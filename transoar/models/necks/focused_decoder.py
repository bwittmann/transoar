"""Focused decoder class."""

import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import trunc_normal_

from transoar.utils.bboxes import box_cxcyczwhd_to_xyzxyz


class FocusedDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6, 
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        bbox_props=None,
        config=None,
        anchors=None
    ):
        super().__init__()
        self.bbox_props = bbox_props
        self.config = config

        self.d_model = d_model
        self.nhead = nhead
        reg_head = MLP(d_model, d_model, 6, 3)

        decoder_layer = FocusedDecoderLayer(
            d_model, dim_feedforward, dropout, activation, nhead, config['obj_self_attn'], 
            config, bbox_props, anchors
        )
        self.decoder = FocusedDecoderModel(decoder_layer, reg_head, num_decoder_layers, return_intermediate_dec)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.decoder.layers

    def forward(self, src, query_embed, pos):
        assert query_embed is not None

        # prepare input for decoder
        src = src.flatten(2).transpose(1, 2)
        pos = pos.flatten(2).transpose(1, 2)
            
        bs, _, c = src.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)       # Tgt in contrast to detr not zeros, but learnable
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        # decoder
        hs = self.decoder(tgt, src, pos, query_embed)

        return hs

class FocusedDecoderModel(nn.Module):
    def __init__(self, decoder_layer, reg_head, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.reg_head = reg_head
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, src, src_pos, query_pos=None):
        output = tgt
        intermediate = []
        for idx, layer in enumerate(self.layers):

            output = layer(output, query_pos, src_pos, src, idx, self.reg_head)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class FocusedDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1, 
        activation="relu",
        n_heads=8,
        obj_self_attn=False,
        config=None,
        bbox_props=None,
        anchors=None
    ):
        super().__init__()
        self.config = config
        self.obj_self_attn = obj_self_attn
        self.bbox_props= bbox_props
        self.anchors = anchors

        self.shapes = {
            'P0': [160, 160, 256],
            'P1': [80, 80, 128],
            'P2': [40, 40, 64],
            'P3': [20, 20, 32],
            'P4': [10, 10, 16],
            'P5': [5, 5, 8]
        }

        # cross attention
        # self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = FocusedAttn(d_model, n_heads, proj_drop=0.1)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)


    def generate_attn_masks(self, idx, tgt, reg_head):
        batch_size, *_ = tgt.shape
        num_queries_per_organ = int(self.config['num_queries'] / self.config['num_organs'])
        assert num_queries_per_organ in [1, 7, 27]

        input_shape = torch.tensor(self.shapes[self.config['input_levels']])
        padding = self.config['focus_factor'][idx] * input_shape
        num_patches = input_shape.prod()

        # get attn volumes in format x1, y1, z1, x2, y2, z2
        if idx == 0:
            attn_volumes = []
            for props in self.bbox_props.values():
                attn_volume = torch.tensor(props['attn_area'])   # x1, y1, z1, x2, y2, z2
                attn_volumes.append(attn_volume[None])
            attn_volumes =  torch.repeat_interleave(torch.cat(attn_volumes), num_queries_per_organ, dim=0)[None].repeat(batch_size, 1, 1)
        else:
            bbox_proposals = reg_head(tgt)
            bbox_proposals = torch.clamp((bbox_proposals.tanh() * self.config['max_anchor_pred_offset']) + self.anchors, min=0, max=1).detach().cpu()
            attn_volumes = box_cxcyczwhd_to_xyzxyz(bbox_proposals)

        # pad attn volumes
        attn_volumes[:, :, :3] = torch.floor((attn_volumes[:, :, :3] * input_shape) - padding).clamp(min=torch.tensor([0, 0, 0]), max=input_shape)
        attn_volumes[:, :, 3:] = torch.floor((attn_volumes[:, :, 3:] * input_shape) + padding).clamp(min=torch.tensor([0, 0, 0]), max=input_shape)

        # init full attn mask
        attn_mask = torch.ones((batch_size, self.config['num_queries'], num_patches.sum()), dtype=torch.bool)

        # mask out regions not in desired attn volume
        for batch_attn_mask, batch_attn_volume in zip(attn_mask, attn_volumes.int()):
            for query_attn_mask, query_attn_volume in zip(batch_attn_mask, batch_attn_volume):
                # Retrieve class attn volume of current query
                dummy_fmap = torch.zeros(input_shape.tolist())

                # Restrict attn to region of interest
                dummy_fmap[query_attn_volume[0]:query_attn_volume[3], query_attn_volume[1]:query_attn_volume[4], query_attn_volume[2]:query_attn_volume[5]] = 1
                dummy_fmap_flattened_idx = dummy_fmap.flatten().nonzero()
                query_attn_mask[dummy_fmap_flattened_idx] = False

        return attn_mask if self.config['restrict_attn'] else torch.zeros_like(attn_mask, dtype=torch.bool)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, src_pos, src, idx, reg_head):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # generate narrowing attn masks
        attn_mask = self.generate_attn_masks(idx, tgt, reg_head).squeeze()

        # cross attention
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(src, src_pos)
        tgt2 = self.cross_attn(q, k, src, mask=attn_mask.to(device=tgt.device).float())

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class FocusedAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=None,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        B, N_kv, C = k.shape
        _, N_q, _ = q.shape

        # Project and split heads
        k = self.k_proj(k).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        q = self.k_proj(q).reshape(B, N_q, self.num_heads, C // self.num_heads)
        q = q * self.scale

        if mask is not None:
            attn = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)).transpose(0, 1)
            mask[mask > 0] = - torch.inf
            attn = (attn + mask).transpose(0, 1)
        else:
            attn = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v.permute(0, 2, 1, 3)).transpose(1, 2).reshape(B, N_q, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x     

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

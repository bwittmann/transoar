"""Focused decoder class."""

import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import trunc_normal_


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
        config=None
    ):
        super().__init__()
        self.bbox_props = bbox_props
        self.config = config

        self.d_model = d_model
        self.nhead = nhead

        decoder_layer = FocusedDecoderLayer(
            d_model, dim_feedforward, dropout, activation, nhead, config, bbox_props
        )
        self.decoder = FocusedDecoderModel(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos):
        assert query_embed is not None

        # prepare input for decoder
        src = src.flatten(2).transpose(1, 2)
        pos = pos.flatten(2).transpose(1, 2)
            
        bs, _, c = src.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        # decoder
        hs = self.decoder(tgt, src, pos, query_embed)

        return hs

class FocusedDecoderModel(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, src, src_pos, query_pos=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output, _ = layer(output, query_pos, src_pos, src)

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
        config=None,
        bbox_props=None
    ):
        super().__init__()
        self.config = config
        self.bbox_props= bbox_props
        self.num_queries_per_organ = int(self.config['num_queries'] / self.config['num_organs'])
        assert self.num_queries_per_organ in [1, 7, 27, 54]

        if self.config['num_organs'] == 20:
            shapes = {
                'P0': [160, 160, 256],
                'P1': [80, 80, 128],
                'P2': [40, 40, 64],
                'P3': [20, 20, 32],
                'P4': [10, 10, 16],
                'P5': [5, 5, 8]
            }
        else:
            shapes = {
                'P0': [256, 256, 128],
                'P1': [128, 128, 64],
                'P2': [64, 64, 32],
                'P3': [32, 32, 16],
                'P4': [16, 16, 8],
                'P5': [8, 8, 4]
            }
        self.input_shape = torch.tensor(shapes[self.config['input_levels']])

        # cross attention
        self.attn_mask = self.generate_attn_masks().cuda()
        self.cross_attn = FocusedAttn(d_model, n_heads, self.attn_mask, proj_drop=0.1)
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

    def generate_attn_masks(self, padding=0):
        # get attn volumes in format x1, y1, z1, x2, y2, z2
        attn_volumes = []
        for props in self.bbox_props.values():
            attn_volume = torch.tensor(props['attn_area'])   # x1, y1, z1, x2, y2, z2
            attn_volumes.append(attn_volume[None])
        attn_volumes =  torch.repeat_interleave(torch.cat(attn_volumes), self.num_queries_per_organ, dim=0)

        # pad attn volumes
        attn_volumes = ((attn_volumes * self.input_shape.repeat(2)) - padding).clamp(min=torch.zeros(6, dtype=torch.int), max=self.input_shape.repeat(2))
        attn_volumes[:, :3] = torch.floor(attn_volumes[:, :3])
        attn_volumes[:, 3:] = torch.ceil(attn_volumes[:, 3:])
        attn_volumes = attn_volumes.int()
        
        # init full attn mask
        attn_mask = torch.ones(self.config['num_queries'], *self.input_shape.tolist()).bool()

        # mask out regions not in desired attn volume
        for q in range(self.config['num_queries']):
            attn_mask[q, attn_volumes[q, 0]:attn_volumes[q, 3], attn_volumes[q, 1]:attn_volumes[q, 4], attn_volumes[q, 2]:attn_volumes[q, 5]] = False

        return attn_mask.flatten(1) if self.config['restrict_attn'] else torch.zeros_like(attn_mask.flatten(1), dtype=torch.bool)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, src_pos, src):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2= self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(src, src_pos)
        tgt2, weights = self.cross_attn(q, k, src, mask=self.attn_mask.float())

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, weights


class FocusedAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn_mask,
        qkv_bias=None,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0,
        use_pos_bias=False,
        return_weights=True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ret_weights = return_weights
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        # learnable position bias
        if use_pos_bias:
            self.pos_bias = trunc_normal_(nn.Parameter(torch.zeros_like(attn_mask, dtype=torch.float)), std=.02)
        else:
            self.pos_bias = None

    def forward(self, q, k, v, mask=None):
        B, N_kv, C = k.shape
        _, N_q, _ = q.shape

        # Project and split heads
        k = self.k_proj(k).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        q = self.k_proj(q).reshape(B, N_q, self.num_heads, C // self.num_heads)
        q = q * self.scale

        attn = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))

        if self.pos_bias is not None:
            attn += self.pos_bias

        if mask is not None:
            mask[mask > 0] = - torch.inf
            attn += mask

        attn = self.softmax(attn)

        if self.ret_weights:
            weights = attn

        attn = self.attn_drop(attn)

        x = (attn @ v.permute(0, 2, 1, 3)).transpose(1, 2).reshape(B, N_q, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        
        if self.ret_weights:
            return x, weights
        else:
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

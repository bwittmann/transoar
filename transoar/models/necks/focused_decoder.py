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

        self.shapes = {
            'P0': [160, 160, 256],
            'P1': [80, 80, 128],
            'P2': [40, 40, 64],
            'P3': [20, 20, 32],
            'P4': [10, 10, 16],
            'P5': [5, 5, 8]
        }

        self.d_model = d_model
        self.nhead = nhead

        attn_masks = self.generate_attn_masks(anchors)

        decoder_layer = FocusedDecoderLayer(
            d_model, dim_feedforward, dropout, activation, nhead, attn_masks, config['obj_self_attn']
        )
        self.decoder = FocusedDecoderModel(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self._reset_parameters()

    def generate_attn_masks(self, anchors, padding_factor=0.2):
        num_queries_per_organ = int(self.config['num_queries'] / self.config['num_organs'])
        assert num_queries_per_organ in [1, 7, 27]
        anchors = box_cxcyczwhd_to_xyzxyz(anchors.reshape(self.config['num_organs'], -1, 6))

        input_shapes = [self.shapes[input_level] for input_level in self.config['input_levels']]

        # Get per class volume to attend to at different feature map lvls
        attn_volumes = defaultdict(list)
        for idx, class_anchors in enumerate(anchors):
            for fmap_shape in input_shapes:
                padding = [val * padding_factor for val in fmap_shape]

                attn_volume = torch.cat(
                    [
                        torch.floor((class_anchors[:, 0] * fmap_shape[0]) - padding[0])[None].clamp(min=0, max=fmap_shape[0]),   # x1
                        torch.floor((class_anchors[:, 1] * fmap_shape[1]) - padding[1])[None].clamp(min=0, max=fmap_shape[1]),   # y1
                        torch.floor((class_anchors[:, 2] * fmap_shape[2]) - padding[2])[None].clamp(min=0, max=fmap_shape[2]),   # z1
                        torch.ceil((class_anchors[:, 3] * fmap_shape[0]) + padding[0])[None].clamp(min=0, max=fmap_shape[0]),    # x2
                        torch.ceil((class_anchors[:, 4] * fmap_shape[1]) + padding[1])[None].clamp(min=0, max=fmap_shape[1]),    # y2
                        torch.ceil((class_anchors[:, 5] * fmap_shape[2]) + padding[2])[None].clamp(min=0, max=fmap_shape[2])     # z2
                    ]
                )
                attn_volumes[int(idx)].append(attn_volume.to(dtype=torch.int).transpose(0, 1))

        # Init full attn mask
        num_patches_per_lvl = torch.tensor(input_shapes).prod(axis=1)
        attn_masks = [torch.ones((self.config['num_organs'], num_queries_per_organ, num_patches.sum()), dtype=torch.bool) for num_patches in num_patches_per_lvl]


        # Mask out regions not in desired attn volume
        for idx, attn_mask in enumerate(attn_masks):
            for class_, class_attn_mask in enumerate(attn_mask):
                # Retrieve class attn volume of current query
                class_attn_volume = attn_volumes[class_][idx]

                for query_mask, query_vol in zip(class_attn_mask, class_attn_volume):
                    dummy_fmap = torch.zeros(input_shapes[idx])
                    dummy_fmap[query_vol[0]:query_vol[3], query_vol[1]:query_vol[4], query_vol[2]:query_vol[5]] = 1
                    dummy_fmap_flattened_idx = dummy_fmap.flatten().nonzero()
                    query_mask[dummy_fmap_flattened_idx] = False

        attn_masks = [mask.flatten(0, 1) for mask in attn_masks]
        return attn_masks if self.config['restrict_attn'] else [torch.zeros_like(attn_mask, dtype=torch.bool) for attn_mask in attn_masks]

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, srcs, query_embed, poss):
        assert query_embed is not None

        # prepare input for decoder
        for idx in range(len(srcs)):
            srcs[idx] = srcs[idx].flatten(2)
            poss[idx] = poss[idx].flatten(2)
        src, pos = torch.cat(srcs, dim=-1), torch.cat(poss, dim=-1)

        bs, c, _ = src.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)       # Tgt in contrast to detr not zeros, but learnable
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

    def forward(self, tgt, srcs, srcs_pos, query_pos=None):
        output = tgt


        intermediate = []
        for idx, layer in enumerate(self.layers):

            if False: #len(srcs) == len(self.layers):
                output = layer(output, query_pos, srcs_pos[idx], srcs[idx], idx)
            else:
                output = layer(output, query_pos, srcs_pos, srcs, 0)

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
        attn_masks=None,
        obj_self_attn=False
    ):
        super().__init__()
        self.attn_mask = torch.cat(attn_masks, dim=-1)
        self.obj_self_attn = obj_self_attn

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # obj self attention
        if obj_self_attn:
            self.self_attn_obj = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout_obj = nn.Dropout(dropout)
            self.norm_obj = nn.LayerNorm(d_model)
            self.obj_attn_mask = self.generate_obj_attn_mask()

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def generate_obj_attn_mask(self):
        attn_mask = torch.ones((540, 540))

        for idx, query in enumerate(attn_mask):
            start_idx = (idx // 27) * 27
            query[start_idx:start_idx + 27] = 0

        return attn_mask.to(dtype=torch.bool)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, src_pos, src, idx):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # obj self attention
        if self.obj_self_attn:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn_obj(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=self.obj_attn_mask.to(device=tgt.device))[0].transpose(0, 1)
            tgt = tgt + self.dropout_obj(tgt2)
            tgt = self.norm_obj(tgt)

        # cross attention
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(src, src_pos)
        tgt2 = self.cross_attn(q.transpose(0, 1), k.permute(2, 0, 1), src.permute(2, 0, 1), attn_mask=self.attn_mask.to(device=tgt.device))[0].transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

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



class FocusedAttnV1(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias,
        qk_scale,
        attn_drop,
        proj_drop
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # learnable position bias
        self.position_bias = nn.Parameter(
            torch.zeros(6, 27, self.window_size.prod())
        )

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.position_bias, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src_k, src_q):
        B, N_k, C = src_k.shape
        B, N_q, C = src_q.shape

        # Project and split heads
        k = self.k_proj(src_k).reshape(B, N_k, self.num_heads, C // self.num_heads)
        v = self.v_proj(src_k).reshape(B, N_k, self.num_heads, C // self.num_heads)
        q = self.k_proj(src_q).reshape(B, N_q, self.num_heads, C // self.num_heads)

        q = q * self.scale
        attn = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        attn = attn + self.position_bias

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v.permute(0, 2, 1, 3)).transpose(1, 2).reshape(B, N_q, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FocusedAttnV2(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn_mask,
        qkv_bias=None,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0,
        use_pos_bias=False
    ):
        super().__init__()
        self.use_pos_bias = use_pos_bias
        self.attn_mask = attn_mask
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # learnable position bias
        if use_pos_bias:
            self.pos_bias = nn.ParameterList()
            for class_mask in self.attn_mask[0]:
                params = trunc_normal_(nn.Parameter(torch.zeros((class_mask == False).sum())), std=.02)
                self.pos_bias.append(params)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        B, N_kv, C = k.shape
        _, N_q, _ = q.shape

        # Project and split heads
        k = self.k_proj(k).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        q = self.k_proj(q).reshape(B, N_q, self.num_heads, C // self.num_heads)
        q = q * self.scale

        # Attend solely to non-masked region
        attn = (- torch.inf * torch.ones(B, self.num_heads, N_q, N_kv)).to(dtype=q.dtype, device=q.device)
        for class_ in range(N_q):
            mask = ~self.attn_mask[0][class_]
            k_masked, q_class = k[:, mask].permute(0, 2, 3, 1), q[:, class_][:, None].permute(0, 2, 1, 3)
            class_attn = torch.matmul(q_class, k_masked).squeeze(2)

            if self.use_pos_bias:
                class_attn += self.pos_bias[class_]

            attn[:, :, class_, mask] = class_attn

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v.permute(0, 2, 1, 3)).transpose(1, 2).reshape(B, N_q, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FocusedAttnV3(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=None,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0,
        use_pos_bias=False
    ):
        super().__init__()
        self.use_pos_bias = use_pos_bias
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # learnable position bias
        if use_pos_bias:
            self.pos_bias = nn.ParameterList()
            for class_mask in self.attn_mask[0]:
                params = trunc_normal_(nn.Parameter(torch.zeros((class_mask == False).sum())), std=.02)
                self.pos_bias.append(params)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask, dyn_attn_mask):
        B, N_kv, C = k.shape
        _, N_q, _ = q.shape

        # Project and split heads
        k = self.k_proj(k).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        q = self.k_proj(q).reshape(B, N_q, self.num_heads, C // self.num_heads)
        q = q * self.scale

        # # Attend solely to non-masked region
        # attn = (- torch.inf * torch.ones(B, self.num_heads, N_q, N_kv)).to(dtype=q.dtype, device=q.device)
        # for class_ in range(N_q):
        #     mask = ~attn_mask[:, class_]
        #     k_masked, q_class = k[mask][None].permute(0, 2, 3, 1), q[:, class_][:, None].permute(0, 2, 1, 3)
        #     class_attn = torch.matmul(q_class, k_masked).squeeze(2)

        #     if self.use_pos_bias:
        #         class_attn += self.pos_bias[class_]

        #     attn[:, :, class_, mask] = class_attn

        # Adjust combine fixed and dynamic attn masks
        # self.dyn_rest(tgt).tanh() * 5
        attn_mask = attn_mask.float().cuda()
        attn_mask[attn_mask > 0] = - torch.inf
        attn_mask += dyn_attn_mask

        attn = (torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)).transpose(0, 1) + attn_mask).transpose(0, 1)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v.permute(0, 2, 1, 3)).transpose(1, 2).reshape(B, N_q, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def generate_attn_masks_V2(self, padding=0, restriction=None):
        num_queries_per_organ = int(self.num_queries / self.num_organs)
        assert num_queries_per_organ in [1, 7, 27]
        #batch_size, *_ = restriction.shape
        batch_size = 2

        input_shapes = [self.shapes[input_level] for input_level in self.input_levels]

        # Get per class volume to attend to at different feature map lvls
        attn_volumes = defaultdict(list)
        for class_, props in self.bbox_props.items():
            attn_volume_normalized = torch.tensor(props['attn_area'])   # x1, y1, z1, x2, y2, z2

            for fmap_shape in input_shapes:
                attn_volume = torch.tensor(
                    [
                        torch.floor(attn_volume_normalized[0] * fmap_shape[0]) - padding,   # x1
                        torch.floor(attn_volume_normalized[1] * fmap_shape[1]) - padding,   # y1
                        torch.floor(attn_volume_normalized[2] * fmap_shape[2]) - padding,   # z1
                        torch.ceil(attn_volume_normalized[3] * fmap_shape[0]) + padding,    # x2
                        torch.ceil(attn_volume_normalized[4] * fmap_shape[1]) + padding,    # y2
                        torch.ceil(attn_volume_normalized[5] * fmap_shape[2]) + padding     # z2
                    ]
                )
                attn_volumes[int(class_)].append(attn_volume.to(dtype=torch.int))

        # Init full attn mask
        num_patches_per_lvl = torch.tensor(input_shapes).prod(axis=1)
        attn_masks = [[torch.ones((self.num_queries, num_patches.sum()), dtype=torch.bool) for num_patches in num_patches_per_lvl] for b in range(batch_size)]

        # Get query classes
        query_classes = torch.repeat_interleave(torch.arange(1, self.num_organs + 1), num_queries_per_organ)

        # Mask out regions not in desired attn volume
        # for batch_restriction, batch_attn_masks in zip(restriction, attn_masks):   # loop over batches
        for batch_attn_masks in attn_masks:   # loop over batches
            for idx, attn_mask in enumerate(batch_attn_masks):    # loop over individual masks

                # for query_attn_volume, query_class, query_restriction in zip(attn_mask, query_classes, batch_restriction):
                for query_attn_volume, query_class in zip(attn_mask, query_classes):
                    # Retrieve class attn volume of current query
                    dummy_fmap = torch.zeros(input_shapes[idx])
                    class_attn_volume = attn_volumes[query_class.item()][idx]

                    # Learned additional restriction of attn volume
                    # delta_attn_volume = (class_attn_volume[-3:] - class_attn_volume[:3]).repeat(2)
                    # query_restriction_xyzxyz = (box_cxcyczwhd_to_xyzxyz(query_restriction) * delta_attn_volume).float()

                    # min_vol = (class_attn_volume[:3] + query_restriction_xyzxyz[:3]).floor()
                    # max_vol = (class_attn_volume[-3:] - delta_attn_volume[-3:] + query_restriction_xyzxyz[-3:]).ceil()
                    # class_attn_volume = torch.cat([min_vol, max_vol], dim=-1).long()

                    # Restrict attn to region of interest
                    dummy_fmap[class_attn_volume[0]:class_attn_volume[3], class_attn_volume[1]:class_attn_volume[4], class_attn_volume[2]:class_attn_volume[5]] = 1
                    dummy_fmap_flattened_idx = dummy_fmap.flatten().nonzero()
                    query_attn_volume[dummy_fmap_flattened_idx] = False

        attn_masks = torch.cat([torch.cat(batch_attn_masks, dim=-1)[None] for batch_attn_masks in attn_masks], dim=0)
        return attn_masks if self.restrict_attn else torch.zeros_like(attn_mask, dtype=torch.bool)


"""Focused decoder class."""

import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn


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

        attn_masks = self.generate_attn_mask()
        attn_mask = attn_masks[0]   # TODO: pyramid scheme

        decoder_layer = FocusedDecoderLayer(
            d_model, dim_feedforward, dropout, activation, nhead, attn_mask, config['obj_self_attn']
        )
        self.decoder = FocusedDecoderModel(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self._reset_parameters()

    def generate_attn_mask(self, padding=0):
        num_queries_per_organ = int(self.config['num_queries'] / self.config['num_organs'])
        assert num_queries_per_organ in [1, 7, 27]

        input_shapes = [self.shapes[input_level] for input_level in self.config['input_levels']]

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
        attn_masks = [torch.ones((self.config['num_queries'], num_patches.sum()), dtype=torch.bool) for num_patches in num_patches_per_lvl]

        # Get query classes
        query_classes = torch.repeat_interleave(torch.arange(1, self.config['num_organs'] + 1), num_queries_per_organ)

        # Mask out regions not in desired attn volume
        for idx, attn_mask in enumerate(attn_masks):

            for query_attn_volume, query_class in zip(attn_mask, query_classes):
                # Retrieve class attn volume of current query
                dummy_fmap = torch.zeros(input_shapes[idx])
                class_attn_volume = attn_volumes[query_class.item()][idx]

                # Restrict attn to region of interest
                dummy_fmap[class_attn_volume[0]:class_attn_volume[3], class_attn_volume[1]:class_attn_volume[4], class_attn_volume[2]:class_attn_volume[5]] = 1
                dummy_fmap_flattened_idx = dummy_fmap.flatten().nonzero()
                query_attn_volume[dummy_fmap_flattened_idx] = False

        return attn_masks if self.config['restrict_attn'] else [torch.zeros_like(attn_mask, dtype=torch.bool) for attn_mask in attn_masks]

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, srcs, query_embed, pos_embeds):
        assert query_embed is not None

        # prepare input for decoder
        src_flatten = srcs.flatten(2).transpose(1, 2)                                # [Batch, Patches, HiddenDim] 
        pos_embed_flatten = pos_embeds.flatten(2).transpose(1, 2)                    # [Batch, Patches, HiddenDim] 
        
        bs, _, c = src_flatten.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)                        # Tgt in contrast to detr not zeros, but learnable
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        # decoder
        hs = self.decoder(tgt, src_flatten, pos_embed_flatten, query_embed)

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
            output = layer(output, query_pos, src_pos, src)

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
        attn_mask=None,
        obj_self_attn=False
    ):
        super().__init__()
        self.attn_mask = attn_mask
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

    def forward(self, tgt, query_pos, src_pos, src):
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
        tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), src.transpose(0, 1), attn_mask=self.attn_mask.to(device=tgt.device))[0].transpose(0, 1)

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

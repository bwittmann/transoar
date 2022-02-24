"""Module containing code of the backbones decoder blocks."""

import copy
from numpy import ones_like

import torch
import torch.nn as nn
import torch.nn.functional as F

from transoar.models.ops.modules import MSDeformAttn


class DecoderDefAttnBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        feature_levels,
        n_points,
        use_cuda=True,
        activation="relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.feature_levels = feature_levels
        num_feature_levels = len(feature_levels)

        attn_layer = DefAttnLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels,
            nhead, n_points, use_cuda
        )
        self.refine_def_attn = DefAttnTransformer(attn_layer, num_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, fmap):
        # Generate empty mask
        mask = torch.zeros(fmap[:, 0].shape, dtype=torch.bool)

        _, D, H, W = mask.shape
        valid_D = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)
        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_d], -1)
        return valid_ratio

    def forward(self, fmaps, pos_embeds):
        # prepare input for refinement
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(fmaps, pos_embeds)):
            bs, c, d, h, w = src.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)                                # [Batch, Patches, HiddenDim]   
            pos_embed = pos_embed.flatten(2).transpose(1, 2)                    # [Batch, Patches, HiddenDim]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)    # [Batch, Patches, HiddenDim]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)                                 # [Batch, AllLvlPatches, HiddenDim]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)             # [Batch, AllLvlPatches, HiddenDim]

        # Shapes of feature maps of levels in use
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)

        # Determine indices of batches that mark the start of a new feature level
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # Refine fmaps
        memory = self.refine_def_attn(
            src_flatten, spatial_shapes, level_start_index, lvl_pos_embed_flatten
        )

        # Prepare output
        out_shapes = [[bs, c] + shape.tolist() for shape in spatial_shapes]
        fmaps = torch.split(memory, spatial_shapes.prod(axis=-1).tolist(), dim=1)
        fmaps = [fmap.transpose(-1, -2).reshape(shape) for fmap, shape in zip(fmaps, out_shapes)]

        return fmaps


class DefAttnTransformer(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        # No mask in use
        valid_ratios = torch.ones_like(spatial_shapes, dtype=torch.float)[None]

        reference_points_list = []
        for _, (D_, H_, W_) in enumerate(spatial_shapes):

            ref_z, ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
            )

            # Get relative coords in range [0, 1]
            ref_z = ref_z.reshape(-1)[None] / D_ 
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_

            ref = torch.stack((ref_x, ref_y, ref_z), -1)    # Coords in format WHD/XYZ
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)  # [Batch, AllLvlPatches, RelativeRefCoords]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] #WHD/XYZ

        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, pos=None):
        output = src

        # Get reference points normalized and in valid areas, [Batch, AllLvlPatches, NumLevels, RefCoords]
        reference_points = self.get_reference_points(spatial_shapes, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index)

        return output

class DefAttnLayer(nn.Module):
    def __init__(self, d_model, d_ffn, dropout, activation, n_levels, n_heads, n_points, use_cuda):
        super().__init__()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, use_cuda)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention -> query and input_flatten are the same
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src


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

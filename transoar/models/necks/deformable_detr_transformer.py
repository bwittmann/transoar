"""Deformable DETR Transformer class adapted from https://github.com/fundamentalvision/Deformable-DETR."""

import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from transoar.models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6, 
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        enc_n_points=4,
        use_cuda=True,
        bbox_props=None,
        config=None
    ):
        super().__init__()
        self.bbox_props = bbox_props
        self.config = config

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels,
            nhead, enc_n_points, use_cuda
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        attn_mask = self.generate_attn_mask()
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, nhead, attn_mask)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.dropout = nn.Dropout2d(0.5)

        self._reset_parameters()

    def generate_attn_mask(self, padding=0):
        assert self.config['num_queries'] == self.config['queries_per_organ'] * self.config['num_feature_levels'] * self.config['num_organs']

        # Init full attn mask
        num_patches_per_lvl = torch.tensor(self.config['input_shapes']).prod(axis=1)
        attn_mask = torch.ones((self.config['num_queries'], num_patches_per_lvl.sum()), dtype=torch.bool)

        # Get per class volume to attend to at different feature map lvls
        attn_volumes = defaultdict(list)
        for class_, props in self.bbox_props.items():
            attn_volume_normalized = torch.tensor(props['attn_area'])   # x1, y1, z1, x2, y2, z2

            for fmap_shape in self.config['input_shapes']:
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

        # Set attn mask to mask out region which is not in desired attn volume
        query_classes = torch.repeat_interleave(torch.arange(1, self.config['num_organs'] + 1), self.config['queries_per_organ'] * self.config['num_feature_levels'])
        query_fmap_lvls = torch.repeat_interleave(torch.arange(self.config['num_feature_levels']), self.config['queries_per_organ']).repeat(self.config['num_organs']) 
        for query_attn_volume, query_class, query_fmap_lvl in zip(attn_mask, query_classes, query_fmap_lvls):
            # Retrieve class attn volume of current query
            dummy_fmap = torch.zeros(self.config['input_shapes'][query_fmap_lvl.item()])
            class_attn_volume = attn_volumes[query_class.item()][query_fmap_lvl.item()]

            # Restrict attn to region of interest
            dummy_fmap[class_attn_volume[0]:class_attn_volume[3], class_attn_volume[1]:class_attn_volume[4], class_attn_volume[2]:class_attn_volume[5]] = 1
            dummy_fmap_flattened = dummy_fmap.flatten().nonzero()

            if query_fmap_lvl.item() > 0:
                dummy_fmap_flattened += num_patches_per_lvl[query_fmap_lvl.item() - 1]

            query_attn_volume[dummy_fmap_flattened] = False

        return attn_mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, D, H, W = mask.shape
        valid_D = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)
        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_d], -1)
        return valid_ratio

    def forward(self, srcs, masks, query_embed, pos_embeds):
        assert query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, d, h, w = src.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)                                # [Batch, Patches, HiddenDim]   
            mask = mask.flatten(1)                                              # [Batch, Patches]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)                    # [Batch, Patches, HiddenDim]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)    # [Batch, Patches, HiddenDim]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)                                 # [Batch, AllLvlPatches, HiddenDim]
        mask_flatten = torch.cat(mask_flatten, 1)                               # [Batch, AllLvlPatches]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)             # [Batch, AllLvlPatches, HiddenDim]

        # Shapes of feature maps of levels in use
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)

        # Determine indices of batches that mark the start of a new feature level
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # Determine the ratios of valid regions based on the mask for in format WHD/XYZ
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten
        )
        # memory = src_flatten  

        # prepare input for decoder
        bs, _, c = memory.shape                                                 # [Batch, AllLvlPatches, HiddenDim]
        query_embed, tgt = torch.split(query_embed, c, dim=1)                   # Tgt in contrast to detr not zeros, but learnable
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        # decoder
        hs = self.decoder(tgt, memory, spatial_shapes, level_start_index, lvl_pos_embed_flatten, query_embed)

        # query drop
        # hs = hs.permute((0, 2, 1, 3))
        # hs = self.dropout(hs)
        # hs = hs.permute((0, 2, 1, 3))
        return hs


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_cuda=True
    ):
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


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):

            ref_z, ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
            )
            # Get relative coords in range [0, 1], ref points in masked areas have values > 1
            ref_z = ref_z.reshape(-1)[None] / D_ #(valid_ratios[:, None, lvl, 2] * D_)  # TODO
            ref_y = ref_y.reshape(-1)[None] / H_ #(valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / W_ #(valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y, ref_z), -1)    # Coords in format WHD/XYZ
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [Batch, AllLvlPatches, RelativeRefCoords]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] # Valid ratio also in format WHD/XYZ
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src

        # Get reference points normalized and in valid areas, [Batch, AllLvlPatches, NumLevels, RefCoords]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1, 
        activation="relu",
        n_heads=8,
        attn_mask=None
    ):
        super().__init__()
        self.attn_mask = attn_mask

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
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

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, lvl_pos, src, src_spatial_shapes, level_start_index):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(src, lvl_pos)
        tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), src.transpose(0, 1), attn_mask=self.attn_mask.to(device=tgt.device))[0].transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, src, src_spatial_shapes, src_level_start_index, lvl_pos, query_pos=None):
        output = tgt

        intermediate = []
        for layer in self.layers:
            output = layer(output, query_pos, lvl_pos, src, src_spatial_shapes, src_level_start_index)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


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

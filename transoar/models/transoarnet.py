"""Main model of the transoar project."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transoar.models.build import build_backbone, build_neck, build_pos_enc

class TransoarNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['neck']['hidden_dim']
        num_queries = config['neck']['num_queries']
        num_channels = config['backbone']['num_channels']
        num_classes = config['num_classes']

        # Use auxiliary decoding losses if required
        self._aux_loss = config['neck']['aux_loss']

        # Skip connection from backbone outputs to heads
        self._skip_con = config['neck']['skip_con']
        if self._skip_con:
            self._skip_proj = nn.Linear(
                config['backbone']['num_feature_patches'], 
                config['neck']['num_queries']
            )

        # Get backbone
        self._backbone = build_backbone(config['backbone'])

        # Get neck
        self._neck = build_neck(config['neck'])

        # Get heads
        self._cls_head = nn.Linear(hidden_dim, num_classes + 1)
        self._bbox_reg_head = MLP(hidden_dim, hidden_dim, 6, 3)

        # Get projections and embeddings
        if 'num_feature_levels' in config['neck']:
            self._query_embed = nn.Embedding(num_queries, hidden_dim * 2)   # 2 -> tgt + query_pos

            # Get individual input projection for each feature level
            num_feature_levels = config['neck']['num_feature_levels']
            if num_feature_levels > 1:
                num_backbone_outs = len(config['backbone']['num_channels'])
                input_proj_list = []
                for _ in range(num_backbone_outs):
                    in_channels = config['backbone']['num_channels'][_]
                    input_proj_list.append(nn.Sequential(
                        nn.Conv3d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))

                self._input_proj = nn.ModuleList(input_proj_list)
            else:
                self._input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(config['backbone']['num_channels'][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
            
            self._reset_parameter()
        else:
            self._query_embed = nn.Embedding(num_queries, hidden_dim)
            self._input_proj = nn.Conv3d(num_channels, hidden_dim, kernel_size=1)

        # Get positional encoding
        self._pos_enc = build_pos_enc(config['neck'])

    def _reset_parameter(self):
        nn.init.constant_(self._bbox_reg_head.layers[-1].weight.data, 0)
        nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data, 0)
        nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data[2:], -2.0)

        for proj in self._input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, x, mask):
        out_backbone = self._backbone(x, mask)

        if len(out_backbone) > 1:   # For approaches that utilize multiple feature maps
            srcs = []
            masks = []
            pos = []
            for idx, (src, mask) in enumerate(out_backbone):
                srcs.append(self._input_proj[idx](src))
                masks.append(mask)
                pos.append(self._pos_enc(mask))
        else:
            srcs = self._input_proj(out_backbone[0][0])
            masks = out_backbone[0][1]
            pos = self._pos_enc(masks)

        out_neck = self._neck(             # [Batch, Queries, HiddenDim]         
            srcs,
            masks,
            self._query_embed.weight,
            pos
        )

        if self._skip_con:
            if isinstance(srcs, torch.Tensor):
                out_backbone_proj = srcs.flatten(2)
            else:
                out_backbone_proj = torch.cat([src.flatten(2) for src in srcs], dim=-1)
            out_backbone_skip_proj = self._skip_proj(out_backbone_proj).permute(0, 2, 1)
            out_neck = out_neck + out_backbone_skip_proj

        if len(out_neck) > 2:   # In the case of relative offset prediction to references
            hs, init_reference_out, inter_references_out = out_neck

            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference_out
                else:
                    reference = inter_references_out[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self._cls_head(hs[lvl])
                tmp = self._bbox_reg_head(hs[lvl])

                assert reference.shape[-1] == 3
                tmp[..., :3] += reference

                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            pred_logits = torch.stack(outputs_classes)
            pred_boxes = torch.stack(outputs_coords)
        else:
            pred_logits = self._cls_head(out_neck)
            pred_boxes = self._bbox_reg_head(out_neck).sigmoid()

        out = {
            'pred_logits': pred_logits[-1], # Take output of last layer
            'pred_boxes': pred_boxes[-1]
        }

        if self._aux_loss:
            out['aux_outputs'] = self._set_aux_loss(pred_logits, pred_boxes)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, pred_logits, pred_boxes):
        # Hack to support dictionary with non-homogeneous values
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(pred_logits[:-1], pred_boxes[:-1])]


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

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

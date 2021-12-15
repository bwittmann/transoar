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
                np.prod(config['backbone']['feature_map_dim']), 
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
            self._query_embed = nn.Embedding(num_queries, hidden_dim * 2)   # TODO: Why?

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
                    nn.Conv3d(config['backbone']['num_channels'][0], hidden_dim, kernel_size=1),    # TODO
                    nn.GroupNorm(32, hidden_dim),
                )])

            # Initialize learnable params
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self._cls_head.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(self._bbox_reg_head.layers[-1].weight.data, 0)
            nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data, 0)
            for proj in self._input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)

            num_pred = config['neck']['dec_layers']
            nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data[2:], -2.0)
            self._cls_head = nn.ModuleList([self._cls_head for _ in range(num_pred)])   # TODO check logic
            self._bbox_reg_head = nn.ModuleList([self._bbox_reg_head for _ in range(num_pred)])
            self._neck.decoder.bbox_embed = None
        else:
            self._query_embed = nn.Embedding(num_queries, hidden_dim)
            self._input_proj = nn.Conv3d(num_channels, hidden_dim, kernel_size=1)

        # Get positional encoding
        self._pos_enc = build_pos_enc(config['neck'])


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

        x_neck = self._neck(             # [Batch, Queries, HiddenDim]         
            srcs,
            masks,
            self._query_embed.weight,
            pos
        )

        if self._skip_con:
            x_backbone_proj = x_backbone_proj.flatten(2)
            x_backbone_skip_proj = self._skip_proj(x_backbone_proj).permute(0, 2, 1)
            x_neck = x_neck + x_backbone_skip_proj

        pred_logits = self._cls_head(x_neck)
        pred_boxes = self._bbox_reg_head(x_neck).sigmoid()

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

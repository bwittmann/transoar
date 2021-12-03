"""Main model of the transoar project."""

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
        self._query_embed = nn.Embedding(num_queries, hidden_dim)
        self._input_proj = nn.Conv3d(num_channels, hidden_dim, kernel_size=1)

        # Get positional encoding
        self._pos_enc = build_pos_enc(config['neck'])

    def forward(self, x, mask):
        x_backbone, mask = self._backbone(x, mask)
        x_backbone_proj = self._input_proj(x_backbone)

        x_neck = self._neck(             # [Batch, Queries, HiddenDim]         
            x_backbone_proj,
            mask,
            self._query_embed.weight,
            self._pos_enc(mask)
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

"""Main model of the transoar project."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from transoar.models.build import build_backbone, build_neck, build_pos_enc

class TransoarNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['neck']['hidden_dim']
        num_queries = config['neck']['num_queries']
        num_channels = config['backbone']['fpn_channels']
        num_classes = config['num_classes']
        self.input_level = config['neck']['input_level']

        # Use auxiliary decoding losses if required
        self._aux_loss = config['neck']['aux_loss']

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


    def forward(self, x):
        out_backbone = self._backbone(x)
        out_backbone = out_backbone[self.input_level]

        src = self._input_proj(out_backbone)
        mask = torch.zeros_like(src[:, 0], dtype=torch.bool)    # No mask needed

        pos = self._pos_enc(src)

        out_neck = self._neck(             # [Batch, Queries, HiddenDim]         
            src,
            mask,
            self._query_embed.weight,
            pos
        )

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

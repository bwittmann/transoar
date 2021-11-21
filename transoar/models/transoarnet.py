"""Main model of the transoar project."""

import torch.nn as nn
import torch.nn.functional as F

from transoar.models.build import build_backbone, build_neck
from transoar.models.position_encoding import PositionEmbeddingSine3D

class TransoarNet(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        hidden_dim = config['neck']['hidden_dim']
        num_queries = config['neck']['num_queries']
        num_channels = config['backbone']['num_channels']
        num_classes = num_classes
        
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
        self._pos_enc = PositionEmbeddingSine3D(channels=hidden_dim)


    def forward(self, x, mask):
        x, mask = self._backbone(x, mask)

        x = self._neck(             # [Batch, Queries, HiddenDim]         
            self._input_proj(x),
            mask,
            self._query_embed.weight,
            self._pos_enc(mask)
        )[0]

        out = {
            'pred_logits': self._cls_head(x),
            'pred_bboxes': self._bbox_reg_head(x).sigmoid()
        }

        return out


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

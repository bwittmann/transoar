"""Main model of the transoar project."""

import torch.nn as nn

from transoar.models.build import build_backbone

class TransoarNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self._backbone = build_backbone(config['backbone'])

    def forward(self, x):
        x = self._backbone(x)

        return x
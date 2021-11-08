"""Main model of the transoar project."""


import torch.nn as nn

from transoar.models.backbones.swin_transformer import SwinTransformer3D

class TransoarNet(nn.Module):
    def __init__(self, config):
        self._backbone = SwinTransformer3D(**config)
        self._backbone.init_weights()

    def forward(self, x):
        x = self._backbone(x)

        return x
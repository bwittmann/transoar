"""Main model of the transoar project."""

import torch
import torch.nn as nn

from transoar.models.backbones import swin_transformer

class TransoarNet(nn.Module):
    def __init__(self, config):
        self._backbone = swin_transformer()
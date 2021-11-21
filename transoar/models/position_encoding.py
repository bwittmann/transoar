"""Various positional encodings for the transformer."""

import math

import torch
import numpy as np
from torch import nn


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, channels=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.orig_channels = channels
        self.channels = int(np.ceil(channels/6)*2)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        """[summary]
        Args:
            mask ([type]): [shape BxHxWxD]
        Raises:
            RuntimeError: [description]
        Returns:
            [type]: [description]
        """""""""
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        z_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            x_embed = (x_embed - 0.5) / (x_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = (y_embed - 0.5) / (y_embed[:, :, -1:, :] + eps) * self.scale
            z_embed = (z_embed - 0.5) / (z_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.channels, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.channels)  # TODO: check 6 or something else?

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_z = z_embed[..., None] / dim_t
        
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(4)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(4)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=4).flatten(4)
        pos = torch.cat((pos_y, pos_x, pos_z), dim=4).permute(0, 4, 1, 2, 3)
        return pos[:,:self.orig_channels,...]

"""Various positional encodings for the transformer."""

import math

import torch
import numpy as np
from torch import nn


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on 3D data.
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

    def forward(self, src):
        mask = torch.zeros_like(src[:, 0], dtype=torch.bool)
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


class PositionEmbeddingLearned3D(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, channels=128):
        super().__init__()
        self.orig_channels = channels
        channels = int(np.ceil(channels/6)*2)

        self.row_embed = nn.Embedding(50, channels)
        self.col_embed = nn.Embedding(50, channels)
        self.depth_embed = nn.Embedding(50, channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.depth_embed.weight)

    def forward(self, x):
        h, w, d = x.shape[-3:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        k = torch.arange(d, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        z_emb = self.depth_embed(k)
        pos = torch.cat([
            x_emb.unsqueeze(0).unsqueeze(2).repeat(h, 1, d, 1),
            y_emb.unsqueeze(1).unsqueeze(1).repeat(1, w, d, 1),
            z_emb.unsqueeze(0).unsqueeze(0).repeat(h, w, 1, 1),
        ], dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
        return pos[:,:self.orig_channels,...]

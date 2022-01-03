from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetLight(nn.Module):

    def __init__(
        self,
        out_channels,
        kernel_sizes,
        strides,
        return_intermediate_outputs=True
    ):
        super().__init__()

        self._return_intermediate_outputs = return_intermediate_outputs

        layer1 = Block(
            1, out_channels[0], kernel_sizes[0], strides[0]
        )
        layer2 = Block(
            out_channels[0], out_channels[1], kernel_sizes[1], strides[1]
        )
        layer3 = Block(
            out_channels[1], out_channels[2], kernel_sizes[2], strides[2]
        )

        self._layers = nn.ModuleList(
            [
                layer1,
                layer2,
                layer3
            ]
        )
        
    def forward(self, x, mask):
        out = []
        for layer in self._layers:
            x = layer(x)

            # Adjust mask via interpolation - True: masked, False: not masked
            mask_inter = F.interpolate(mask.float(), size=x.shape[-3:]).to(torch.bool).squeeze(1)
            out.append((x, mask_inter))

        # Decide which layer outputs to return
        if self._return_intermediate_outputs:
            return out
        else:
            return out[-1:]  # only return last output  



class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation=1
    ):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation
        )
        self.bn = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
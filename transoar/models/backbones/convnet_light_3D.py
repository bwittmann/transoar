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
        padding,
        return_intermediate_outputs=True,
        learnable=False
    ):
        super().__init__()

        self._return_intermediate_outputs = return_intermediate_outputs
        self._learnable = learnable

        layer1 = Block(
            1, out_channels[0], kernel_sizes[0], strides[0], padding[0]
        )
        layer2 = Block(
            out_channels[0], out_channels[1], kernel_sizes[1], strides[1], padding[1]
        )
        layer3 = Block(
            out_channels[1], out_channels[2], kernel_sizes[2], strides[2], padding[2]
        )
        layer4 = Block(
            out_channels[2], out_channels[3], kernel_sizes[3], strides[3], padding[3]
        )
        layer5 = Block(
            out_channels[3], out_channels[4], kernel_sizes[4], strides[4], padding[4]
        )

        if learnable:
            self._layers = nn.ModuleList(
                [
                    layer1,
                    layer2,
                    layer3,
                    layer4,
                    layer5
                ]
            )
        else:
            self._layers = [None, None, None]
        
    def forward(self, x, mask):
        out = []
        for layer in self._layers:
            if self._learnable:
                x = layer(x)
            else:
                x = F.interpolate(x, scale_factor=0.5)

            # Adjust mask via interpolation - True: masked, False: not masked
            mask_inter = F.interpolate(mask.float(), size=x.shape[-3:]).to(torch.bool).squeeze(1)
            out.append((x, mask_inter))

        # Decide which layer outputs to return
        if self._return_intermediate_outputs:
            return out[-3:]
        else:
            return out[-1:]  # only return last output  



class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1
    ):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )
        self.bn = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
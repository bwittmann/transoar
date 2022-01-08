import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(
    in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=False
):
    "3x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
    )


def norm_layer(norm, inplanes):
    if norm == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)
    return out


def activation_layer(activation, inplace=True):
    if activation == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)
    return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None
    ):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_layer(norm_cfg, planes)
        self.nonlin = activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out


class Backbone(nn.Module):

    arch_settings = {
        9: (ResBlock, (3, 3, 2))
    }


    def __init__(
        self,
        depth,
        in_channels=1,
        norm_cfg='BN',
        activation_cfg='ReLU',
    ):
        super(Backbone, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        self.conv1 = conv3x3x3(in_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=3, bias=False)
        self.norm1 = norm_layer(norm_cfg, 64)
        self.nonlin = activation_layer(activation_cfg, inplace=True)
        self.layer1 = self._make_layer(block, 192, layers[0], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        self.layer2 = self._make_layer(block, 384, layers[1], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        self.layer3 = self._make_layer(block, 384, layers[2], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        self.layers = []

        for m in self.modules():
            if isinstance(m, (nn.Conv3d)):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='ReLU'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3x3(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                norm_layer(norm_cfg, planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample,))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        out = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        out.append((x, F.interpolate(mask.float(), size=x.shape[-3:]).to(torch.bool).squeeze(1)))

        x = self.layer1(x)
        out.append((x, F.interpolate(mask.float(), size=x.shape[-3:]).to(torch.bool).squeeze(1)))
        x = self.layer2(x)
        out.append((x, F.interpolate(mask.float(), size=x.shape[-3:]).to(torch.bool).squeeze(1)))
        x = self.layer3(x)
        out.append((x, F.interpolate(mask.float(), size=x.shape[-3:]).to(torch.bool).squeeze(1)))

        return out[-2:]

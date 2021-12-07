from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.GroupNorm(num_groups=4, num_channels=planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=1,
        num_layers=4,
        conv1_t_size=7,
        conv1_t_stride=2,
        shortcut_type='B',
        widen_factor=1.0,
        strides=[1, 2, 2, 2],
        max_pool=True
    ):
        super().__init__()
        self.max_pool = max_pool

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, conv1_t_size, conv1_t_size),
                               stride=(conv1_t_stride, conv1_t_stride, conv1_t_stride),
                               padding=(conv1_t_size // 2, conv1_t_size // 2, conv1_t_size // 2),
                               bias=False)
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Changed kernel size to make deterministic, https://github.com/pytorch/pytorch/issues/23550
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

        layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type, stride=strides[0]
        )
        layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=strides[1]
        )

        layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=strides[2]
        )

        layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=strides[3]
        )

        all_layers = [layer1, layer2, layer3, layer4]
        self.layers = nn.ModuleList(all_layers[:num_layers])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.GroupNorm(num_groups=4, num_channels=planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mask):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.max_pool:
            x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        # Adjust mask via interpolation - True: masked, False: not masked
        mask = F.interpolate(mask.float(), size=x.shape[-3:]).to(torch.bool).squeeze(1)
        return x, mask

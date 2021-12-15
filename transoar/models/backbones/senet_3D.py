"""Module containing the SENet backbone, adapted from the monai repo."""

from collections import OrderedDict
from typing import Any, List, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.squeeze_and_excitation import SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck
from monai.networks.layers.factories import Act, Conv, Norm, Pool

class SENet(nn.Module):
    """
    SENet based on `Squeeze-and-Excitation Networks <https://arxiv.org/pdf/1709.01507.pdf>`_.
    Adapted from `Cadene Hub 2D version
    <https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py>`_.
    Args:
        spatial_dims: spatial dimension of the input data.
        in_channels: channel number of the input data.
        block: SEBlock class.
            for SENet154: SEBottleneck
            for SE-ResNet models: SEResNetBottleneck
            for SE-ResNeXt models:  SEResNeXtBottleneck
        layers: number of residual blocks for 4 layers of the network (layer1...layer4).
        groups: number of groups for the 3x3 convolution in each bottleneck block.
            for SENet154: 64
            for SE-ResNet models: 1
            for SE-ResNeXt models:  32
        reduction: reduction ratio for Squeeze-and-Excitation modules.
            for all models: 16
        dropout_prob: drop probability for the Dropout layer.
            if `None` the Dropout layer is not used.
            for SENet154: 0.2
            for SE-ResNet models: None
            for SE-ResNeXt models: None
        dropout_dim: determine the dimensions of dropout. Defaults to 1.
            When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).
        inplanes:  number of input channels for layer1.
            for SENet154: 128
            for SE-ResNet models: 64
            for SE-ResNeXt models: 64
        downsample_kernel_size: kernel size for downsampling convolutions in layer2, layer3 and layer4.
            for SENet154: 3
            for SE-ResNet models: 1
            for SE-ResNeXt models: 1
        input_3x3: If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        num_layers: number of layers in forward pass.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        block: Type[Union[SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck]],
        layers: Sequence[int],
        groups: int,
        reduction: int,
        strides: List[int],
        max_pool: True,
        inplanes: int = 128,
        downsample_kernel_size: int = 3,
        input_3x3: bool = True,
        num_layers: int = -1,
        return_intermediate_outputs=True
    ) -> None:

        super().__init__()

        relu_type: Type[nn.ReLU] = Act[Act.RELU]
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]

        self.inplanes = inplanes
        self.spatial_dims = spatial_dims
        self.return_intermediate_outputs = return_intermediate_outputs

        layer0_modules: List[Tuple[str, Any]]

        if input_3x3:
            layer0_modules = [
                (
                    "conv1",
                    conv_type(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                ),
                ("bn1", norm_type(num_features=64)),
                ("relu1", relu_type(inplace=True)),
                ("conv2", conv_type(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)),
                ("bn2", norm_type(num_features=64)),
                ("relu2", relu_type(inplace=True)),
                (
                    "conv3",
                    conv_type(in_channels=64, out_channels=inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                ),
                ("bn3", norm_type(num_features=inplanes)),
                ("relu3", relu_type(inplace=True)),
            ]
        else:
            layer0_modules = [
                (
                    "conv1",
                    conv_type(
                        in_channels=in_channels, out_channels=inplanes, kernel_size=7, stride=2, padding=3, bias=False
                    ),
                ),
                ("bn1", norm_type(num_features=inplanes)),
                ("relu1", relu_type(inplace=True)),
            ]

        if max_pool:
            # Changed kernel size to make deterministic, https://github.com/pytorch/pytorch/issues/23550
            layer0_modules.append(("pool", pool_type(kernel_size=2, stride=2, ceil_mode=True)))

        layer0 = nn.Sequential(OrderedDict(layer0_modules))
        layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            stride=strides[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1
        )
        layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=strides[1],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
        )
        layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=strides[2],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
        )
        layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=strides[3],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
        )
        all_layers = [layer0, layer1, layer2, layer3, layer4]
        self.layers = nn.ModuleList(all_layers[:num_layers + 1])

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _make_layer(
        self,
        block: Type[Union[SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck]],
        planes: int,
        blocks: int,
        groups: int,
        reduction: int,
        stride: int = 1,
        downsample_kernel_size: int = 1,
    ) -> nn.Sequential:

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.inplanes,
                out_channels=planes * block.expansion,
                strides=stride,
                kernel_size=downsample_kernel_size,
                act=None,
                norm=Norm.BATCH,
                bias=False,
            )

        layers = []
        layers.append(
            block(
                spatial_dims=self.spatial_dims,
                inplanes=self.inplanes,
                planes=planes,
                groups=groups,
                reduction=reduction,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for _num in range(1, blocks):
            layers.append(
                block(
                    spatial_dims=self.spatial_dims,
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=groups,
                    reduction=reduction,
                )
            )

        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor, mask: torch.Tensor):

        out = []
        for layer in self.layers:
            x = layer(x)
            # Adjust mask via interpolation - True: masked, False: not masked
            mask_inter = F.interpolate(mask.float(), size=x.shape[-3:]).to(torch.bool).squeeze(1)
            out.append((x, mask_inter))

        # Decide which layer outputs to return
        if self.return_intermediate_outputs:
            return out[2:]
        else:
            return out[-1:]  # only return last output

    def forward(self, x: torch.Tensor, mask: torch.tensor):
        return self.features(x, mask)

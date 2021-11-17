"""Module containing functionality to build different parts of the model."""

from transoar.models.backbones.senet_3D import SENet, SEResNetBottleneck

def build_backbone(backbone_config):
    if backbone_config['name'] == 'senet':
        model = SENet(
            block=SEResNetBottleneck,
            spatial_dims=3,
            in_channels=backbone_config['in_chans'],
            layers=backbone_config['depths'],
            groups=1,
            reduction=16,
            inplanes=64,
            downsample_kernel_size=1,
            input_3x3=False
        )
        
    return model

def build_neck():
    pass
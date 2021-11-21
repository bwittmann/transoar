"""Module containing functionality to build different parts of the model."""

from transoar.utils.io import get_config
from transoar.models.transoarnet import TransoarNet
from transoar.models.backbones.senet_3D import SENet, SEResNetBottleneck
from transoar.models.necks.detr_transformer import DetrTransformer

data_config = get_config('data_main')

def build_model(config):
    model = TransoarNet(
        config,
        data_config['num_classes']
    )
    return model

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

def build_neck(neck_config):
    if neck_config['name'] == 'detr':
        model = DetrTransformer(
            d_model=neck_config['hidden_dim'],
            dropout=neck_config['dropout'],
            nhead=neck_config['nheads'],
            dim_feedforward=neck_config['dim_feedforward'],
            num_encoder_layers=neck_config['enc_layers'],
            num_decoder_layers=neck_config['dec_layers'],
            normalize_before=neck_config['pre_norm'],
            return_intermediate_dec=True
        )

    return model

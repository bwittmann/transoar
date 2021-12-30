"""Module containing functionality to build different parts of the model."""

from transoar.models.matcher import HungarianMatcher
from transoar.models.criterion import TransoarCriterion
from transoar.models.backbones.senet_3D import SENet, SEResNetBottleneck
from transoar.models.backbones.resnet_3D import ResNet, Bottleneck, get_inplanes
from transoar.models.necks.detr_transformer import DetrTransformer
from transoar.models.necks.deformable_detr_transformer import DeformableTransformer
from transoar.models.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingLearned3D


def build_backbone(config):
    if config['name'] == 'senet':
        model = SENet(
            block=SEResNetBottleneck,
            spatial_dims=3,
            in_channels=config['in_chans'],
            layers=config['depths'],
            num_layers=config['num_layers'],
            groups=1,
            reduction=config['reduction'],
            strides=config['strides'],
            max_pool=config['pool'],
            inplanes=64,
            downsample_kernel_size=1,
            input_3x3=False,
            return_intermediate_outputs=config['return_intermediate_outputs']
        )
    if config['name'] == 'resnet':
        model = ResNet(
            block=Bottleneck,
            layers=config['depths'],
            block_inplanes=get_inplanes(),
            n_input_channels=config['in_chans'],
            num_layers=config['num_layers'],
            strides=config['strides'],
            max_pool=config['pool'],
            conv1_t_size=7,
            conv1_t_stride=2,
            shortcut_type='B',
            widen_factor=1.0,
            return_intermediate_outputs=config['return_intermediate_outputs']
        )

    return model

def build_neck(config):
    if config['name'] == 'detr':
        model = DetrTransformer(
            d_model=config['hidden_dim'],
            dropout=config['dropout'],
            nhead=config['nheads'],
            dim_feedforward=config['dim_feedforward'],
            num_encoder_layers=config['enc_layers'],
            num_decoder_layers=config['dec_layers'],
            normalize_before=config['pre_norm'],
            return_intermediate_dec=True
        )
    elif config['name'] == 'def_detr':
        model = DeformableTransformer(
            d_model=config['hidden_dim'],
            nhead=config['nheads'],
            num_encoder_layers=config['enc_layers'],
            num_decoder_layers=config['dec_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=config['num_feature_levels'],
            dec_n_points=config['dec_n_points'],
            enc_n_points=config['enc_n_points'],
            use_cuda=config['use_cuda']
        )  

    return model

def build_criterion(config):
    matcher = HungarianMatcher(
        cost_class=config['set_cost_class'],
        cost_bbox=config['set_cost_bbox'],
        cost_giou=config['set_cost_giou']
    )

    criterion = TransoarCriterion()

    return criterion

def build_pos_enc(config):
    channels = config['hidden_dim']
    if config['pos_encoding'] == 'sine':
        return PositionEmbeddingSine3D(channels=channels)
    elif config['pos_encoding'] == 'learned':
        return PositionEmbeddingLearned3D(channels=channels)
    else:
        raise ValueError('Please select a implemented pos. encoding.')

"""Module containing functionality to build different parts of the model."""

from transoar.models.matcher import HungarianMatcher
from transoar.models.criterion import TransoarCriterion
from transoar.models.backbones.attn_fpn.attn_fpn import AttnFPN
from transoar.models.necks.def_detr_transformer import DeformableTransformer
from transoar.models.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingLearned3D


def build_backbone(config):
    model = AttnFPN(
        config
    )
    return model

def build_neck(config):
    model = DeformableTransformer(
        d_model=config['hidden_dim'],
        nhead=config['nheads'],
        num_encoder_layers=config['enc_layers'],
        num_decoder_layers=config['dec_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        activation="relu",
        return_intermediate_dec=True,
        dec_n_points=config['dec_n_points'],
        enc_n_points=config['enc_n_points'],
        use_cuda=config['use_cuda'],
        use_encoder=config['use_encoder'],
        num_feature_levels=config['num_feature_levels']
    ) 

    return model

def build_criterion(config):
    matcher = HungarianMatcher(
        cost_class=config['set_cost_class'],
        cost_bbox=config['set_cost_bbox'],
        cost_giou=config['set_cost_giou']
    )

    criterion = TransoarCriterion(
        num_classes=config['num_classes'],
        matcher=matcher,
        seg_proxy=config['backbone']['use_seg_proxy_loss'],
        seg_fg_bg=config['backbone']['fg_bg']
    )

    return criterion

def build_pos_enc(config):
    channels = config['hidden_dim']
    if config['pos_encoding'] == 'sine':
        return PositionEmbeddingSine3D(channels=channels)
    elif config['pos_encoding'] == 'learned':
        return PositionEmbeddingLearned3D(channels=channels)
    else:
        raise ValueError('Please select a implemented pos. encoding.')

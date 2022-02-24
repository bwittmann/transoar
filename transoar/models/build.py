"""Module containing functionality to build different parts of the model."""

from transoar.models.matcher import HungarianMatcher
from transoar.models.criterion import TransoarCriterion
from transoar.models.necks.detr_transformer import DetrTransformer
from transoar.models.necks.deformable_detr_transformer import DeformableTransformer
from transoar.models.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingLearned3D


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

def build_def_attn_encoder(config):
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

    criterion = TransoarCriterion(
        num_classes=config['num_classes'],
        matcher=matcher
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

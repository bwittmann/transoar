"""Module containing functionality to build different parts of the model."""

from transoar.models.matcher import HungarianMatcher
from transoar.models.criterion import TransoarCriterion
from transoar.models.backbones.attn_fpn.attn_fpn import AttnFPN
from transoar.models.backbones.attn_fpn.resnet import resnet10, resnet18, resnet34, resnet50
from transoar.models.necks.focused_decoder import FocusedDecoder
from transoar.models.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingLearned3D


def build_backbone(config):
    if config['name'] =='attnfpn':
        model = AttnFPN(
            config
        )
    elif config['name'] == 'resnet10':
        model = resnet10()
    elif config['name'] == 'resnet18':
        model = resnet18()
    elif config['name'] == 'resnet34':
        model = resnet34()
    elif config['name'] == 'resnet50':
        model = resnet50()
    return model

def build_neck(config, bbox_props, anchors):
    model = FocusedDecoder(
        d_model=config['hidden_dim'],
        nhead=config['nheads'],
        num_decoder_layers=config['dec_layers'], 
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        activation="relu",
        return_intermediate_dec=True,
        bbox_props=bbox_props,
        config=config,
        anchors=anchors
    )

    return model

def build_criterion(config):
    matcher = HungarianMatcher(
        cost_class=config['set_cost_class'],
        cost_bbox=config['set_cost_bbox'],
        cost_giou=config['set_cost_giou'],
        anchor_matching=config['anchor_matching']
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

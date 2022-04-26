"""Module containing code of the transoar projects backbone."""

from copy import deepcopy

import torch
import torch.nn as nn

from transoar.models.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingLearned3D
from transoar.models.backbones.attn_fpn.decoder_blocks import DecoderDefAttnBlock
from transoar.models.backbones.attn_fpn.encoder_blocks import (
    EncoderCnnBlock,
    EncoderSwinBlock,
    PatchMerging,
    ConvPatchMerging
)


class AttnFPN(nn.Module):
    def __init__(self, fpn_config, debug=False):
        super().__init__()

        # Build encoder and decoder
        self._encoder = Encoder(fpn_config, debug)
        self._decoder = Decoder(fpn_config, debug)

    def forward(self, src):
        down = self._encoder(src)
        up = self._decoder(down)
        return up

    def init_weights(self):
        pass    # TODO

class Decoder(nn.Module):
    def __init__(self, config, debug):
        super().__init__()
        self._debug = debug
        self._num_stages = len(config['conv_kernels'])
        self._refine_fmaps = config['use_decoder_attn']
        self._refine_feature_levels = config['feature_levels']
        self._seg_proxy = config['use_seg_proxy_loss']

        # Determine channels of encoder fmaps
        encoder_out_channels = torch.tensor([config['start_channels'] * 2**stage for stage in range(self._num_stages)])

        # Estimate required stages
        all_stages = config['out_fmaps'] + config['feature_levels'] if config['use_decoder_attn'] else config['out_fmaps']
        required_stages = set([int(fmap[-1]) for fmap in all_stages])
        if self._seg_proxy:
            required_stages.add(0)
        self._required_stages = required_stages

        earliest_required_stage = min(required_stages)

        # LATERAL
        # Reduce lateral connections if not needed
        lateral_in_channels = encoder_out_channels if self._seg_proxy else encoder_out_channels[earliest_required_stage:]
        lateral_out_channels = lateral_in_channels.clip(max=config['fpn_channels'])

        self._lateral = nn.ModuleList()
        for in_channels, out_channels in zip(lateral_in_channels, lateral_out_channels):
            self._lateral.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        self._lateral_levels = len(self._lateral)

        # OUT
        # Ensure that relevant stages have channels according to fpn_channels
        out_in_channels = [lateral_out_channels[-self._num_stages + required_stage].item() for required_stage in required_stages]
        out_out_channels = torch.full((len(out_in_channels),), int(config['fpn_channels'])).tolist()
        out_out_channels[0] = encoder_out_channels[0] if self._seg_proxy else int(config['fpn_channels'])

        self._out = nn.ModuleList()
        for in_channels, out_channels in zip(out_in_channels, out_out_channels):
            self._out.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))

        #  UP
        self._up = nn.ModuleList()
        for level in range(len(lateral_out_channels)-1):
            self._up.append(
                nn.ConvTranspose3d(
                    in_channels=list(reversed(lateral_out_channels))[level], out_channels=list(reversed(lateral_out_channels))[level+1],
                    kernel_size=list(reversed(config['strides']))[level], stride=list(reversed(config['strides']))[level]
                )
            )
        
        # REFINE
        if self._refine_fmaps:
            # Build positional encoding
            if config['pos_encoding'] == 'sine':
                self._pos_enc = PositionEmbeddingSine3D(channels=config['hidden_dim'])
            elif config['pos_encoding'] == 'learned':
                self._pos_enc = PositionEmbeddingLearned3D(channels=config['hidden_dim'])

            # Build deformable arrention module
            self._refine = DecoderDefAttnBlock(
                d_model=config['hidden_dim'],
                nhead=config['nheads'],
                num_layers=config['layers'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout'],
                feature_levels=config['feature_levels'],
                n_points=config['n_points'],
                use_cuda=config['use_cuda']
            )

    def forward(self, x):
        # Forward lateral
        lateral_out = [lateral(fmap) for lateral, fmap in zip(self._lateral, list(x.values())[-self._lateral_levels:])]

        # Forward up
        up_out = []
        for idx, x in enumerate(reversed(lateral_out)):
            if idx != 0:
                x = x + up
            
            if idx < self._lateral_levels - 1:
                up = self._up[idx](x)

            up_out.append(x)

        # Forward out
        if self._seg_proxy:
            out_fmaps = [(list(reversed(up_out))[stage], stage) for stage in self._required_stages]
        else:
            out_fmaps = zip(reversed(up_out), self._required_stages)

        outputs = {'P' + str(stage): self._out[idx](fmap) for idx, (fmap, stage) in enumerate(out_fmaps)}

        # Forward refine
        if self._refine_fmaps:
            fmaps = [outputs[fmap_id] for fmap_id in self._refine_feature_levels]
            pos_enc = [self._pos_enc(fmap) for fmap in fmaps]
            fmaps_refined = self._refine(fmaps, pos_enc)

            # Update output dict
            for fmap_id, fmap_refined in zip(self._refine_feature_levels, fmaps_refined):
                outputs[fmap_id] = fmap_refined

        # Print shapes for debugging
        if self._debug:
            print('AttnFPN decoder shapes:')
            for fmap_id, fmap in outputs.items():
                print(fmap_id, list(fmap.shape))
                self._debug = False

        return outputs


class Encoder(nn.Module):
    def __init__(self, config, debug):
        super().__init__()
        self._debug = debug

        # Get initial channels
        in_channels = config['in_channels']
        out_channels = config['start_channels']

        # Get number of encoder stages
        num_stages = len(config['conv_kernels'])

        # Define stochastic depth for drop path of swin blocks
        swin_depth = config['depths']
        drop_path_rate = [x.item() for x in torch.linspace(0, config['drop_path_rate'], sum(swin_depth))]

        # Define downsample operation for swin blocks
        downsample_layer = ConvPatchMerging if config['conv_merging'] else PatchMerging

        # Down
        self._stages = nn.ModuleList()
        for stage_id in range(num_stages):

            # Get encoder blocks
            if config['use_encoder_attn'] and stage_id > 1: # Initial patch embedding done with convs
                stage = EncoderSwinBlock(
                    dim=in_channels,
                    depth=config['depths'][stage_id - 2],
                    num_heads=config['num_heads'][stage_id - 2],
                    window_size=config['window_size'],
                    mlp_ratio=config['mlp_ratio'],
                    qkv_bias=config['qkv_bias'],
                    qk_scale=config['qk_scale'],
                    drop=config['drop_rate'],
                    attn_drop=config['attn_drop_rate'],
                    drop_path=drop_path_rate[sum(swin_depth[:stage_id - 2]):sum(swin_depth[:stage_id - 1])],
                    downsample=downsample_layer
                )
            else:
                stage = EncoderCnnBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config['conv_kernels'][stage_id],
                    stride=config['strides'][stage_id]
                )

            self._stages.append(stage)

            in_channels = out_channels
            out_channels *= 2

    def forward(self, x):
        # Forward down
        outputs = {}
        for stage_id, module in enumerate(self._stages):
            x = module(x)
            outputs['C' + str(stage_id)] = x

        # Print shapes for debugging
        if self._debug:
            print('AttnFPN encoder shapes:')
            for fmap_id, fmap in outputs.items():
                print(fmap_id, list(fmap.shape))
                self._debug = False

        return outputs

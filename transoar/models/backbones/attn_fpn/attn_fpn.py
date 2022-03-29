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

        # Determine channels
        out_channels = torch.tensor([config['start_channels'] * 2**stage for stage in range(self._num_stages)])
        encoder_out_channels = out_channels.tolist()
        decoder_out_channels = out_channels.clip(max=config['fpn_channels']).tolist()

        # Lateral 
        self._lateral = nn.ModuleList()
        for in_channels, out_channels in zip(encoder_out_channels, decoder_out_channels):
            self._lateral.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))

        # Out
        self._out = nn.ModuleList()
        # Ensure that relevant stages have channels according to fpn_channels
        earliest_required_stage = min([int(config['out_fmap'][-1]), int(config['feature_levels'][0][-1])])
        num_required_stages = self._num_stages - earliest_required_stage    
        final_out_channels = deepcopy(decoder_out_channels)
        final_out_channels[-num_required_stages:] = [config['fpn_channels'] for _ in final_out_channels[-num_required_stages:]]

        for out_channels, in_channels in zip(final_out_channels, decoder_out_channels):
            self._out.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))  # TODO: kernel_size 1

        #  Up
        self._up = nn.ModuleList()
        for level in range(1, len(decoder_out_channels)):
            self._up.append(
                nn.ConvTranspose3d(
                    in_channels=decoder_out_channels[level], out_channels=decoder_out_channels[level-1],
                    kernel_size=config['strides'][level], stride=config['strides'][level]
                )
            )
        
        # Refine
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
        fpn_maps = [self._lateral[level](fm) for level, fm in enumerate(x.values())]

        # Forward up
        out_up = []
        for idx, x in enumerate(reversed(fpn_maps), 1):
            level = self._num_stages - idx - 1

            if idx != 1:
                x = x + up

            if idx != self._num_stages:
                up = self._up[level](x)

            out_up.append(x)

        # Forward out
        outputs = {'P' + str(level): self._out[level](fm) for level, fm in enumerate(reversed(out_up))}

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

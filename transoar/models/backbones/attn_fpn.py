"""Module containing code of the transoar projects backbone."""

from math import floor

import torch
import torch.nn as nn

from transoar.models.backbones.encoder_blocks import (
    EncoderCnnBlock,
    EncoderSwinBlock,
    PatchMerging,
    ConvPatchMerging
)

class AttnFPN(nn.Module):
    def __init__(self, fpn_config):
        super().__init__()

        # Build encoder and decoder
        self._encoder = Encoder(fpn_config)
        self._decoder = Decoder(fpn_config)

    def forward(self, src):
        down = self._encoder(src)
        up = self._decoder(down)
        return up

    def init_weights(self):
        pass    # TODO

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._num_stages = len(config['conv_kernels'])

        # Determine channels
        out_channels = torch.tensor([config['start_channels'] * 2**stage for stage in range(self._num_stages)])
        encoder_out_channels = out_channels.clip(max=config['max_channels']).tolist()
        decoder_out_channels = out_channels.clip(max=config['fpn_channels']).tolist()

        # Lateral 
        self._lateral = nn.ModuleList()
        for in_channels, out_channels in zip(encoder_out_channels, decoder_out_channels):
            self._lateral.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))

        # Out
        self._out = nn.ModuleList()
        for out_channels in decoder_out_channels:
            self._out.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1))

        #  Up
        self._up = nn.ModuleList()
        for level in range(1, len(decoder_out_channels)):
            self._up.append(
                nn.ConvTranspose3d(
                    in_channels=decoder_out_channels[level], out_channels=decoder_out_channels[level-1],
                    kernel_size=config['strides'][level], stride=config['strides'][level]
                )
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
        return outputs

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        return outputs
import torch
import torch.nn as nn

from transoar.models.head import Head


class RetinaUNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._encoder = Encoder(config)
        decoder_channels = self._encoder.out_channels
        decoder_strides = self._encoder.out_strides

        self._decoder = Decoder(config, decoder_channels, decoder_strides)

        self._head = Head(config)
        pass

    def forward(self, x):
        encoder_out = self._encoder(x)
        decoder_out = self._decoder(encoder_out)
 
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self._config = config
        in_channels = config['in_channels']
        out_channels = config['start_channels']

        num_stages = len(config['conv_kernels'])
        self._out_stages = list(range(num_stages))

        self.out_channels = []
        self.out_strides = []
        self._stages = nn.ModuleList()
        for stage_id in range(num_stages):
            self.out_channels.append(out_channels)

            if len(self.out_strides) == 0:
                self.out_strides.append(config['strides'][stage_id])
            else:
                self.out_strides.append(config['strides'][stage_id]) # * self.out_strides[-1])

            
            stage = EncoderBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config['conv_kernels'][stage_id],
                stride=config['strides'][stage_id]
            )
            self._stages.append(stage)

            in_channels = out_channels
            out_channels = out_channels * 2

            if out_channels > config['max_channels']:
                out_channels = config['max_channels']

    def forward(self, x):
        outputs = []
        for stage_id, module in enumerate(self.stages):
            x = module(x)
            if stage_id in self.out_stages:
                outputs.append(x)
        return outputs 

class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=1,
        bias=False,
        affine=True,
        eps=1e-05

    ):
        super().__init__()

        conv_block_1 = [
            nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=bias
            ),
            nn.InstanceNorm3d(num_features=out_channels, affine=affine, eps=eps),
            nn.ReLU(inplace=True)
        ]

        conv_block_2 = [
            nn.Conv3d(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, padding=padding,
                bias=bias
            ),
            nn.InstanceNorm3d(num_features=out_channels, affine=affine, eps=eps),
            nn.ReLU(inplace=True)
        ]

        self._block = nn.Sequential(
            *conv_block_1,
            *conv_block_2
        )

    def forward(self, x):
        return self._block(x)

class Decoder(nn.Module):
    def __init__(self, config, encoder_out_channels, strides):
        super().__init__() 
        self._num_levels = len(encoder_out_channels)

        decoder_out_channels = torch.clip(torch.tensor(encoder_out_channels), max=(config['fpn_channels'])).tolist()

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
            self._up.append(nn.ConvTranspose3d(
                in_channels=decoder_out_channels[level], out_channels=decoder_out_channels[level-1],
                kernel_size=config['strides'][level], stride=config['strides'][level]
                ))

    def forward(self, x):
        out_list = []

        # Forward lateral
        fpn_maps = [self._lateral[level](fm) for level, fm in enumerate(x)]

        # Forward up
        for idx, x in enumerate(reversed(fpn_maps), 1):
            level = self._num_level - idx

            if idx != 1:
                x = x + up

            if idx != self._num_level:
                up = self._up[level](x)

            out_list.append(x)

        # Forward out
        out_list = [self.out[level](fm) for level, fm in enumerate(reversed(out_list))]

        return out_list

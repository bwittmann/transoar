""" 
This module contains the FPN backbone.
adapted from: https://github.com/MIC-DKFZ/medicaldetectiontoolkit/blob/master/models/backbone.py
"""

import torch.nn as nn
import torch.nn.functional as F

from transoar.models.backbones.down import NDConvGenerator, ResDown, SwinDown

class SwinFPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(
        self,
        # FPN
        start_filts, end_filts, res_architecture, sixth_pooling, operate_stride1, 
        n_channels, norm, relu, n_latent_dims,
        # Swin
        use_swin, pretrained, pretrained2d, patch_size, in_chans, embed_dim, depths, num_heads,
        window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
        norm_layer, patch_norm, frozen_stages, use_checkpoint
    ):
        """
        from configs:
        :param input_channels: number of channel dimensions in input data.
        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
        :param out_channels: number of feature_maps for output_layers of all levels in decoder.
        :param conv: instance of custom conv class containing the dimension info.
        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super().__init__()
        conv = NDConvGenerator(dim=3)
        self.sixth_pooling = sixth_pooling
        self.operate_stride1 = operate_stride1

        # Down
        if not use_swin:
            self.down = ResDown(
                operate_stride1, sixth_pooling, n_channels, start_filts, norm, relu, res_architecture
            )
        else:
            self.down = SwinDown(
                pretrained, pretrained2d, patch_size, in_chans, embed_dim, depths, num_heads, window_size,
                mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, patch_norm,
                frozen_stages, use_checkpoint
            )

        # Up
        self.P1_upsample = Interpolate(scale_factor=(2, 2, 2), mode='trilinear')
        self.P2_upsample = Interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        self.out_channels = end_filts
        self.P5_conv1 = conv(start_filts*32 + n_latent_dims, self.out_channels, ks=1, stride=1, relu=None) #
        self.P4_conv1 = conv(start_filts*16, self.out_channels, ks=1, stride=1, relu=None)
        self.P3_conv1 = conv(start_filts*8, self.out_channels, ks=1, stride=1, relu=None)
        self.P2_conv1 = conv(start_filts*4, self.out_channels, ks=1, stride=1, relu=None)
        self.P1_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)

        if operate_stride1:
            self.P0_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)
            self.P0_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        self.P1_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P2_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P3_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P4_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P5_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        if self.sixth_pooling:
            self.P6_conv1 = conv(start_filts * 64, self.out_channels, ks=1, stride=1, relu=None)
            self.P6_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        # Down
        out_down = self.down(x)

        # Up
        if self.sixth_pooling:
            p6_pre_out = self.P6_conv1(out_down[6])
            p5_pre_out = self.P5_conv1(out_down[5]) + F.interpolate(p6_pre_out, scale_factor=2)
        else:
            p5_pre_out = self.P5_conv1(out_down[5])

        p4_pre_out = self.P4_conv1(out_down[4]) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(out_down[3]) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(out_down[2]) + F.interpolate(p3_pre_out, scale_factor=2)

        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]

        if self.sixth_pooling:
            p6_out = self.P6_conv2(p6_pre_out)
            out_list.append(p6_out)

        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(out_down[1]) + self.P2_upsample(p2_pre_out)
            p0_pre_out = self.P0_conv1(out_down[0]) + self.P1_upsample(p1_pre_out)
            # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list

        return out_list


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

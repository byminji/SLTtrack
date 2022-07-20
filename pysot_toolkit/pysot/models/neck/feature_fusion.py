import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from pysot.models.head.conv_module import ConvModule
from pysot.models.head.rpn import DepthwiseXCorr

class FeatureFusionNeck(nn.Module):
    def __init__(self,
                 num_ins,
                 fusion_level,
                 in_channels=[256, 256, 256],
                 conv_out_channels=256):
        super(FeatureFusionNeck, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels

        assert num_ins == len(in_channels), "num_ins must equal to length of in_channel."

        self.xcorr = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.xcorr.append(DepthwiseXCorr(self.in_channels[i],
                                             self.in_channels[i],
                                             self.conv_out_channels))

        self.convs = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.convs.append(
                ConvModule(
                    self.in_channels[i],
                    self.conv_out_channels,
                    1,
                    inplace=False))

    def forward(self, z_fs, x_fs):
        # xcorr each layer -> elementwise sum
        b_feats = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs)):
            out, _ = self.xcorr[idx](z_f, x_f)
            b_feats.append(out)

        y = self.convs[0](b_feats[0])
        for i, feat in enumerate(b_feats[1:], start=1):
            y = y + self.convs[i](feat)
        return y


class FeatureFusionAllNeck(nn.Module):
    def __init__(self,
                 num_ins,
                 fusion_level,
                 in_channels=[64, 256, 256, 256, 256],
                 conv_out_channels=256):
        super(FeatureFusionAllNeck, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels

        assert num_ins == len(in_channels), "num_ins must equal to length of in_channel."

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                ConvModule(
                    self.in_channels[i],
                    self.conv_out_channels,
                    1,
                    inplace=False))

        self.xcorr = nn.ModuleList()
        for i in range(len(self.in_channels[2:])):
            self.xcorr.append(DepthwiseXCorr(self.in_channels[i+2],
                                             self.in_channels[i+2],
                                             self.conv_out_channels))

        self.convs = nn.ModuleList()
        for i in range(len(self.in_channels[2:])):
            self.convs.append(
                ConvModule(
                    self.in_channels[i+2],
                    self.conv_out_channels,
                    1,
                    inplace=False))

    def forward(self, z_fs, x_fs):
        m_feats = x_fs
        b_feats = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs[2:], x_fs[2:])):
            out, _ = self.xcorr[idx](z_f, x_f)
            b_feats.append(out)

        x = self.lateral_convs[self.fusion_level](m_feats[self.fusion_level])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(m_feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                x = x + self.lateral_convs[i](feat)

        y = self.convs[0](b_feats[0])
        for i, feat in enumerate(b_feats[1:], start=1):
            y = y + self.convs[i](feat)

        return y, x

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.head.rpn import DepthwiseXCorr
from pysot.core.xcorr import xcorr_depthwise
from pysot.models.head.conv_module import ConvModule
from pysot.models.init_weight import kaiming_init
from pysot.models.head.dcn.deform_pool import DeformRoIPoolingPack, ModulatedDeformRoIPoolingPack
import torchvision.ops as ops


class MaskCorr(DepthwiseXCorr):
    def __init__(self, in_channels, hidden, out_channels,
                 kernel_size=3, hidden_kernel_size=5):
        super(MaskCorr, self).__init__(in_channels, hidden,
                                       out_channels, kernel_size,
                                       hidden_kernel_size)

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out, feature


class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.v1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.v2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.h2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.h1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.h0 = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)
        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, f, corr_feature, pos):
        p0 = F.pad(f[0], [16, 16, 16, 16])[:, :, 4 * pos[0]:4 * pos[0] + 61, 4 * pos[1]:4 * pos[1] + 61]
        p1 = F.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
        p2 = F.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]

        p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127 * 127)
        return out


class Mask_FCN_Head(nn.Module):

    def __init__(self, dim_in, out_res, upsample_ratio, use_fc=False):
        super(Mask_FCN_Head, self).__init__()
        self.dim_in = dim_in
        self.use_fc = use_fc
        self.out_res = out_res
        self.upsample_ratio = upsample_ratio

        n_classes = 1  # foreground and background

        if self.use_fc:
            self.classify = nn.Linear(dim_in, n_classes * self.out_res ** 2)
        else:
            # Predict mask using Conv
            # self.classify = nn.Conv2d(dim_in, n_classes, 1, 1, 0)
            self.classify = nn.Sequential(
                nn.Conv2d(dim_in, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
            if self.upsample_ratio > 1:
                self.upsample = BilinearInterpolation2d(
                    n_classes, n_classes, self.upsample_ratio)

        # self._init_weights()

    def _init_weights(self):
        init.normal_(self.classify[0].weight, std=0.001)
        init.constant_(self.classify[0].bias, 0)
        init.normal_(self.classify[2].weight, std=0.001)
        init.constant_(self.classify[2].bias, 0)

    def forward(self, x):
        x = self.classify(x)
        if self.upsample_ratio > 1:
            x = self.upsample(x)
        if not self.training:
            x = torch.sigmoid(x)
        return x


class BilinearInterpolation2d(nn.Module):
    """Bilinear interpolation in space of scale.
    Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale
    Adapted from the CVPR'15 FCN code.
    See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """

    def __init__(self, in_channels, out_channels, up_scale):
        super().__init__()
        assert in_channels == out_channels
        assert up_scale % 2 == 0, 'Scale should be even'
        self.in_channes = in_channels
        self.out_channels = out_channels
        self.up_scale = int(up_scale)
        self.padding = up_scale // 2

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return ((1 - abs(og[0] - center) / factor) *
                    (1 - abs(og[1] - center) / factor))

        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)

        kernel = np.zeros(
            (in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32
        )
        kernel[range(in_channels), range(out_channels), :, :] = bil_filt

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=self.up_scale, padding=self.padding)

        self.upconv.weight.data.copy_(torch.from_numpy(kernel))
        self.upconv.bias.data.fill_(0)
        self.upconv.weight.requires_grad = False
        self.upconv.bias.requires_grad = False

    def forward(self, x):
        return self.upconv(x)


class FCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=1
                 ):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred


class FusedSemanticHead(nn.Module):

    def __init__(self,
                 pooling_func,
                 num_convs=4,
                 in_channels=256,
                 conv_out_channels=256,
                 upsample_ratio=2,
                 num_classes=1):
        super(FusedSemanticHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.pooling_func = pooling_func

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.conv_out_channels
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    padding=1))

        self.conv_logits = nn.Conv2d(self.conv_out_channels, self.num_classes, 1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.ConvTranspose2d(
            self.conv_out_channels,
            self.conv_out_channels,
            self.upsample_ratio,
            stride=self.upsample_ratio)

        self.roi_pool_m = DeformRoIPoolingPack(spatial_scale=1 / (cfg.TRAIN.SEARCH_SIZE / 63),
                                               out_size=cfg.TRAIN.ROIPOOL_OUTSIZE,
                                               out_channels=256,
                                               no_trans=False,
                                               trans_std=0.1)
        '''
        self.roi_pool_m = ModulatedDeformRoIPoolingPack(spatial_scale=1 / (cfg.TRAIN.SEARCH_SIZE / 63),
                                               out_size=cfg.TRAIN.ROIPOOL_OUTSIZE,
                                               out_channels=256,
                                               no_trans=False,
                                               trans_std=0.1,
                                               deform_fc_channels=512)

        self.roi_pool_m = ops.RoIAlign(output_size=(cfg.TRAIN.ROIPOOL_OUTSIZE, cfg.TRAIN.ROIPOOL_OUTSIZE),
                                     spatial_scale=1 / (cfg.TRAIN.SEARCH_SIZE / 63),
                                     sampling_ratio=-1)
        '''

        self.init_weights()

    def init_weights(self):
        kaiming_init(self.conv_logits)

    def forward(self, x, roi_list):

        x = self.roi_pool_m(x, roi_list)
        for i in range(self.num_convs):
            x = self.convs[i](x)

        if self.upsample is not None:
            x = self.upsample(x)
            x = self.relu(x)

        mask_pred = self.conv_logits(x)

        return mask_pred

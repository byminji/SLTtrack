# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, select_focal_loss, det_loss_smooth_l1, mask_loss_bce
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head
from pysot.models.neck import get_neck
from pysot.models.neck.feature_fusion import FeatureFusionAllNeck
from pysot.models.neck.enhance import FeatureEnhance
from pysot.models.head.detection import FCx2DetHead
from pysot.models.head.mask import FusedSemanticHead

from pysot.utils.mask_target_builder import build_proposal_target, build_mask_target

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        if cfg.ENHANCE.ENHANCE:
            self.feature_enhance = FeatureEnhance(in_channels=256, out_channels=256)

        if cfg.REFINE.REFINE:
            self.feature_fusion = FeatureFusionAllNeck(num_ins=5, fusion_level=1,
                                                    in_channels=[64, 256, 256, 256, 256], conv_out_channels=256)
            self.refine_head = FCx2DetHead(pooling_func=None,
                                         in_channels=256 * (cfg.TRAIN.ROIPOOL_OUTSIZE // 4)**2)

        if cfg.MASK.MASK:
            self.mask_head = FusedSemanticHead(pooling_func=None,
                                               num_convs=4, in_channels=256,
                                               upsample_ratio=(cfg.MASK.MASK_OUTSIZE // cfg.TRAIN.ROIPOOL_OUTSIZE))

    def template(self, z):
        with torch.no_grad():
            zf = self.backbone(z)
            if cfg.ADJUST.ADJUST:
                zf[2:] = self.neck(zf[2:])
            self.zf = zf

    def track(self, x):
        with torch.no_grad():
            xf = self.backbone(x)
            if cfg.ADJUST.ADJUST:
                xf[2:] = self.neck(xf[2:])

            zf, xf[2:] = self.feature_enhance(self.zf[2:], xf[2:])
            cls, loc, _, _ = self.rpn_head(zf, xf[2:])
            enhanced_zf = self.zf[:2] + zf
            if cfg.MASK.MASK:
                self.b_fused_features, self.m_fused_features = self.feature_fusion(enhanced_zf, xf)
            return {
                'cls': cls,
                'loc': loc
            }

    def bbox_refine(self, roi):
        with torch.no_grad():
            bbox_pred = self.refine_head(self.b_fused_features, roi)
            return bbox_pred

    def mask_refine(self, roi):
        with torch.no_grad():
            mask_pred = self.mask_head(self.m_fused_features, roi)
        return mask_pred

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.contiguous().view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1)
        cls = F.log_softmax(cls, dim=4)
        return cls

    def convert_loc(self, loc, anchor):
        b, k4, h, w = loc.size() # batch_num, anchor_num*4, size1, size2
        k = k4//4
        loc = loc.contiguous().view(b, 4, k, h, w)
        loc = loc.permute(0, 2, 3, 4, 1).contiguous().view(b, k * h * w, 4)  # b, (k*h*w), 4
        anchor = anchor.permute(0, 2, 3, 4, 1).contiguous().view(b, k*h*w, 4)  # b, (k*h*w), 4
        loc[:, :, 0] = loc[:, :, 0] * anchor[:, :, 2] + anchor[:, :, 0]   # cx
        loc[:, :, 1] = loc[:, :, 1] * anchor[:, :, 3] + anchor[:, :, 1]   # cy
        loc[:, :, 2] = torch.exp(loc[:, :, 2]) * anchor[:, :, 2]       # w
        loc[:, :, 3] = torch.exp(loc[:, :, 3]) * anchor[:, :, 3]       # h

        # x1, y1, x2, y2
        x1 = loc[:, :, 0] - loc[:, :, 2] / 2.
        y1 = loc[:, :, 1] - loc[:, :, 3] / 2.
        x2 = loc[:, :, 0] + loc[:, :, 2] / 2.
        y2 = loc[:, :, 1] + loc[:, :, 3] / 2.
        return torch.stack((x1, y1, x2, y2), dim=2)

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        search_mask = data['search_mask'].cuda()
        mask_weight = data['mask_weight'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        anchor = data['anchor_center'].cuda()
        gt_bboxes = data['bbox'].cuda()
        neg = data['neg'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf[2:] = self.neck(zf[2:])
            xf[2:] = self.neck(xf[2:])

        if cfg.ENHANCE.ENHANCE:
            zf[2:], xf[2:] = self.feature_enhance(zf[2:], xf[2:])
        cls, loc, _, _ = self.rpn_head(zf[2:], xf[2:])

        # get loss
        cls = self.log_softmax(cls)

        if cfg.TRAIN.FOCAL_LOSS:
            cls_loss = select_focal_loss(cls, label_cls, cfg.TRAIN.FOCAL_GAMMA)
        else:
            cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.REFINE.REFINE:
            # Convert loc coordinate to (x1,y1,x2,y2)
            loc = loc.detach()
            bbox = self.convert_loc(loc, anchor)

            # Randomly sample proposals from rpn prediction
            # Select proposals only for positive training pairs
            batch_select = neg.data.eq(0).nonzero().squeeze().cuda()
            rois, cls_labels, regression_target, weights = build_proposal_target(bbox, gt_bboxes, batch_select)

            mask_targets = build_mask_target(rois, cls_labels, search_mask)

            # For deformable RoI pooling
            num_batch = batch_select.size()[0]
            batch_select = batch_select.unsqueeze(1).expand(num_batch, cfg.TRAIN.ROI_PER_IMG)\
                            .reshape(num_batch*cfg.TRAIN.ROI_PER_IMG, 1).float()
            rois = torch.cat((batch_select, rois.view(-1, 4).float()), dim=1)

            # Feature fusion
            # b_fused_features : b, 256, 25, 25
            # m_fused_features : b, 256, 63, 63
            b_fused_features, m_fused_features = self.feature_fusion(zf, xf)

            # Refinement
            bbox_refine = self.refine_head(b_fused_features, rois)

            # Mask prediction
            mask_pred = self.mask_head(m_fused_features, rois)

            mask_pred = mask_pred.view_as(mask_targets)

            # Compute loss
            # loss_refine = refine_weight_l1_loss(bbox_refine, regression_target, weights)
            loss_refine = det_loss_smooth_l1(bbox_refine, regression_target)

            # mask_weight = (~(search_mask.sum(dim=(1,2)).data.eq(0))).int()
            loss_mask = mask_loss_bce(mask_pred, mask_targets, mask_weight)

            outputs['total_loss'] += (cfg.TRAIN.REFINE_WEIGHT*loss_refine + cfg.TRAIN.MASK_WEIGHT*loss_mask)
            outputs['refine_loss'] = loss_refine
            outputs['mask_loss'] = loss_mask

        return outputs
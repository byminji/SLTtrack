# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def select_focal_loss(pred, label, gamma=0):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.eq(1).nonzero().squeeze()
    neg = label.eq(0).nonzero().squeeze()

    if len(pos) > 0:
        loss_pos = - pred[:, 1][pos]
        loss_pos = loss_pos * (1. - torch.exp(-loss_pos)).pow(gamma)
        loss_pos = loss_pos.mean()
    else:
        loss_pos = 0

    if len(neg) > 0:
        loss_neg = - pred[:, 0][neg]
        loss_neg = loss_neg * (1. - torch.exp(-loss_neg)).pow(gamma)
        loss_neg = loss_neg.mean()
    else:
        loss_neg = 0

    return loss_pos * 0.5 + loss_neg * 0.5


def top1_reg_loss(pred, pred_loc, label_loc, loss_weight):
    b, a, h, w, _ = pred.size()
    pred = pred.view(b, -1, 2)[:, :, 1]
    num_anchors = pred.size(1)

    _,  top1_indices = torch.max(pred, dim=1)

    pred_loc = pred_loc.view(b, 4, -1)
    label_loc = label_loc.view(b, 4, -1)
    assert pred_loc.size(2) == num_anchors

    loss_weight = loss_weight.view(b, -1)
    assert num_anchors == loss_weight.size(1)

    top1_losses = []
    #print(top1_indices)
    for i in range(b):
        here_pred_loc = pred_loc[i, :, top1_indices[i]]
        here_label_loc = label_loc[i, :, top1_indices[i]]
        diff = (here_label_loc - here_pred_loc).abs()
        here_loss = diff.sum()
        top1_losses.append(here_loss)

    top1_losses = torch.stack(top1_losses)
    return top1_losses.mean()


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    # pred_loc : b * 4k * sh * sw
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)  # b 4 k sh sw
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw) # b k sh sw
    loss = diff * loss_weight
    return loss.sum().div(b)


def refine_weight_l1_loss(pred_loc, label_loc, loss_weight):
    # pred_loc : 368*4
    # label_loc : 23*16*4
    # loss_weight : 23*16*1
    b, n, _ = label_loc.size()
    pred_loc = pred_loc.view(b, n, -1)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=2)
    loss = diff * loss_weight.squeeze()
    return loss.sum().div(b)


def det_loss_smooth_l1(bboxes_pred, bboxes_gt):
    b, n, _ = bboxes_gt.size()
    bboxes_pred = bboxes_pred.view(b, n, -1)
    bboxes_gt = bboxes_gt.float()
    bbox_loss = F.smooth_l1_loss(bboxes_pred, bboxes_gt)
    return bbox_loss


def mask_loss_bce(masks_pred, masks_gt, mask_weight, ohem=True):
    """Mask R-CNN specific losses."""
    mask_weight = mask_weight.view(-1)
    pos = mask_weight.data.eq(1).nonzero().squeeze()
    if pos.nelement() == 0:
        return masks_pred.sum() * 0

    masks_pred = torch.index_select(masks_pred, 0, pos)
    masks_gt = torch.index_select(masks_gt, 0, pos)

    b, n, h, w = masks_pred.size()
    masks_pred = masks_pred.view(-1, h*w)
    # masks_gt = Variable(masks_gt.view(-1, h*w), requires_grad=False)
    masks_gt = masks_gt.view(-1, h*w).float()

    if ohem:
        top_k = 0.7
        loss = F.binary_cross_entropy_with_logits(masks_pred, masks_gt, reduction='none')
        loss = loss.view(-1)
        index = torch.topk(loss, int(top_k * loss.size()[0]))
        loss = torch.mean(loss[index[1]])
    else:
        loss = F.binary_cross_entropy_with_logits(masks_pred, masks_gt)

    # iou_m, iou_5, iou_7 = iou_measure(masks_pred, masks_gt)
    # return loss, iou_m, iou_5, iou_7
    return loss


def per_image_cross_entropy(pred, label):
    b, anchor, h, w, _ = pred.size()

    image_ce_loss = []

    pred = pred.view(b, -1, 2)
    label = label.view(b, -1)

    for i in range(b):
        pos = label[i].eq(1).nonzero().squeeze()
        neg = label[i].eq(0).nonzero().squeeze()

        pos_loss = get_cls_loss(pred[i], label[i], pos)
        neg_loss = get_cls_loss(pred[i], label[i], neg)
        image_ce_loss.append(pos_loss * 0.5 + neg_loss * 0.5)
    return torch.stack(image_ce_loss)


def per_image_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, h, w = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, h, w)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, h, w)

    image_l1_loss = []

    for i in range(b):
        loss = (diff[i] * loss_weight[i]).sum()
        image_l1_loss.append(loss)
    return torch.stack(image_l1_loss)


def _convert_bbox(anchor, delta):
    num_anchors = anchor.size(0)
    batch_size = delta.size(0)
    anchor = anchor.view(1, num_anchors, 4).expand(batch_size, num_anchors, 4)
    delta = delta.view(batch_size, 4, num_anchors)

    cx = delta[:, 0, :] * anchor[:, :, 2] + anchor[:, :, 0]
    cy = delta[:, 1, :] * anchor[:, :, 3] + anchor[:, :, 1]
    w = torch.exp(delta[:, 2, :]) * anchor[:, :, 2]
    h = torch.exp(delta[:, 3, :]) * anchor[:, :, 3]

    return torch.stack([cx, cy, w, h], dim=2)


def _intersect(box_a, box_b):
    box_a_max_xy = box_a[:, :2] + box_a[:, 2:] / 2
    box_a_min_xy = box_a[:, :2] - box_a[:, 2:] / 2
    box_b_max_xy = box_b[:, :2] + box_b[:, 2:] / 2
    box_b_min_xy = box_b[:, :2] - box_b[:, 2:] / 2

    max_xy = torch.min(box_a_max_xy, box_b_max_xy)
    min_xy = torch.max(box_a_min_xy, box_b_min_xy)
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def _compute_iou(box_a, box_b):
    inter = _intersect(box_a, box_b)
    area_a = box_a[:, 2] * box_a[:, 3]
    area_b = box_b[:, 2] * box_b[:, 3]
    union = area_a + area_b - inter
    iou = inter / union  # [A,B]
    return iou

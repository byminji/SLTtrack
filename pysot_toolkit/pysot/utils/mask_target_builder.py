from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import cv2
from pysot.core.config import cfg


def build_proposal_target(rois, gt_boxes, select):
    '''
    Args:
        rois: rois from rpn (x1, y1, x2, y2)
        gt_boxes: gt bboxes (x1, y1, x2, y2)
        select: selected batch indices
    Returns:

    '''
    rois = rois.detach()
    rois, cls_ind, regression_target, weights = proposal_target_layer(rois, gt_boxes, select)
    cls_ind = cls_ind.long().detach()
    weights = weights.detach()
    return rois, cls_ind, regression_target, weights


def proposal_target_layer(rpn_rois, gt_boxes, select):
    '''

    Args:
        rpn_rois: [b, k*w*h, 4] -> x1, y1, x2, y2
        gt_boxes: [b, 4] -> x1, y1, x2, y2
        select: [b]
    Returns:

    '''
    batch_size = len(gt_boxes)
    sample_num = rpn_rois.size()[1]
    gt_boxes = gt_boxes.unsqueeze(1) # [b, 1, 4]

    if batch_size > 0:
        # multiple images
        rois_list = []
        labels_list = []
        bbox_targets_list = []
        weights_list = []

        for i in range(batch_size):
            if i not in select:
                continue

            rois, labels, bbox_targets, weights = single_proposal_target_layer(i * sample_num, rpn_rois[i], gt_boxes[i])

            rois_list.append(rois.unsqueeze(0))
            labels_list.append(labels.unsqueeze(0))
            bbox_targets_list.append(bbox_targets.unsqueeze(0))
            weights_list.append(weights.unsqueeze(0))
        return torch.cat(rois_list, dim=0), torch.cat(labels_list, dim=0), torch.cat(bbox_targets_list, dim=0), torch.cat(weights_list, dim=0)

    else:
        # single image
        rois, labels, bbox_targets, weights = single_proposal_target_layer(0, rpn_rois, gt_boxes)
        return rois.unsqueeze(0), labels.unsqueeze(0), bbox_targets.unsqueeze(0), weights.unsqueeze(0)


def single_proposal_target_layer(start_idx, rpn_rois, gt_box):
    """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """
    # Proposal RoIs coming from RPN
    # rpn_rois: [N, 4]
    # gt_rbbox : [1, 4]

    all_rois = rpn_rois
    # Include ground-truth boxes in the set of candidate rois
    TRAIN_USE_GT = True  # TO BE WRITTEN IN CFG
    if TRAIN_USE_GT:
        all_rois = torch.cat((all_rois.double(), gt_box), 0)

    rois_per_image = cfg.TRAIN.ROI_PER_IMG
    fg_rois_per_image = int(round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    labels, rois, bbox_target, weights = _sample_rois(
                                            all_rois,
                                            gt_box,
                                            fg_rois_per_image,
                                            rois_per_image
                                            )

    rois = rois.view(-1, 4)
    labels = labels.view(-1, 1)
    bbox_target = bbox_target.view(-1, 4)
    weights = weights.view(-1, 1)
    return rois, labels, bbox_target, weights


def _sample_rois(all_rois, gt_box, fg_rois_per_image, rois_per_image):
    """
        Generate a random sample of RoIs comprising foreground and background examples.
        all_rois: [N, 4]
        gt_rbbox: [1, 4]
    """

    def _rand_choice_idx(x, k, to_replace=False):
        idxs = np.random.choice(x.numel(), k, replace=to_replace)
        return x[torch.cuda.LongTensor(idxs)]

    fg_inds, bg_inds = _bbox_assignment(
        all_rois, gt_box,
        cfg.TRAIN.FG_THRESH, cfg.TRAIN.BG_THRESH_HIGH,
        cfg.TRAIN.BG_THRESH_LOW)

    # balanced sample rois
    weights = (1/rois_per_image)*torch.ones(rois_per_image).cuda()
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    if fg_inds.numel() > fg_rois_per_image:
        fg_inds = _rand_choice_idx(fg_inds, fg_rois_per_image)
    else:
        if fg_inds.numel() == 0:
            fg_inds = torch.tensor([all_rois.size(0)]).cuda() # gt bbox
        # TODO random sample some area around the gt box
        times = fg_rois_per_image // fg_inds.numel()
        mod = fg_rois_per_image % fg_inds.numel()

        # consider sample duplication
        weights[:int(fg_inds.numel()*times)] /= times
        fg_inds = torch.cat((fg_inds.repeat(times), _rand_choice_idx(fg_inds, mod)), dim=0)

    if bg_inds.numel() > bg_rois_per_image:
        bg_inds = _rand_choice_idx(bg_inds, bg_rois_per_image)
    elif bg_inds.numel() < 0 and bg_inds.numel() < bg_rois_per_image:
        times = bg_rois_per_image // bg_inds.numel()
        mod = bg_rois_per_image % bg_inds.numel()
        bg_inds = torch.cat((bg_inds.repeat(times), _rand_choice_idx(bg_inds, mod)), dim=0)

    # The indices that we're selecting (both fg and bg)
    keep_inds = torch.cat([fg_inds, bg_inds], 0)

    # Select sampled values from various arrays:
    labels = torch.ones(keep_inds.size(0)).cuda()
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds].contiguous()

    bbox_target = _compute_targets(rois, gt_box.expand(rois.shape[0], -1))

    return labels, rois, bbox_target, weights


def _bbox_assignment(boxes, gt_boxes, pos_iou_th, neg_iou_th_high, neg_iou_th_low):
    overlaps = bbox_overlaps(boxes, gt_boxes, mode="iou")
    max_overlaps, _ = overlaps.max(dim=1)

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = max_overlaps >= pos_iou_th
    fg_inds = fg_inds.nonzero().view(-1)

    #bg_inds = ((max_overlaps < neg_iou_th_high) & (max_overlaps >= neg_iou_th_low)).nonzero().view(-1)
    bg_inds = (max_overlaps < neg_iou_th_high).nonzero().view(-1)
    #if bg_inds.numel() == 0:
    #    bg_inds = (max_overlaps < neg_iou_th_high).nonzero().view(-1)

    return fg_inds, bg_inds


def bbox_overlaps(boxes, query_boxes, mode="iou"):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0]) * \
                (boxes[:, 3] - boxes[:, 1])
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0]) * \
                  (query_boxes[:, 3] - query_boxes[:, 1])

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) -
          torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t())).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) -
          torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t())).clamp(min=0)
    iarea = iw * ih
    if mode == "iou":
        ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iarea
    elif mode == "iof":
        ua = box_areas.view(-1, 1)
    else:
        raise NotImplementedError

    overlaps = iarea / ua
    return out_fn(overlaps)


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    # Inputs are tensor

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS)) /
                   targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return targets


def bbox_transform(ex_rois, gt_rois, clip=-1):
    # type: (Tensor, Tensor) -> Tensor
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
    ex_ctr_x = (ex_rois[:, 0] + ex_rois[:, 2]) * 0.5
    ex_ctr_y = (ex_rois[:, 1] + ex_rois[:, 3]) * 0.5

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
    gt_ctr_x = (gt_rois[:, 0] + gt_rois[:, 2]) * 0.5
    gt_ctr_y = (gt_rois[:, 1] + gt_rois[:, 3]) * 0.5

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    if clip > 0:
        targets_dw = torch.clamp_max(targets_dw, clip)
        targets_dh = torch.clamp_max(targets_dh, clip)

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 1)

    return targets



def build_mask_target(rois, cls_ind, gt_masks):
    pred_boxes = rois
    # crop mask wrt roi
    mask_targets = _build_mask_target(
        pred_boxes.data.cpu().numpy(),
        gt_masks.data.cpu().numpy(),
        cls_ind.data.cpu().numpy(),
        cfg.MASK.MASK_OUTSIZE)
    return mask_targets


def _build_mask_target(boxes, masks, label, mask_size):

    def mask_to_target(mask, roi, mask_size, is_binary=False):
        x1 = int(min(max(roi[0], 0), gt_mask.shape[1] - 1))
        y1 = int(min(max(roi[1], 0), gt_mask.shape[0] - 1))
        x2 = int(min(max(roi[2], 0), gt_mask.shape[1] - 1))
        y2 = int(min(max(roi[3], 0), gt_mask.shape[0] - 1))
        mask = mask[y1:y2 + 1, x1:x2 + 1]
        if is_binary is True:
            target = cv2.resize(mask, tuple([mask_size, mask_size]), interpolation=cv2.INTER_LINEAR)
            target = target > 128
        else:
            target = cv2.resize(mask, tuple([mask_size, mask_size]), interpolation=cv2.INTER_NEAREST)
        return target

    num_batches = boxes.shape[0]
    crop_masks = []
    for i in range(num_batches):
        batch_label = label[i]
        num_rois = boxes[i].shape[0]
        batch_crop_masks = []
        for j in range(num_rois):
            if batch_label[j] <= 0:
                continue
            # gt_mask = masks[i].squeeze(0)
            gt_mask = masks[i]
            roi = boxes[i, j, :]
            gt_mask = mask_to_target(gt_mask, roi, mask_size)
            batch_crop_masks.append(gt_mask)
        crop_masks.append(np.array(batch_crop_masks, np.float32))

    if len(crop_masks) > 0:
        mask_targets = torch.from_numpy(np.array(crop_masks, np.float32)).cuda()
        return mask_targets
    else:
        return None
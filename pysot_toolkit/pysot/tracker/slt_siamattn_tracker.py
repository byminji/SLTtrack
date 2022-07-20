# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.bbox import *
from torch.distributions.categorical import Categorical
import time
import cv2

def batch_center2corner(boxes):
    xmin = boxes[:, 0] - boxes[:, 2] * 0.5
    ymin = boxes[:, 1] - boxes[:, 3] * 0.5
    xmax = boxes[:, 0] + boxes[:, 2] * 0.5
    ymax = boxes[:, 1] + boxes[:, 3] * 0.5

    if isinstance(boxes, np.ndarray):
        return np.stack([xmin, ymin, xmax, ymax], 1)
    else:
        return torch.stack([xmin, ymin, xmax, ymax], 1)


def batch_corner2center(boxes):
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w = (boxes[:, 2] - boxes[:, 0])
    h = (boxes[:, 3] - boxes[:, 1])

    if isinstance(boxes, np.ndarray):
        return np.stack([cx, cy, w, h], 1)
    else:
        return torch.stack([cx, cy, w, h], 1)


def inverse_sigmoid(x, eps=1e-3):
    return torch.log((x / (1 - x + eps)) + eps)


def stable_inverse_sigmoid(x, eps=0.01):
    x = (x * (1 - eps * 2)) + eps
    return torch.log(x / (1 - x))

from pysot.datasets.anchor_target_attn import AnchorTarget


class SltSiamAttnTracker(SiameseTracker):
    def __init__(self, model):
        super(SltSiamAttnTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.gpu_window = torch.from_numpy(self.window).cuda()
        self.anchors = self.generate_anchor(self.score_size)
        self.gpu_anchors = torch.from_numpy(self.anchors).cuda()
        self.model = torch.nn.DataParallel(model)
        self.anchor_target = AnchorTarget()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_bbox_torch(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = torch.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = torch.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1)[:, 1].detach().cpu().numpy()
        return score

    def _convert_score_no_softmax(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)[:, 1]
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def _bbox_clip_batch(self, cx, cy, width, height, boundary):
        cx = np.maximum(0, np.minimum(cx, boundary[1]))
        cy = np.maximum(0, np.minimum(cy, boundary[0]))
        width = np.maximum(10, np.minimum(width, boundary[1]))
        height = np.maximum(10, np.minimum(height, boundary[0]))
        return cx, cy, width, height

    def _post_process_baseline(self, cls, loc, scale_z):
        scale_z = scale_z.reshape(-1)
        bs = cls.size(0)
        score = cls.view(bs, 2, -1).permute(0, 2, 1).contiguous()  # tensor(num_seq, num_anchors, 2)
        if cfg.TRAIN.NO_NEG_LOGIT:
            score = score[:, :, 1]
        else:
            score = score[:, :, 1] - score[:, :, 0]
        score = torch.sigmoid(score)

        def batch_convert_bbox(delta):
            anchor = self.gpu_anchors
            bs = delta.size(0)
            delta = delta.view(bs, 4, -1)  # batch * 4 * num_anchors

            delta[:, 0, :] = delta[:, 0, :] * anchor[:, 2] + anchor[:, 0]
            delta[:, 1, :] = delta[:, 1, :] * anchor[:, 3] + anchor[:, 1]
            delta[:, 2, :] = torch.exp(delta[:, 2, :]) * anchor[:, 2]
            delta[:, 3, :] = torch.exp(delta[:, 3, :]) * anchor[:, 3]
            return delta

        pred_bbox = batch_convert_bbox(loc).cpu().numpy()  # np.array(num_seq, 4, num_anchors)

        def change(a, b):
            r = a / b
            r = np.log(r)
            r = np.abs(r)
            r = np.exp(r)
            return r

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        old_bbox_sz = sz(self.size[:, 0] * scale_z, self.size[:, 1] * scale_z)  # np.array(num_seq)
        pred_bbox_sz = sz(pred_bbox[:, 2, :], pred_bbox[:, 3, :])  # np.array(num_seq, num_anchors)
        s_c = change(old_bbox_sz.reshape(bs, 1), pred_bbox_sz)  # np.array(num_seq, num_anchors)

        # aspect ratio penalty
        old_bbox_ratio = (self.size[:, 0] * scale_z) / (self.size[:, 1] * scale_z)  # np.array(num_seq)
        pred_bbox_ratio = pred_bbox[:, 2, :] / pred_bbox[:, 3, :]  # np.array(num_seq, num_anchors)
        r_c = change(old_bbox_ratio.reshape(bs, 1), pred_bbox_ratio)  # np.array(num_seq, num_anchors)

        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        penalty = torch.tensor(penalty, dtype=torch.float, device=score.device)
        pscore = score * penalty  # tensor(num_seq, num_anchors)
        final_pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.gpu_window.view(1, -1) * cfg.TRACK.WINDOW_INFLUENCE

        return final_pscore, pscore, pred_bbox, penalty

    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > cfg.TRACK.MASK_THERSHOLD)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(self.center_pos, self.size)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        return rbox_in_img

    def batch_init(self, images, template_bbox, initial_bbox):
        template_bbox = batch_corner2center(template_bbox)
        initial_bbox = batch_corner2center(initial_bbox)
        self.center_pos = initial_bbox[:, :2]
        self.size = initial_bbox[:, 2:]

        # calculate z crop size
        w_z = template_bbox[:, 2] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(template_bbox[:, 2:], 1)
        h_z = template_bbox[:, 3] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(template_bbox[:, 2:], 1)
        s_z = np.round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = []
        for img in images:
            self.channel_average.append(np.mean(img, axis=(0, 1)))
        self.channel_average = np.array(self.channel_average)

        # get crop
        z_crop_list = []
        for i in range(len(images)):
            here_crop, np_img = self.get_subwindow(images[i], template_bbox[i, :2],
                                              cfg.TRACK.EXEMPLAR_SIZE, int(s_z[i]), self.channel_average[i])
            z_crop_list.append(here_crop)

        z_crop = torch.cat(z_crop_list, dim=0)
        self.z_crop = z_crop
        return z_crop  # tensor(num_seq, 3, 127, 127)

    def batch_track(self, img, gt_boxes, mask, action_mode='max'):
        w_z = self.size[:, 0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size, 1)  # np.array(num_seq)
        h_z = self.size[:, 1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size, 1)  # np.array(num_seq)
        s_z = np.sqrt(w_z * h_z)  # np.array(num_seq)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)  # np.array(num_seq)

        x_crop_list = []
        search_img_list = []
        m_crop_list = []
        label_loc_list = []
        label_loc_weight_list = []
        label_cls_list = []
        anchor_list = []
        gt_in_crop_list = []
        for i in range(len(img)):
            try:
                x_crop, np_img, m_crop = self.get_subwindow_mask(img[i], mask[i], self.center_pos[i],
                                                     cfg.TRACK.INSTANCE_SIZE,
                                                     round(s_x[i]), self.channel_average[i])
            except cv2.error as e:
                print('print img i shape', img[i].shape)
                print('s_x[i]', s_x[i])
                print('print mask i shape', mask[i].shape)
                exit(0)

            if gt_boxes is not None and np.sum(np.abs(gt_boxes[i] - np.zeros(4))) > 10:
                gt_in_crop = np.zeros(4)
                offset = self.center_pos[i] - s_x[i] * 0.5
                gt_in_crop[:2] = gt_boxes[i, :2] - offset
                gt_in_crop[2:] = gt_boxes[i, 2:] - offset
                gt_in_crop = gt_in_crop * scale_z[i]

                cls_target, delta, delta_weight, anchor_center, overlap = self.anchor_target(gt_in_crop, cfg.TRAIN.OUTPUT_SIZE)
                label_loc_list.append(delta)
                label_loc_weight_list.append(delta_weight)
                label_cls_list.append(cls_target)
                anchor_list.append(anchor_center)
                gt_in_crop_list.append(gt_in_crop)
            else:
                label_loc_list.append(np.zeros((4, cfg.ANCHOR.ANCHOR_NUM, 25, 25)))
                label_loc_weight_list.append(np.zeros((cfg.ANCHOR.ANCHOR_NUM, 25, 25)))
                label_cls_list.append(np.zeros((5, 25, 25)))
                anchor_list.append(np.zeros((4, cfg.ANCHOR.ANCHOR_NUM, 25, 25)))
                gt_in_crop_list.append(np.zeros(4))

            x_crop_list.append(x_crop)
            m_crop_list.append(m_crop)
            search_img_list.append(np_img)
        x_crop = torch.cat(x_crop_list, dim=0)  # tensor(num_seq, 3, 255, 255)
        m_crop = torch.cat(m_crop_list, dim=0)  # tensor(num_seq, 255, 255)

        outputs = self.model({'search': x_crop, 'template': self.z_crop})
        final_score, pscore, pred_bbox, penalty = self._post_process_baseline(outputs['cls'], outputs['loc'].detach(), scale_z)

        batch_size = len(img)
        inv_score = stable_inverse_sigmoid(final_score, cfg.TRAIN.SIG_EPS)
        softmax_score = F.softmax(inv_score * cfg.TRAIN.TEMP, dim=1)

        epsilon = 1 - torch.max(softmax_score, dim=1)[0]

        prob = Categorical(softmax_score)

        not_max_selected = None
        if action_mode == 'max':
            selected_indices = torch.argmax(softmax_score, 1)
        elif action_mode == 'sample':
            selected_indices = prob.sample()
        elif action_mode == 'half':
            max_indices = torch.argmax(softmax_score, 1)
            sampled_indices = prob.sample()
            bs = len(max_indices) // 2
            assert len(max_indices) % 2 == 0
            selected_indices = torch.cat([max_indices[:bs], sampled_indices[bs:]], dim=0)
            not_max_selected = max_indices == selected_indices

        selected_indices = selected_indices.detach()
        rpn_bbox = pred_bbox[range(batch_size), :, selected_indices.cpu().numpy()]  # np.array(num_anchors, 4)

        # refine bounding box
        crop_center_pos = np.array([cfg.TRACK.INSTANCE_SIZE / 2, cfg.TRACK.INSTANCE_SIZE / 2], dtype=np.float32)
        crop_center_pos = np.tile(crop_center_pos, (batch_size, 1))
        rpn_cx = rpn_bbox[:, 0] + crop_center_pos[:, 0]
        rpn_cy = rpn_bbox[:, 1] + crop_center_pos[:, 1]

        rpn_cx, rpn_cy, rpn_width, rpn_height = self._bbox_clip_batch(rpn_cx, rpn_cy, rpn_bbox[:, 2], rpn_bbox[:, 3],
                                                                [cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE])

        rpn_roi = np.stack((np.arange(batch_size),
                            rpn_cx - rpn_width * 0.5 + 0.5,
                            rpn_cy - rpn_height * 0.5 + 0.5,
                            rpn_cx + rpn_width * 0.5 - 0.5,
                            rpn_cy + rpn_height * 0.5 - 0.5), axis=1)
        rpn_roi = torch.from_numpy(rpn_roi).float().cuda()

        b_fused_features, _ = self.model.module.feature_fusion(outputs['zf'], outputs['xf'])
        pb_bbox = self.model.module.refine_head(b_fused_features, rpn_roi).squeeze().data.cpu().numpy()

        # normalize
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            stds = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS)
            means = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            pb_bbox = pb_bbox * stds + means
        else:
            pb_bbox = pb_bbox

        pb_bbox[:, 0] = pb_bbox[:, 0] * rpn_width + rpn_cx - crop_center_pos[:, 0]
        pb_bbox[:, 1] = pb_bbox[:, 1] * rpn_height + rpn_cy - crop_center_pos[:, 1]
        pb_bbox[:, 2] = np.exp(pb_bbox[:, 2]) * rpn_width
        pb_bbox[:, 3] = np.exp(pb_bbox[:, 3]) * rpn_height

        bbox = pb_bbox / scale_z.reshape(-1, 1)
        lr = pscore.detach()[range(batch_size), selected_indices].cpu().numpy() * cfg.TRACK.LR

        cx = bbox[:, 0] + self.center_pos[:, 0]
        cy = bbox[:, 1] + self.center_pos[:, 1]

        # smooth bbox
        width = self.size[:, 0] * (1 - lr) + bbox[:, 2] * lr
        height = self.size[:, 1] * (1 - lr) + bbox[:, 3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip_batch(cx, cy, width, height, img[i].shape[:2])

        # udpate state
        self.center_pos = np.stack([cx, cy], 1)
        self.size = np.stack([width, height], 1)

        bbox = np.stack([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], 1)
        return {
            'x_crop': x_crop,  # tensor(num_seq, 3, 255, 255)
            'm_crop': m_crop,  # tensor(num_seq, 255, 255)
            'bbox': bbox,  # np.array(num_seq, 4)
            'selected_indices': selected_indices.cpu(),  # np.array(num_seq)
            'prob': softmax_score[range(batch_size), selected_indices].detach().cpu(),  # np.array(num_seq)
            'not_max_selected': not_max_selected.cpu() if not_max_selected is not None else None,
            'label_loc': torch.tensor(np.stack(label_loc_list, axis=0), dtype=torch.float),
            'label_loc_weight': torch.tensor(np.stack(label_loc_weight_list, axis=0), dtype=torch.float),
            'epsilon': epsilon.detach().cpu(),
            'penalty': penalty.cpu(),
            'label_cls': torch.tensor(np.stack(label_cls_list, axis=0), dtype=torch.long),
            'anchor': torch.tensor(np.stack(anchor_list, axis=0), dtype=torch.float),
            'gt_in_crop': torch.tensor(np.stack(gt_in_crop_list, axis=0), dtype=torch.double),
        }

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop, raw_img = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.module.template(z_crop.cuda())
        return raw_img

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        s_x = round(s_x)

        x_crop, raw_img = self.get_subwindow(img, self.center_pos,
                                             cfg.TRACK.INSTANCE_SIZE,
                                             s_x, self.channel_average)

        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]

        outputs = self.model.module.track(x_crop.cuda())

        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        final_score, pscore, logit = self.convert_score(outputs['cls'], pred_bbox, scale_z)

        best_idx = np.argmax(final_score)
        rpn_bbox = pred_bbox[:, best_idx]

        # Refine box
        center_pos = np.array([cfg.TRACK.INSTANCE_SIZE / 2, cfg.TRACK.INSTANCE_SIZE / 2], dtype=np.float32)
        rpn_cx = rpn_bbox[0] + center_pos[0]
        rpn_cy = rpn_bbox[1] + center_pos[1]
        rpn_cx, rpn_cy, rpn_width, rpn_height = self._bbox_clip(rpn_cx, rpn_cy, rpn_bbox[2], rpn_bbox[3],
                                                                [cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE])

        rpn_roi = [rpn_cx - rpn_width * 0.5 + 0.5,
                   rpn_cy - rpn_height * 0.5 + 0.5,
                   rpn_cx + rpn_width * 0.5 - 0.5,
                   rpn_cy + rpn_height * 0.5 - 0.5]
        rpn_roi = torch.cat((torch.Tensor([0]).unsqueeze(0), torch.Tensor(rpn_roi).unsqueeze(0)), dim=1).cuda()

        pb_bbox = self.model.module.bbox_refine(rpn_roi).squeeze().data.cpu().numpy()

        # normalize
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            stds = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS)
            means = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            pb_bbox = pb_bbox * stds + means
        else:
            pb_bbox = pb_bbox

        pb_bbox[0] = pb_bbox[0] * rpn_width + rpn_cx
        pb_bbox[1] = pb_bbox[1] * rpn_height + rpn_cy
        pb_bbox[2] = np.exp(pb_bbox[2]) * rpn_width
        pb_bbox[3] = np.exp(pb_bbox[3]) * rpn_height

        # update state
        bbox = pb_bbox / scale_z
        lr = pscore[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + crop_box[0]
        cy = bbox[1] + crop_box[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy,
                                                width, height, img.shape[:2])

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                cx + width / 2,
                cy + height / 2]

        # Mask prediction
        pb_cx = (cx - crop_box[0]) * scale_z
        pb_cy = (cy - crop_box[1]) * scale_z
        pb_width = width * scale_z
        pb_height = height * scale_z

        bbox_roi = [pb_cx - pb_width * 0.5,
                    pb_cy - pb_height * 0.5,
                    pb_cx + pb_width * 0.5,
                    pb_cy + pb_height * 0.5]
        bbox_roi = torch.cat((torch.Tensor([0]).unsqueeze(0), torch.Tensor(bbox_roi).unsqueeze(0)), dim=1).cuda()

        mask = self.model.module.mask_refine(bbox_roi).view(-1)
        mean = torch.mean(mask)
        mask = (mask - mean).sigmoid()

        out_size = cfg.TRACK.MASK_OUTPUT_SIZE
        mask = mask.view(out_size, out_size).cpu().data.numpy()

        s = crop_box[2] / cfg.TRACK.INSTANCE_SIZE
        sub_box = [crop_box[0] + (pb_cx-pb_width/2) * s,
                   crop_box[1] + (pb_cy-pb_height/2) * s,
                   s * pb_width,
                   s * pb_height]
        s_w = out_size / sub_box[2]
        s_h = out_size / sub_box[3]

        im_h, im_w = img.shape[:2]
        back_box = [-sub_box[0] * s_w, -sub_box[1] * s_h, im_w * s_w, im_h * s_h]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        mask_for_show = (mask_in_img > cfg.TRACK.MASK_THERSHOLD) * 255
        mask_for_show = mask_for_show.astype(np.uint8)
        polygon = self._mask_post_processing(mask_in_img)
        polygon = polygon.flatten().tolist()

        return {
                'best_index': best_idx,
                'best_score': pscore[best_idx],
                'x_crop': x_crop,
                'bbox': bbox,
                'search_img': raw_img,
                'mask': mask_for_show,
                'polygon': polygon
               }

    def convert_score(self, score, pred_bbox, scale_z):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)

        if cfg.TRAIN.NO_NEG_LOGIT:
            score = score[:, 1]
        else:
            score = score[:, 1] - score[:, 0]
        logit = score.clone()
        score = torch.sigmoid(score)
        score = score.detach().cpu().numpy()

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        final_score = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                      self.window * cfg.TRACK.WINDOW_INFLUENCE
        return final_score, pscore, logit
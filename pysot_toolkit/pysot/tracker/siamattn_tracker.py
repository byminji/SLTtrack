from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch

from pysot.core.config import cfg
from pysot.utils.bbox import cxy_wh_2_rect
from pysot.tracker.siamrpn_tracker import SiamRPNTracker


class SiamAttnTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamAttnTracker, self).__init__(model)
        assert hasattr(self.model, 'mask_head'), \
            "SiamMaskTracker must have mask_head"

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

        x_crop, _ = self.get_subwindow(img,
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x,
                                    self.channel_average)
        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]

        outputs = self.model.track(x_crop.cuda())
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        # processing bbox & mask
        rpn_bbox = pred_bbox[:, best_idx]
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

        pb_bbox = self.model.bbox_refine(rpn_roi).squeeze().data.cpu().numpy()
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

        # udpate state
        bbox = pb_bbox / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

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
        best_score = score[best_idx]

        pb_cx = (cx - crop_box[0]) * scale_z
        pb_cy = (cy - crop_box[1]) * scale_z
        pb_width = width * scale_z
        pb_height = height * scale_z

        bbox_roi = [pb_cx - pb_width * 0.5,
                    pb_cy - pb_height * 0.5,
                    pb_cx + pb_width * 0.5,
                    pb_cy + pb_height * 0.5]
        bbox_roi = torch.cat((torch.Tensor([0]).unsqueeze(0), torch.Tensor(bbox_roi).unsqueeze(0)), dim=1).cuda()
        mask = self.model.mask_refine(bbox_roi).sigmoid().squeeze()
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
                'bbox': bbox,
                'best_score': best_score,
                'mask': mask_for_show,
                'polygon': polygon
               }

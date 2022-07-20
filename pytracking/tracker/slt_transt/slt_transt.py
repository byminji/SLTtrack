from pytracking.tracker.base import BaseTracker, SiameseTracker
import torch
import torch.nn.functional as F
import math
import time
import numpy as np
import cv2

from pytracking.features.preprocessing import numpy_to_torch
from pytracking.tracker.slt_transt.config import cfg
# from pytracking.tracker.slt_transt.config_ht import cfg
import ltr.data.bounding_box_utils as bbutils
import torchvision.transforms.functional as tvisf
from torch.distributions.categorical import Categorical


def stable_inverse_sigmoid(x, eps=0.01):
    x = (x * (1 - eps * 2)) + eps
    return torch.log(x / (1 - x))

class SLTTransT(SiameseTracker):

    multiobj_mode = 'parallel'

    def _convert_score(self, score):
        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_score_new(self, score):
        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = score[:, 0] - score[:, 1]
        logit = score.clone()
        score = torch.sigmoid(score)
        score = score.detach().cpu().numpy()
        return score, logit

    def _convert_bbox(self, delta):
        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()
        return delta

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def batch_init(self, images, template_bbox, initial_bbox) -> dict:
        """
        For inference in sequence-level training
        Note that sequences are duplicated for argmax & sampling tracker
        image - template bbox : pair for template frame (image: list (num_seq) of np.ndarray (hxwx3))
        initial bbox : gt in the first frame of test sequences (bbox: np.ndarray: num_seq x 4)
        """

        hanning = np.hanning(32)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        self.gpu_window = torch.from_numpy(self.window).cuda()
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # The TransT network
        self.net = self.params.net

        # Convert bbox (x1, y1, w, h) -> (cx, cy, w, h)
        template_bbox = bbutils.batch_xywh2center2(template_bbox) # ndarray:(2*num_seq,4)
        initial_bbox = bbutils.batch_xywh2center2(initial_bbox) # ndarray:(2*num_seq,4)

        self.center_pos = initial_bbox[:, :2] # ndarray:(2*num_seq,2)
        self.size = initial_bbox[:, 2:] # ndarray:(2*num_seq,2)

        # calculate z crop size
        w_z = template_bbox[:, 2] + (2 - 1) * ((template_bbox[:, 2] + template_bbox[:, 3]) * 0.5)# ndarray:(2*num_seq)
        h_z = template_bbox[:, 3] + (2 - 1) * ((template_bbox[:, 2] + template_bbox[:, 3]) * 0.5)# ndarray:(2*num_seq)
        s_z = np.ceil(np.sqrt(w_z * h_z))# ndarray:(2*num_seq)

        # calculate channle average
        self.channel_average = []
        for img in images:
            self.channel_average.append(np.mean(img, axis=(0, 1)))
        self.channel_average = np.array(self.channel_average)# ndarray:(2*num_seq,3)

        # get crop
        z_crop_list = []
        for i in range(len(images)):
            here_crop = self.get_subwindow(images[i], template_bbox[i, :2],
                                           cfg.TRACK.EXEMPLAR_SIZE, s_z[i], self.channel_average[i])
            z_crop = here_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.inplace = False
            z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)
            z_crop_list.append(z_crop)
        z_crop = torch.cat(z_crop_list, dim=0)  # Tensor(2*num_seq,3,128,128)

        self.net.template_batch(z_crop)

        out = {'template_images': z_crop} # Tensor(2*num_seq,3,128,128)
        return out

    def batch_track(self, img, gt_boxes, action_mode='max') -> dict:
        w_x = self.size[:, 0] + (4 - 1) * ((self.size[:, 0] + self.size[:, 1]) * 0.5)
        h_x = self.size[:, 1] + (4 - 1) * ((self.size[:, 0] + self.size[:, 1]) * 0.5)
        s_x = np.ceil(np.sqrt(w_x * h_x))

        # Convert bbox (x1, y1, w, h) -> (x1, y1, x2, y2)
        gt_boxes_corner = bbutils.batch_xywh2corner(gt_boxes) # ndarray:(2*num_seq,4)

        x_crop_list = []
        gt_in_crop_list = []
        for i in range(len(img)):
            try:
                x_crop = self.get_subwindow(img[i], self.center_pos[i],
                                                     cfg.TRACK.INSTANCE_SIZE,
                                                     round(s_x[i]), self.channel_average[i])

            except cv2.error as e:
                print('print img i shape', img[i].shape)
                print('s_x[i]', s_x[i])
                exit(0)

            if gt_boxes_corner is not None and np.sum(np.abs(gt_boxes_corner[i] - np.zeros(4))) > 10:
                gt_in_crop = np.zeros(4)
                gt_in_crop[:2] = gt_boxes_corner[i, :2] - self.center_pos[i]
                gt_in_crop[2:] = gt_boxes_corner[i, 2:] - self.center_pos[i]
                gt_in_crop = gt_in_crop * (cfg.TRACK.INSTANCE_SIZE / s_x[i]) + cfg.TRACK.INSTANCE_SIZE / 2
                gt_in_crop[2:] = gt_in_crop[2:] - gt_in_crop[:2] # (x1,y1,x2,y2) to (x1,y1,w,h)
                gt_in_crop_list.append(gt_in_crop)
            else:
                gt_in_crop_list.append(np.zeros(4))

            x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
            x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)
            x_crop_list.append(x_crop)

        x_crop = torch.cat(x_crop_list, dim=0)  # tensor(2*num_seq, 3, 256, 256)

        # Get outputs
        # outputs: {'pred_logits': Tensor(2*num_seq,1024,2),'pred_boxes': Tensor(2*num_seq,1024,4)}
        outputs = self.net.track_batch(x_crop)

        # sigmoid
        cls = outputs['pred_logits']
        if self.params.no_neg_logit:
            score = cls[:, :, 0]
        else:
            score = cls[:, :, 0] - cls[:, :, 1]
        score = torch.sigmoid(score) # (2*num_seq,1024)
        pscore = score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.gpu_window.view(1, -1) * cfg.TRACK.WINDOW_INFLUENCE

        # inverse sigmoid -> softmax
        inv_score = stable_inverse_sigmoid(pscore, self.params.sig_eps)
        softmax_score = F.softmax(inv_score * self.params.temp, dim=1)

        prob = Categorical(softmax_score)

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
        selected_indices = selected_indices.detach()

        # Convert bbox (2*num_seq,1024,4) tensor -> numpy
        pred_bbox = outputs['pred_boxes'].data.cpu().numpy()
        bbox = pred_bbox[range(len(img)), selected_indices.cpu().numpy(), :]

        bbox = bbox * s_x.reshape(-1, 1)
        cx = bbox[:, 0] + self.center_pos[:, 0] - s_x/2
        cy = bbox[:, 1] + self.center_pos[:, 1] - s_x/2
        width = bbox[:, 2]
        height = bbox[:, 3]

        # clip boundary
        for i in range(len(img)):
            cx[i], cy[i], width[i], height[i] = self._bbox_clip(cx[i], cy[i], width[i],
                                                    height[i], img[i].shape[:2])

        # update state
        self.center_pos = np.stack([cx, cy], 1)
        self.size = np.stack([width, height], 1)

        bbox = np.stack([cx - width / 2, cy - height / 2, width, height], 1)

        out = {'search_images': x_crop,  # tensor(num_seq, 3, 256, 256)
               'pred_bboxes': bbox,  # np.array(num_seq, 4)
               'selected_indices': selected_indices.cpu(),  # np.array(num_seq)
               'gt_in_crop': torch.tensor(np.stack(gt_in_crop_list, axis=0), dtype=torch.float)}

        return out

    def initialize(self, image, info: dict) -> dict:
        hanning = np.hanning(32)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'
        # Initialize network
        self.initialize_features()
        # The DiMP network
        self.net = self.params.net
        # Time initialization
        tic = time.time()
        bbox = info['init_bbox']
        self.center_pos = np.array([bbox[0]+bbox[2]/2,
                                    bbox[1]+bbox[3]/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channel average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)
        self.net.template(z_crop)
        out = {'time': time.time() - tic}
        return out

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def track(self, image, info: dict = None) -> dict:
        w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))

        x_crop = self.get_subwindow(image, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)
        outputs = self.net.track(x_crop)
        score, logit = self._convert_score_new(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])

        # window penalty
        pscore = score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        # pscore = score
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:,best_idx]

        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x/2
        cy = bbox[1] + self.center_pos[1] - s_x/2
        width = bbox[2]
        height = bbox[3]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        out = {'target_bbox': bbox,
               'best_score': pscore,
               'best_idx': best_idx}
        return out



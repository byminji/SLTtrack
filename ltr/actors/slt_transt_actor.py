from . import BaseActor
import torch
import numpy as np
import torch.nn.functional as F

from ltr.data.bounding_box_utils import batch_xywh2corner
from ltr.data.bbox import IoU
from pytracking.tracker.slt_transt import SLTTransT
from pytracking.tracker.slt_transt.slt_transt import stable_inverse_sigmoid
from pytracking.tracker.transt.config import cfg


class SLTTransTActor(BaseActor):
    """Actor for sequence-level-training the TrDiMP network."""
    def __init__(self, net, objective, loss_weight=None, params=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.params = params # parameter for tracker

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)
        self.objective.train(mode)

    def explore(self, data):
        """
        Exploration in Sequence-Level Training to collect data from inference mode
        Args:
            data: The input data, should contain the fields 'search_images', 'search_annos', 'template_images', 'template_annos'.
            - search_images: Search frames. list (num_seq) of array (num_frames, h, w, 3)
            - search_annos: Seqrch annos. list (num_seq) of array (num_frames, 4)
            - template_images: list (num_seq) of array (h, w, 3)
            - template_annos: list (num_seq) of array (4)

        Returns:
            results: dictionary of exploration results
            - template_images: Tensor (num_seq, 3, 127, 127)
            - search_images: Sequence-level sampled image patches. Tensor (num_frames-1, num_seq, 3, 256, 256)
            - search_anno: Sequence-level sampled annotations. Tensor (num_frames-1, num_seq, 4)
            - action_tensor: Selected indices. Tensor (num_frames-1, num_seq)
            - baseline_iou: IoU of prediction trajectory of baseline (argmax tracker). Tensor (num_frames-1, num_seq)
            - explore_iou: IoU of prediction trajectory of exploration (sampling tracker). Tensor (num_frames-1, num_seq)
        """
        params = self.params
        params.net.net = self.net
        tracker = SLTTransT(params)

        results = {}
        search_images_list = []
        search_anno_list = []
        action_tensor_list = []
        iou_list = []

        num_frames = data['num_frames']
        images = data['search_images']
        gt_bbox = data['search_annos']
        template = data['template_images']
        template_bbox = data['template_annos']

        template = template + template
        template_bbox = template_bbox + template_bbox
        template_bbox = np.array(template_bbox)
        num_seq = len(num_frames)

        for idx in range(np.max(num_frames)):
            here_images = [img[idx] for img in images]
            here_gt_bbox = np.array([gt[idx] for gt in gt_bbox])

            here_images = here_images + here_images
            here_gt_bbox = np.concatenate([here_gt_bbox, here_gt_bbox], 0)

            if idx == 0:
                outputs = tracker.batch_init(template, template_bbox, here_gt_bbox)
                results['template_images'] = outputs['template_images'][num_seq:]

            else:
                outputs = tracker.batch_track(here_images, here_gt_bbox, action_mode='half')

                pred_bbox = outputs['pred_bboxes']
                search_images_list.append(outputs['search_images'][num_seq:])
                search_anno_list.append(outputs['gt_in_crop'][num_seq:])
                action_tensor_list.append(outputs['selected_indices'][num_seq:])

                pred_bbox_corner = batch_xywh2corner(pred_bbox)
                gt_bbox_corner = batch_xywh2corner(here_gt_bbox)
                here_iou = []
                for i in range(num_seq * 2):
                    bbox_iou = IoU(pred_bbox_corner[i], gt_bbox_corner[i])
                    here_iou.append(bbox_iou)
                iou_list.append(here_iou)

        results['search_images'] = torch.stack(search_images_list)
        results['search_anno'] = torch.stack(search_anno_list)

        iou_tensor = torch.tensor(iou_list, dtype=torch.float)
        results['baseline_iou'] = iou_tensor[:, :num_seq]
        results['explore_iou'] = iou_tensor[:, num_seq:]
        results['action_tensor'] = torch.stack(action_tensor_list)

        return results


    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.
            template_images : Tensor(bs, 3, 128, 128)
            search_images : Tensor(num_frames, bs, 3, 128, 128)
            search_anno : Tensor(num_frames, bs, 4)
            action_tensor: Tensor(num_frames, bs)
            reward_tensor: Tensor(num_frames, bs)
            slt_loss_weight
        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        num_frames = data['search_images'].shape[0]
        template_images = data['template_images'].repeat(num_frames,1,1,1,1)
        template_images = template_images.view(-1, *template_images.size()[2:])
        search_images = data['search_images'].reshape(-1, *data['search_images'].size()[2:])
        search_anno = data['search_anno'].reshape(-1, *data['search_anno'].size()[2:])

        outputs = self.net(search_images, template_images)

        # generate labels
        targets =[]
        targets_origin = search_anno
        for i in range(len(targets_origin)):
            h, w = search_images[i][0].shape
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        # Compute bbox loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        bbox_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Self-critical SLT loss
        score = outputs['pred_logits']
        score = score[:, :, 0] - score[:, :, 1]
        score = torch.sigmoid(score)
        bs = score.shape[0]

        hanning = np.hanning(32)
        window = np.outer(hanning, hanning)
        gpu_window = torch.from_numpy(window.flatten()).cuda()

        score = score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + gpu_window.view(1, -1) * cfg.TRACK.WINDOW_INFLUENCE
        inv_score = stable_inverse_sigmoid(score, self.params.sig_eps)

        log_softmax_score = F.log_softmax(inv_score * self.params.temp, dim=1)

        selected_logprobs = log_softmax_score[range(bs), data['action_tensor'].view(-1)]
        cls_loss = - selected_logprobs * data['reward_tensor'].view(-1)
        loss_dict['sl_clf'] = cls_loss.mean()

        total_losses = loss_dict['sl_clf']*data['slt_loss_weight'] + bbox_losses

        # Return training stats
        stats = {'Loss/total': total_losses.item(),
                 'Loss/sl_clf': loss_dict['sl_clf'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'bbox_iou': loss_dict['iou'].item()
                 }

        return total_losses, stats
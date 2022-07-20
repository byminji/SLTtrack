from . import BaseActor
import torch
import numpy as np
import torch.nn.functional as F

from ltr.data.bounding_box_utils import batch_xywh2corner
from ltr.data.bbox import IoU
from pytracking.tracker.slt_trdimp import SLTTrDiMP


class SLTTrDiMPActor(BaseActor):
    """Actor for sequence-level-training the TrDiMP network."""
    def __init__(self, net, objective, loss_weight=None, params=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.params = params # parameter for tracker

    def explore(self, data):
        """
        Exploration in Sequence-Level Training to collect data from inference mode
        Args:
            data: The input data, should contain the fields 'search_images', 'search_annos', 'template_images', 'template_annos'.
            - search_images: Search frames. list (num_seq) of array (num_frames, h, w, 3)
            - search_annos: Search annos. list (num_seq) of array (num_frames, 4)
            - template_images: list (num_seq) of array (h, w, 3)
            - template_annos: list (num_seq) of array (4)

        Returns:
            results: dictionary of exploration results
            - template_images: Template image patches. Tensor (num_seq, 3, 352, 352)
            - search_images: Sequence-level sampled image patches. Tensor (num_frames-1, num_seq, 3, 352, 352)
            - search_anno: Sequence-level sampled annotations. Tensor (num_frames-1, num_seq, 4)
            - action_tensor: Selected indices. Tensor (num_frames-1, num_seq)
            - baseline_iou: IoU of prediction trajectory of baseline (argmax tracker). Tensor (num_frames-1, num_seq)
            - explore_iou: IoU of prediction trajectory of exploration (sampling tracker). Tensor (num_frames-1, num_seq)
        """
        ############ Tracker
        params = self.params
        params.net.net = self.net
        tracker = SLTTrDiMP(params)

        results = {}
        search_images_list = []
        search_anno_list = []
        search_label_list = []
        search_proposals_list = []
        proposal_density_list = []
        gt_density_list = []
        action_tensor_list = []
        iou_list = []

        num_frames = data['num_frames']
        images = data['search_images']  # list num_seq(8) of array num_frames(16)xhxwx3
        gt_bbox = data['search_annos']  # list num_seq(8) of array num_frames(16)x4
        template = data['template_images']  # list num_seq(8) of array hxwx3
        template_bbox = data['template_annos']  # list num_seq(8) of array 4

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
                outputs = tracker.batch_init(template, template_bbox, here_gt_bbox) # z_crop : (13xnum_seqx2)x3x352x352
                results['template_images'] = outputs['template_images'][:, num_seq:] # num_samplesxnum_seqx3x352x352
                results['template_anno'] = outputs['template_anno'][:, num_seq:]     # num_samplesxnum_seqx4
                results['template_label'] = outputs['template_label'][:, num_seq:]   # num_samplesxnum_seqx22x22

            else:
                outputs = tracker.batch_track(here_images, here_gt_bbox, action_mode='half')

                pred_bbox = outputs['target_bbox']
                search_images_list.append(outputs['search_images'][num_seq:])
                search_anno_list.append(outputs['search_anno'][num_seq:])
                search_label_list.append(outputs['search_label'][num_seq:])
                search_proposals_list.append(outputs['search_proposals'][num_seq:])
                proposal_density_list.append(outputs['proposal_density'][num_seq:])
                gt_density_list.append(outputs['gt_density'][num_seq:])
                action_tensor_list.append(outputs['selected_indices_1d'][num_seq:])

                pred_bbox_corner = batch_xywh2corner(pred_bbox)
                gt_bbox_corner = batch_xywh2corner(here_gt_bbox)
                here_iou = []
                for i in range(num_seq * 2):
                    bbox_iou = IoU(pred_bbox_corner[i], gt_bbox_corner[i])
                    here_iou.append(bbox_iou)
                iou_list.append(here_iou)

        results['search_images'] = torch.stack(search_images_list)    # (num_frames-1)xnum_seqx3x352x352
        results['search_anno'] = torch.stack(search_anno_list)        # (num_frames-1)xnum_seqx4
        results['search_label'] = torch.stack(search_label_list)      # (num_frames-1)xnum_seqx23x23
        results['search_proposals'] = torch.stack(search_proposals_list)      # (num_frames-1)xnum_seqxnum_proposalx4
        results['proposal_density'] = torch.stack(proposal_density_list)  # (num_frames-1)xnum_seqxnum_proposal
        results['gt_density'] = torch.stack(gt_density_list)              # (num_frames-1)xnum_seqxnum_proposal

        iou_tensor = torch.tensor(iou_list, dtype=torch.float)
        results['baseline_iou'] = iou_tensor[:, :num_seq]       # (num_frames-1) x num_seq
        results['explore_iou'] = iou_tensor[:, num_seq:]        # (num_frames-1) x num_seq
        results['action_tensor'] = torch.stack(action_tensor_list)    # (num_frames-1) x num_seq

        return results


    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template_images', 'search_images', 'template_anno', 'search_anno'

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """

        # target_scores: list(6) of Tensor(23(num_frames-1)xbsx23x23), bb_scores : 23(num_frames-1)xbsx128
        target_scores, bb_scores = self.net(train_imgs=data['template_images'],
                                            test_imgs=data['search_images'],
                                            train_bb=data['template_anno'],
                                            train_label=data['template_label'],  ##
                                            test_proposals=data['search_proposals'])

        # Reshape bb reg variables
        is_valid = data['search_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # Sequence-Level classification loss
        loss_sl_target_classifier = 0
        loss_sl_init_clf = 0
        loss_sl_iter_clf = 0
        if 'sl_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_sl_losses_test = []
            for s in target_scores:
                bs = s.shape[0]*s.shape[1]
                scores_view = s.view(bs, -1)

                # mean max norm
                mean_val = torch.mean(scores_view, dim=-1)
                max_val = torch.max(scores_view, dim=-1)[0]
                scores_view = (scores_view - mean_val.unsqueeze(1)) / (max_val - mean_val).unsqueeze(1)
                range_mean, range_max = self.params.score_norm_meanmax
                scores_view = scores_view * (range_max - range_mean) + range_mean

                log_softmax_score = F.log_softmax(scores_view, dim=1)

                selected_logprobs = log_softmax_score[range(bs), data['action_tensor'].view(-1)]
                cls_loss = - selected_logprobs * data['reward_tensor'].view(-1)
                clf_sl_losses_test.append(cls_loss.mean())

            # Loss of the final filter
            clf_sl_loss_test = clf_sl_losses_test[-1]
            loss_sl_target_classifier = self.loss_weight['sl_clf'] * clf_sl_loss_test

            # Loss for the initial filter iteration
            if 'sl_init_clf' in self.loss_weight.keys():
                loss_sl_init_clf = self.loss_weight['sl_init_clf'] * clf_sl_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'sl_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['sl_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_sl_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_sl_losses_test[1:-1])])
                else:
                    loss_sl_iter_clf = (test_iter_weights / (len(clf_sl_losses_test) - 2)) * sum(clf_sl_losses_test[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_sl_target_classifier + loss_sl_init_clf + loss_sl_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'sl_clf' in self.loss_weight.keys():
            stats['Loss/sl_target_clf'] = loss_sl_target_classifier.item()
        if 'sl_init_clf' in self.loss_weight.keys():
            stats['Loss/sl_init_clf'] = loss_sl_init_clf.item()
        if 'sl_iter_clf' in self.loss_weight.keys():
            stats['Loss/sl_iter_clf'] = loss_sl_iter_clf.item()

        if 'sl_clf' in self.loss_weight.keys():
            stats['ClfTrain/sl_loss'] = clf_sl_loss_test.item()
            if len(clf_sl_losses_test) > 0:
                stats['ClfTrain/sl_init_loss'] = clf_sl_losses_test[0].item()
                if len(clf_sl_losses_test) > 2:
                    stats['ClfTrain/sl_iter_loss'] = sum(clf_sl_losses_test[1:-1]).item() / (len(clf_sl_losses_test) - 2)

        return loss, stats
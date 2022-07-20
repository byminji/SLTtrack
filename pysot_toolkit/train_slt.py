
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import random
import numpy as np
import argparse
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)

from pysot.utils.bbox import IoU
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.log_helper import print_speed
from pysot.models.slt_model_builder import ModelBuilder
from pysot.datasets.sequence_dataset import SequenceDataset
from pysot.core.config import cfg
from pysot.tracker.tracker_builder import build_tracker
from pysot.tracker.slt_siamrpn_tracker import stable_inverse_sigmoid



parser = argparse.ArgumentParser(description='SLT trainig for SiamRPN++')
parser.add_argument('--expr', type=str,
                    help='name of experiments')
parser.add_argument('--cfg', type=str,
                    help='path to configuration of tracking. ex) experiments/slt_siamrpnpp/slt_siamrpnpp.yaml')
parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                    help='Set cudnn benchmark on (1) or off (0) (default is on).')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=12345,
                    help='random seed')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if args.cudnn_benchmark:
        print("Using CuDNN Benchmark")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        print("Disabling CuDNN Benchmark")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def explore(tracker, batch_video):
    """
    Exploration in Sequence-Level Training to collect data from inference mode
    Args:
        tracker: Tracker model
        batch_video: The input data, should contain the fields 'images', 'gt_orig', 'template', 'template_bbox'.
            - images: Sequence frames. list (num_seq) of array (num_frames, h, w, 3)
            - gt_orig: Sequence annos. list (num_seq) of array (num_frames, 4)
            - template: Template frame. list (num_seq) of array (h, w, 3)
            - template_bbox: Template bbox. list (num_seq) of list (4)

    Returns:
        Dictionary of exploration results
            - z_crop: Template image patches. Tensor (num_seq, 3, 127, 127)
            - x_crop: Sequence-level sampled image patches. Tensor (num_frames-1, num_seq, 3, 255, 255)
            - label_loc: Sequence-level sampled annotations (normalized coordinates). Tensor (num_frames-1, num_seq, 4, 5, 25, 25)
            - label_loc_weight: Weights for localization. Tensor (num_frames-1, num_seq, 5, 25, 25)
            - label_cls: Weights for localization. Tensor (num_frames-1, num_seq, 5, 25, 25)
            - baseline_iou: IoU of prediction trajectory of baseline (argmax tracker). Tensor (num_seq, num_frames-1)
            - explore_iou: IoU of prediction trajectory of exploration (sampling tracker). Tensor (num_seq, num+frames-1)
            - action_tensor: Selected indices. Tensor (num_frames-1, num_seq)
            - penalty: Aspect ratio & window penalty calculated during inference. Tensor (num_frames-1, num_seq, 3125)
    """

    tracker.model.eval()

    x_crop_list = []
    label_loc_list = []
    label_loc_weight_list = []
    label_cls_list = []
    iou_list = []
    action_list = []
    penalty_list = []

    num_frames = batch_video['num_frames']
    images = batch_video['images']
    gt_bbox = batch_video['gt_orig']
    template = batch_video['template']
    template_bbox = batch_video['template_bbox']

    template = template + template
    template_bbox = template_bbox + template_bbox
    template_bbox = np.array(template_bbox)
    num_seq = len(num_frames)
    z_crop = None
    st = time.time()

    for idx in range(np.max(num_frames)):
        # Duplicate twice for argmax & sampling trackers
        here_images = [img[idx] for img in images]
        here_images = here_images + here_images  # *2
        here_gt_bbox = np.array([gt[idx] for gt in gt_bbox])
        here_gt_bbox = np.concatenate([here_gt_bbox, here_gt_bbox], 0)

        # Batch tracking
        if idx == 0:
            with torch.no_grad():
                z_crop = tracker.batch_init(template, template_bbox, here_gt_bbox)
        else:
            with torch.no_grad():
                outputs = tracker.batch_track(here_images, here_gt_bbox, action_mode='half')

            pred_bbox = outputs['bbox']
            x_crop_list.append(outputs['x_crop'][num_seq:])
            label_loc_list.append(outputs['label_loc'][num_seq:])
            label_loc_weight_list.append(outputs['label_loc_weight'][num_seq:])
            label_cls_list.append(outputs['label_cls'][num_seq:])

            action_list.append(outputs['selected_indices'][num_seq:])
            penalty_list.append(outputs['penalty'][num_seq:])

            here_iou = []
            for i in range(num_seq * 2):
                iou = IoU(pred_bbox[i], here_gt_bbox[i])
                here_iou.append(iou)
            iou_list.append(here_iou)

    iou_tensor = torch.tensor(iou_list, dtype=torch.float).permute(1, 0).contiguous()
    elapsed_time = time.time() - st
    fps = (np.max(num_frames) * num_seq * 2) / elapsed_time

    return {'z_crop': z_crop[num_seq:],
            'x_crop': torch.stack(x_crop_list),
            'label_loc': torch.stack(label_loc_list),
            'label_loc_weight': torch.stack(label_loc_weight_list),
            'label_cls': torch.stack(label_cls_list),
            'baseline_iou': iou_tensor[:num_seq],
            'explore_iou': iou_tensor[num_seq:],
            'action_tensor': torch.stack(action_list),
            'penalty': torch.stack(penalty_list),
            'elapsed_time': elapsed_time,
            'fps': fps}


def build_optimizer(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

    if cfg.TRAIN.RL_TRAIN_BACKBONE:
        print('training backbone.')
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    # add neck module
    if cfg.ADJUST.LR > 0.0000001:
        print('Train Neck')
        trainable_params += [{'params': model.neck.parameters(), 'lr': cfg.ADJUST.LR * cfg.TRAIN.BASE_LR}]
    else:
        for param in model.neck.parameters():
            param.requires_grad = False

    # add rpn module
    if cfg.RPN.LR > 0.00000001:
        print('Train RPN')

        trainable_params += [
            {'params': model.rpn_head.parameters(), 'lr': cfg.RPN.LR * cfg.TRAIN.BASE_LR}]
    else:
        for param in model.rpn_head.parameters():
            param.requires_grad = False

    if cfg.TRAIN.OPTIM == 'sgd':
        optimizer = torch.optim.SGD(trainable_params,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIM == 'adam':
        optimizer = torch.optim.Adam(trainable_params, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    return optimizer, lr_scheduler


miou_list = []
def train(tracker, dataloader, optimizer, lr_scheduler):
   num_per_epoch = len(dataloader.dataset) // cfg.TRAIN.EPOCH // cfg.TRAIN.NUM_SEQ
   start_epoch = cfg.TRAIN.START_EPOCH
   epoch = start_epoch
   avg_time = 0

   end = time.time()
   for step, batch in enumerate(dataloader):
        if epoch != step // num_per_epoch + start_epoch:
            epoch = step // num_per_epoch + start_epoch

            torch.save(
                {'epoch': epoch,
                 'state_dict': tracker.model.module.state_dict(),
                 'optimizer': optimizer.state_dict()},
                os.path.join(cfg.TRAIN.SNAPSHOT_DIR, '%s_%depoch.pth' % (args.expr, epoch)))

            if epoch == cfg.TRAIN.EPOCH:
                return
            if epoch != start_epoch:
                lr_scheduler.step(epoch)

        data_time = time.time() - end

        # Exploration in Sequence-Level Training to collect data from inference mode
        explore_result = explore(tracker, batch)

        # For writing log
        reward_record = []
        miou_record = []
        e_miou_record = []

        # Calculate reward tensor
        num_seq = len(batch['num_frames'])
        reward_tensor = torch.zeros(explore_result['action_tensor'].size())
        baseline_iou = explore_result['baseline_iou']
        explore_iou = explore_result['explore_iou']
        for seq_idx in range(num_seq):
            num_frames = batch['num_frames'][seq_idx] - 1
            b_miou = torch.mean(baseline_iou[seq_idx, :num_frames])
            e_miou = torch.mean(explore_iou[seq_idx, :num_frames])
            miou_record.append(b_miou.item())
            e_miou_record.append(e_miou.item())

            b_reward = b_miou.item()
            e_reward = e_miou.item()
            iou_gap = e_reward - b_reward
            reward_record.append(iou_gap)
            reward_tensor[:num_frames, seq_idx] = iou_gap

        num_frames = explore_result['x_crop'].size(0)
        x_crop = explore_result['x_crop'].view(-1, 3, 255, 255)
        z_crop = explore_result['z_crop'].view(1, num_seq, 3, 127, 127).repeat(num_frames, 1, 1, 1, 1).view(-1, 3, 127, 127)
        label_cls = explore_result['label_cls'].view(num_seq * num_frames, *explore_result['label_cls'].size()[2:])
        label_loc = explore_result['label_loc'].view(num_seq * num_frames, *explore_result['label_loc'].size()[2:])
        label_loc_weight = explore_result['label_loc_weight'].view(num_seq * num_frames,
                                                                   *explore_result['label_loc_weight'].size()[2:])

        action_tensor = explore_result['action_tensor'].view(-1).cuda()
        reward_tensor = reward_tensor.view(-1).cuda()
        penalty = explore_result['penalty'].view(-1, *explore_result['penalty'].size()[2:])
        explore_fps = explore_result['fps']

        # Training mode
        cursor = 0
        num_samples = x_crop.size(0)
        loss_sum = 0
        reg_loss_sum = 0
        print('Backward with %d samples' % num_samples)
        optimizer.zero_grad()
        while cursor < num_samples:
            model_inputs = {
                'template': z_crop[cursor:cursor + cfg.TRAIN.BATCH_SIZE],
                'search': x_crop[cursor:cursor + cfg.TRAIN.BATCH_SIZE],
            }
            if cfg.TRAIN.RL_TRAIN_LOC:
                model_inputs['label_cls'] = label_cls[cursor:cursor + cfg.TRAIN.BATCH_SIZE]
                model_inputs['label_loc'] = label_loc[cursor:cursor + cfg.TRAIN.BATCH_SIZE]
                model_inputs['label_loc_weight'] = label_loc_weight[cursor:cursor + cfg.TRAIN.BATCH_SIZE]

            outputs = tracker.model(model_inputs)
            bs = outputs['cls'].size(0)

            # Self-critical SLT loss
            score = outputs['cls']
            score = score.view(bs, 2, -1).permute(0, 2, 1).contiguous()  # tensor(num_seq, num_anchors, 2)
            score = score[:, :, 1] - score[:, :, 0]
            score = torch.sigmoid(score)
            score = score * penalty[cursor:cursor + cfg.TRAIN.BATCH_SIZE].to(score.device)
            score = score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + tracker.gpu_window.view(1, -1) * cfg.TRACK.WINDOW_INFLUENCE
            inv_score = stable_inverse_sigmoid(score, cfg.TRAIN.SIG_EPS)

            log_softmax_score = F.log_softmax(inv_score * cfg.TRAIN.TEMP, dim=1)

            selected_logprobs = log_softmax_score[range(bs), action_tensor[cursor:cursor + cfg.TRAIN.BATCH_SIZE]]
            cls_loss = - selected_logprobs * reward_tensor[cursor:cursor + cfg.TRAIN.BATCH_SIZE]
            cls_loss = cls_loss.mean() * (bs / num_samples)

            # Bounding box regression loss (baseline)
            if cfg.TRAIN.RL_TRAIN_LOC:
                reg_loss = outputs['reg_loss'].mean() * (bs / num_samples)
            else:
                reg_loss = torch.tensor(0)
            final_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * reg_loss

            final_loss.backward()

            reg_loss_sum += reg_loss.item()
            loss_sum += cls_loss.item()
            cursor += cfg.TRAIN.BATCH_SIZE

        grad_norm = clip_grad_norm_(tracker.model.parameters(), cfg.TRAIN.GRAD_CLIP)
        optimizer.step()
        tracker.model.module.eval()

        batch_time = time.time() - end
        avg_time = (avg_time * step + batch_time) / (step + 1)
        print('[%s][%d epoch][%d step] loss %.4f, reg_loss: %.4f, reward: %.4f, b_miou: %.4f, e_miou: %.4f, elapsed_time: %.1f,explore_fps: %.1f' %
              (args.expr, epoch+1, step, loss_sum, reg_loss_sum, np.mean(reward_record), np.mean(miou_record), np.mean(e_miou_record), batch_time - data_time, explore_fps))
        if (step + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            print_speed(step + 1 + start_epoch * num_per_epoch,
                        avg_time,
                        cfg.TRAIN.EPOCH * num_per_epoch)
        miou = np.mean(miou_record)
        miou_list.append(miou)
        log = {
            'loss': loss_sum,
            'reg_loss': reg_loss_sum,
            'reward': np.mean(reward_record),
            'mIoU': miou,
            'mIoU10': np.mean(miou_list[-10:]),
            'mIoU100': np.mean(miou_list[-100:]),
            'backbone_lr': optimizer.param_groups[0]['lr'],
            'top_lr': optimizer.param_groups[-1]['lr'],
            'grad_norm': grad_norm,
            'batch_time': batch_time,
            'data_time': data_time,
        }

        # Add your log function in here
        # wandb.log(log)

        end = time.time()


def main():
    # load cfg
    cfg.merge_from_file(args.cfg)

    seq_dataset = SequenceDataset(cfg.TRAIN.NUM_FRAMES)

    print('Total %d sequence' % len(seq_dataset))
    def collate_fn(batch):
        ret = {}
        for k in batch[0].keys():
            here_list = []
            for ex in batch:
                here_list.append(ex[k])
            ret[k] = here_list
        return ret
    seq_dataloader = DataLoader(seq_dataset, shuffle=False, batch_size=cfg.TRAIN.NUM_SEQ, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    print('Load model')
    model = ModelBuilder().cuda().train()

    optim, lr_scheduler = build_optimizer(model)

    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '/', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)

    # resume training
    if cfg.TRAIN.RESUME:
        resume_snapshot_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optim, resume_snapshot_path)
        print('Resume training...')
    elif cfg.TRAIN.PRETRAINED:
        snapshot_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, cfg.TRAIN.PRETRAINED)
        snapshot = torch.load(snapshot_path, map_location='cpu')
        model.cpu()
        if 'state_dict' in snapshot:
            model.load_state_dict(snapshot['state_dict'])
        else:
            model.load_state_dict(snapshot)

    model.cuda().eval()
    tracker = build_tracker(model)

    train(tracker, seq_dataloader, optim, lr_scheduler)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()

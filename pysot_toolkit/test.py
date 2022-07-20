# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.core.local_config import local_config  as local
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot_toolkit.toolkit.datasets import DatasetFactory
from pysot_toolkit.toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--expr', default='', type=str)
parser.add_argument('--e', dest='epoch',
                    help='training epoch', default=0, type=int)
parser.add_argument('--dataset', type=str,
                    help='datasets')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
                    help='whether visualize result')
parser.add_argument('--g', dest='gpu_num',
                    help='gpu number', default='0', type=str)
parser.add_argument('--start_seq', default=1, type=int)
parser.add_argument('--end_seq', default=700, type=int)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
print("GPU_NUM: ", args.gpu_num)


def main():
    cfg_file_name = os.path.join('%s_%d.yaml' % (args.expr_name, args.session))
    cfg.merge_from_file(cfg_file_name)

    if 'slt' in args.expr_name:
        if 'siamrpnn' in args.expr_name:
            from pysot.models.slt_model_builder import ModelBuilder
        elif 'siamattn' in args.expr_name:
            from pysot.models.slt_mask_model_builder import ModelBuilder
        else:
            raise NotImplementedError
    else:
        if 'siamrpnpp' in args.expr_name:
            from pysot.models.model_builder import ModelBuilder
        elif 'siamattn' in args.expr_name:
            from pysot.models.mask_model_builder import ModelBuilder
        else:
            raise NotImplementedError

    # create model
    model = ModelBuilder()

    # load model
    snapshot_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, '%s_%depoch.pth' % (args.expr_name, args.epoch))
    snapshot = torch.load(snapshot_path)
    if 'state_dict' in snapshot:
        snapshot = snapshot['state_dict']
    for k in list(snapshot):
        if 'bbox_head' in k:  # bbox_head -> refine_head
            new_k = k.replace('bbox', 'refine')
            snapshot[new_k] = snapshot.pop(k)
    model.load_state_dict(snapshot, strict=False)
    model.cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset_name = args.dataset
    dataset_root = os.path.join(local.ORIG_ROOT, args.dataset)
    if args.dataset == 'GOT-10k-Test':
        dataset_root = os.path.join(local.GOT10K_ORIG_PATH, 'test')
    elif 'GOT-10k' in args.dataset:
        dataset_root = os.path.join(local.GOT10K_ORIG_PATH, 'val')
    elif 'LaSOT' in args.dataset:
        dataset_root = os.path.join(local.LASOT_ORIG_PATH, 'test')
    elif 'NFS' in args.dataset:
        dataset_root = os.path.join(local.ORIG_ROOT, 'NFS/30')
    elif 'UAV' in args.dataset:
        dataset_root = os.path.join(local.ORIG_ROOT, 'UAV123/data_seq/UAV123')
    elif 'TNL2K' in args.dataset:
        dataset_root = os.path.join(local.ORIG_ROOT, args.dataset, 'test')
    elif args.dataset == 'TrackingNet':
        dataset_root = os.path.join(local.TRACKINGNET_ORIG_PATH, 'TEST')

    dataset = DatasetFactory.create_dataset(name=dataset_name,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = '%s_%depoch' % (args.expr_name, args.epoch)

    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bbox = [
                        pred_bbox[0],  # x1
                        pred_bbox[1],  # y1
                        pred_bbox[2] - pred_bbox[0],  # w = x2 - x1
                        pred_bbox[3] - pred_bbox[1]  # h = y2 - y1
                    ]
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        dataset_name = args.dataset
        for v_idx, video in enumerate(dataset):
            if v_idx < args.start_seq - 1:
                continue
            if v_idx == args.end_seq:
                break
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            # scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    # scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bbox = [
                        pred_bbox[0],  # x1
                        pred_bbox[1],  # y1
                        pred_bbox[2] - pred_bbox[0],  # w = x2 - x1
                        pred_bbox[3] - pred_bbox[1]  # h = y2 - y1
                    ]
                    if 'VOT' in args.dataset and cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    pred_bboxes.append(pred_bbox)
                    # scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k-Test' == args.dataset:
                video_path = os.path.join('results', dataset_name, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                for j in range(3):
                    result_path = os.path.join(video_path, '{}_{:03d}.txt'.format(video.name, j))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' in args.dataset:
                video_path = os.path.join('results', dataset_name, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', dataset_name, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()

import numpy as np
from os.path import join
from os import listdir
import json
import cv2
import pysot.core.local_config as local
original_got_path = ''
cropped_got_path = ''

for split in ['train', 'val']: #['val']:
    dataset = dict()
    video_base_path = join(original_got_path, split)
    videos = sorted(listdir(video_base_path))[:-1]
    n_videos = len(videos)
    for n, video in enumerate(videos):
        print('subset: {} video id: {:04d} / {:04d}'.format(split, n, n_videos))
        ann_path = join(video_base_path, video, 'groundtruth.txt')
        anns = np.genfromtxt(ann_path, dtype=float, delimiter=',')
        video_crop_base_path = join(split, video)

        if len(anns) > 0:
            dataset[video_crop_base_path] = dict()

        # im = cv2.imread(join(video_base_path, video, '00000001.jpg'))
        # im_size = im.shape[:2]
        # dataset[video_crop_base_path]['im_size'] = im_size

        box_dict = dict()
        for fid, ann in enumerate(anns):
            rect = list(map(int, anns[fid]))
            bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                continue
            box_dict['%06d' % fid] = bbox
        dataset[video_crop_base_path]['00'] = box_dict

    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open(join(cropped_got_path, '{}.json'.format(split)), 'w'), indent=4, sort_keys=True)
    print('done!')


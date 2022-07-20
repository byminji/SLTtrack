import numpy as np
from os.path import join
from os import listdir
import json
import cv2
original_data_path = ''

dataset = dict()
# for split in ['TRAIN_0', 'TRAIN_1', 'TRAIN_2', 'TRAIN_3', 'TRAIN_4', 'TRAIN_5', 'TRAIN_6', 'TRAIN_7', 'TRAIN_8', 'TRAIN_9', 'TRAIN_10', 'TRAIN_11']: #['val']:
for split in ['TRAIN_6', 'TRAIN_7', 'TRAIN_8', 'TRAIN_9', 'TRAIN_10', 'TRAIN_11']: #['val']:
    anno_base_path = join(original_data_path, split, 'anno')
    anno_fnames = sorted(listdir(anno_base_path))
    n_videos = len(anno_fnames)
    for n, anno_fn in enumerate(anno_fnames):
        print('subset: {} video id: {:04d} / {:04d}'.format(split, n, n_videos))
        ann_path = join(anno_base_path, anno_fn)
        anns = np.genfromtxt(ann_path, dtype=float, delimiter=',')
        video_name = join(split, 'frames', anno_fn[:-4])

        if len(anns) > 0:
            dataset[video_name] = {
                'tracks': {},
                'frame_names': []
            }

        box_dict = dict()
        for fid, ann in enumerate(anns):
            frame_name = '%d' % fid
            dataset[video_name]['frame_names'].append(frame_name)
            rect = list(map(int, anns[fid]))
            bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                continue
            box_dict[frame_name] = bbox
        dataset[video_name]['tracks']['00'] = box_dict

print('save json (dataset), please wait 20 seconds~')
json.dump(dataset, open(join(original_data_path, 'TRAIN_6to11_original.json'), 'w'), indent=4, sort_keys=True)
print('done!')


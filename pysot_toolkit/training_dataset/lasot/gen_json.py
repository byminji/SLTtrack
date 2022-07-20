import numpy as np
from os.path import join
from os import listdir
import json

original_data_path = ''
cropped_data_path = ''

##### Directories split by classes
# dataset = dict()
# classes = sorted(listdir(original_data_path))
#
# for cls in classes:
#     videos = sorted(listdir(join(original_data_path, cls)))
#
#     n_videos = len(videos)
#     for n, video in enumerate(videos):
#         print('subset: {} video id: {:04d} / {:04d}'.format(cls, n, n_videos))
#         ann_path = join(original_data_path, cls, video, 'groundtruth.txt')
#         anns = np.genfromtxt(ann_path, dtype=float, delimiter=',')
#         video_crop_base_path = join(cls, video, 'img')
#
#         if len(anns) > 0:
#             dataset[video_crop_base_path] = dict()
#
#         box_dict = dict()
#         for fid, ann in enumerate(anns):
#             rect = list(map(int, anns[fid]))
#             bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
#             if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
#                 continue
#             box_dict['%06d' % fid] = bbox
#         dataset[video_crop_base_path]['00'] = box_dict
#
# print('save json (dataset), please wait 20 seconds~')
# json.dump(dataset, open(join(cropped_data_path, 'train.json'), 'w'), indent=4, sort_keys=True)
# print('done!')


##### Directories split by train/test subset
for split in ['train', 'test']: #['val']:
    dataset = dict()
    video_base_path = join(original_data_path, split)
    videos = sorted(listdir(video_base_path))[:-1]
    n_videos = len(videos)
    for n, video in enumerate(videos):
        print('subset: {} video id: {:04d} / {:04d}'.format(split, n, n_videos))
        ann_path = join(original_data_path, split, video, 'groundtruth.txt')
        anns = np.genfromtxt(ann_path, dtype=float, delimiter=',')
        video_crop_base_path = join(split, video, 'img')

        if len(anns) > 0:
            dataset[video_crop_base_path] = dict()

        box_dict = dict()
        for fid, ann in enumerate(anns):
            rect = list(map(int, anns[fid]))
            bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                continue
            box_dict['%06d' % fid] = bbox
        dataset[video_crop_base_path]['00'] = box_dict

    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open(join(cropped_data_path, '{}.json'.format(split)), 'w'), indent=4, sort_keys=True)
    print('done!')
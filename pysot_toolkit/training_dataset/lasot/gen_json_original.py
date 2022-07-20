import numpy as np
from os.path import join
from os import listdir
import json

data_root = ''

##### Directories split by classes
for split in ['train']:
    original_data_path = join(data_root, split)
    dataset = dict()
    classes = sorted(listdir(original_data_path))

    for cls in classes:
        videos = sorted(listdir(join(original_data_path, cls)))

        n_videos = len(videos)
        for n, video in enumerate(videos):
            print('subset: {} video id: {:04d} / {:04d}'.format(cls, n, n_videos))
            ann_path = join(original_data_path, cls, video, 'groundtruth.txt')
            anns = np.genfromtxt(ann_path, dtype=float, delimiter=',')

            video_name = join(split, cls, video, 'img')

            if len(anns) > 0:
                dataset[video_name] = {
                    'tracks': {},
                    'frame_names': []
                }

            box_dict = dict()
            for fid, ann in enumerate(anns):
                frame_name = '%08d' % (fid + 1)
                dataset[video_name]['frame_names'].append(frame_name)

                rect = list(map(int, anns[fid]))
                bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
                if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                    continue
                box_dict[frame_name] = bbox
            dataset[video_name]['tracks']['00'] = box_dict

    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open(join(data_root, 'original_{}.json'.format(split)), 'w'), indent=4, sort_keys=True)
    print('done!')


##### Directories split by train/test subset
# for split in ['train', 'test']: #['val']:
#     dataset = dict()
#     video_base_path = join(original_data_path, split)
#     videos = sorted(listdir(video_base_path))[:-1]
#     n_videos = len(videos)
#     for n, video in enumerate(videos):
#         print('subset: {} video id: {:04d} / {:04d}'.format(split, n, n_videos))
#         ann_path = join(video_base_path, video, 'groundtruth.txt')
#         anns = np.genfromtxt(ann_path, dtype=float, delimiter=',')
#
#         video_name = join(split, video, 'img')
#
#         if len(anns) > 0:
#             dataset[video_name] = {
#                 'tracks': {},
#                 'frame_names': []
#             }
#
#         box_dict = dict()
#         for fid, ann in enumerate(anns):
#             frame_name = '%08d' % (fid + 1)
#             dataset[video_name]['frame_names'].append(frame_name)
#             rect = list(map(int, anns[fid]))
#             bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
#             if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
#                 continue
#             box_dict[frame_name] = bbox
#         dataset[video_name]['tracks']['00'] = box_dict
#
#     print('save json (dataset), please wait 20 seconds~')
#     json.dump(dataset, open(join(original_data_path, 'original_{}.json'.format(split)), 'w'), indent=4, sort_keys=True)
#     print('done!')
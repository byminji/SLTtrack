# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import json
from os import makedirs
from os.path import join, isdir
import numpy as np
import cv2

data_root = ''
save_root = ''
video_base_path = join(data_root, 'train')

print('load json (raw ytb_vos info), please wait 10 seconds~')
ytb_vos = json.load(open('instances_train.json', 'r'))
dataset = dict()
for n, (k, v) in enumerate(ytb_vos.items()):
    print('video id: {:04d} / {:04d}'.format(n, len(ytb_vos)))
    video_name = join('train', k)

    dataset[video_name] = {
        'tracks': {},
        'frame_names': []
    }

    # Make mask dir
    bin_mask_save_path = join(save_root, 'mask', k)
    if not isdir(bin_mask_save_path): makedirs(bin_mask_save_path)

    # for i, o in enumerate(list(v)):
    for i, (obj_id, obj) in enumerate(v.items()):
        obj_id_int = int(obj_id)
        box_dict = dict()
        trackid = "{:02d}".format(i)
        for frame in obj:
            file_name = frame['file_name']
            frame_name = '{:05d}'.format(int(file_name.split('/')[-1]))

            # save mask
            mask_name = join(video_base_path, 'Annotations', k, '{}.png'.format(frame_name))
            img = cv2.imread(mask_name, 0)
            mask = np.uint8(img == obj_id_int)
            if mask is None:
                print('no ann')
                continue
            cv2.imwrite(join(bin_mask_save_path, '{}_{}.png'.format(frame_name, trackid)), mask * 255)

            # if i == 0:
            #     dataset[video_name]['frame_names'].append(frame_name)
            if frame_name not in dataset[video_name]['frame_names']:
                dataset[video_name]['frame_names'].append(frame_name)
            bbox = frame['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            box_dict[frame_name] = bbox

        dataset[video_name]['tracks'][trackid] = box_dict
    dataset[video_name]['frame_names'].sort(key=int)

json.dump(dataset, open('original_train.json', 'w'), indent=4, sort_keys=True)
print('done!')

from pycocotools.coco import COCO
from os.path import join, isdir
from os import makedirs, rmdir
import numpy as np
import cv2
import json

# Generate binary masks
# siam_orig
#     - <dataset>
#         - mask
#             - <video id>
#                 - <frameid>_<objectid>.png
#         - train2017
#             - <video id >
#                 - <frameid>.jpg

data_root = ''
save_root = ''
for data_type in ['train2017']:
    dataset = dict()
    ann_file = '{}/annotations/instances_{}.json'.format(data_root, data_type)
    coco = COCO(ann_file)
    n_imgs = len(coco.imgs)

    for n, img_id in enumerate(coco.imgs):
        print('subset: {} image id: {:04d} / {:04d}'.format(data_type, n, n_imgs))
        img = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        img_name = img['file_name'].split('/')[-1].split('.')[0]
        img = cv2.imread(join(data_root, data_type, '{}.jpg'.format(img_name)))

        if len(anns) > 0:
            bin_mask_save_path = join(save_root, 'mask', img_name)
            if not isdir(bin_mask_save_path): makedirs(bin_mask_save_path)
            img_save_path = join(save_root, data_type, img_name)
            if not isdir(img_save_path): makedirs(img_save_path)

            dataset[img_save_path] = {
                'tracks': {},
                'frame_names': ['000000']
            }
        else:
            continue

        for trackid, ann in enumerate(anns):
            rect = ann['bbox']
            c = ann['category_id']

            # Find bounding box
            bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                continue

            # Generate binary mask
            bin_mask = coco.annToMask(ann).astype(np.uint8)

            # Save binary segmentation mask
            cv2.imwrite(join(bin_mask_save_path, '000000_{:02d}.png'.format(trackid)), bin_mask*255)

            # Save bounding box
            dataset[img_save_path]['tracks']['{:02d}'.format(trackid)] = {'000000': bbox}

        # Save image (for multiple objects)
        cv2.imwrite(join(img_save_path, '000000.jpg'), img)

    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open(join(save_root, 'original_{}.json'.format(data_type)), 'w'), indent=4, sort_keys=True)
    print('done!')

import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import json

# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return x


def crop_img(video, anns, set_crop_base_path, set_img_base_path, set_mask_base_path, instanc_size=511):
    video_name = video.split('/')[-1]
    frame_crop_base_path = join(set_crop_base_path, video_name)
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    for frameid in anns['frame_names']:
        img = cv2.imread('{}/{}/{:05d}.jpg'.format(set_img_base_path, video_name, int(frameid)))
        avg_chans = np.mean(img, axis=(0, 1))

        for trackid, ann in anns['tracks'].items():
            rect = ann[frameid]
            bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
            if rect[2] <= 0 or rect[3] <=0:
                continue

            # crop image
            x = crop_like_SiamFC(img, bbox, instanc_size=instanc_size, padding=avg_chans)
            cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, int(trackid))), x)

            # crop mask
            mask = cv2.imread('{}/{}/{:05d}_{:02d}.png'.format(set_mask_base_path, video_name, int(frameid), int(trackid)))
            m = crop_like_SiamFC(mask, bbox, instanc_size=instanc_size)
            cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.m.jpg'.format(0, int(trackid))), m)


def main(instanc_size=511, num_threads=12):
    data_root = ''
    crop_root = ''
    crop_path = join(crop_root)
    if not isdir(crop_path): mkdir(crop_path)

    for dataType in ['train']:
        set_img_base_path = join(data_root, dataType)
        set_mask_base_path = join(data_root, 'mask')
        set_crop_base_path = join(crop_path, dataType)

        ann_path = join(data_root, 'original_{}.json'.format(dataType))
        ann = json.load(open(ann_path, 'r'))
        # e.g. ann['train/video_id'] = {
        # 'frame_names': [<frame_id>, ...]
        # 'tracks': { <instance id>:
        #               {<frame_id>: [bbox], ...}}}

        n_imgs = len(ann)
        print(n_imgs)

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_img, video_name, video_ann,
                                  set_crop_base_path, set_img_base_path, set_mask_base_path,
                                  instanc_size) for video_name, video_ann in ann.items()]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_imgs, prefix=dataType, suffix='Done ', barLength=40)
    print('done')


if __name__ == '__main__':
    since = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

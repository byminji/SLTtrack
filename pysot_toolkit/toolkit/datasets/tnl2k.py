
import json
import os
import configparser
import csv

from tqdm import tqdm

from .dataset import Dataset
from .video import Video

class TNL2KVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(TNL2KVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)


class TNL2KDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "TNL2K"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(TNL2KDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            video_name = video.replace(' ', '').replace('&', '')   # remove white space
            pbar.set_postfix_str(video)
            img_names = [os.path.join(video_name, 'imgs', k) for k in meta_data[video]['image_files']]

            # Exception
            if video_name in ['advSamp_Baseball_game_002-Done', 'advSamp_monitor_bikeyellow']:
                gts = meta_data[video]['gt_rect'][:-1]
                print(gts[0])
            else:
                gts = meta_data[video]['gt_rect']

            self.videos[video] = TNL2KVideo(name=video,
                                             root=dataset_root,
                                             video_dir=os.path.join(video_name, 'imgs'),
                                             init_rect=meta_data[video]['gt_rect'][0],
                                             img_names=img_names,
                                             gt_rect=gts,
                                            attr=None)

        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())

import json
import os
import configparser
import csv

from tqdm import tqdm

from .dataset import Dataset
from .video import Video

class GOT10kVideo(Video):
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
        super(GOT10kVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                             if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        self.pred_trajs = {}
        for name in tracker_names:
            #os.path.join(video_path, '{}_001.txt'.format(video.name))
            traj_file = os.path.join(path, name, self.name, self.name +'_001.txt')
            # print(traj_file)
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f:
                    pred_traj = [list(map(float, x.strip().split(',')))
                                             for x in f.readlines()]
                    if len(pred_traj) != len(self.gt_traj):
                        print(name, len(pred_traj), len(self.gt_traj), self.name)
                    if store:
                        self.pred_trajs[name] = pred_traj
                    else:
                        return pred_traj
            else:
                print(traj_file)
        self.tracker_names = list(self.pred_trajs.keys())

class GOT10kDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS30" or "NFS240"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(GOT10kDataset, self).__init__(name, dataset_root)
        #with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
        #    meta_data = json.load(f)
        parser = configparser.ConfigParser()
        with open(os.path.join(dataset_root, 'list.txt'), 'r') as f:
            video_dirs = f.read().splitlines()

        #load videos
        pbar = tqdm(video_dirs, desc='loading ' + name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            video_path = os.path.join(dataset_root, video)
            metainfo_path = os.path.join(video_path, 'meta_info.ini')
            gt_path = os.path.join(video_path, 'groundtruth.txt')
            with open(gt_path, 'r') as vid_f:
                img_names = []
                gts = []
                for img_idx, gt in enumerate(csv.reader(vid_f), start=1):
                    img_names.append(os.path.join(video, '%08d.jpg' % (img_idx)))
                    gts.append([float(gt[0]), float(gt[1]), float(gt[2]), float(gt[3])])
            # meta
            parser.read(metainfo_path)
            metainfo = {}
            for metakey in parser.options('METAINFO'):
                metainfo[metakey] = parser.get('METAINFO', metakey)

            self.videos[video] = GOT10kVideo(name=video, #parser.get('METAINFO', 'object_class'),
                                             root=dataset_root,
                                             video_dir=video,
                                             init_rect=gts[0],
                                             img_names=img_names,
                                             gt_rect=gts,
                                             attr=metainfo)


class GOT10kTestDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS30" or "NFS240"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(GOT10kTestDataset, self).__init__(name, dataset_root)
        parser = configparser.ConfigParser()
        with open(os.path.join(dataset_root, 'list.txt'), 'r') as f:
            video_dirs = f.read().splitlines()

        #load videos
        pbar = tqdm(video_dirs, desc='loading ' + name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            video_path = os.path.join(dataset_root, video)
            gt_path = os.path.join(video_path, 'groundtruth.txt')

            img_names = [img for img in os.listdir(video_path) if img.endswith(".jpg")]
            img_names.sort(key=lambda f: int(f[:-4]))
            img_names = [os.path.join(video, img) for img in img_names]

            with open(gt_path, 'r') as vid_f:
                gts = []
                for img_idx, gt in enumerate(csv.reader(vid_f), start=1):
                    gts.append([float(gt[0]), float(gt[1]), float(gt[2]), float(gt[3])])

            init_rect = gts[0]
            gt_rect = [init_rect for i in range(len(img_names))]

            self.videos[video] = GOT10kVideo(name=video,
                                             root=dataset_root,
                                             video_dir=video,
                                             init_rect=init_rect,
                                             img_names=img_names,
                                             gt_rect=gt_rect,
                                             attr=None)

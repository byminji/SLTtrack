import os
from yacs.config import CfgNode as CN

__C = CN()

local_config = __C

__C.SNAPSHOT_DIR = ''   # path where snapshot (checkpoint) of models should be saved
__C.CROP_ROOT = ''  # root of cropped images for baseline pre-training
__C.ORIG_ROOT = ''  # root of original datasets for SLT fine-tuning

# path of cropped images for baseline pre-training
__C.VID_PATH = ''
__C.DET_PATH = ''
__C.COCO_PATH = ''
__C.YTVOS_PATH = ''
__C.LASOT_PATH = ''
__C.TRACKINGNET_PATH = ''
__C.GOT10K_PATH = ''

# root of original datasets for SLT fine-tuning
__C.LASOT_ORIG_PATH = ''
__C.TRACKINGNET_ORIG_PATH =''
__C.GOT10K_ORIG_PATH = ''
__C.COCO_ORIG_PATH = ''
__C.YTVOS_ORIG_PATH = ''

__C.OTB_PATH = ''
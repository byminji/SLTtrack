# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN
# import pysot.core.local_config as local
from pysot.core.local_config import local_config as local
import os

__C = CN()

cfg = __C

__C.META_ARC = "siamrpn_r50_l234_dwxcorr"

__C.CUDA = True

# cam
__C.GRAD_CAM = False

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()
__C.TRAIN.REWARD_WEIGHT = 1
__C.TRAIN.RL_TRAIN_LOC = True
__C.TRAIN.RL_TRAIN_BACKBONE = False
__C.TRAIN.LOSS_TEMP = False
__C.TRAIN.TEMP_DIV = False
__C.TRAIN.TEMP_INC = False
__C.TRAIN.NO_NEG_LOGIT = False
__C.TRAIN.LOST_PENALTY = False
__C.TRAIN.STD_INIT = 0.0
__C.TRAIN.FRAME_LEVEL_LOC = False

__C.TRAIN.TEMP_LR = 1.0
__C.TRAIN.ALPHA = 1
__C.TRAIN.TEMP = 1.0
__C.TRAIN.NUM_SEQ = 1
__C.TRAIN.NUM_FRAMES = 100
__C.TRAIN.TEMPLATE_GAP = 100
__C.TRAIN.START_MEM_LEN = 1000
__C.TRAIN.STORE_TH = 0.0001
__C.TRAIN.ITER_PER_EPISODE = 10
__C.TRAIN.SIG_EPS = 0.01
__C.TRAIN.PROB_CLAMP_TH = 0.0




__C.TRAIN.OPTIM = 'sgd'

# Anchor Target
# Positive anchor threshold
__C.TRAIN.THR_HIGH = 0.6

# Negative anchor threshold
__C.TRAIN.THR_LOW = 0.3

# Number of negative
__C.TRAIN.NEG_NUM = 16

# Number of positive
__C.TRAIN.POS_NUM = 16

# Number of anchors per images
__C.TRAIN.TOTAL_NUM = 64

__C.TRAIN.HEM = False
__C.TRAIN.HEM_RATIO = 0.5


__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = local.SNAPSHOT_DIR

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 0 #1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 1.2

__C.TRAIN.MASK_WEIGHT = 0.4

__C.TRAIN.REFINE_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.FOCAL_LOSS = False

##### SiamAttn #####
__C.TRAIN.ROIPOOL_OUTSIZE = 16

__C.TRAIN.ROI_PER_IMG = 16

__C.TRAIN.FG_FRACTION = 1

__C.TRAIN.FG_THRESH = 0.5

__C.TRAIN.BG_THRESH_HIGH = 0.5

__C.TRAIN.BG_THRESH_LOW = 0.0

__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True

__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)

__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# To extend the number of frames with duplicated images (ex. COCO)
__C.DATASET.SEQUENCE = CN()
__C.DATASET.SEQUENCE.SHIFT = 0.1
__C.DATASET.SEQUENCE.SCALE = 0.18
__C.DATASET.SEQUENCE.BLUR = 0.0
__C.DATASET.SEQUENCE.COLOR = 1.0

#  Frame sampling augmentation
__C.DATASET.SEQUENCE.SAMPLE_MODE = 'sequential'
__C.DATASET.SEQUENCE.SAMPLE_PROB = 1.0
__C.DATASET.SEQUENCE.MAX_INTERVAL = 1

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'DET', 'YOUTUBEBB', 'LASOT', 'LASOT_ORIG', 'TRACKINGNET', 'TRACKINGNET_ORIG',
                     'GOT10K', 'GOT_ORIG', 'COCO', 'COCO_ORIG', 'YTVOS', 'YTVOS_ORIG')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = os.path.join(local.VID_PATH, 'crop511')
__C.DATASET.VID.ANNO = os.path.join(local.VID_PATH, 'train.json')
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = local.COCO_PATH #'training_dataset/coco/crop511'
__C.DATASET.COCO.ANNO = os.path.join(local.COCO_PATH,  'train_coco2017.json') #'train2017.json') #'training_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1

__C.DATASET.COCO_ORIG = CN()
__C.DATASET.COCO_ORIG.ROOT = local.COCO_ORIG_PATH
__C.DATASET.COCO_ORIG.ANNO = os.path.join(local.COCO_ORIG_PATH, 'original_train2017.json')
__C.DATASET.COCO_ORIG.FRAME_RANGE = 1
__C.DATASET.COCO_ORIG.NUM_USE = -1

__C.DATASET.YTVOS = CN()
__C.DATASET.YTVOS.ROOT = local.YTVOS_PATH
__C.DATASET.YTVOS.ANNO = os.path.join(local.YTVOS_PATH, 'train.json')
__C.DATASET.YTVOS.FRAME_RANGE = 20
__C.DATASET.YTVOS.NUM_USE = 150000

__C.DATASET.YTVOS_ORIG = CN()
__C.DATASET.YTVOS_ORIG.ROOT = local.YTVOS_ORIG_PATH
__C.DATASET.YTVOS_ORIG.ANNO = os.path.join(local.YTVOS_ORIG_PATH, 'original_train.json')
__C.DATASET.YTVOS_ORIG.FRAME_RANGE = 30
__C.DATASET.YTVOS_ORIG.NUM_USE = 150000

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = os.path.join(local.DET_PATH, 'crop511')
__C.DATASET.DET.ANNO = os.path.join(local.DET_PATH, 'train.json')
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1

__C.DATASET.LASOT = CN()
__C.DATASET.LASOT.ROOT = os.path.join(local.LASOT_PATH, 'train')
__C.DATASET.LASOT.ANNO = os.path.join(local.LASOT_PATH, 'train.json')
__C.DATASET.LASOT.FRAME_RANGE = 100 # max_gap = 200 for template & search
__C.DATASET.LASOT.NUM_USE = -1

__C.DATASET.LASOT_ORIG = CN()
__C.DATASET.LASOT_ORIG.ROOT = local.LASOT_ORIG_PATH
__C.DATASET.LASOT_ORIG.ANNO = os.path.join(local.LASOT_ORIG_PATH, 'original_train.json')
__C.DATASET.LASOT_ORIG.FRAME_RANGE = 100
__C.DATASET.LASOT_ORIG.NUM_USE = -1

__C.DATASET.TRACKINGNET = CN()
__C.DATASET.TRACKINGNET.ROOT = local.TRACKINGNET_PATH
__C.DATASET.TRACKINGNET.ANNO = os.path.join(local.TRACKINGNET_PATH, 'TRAIN_0to3.json')
__C.DATASET.TRACKINGNET.FRAME_RANGE = 100
__C.DATASET.TRACKINGNET.NUM_USE = -1

__C.DATASET.TRACKINGNET_6to11 = CN()
__C.DATASET.TRACKINGNET_6to11.ROOT = local.TRACKINGNET_PATH
__C.DATASET.TRACKINGNET_6to11.ANNO = os.path.join(local.TRACKINGNET_PATH, 'TRAIN_6to11.json')
__C.DATASET.TRACKINGNET_6to11.FRAME_RANGE = 100
__C.DATASET.TRACKINGNET_6to11.NUM_USE = -1

__C.DATASET.TRACKINGNET_ORIG = CN()
__C.DATASET.TRACKINGNET_ORIG.ROOT = local.TRACKINGNET_ORIG_PATH
__C.DATASET.TRACKINGNET_ORIG.ANNO = os.path.join(local.TRACKINGNET_ORIG_PATH, 'TRAIN_6to11_original.json')
__C.DATASET.TRACKINGNET_ORIG.FRAME_RANGE = 100
__C.DATASET.TRACKINGNET_ORIG.NUM_USE = -1

__C.DATASET.TRACKINGNET_ALL = CN()
__C.DATASET.TRACKINGNET_ALL.ROOT = local.TRACKINGNET_ORIG_PATH
__C.DATASET.TRACKINGNET_ALL.ANNO = os.path.join(local.TRACKINGNET_PATH, 'TRAIN_0to11.json')
__C.DATASET.TRACKINGNET_ALL.FRAME_RANGE = 100
__C.DATASET.TRACKINGNET_ALL.NUM_USE = -1

__C.DATASET.GOT10K = CN()
__C.DATASET.GOT10K.ROOT = local.GOT10K_PATH
__C.DATASET.GOT10K.ANNO = os.path.join(local.GOT10K_PATH, 'train.json')
__C.DATASET.GOT10K.FRAME_RANGE = 100
__C.DATASET.GOT10K.NUM_USE = -1

__C.DATASET.GOT10K_VAL = CN()
__C.DATASET.GOT10K_VAL.ROOT = local.GOT10K_PATH
__C.DATASET.GOT10K_VAL.ANNO = os.path.join(local.GOT10K_PATH, 'val.json')
__C.DATASET.GOT10K_VAL.FRAME_RANGE = 100
__C.DATASET.GOT10K_VAL.NUM_USE = -1

__C.DATASET.GOT10K_ORIG = CN()
__C.DATASET.GOT10K_ORIG.ROOT = local.GOT10K_ORIG_PATH
__C.DATASET.GOT10K_ORIG.ANNO = os.path.join(local.GOT10K_ORIG_PATH, 'original_train.json')
__C.DATASET.GOT10K_ORIG.FRAME_RANGE = 100
__C.DATASET.GOT10K_ORIG.NUM_USE = -1

__C.DATASET.OTB = CN()
__C.DATASET.OTB.ROOT = local.OTB_PATH

__C.DATASET.VIDEOS_PER_EPOCH = 600000

__C.DATASET.PERTURB = CN()
__C.DATASET.PERTURB.PERTURB = False
__C.DATASET.PERTURB.MEAN = 0.0
__C.DATASET.PERTURB.VAR = 3.0
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Hyperpixel
__C.BACKBONE.HPF = False
__C.BACKBONE.HPF_KWARGS = CN(new_allowed=True)

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# attention options
# ------------------------------------------------------------------------ #
__C.ATTENTION = CN()
__C.ATTENTION.TYPE = 'ChannelAttention'
__C.ATTENTION.LR = 1.0
__C.ATTENTION.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Layer selection options
# ------------------------------------------------------------------------ #
__C.LAYER_SELECTION = CN()
__C.LAYER_SELECTION.TYPE = 'Static'
__C.LAYER_SELECTION.LR = 1.0
__C.LAYER_SELECTION.KWARGS = CN(new_allowed=True)
__C.LAYER_SELECTION.USE_LAYERS = [3, 9, 12]

__C.LAYER_SELECTION.REG = 'no'
__C.LAYER_SELECTION.TARGET_RATE = 0.0
__C.LAYER_SELECTION.REG_WEIGHT = 1.0

# ------------------------------------------------------------------------ #
# Dynamic layer gating options
# ------------------------------------------------------------------------ #
__C.DYNAMIC = CN()

# Dynamic layer gating
__C.DYNAMIC.DYNAMIC = False
__C.DYNAMIC.CHANNELS = []
__C.DYNAMIC.HIDDEN_CHANNELS = []
__C.DYNAMIC.COMBINED = True
__C.DYNAMIC.LR = 1.0
__C.DYNAMIC.COMBINED_LR = 1.0
__C.DYNAMIC.REGULARIZE = False
__C.DYNAMIC.REG_WEIGHT = 1.0
__C.DYNAMIC.TARGET_RATE = 0.0
__C.DYNAMIC.PER_LAYER = False
__C.DYNAMIC.PER_BATCH = False
__C.DYNAMIC.USE_ALL = False
__C.DYNAMIC.gumbel_hidden_size = 32
__C.DYNAMIC.FUSE_METHOD = 'sum'
__C.DYNAMIC.SOFT = False
__C.DYNAMIC.USE_BIAS = False
__C.DYNAMIC.temperature = 1.
__C.DYNAMIC.USE_AUX_LOSS = False
__C.DYNAMIC.TYPE = 'Hard'
__C.DYNAMIC.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

__C.ADJUST.LR = 1.0


# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.RPN = CN()

# RPN type
__C.RPN.TYPE = 'MultiRPN'

__C.RPN.KWARGS = CN(new_allowed=True)

__C.RPN.LR = 1.0
__C.RPN.TRAIN_LOC = True

__C.RPN.FINAL_LR = 1.0

# ------------------------------------------------------------------------ #
# mask options
# ------------------------------------------------------------------------ #
__C.MASK = CN()

# Whether to use mask generate segmentation
__C.MASK.MASK = False

# Mask type
__C.MASK.TYPE = "MaskCorr"

__C.MASK.KWARGS = CN(new_allowed=True)

__C.MASK.MASK_OUTSIZE = 64

# attention layer
__C.ENHANCE = CN()

__C.ENHANCE.ENHANCE = False

__C.REFINE = CN()

# Mask refine
__C.REFINE.REFINE = False

# Refine type
__C.REFINE.TYPE = "Refine"

# ------------------------------------------------------------------------ #
# Anchor options
# ------------------------------------------------------------------------ #
__C.ANCHOR = CN()

# Anchor stride
__C.ANCHOR.STRIDE = 8

# Anchor ratios
__C.ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]

# Anchor scales
__C.ANCHOR.SCALES = [8]

# Anchor number
__C.ANCHOR.ANCHOR_NUM = len(__C.ANCHOR.RATIOS) * len(__C.ANCHOR.SCALES)


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamRPNTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# Long term lost search size
__C.TRACK.LOST_INSTANCE_SIZE = 831

# Long term confidence low
__C.TRACK.CONFIDENCE_LOW = 0.85

# Long term confidence high
__C.TRACK.CONFIDENCE_HIGH = 0.998

# Mask threshold
__C.TRACK.MASK_THERSHOLD = 0.30

# Mask output size
__C.TRACK.MASK_OUTPUT_SIZE = 127

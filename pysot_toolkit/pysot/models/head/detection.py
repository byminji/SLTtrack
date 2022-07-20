import torch.nn as nn
from pysot.models.init_weight import xavier_fill, gauss_fill
from pysot.core.config import cfg
from pysot.models.head.dcn.deform_pool import DeformRoIPoolingPack, ModulatedDeformRoIPoolingPack
import torchvision.ops as ops

class FCx2DetHead(nn.Module):

    def __init__(self, pooling_func, in_channels, fc_channels=512):
        super(FCx2DetHead, self).__init__()
        self.in_chns = in_channels
        self.tail_det = nn.Sequential(
            nn.Linear(in_channels, fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(fc_channels, fc_channels),
            nn.ReLU(inplace=True)
        )
        self.tail_det_box = nn.Linear(fc_channels, 4)
        self.pooling_func = pooling_func

        self.roi_pool_b = DeformRoIPoolingPack(spatial_scale=1 / (cfg.TRAIN.SEARCH_SIZE / 25),
                                               out_size=cfg.TRAIN.ROIPOOL_OUTSIZE // 4,
                                               out_channels=256,
                                               no_trans=False,
                                               trans_std=0.1)
        '''
        self.roi_pool_b = ModulatedDeformRoIPoolingPack(spatial_scale=1 / (cfg.TRAIN.SEARCH_SIZE / 25),
                                               out_size=cfg.TRAIN.ROIPOOL_OUTSIZE // 4,
                                               out_channels=256,
                                               no_trans=False,
                                               trans_std=0.1,
                                               deform_fc_channels=512)
        
        self.roi_pool_b = ops.RoIAlign(output_size=(cfg.TRAIN.ROIPOOL_OUTSIZE // 4, cfg.TRAIN.ROIPOOL_OUTSIZE // 4),
                                     spatial_scale=1 / (cfg.TRAIN.SEARCH_SIZE / 25),
                                     sampling_ratio=-1)
        '''

        # init parameters
        xavier_fill(self.tail_det)
        gauss_fill(self.tail_det_box, std=0.001)

    def forward(self, x, roi_list):

        x = self.roi_pool_b(x, roi_list)
        x = x.view(-1, self.in_chns)
        x = self.tail_det(x)
        bbox_pred = self.tail_det_box(x)
        return bbox_pred

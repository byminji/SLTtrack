# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siamattn_tracker import SiamAttnTracker
from pysot.tracker.slt_siamrpn_tracker import SltSiamRPNTracker
from pysot.tracker.slt_siamattn_tracker import SltSiamAttnTracker


TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamAttnTracker': SiamAttnTracker,
          'SltSiamRPNTracker': SltSiamRPNTracker,
          'SltSiamAttnTracker': SltSiamAttnTracker
}


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)

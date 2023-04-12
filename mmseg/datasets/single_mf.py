# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .single_modal import SingleCustomDataset


@DATASETS.register_module()
class SingleMFDataset(SingleCustomDataset):
    """MF dataset.

    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '.png' for MF dataset.
    """

    CLASSES = ('unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop',
               'guardrail', 'color_cone', 'bump')

    PALETTE = [[0, 0, 0], [64, 0, 128], [64, 64, 0], [0, 128, 192],
               [0, 0, 192], [128, 128, 0], [64, 64, 128], [192, 128, 128],
               [192, 64, 0]]

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs):
        super(SingleMFDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

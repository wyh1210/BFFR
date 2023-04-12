# Copyright (c) OpenMMLab. All rights reserved.
from asyncore import file_dispatcher
from pickletools import uint8
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
import numpy as np
import cv2
import math
import numpy
# import scipy.signal
# import scipy.ndimage

@LOSSES.register_module()
class reg_loss(nn.Module):

    def __init__(self):
        super(reg_loss, self).__init__()
        self._loss_name = 'reg_loss'
        self.reg = REG()

    def forward(self, image_vis_ycrcb, image_ir, img_fusion, bin_img=None):
        reg = self.reg(image_ir, image_vis_ycrcb, img_fusion)
        reg_loss = 1./reg
        return reg_loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)


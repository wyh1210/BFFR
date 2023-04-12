# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
# print(LOSSES)
# exit()
@LOSSES.register_module()
class avg_loss(nn.Module):
    def __init__(self):
        super(avg_loss, self).__init__()
        self._loss_name = 'avg_loss'

    def forward(self, image_vis_ycrcb, image_ir, img_fusion):

        image_y=image_vis_ycrcb[:,:1,:,:]

        x_in_avg=(image_y + image_ir)/2
        loss_in=F.l1_loss(x_in_avg,img_fusion)
        loss_total=loss_in
        return loss_total

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
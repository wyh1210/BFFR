# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fuser_decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class FuserformerHead(BaseDecodeHead):
    """The all mlp Head of fuserformer.


    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv1 = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv2 = ConvModule(
            in_channels=self.channels,
            out_channels=96,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        # self.fusion_conv3 = ConvModule(
        #     in_channels=128,
        #     out_channels=96,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg)

    def forward(self, inputs, x1, x2):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        # for i in outs:
        #     print(i.shape)
        # print(torch.cat(outs, dim=1))
        # exit()
        out = self.fusion_conv1(torch.cat(outs, dim=1))
        out = self.fusion_conv2(out)
        # out = self.fusion_conv3(out)
        out = resize(input=out,
                     size=x1.shape[2:],
                     mode=self.interpolate_mode,
                     align_corners=self.align_corners)
        out = self.fuse(out, x1, x2)
        # out = self.fuse(out)
        # print(out.shape)

        return out

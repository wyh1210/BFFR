# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fuser_decode_head_v2 import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class FuserformerHead_v2(BaseDecodeHead):
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

        self.convs_vis = nn.ModuleList()
        for i in range(num_inputs):
            self.convs_vis.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.convs_ir = nn.ModuleList()
        for i in range(num_inputs):
            self.convs_ir.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv1_vis = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv2_vis = ConvModule(
            in_channels=self.channels,
            out_channels=1,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.fusion_conv1_ir = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv2_ir = ConvModule(
            in_channels=self.channels,
            out_channels=1,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, seg1, seg2, x1, x2):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        # vis
        inputs1 = self._transform_inputs(seg1)
        outs1 = []
        for idx in range(len(inputs1)):
            x = inputs1[idx]
            conv = self.convs_vis[idx]
            outs1.append(
                resize(
                    input=conv(x),
                    size=inputs1[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        out1 = self.fusion_conv1_vis(torch.cat(outs1, dim=1))
        out1 = self.fusion_conv2_vis(out1)
        out1 = resize(input=out1,
                     size=x1.shape[2:],
                     mode=self.interpolate_mode,
                     align_corners=self.align_corners)
        # ir
        inputs2 = self._transform_inputs(seg2)
        outs2 = []
        for idx in range(len(inputs2)):
            x = inputs2[idx]
            conv = self.convs_ir[idx]
            outs2.append(
                resize(
                    input=conv(x),
                    size=inputs2[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        out2 = self.fusion_conv1_ir(torch.cat(outs2, dim=1))
        out2 = self.fusion_conv2_ir(out2)
        out2 = resize(input=out2,
                     size=x1.shape[2:],
                     mode=self.interpolate_mode,
                     align_corners=self.align_corners)

        out = self.fuse(out1, out2, x1, x2)

        return out

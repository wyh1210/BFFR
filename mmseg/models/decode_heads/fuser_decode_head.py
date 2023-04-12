# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
import torch.nn.functional as F

class ConvBnRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.relu(self.conv(x))
class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

# class ChannelWeights(nn.Module):
#     def __init__(self, dim, reduction=1):
#         super(ChannelWeights, self).__init__()
#         self.dim = dim
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.dim * 2, self.dim * 2 // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.dim * 2 // reduction, self.dim),
#             nn.Sigmoid())

#     def forward(self, x):
#         B, _, H, W = x.shape
#         avg = self.avg_pool(x).view(B, self.dim)
#         glo = self.max_pool(x).view(B, self.dim)
#         y = torch.cat((avg, glo), dim=1)  # B 4C
#         y = self.mlp(y).view(B, self.dim, 1)
#         channel_weights = y.reshape(B, self.dim, 1, 1)  # 2 B C 1 1
#         return channel_weights


# class SpatialWeights(nn.Module):
#     def __init__(self, dim, reduction=1):
#         super(SpatialWeights, self).__init__()
#         self.dim = dim
#         self.mlp = nn.Sequential(
#             nn.Conv2d(self.dim, self.dim // reduction, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.dim // reduction, 1, kernel_size=1),
#             nn.Sigmoid())

#     def forward(self, x):
#         B, _, H, W = x.shape
#         spatial_weights = self.mlp(x).reshape(B, 1, H, W)  # 2 B 1 H W
#         return spatial_weights
        
class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 out_channels,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=None):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.out_channels = out_channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = ConvBnLeakyRelu2d(96, 96, kernel_size=1, padding=0)
        # self.conv_fea = ConvBnLeakyRelu2d(96, 96, kernel_size=1, padding=0)
        self.conv_fuse1 = ConvBnLeakyRelu2d(96, 64, kernel_size=1, padding=0)
        self.conv_fuse2 = ConvBnLeakyRelu2d(64, 32, kernel_size=1, padding=0)
        self.conv_fuse3 = ConvBnTanh2d(32, 1, kernel_size=1, padding=0)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass


    def fuse(self, seg_fea, x1, x2):
        img_fea = torch.cat([x1, x2], dim=1)
        # img_fea = self.conv_fea(img_fea)
        # feat = torch.cat([seg_fea, img_fea], dim=1)
        # """Pixel fuse."""
        if self.dropout is not None:
            seg_fea = self.dropout(seg_fea)
        # img_fea = self.conv_low(img_fea)
        w = self.conv_seg(seg_fea)
        out = w*img_fea + img_fea
        out = self.conv_fuse1(out)
        out = self.conv_fuse2(out)
        out = self.conv_fuse3(out)
        return out


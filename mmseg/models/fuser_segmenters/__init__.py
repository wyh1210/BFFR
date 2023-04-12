# Copyright (c) OpenMMLab. All rights reserved.
from .dual_base import DualBaseSegmenter
from .transfuser_segformer import FuserSegmenter

__all__ = ['DualBaseSegmenter', 'FuserSegmenter']

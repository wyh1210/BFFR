# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder
from .dual_encoder_decoder import DualEncoderDecoder
from .dual_encoder_decoder_fusion import DualEncoderDecoderFusion
from .dual_encoder_decoder_fusion_inbackbone import DualEncoderDecoderFusionInbackbone
# from .dual_encoder_decoder_fusion_inbackbone_resnet import DualEncoderDecoderFusionInbackboneResnet

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', 'DualEncoderDecoder', 'DualEncoderDecoderFusion',
             'DualEncoderDecoderFusionInbackbone']

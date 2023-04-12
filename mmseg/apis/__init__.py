# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test import multi_gpu_test, single_gpu_test
from .test_fusion import single_gpu_test_fusion
from .train import (get_root_logger, init_random_seed, set_random_seed,
                    train_segmentor)
from .train_fuser import train_fusion
from .train_fuser_segmenter import train_fuser_segmenter
from .test_fuser_segmenter import test_fuser_segmenter

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'init_segmentor',
    'inference_segmentor', 'multi_gpu_test', 'single_gpu_test',
    'show_result_pyplot', 'init_random_seed',
    'search_segmentor', 'train_fusion', 'single_gpu_test_fusion',
    'train_fuser_segmenter',
    'test_fuser_segmenter'
]

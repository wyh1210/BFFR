# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict
from turtle import back

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_mit(ckpt):
    backbone_ckpt = OrderedDict()
    decode_head_ckpt = OrderedDict()
    for k, v in ckpt.items():
        # print(k, v)
        # print(type(k))
        if k.startswith('backbone'):

            new_k = k.replace('backbone.', '')
            backbone_ckpt[new_k] = v
        elif k.startswith('decode_head'):
            new_k = k.replace('decode_head.', '')
            decode_head_ckpt[new_k] = v
        # print(new_k)
        # exit()
    return backbone_ckpt, decode_head_ckpt

def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained segformer to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst1', help='save path1')
    parser.add_argument('dst2', help='save path2')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    # print(checkpoint)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # print(state_dict)
    weight_backbone, weight_head = convert_mit(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst1))
    mmcv.mkdir_or_exist(osp.dirname(args.dst2))
    torch.save(weight_backbone, args.dst1)
    torch.save(weight_head, args.dst2)


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import os

def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test_fusion(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler
    # print(out_dir)
    # exit()
    fusion_out_dir = osp.join(out_dir, 'test_result')
    if not osp.exists(fusion_out_dir):
        os.mkdir(fusion_out_dir)
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            if hasattr(model.module, 'fuser'):
                inputs, kwargs = model.scatter(data, {}, model.device_ids)
                img_fusion =  model.module.fuser(return_loss=False, **inputs[0])
                # img_fusion = model.module.fuser(return_loss=False, **data)
            else:
                # print('22'*50)
                # print(model.device)
                # exit()
                # print(data)
                # exit()
                # input()
                inputs, kwargs = model.scatter(data, {}, model.device_ids)
                img_fusion =  model.module(return_loss=False, **inputs[0])
                # input()
                # img_fusion = model(return_loss=False, **data)

            ones = torch.ones_like(img_fusion)
            zeros = torch.zeros_like(img_fusion)
            img_fusion = torch.where(img_fusion > ones, ones, img_fusion)
            img_fusion = torch.where(img_fusion < zeros, zeros, img_fusion)


            # print(img_rgb.shape, img_fusion.shape)
            # exit()
            # img_fusion = img_rgb

            img_fusion = img_fusion.cpu().numpy()
            
            img_fusion = img_fusion.transpose((0, 2, 3, 1))
            img_fusion = (img_fusion - np.min(img_fusion)) / (
                np.max(img_fusion) - np.min(img_fusion)
            )
            img_fusion = np.uint8(255.0 * img_fusion)
        if out_dir:
            # print(result[0].shape)
            # print(len(result))
            # print(batch_indices)
            # exit()
            img_metas = data['img_metas'][0].data[0]
            
            for img_meta in img_metas:
                out_file = osp.join(fusion_out_dir, img_meta['ori_filename'])
                # print(out_file)
                # exit()
                # print(img_fusion)
                # print(img_fusion.shape)
                # print(out_file)
                # exit()
                mmcv.imwrite(img_fusion[0], out_file)

        # results.extend(result)

        batch_size = len(img_fusion)
        for _ in range(batch_size):
            prog_bar.update()

    # return results



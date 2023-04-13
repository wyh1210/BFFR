# Introduction
This is the implementation of the paper [Breaking Free from Fusion Rule: A Fully Semantic-driven Infrared and Visible Image Fusion
](https://arxiv.org/abs/2211.12286).
The main contribution is that we prove the feasibility of discarding hand-crafted fusion rules and only let the semantic segmentation task determine the fusion effect.

Here is a comparison with two typical semantic-driven fusion methods, the results show that ours outperform other semantic-driven methods by discarding fusion rules, and liberate the functionality of the semantic task.

![a](https://github.com/wyh1210/BFFR/blob/master/figs/comparison.png)

We remove some semantic classes during the training process, results show that the semantic information is enough to determine how to fuse.

![b](https://github.com/wyh1210/BFFR/blob/master/figs/remove_class.png)

# Requirements
Install mmsegmentation package as following:
```
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
cd mmseg_bffr
pip install -e .
```
The `cuda` and `torch` version can be arbitrary, but you have to install `mmcv-full<=1.5.0` to meet the requirements of the `./mmseg` module in this repository.

# Datasets
You can download the [MFNet dataset](https://drive.google.com/file/d/1VxTVJ0O72i-dcmfwUThPCDoYwVKIuyri/view?usp=sharing), and put it in ./data/ . We also provided [TNO](https://drive.google.com/file/d/1gWGFUSbhGvXOKlfUNxi5pkmSlrUghpuy/view?usp=share_link) and [RoadScene](https://drive.google.com/file/d/1eRRqrlRkKSHK4M0toPbz292kKUCKB1Fw/view?usp=share_link) testing datasets, too.

# Test with the given model
Download the given model [here](https://drive.google.com/file/d/1BhkaEmb9AipI2Ib7pjSw1zwWoSy86X00/view?usp=share_link). Put it in ./weights/.
This project doesn't support distributed training. You'd better specify an idle GPU card like this: `export CUDA_VISIBLE_DEVICES=X`, `X` denotes the sequence number of the idle card.

- Test fusion effect
```
python tools/test_fusion.py configs/fuser_segmenters/transfuser_segformer_mit-b0_preavgfuser6.py weights/bffr.pth --gpu-id 0
```
* Calculate per-class IoU
```
python tools/test_fuser_segmenter.py configs/fuser_segmenters/transfuser_segformer_mit-b0_preavgfuser6.py weights/bffr.pth --gpu-id 0 --eval mIoU
```
+ Save the segmentation results in color
```
python tools/test_fuser_segmenter.py configs/fuser_segmenters/transfuser_segformer_mit-b0_preavgfuser6.py weights/bffr.pth --gpu-id 0 --show-dir ./seg_result
```

# Train from scratch
Download the pretrained backbone [mit_b0.pth](https://drive.google.com/file/d/1zhcQCl7-lIQh0sTGorhDJDo5kv9fBlGs/view?usp=sharing), put it in `./pretrain/`.
## Step 1: Pretrain the image fusion network (warm-start phase).
    python tools/train_fuser.py configs/fuser/transfuser6_avg.py --gpu-id 0
Put the obtained `work_dirs/transfuser6_avg/iter_20000.pth` into `./pretrain/`  and rename it as `avg_fuser.pth`.
In fact, you can name it whatever you like. It's ok as long as you confirm that the 10th line in `./configs/fuser/transfuser_segformer_mit-b0_preavgfuser6.py`, the `checkpoint='pretrain/XXX.pth` can locate to the path of the weight dict.

By the way, `avg_fuser.pth` can generate average fusion results. If interested, You can also test it by running: `python tools/test_fusion.py configs/fuser/transfuser6_avg.py pretrain/avg_20000.pth --gpu-id 0`. Average fusion is not the goal, but it can provide a good initialization for the next step.

## Step 2: Jointly train the fusion network and the segformer_b0 (semantic training phase).
    python tools/train_fuser_segmenter.py configs/fuser_segmenters/transfuser_segformer_mit-b0_preavgfuser6.py --gpu-id 0
You will get some weight dict in `./work_dirs/transfuser_segformer_mit-b0_preavgfuser6/`, you can test them in the way mentioned above.

# Citation
If this is helpful in your research, please cite the [paper](https://arxiv.org/abs/2211.12286):

@misc{https://doi.org/10.48550/arxiv.2211.12286,
  doi = {10.48550/ARXIV.2211.12286},
  
  url = {https://arxiv.org/abs/2211.12286},
  
  author = {Wu, Yuhui and Liu, Zhu and Liu, Jinyuan and Fan, Xin and Liu, Risheng},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Breaking Free from Fusion Rule: A Fully Semantic-driven Infrared and Visible Image Fusion},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}


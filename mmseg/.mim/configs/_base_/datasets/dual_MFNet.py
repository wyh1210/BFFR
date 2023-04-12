# dataset settings
dataset_type = 'DualMFDataset'
data_root = 'data/MFNet_test/'
img_norm_cfg = dict(
    mean1=[123.675, 116.28, 103.53],
    mean2=[114.478, 114.478, 114.478],
    std1=[58.395, 57.12, 57.375], 
    std2=[57.63, 57.63, 57.63], 
    to_rgb=True)
crop_size = (240, 320)
train_pipeline = [
    dict(type='DualLoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(640, 480), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='DualLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        # img_scale=(640, 480),
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='DualResize', keep_ratio=True),
            dict(type='DualRandomFlip'),
            dict(type='DualNormalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img1', 'img2']),
            dict(type='DualCollect', keys=['img1', 'img2']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        model1_dir='vis',
        model2_dir='ir',
        ann_dir='labels',
        pipeline=test_pipeline,
        split='test.txt'),
    # train_val=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     model1_dir='vis',
    #     model2_dir='ir',
    #     ann_dir='labels',
    #     pipeline=train_pipeline,
    #     split='train_val.txt'),
    # visual=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     model1_dir='vis',
    #     model2_dir='ir',
    #     ann_dir='labels',
    #     pipeline=test_pipeline,
    #     split='visual.txt'),
    fusion=dict(
        type=dataset_type,
        data_root=data_root,
        model1_dir='vis',
        model2_dir='ir',
        ann_dir='labels',
        pipeline=test_pipeline,
        split='fusion.txt'))

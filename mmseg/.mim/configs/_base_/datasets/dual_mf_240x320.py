_base_ = './dual_mf.py'
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
    dict(type='DualResize', img_scale=(640, 480), ratio_range=(0.5, 2.0)),
    dict(type='DualRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='DualRandomFlip', prob=0.5),
    dict(type='DualPhotoMetricDistortion'),
    dict(type='DualNormalize', **img_norm_cfg),
    dict(type='DualPad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DualDefaultFormatBundle'),
    dict(type='DualCollect', keys=['img1', 'img2', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='DualLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
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
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    train_val=dict(pipeline=train_pipeline),
    visual=dict(pipeline=test_pipeline),
    fusion=dict(pipeline=test_pipeline))

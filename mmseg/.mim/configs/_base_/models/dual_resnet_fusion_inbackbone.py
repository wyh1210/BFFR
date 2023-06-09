# model settings

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DualEncoderDecoderFusionInbackbone',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c_mlp',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    frozen=None,
    fusion=dict(type='AddsInbackbone'),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

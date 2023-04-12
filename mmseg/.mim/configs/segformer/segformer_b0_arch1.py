_base_ = [
    '../_base_/models/dual_segformer_mit-b0_fusion_inbackbone.py',
    '../_base_/datasets/dual_mf_240x320.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k_share.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b0.pth')),
    fusion = dict(type='Arch1', dims=[32, 64, 160, 256], heads=[4, 8, 16, 32]),
    test_cfg=dict(mode='slide', crop_size=(240, 320), stride=(120, 160)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=8, workers_per_gpu=1)

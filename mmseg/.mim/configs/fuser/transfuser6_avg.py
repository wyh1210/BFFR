_base_ = [
    '../_base_/datasets/dual_mf_240x320.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_share.py'
]
# model settings


model = dict(
    type='TransFuser6',
    loss_fusion=dict(type='avg_loss'))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0006,
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
evaluation = dict(interval=2000, metric='mIoU', pre_eval=False)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        split='train.txt'),
    val=dict(
        split='val.txt'),
    test=dict(
        split='fusion_53.txt'),
    visual=dict(
        split='visual.txt'),
    fusion=dict(
        split='fusion_53.txt'))
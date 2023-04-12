_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/dual_mf_240x320.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k_share.py'
]

model = dict(
    type='FuserSegmenter',
    fuser=dict(type='TransFuser6', 
            init_cfg=dict(type='Pretrained', checkpoint='pretrain/avg_20000.pth'), 
            loss_fusion=dict(type='reg_loss'),
            lam=0.001),
    no_fusionloss=False,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b0.pth')),
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

data = dict(samples_per_gpu=4, workers_per_gpu=1,
    fusion=dict(
        split='fusion_abl.txt'),
    # test=dict(
    # split='fusion_53.txt')
    )

evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)

checkpoint_config = dict(by_epoch=False, interval=2000, create_symlink=False)

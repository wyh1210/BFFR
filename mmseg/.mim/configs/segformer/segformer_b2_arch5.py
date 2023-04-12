_base_ = ['./segformer_b0_arch1.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b2.pth'),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    fusion = dict(type='Arch5', dims=[64, 128, 320, 512], heads=[8, 16, 32, 64]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

data = dict(samples_per_gpu=4, workers_per_gpu=1)

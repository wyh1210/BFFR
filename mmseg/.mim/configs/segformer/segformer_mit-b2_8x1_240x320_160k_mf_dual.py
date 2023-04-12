_base_ = ['./segformer_mit-b0_8x1_240x320_160k_mf_dual.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b2.pth'),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512, 64, 128, 320, 512]))

data = dict(samples_per_gpu=4, workers_per_gpu=1)

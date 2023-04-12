_base_ = ['./segformer_mit-b0_8x1_480x640_160k_mf_rgb.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b4.pth'),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

data = dict(samples_per_gpu=8, workers_per_gpu=1)


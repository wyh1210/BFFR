_base_ = ['./segformer_mit-b0_8x1_240x320_160k_mf_dual_add_inbackbone.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b3.pth'),
        embed_dims=64,
        num_layers=[3, 4, 18, 3]),
    fusion=dict(type='AddsInbackbone'),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
data = dict(samples_per_gpu=8, workers_per_gpu=1)

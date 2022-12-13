_base_ = [
    '../_base_/models/simclr.py',                  # model
    '../_base_/datasets/imagenet_mocov2.py',       # data TODO PASCAL VOC
    '../_base_/schedules/sgd_coslr-200e_in1k.py',  # training schedule
    '../_base_/default_runtime.py',                # runtime setting
]

model = dict(
    type='SimClr',  # Algorithm name
    queue_len=65536,  # Number of negative keys maintained in the queue
    feat_dim=128,  # Dimension of compact feature vectors, equal to the out_channels of the neck
    momentum=0.999,  # Momentum coefficient for the momentum-updated encoder
    backbone=dict(
        type='ResNet',  # Backbone name
        depth=50,  # Depth of backbone, ResNet has options of 18, 34, 50, 101, 152
        in_channels=3,  # The channel number of the input images
        out_indices=[4],  # The output index of the output feature maps, 0 for conv-1, x for stage-x
        norm_cfg=dict(type='BN')),  # Dictionary to construct and config norm layer
    neck=dict(
        type='MoCoV2Neck',  # Neck name
        in_channels=2048,  # Number of input channels
        hid_channels=2048,  # Number of hidden channels
        out_channels=128,  # Number of output channels
        with_avg_pool=True),  # Whether to apply the global average pooling after backbone
    head=dict(
        type='ContrastiveHead',  # Head name, indicates that the MoCo v2 use contrastive loss
        temperature=0.2))  # The temperature hyper-parameter that controls the concentration level of the distribution.

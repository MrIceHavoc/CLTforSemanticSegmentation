_base_ = [
    './mmselfsup/mmselfsup/models/algorithms/simclr.py',                  # model
    './mmselfsup/mmselfsup/datasets/imagenet_mocov2.py',       # data TODO PASCAL VOC
    './mmselfsup/mmselfsup/schedules/sgd_coslr-200e_in1k.py',  # training schedule
    './mmselfsup/mmselfsup/default_runtime.py',                # runtime setting
]

model = dict(
    type='SimCLR',  # Algorithm name
    queue_len=65536,  # Number of negative keys maintained in the queue
    feat_dim=128,  # Dimension of compact feature vectors, equal to the out_channels of the neck
    momentum=0.999,  # Momentum coefficient for the momentum-updated encoder
    backbone=dict(
        type='ResNet',  # Backbone name
        depth=50,  # Depth of backbone, ResNet has options of 18, 34, 50, 101, 152
        in_channels=3,  # The channel number of the input images
        out_indices=[4],  # The output index of the output feature maps, 0 for conv-1, x for stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='MoCoV2Neck',  # Neck name
        in_channels=2048,  # Number of input channels
        hid_channels=2048,  # Number of hidden channels
        out_channels=128,  # Number of output channels
        with_avg_pool=True,  # Whether to apply the global average pooling after backbone
        vit_backbone=True),
    head=dict(
        type='ContrastiveHead',  # Head name, indicates that the MoCo v2 use contrastive loss
        temperature=0.2))  # The temperature hyper-parameter that controls the concentration level of the distribution.

data_source = 'ImageNet'  # data source name
dataset_type = 'MultiViewDataset' # dataset type is related to the pipeline composing
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],  # Mean values used to pre-training the pre-trained backbone models
    std=[0.229, 0.224, 0.225])  # Standard variance used to pre-training the pre-trained backbone models
# The difference between mocov2 and mocov1 is the transforms in the pipeline
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),  # RandomResizedCrop
    dict(
        type='RandomAppliedTrans',  # Random apply ColorJitter augment method with probability 0.8
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),  # RandomGrayscale with probability 0.2
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),  # Random GaussianBlur with probability 0.5
    dict(type='RandomHorizontalFlip'),  # Randomly flip the picture horizontally
]

# prefetch
prefetch = False  # Whether to using prefetch to speed up the pipeline
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU, total 32*8=256
    workers_per_gpu=0,  # Worker to pre-fetch data for each single GPU
    drop_last=True,  # Whether to drop the last batch of data
    train=dict(
        type=dataset_type,  # dataset name
        data_source=dict(
            type=data_source,  # data source name
            data_prefix='data/imagenet/train',  # Dataset root, when ann_file does not exist, the category information is automatically obtained from the root folder
            ann_file='data/imagenet/meta/train.txt',  #  ann_file existes, the category information is obtained from file
        ),
        num_views=[2],  # The number of different views from pipeline
        pipelines=[train_pipeline],  # The train pipeline
        prefetch=prefetch,  # The boolean value
    ))

# optimizer
optimizer = dict(
    type='SGD',  # Optimizer type
    lr=0.03,  # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
    weight_decay=1e-4,  # Momentum parameter
    momentum=0.9)  # Weight decay of SGD
# Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
optimizer_config = dict()  # this config can set grad_clip, coalesce, bucket_size_mb, etc.

# learning policy
# Learning rate scheduler config used to register LrUpdater hook
lr_config = dict(
    policy='CosineAnnealing',  # The policy of scheduler, also support Step, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    min_lr=0.)  # The minimum lr setting in CosineAnnealing

# runtime settings
runner = dict(
    type='EpochBasedRunner',  # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=200) # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`

# checkpoint saving
checkpoint_config = dict(interval=10)  # The save interval is 10

# yapf:disable
log_config = dict(
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook'),  # The Tensorboard logger is also supported
        # dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

# runtime settings
dist_params = dict(backend='nccl') # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # The output level of the log.
load_from = None  # Runner to load ckpt
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.
persistent_workers = False  # The boolean type to set persistent_workers in Dataloader. see detail in the documentation of PyTorch

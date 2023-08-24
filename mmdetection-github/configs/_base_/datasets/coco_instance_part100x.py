# dataset settings
dataset_type = 'CocoDataset_Part100x'
data_root = 'data/part100x/'
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPart100x',with_bbox=True,with_mask=True,poly2mask=False),
    dict(type='ResizePart100x',img_scale=(1280, 1024), multiscale_mode='value',keep_ratio=True),
    dict(type='RandomFlipPart100x', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PadPart100x', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1024),
        flip=False,
        transforms=[
            dict(type='ResizePart100x', keep_ratio=True),
            dict(type='RandomFlipPart100x'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='PadPart100x', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
#data = dict(
 #   samples_per_gpu=2,
 #   workers_per_gpu=2,
 #   train=dict(
 #       type=dataset_type,
 #       ann_file=data_root + 'annotations/instances_train2017.json',
 #       img_prefix=data_root + 'train2017/',
 #       pipeline=train_pipeline),
 #   val=dict(
 #       type=dataset_type,
 #       ann_file=data_root + 'annotations/instances_val2017.json',
 #       img_prefix=data_root + 'val2017/',
 #       pipeline=test_pipeline),
 #   test=dict(
#        type=dataset_type,
#        ann_file=data_root + 'annotations/instances_val2017.json',
#        img_prefix=data_root + 'val2017/',
#        pipeline=test_pipeline))
classes = ('sperm',)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        img_prefix='data/part100x',
        classes=classes,
        ann_file='data/part100x/annotations.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix='data/part100x',
        classes=classes,
        ann_file='data/part100x/annotations.json',
        pipeline=test_pipeline),
     test=dict(
        type=dataset_type,
        img_prefix='data/part100x',
        classes=classes,
        ann_file='data/part100x/annotations.json',
        pipeline=test_pipeline))

evaluation = dict(metric=['bbox', 'segm'])
#evaluation = dict(interval=1)

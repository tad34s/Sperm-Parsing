_base_ = ['./cascade_mask_rcnn_r50_fpn_1x_sperm.py']

model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')))
        
        
dataset_type = 'CocoDataset'
classes = ('sperm',)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPart', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='ResizePart', img_scale=(1200, 900), multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlipPart', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PadPart', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200, 900),
        flip=False,
        transforms=[
            dict(type='ResizePart', keep_ratio=True),
            dict(type='RandomFlipPart'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='PadPart', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
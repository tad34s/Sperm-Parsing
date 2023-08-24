_base_ = './mask_rcnn_r50_fpn_1x_coco-backup.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPart',with_bbox=True,with_mask=True,poly2mask=False),
    dict(type='ResizePart',img_scale=(1200, 900), multiscale_mode='value',keep_ratio=True),
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

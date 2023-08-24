# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = 'mask_rcnn_r101_fpn_1x_coco.py'

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('sperm',)
    #val=dict(
    #    img_prefix='balloon/val/',
    #    classes=classes,
    #    ann_file='balloon/val/annotation_coco.json'),
    #test=dict(
     #   img_prefix='balloon/val/',
     #   classes=classes,
    #    ann_file='balloon/val/annotation_coco.json'))

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

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
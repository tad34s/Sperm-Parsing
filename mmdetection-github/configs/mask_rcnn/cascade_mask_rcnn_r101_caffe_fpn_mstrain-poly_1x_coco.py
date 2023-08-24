_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')))


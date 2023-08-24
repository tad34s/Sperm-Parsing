# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = 'cascade_mask_rcnn_r101_caffe_fpn_mstrain-poly_1x_coco.py'

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        #bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1, num_parsing=3)))

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('sperm',)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])

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
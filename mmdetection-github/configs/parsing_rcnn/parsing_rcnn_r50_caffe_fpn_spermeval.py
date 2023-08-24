# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = 'parsing_rcnn_r50_caffe_fpn_cocoeval.py'

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1, num_parsing=5)))

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('sperm',)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
#embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
#optimizer = dict(
#    _delete_=True,
#    type='AdamW',
#    lr=0.0001,
#    weight_decay=0.05,
#    eps=1e-8,
#    betas=(0.9, 0.999),
#    paramwise_cfg=dict(
#        custom_keys={
#            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
#            'query_embed': embed_multi,
#            'query_feat': embed_multi,
#            'level_embed': embed_multi,
#        },
#        norm_decay_mult=0.0))
#optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
#lr_config = dict(
#    policy='step',
#    gamma=0.1,
#    by_epoch=False,
#    step=[327778, 355092],
#    warmup='linear',
#    warmup_by_epoch=False,
#    warmup_ratio=1.0,  # no warmup
#    warmup_iters=10)
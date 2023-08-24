_base_ = [
    '../_base_/models/mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance_part.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])


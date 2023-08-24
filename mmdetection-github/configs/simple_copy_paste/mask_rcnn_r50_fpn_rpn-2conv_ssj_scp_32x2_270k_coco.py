_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_ori.py',
    # 270k iterations with batch_size 64 is roughly equivalent to 144 epochs
    '../common/ssj_scp_270k_coco_instance.py'
]


# Use MMSyncBN that handles empty tensor in head. It can be changed to
# SyncBN after https://github.com/pytorch/pytorch/issues/36530 is fixed.

model = dict(
    backbone=dict(frozen_stages=-1, norm_eval=False),
    rpn_head=dict(num_convs=2),  # leads to 0.1+ mAP
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256)))

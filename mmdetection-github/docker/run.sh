#!/bin/bash

sudo docker run --gpus all \
           sperm-parsing \
           bash -c "cd /sperm/mmdetection-github && python tools/test.py configs/git_fusionrcnn/cascade_mask_rcnn_r101_caffe_feature_blend_coarse_fine_edge_fpn_1x_spermparsingeval.py epoch_35.pth --eval bbox segm"

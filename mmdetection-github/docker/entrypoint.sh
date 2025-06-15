#!/bin/bash
set -e

# Download model weights if not already present
if [ ! -f "epoch_35.pth" ]; then
  gdown 1bFhdgD3SrSB7gvvRKnx_sX_KjFaf85q_
fi

# Run your test command (uncomment if needed)
# python tools/test.py configs/git_fusionrcnn/cascade_mask_rcnn_r101_caffe_feature_blend_coarse_fine_edge_fpn_1x_spermparsingeval.py epoch_35.pth --eval bbox segm

exec "$@"

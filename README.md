# Cell-Parsing
Official implementation of **CP-Net: Instance-Aware Part Segmentation Network for Biological Cell Parsing**

In this repository, we release the CP-Net code in Pytorch and the proposed Sperm Parsing Dataset

The dataset is in mmdetection-github/data/spermparsing

This network is based on the framework of [mmdetection](https://github.com/open-mmlab/mmdetection)

<hr />

> **Abstract:** *Instance segmentation of biological cells is important in medical image analysis for identifying and segmenting individual cells, and quantitative measurement of subcellular structures requires further cell-level subcellular part segmentation. Subcellular structure measurements are critical for cell phenotyping and quality analysis. For these purposes, instance-aware part segmentation network is first introduced to distinguish individual cells and segment subcellular structures for each detected cell. This approach is demonstrated on human sperm cells since the World Health Organization has established quantitative standards for sperm quality assessment. Specifically, a novel Cell Parsing Net (CP-Net) is proposed for accurate instance-level cell parsing. An attention-based feature fusion module is designed to alleviate contour misalignments for cells with an irregular shape by using instance masks as spatial cues instead of as strict constraints to differentiate various instances. A coarse-to-fine segmentation module is developed to effectively segment tiny subcellular structures within a cell through hierarchical segmentation from whole to part instead of directly segmenting each cell part. Moreover, a sperm parsing dataset is built including 320 annotated sperm images with five semantic subcellular part labels. Extensive experiments on the collected dataset demonstrate that the proposed CP-Net outperforms state-of-the-art instance-aware part segmentation networks.* 
<hr />

## Netword architecture:
<p align="center"><img width="90%" src="mmdetection-github/data/CP-Net.jpg" /></p>

## Sample output:
<p align="center"><img width="90%" src="mmdetection-github/data/result.jpg" /></p>

## Installation
- mmcv.1.6.0
- pytorch.1.12.1

## Results and Models
**On Sperm Parsing Dataset**

|Methods      |  Backbone  | mIoU |Parsing (APp50/APvol/PCP50) | DOWNLOAD |
|-------------|------------|:----:|:--------------------------:| :-------:|
|CP-Net-light |  R-50-FPN  | 70.4 |      59.5/78.3/70.1        |          |
|CP-Net       |  R-101-FPN | 70.6 |      61.5/83.3/72.5        | [GoogleDrive](https://drive.google.com/file/d/1bFhdgD3SrSB7gvvRKnx_sX_KjFaf85q_/view?usp=drive_link)|

## Evaluation
```
python tools/test.py configs/git_fusionrcnn/cascade_mask_rcnn_r101_caffe_feature_blend_coarse_fine_edge_fpn_1x_spermparsingeval.py epoch_35.pth --eval bbox segm
```

# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import ModuleList

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import torch.nn.functional as F
BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit
import pdb

@HEADS.register_module()
class CascadeRoIHeadGit(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_head = None,
                 fusion_head = None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(CascadeRoIHeadGit, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.semantic_head = None
        self.fusion_head = None
        
        if semantic_head is not None:
            #pdb.set_trace()
            self.semantic_head = build_head(semantic_head)
            
            
        if fusion_head is not None:
            self.fusion_head = build_head(fusion_head)
            
        self.semantic_traced_model = torch.jit.load("cascadercnnfusion_tracedsemantic.pt")    
        self.fusion_traced_model = torch.jit.load("cascadercnnfusion_tracedfusion.pt") 
        
        
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        #pdb.set_trace()
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'], )
        return outs

    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        #pdb.set_trace()
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, x, rois):
        """Mask head forward function used in both training and testing."""

        mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred, mask_features, mask_pred2 = self.mask_head(mask_feats)
        #mask_pred = self.mask_traced_model(mask_feats)
        #pdb.set_trace()
        
        mask_results = dict(mask_pred=mask_pred, mask_pred2=mask_pred2)
        return mask_results

    def _mask_forward_train(self,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            img_metas,
                            bbox_feats=None):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(x, pos_rois)

        #pdb.set_trace()

        mask_targets, mask_edges = self.mask_head.get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        pad_shape = img_metas[0]['pad_shape']
        parsing_pred = self.mask_head.parsing_results(mask_results['mask_pred'], [res.pos_bboxes for res in sampling_results][0] , pad_shape, 4.0, True)
        
        loss_mask, loss_mask2, loss_mask3, loss_mask4 = self.mask_head.loss(mask_results['mask_pred'],
                                               mask_targets, mask_results['mask_pred2'], mask_edges, pos_labels)

        mask_results.update(loss_mask=loss_mask, parsing_pred=parsing_pred, loss_mask2 = loss_mask2, loss_mask3 = loss_mask3, loss_mask4 = loss_mask4)
        #pdb.set_trace()
        return mask_results

    def _seg_forward_train(self, x, gt_masks, img_metas):
        pad_shape = img_metas[0]['pad_shape']
        seg_results = self._seg_forward(x, True)
        device = seg_results['seg_pred'].device
        seg_targets, gt_edges, seg_instance_targets, gt_instance_edges = self.semantic_head.get_targets(gt_masks, device, self.train_cfg, pad_shape)
        #pdb.set_trace()
        loss_seg, loss_seg2, loss_seg3, loss_seg4, loss_seg5, loss_seg6, loss_seg7, loss_seg8, loss_seg9 = self.semantic_head.loss(seg_results['seg_pred'], seg_targets, seg_results['seg_edges'], gt_edges, seg_results['seg_pred2'], seg_instance_targets, seg_results['seg_edges2'], gt_instance_edges)

        seg_results.update(loss_seg=loss_seg, seg_targets=seg_targets, loss_seg2=loss_seg2, loss_seg3=loss_seg3, loss_seg4=loss_seg4, loss_seg5 = loss_seg5, loss_seg6 = loss_seg6, loss_seg7 = loss_seg7, loss_seg8 = loss_seg8, loss_seg9 = loss_seg9)
        return seg_results
        
    def _seg_forward(self, x, is_train):
        #pdb.set_trace()
        #seg_pred, seg_pred2, seg_features, seg_edges, seg_edges2 = self.semantic_head(x, is_train)
        seg_pred, seg_pred2, seg_features, seg_edges, seg_edges2 = self.semantic_traced_model(x[0], x[1], x[2], x[3], x[4])
        seg_results = dict(seg_pred=seg_pred, seg_pred2=seg_pred2, seg_features=seg_features, seg_edges=seg_edges, seg_edges2=seg_edges2)
        return seg_results
        
    def _fusion_forward_train(self, seg_pred, mask_pred, seg_features, sampling_results, gt_masks, img_metas):
        #pdb.set_trace()
        pad_shape = img_metas[0]['pad_shape']
        fusion_results = self._fusion_forward(seg_pred, mask_pred, seg_features, pad_shape, [res.pos_bboxes for res in sampling_results][0], True)
        device = fusion_results['final_pred'].device
        fusion_targets, fusion_edges, fusion_instance_targets = self.fusion_head.get_targets(gt_masks, device, sampling_results, pad_shape)
        #pdb.set_trace()
        loss_fusion, loss_fusion2, loss_fusion4, loss_fusion5, loss_fusion6, loss_fusion7 = self.fusion_head.loss(fusion_results['final_pred'], fusion_targets, fusion_results['final_edge'], fusion_edges, fusion_results['final_pred2'], fusion_instance_targets)
        
        fusion_results.update(loss_fusion=loss_fusion, fusion_targets = fusion_targets, loss_fusion2=loss_fusion2, loss_fusion4 = loss_fusion4, loss_fusion5 = loss_fusion5, loss_fusion6 = loss_fusion6, loss_fusion7 = loss_fusion7)
        return fusion_results
        
    def _fusion_forward(self, seg_pred, mask_pred, seg_features, pad_shape, boxes, is_Train):
        #pdb.set_trace()
        #if is_Train:
        #    mask_pred = mask_pred.detach()
        #    seg_features = seg_features.detach()
        #    seg_pred = seg_pred.detach()
        #else:
        #    mask_pred = mask_pred
        #    seg_features = seg_features
        #    seg_pred = seg_pred

        #final_pred, final_edge, final_pred2 = self.fusion_head(seg_pred, seg_features, mask_pred, is_Train, boxes)
        final_pred = [] 
        for i in range(mask_pred.shape[0]):
            mask_attn = torch.where(mask_pred[i].unsqueeze(0) == 0, torch.tensor(float(torch.min(mask_pred[i]))).cuda(), mask_pred[i].unsqueeze(0))
            fusion_pred = self.fusion_traced_model(seg_features, mask_attn)
            final_pred.append(fusion_pred)
        final_pred = torch.cat(final_pred,0) 
        fusion_results = dict(final_pred = final_pred, final_edge = None, final_pred2 = None)
        return fusion_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            #pdb.set_trace()
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)
            
            #pdb.set_trace()
            
            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)

                    # Empty proposal.
                    if cls_score.numel() == 0:
                        break

                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)
            if i == self.num_stages - 1:
                # mask head forward and loss
                #pdb.set_trace()
                if self.with_mask:
                    mask_results = self._mask_forward_train(
                        x, sampling_results, gt_masks, rcnn_train_cfg, img_metas,
                        bbox_results['bbox_feats'])
                    #pdb.set_trace()
                    #for name, value in mask_results['loss_mask'].items():
                    #    losses[f's{i}.{name}'] = (
                    #        value * lw if 'loss' in name else value)
                    losses.update(mask_results['loss_mask'])
                    losses.update(mask_results['loss_mask2'])
                    #losses.update(mask_results['loss_mask3'])
                    #losses.update(mask_results['loss_mask4'])
                    
                if self.semantic_head is not None:
                    #pdb.set_trace()
                    seg_results = self._seg_forward_train(x, gt_masks, img_metas)
                    losses.update(seg_results['loss_seg'])
                    losses.update(seg_results['loss_seg2'])
                    #losses.update(seg_results['loss_seg3'])
                    losses.update(seg_results['loss_seg4'])
                    losses.update(seg_results['loss_seg5'])
                    losses.update(seg_results['loss_seg6'])
                    losses.update(seg_results['loss_seg7'])
                    losses.update(seg_results['loss_seg8'])
                    losses.update(seg_results['loss_seg9'])
                    
                if self.fusion_head is not None:
                    #pdb.set_trace()
                    fusion_results = self._fusion_forward_train(seg_results['seg_pred2'], mask_results['parsing_pred'], seg_results['seg_features'], sampling_results, gt_masks, img_metas)
                    losses.update(fusion_results['loss_fusion'])
                    losses.update(fusion_results['loss_fusion2'])
                    losses.update(fusion_results['loss_fusion4'])
                    losses.update(fusion_results['loss_fusion5'])
                    #losses.update(fusion_results['loss_fusion6'])
                    #losses.update(fusion_results['loss_fusion7'])
        #pdb.set_trace()
        #print(losses)
        return losses



    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        #pdb.set_trace()
        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)
            #pdb.set_trace()

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            #pdb.set_trace()
        
        #pdb.set_trace()
        
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                #pdb.set_trace()
                mask_classes = self.mask_head.num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)

                mask_results = self._mask_forward(x, mask_rois)
                mask_pred = mask_results['mask_pred']
                # split batch mask prediction back to each image
                mask_preds = mask_pred.split(num_mask_rois_per_img, 0)

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head.num_classes)])
                    else:
                        #pdb.set_trace()
                        pad_shapes = tuple(meta['pad_shape'] for meta in img_metas)
                        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
                        
                        mask_result = self.mask_head.parsing_results(
                            mask_preds[i], _bboxes[i], pad_shapes[i], 4.0, True)

                        seg_results = self._seg_forward(x, False)
                    
                        fusion_results = self._fusion_forward(seg_results['seg_pred2'], mask_result, seg_results['seg_features'], pad_shapes[i], det_bboxes[i][:, :4], False)
                            
                        fusion_results['final_pred'] = F.interpolate(fusion_results['final_pred'], size=pad_shapes[i][:2], mode='bilinear', align_corners=True)
                        segm_results.append(self.simple_get_targets(fusion_results['final_pred'], self.test_cfg, ori_shapes[i]))
                            
                        
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']
        
        return results

    def simple_get_targets(self, mask_pred, rcnn_test_cfg, ori_shape):
        #pdb.set_trace()
        device = mask_pred.device
        img_h, img_w = ori_shape[:2]
        mask_pred = mask_pred.sigmoid()
        threshold = rcnn_test_cfg.mask_thr_binary
        
        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we need to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT /
                        GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
                    
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
        
        cls_segms = [[] for _ in range(self.mask_head.num_parsing)
                    ]  # BG is not included in num_classes

        for j in range(0, self.mask_head.num_parsing):


            im_mask = torch.zeros(
                N,
                img_h,
                img_w,
                device=device,
                dtype=torch.bool if threshold >= 0 else torch.uint8)

            for inds in chunks:
                masks_chunk = mask_pred[inds,torch.tensor([j]*N)]

                if threshold >= 0:
                    masks_chunk = (masks_chunk > threshold).to(dtype=torch.bool)
                else:
                    # for visualization and debugging
                    masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

                im_mask[(inds, )] = masks_chunk[:, :img_h, :img_w]
            
            #pdb.set_trace()
            
            for i in range(N):
                cls_segms[j].append(im_mask[i].detach().cpu().numpy())
        return cls_segms

    def simple_get_targets2(self, mask_pred, rcnn_test_cfg, ori_shape):
        #pdb.set_trace()
        device = mask_pred.device
        img_h, img_w = ori_shape[:2]
        mask_pred = mask_pred.sigmoid()
        threshold = rcnn_test_cfg.mask_thr_binary
        
        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we need to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT /
                        GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
                    
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
        
        cls_segms = [[] for _ in range(self.mask_head.num_classes)
                    ]  # BG is not included in num_classes

        for j in range(0, self.mask_head.num_classes):


            im_mask = torch.zeros(
                N,
                img_h,
                img_w,
                device=device,
                dtype=torch.bool if threshold >= 0 else torch.uint8)

            for inds in chunks:
                masks_chunk = mask_pred[inds,torch.tensor([j]*N)]

                if threshold >= 0:
                    masks_chunk = (masks_chunk > threshold).to(dtype=torch.bool)
                else:
                    # for visualization and debugging
                    masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

                im_mask[(inds, )] = masks_chunk[:, :img_h, :img_w]
            
            #pdb.set_trace()
            
            for i in range(N):
                cls_segms[j].append(im_mask[i].detach().cpu().numpy())
        return cls_segms

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])

            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)
                    bbox_label = cls_score[:, :-1].argmax(dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                dummy_scale_factor = np.ones(4)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=dummy_scale_factor,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]

    def onnx_export(self, x, proposals, img_metas):

        assert self.with_bbox, 'Bbox head must be implemented.'
        assert proposals.shape[0] == 1, 'Only support one input image ' \
                                        'while in exporting to ONNX'
        # remove the scores
        rois = proposals[..., :-1]
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]
        # Eliminate the batch dimension
        rois = rois.view(-1, 4)

        # add dummy batch index
        rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)

        max_shape = img_metas[0]['img_shape_for_onnx']
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            # Recover the batch dimension
            rois = rois.reshape(batch_size, num_proposals_per_img,
                                rois.size(-1))
            cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                          cls_score.size(-1))
            bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                assert self.bbox_head[i].reg_class_agnostic
                new_rois = self.bbox_head[i].bbox_coder.decode(
                    rois[..., 1:], bbox_pred, max_shape=max_shape)
                rois = new_rois.reshape(-1, new_rois.shape[-1])
                # add dummy batch index
                rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois],
                                 dim=-1)

        cls_score = sum(ms_scores) / float(len(ms_scores))
        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
        rois = rois.reshape(batch_size, num_proposals_per_img, -1)
        det_bboxes, det_labels = self.bbox_head[-1].onnx_export(
            rois, cls_score, bbox_pred, max_shape, cfg=rcnn_test_cfg)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            batch_index = torch.arange(
                det_bboxes.size(0),
                device=det_bboxes.device).float().view(-1, 1, 1).expand(
                    det_bboxes.size(0), det_bboxes.size(1), 1)
            rois = det_bboxes[..., :4]
            mask_rois = torch.cat([batch_index, rois], dim=-1)
            mask_rois = mask_rois.view(-1, 5)
            aug_masks = []
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                mask_pred = mask_results['mask_pred']
                aug_masks.append(mask_pred)
            max_shape = img_metas[0]['img_shape_for_onnx']
            # calculate the mean of masks from several stage
            mask_pred = sum(aug_masks) / len(aug_masks)
            segm_results = self.mask_head[-1].onnx_export(
                mask_pred, rois.reshape(-1, 4), det_labels.reshape(-1),
                self.test_cfg, max_shape)
            segm_results = segm_results.reshape(batch_size,
                                                det_bboxes.shape[1],
                                                max_shape[0], max_shape[1])
            return det_bboxes, det_labels, segm_results

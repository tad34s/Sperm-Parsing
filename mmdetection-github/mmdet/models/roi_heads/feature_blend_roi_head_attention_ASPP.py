# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import numpy as np
import torch.nn.functional as F
BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit
import pdb

@HEADS.register_module()
class FeatureBlendHeadAttentionASPP(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
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
        super(FeatureBlendHeadAttentionASPP, self).__init__(
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
            
            
            
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        #pdb.set_trace()
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
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
        ##self.semantic_head = self.semantic_head.to('cuda:1')
        #for i in range(self.fusion_head.num_convs):
        #    if i == 1:
        #        self.fusion_head.convs[i] = self.fusion_head.convs[i].to('cuda:1')
        #    elif i == 2:
        #        self.fusion_head.convs[i] = self.fusion_head.convs[i].to('cuda:2')
        ##self.fusion_head.convs = self.fusion_head.convs.to('cuda:1')
        #self.fusion_head.upsample = self.fusion_head.upsample.to('cuda:3')
        #self.fusion_head.relu = self.fusion_head.relu.to('cuda:3')
        ##self.fusion_head.lateral_convs = self.fusion_head.lateral_convs.to('cuda:2')
        #self.fusion_head.conv_logits = self.fusion_head.conv_logits.to('cuda:3')
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        #pdb.set_trace()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
            #losses.update(mask_results['loss_mask2'])
            #losses.update(mask_results['loss_mask3'])
        #pdb.set_trace()
        
        if self.semantic_head is not None:
            #pdb.set_trace()
            seg_results = self._seg_forward_train(x, gt_masks, img_metas)
            losses.update(seg_results['loss_seg'])
            losses.update(seg_results['loss_seg2'])
            #losses.update(seg_results['loss_seg3'])
            losses.update(seg_results['loss_seg4'])
            
        if self.fusion_head is not None:
            #pdb.set_trace()
            fusion_results = self._fusion_forward_train(seg_results['seg_pred'], mask_results['parsing_pred'], seg_results['seg_features'], sampling_results, gt_masks, img_metas)
            losses.update(fusion_results['loss_fusion'])
            losses.update(fusion_results['loss_fusion2'])
            #losses.update(fusion_results['loss_fusion3'])
            #losses.update(fusion_results['loss_fusion4'])
        
        #pdb.set_trace()
        print(losses)
        
        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        
        pad_shape = img_metas[0]['pad_shape']
        
        parsing_pred = self.mask_head.parsing_results(mask_results['mask_pred'], [res.pos_bboxes for res in sampling_results][0] , pad_shape, 4.0, True)
        
        #ori_shape = img_metas[0]['ori_shape']
        
        #parsing_pred1 = self.mask_head.parsing_results(mask_results['mask_pred'], [res.pos_bboxes for res in sampling_results] ,ori_shape)
        
        #pdb.set_trace()
        loss_mask= self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)
        
        #mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets, parsing_pred=parsing_pred)#, loss_mask2 = loss_mask2)#, loss_mask3 = loss_mask3)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]
        
        #pdb.set_trace()
        mask_pred, mask_features = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats, mask_features=mask_features)
        return mask_results

    def _seg_forward_train(self, x, gt_masks, img_metas):
        pad_shape = img_metas[0]['pad_shape']
        seg_results = self._seg_forward(x, True)
        device = seg_results['seg_pred'].device
        seg_targets, gt_edges = self.semantic_head.get_targets(gt_masks, device, self.train_cfg, pad_shape)
        #pdb.set_trace()
        loss_seg, loss_seg2, loss_seg3, loss_seg4 = self.semantic_head.loss(seg_results['seg_pred'], seg_targets, seg_results['seg_edges'], gt_edges)

        seg_results.update(loss_seg=loss_seg, seg_targets=seg_targets, loss_seg2=loss_seg2, loss_seg3=loss_seg3, loss_seg4=loss_seg4)
        return seg_results
        
    def _seg_forward(self, x, is_train):
        #pdb.set_trace()
        seg_pred, seg_features, seg_edges = self.semantic_head(x, is_train)
        seg_results = dict(seg_pred=seg_pred, seg_features=seg_features, seg_edges=seg_edges)
        return seg_results
        
    def _fusion_forward_train(self, seg_pred, mask_pred, seg_features, sampling_results, gt_masks, img_metas):
        #pdb.set_trace()
        pad_shape = img_metas[0]['pad_shape']
        fusion_results = self._fusion_forward(seg_pred, mask_pred, seg_features, pad_shape, [res.pos_bboxes for res in sampling_results][0], True)
        device = fusion_results['final_pred'].device
        #device = mask_pred.device
        fusion_targets = self.fusion_head.get_targets(gt_masks, device, sampling_results, pad_shape)
        #pdb.set_trace()
        loss_fusion, loss_fusion2 = self.fusion_head.loss(fusion_results['final_pred'], fusion_targets)
        
        fusion_results.update(loss_fusion=loss_fusion, fusion_targets = fusion_targets, loss_fusion2=loss_fusion2)#, loss_fusion3 = loss_fusion3, loss_fusion4 = loss_fusion4)
        return fusion_results
        
    def _fusion_forward(self, seg_pred, mask_pred, seg_features, pad_shape, boxes, is_Train):
        #pdb.set_trace()
        #seg_pred = seg_pred.sigmoid()
        if is_Train:
            #seg_features.requires_grad=True
            mask_pred = mask_pred.detach()
            #mask_pred.requires_grad = True
            seg_features = seg_features.detach()
        else:
            mask_pred = mask_pred
            seg_features = seg_features

        final_pred = self.fusion_head(seg_features, mask_pred, is_Train)
        fusion_results = dict(final_pred = final_pred)
        #fusion_results = dict()
        return fusion_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        #pdb.set_trace()
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
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
        #pdb.set_trace()
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            #segm_results = self.simple_test_mask(
            #    x, img_metas, det_bboxes, det_labels, rescale=rescale)
            mask_results = self.simple_test_mask_feature_blend(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
                
            #pdb.set_trace()
            ori_shape = img_metas[0]['ori_shape']
            pad_shape = img_metas[0]['pad_shape']
            
            segm_results = []
            
            if det_bboxes[0].shape[0] == 0:
                segm_results.append(
                    [[] for _ in range(self.mask_head.num_classes)])
            else:
                if self.semantic_head is not None:
                    seg_results = self._seg_forward(x, False)
                    
                if self.fusion_head is not None:
                    #pdb.set_trace()
                    fusion_results = self._fusion_forward(seg_results['seg_pred'], mask_results[0], seg_results['seg_features'], pad_shape, det_bboxes[0][:, :4], False)
                    fusion_results['final_pred'] = F.interpolate(fusion_results['final_pred'], size=pad_shape[:2], mode='bilinear', align_corners=True)
                    segm_results.append(self.simple_get_targets(fusion_results['final_pred'], self.test_cfg, ori_shape))
                    #mask_result = F.interpolate(mask_results[0], size=pad_shape[:2], mode='bilinear', align_corners=True)
                    #segm_results.append(self.simple_get_targets(mask_result, self.test_cfg, ori_shape))
                else:
                    mask_result = F.interpolate(mask_results[0], size=pad_shape[:2], mode='bilinear', align_corners=True)
                    segm_results.append(self.simple_get_targets(mask_result, self.test_cfg, ori_shape))
                
            return list(zip(bbox_results, segm_results))

    def simple_test_mask_feature_blend(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        #pdb.set_trace()
        pad_shapes = tuple(meta['pad_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):     
            mask_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
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
            mask_results = self._mask_forward(x, mask_rois)
                    
            mask_pred = mask_results['mask_pred']
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            mask_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    mask_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    mask_result = self.mask_head.parsing_results(
                        mask_preds[i], _bboxes[i], pad_shapes[i], 4.0, True)
                    mask_results.append(mask_result)
        return mask_results
    
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

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        pdb.set_trace()
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        pdb.set_trace()
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels

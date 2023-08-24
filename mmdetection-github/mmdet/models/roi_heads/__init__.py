# Copyright (c) OpenMMLab. All rights reserved.
from .base_roi_head import BaseRoIHead
from .bbox_heads import (BBoxHead, ConvFCBBoxHead, DIIHead,
                         DoubleConvFCBBoxHead, SABLHead, SCNetBBoxHead,
                         Shared2FCBBoxHead, Shared4Conv1FCBBoxHead)
from .cascade_roi_head import CascadeRoIHead
from .cascade_roi_head100x import CascadeRoIHead100x
from .cascade_feature_blend_roi_head_attention_instance import CascadeRoIHeadAttentionInstance
from .cascade_feature_blend_roi_head_attention_coarse_fine_edge import CascadeRoIHeadAttentionCoarserFineEdge
from .cascade_feature_blend_roi_head_attention_edge import CascadeRoIHeadAttentionEdge
from .cascade_feature_blend_roi_head_crop_edge import CascadeRoIHeadCropEdge
from .cascade_feature_blend_roi_head_git import CascadeRoIHeadGit
from .double_roi_head import DoubleHeadRoIHead
from .dynamic_roi_head import DynamicRoIHead
from .grid_roi_head import GridRoIHead
from .htc_roi_head import HybridTaskCascadeRoIHead
from .mask_heads import (CoarseMaskHead, FCNMaskHead, FCNMaskHeadPart, FCNMaskHeadPart100x, FeatureRelayHead,
                         FusedSemanticHead, GlobalContextHead, GridHead,
                         HTCMaskHead, MaskIoUHead, MaskPointHead,
                         SCNetMaskHead, SCNetSemanticHead)
from .mask_scoring_roi_head import MaskScoringRoIHead
from .pisa_roi_head import PISARoIHead
from .point_rend_roi_head import PointRendRoIHead
from .roi_extractors import (BaseRoIExtractor, GenericRoIExtractor,
                             SingleRoIExtractor, SingleRoIExtractorPart, SingleRoIExtractorPart2, SingleRoIExtractorPartNew)
from .scnet_roi_head import SCNetRoIHead
from .shared_heads import ResLayer
from .sparse_roi_head import SparseRoIHead
from .standard_roi_head import StandardRoIHead
from .standard_roi_head_part import StandardRoIHeadPart
from .trident_roi_head import TridentRoIHead
from .feature_blend_roi_head import FeatureBlendHead
from .feature_blend_roi_head_1 import FeatureBlendHead1
from .feature_blend_roi_head_attention_ASPP import FeatureBlendHeadAttentionASPP
from .feature_blend_roi_head_attention_edge import FeatureBlendHeadAttentionEdge
from .feature_blend_roi_head_attention_instance import FeatureBlendHeadAttentionInstance
from .fusion_heads import (PartSegmentationHead, PartSegmentationHead1, PartSegmentationHeadDetachASPP,
                           PartSegmentationHeadDetachEdge, PartSegmentationHeadDetachAttentionEdge, PartSegmentationHeadDetachInstance, FusionHeadDetach, PartSegmentationHeadDetachAttentionEdgeDouble,
                           FCNMaskHeadDetach, FusionHeadAttentionASPP, FusionHeadAttentionEdge, FusionHeadAttentionInstance, FusionHeadAdditiveAttentionEdge)
from .ablation_study import (CascadeRoIHeadAttentionCoarserFineEdgeNO1, CascadeRoIHeadAttentionCoarserFineEdgeNO3,
                          CascadeRoIHeadAttentionCoarserFineEdgeNO5, CascadeRoIHeadAttentionCoarserFineEdgeNO6,
                          PartSegmentationHeadDetachAttentionEdgeNO1, FCNMaskHeadDetachNO1, FeatureBlendHeadAttentionEdgeNO2,       CascadeRoIHeadCropEdgeNO4, CascadeRoIHeadInstance, FCNMaskHeadDetachInstance)
from .parsing_heads import (ParsingHead)
from .deploy_heads import (CascadeRoIHeadDeploy, FCNMaskHeadDeploy, FeatureBlendDeploy, PartSegmentationDeploy, FusionDeploy)
from .ICRA import (CascadeRoIHeadICRA, FeatureBlendICRA, FCNMaskHeadICRA, FusionICRA)
__all__ = [
    'BaseRoIHead', 'CascadeRoIHead', 'CascadeRoIHead100x', 'CascadeRoIHeadAttentionInstance', 'CascadeRoIHeadAttentionEdge', 'CascadeRoIHeadAttentionCoarserFineEdge', 'CascadeRoIHeadGit',
    'CascadeRoIHeadCropEdge', 'DoubleHeadRoIHead', 'MaskScoringRoIHead',
    'HybridTaskCascadeRoIHead', 'GridRoIHead', 'ResLayer', 'BBoxHead',
    'ConvFCBBoxHead', 'DIIHead', 'SABLHead', 'Shared2FCBBoxHead',
    'StandardRoIHead', 'StandardRoIHeadPart', 'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead',
    'FCNMaskHead', 'FCNMaskHeadPart', 'FCNMaskHeadPart100x', 'HTCMaskHead', 'FusedSemanticHead', 'GridHead',
    'MaskIoUHead', 'BaseRoIExtractor', 'GenericRoIExtractor',
    'SingleRoIExtractor', 'SingleRoIExtractorPart', 'SingleRoIExtractorPart2', 'SingleRoIExtractorPartNew', 'PISARoIHead', 'PointRendRoIHead', 'MaskPointHead',
    'CoarseMaskHead', 'DynamicRoIHead', 'SparseRoIHead', 'TridentRoIHead',
    'SCNetRoIHead', 'SCNetMaskHead', 'SCNetSemanticHead', 'SCNetBBoxHead',
    'FeatureRelayHead', 'GlobalContextHead', 'FeatureBlendHead', 'FeatureBlendHead1', 'FeatureBlendHeadAttentionASPP', 
    'FeatureBlendHeadAttentionEdge', 'FeatureBlendHeadAttentionInstance', 'FusionHeadAttentionASPP',
    'PartSegmentationHead', 'PartSegmentationHead1', 'PartSegmentationHeadDetachEdge', 'PartSegmentationHeadDetachAttentionEdge', 'PartSegmentationHeadDetachAttentionEdgeDouble', 'PartSegmentationHeadDetachASPP', 'FusionHeadDetach', 
    'FCNMaskHeadDetach', 'FusionHeadAttentionEdge', 'FusionHeadAttentionInstance', 'FusionHeadAdditiveAttentionEdge', 'CascadeRoIHeadAttentionCoarserFineEdgeNO1',
    'CascadeRoIHeadAttentionCoarserFineEdgeNO3', 'PartSegmentationHeadDetachAttentionEdgeNO1', 'FCNMaskHeadDetachNO1', 'FeatureBlendHeadAttentionEdgeNO2',
    'CascadeRoIHeadAttentionCoarserFineEdgeNO5', 'CascadeRoIHeadAttentionCoarserFineEdgeNO6', 'CascadeRoIHeadCropEdgeNO4', 'ParsingHead', 'FCNMaskHeadDeploy', 'CascadeRoIHeadInstance',
    'FeatureBlendDeploy', 'PartSegmentationDeploy', 'FusionDeploy', 'FCNMaskHeadDetachInstance', 'CascadeRoIHeadICRA', 'FeatureBlendICRA', 'FCNMaskHeadICRA', 'FusionICRA'
]

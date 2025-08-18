"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

YOLO-ONE MAIN MODEL
This file defines the main YoloOne class, which integrates the backbone,
neck, and head, and includes the Mixture of Experts (MoE) routing logic.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any

from yolo_one.models.common import GatingNetwork
from yolo_one.models.yolo_one_backbone import create_yolo_one_backbone
from yolo_one.models.yolo_one_neck import create_yolo_one_neck
from yolo_one.models.yolo_one_head import create_yolo_one_head


class YoloOne(nn.Module):
    """
    YOLO-One: A complete model combining Backbone, Neck, and Head.
    Includes optional Mixture of Experts (MoE) routing for inference.
    """
    def __init__(self, model_size: str = 'nano', **kwargs):
        super().__init__()
        self.model_size = model_size
        
        # 1. Backbone
        self.backbone = create_yolo_one_backbone(model_size=model_size)
        
        # 2. Neck
        self.neck = create_yolo_one_neck(
            model_size=model_size,
            in_channels=self.backbone.out_channels
        )
        
        # 3. Head
        self.head = create_yolo_one_head(
            model_size=model_size,
            in_channels=self.neck.out_channels
        )
        
        # 4. MoE Gating Network (now a native component)
        # The gating network takes a global feature representation as input.
        # We'll use the output of the last stage of the backbone (P5).
        gating_in_channels = self.backbone.out_channels[-1]
        num_experts = len(self.neck.out_channels) # Typically 3 (P3, P4, P5)
        self.gating_network = GatingNetwork(gating_in_channels, num_experts)

    def forward(self, x: torch.Tensor, decode: bool = False, img_size=None):
        features = self.backbone(x)
        fused_features = self.neck(features)

        # MoE Gating is now native
        # The GatingNetwork from common.py handles pooling internally
        gate_scores = self.gating_network(features[-1])

        # The head now receives the soft gate_scores for routing.
        # It can decide to perform hard routing (argmax) internally during inference.
        outputs = self.head(
            fused_features, 
            decode=decode, 
            img_size=img_size,
            gate_scores=gate_scores
        )
        
        # Add gate_scores to the output dict for the loss function
        if isinstance(outputs, dict):
            outputs['gate_scores'] = gate_scores
        
        return outputs
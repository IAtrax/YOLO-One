"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

YOLO-ONE MAIN MODEL
This file defines the main YoloOne class, which integrates the backbone,
"""

import torch
import torch.nn as nn
from yolo_one.models.yolo_one_backbone import create_yolo_one_backbone
from yolo_one.models.yolo_one_neck import create_yolo_one_neck
from yolo_one.models.yolo_one_head import create_yolo_one_head


class YoloOne(nn.Module):
    """
    YOLO-One: A complete model combining Backbone, Neck, and Head.
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
        
    def forward(self, x: torch.Tensor, decode: bool = False, img_size=None):
        features = self.backbone(x)
        fused_features = self.neck(features)

        outputs = self.head(
            fused_features, 
            decode=decode, 
            img_size=img_size,
            gate_scores= None # gate_scores
        )
        
        return outputs
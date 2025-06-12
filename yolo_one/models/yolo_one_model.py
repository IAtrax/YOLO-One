"""
Iatrax Team - 2025 - https://iatrax.com

COMPLETE YOLO-ONE MODEL ASSEMBLY
"""
from .backbone import  create_yolo_one_backbone
from .yolo_one_neck import PAFPN 
from .yolo_one_head import YoloOneDetectionHead

import torch.nn as nn

class YoloOne(nn.Module):
    def __init__(self, model_size='nano'):
        super().__init__()
        self.backbone = create_yolo_one_backbone(model_size)
        self.num_classes = 1
        self.neck = PAFPN(self.backbone.out_channels)
        self.head = YoloOneDetectionHead(
            self.neck.out_channels, 
            num_classes= self.num_classes
        )
    
    def forward(self, x):
        features = self.backbone(x)      # [P3, P4, P5]
        features = self.neck(features)   # Enhanced features
        outputs = self.head(features)    # Detections
        return outputs
        
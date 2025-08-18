"""
Iatrax Team - 2025 - https://iatrax.com

COMPLETE YOLO-ONE MODEL ASSEMBLY
"""
from .yolo_one_backbone import create_yolo_one_backbone
from .yolo_one_neck import create_yolo_one_neck
from .yolo_one_head import create_yolo_one_head
import torch.nn as nn

class YoloOne(nn.Module):
    def __init__(self, model_size='nano'):
        super().__init__()
        # Create each component using its factory function
        self.backbone = create_yolo_one_backbone(model_size)
        self.neck = create_yolo_one_neck(model_size, self.backbone.out_channels)
        self.head = create_yolo_one_head(model_size, in_channels=self.neck.out_channels)
    
    def forward(self, x, decode=False, img_size=None):
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features, decode=decode, img_size=img_size)
        return outputs

        
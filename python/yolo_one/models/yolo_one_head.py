"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

DETECTION HEAD MODULE FOR YOLO-ONE
"""
import torch
import torch.nn as nn
from typing import List

class YoloOneDetectionHead(nn.Module):
    """
     YOLO-One Detection Head
    Optimized for single-class detection
    """
    def __init__(self, in_channels: List[int], num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # cls + bbox + conf
        
        # Detection heads for each scale [P3, P4, P5]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, self.num_outputs, 1)
            )
            for ch in in_channels
        ])
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(feat) for head, feat in zip(self.heads, x)]
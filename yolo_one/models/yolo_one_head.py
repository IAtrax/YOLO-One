"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

DETECTION HEAD MODULE FOR YOLO-ONE
"""
import torch
import torch.nn as nn
from typing import List, Dict

class YoloOneDetectionHead(nn.Module):
    """
    YOLO-One Detection Head - Anchor-Free Architecture
    Optimized for single-class detection with direct regression
    """
    def __init__(self, in_channels: List[int], num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 5  # x, y, w, h, conf (anchor-free)
        
        # Main detection heads for each scale [P3, P4, P5]
        self.detection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, self.num_outputs, 1)
            )
            for ch in in_channels
        ])
        
        # Aspect ratio heads for shape capture
        self.aspect_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch//2, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch//2),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch//2, 1, 1),
                nn.Sigmoid()
            )
            for ch in in_channels
        ])
        
        # Shape confidence heads
        self.shape_confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch//4, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch//4),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch//4, 1, 1),
                nn.Sigmoid()
            )
            for ch in in_channels
        ])
    
    def forward(self, x: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """Forward pass with multi-head predictions"""
        detections = []
        aspects = []
        shape_confs = []
        
        for feat, det_head, asp_head, conf_head in zip(
            x, self.detection_heads, self.aspect_heads, self.shape_confidence_heads
        ):
            detection = det_head(feat)
            aspect = asp_head(feat)
            shape_conf = conf_head(feat)
            
            detections.append(detection)
            aspects.append(aspect)
            shape_confs.append(shape_conf)
        
        return {
            'detections': detections,
            'aspects': aspects,
            'shape_confidences': shape_confs
        }
"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

BACKBONE MODULE FOR YOLO-ONE

This module contains the backbone for YOLO-ONE based on ROI detection
Principe: 
    We propose a backbone with 2 large main blocks: 
       1) A coarse area of interest detection block
       2) A fine box detection block based on the areas of interest found (ROI).
"""
import torch
import torch.nn as nn
from yolo_one.models.common import Conv, CSPBlock
from typing import List, Dict, Any
from yolo_one.configs.config import MODEL_SIZE_MULTIPLIERS as size_multipliers

class RoiDetectionBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        pass

class BoxDetectionBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        pass

class YoloOneBackbone(nn.Module, RoiDetectionBlock, BoxDetectionBlock):
    """
        YOLO-One Backbone.
        This backbone is built dynamically based on a configuration dictionary,
        making it highly flexible and easy to experiment with.
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        pass
    
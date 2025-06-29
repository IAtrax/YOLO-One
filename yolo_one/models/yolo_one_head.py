"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

REFACTORED DETECTION HEAD MODULE FOR YOLO-ONE
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any

# Import reusable blocks from the backbone
from .yolo_one_backbone import Conv

# --- Main Head ---

class YoloOneDetectionHead(nn.Module):
    """
    Refactored YOLO-One Detection Head.

    This head is built dynamically based on a configuration dictionary.
    It separates the regression and classification tasks for clarity.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.num_outputs = 5  # x, y, w, h, conf
        in_channels = config['in_channels']

        # Create the detection heads for each feature map level
        self.detection_heads = nn.ModuleList()
        for in_ch in in_channels:
            head = nn.Sequential(
                Conv(in_ch, in_ch, kernel_size=3),
                nn.Conv2d(in_ch, self.num_outputs, 1)
            )
            self.detection_heads.append(head)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """The forward pass now returns a list of detection tensors."""
        detections = []
        for i, feat in enumerate(x):
            detections.append(self.detection_heads[i](feat))
        return detections

# --- Factory Function ---

def create_yolo_one_head(model_size: str, in_channels: List[int]) -> YoloOneDetectionHead:
    """
    Factory function to create a YOLO-One detection head.
    """
    # The head's configuration is currently simple and doesn't need size-based variations,
    # but the factory function is here for future flexibility.
    
    config = {
        'in_channels': in_channels,
        # Add other head-specific configurations here if needed in the future
    }

    return YoloOneDetectionHead(config)

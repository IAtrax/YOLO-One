"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

REFACTORED NECK MODULE FOR YOLO-ONE (PAFPN Implementation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from yolo_one.configs.config import MODEL_SIZE_MULTIPLIERS as size_multipliers

# Import reusable blocks from the backbone to ensure consistency
from .common import Conv, CSPBlock

# --- Main Neck ---

class PAFPN(nn.Module):
    """
     Path Aggregation Feature Pyramid Network (PAFPN).

    This neck is built dynamically based on a configuration dictionary,
    making it highly flexible and easy to experiment with.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PAFPN (Path Aggregation Feature Pyramid Network).

        This constructor sets up the PAFPN with lateral convolutions, top-down,
        and bottom-up pathways based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary with keys:
                - 'in_channels': List of input channel sizes from the backbone.
                - 'out_channels': List of output channel sizes for the neck.
                - 'num_blocks': Number of blocks in CSP layers.
        """

        super().__init__()
        self.config = config
        in_channels = config['in_channels']
        out_channels = config['out_channels']

        # Lateral convolutions to unify channel dimensions
        self.lateral_convs = nn.ModuleList([
            Conv(in_ch, out_ch, kernel_size=1) for in_ch, out_ch in zip(in_channels, out_channels)
        ])

        # Top-down pathway (from P5 to P3)
        self.top_down_blocks = nn.ModuleList([
            CSPBlock(out_channels[1] + out_channels[2], out_channels[1], num_blocks=config['num_blocks']),
            CSPBlock(out_channels[0] + out_channels[1], out_channels[0], num_blocks=config['num_blocks'])
        ])

        # Bottom-up pathway (from P3 to P5)
        self.downsample_convs = nn.ModuleList([
            Conv(out_channels[0], out_channels[0], kernel_size=3, stride=2),
            Conv(out_channels[1], out_channels[1], kernel_size=3, stride=2)
        ])
        self.bottom_up_blocks = nn.ModuleList([
            CSPBlock(out_channels[0] + out_channels[1], out_channels[1], num_blocks=config['num_blocks']),
            CSPBlock(out_channels[1] + out_channels[2], out_channels[2], num_blocks=config['num_blocks'])
        ])

        # Store final output channels for the head
        self.out_channels = out_channels

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # inputs are [P3, P4, P5] from the backbone
        p3, p4, p5 = inputs

        # Apply lateral convolutions
        lat_p3 = self.lateral_convs[0](p3)
        lat_p4 = self.lateral_convs[1](p4)
        lat_p5 = self.lateral_convs[2](p5)

        # Top-down pathway
        td_p4 = self.top_down_blocks[0](torch.cat([F.interpolate(lat_p5, size=lat_p4.shape[2:], mode='nearest'), lat_p4], 1))
        td_p3 = self.top_down_blocks[1](torch.cat([F.interpolate(td_p4, size=lat_p3.shape[2:], mode='nearest'), lat_p3], 1))

        # Bottom-up pathway
        bu_p4 = self.bottom_up_blocks[0](torch.cat([self.downsample_convs[0](td_p3), td_p4], 1))
        bu_p5 = self.bottom_up_blocks[1](torch.cat([self.downsample_convs[1](bu_p4), lat_p5], 1))

        return [td_p3, bu_p4, bu_p5]

# --- BUILD NECK ---

def create_yolo_one_neck(model_size: str, in_channels: List[int]) -> PAFPN:
    """
    Function to create a YOLO-One neck of a specific size.
    """

    if model_size not in size_multipliers:
        raise ValueError(f"Model size '{model_size}' not supported.")

    w = size_multipliers[model_size]['width']
    d = size_multipliers[model_size]['depth']
    neck_channels = int(in_channels[1] * w)

    config = {
        'in_channels': in_channels,
        'out_channels': [neck_channels] * len(in_channels),
        'num_blocks': max(1, round(2 * d)),
    }

    return PAFPN(config)

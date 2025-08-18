"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

YOLO-One Neck (PAFPN)
Configurable PAFPN with concat or sum fusion, quantization-friendly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_one.configs.config import MODEL_SIZE_MULTIPLIERS as size_multipliers
# Reuse the same blocks for consistency with the backbone
from yolo_one.models.common import Conv, CSPBlock, Bottleneck


class PAFPN(nn.Module):
    """
    Path Aggregation Feature Pyramid Network (PAFPN).

    Inputs:
        - [P3, P4, P5] feature maps from the backbone

    Outputs:
        - [P3', P4', P5'] fused feature maps (same order, low → high resolution)

    Config keys:
        - in_channels:  List[int] input channels (P3, P4, P5)
        - out_channels: List[int] output channels (often equal, e.g., C4)
        - num_blocks:   int, number of internal blocks inside CSP stages
        - sum_fusion:   bool, if True use element-wise sum instead of concat (lighter)
        - block_type:   str, 'csp' or 'bottleneck' to select the building block
        - upsample_mode:str, 'nearest' (default) or 'nearest-exact'
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        in_channels: List[int] = list(config["in_channels"])
        out_channels: List[int] = list(config["out_channels"])
        if len(in_channels) != 3 or len(out_channels) != 3:
            raise ValueError("Expected 3 input and 3 output channels for [P3, P4, P5]")

        self.sum_fusion: bool = bool(config.get("sum_fusion", False))
        self.upsample_mode: str = str(config.get("upsample_mode", "nearest"))
        block_type = config.get("block_type", "csp")
        Block = {"csp": CSPBlock, "bottleneck": Bottleneck}[block_type]
        block_kwargs = {"num_blocks": config["num_blocks"]} if block_type == "csp" else {}

        # Lateral 1x1 convs to unify channels per level
        self.lateral_convs = nn.ModuleList([
            Conv(c_in, c_out, kernel_size=1, stride=1) for c_in, c_out in zip(in_channels, out_channels)
        ])

        # Top-down pathway: P5 -> P4 -> P3
        if self.sum_fusion:
            # Sum fusion keeps the same channel count before the block
            self.top_down_blocks = nn.ModuleList([
                Block(out_channels[1], out_channels[1], **block_kwargs),
                Block(out_channels[0], out_channels[0], **block_kwargs),
            ])
        else:
            # Concat fusion doubles input channels before the block
            self.top_down_blocks = nn.ModuleList([
                Block(out_channels[1] + out_channels[2], out_channels[1], **block_kwargs),
                Block(out_channels[0] + out_channels[1], out_channels[0], **block_kwargs),
            ])

        # Bottom-up pathway: P3' -> P4' -> P5'
        self.downsample_convs = nn.ModuleList([
            Conv(out_channels[0], out_channels[0], kernel_size=3, stride=2),
            Conv(out_channels[1], out_channels[1], kernel_size=3, stride=2),
        ])

        if self.sum_fusion:
            self.bottom_up_blocks = nn.ModuleList([
                Block(out_channels[1], out_channels[1], **block_kwargs),
                Block(out_channels[2], out_channels[2], **block_kwargs),
            ])
        else:
            self.bottom_up_blocks = nn.ModuleList([
                Block(out_channels[0] + out_channels[1], out_channels[1], **block_kwargs),
                Block(out_channels[1] + out_channels[2], out_channels[2], **block_kwargs),
            ])

        # Expose final channels for the head
        self.out_channels = out_channels

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError("Expected [P3, P4, P5] inputs")
        p3, p4, p5 = inputs

        # Lateral projections
        lat_p3 = self.lateral_convs[0](p3)
        lat_p4 = self.lateral_convs[1](p4)
        lat_p5 = self.lateral_convs[2](p5)

        # Top-down fusion
        up_p5_to_p4 = F.interpolate(lat_p5, size=lat_p4.shape[2:], mode=self.upsample_mode)
        td_p4_in = (up_p5_to_p4 + lat_p4) if self.sum_fusion else torch.cat([up_p5_to_p4, lat_p4], dim=1)
        td_p4 = self.top_down_blocks[0](td_p4_in)

        up_p4_to_p3 = F.interpolate(td_p4, size=lat_p3.shape[2:], mode=self.upsample_mode)
        td_p3_in = (up_p4_to_p3 + lat_p3) if self.sum_fusion else torch.cat([up_p4_to_p3, lat_p3], dim=1)
        td_p3 = self.top_down_blocks[1](td_p3_in)

        # Bottom-up aggregation
        p3_down = self.downsample_convs[0](td_p3)
        bu_p4_in = (p3_down + td_p4) if self.sum_fusion else torch.cat([p3_down, td_p4], dim=1)
        bu_p4 = self.bottom_up_blocks[0](bu_p4_in)

        p4_down = self.downsample_convs[1](bu_p4)
        bu_p5_in = (p4_down + lat_p5) if self.sum_fusion else torch.cat([p4_down, lat_p5], dim=1)
        bu_p5 = self.bottom_up_blocks[1](bu_p5_in)

        # Return in ascending resolution: [P3', P4', P5']
        return [td_p3, bu_p4, bu_p5]


def create_yolo_one_neck(
    model_size: str,
    in_channels: List[int],
    **kwargs: Any,
) -> PAFPN:
    """
    Build the YOLO-One PAFPN neck.

    Args:
        model_size:   'nano' | 'small' | 'medium' | 'large'
        in_channels:  [C3, C4, C5] from the backbone (already width-scaled)
        kwargs:
            - num_blocks:   int, defaults to max(1, round(2 * depth_mult))
            - neck_channels:int, defaults to in_channels[1] (avoid double width scaling)
            - sum_fusion:   bool, default True for 'nano', False otherwise
        - block_type:   str, default 'bottleneck' for 'nano', 'csp' otherwise
            - upsample_mode:str, default 'nearest'

    Returns:
        PAFPN instance.
    """
    if model_size not in size_multipliers:
        raise ValueError(f"Model size '{model_size}' not supported.")

    depth_mult = float(size_multipliers[model_size]["depth"])
    num_blocks = int(kwargs.get("num_blocks", max(1, round(2 * depth_mult))))

    # For nano models, use the smallest input channel count to create a much lighter neck.
    # For other models, use the middle channel count as a robust default.
    default_neck_channels = in_channels[0] if model_size == "nano" else in_channels[1]
    neck_channels = int(kwargs.get("neck_channels", default_neck_channels))

    default_block_type = "bottleneck" if model_size == "nano" else "csp"
    block_type = kwargs.get("block_type", default_block_type)

    default_sum = True if model_size == "nano" else False
    sum_fusion = bool(kwargs.get("sum_fusion", default_sum))
    upsample_mode = str(kwargs.get("upsample_mode", "nearest"))

    config: Dict[str, Any] = {
        "in_channels": in_channels,
        "out_channels": [neck_channels] * len(in_channels),
        "num_blocks": num_blocks,
        "sum_fusion": sum_fusion,
        "upsample_mode": upsample_mode,
        "block_type": block_type,
    }
    return PAFPN(config)
"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

Common class or function definitions for YOLO-One models
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


class Conv(nn.Module):
    """Standard Convolution Block: Conv2d + BatchNorm2d + SiLU activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """
    Standard Bottleneck block with residual connection.
    Reduces channels with a 1x1 conv, applies a 3x3 conv, then restores channels.
    """
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True,
                 expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3)
        self.use_residual = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If residual is possible, add input to output, otherwise just return output
        return x + self.cv2(self.cv1(x)) if self.use_residual else self.cv2(self.cv1(x))
    
class SpatialAttention(nn.Module):
    """Spatial attention module for single-class focus"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * attention
    
class CSPBlock(nn.Module):
    """
    Cross Stage Partial (CSP) block with two branches.
    - Branch 1: Goes through bottleneck blocks.
    - Branch 2: Is passed through directly.
    The two branches are then concatenated.
    """
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1,
                 shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # Main convolution for the entire block
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1)
        # The "shortcut" branch
        self.cv2 = Conv(in_channels, hidden_channels, kernel_size=1)
        # The branch with bottleneck transformations
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0) for _ in range(num_blocks)]
        )
        # Final convolution to merge the two branches
        self.cv3 = Conv(2 * hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The main branch passes through the bottlenecks
        main_branch = self.bottlenecks(self.cv1(x))
        # The shortcut branch is a simple convolution
        shortcut_branch = self.cv2(x)
        # Concatenate and merge
        return self.cv3(torch.cat((main_branch, shortcut_branch), dim=1))

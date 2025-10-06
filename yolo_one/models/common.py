"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

Common class or function definitions for YOLO-One models

"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple


class Conv(nn.Module):
    """Standard Convolution Block: Conv2d + BatchNorm2d + SiLU + Dropout."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.conv(x))))


class Bottleneck(nn.Module):
    """
    Standard Bottleneck block with residual connection and dropout.
    """
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True,
                 expansion: float = 0.5, dropout: float = 0.1):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, dropout=dropout)
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3, dropout=dropout)
        self.use_residual = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.cv2(self.cv1(x))
        return out + residual if self.use_residual else out


class SpatialAttention(nn.Module):
    """Spatial attention module with dropout for regularization"""
    def __init__(self, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1).to(x.dtype)))
        attention = self.dropout(attention)
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel-attention module with dropout for overfitting prevention.
    Based on RTMDet - https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.act(self.fc(self.pool(x)))
        attention = self.dropout(attention)
        return x * attention


class CSPBlock(nn.Module): 
    """
    Cross Stage Partial (CSP) block with strategic dropout placement.
    """
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1,
                 shortcut: bool = True, expansion: float = 0.5, dropout: float = 0.2):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        # Lower dropout on input convs
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, dropout=dropout*0.5)
        self.cv2 = Conv(in_channels, hidden_channels, kernel_size=1, dropout=dropout*0.5)
        
        # Higher dropout in bottlenecks (main memorization risk)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut, 
                        expansion=1.0, dropout=dropout) for _ in range(num_blocks)]
        )
        
        # Moderate dropout on output
        self.cv3 = Conv(2 * hidden_channels, out_channels, kernel_size=1, dropout=dropout*0.7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_branch = self.bottlenecks(self.cv1(x))
        shortcut_branch = self.cv2(x)
        return self.cv3(torch.cat((main_branch, shortcut_branch), dim=1))


class GatingNetwork(nn.Module):
    """
    Gating network for MoE with aggressive dropout.
    This is where overfitting happens most!
    """
    def __init__(self, in_channels: int, num_experts: int, hidden_dim: int = 16, 
                 dropout: float = 0.4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # MLP with dropout between each layer
        self.gate = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # Feature dropout
            nn.Linear(hidden_dim, num_experts),
            nn.Dropout(dropout * 0.5),  # Lighter dropout before sigmoid
            nn.Sigmoid()
        )
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_features = self.pool(x)
        flat_features = self.flatten(pooled_features)
        return self.gate(flat_features.to(x.dtype))

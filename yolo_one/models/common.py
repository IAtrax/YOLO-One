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
        """
        Forward pass through the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, height, width).
        """
        return x + self.cv2(self.cv1(x)) if self.use_residual else self.cv2(self.cv1(x))
    
class SpatialAttention(nn.Module):
    """Spatial attention module for single-class focus"""
    def __init__(self, kernel_size: int = 7):
        """
        Initialize the SpatialAttention module.

        Args:
            kernel_size (int, optional): Kernel size of the convolutional layer used in the spatial attention module. Defaults to 7.
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch.mean can upcast to float32 to preserve precision.
        # We must cast it back to the input's dtype to avoid type mismatch in the following conv layer.
        avg_pool = torch.mean(x, dim=1, keepdim=True).to(x.dtype)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * attention

class ChannelAttention(nn.Module): # from https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """
    Channel-attention module for feature recalibration.
    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
        
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))    
class CSPBlock(nn.Module): 
    """
    Cross Stage Partial (CSP) block with two branches.
    - Branch 1: Goes through bottleneck blocks.
    - Branch 2: Is passed through directly.
    The two branches are then concatenated.
    """
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1,
                 shortcut: bool = True, expansion: float = 0.5):
        """
        Initialize the CSPBlock.

        Args:
            in_channels (int): Number of channels in the input feature map.
            out_channels (int): Number of channels in the output feature map.
            num_blocks (int, optional): Number of Bottleneck blocks in the CSPBlock. Defaults to 1.
            shortcut (bool, optional): Whether to use shortcuts in Bottleneck blocks. Defaults to True.
            expansion (float, optional): Expansion factor of Bottleneck blocks. Defaults to 0.5.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = Conv(in_channels, hidden_channels, kernel_size=1)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0) for _ in range(num_blocks)]
        )
        self.cv3 = Conv(2 * hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CSPBlock.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, height, width).
        """

        main_branch = self.bottlenecks(self.cv1(x))
        shortcut_branch = self.cv2(x)
        return self.cv3(torch.cat((main_branch, shortcut_branch), dim=1))

class GatingNetwork(nn.Module):
    """
    A lightweight gating network for Mixture of Experts (MoE).
    Decides which experts (e.g., detection heads) to use based on a global feature map.
    """
    def __init__(self, in_channels: int, num_experts: int, hidden_dim: int = 16):
        """
        Initialize the GatingNetwork.

        Args:
            in_channels (int): Number of channels of the input feature map (e.g., from P5).
            num_experts (int): Number of experts to gate (e.g., 3 for P3, P4, P5 heads).
            hidden_dim (int): Dimension of the hidden layer in the MLP.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.gate = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
            nn.Sigmoid()
        )
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get gating scores.
        """
        # AdaptiveAvgPool2d can upcast to float32. Cast back to input dtype
        # to avoid mixed-precision errors in the following linear layers.
        pooled_features = self.pool(x).to(x.dtype)
        flat_features = self.flatten(pooled_features)
        return self.gate(flat_features)

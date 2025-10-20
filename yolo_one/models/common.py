"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

Common class or function definitions for YOLO-One models

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """
        Forward pass through the standard convolution block.

        Parameters:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after applying the convolution block
        """
        return self.dropout(self.act(self.bn(self.conv(x))))


class Bottleneck(nn.Module):
    """
    Standard Bottleneck block with residual connection and dropout.
    """
    def __init__(self, in_channels: int, out_channels: int, expansion: float = 0.5, dropout: float = 0.1):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, dropout=dropout)
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3, dropout=dropout)
        self.use_residual = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return out + x if self.use_residual else out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7, dropout: float = 0.1):
        """
        Initialize the SpatialAttention module.

        Parameters:
            kernel_size (int): kernel size for the attention convolution (default=7)
            dropout (float): dropout rate for the attention convolution (default=0.1)
        """
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
    Based on RTMDet - https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.act(self.fc(self.pool(x)))
        attention = self.dropout(attention)
        return x * attention


class CSPBlock(nn.Module): 
    """
    Cross Stage Partial (CSP) block with strategic dropout placement.
    """
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1, expansion: float = 0.5, dropout: float = 0.2):
        super().__init__()
        hidden_channels = int(out_channels * expansion)        
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, dropout=dropout*0.5)
        self.cv2 = Conv(in_channels, hidden_channels, kernel_size=1, dropout=dropout*0.5)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, expansion=1.0, dropout=dropout) for _ in range(num_blocks)]
        )
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

class AxialSpatialAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Axial attention: separate attention along height and width dimensions.
        
        Args:
            channels: number of input channels (REQUIRED)
            num_heads: number of attention heads
            dropout: dropout rate
        """
        super().__init__()
        assert channels is not None, "channels must be provided"
        
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        self.scale = self.head_dim ** -0.5
        
        # Horizontal attention
        self.qkv_h = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj_h = nn.Conv2d(channels, channels, 1, bias=False)
        
        # Vertical attention
        self.qkv_v = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj_v = nn.Conv2d(channels, channels, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm_h = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.norm_v = nn.GroupNorm(num_groups=1, num_channels=channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Horizontal attention
        qkv_h = self.qkv_h(x).reshape(B, 3, self.num_heads, self.head_dim, H, W)
        qkv_h = qkv_h.permute(1, 0, 4, 2, 5, 3)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        attn_h = self.dropout(F.softmax(attn_h, dim=-1))
        
        out_h = (attn_h @ v_h).permute(0, 2, 4, 1, 3).reshape(B, C, H, W)
        x = self.norm_h(x + self.proj_h(out_h))
        
        # Vertical attention
        qkv_v = self.qkv_v(x).reshape(B, 3, self.num_heads, self.head_dim, H, W)
        qkv_v = qkv_v.permute(1, 0, 5, 2, 4, 3)
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]
        
        attn_v = (q_v @ k_v.transpose(-2, -1)) * self.scale
        attn_v = self.dropout(F.softmax(attn_v, dim=-1))
        
        out_v = (attn_v @ v_v).permute(0, 2, 4, 3, 1).reshape(B, C, H, W)
        x = self.norm_v(x + self.proj_v(out_v))
        
        return x


class HybridEfficientAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, 
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            channels: number of input channels (REQUIRED)
            reduction: channel reduction ratio
            num_heads: number of attention heads
            dropout: dropout rate
        """
        super().__init__()
        assert channels is not None, "channels must be provided"
        
        self.channels = channels
        reduced_channels = channels // reduction
        
        # Global and local branches
        self.axial_attn = AxialSpatialAttention(channels, num_heads, dropout)
        self.local_conv = Bottleneck(channels, reduced_channels, dropout=dropout)
        self.local_proj = Conv(reduced_channels, channels, kernel_size=1, dropout=0)
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two parallel branches
        global_feat = self.axial_attn(x)
        local_feat = self.local_proj(self.local_conv(x))
        
        # Dynamic fusion
        combined = torch.cat([global_feat, local_feat], dim=1)
        weights = self.gate(combined)
        
        return weights[:, 0:1] * global_feat + weights[:, 1:2] * local_feat
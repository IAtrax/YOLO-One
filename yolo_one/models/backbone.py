"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

BACKBONE MODULE FOR YOLO-ONE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class YoloOneBlock(nn.Module):
    """Single-class optimized block with reduced complexity"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual and stride == 1 and in_channels == out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        
        self.conv2 = DepthwiseSeparableConv(out_channels // 2, out_channels // 2, 
                                          kernel_size=3, stride=stride)
        
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)
        
        # Downsample if needed
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Bottleneck design
        out = self.act(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        if self.use_residual:
            out += identity
            
        return self.act(out)

class CSPLayer(nn.Module):
    """Cross Stage Partial layer optimized for single class"""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3):
        super().__init__()
        hidden_channels = out_channels // 2
        
        # Split channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        
        # Bottleneck blocks
        self.blocks = nn.Sequential(*[
            YoloOneBlock(hidden_channels, hidden_channels) 
            for _ in range(num_blocks)
        ])
        
        # Combine features
        self.conv3 = nn.Conv2d(hidden_channels * 2, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        # Process one branch
        x2 = self.blocks(x2)
        
        # Concatenate and combine
        out = torch.cat([x1, x2], dim=1)
        out = self.act(self.bn(self.conv3(out)))
        
        return out

class YoloOneBackbone(nn.Module):
    """
    YOLO-One Backbone optimized for single-class detection
    
    Key optimizations:
    - Reduced channel count since only one class
    - Simplified feature extraction
    - Focus on speed while maintaining accuracy
    - Efficient spatial attention for single object type
    """
    
    def __init__(self, width_multiple: float = 1.0, depth_multiple: float = 1.0):
        super().__init__()
        
        # Scale factors for different model sizes
        self.width_multiple = width_multiple
        self.depth_multiple = depth_multiple
        
        # Base channel configuration (reduced for single class)
        base_channels = [32, 64, 128, 256, 512]
        channels = [self._make_divisible(ch * width_multiple) for ch in base_channels]
        
        # Stem - Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 320x320 -> 160x160
        self.stage1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(inplace=True),
            CSPLayer(channels[1], channels[1], num_blocks=self._scale_depth(2))
        )
        
        # Stage 2: 160x160 -> 80x80
        self.stage2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU(inplace=True),
            CSPLayer(channels[2], channels[2], num_blocks=self._scale_depth(4))
        )
        
        # Stage 3: 80x80 -> 40x40
        self.stage3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.SiLU(inplace=True),
            CSPLayer(channels[3], channels[3], num_blocks=self._scale_depth(6))
        )
        
        # Stage 4: 40x40 -> 20x20
        self.stage4 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[4]),
            nn.SiLU(inplace=True),
            CSPLayer(channels[4], channels[4], num_blocks=self._scale_depth(4))
        )
        
        # Spatial attention for single class focus
        self.spatial_attention = SpatialAttention(channels[4])
        
        # Store channel info for neck
        self.out_channels = [channels[2], channels[3], channels[4]]  # P3, P4, P5
        
        self._initialize_weights()
    
    def _make_divisible(self, value: float, divisor: int = 8) -> int:
        """Make channels divisible by divisor for efficient computation"""
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value
    
    def _scale_depth(self, depth: int) -> int:
        """Scale depth based on depth_multiple"""
        return max(1, round(depth * self.depth_multiple))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            List of feature maps [P3, P4, P5] for FPN
        """
        # Stem
        x = self.stem(x)
        
        # Multi-scale feature extraction
        x = self.stage1(x)
        
        x = self.stage2(x)
        p3 = x  # 80x80 features
        
        x = self.stage3(x)
        p4 = x  # 40x40 features
        
        x = self.stage4(x)
        x = self.spatial_attention(x)
        p5 = x  # 20x20 features
        
        return [p3, p4, p5]
    
    def _initialize_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class SpatialAttention(nn.Module):
    """Spatial attention module for single-class focus"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Combine statistics
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        return x * attention

def create_yolo_one_backbone(model_size: str = 'nano') -> YoloOneBackbone:
    """
    Create YOLO-One backbone with different sizes
    
    Args:
        model_size: 'nano', 'small', 'medium', 'large'
    """
    size_configs = {
        'nano': {'width': 0.25, 'depth': 0.33},
        'small': {'width': 0.5, 'depth': 0.33},
        'medium': {'width': 0.75, 'depth': 0.67},
        'large': {'width': 1.0, 'depth': 1.0}
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Model size {model_size} not supported. Choose from {list(size_configs.keys())}")
    
    config = size_configs[model_size]
    return YoloOneBackbone(width_multiple=config['width'], 
                          depth_multiple=config['depth'])

if __name__ == "__main__":
    # Test backbone
    backbone = create_yolo_one_backbone('nano')
    x = torch.randn(1, 3, 640, 640)
    features = backbone(x)
    
    print("YOLO-One Backbone Test:")
    print(f"Input: {x.shape}")
    for i, feat in enumerate(features):
        print(f"P{i+3}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"Total parameters: {total_params:,}")
"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

REFACTORED BACKBONE MODULE FOR YOLO-ONE
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any

# --- Building Blocks ---

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

# --- Main Backbone ---

class YoloOneBackbone(nn.Module):
    """
    Refactored YOLO-One Backbone.

    This backbone is built dynamically based on a configuration dictionary,
    making it highly flexible and easy to experiment with.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Initial stem layer
        self.stem = Conv(3, config['channels'][0], kernel_size=6, stride=2)
        
        # Build the main stages of the backbone in a loop
        self.stages = nn.ModuleList()
        in_ch = config['channels'][0]
        
        # Downsampling layers before CSP stages
        self.downsample_layers = nn.ModuleList([
            Conv(in_ch, config['channels'][0], kernel_size=3, stride=2),
            Conv(config['channels'][0], config['channels'][1], kernel_size=3, stride=2),
            Conv(config['channels'][1], config['channels'][2], kernel_size=3, stride=2),
            Conv(config['channels'][2], config['channels'][3], kernel_size=3, stride=2),
        ])

        in_channels_list = [config['channels'][i] for i in range(len(config['stages']))]

        for i, (out_ch, num_blocks, use_csp) in enumerate(config['stages']):
            in_ch = in_channels_list[i]
            if use_csp:
                stage = CSPBlock(in_ch, out_ch, num_blocks=num_blocks)
            else:
                stage = Conv(in_ch, out_ch, kernel_size=3, stride=2)
            self.stages.append(stage)
        
        # Spatial attention for the final feature map
        self.spatial_attention = SpatialAttention()
        
        # Define which feature maps to return for the neck
        self.out_channels = [config['channels'][i] for i in config['output_indices']]

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        x = self.stem(x)
        
        # Pass input through all stages
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            # If the index of this stage is one of our desired outputs, save it
            if i in self.config['output_indices']:
                outputs.append(x)

        # Apply spatial attention to the last output
        outputs[-1] = self.spatial_attention(outputs[-1])
        
        return outputs

    def _initialize_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# --- Factory Function ---

def _make_divisible(value: float, divisor: int = 8) -> int:
    """Make channels divisible by a divisor for hardware efficiency."""
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value

def create_yolo_one_backbone(model_size: str = 'nano') -> YoloOneBackbone:
    """
    Factory function to create a YOLO-One backbone of a specific size.

    This function defines the architecture configurations and passes the
    correct one to the YoloOneBackbone class.
    """
    size_multipliers = {
        'nano':   {'width': 0.25, 'depth': 0.33},
        'small':  {'width': 0.50, 'depth': 0.33},
        'medium': {'width': 0.75, 'depth': 0.67},
        'large':  {'width': 1.00, 'depth': 1.00},
    }
    if model_size not in size_multipliers:
        raise ValueError(f"Model size '{model_size}' not supported.")

    w = size_multipliers[model_size]['width']
    d = size_multipliers[model_size]['depth']

    # Base configuration
    # Format: [output_channels, num_blocks, use_csp_block]
    base_channels = [32, 64, 128, 256, 512]
    
    final_channels = [_make_divisible(c * w) for c in base_channels]

    base_config = {
        'channels': final_channels,
        'stages': [
            # out_ch, num_blocks, use_csp
            ( final_channels[1], max(1, round(2 * d)), True), # Stage 1
            ( final_channels[2], max(1, round(4 * d)), True), # Stage 2 (P3 out)
            ( final_channels[3], max(1, round(6 * d)), True), # Stage 3 (P4 out)
            ( final_channels[4], max(1, round(4 * d)), True), # Stage 4 (P5 out)
        ],
        'output_indices': [1, 2, 3] # Corresponds to P3, P4, P5
    }

    return YoloOneBackbone(base_config)

"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

BACKBONE MODULE FOR YOLO-ONE
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any
from yolo_one.models.common import Conv, CSPBlock, SpatialAttention, Bottleneck
from yolo_one.configs.config import MODEL_SIZE_MULTIPLIERS as size_multipliers


# --- Main Backbone ---

class YoloOneBackbone(nn.Module):
    """
    YOLO-One Backbone!
    The backbone is built dynamically based on a configuration dictionary,
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the YOLO-One Backbone.

        Args:
        - config (Dict[str, Any]): Configuration dictionary for the backbone.
        """
        super().__init__()
        self.config = config
        stem_kernel_size = config.get('stem_kernel_size', 3)
        self.stem = Conv(3, config['channels'][0], kernel_size=stem_kernel_size, stride=2)
        self.out_channels = [config['stages'][i][0] for i in range(1, len(config['stages'])) ]
        self.layers = nn.ModuleList()
        in_ch = config['channels'][0]
        num_stage = len(config['stages'])
        for index, (out_ch, num_blocks) in enumerate(config['stages']):
            stage_layers = []
            stage_layers.append(Conv(in_ch, out_ch, kernel_size=3, stride=2))
            stage_layers.extend([Bottleneck(out_ch, out_ch) for _ in range(num_blocks)])
            if index == num_stage - 1: # Add spatial attention to the last stage (P5)
                stage_layers.append(SpatialAttention())
            self.layers.append(nn.Sequential(*stage_layers))
            in_ch = out_ch
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the YOLO-One backbone.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: List of output tensors, each from a different scale.
        """
        outputs = []
        x = self.stem(x)     # P1   
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0 : # Skip the second stages ( P2)
                continue
            outputs.append(x)
        return outputs # [ P3, P4, P5 ]

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
    Create a YOLO-One backbone.

    Args:
        model_size: str - Size like 'nano' (defines width/depth multipliers).
        multipliers: dict - Mapping of sizes to {'width': float, 'depth': float}.
    Raises:
        ValueError: If model_size not in multipliers.
    Returns:
        YoloOneBackbone
   
    """
    stem_kernel_size = 3 
    if model_size not in size_multipliers:
        raise ValueError(f"Model size '{model_size}' not supported.")
    w = size_multipliers[model_size]['width']          
    # Format: [output_channels, num_blocks]
    base_channels = [64, 64, 128, 256, 512]
    final_channels = [_make_divisible(c * w) for c in base_channels]
    base_config = {
        'channels': final_channels,
        'stages': [
            # [(out_ch, num_blocks), (out_ch, num_blocks), ...]
            ( final_channels[1], 1), # Stage 2
            ( final_channels[2], 4), # Stage 3 (P3 out) 
            ( final_channels[3], 2), # Stage 4 (P4 out) 
            ( final_channels[4], 1), # Stage 5 (P5 out)
        ],
        'stem_kernel_size': stem_kernel_size,
    }

    return YoloOneBackbone(base_config)

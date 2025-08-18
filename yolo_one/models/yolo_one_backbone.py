"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

BACKBONE MODULE FOR YOLO-ONE
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any, Type
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
        self.use_attention = config.get('use_attention', False)
        self.stem = Conv(3, config['channels'][0], kernel_size=stem_kernel_size, stride=2)
        
        self.layers = nn.ModuleList()
        in_ch = config['channels'][0]
        for out_ch, num_blocks, use_csp in config['stages']:
            # Downsampling layer that also changes channel dimension
            self.layers.append(Conv(in_ch, out_ch, kernel_size=3, stride=2))
            
            # CSP or a lighter Bottleneck stage for nano models
            if use_csp:
                stage = CSPBlock(out_ch, out_ch, num_blocks=num_blocks)
            else:
                # For 'nano' models, we use a sequence of lighter Bottleneck blocks
                stage = nn.Sequential(*[Bottleneck(out_ch, out_ch, shortcut=True) for _ in range(num_blocks)])
            self.layers.append(stage)
            in_ch = out_ch  # Update in_ch for the next iteration's downsample_conv

        # The output indices now need to be mapped to the new `layers` structure
        self.output_indices = [idx * 2 + 1 for idx in config['output_indices']]
        self.out_channels = [config['stages'][i][0] for i in config['output_indices']]

        # Create a separate attention layer for each output feature map.
        # This allows each scale to learn its own spatial focus.
        if self.use_attention:
            self.attention_layers = nn.ModuleList([SpatialAttention() for _ in self.output_indices])
        
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
        x = self.stem(x)        
        output_count = 0    
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # If the index of this layer is one of our desired outputs, save it
            if i in self.output_indices:
                if self.use_attention:
                    attended_x = self.attention_layers[output_count](x)
                    outputs.append(attended_x)
                    output_count += 1
                else:
                    outputs.append(x)
        
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
    Create a YOLO-One backbone.

    Args:
        model_size: str - Size like 'nano' (defines width/depth multipliers).
        multipliers: dict - Mapping of sizes to {'width': float, 'depth': float}.
        base_channels: List[int] - Base channel counts for scaling.

    Raises:
        ValueError: If model_size not in multipliers.
   
    """
   
    if model_size not in size_multipliers:
        raise ValueError(f"Model size '{model_size}' not supported.")

    w = size_multipliers[model_size]['width']
    d = size_multipliers[model_size]['depth']

    # Base configuration
    # Format: [output_channels, num_blocks, use_csp_block]
    base_channels = [32, 64, 128, 256, 512]

    # For the nano model, we adjust the base channels to start with a more
    # hardware-friendly width (16 instead of 8) to improve GPU utilization.
    if model_size == 'nano':
        base_channels = [64, 64, 128, 256, 512]
    
    final_channels = [_make_divisible(c * w) for c in base_channels]

    # Nano models prioritize speed: smaller stem and no attention
    use_attention = model_size not in ['nano', 'small']
    stem_kernel_size = 3 if model_size in ['nano', 'small'] else 6
    # For nano models, we replace heavy CSPBlocks with lighter Bottleneck sequences
    use_csp_blocks = model_size != 'nano'

    base_config = {
        'channels': final_channels,
        'stages': [
            # out_ch, num_blocks, use_csp_block
            ( final_channels[1], max(1, round(2 * d)), use_csp_blocks), # Stage 1
            ( final_channels[2], max(1, round(4 * d)), use_csp_blocks), # Stage 2 (P3 out) - Kept
            ( final_channels[3], max(1, round(4 * d)), use_csp_blocks), # Stage 3 (P4 out) - Reduced depth
            ( final_channels[4], max(1, round(2 * d)), use_csp_blocks), # Stage 4 (P5 out) - Reduced depth
        ],
        'output_indices': [1, 2, 3], # Corresponds to P3, P4, P5
        'stem_kernel_size': stem_kernel_size,
        'use_attention': use_attention
    }

    return YoloOneBackbone(base_config)

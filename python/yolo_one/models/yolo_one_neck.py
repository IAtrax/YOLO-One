"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

NECK MODULE FOR YOLO-ONE (PAFPN Implementation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ConvBNAct(nn.Module):
    """Basic convolution block with BatchNorm and activation"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, 
                 stride: int = 1, padding: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    """CSP Block for neck feature fusion"""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        super().__init__()
        hidden_channels = out_channels // 2
        
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNAct(in_channels, hidden_channels, 1)
        
        self.bottlenecks = nn.Sequential(*[
            nn.Sequential(
                ConvBNAct(hidden_channels, hidden_channels, 1),
                ConvBNAct(hidden_channels, hidden_channels, 3, padding=1)
            ) for _ in range(num_blocks)
        ])
        
        self.conv3 = ConvBNAct(hidden_channels * 2, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        for bottleneck in self.bottlenecks:
            x2 = x2 + bottleneck(x2)  # Residual connection
        
        out = torch.cat([x1, x2], dim=1)
        return self.conv3(out)

class PAFPN(nn.Module):
    """
    Path Aggregation Feature Pyramid Network (PAFPN)
    Optimized for YOLO-One single-class detection
    
    Features:
    - Top-down pathway for semantic information
    - Bottom-up pathway for localization information  
    - Lateral connections for feature fusion
    """
    
    def __init__(self, in_channels: List[int], out_channels: int = None):
        super().__init__()
        
        # Auto-determine output channels if not specified
        if out_channels is None:
            out_channels = in_channels[1]  # Use P4 channels as default
        
        self.in_channels = in_channels  # [P3, P4, P5]
        self.out_channels = [out_channels] * 3  # [P3_out, P4_out, P5_out]
        
        # Lateral convolutions to normalize channel dimensions
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(ch, out_channels, 1) for ch in in_channels
        ])
        
        # Top-down pathway
        self.td_convs = nn.ModuleList([
            CSPBlock(out_channels, out_channels, num_blocks=1),
            CSPBlock(out_channels, out_channels, num_blocks=1)
        ])
        
        # Bottom-up pathway  
        self.bu_convs = nn.ModuleList([
            CSPBlock(out_channels, out_channels, num_blocks=1),
            CSPBlock(out_channels, out_channels, num_blocks=1)
        ])
        
        # Downsampling for bottom-up pathway
        self.downsample_convs = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, 3, stride=2, padding=1),
            ConvBNAct(out_channels, out_channels, 3, stride=2, padding=1)
        ])
        
        # Final output convolutions
        self.out_convs = nn.ModuleList([
            CSPBlock(out_channels, out_channels, num_blocks=1) for _ in range(3)
        ])
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass of PAFPN
        
        Args:
            inputs: List of feature maps [P3, P4, P5] from backbone
            
        Returns:
            List of enhanced feature maps [P3_out, P4_out, P5_out]
        """
        # inputs: [P3, P4, P5]
        p3, p4, p5 = inputs
        
        # Lateral connections - normalize channels
        lat_p3 = self.lateral_convs[0](p3)  # P3 lateral
        lat_p4 = self.lateral_convs[1](p4)  # P4 lateral  
        lat_p5 = self.lateral_convs[2](p5)  # P5 lateral
        
        # Top-down pathway (high-level semantic info)
        # P5 -> P4
        td_p4 = lat_p4 + F.interpolate(lat_p5, size=lat_p4.shape[2:], 
                                       mode='nearest')
        td_p4 = self.td_convs[0](td_p4)
        
        # P4 -> P3  
        td_p3 = lat_p3 + F.interpolate(td_p4, size=lat_p3.shape[2:], 
                                       mode='nearest')
        td_p3 = self.td_convs[1](td_p3)
        
        # Bottom-up pathway (low-level localization info)
        # P3 -> P4
        bu_p4 = td_p4 + self.downsample_convs[0](td_p3)
        bu_p4 = self.bu_convs[0](bu_p4)
        
        # P4 -> P5
        bu_p5 = lat_p5 + self.downsample_convs[1](bu_p4)
        bu_p5 = self.bu_convs[1](bu_p5)
        
        # Final outputs
        out_p3 = self.out_convs[0](td_p3)
        out_p4 = self.out_convs[1](bu_p4)  
        out_p5 = self.out_convs[2](bu_p5)
        
        return [out_p3, out_p4, out_p5]

def create_pafpn(in_channels: List[int], out_channels: int = None) -> PAFPN:
    """
    Create PAFPN neck for YOLO-One
    
    Args:
        in_channels: Channels from backbone [P3, P4, P5]
        out_channels: Output channels (auto if None)
    """
    return PAFPN(in_channels, out_channels)

if __name__ == "__main__":
    # Test PAFPN
    # Simulate backbone outputs
    p3 = torch.randn(1, 128, 80, 80)   # P3: 80x80
    p4 = torch.randn(1, 256, 40, 40)   # P4: 40x40  
    p5 = torch.randn(1, 512, 20, 20)   # P5: 20x20
    
    # Create PAFPN
    neck = create_pafpn([128, 256, 512], out_channels=256)
    
    # Forward pass
    outputs = neck([p3, p4, p5])
    
    print("PAFPN Neck Test:")
    print(f"Input P3: {p3.shape}")
    print(f"Input P4: {p4.shape}")  
    print(f"Input P5: {p5.shape}")
    print("---")
    for i, out in enumerate(outputs):
        print(f"Output P{i+3}: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in neck.parameters())
    print(f"PAFPN parameters: {total_params:,}")
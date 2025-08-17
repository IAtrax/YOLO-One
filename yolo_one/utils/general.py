"""
IAtrax Team - 2025 - https://iatrax.com

LICENSE: MIT

Exponential Moving Average for YOLO-One model

Utility functions for YOLO-One training

"""

import torch
import torch.nn as nn
import copy
import torch
import random
import numpy as np
import logging
from pathlib import Path


class EMAModel:
    """
    Exponential Moving Average model for YOLO-One
    Maintains shadow parameters for better generalization
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initialize EMA model
        
        Args:
            model: Source model
            decay: EMA decay factor
        """
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        self.updates = 0
        
        # Disable gradients for EMA model
        for param in self.ema.parameters():
            param.requires_grad_(False)
    
    def update(self, model: nn.Module):
        """Update EMA model parameters"""
        
        with torch.no_grad():
            self.updates += 1
            decay = self.decay * (1 - torch.exp(torch.tensor(-self.updates / 2000.0)))
            
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                ema_param.mul_(decay).add_(model_param, alpha=1 - decay)
    
    def state_dict(self):
        """Get EMA state dict"""
        return {
            'ema': self.ema.state_dict(),
            'updates': self.updates,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        self.ema.load_state_dict(state_dict['ema'])
        self.updates = state_dict['updates']
        self.decay = state_dict['decay']

def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging configuration"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def get_device(device: str = None) -> torch.device:
    """Get training device"""
    
    if device:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(checkpoint: dict, filepath: str):
    """Save checkpoint to file"""
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, device: str = 'cpu') -> dict:
    """Load checkpoint from file"""
    return torch.load(filepath, map_location=device)

def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU between boxes
        
        Args:
            boxes1: First set of boxes [N, 4]
            boxes2: Second set of boxes [M, 4]
            
        Returns:
            IoU matrix [N, M]
        """
        # Calculate intersection
        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate union
        union_area = area1[:, None] + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        
        return iou

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2) format.
    
    Args:
        boxes (torch.Tensor): Boxes in [N, 4] (cx, cy, w, h) format.
    Returns:
        torch.Tensor: Boxes in [N, 4] (x1, y1, x2, y2) format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    b = [(cx - w / 2), (cy - h / 2), (cx + w / 2), (cy + h / 2)]
    return torch.stack(b, dim=-1)

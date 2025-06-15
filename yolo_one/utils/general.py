"""
IAtrax Team - 2025 - https://iatrax.com

LICENSE: MIT

Exponential Moving Average for YOLO-One model

Utility functions for YOLO-One training

"""

import torch
import torch.nn as nn
from typing import Optional
import copy
import torch
import random
import numpy as np
import logging
import os
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
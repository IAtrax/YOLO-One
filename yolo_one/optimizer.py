"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

Optimizer configuration for YOLO-One training
Optimized for single-class detection training
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    MultiStepLR, 
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR
)
from typing import Dict, Any, Optional, Union, List


class YoloOneOptimizer:
    """
    Optimizer factory and configuration for YOLO-One
    Supports various optimizers and learning rate schedules
    """
    
    @staticmethod
    def create_optimizer(
        model: torch.nn.Module,
        optimizer_type: str = 'adamw',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        momentum: float = 0.9,
        lr_multipliers: Optional[Dict[str, float]] = None,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create optimizer for YOLO-One model
        
        Args:
            model: YOLO-One model
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Initial learning rate
            weight_decay: Weight decay factor
            momentum: Momentum factor (for SGD)
            lr_multipliers: Optional dictionary to scale LR for different parts of the model.
            betas: Betas for Adam optimizers
            eps: Epsilon for numerical stability
            **kwargs: Additional optimizer arguments
            
        Returns:
            Configured optimizer
        """
        
        # Group parameters by type for different learning rates
        param_groups = YoloOneOptimizer._create_param_groups(
            model, learning_rate, weight_decay, lr_multipliers
        )
        
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                param_groups,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                **kwargs
            )
        
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                **kwargs
            )
        
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True,
                **kwargs
            )
        
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(
                param_groups,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                eps=eps,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        return optimizer
    
    @staticmethod
    def _create_param_groups(
        model: torch.nn.Module, 
        base_lr: float, 
        weight_decay: float,
        lr_multipliers: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create parameter groups with different learning rates
        
        Args:
            model: YOLO-One model
            base_lr: Base learning rate
            weight_decay: Weight decay
            lr_multipliers: Optional dict with keys like 'backbone', 'neck', 'head'.
            
        Returns:
            List of parameter groups
        """
        
        # Default multipliers if not provided
        multipliers = {
            'backbone': 0.1,
            'neck': 0.5,
            'head': 1.0
        }
        if lr_multipliers:
            multipliers.update(lr_multipliers)

        # Different learning rates for different components
        backbone_params = []
        neck_params = []
        head_params = []
        bn_params = []
        bias_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Batch normalization parameters (no weight decay)
            if 'bn' in name or 'batch_norm' in name:
                bn_params.append(param)
            
            # Bias parameters (no weight decay)
            elif 'bias' in name:
                bias_params.append(param)
            
            # Head parameters (higher learning rate)
            elif 'head' in name:
                head_params.append(param)
            
            # Neck parameters (medium learning rate)
            elif 'neck' in name or 'fpn' in name:
                neck_params.append(param)
            
            # Backbone parameters (lower learning rate)
            else:
                backbone_params.append(param)
        
        param_groups = []
        
        # Backbone (lower LR for pre-trained features)
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr * multipliers['backbone'],
                'weight_decay': weight_decay,
                'name': 'backbone'
            })
        
        # Neck (medium LR)
        if neck_params:
            param_groups.append({
                'params': neck_params,
                'lr': base_lr * multipliers['neck'],
                'weight_decay': weight_decay,
                'name': 'neck'
            })
        
        # Head (full LR)
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': base_lr * multipliers['head'],
                'weight_decay': weight_decay,
                'name': 'head'
            })
        
        # Batch norm (no weight decay)
        if bn_params:
            param_groups.append({
                'params': bn_params,
                'lr': base_lr,
                'weight_decay': 0.0,
                'name': 'bn'
            })
        
        # Bias (no weight decay)
        if bias_params:
            param_groups.append({
                'params': bias_params,
                'lr': base_lr,
                'weight_decay': 0.0,
                'name': 'bias'
            })
        
        return param_groups
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'cosine',
        total_epochs: int = 300,
        warmup_epochs: int = 5,
        min_lr_ratio: float = 0.01,
        **kwargs
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler
        
        Args:
            optimizer: Optimizer instance
            scheduler_type: Type of scheduler
            total_epochs: Total training epochs
            warmup_epochs: Warmup epochs
            min_lr_ratio: Minimum LR as ratio of initial LR
            **kwargs: Additional scheduler arguments
            
        Returns:
            Configured scheduler
        """
        
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == 'cosine':
            # Cosine annealing with warmup
            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.1,
                    total_iters=warmup_epochs
                )
                
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_epochs - warmup_epochs),
                    eta_min=optimizer.param_groups[0]['lr'] * min_lr_ratio
                )
                
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_epochs),
                    eta_min=optimizer.param_groups[0]['lr'] * min_lr_ratio
                )
        
        elif scheduler_type == 'multistep':
            milestones = kwargs.get('milestones', [total_epochs * 2//3, total_epochs * 9//10])
            gamma = kwargs.get('gamma', 0.1)
            
            scheduler = MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma
            )
        
        elif scheduler_type == 'onecycle':
            max_lr = kwargs.get('max_lr', optimizer.param_groups[0]['lr'] * 10)
            
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_epochs,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        
        elif scheduler_type == 'cosine_restart':
            T_0 = kwargs.get('T_0', total_epochs // 4)
            T_mult = kwargs.get('T_mult', 2)
            
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=optimizer.param_groups[0]['lr'] * min_lr_ratio
            )
        
        elif scheduler_type == 'none':
            scheduler = None
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return scheduler


def create_yolo_one_optimizer(
    model: torch.nn.Module,
    optimizer_config: Dict[str, Any]
) -> tuple:
    """
    Factory function to create optimizer and scheduler for YOLO-One
    
    Args:
        model: YOLO-One model
        optimizer_config: Configuration dictionary
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    
    # Extract optimizer parameters
    optimizer_params = optimizer_config.get('optimizer', {})
    scheduler_params = optimizer_config.get('scheduler', {})
    
    # Create optimizer
    optimizer = YoloOneOptimizer.create_optimizer(model, **optimizer_params)
    
    # Create scheduler
    scheduler = YoloOneOptimizer.create_scheduler(optimizer, **scheduler_params)
    
    return optimizer, scheduler
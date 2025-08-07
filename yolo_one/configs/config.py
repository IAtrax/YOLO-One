"""
Hyperparameter configuration for YOLO-One training
Optimized configurations for different scenarios
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class YoloOneConfig:
    """
    Configuration manager for YOLO-One hyperparameters
    Provides optimized configurations for different use cases
    """
    
    @staticmethod
    def get_nano_config() -> Dict[str, Any]:
        """
        Configuration for YOLO-One Nano model
        Optimized for speed and low resource usage
        """
        return {
            # Model configuration
            'model': {
                'model_size': 'nano',
                'input_size': 640,
                'num_classes': 1,  # Single-class
                'width_multiple': 0.25,
                'depth_multiple': 0.33
            },
            
            # Training configuration
            'training': {
                'epochs': 300,
                'batch_size': 64,
                'accumulate_batches': 1,
                'mixed_precision': True,
                'compile_model': True,
                'use_ema': True,
                'ema_decay': 0.9999
            },
            
            # Loss configuration
            'loss': {
                'box_weight': 7.5,
                'obj_weight': 1.0,
                'focal_alpha': 0.25,
                'focal_gamma': 1.5,
                'iou_type': 'ciou',
                'label_smoothing': 0.0,
                'p5_weight_boost': 1.2
            },
            
            # Optimizer configuration
            'optimizer': {
                'optimizer': {
                    'optimizer_type': 'adamw',
                    'learning_rate': 0.001,
                    'weight_decay': 0.0005,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8
                },
                'scheduler': {
                    'scheduler_type': 'cosine',
                    'total_epochs': 300,
                    'warmup_epochs': 5,
                    'min_lr_ratio': 0.01
                }
            },
            
            # Data augmentation
            'augmentation': {
                'mosaic': 0.5,
                'mixup': 0.1,
                'copy_paste': 0.1,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5
            },
            
            # Metrics configuration
            'metrics': {
                'iou_thresholds': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                'confidence_threshold': 0.25,
                'nms_threshold': 0.45,
                'max_detections': 300
            },
            
            # Validation configuration
            'validation': {
                'val_interval': 10,
                'save_best': True,
                'patience': 50,
                'min_delta': 0.001
            }
        }
    
    @staticmethod
    def get_small_config() -> Dict[str, Any]:
        """
        Configuration for YOLO-One Small model
        Balanced speed and accuracy
        """
        config = YoloOneConfig.get_nano_config()
        
        # Update model configuration
        config['model'].update({
            'model_size': 'small',
            'width_multiple': 0.5,
            'depth_multiple': 0.33
        })
        
        # Update training configuration
        config['training'].update({
            'batch_size': 32,
            'epochs': 400
        })
        
        # Update optimizer
        config['optimizer']['optimizer']['learning_rate'] = 0.0008
        config['optimizer']['scheduler']['total_epochs'] = 400
        
        return config
    
    @staticmethod
    def get_medium_config() -> Dict[str, Any]:
        """
        Configuration for YOLO-One Medium model
        Higher accuracy, moderate speed
        """
        config = YoloOneConfig.get_nano_config()
        
        # Update model configuration
        config['model'].update({
            'model_size': 'medium',
            'width_multiple': 0.75,
            'depth_multiple': 0.67
        })
        
        # Update training configuration
        config['training'].update({
            'batch_size': 16,
            'epochs': 500,
            'accumulate_batches': 2
        })
        
        # Update optimizer
        config['optimizer']['optimizer']['learning_rate'] = 0.0006
        config['optimizer']['scheduler']['total_epochs'] = 500
        
        return config
    
    @staticmethod
    def get_large_config() -> Dict[str, Any]:
        """
        Configuration for YOLO-One Large model
        Maximum accuracy
        """
        config = YoloOneConfig.get_nano_config()
        
        # Update model configuration
        config['model'].update({
            'model_size': 'large',
            'width_multiple': 1.0,
            'depth_multiple': 1.0
        })
        
        # Update training configuration
        config['training'].update({
            'batch_size': 8,
            'epochs': 600,
            'accumulate_batches': 4
        })
        
        # Update optimizer
        config['optimizer']['optimizer']['learning_rate'] = 0.0005
        config['optimizer']['scheduler']['total_epochs'] = 600
        
        return config
    
    @staticmethod
    def get_mobile_config() -> Dict[str, Any]:
        """
        Configuration optimized for mobile deployment
        Ultra-fast inference, minimal size
        """
        config = YoloOneConfig.get_nano_config()
        
        # Update model configuration
        config['model'].update({
            'input_size': 320,  # Smaller input size
            'width_multiple': 0.125,  # Ultra-lightweight
            'depth_multiple': 0.25
        })
        
        # Update training configuration
        config['training'].update({
            'batch_size': 128,
            'epochs': 200,
            'mixed_precision': True
        })
        
        # More aggressive augmentation
        config['augmentation'].update({
            'mosaic': 0.8,
            'mixup': 0.2,
            'copy_paste': 0.2
        })
        
        return config
    
    @staticmethod
    def get_high_accuracy_config() -> Dict[str, Any]:
        """
        Configuration optimized for maximum accuracy
        """
        config = YoloOneConfig.get_large_config()
        
        # Update model configuration
        config['model'].update({
            'input_size': 832,  # Higher resolution
        })
        
        # Update training configuration
        config['training'].update({
            'epochs': 800,
            'batch_size': 4,
            'accumulate_batches': 8
        })
        
        # Update loss for higher accuracy
        config['loss'].update({
            'box_weight': 10.0,  # Higher box loss weight
            'focal_gamma': 2.0,  # Stronger focal loss
            'label_smoothing': 0.1
        })
        
        # Conservative augmentation for accuracy
        config['augmentation'].update({
            'mosaic': 0.3,
            'mixup': 0.05,
            'copy_paste': 0.05,
            'scale': 0.3
        })
        
        return config
    
    @staticmethod
    def get_edge_config() -> Dict[str, Any]:
        """
        Configuration for edge devices (Raspberry Pi, etc.)
        """
        config = YoloOneConfig.get_mobile_config()
        
        # Update for edge deployment
        config['model'].update({
            'input_size': 256,  # Very small input
            'width_multiple': 0.1,
            'depth_multiple': 0.2
        })
        
        # Simple training for edge
        config['training'].update({
            'mixed_precision': False,  # May not be supported
            'compile_model': False
        })
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: str):
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            filepath: Config file path
            
        Returns:
            Configuration dictionary
        """
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = YoloOneConfig.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged


# Factory functions for easy access
def create_yolo_one_config(
    model_size: str = 'nano',
    use_case: str = 'general'
) -> Dict[str, Any]:
    """
    Factory function to create YOLO-One configuration
    
    Args:
        model_size: Model size ('nano', 'small', 'medium', 'large')
        use_case: Use case ('general', 'mobile', 'edge', 'high_accuracy')
        
    Returns:
        Configuration dictionary
    """
    
    if use_case == 'mobile':
        return YoloOneConfig.get_mobile_config()
    elif use_case == 'edge':
        return YoloOneConfig.get_edge_config()
    elif use_case == 'high_accuracy':
        return YoloOneConfig.get_high_accuracy_config()
    else:
        # General use case
        if model_size == 'nano':
            return YoloOneConfig.get_nano_config()
        elif model_size == 'small':
            return YoloOneConfig.get_small_config()
        elif model_size == 'medium':
            return YoloOneConfig.get_medium_config()
        elif model_size == 'large':
            return YoloOneConfig.get_large_config()
        else:
            raise ValueError(f"Unsupported model size: {model_size}")

MODEL_SIZE_MULTIPLIERS = {
        'nano':   {'width': 0.25, 'depth': 0.33},
        'small':  {'width': 0.50, 'depth': 0.33},
        'medium': {'width': 0.75, 'depth': 0.67},
        'large':  {'width': 1.00, 'depth': 1.00},
    }
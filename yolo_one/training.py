"""
IAtrax Team - 2025 - https://iatrax.com

LICENSE: MIT

Training script for YOLO-One single-class object detection
Optimized training pipeline with comprehensive logging and validation
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import argparse
import warnings
from tqdm import tqdm # Import tqdm for progress bars

warnings.filterwarnings('ignore')

# YOLO-One imports
from yolo_one.models.yolo_one_model import YoloOne
from yolo_one.losses import create_yolo_one_loss
# Import the comprehensive metrics creator
from yolo_one.metrics import SimpleYoloOneMetrics
from yolo_one.optimizer import create_yolo_one_optimizer
from yolo_one.configs.config import create_yolo_one_config, YoloOneConfig
from yolo_one.data.preprocessing import create_yolo_one_dataset
from yolo_one.utils.general import EMAModel
from yolo_one.utils.general import (
    setup_logging, save_checkpoint, load_checkpoint, # We'll modify save_checkpoint internally
    get_device, set_random_seed, count_parameters
)


class YoloOneTrainer:
    """
    Comprehensive trainer for YOLO-One single-class detection
    Features: Mixed precision, EMA, advanced scheduling, comprehensive logging
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        output_dir: str = './runs',
        resume_from: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize YOLO-One trainer
        
        Args:
            config: Training configuration dictionary
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            output_dir: Output directory for logs and checkpoints
            resume_from: Path to checkpoint to resume from
            device: Training device
        """
        
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Setup device
        self.device = get_device(device)
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.run_dir / 'train.log')
        self.writer = SummaryWriter(self.run_dir / 'tensorboard')
        
        # Save config
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._create_optimizer()
        
        # Initialize metrics (using the comprehensive YoloOneMetrics)
        self.metrics = self._create_metrics()
        
        # Initialize EMA if enabled
        self.ema_model = self._create_ema_model()
        
        # Mixed precision scaler
        self.use_mixed_precision = (
            self.config['training'].get('mixed_precision', False) and 
            self.device.type == 'cuda' and 
            torch.cuda.is_available()
        )
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_map = 0.0
        self.patience_counter = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)
        
        self.logger.info(f"YOLO-One Trainer initialized")
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")
        self.logger.info(f"Training device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_mixed_precision}")
        self.logger.info(f"Output directory: {self.run_dir}")
    
    def _create_model(self) -> nn.Module:
        """Create and configure YOLO-One model"""
        
        model_config = self.config['model']
        
        # Create model with proper configuration
        model = YoloOne(
            model_size=model_config.get('model_size', 'nano')
        )
        
        model = model.to(self.device)
        
        # Compile model if enabled (PyTorch 2.0+)
        if self.config['training'].get('compile_model', False):
            try:
                model = torch.compile(model)
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function"""
        
        loss_config = self.config['loss']
        
        criterion = create_yolo_one_loss(
            box_weight=loss_config.get('box_weight', 7.5),
            obj_weight=loss_config.get('obj_weight', 1.0),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 1.5),
            iou_type=loss_config.get('iou_type', 'ciou'),
            label_smoothing=loss_config.get('label_smoothing', 0.0),
            p5_weight_boost=loss_config.get('p5_weight_boost', 1.2)
        )
        
        return criterion.to(self.device)
    
    def _create_optimizer(self) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """Create optimizer and scheduler"""
        
        optimizer_config = self.config['optimizer']
        
        # Update total epochs in scheduler config
        if 'scheduler' in optimizer_config:
            optimizer_config['scheduler']['total_epochs'] = self.config['training']['epochs']
        
        optimizer, scheduler = create_yolo_one_optimizer(self.model, optimizer_config)
        
        return optimizer, scheduler
    
    def _create_metrics(self):
        """Create metrics evaluator"""
        
        metrics_config = self.config['metrics']
        

        metrics = SimpleYoloOneMetrics(device=self.device)
        
        
        return metrics
    
    def _create_ema_model(self) -> Optional[EMAModel]:
        """Create EMA model if enabled"""
        
        if not self.config['training'].get('use_ema', False):
            return None
        
        ema_decay = self.config['training'].get('ema_decay', 0.9999)
        ema_model = EMAModel(self.model, decay=ema_decay)
        
        self.logger.info(f"EMA model initialized with decay: {ema_decay}")
        
        return ema_model
    
    def train(self):
        """Main training loop"""
        
        self.logger.info("Starting YOLO-One training")
        self.logger.info(f"Training for {self.config['training']['epochs']} epochs")
        
        total_epochs = self.config['training']['epochs']
        val_interval = self.config['validation']['val_interval']
        
        start_time = time.time()
        
        # TQDM progress bar for epochs
        for epoch in tqdm(range(self.current_epoch, total_epochs), desc="Total Progress"):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            if (epoch + 1) % val_interval == 0 or epoch == total_epochs - 1:
                val_metrics = self._validate_epoch()
                
                # Check for best model
                current_map = val_metrics.get('mAP', 0.0)
                if current_map > self.best_map:
                    self.best_map = current_map
                    self.patience_counter = 0
                    # Save only model state_dict for best model to reduce size
                    self._save_checkpoint(is_best=True, only_model_state=True) 
                    self.logger.info(f"New best mAP: {self.best_map:.4f}")
                else:
                    self.patience_counter += val_interval
                
                # Log validation metrics
                self._log_metrics(val_metrics, epoch, phase='val')
            else:
                val_metrics = {}
            
            # Log training metrics
            self._log_metrics(train_metrics, epoch, phase='train')
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                else:
                    self.scheduler.step()
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('lr/learning_rate', current_lr, epoch)
            
            # Save regular checkpoint (full state for resumption)
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(is_best=False, only_model_state=False)
            
            # Early stopping check
            patience = self.config['validation'].get('patience', 50)
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
            start_time = time.time()
        
        # Final validation and save (only model state_dict)
        final_metrics = self._validate_epoch()
        self._save_checkpoint(is_best=False, filename='final_model.pt', only_model_state=True)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best mAP: {self.best_map:.4f}")
        
        self.writer.close()
        
        return {
            'best_map': self.best_map,
            'final_metrics': final_metrics,
            'total_time': total_time
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        
        running_loss = 0.0
        running_box_loss = 0.0
        running_obj_loss = 0.0
        num_batches = len(self.train_dataloader)
        accumulate_batches = self.config['training'].get('accumulate_batches', 1)
        
        epoch_start_time = time.time()
        
        # TQDM progress bar for batches
        train_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch+1} Training", leave=False)
        for batch_idx, batch in enumerate(train_bar):
            
            # Move data to device (images and targets are already tensors from DataLoader)
            images = batch['images'].to(self.device, non_blocking=True) # non_blocking for faster transfer
            targets = batch['targets'].to(self.device, non_blocking=True) # non_blocking for faster transfer
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_mixed_precision):
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets, self.model)
                loss = loss_dict['total_loss'] / accumulate_batches
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % accumulate_batches == 0:
                
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema_model:
                    self.ema_model.update(self.model)
                
                self.global_step += 1
            
            # Update running losses
            running_loss += loss_dict['total_loss'].item()
            running_box_loss += loss_dict['box_loss'].item()
            running_obj_loss += loss_dict['obj_loss'].item()
            
            # Update progress bar description
            current_lr = self.optimizer.param_groups[0]['lr']
            train_bar.set_postfix(
                loss=f"{loss_dict['total_loss'].item():.4f}",
                box_l=f"{loss_dict['box_loss'].item():.4f}",
                obj_l=f"{loss_dict['obj_loss'].item():.4f}",
                lr=f"{current_lr:.6f}"
            )

            # Log to tensorboard 
            self.writer.add_scalar('batch/total_loss', loss_dict['total_loss'].item(), self.global_step)
            self.writer.add_scalar('batch/box_loss', loss_dict['box_loss'].item(), self.global_step)
            self.writer.add_scalar('batch/obj_loss', loss_dict['obj_loss'].item(), self.global_step)
            self.writer.add_scalar('batch/learning_rate', current_lr, self.global_step)

        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = running_loss / num_batches
        avg_box_loss = running_box_loss / num_batches
        avg_obj_loss = running_obj_loss / num_batches
        
        return {
            'total_loss': avg_loss,
            'box_loss': avg_box_loss,
            'obj_loss': avg_obj_loss,
            'epoch_time': epoch_time,
            'batches_per_second': num_batches / epoch_time
        }
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        
        # Use EMA model for validation if available
        model_to_eval = self.ema_model.ema if self.ema_model else self.model
        model_to_eval.eval()
        
        self.metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        val_start_time = time.time()
        
        # TQDM progress bar for validation batches
        val_bar = tqdm(self.val_dataloader, desc=f"Epoch {self.current_epoch+1} Validation", leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_bar):
                
                # Move data to device (images and targets are already tensors from DataLoader)
                images = batch['images'].to(self.device, non_blocking=True) # non_blocking for faster transfer
                targets = batch['targets'].to(self.device, non_blocking=True) # non_blocking for faster transfer
                
                # Forward pass
                inference_start = time.time()
                with autocast(enabled=self.use_mixed_precision):
                    predictions_raw = model_to_eval(images) # Raw predictions from model (on GPU)
                    loss_dict = self.criterion(predictions_raw, targets, model_to_eval)
                
                inference_time = time.time() - inference_start
                
                # Prepare predictions for metrics (transfer to CPU before update for multiprocessing safety)
                predictions_for_metrics = {}
                if isinstance(predictions_raw, dict): # Handle multi-head output
                    for key, val_list in predictions_raw.items():
                        # Ensure each tensor in the list is moved to CPU
                        predictions_for_metrics[key] = [v.cpu() for v in val_list]
                else: # Handle single tensor output
                    # This case assumes predictions_raw is a single tensor or a list of single tensors
                    # If it's a list of tensors, ensure each is moved to CPU
                    if isinstance(predictions_raw, list):
                        predictions_for_metrics['detections'] = [v.cpu() for v in predictions_raw]
                    else: # A single tensor
                        predictions_for_metrics['detections'] = [predictions_raw.cpu()]


                # Targets are already on device; move to CPU for metrics (if not already done by DataLoader)
                targets_for_metrics = targets.cpu()


                # Update metrics
                self.metrics.update(
                    predictions=predictions_for_metrics, # Pass CPU predictions
                    targets=targets_for_metrics, # Pass CPU targets
                    inference_time=inference_time / len(images)  # Per image
                )
                
                running_loss += loss_dict['total_loss'].item()

                # Update progress bar description
                val_bar.set_postfix(val_loss=f"{loss_dict['total_loss'].item():.4f}")
        
        # Compute metrics
        val_metrics = self.metrics.compute()
        val_metrics['val_loss'] = running_loss / num_batches
        val_metrics['val_time'] = time.time() - val_start_time
        
        return val_metrics
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str):
        """Log metrics to tensorboard and logger"""
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{phase}/{key}', value, epoch)
        
        # Log important metrics
        if phase == 'val':
            self.logger.info(
                f"Validation Epoch [{epoch+1}] - "
                f"mAP: {metrics.get('mAP', 0):.4f} - "
                f"mAP@0.5: {metrics.get('mAP@0.5', 0):.4f} - "
                f"Val Loss: {metrics.get('val_loss', 0):.4f}"
            )
    
    def _save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None, only_model_state: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best (bool): True if this is the best model so far.
            filename (Optional[str]): Custom filename for the checkpoint.
            only_model_state (bool): If True, only saves the model's state_dict
                                     (and EMA model's state_dict if applicable),
                                     resulting in a smaller file for deployment.
        """
        
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch+1}.pt'
        
        if only_model_state:
            # For final or best model, save only model's state_dict (or EMA's)
            model_to_save = self.ema_model.ema if self.ema_model else self.model
            # Ensure model is on CPU before saving if it's for general deployment
            # Otherwise, keep on GPU if it's for GPU-only inference
            torch.save(model_to_save.state_dict(), self.run_dir / filename)
            self.logger.info(f"Lightweight model state saved: {self.run_dir / filename}")
        else:
            # For regular checkpoints, save full state for resuming training
            checkpoint = {
                'epoch': self.current_epoch + 1,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'ema_state_dict': self.ema_model.state_dict() if self.ema_model else None,
                'best_map': self.best_map,
                'config': self.config,
                'model_parameters': count_parameters(self.model)
            }
            checkpoint_path = self.run_dir / filename
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Full checkpoint saved: {checkpoint_path}")
        
        # Always save best_model.pt as a lightweight model state
        if is_best:
            best_path = self.run_dir / 'best_model.pt'
            model_to_save = self.ema_model.ema if self.ema_model else self.model
            torch.save(model_to_save.state_dict(), best_path)
            self.logger.info(f"Best model state (lightweight) saved: {best_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load EMA state
        if checkpoint['ema_state_dict'] and self.ema_model:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_map = checkpoint['best_map']
        
        self.logger.info(f"Resumed training from epoch {self.current_epoch}")
        self.logger.info(f"Best mAP: {self.best_map:.4f}")


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='YOLO-One Training')
    parser.add_argument('--config', type=str, default='configs/yolo_one_nano.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset configuration')
    parser.add_argument('--model-size', type=str, default='nano',
                        choices=['nano', 'small', 'medium', 'large'],
                        help='Model size')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Training device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='./runs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load configuration
    if os.path.exists(args.config):
        config = YoloOneConfig.load_config(args.config)
    else:
        config = create_yolo_one_config(model_size=args.model_size)
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['optimizer']['optimizer']['learning_rate'] = args.lr
    
    root_dir = Path(args.data)
    
    # Create data loaders
    # Added pin_memory=True for faster GPU transfers
    _, train_dataloader = create_yolo_one_dataset(
        root_dir=root_dir,
        split='train',
        batch_size=config['training']['batch_size'],
        img_size=(config['model']['input_size'], config['model']['input_size'] ),
        augmentations=config['augmentation'],
        num_workers=args.workers,   
    )
    _, val_dataloader = create_yolo_one_dataset(
        root_dir=root_dir,
        split='val',
        batch_size=config['training']['batch_size'],
        img_size=(config['model']['input_size'], config['model']['input_size']),
        num_workers=args.workers,
       
    )
    
    # Create trainer
    trainer = YoloOneTrainer(
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_dir=args.output_dir,
        resume_from=args.resume,
        device=args.device
    )
    
    # Start training
    try:
        results = trainer.train()
        print(f"\nTraining completed successfully!")
        print(f"Best mAP: {results['best_map']:.4f}")
        print(f"Final model saved in: {trainer.run_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save interrupted model as lightweight state
        trainer._save_checkpoint(is_best=False, filename='interrupted_model.pt', only_model_state=True) 
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

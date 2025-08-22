"""
IATrax Team - 2025 - https://iatrax.com

LICENSE: MIT

METRICS MODULE FOR YOLO-ONE

"""

import torch
from typing import List, Optional, Tuple, Union, Dict
from torchmetrics.detection import MeanAveragePrecision

class YoloOneMetrics:
    """
    Comprehensive metrics for YOLO-One validation.
    Handles decoded predictions and computes mAP if torchmetrics is available.
    """
    
    def __init__(self, device='cuda', conf_threshold: float = 0.001, iou_thresholds: Optional[List[float]] = None):
        self.device = device
        self.conf_threshold = conf_threshold
        self.map_metric = MeanAveragePrecision(
            box_format='cxcywh',
            iou_thresholds=iou_thresholds,
            class_metrics=False
        ).to(self.device)
        self.reset()
    
    def reset(self):
        """Resets the state of the metrics."""
        if self.map_metric:
            self.map_metric.reset()
        self.inference_times = []
    
    def update(self, predictions: Union[Dict[str, torch.Tensor], List[torch.Tensor]], 
               targets: torch.Tensor, input_size: Tuple[int, int], **kwargs):
        """
        Update metrics with a batch of decoded predictions and targets.

        Args:
            predictions: Either Dict with 'decoded' key or List of decoded prediction tensors
            targets (torch.Tensor): Ground truth targets [N, 6] (batch_idx, class, xc, yc, w, h) on the correct device.
            input_size (Tuple[int, int]): The (height, width) of the model input, for scaling targets.
        """
        if not self.map_metric:
            return

        if isinstance(predictions, dict):
            if 'decoded' in predictions:
                decoded_preds = predictions['decoded']
            else:
                print("Warning: No 'decoded' key found in predictions dict")
                return
        elif isinstance(predictions, list):
            decoded_preds = predictions
        else:
            print(f"Warning: Unexpected predictions type: {type(predictions)}")
            return
        if not decoded_preds or len(decoded_preds) == 0:
            print("Warning: No decoded predictions found")
            return

        # Concatenate predictions from all levels and reshape
        # [B, 5, H, W] -> [B, H, W, 5] -> [B, N, 5]
        try:
            all_preds = [p.permute(0, 2, 3, 1).reshape(p.shape[0], -1, 5) for p in decoded_preds]
            batch_preds = torch.cat(all_preds, dim=1)  # [B, Total_N, 5]
        except Exception as e:
            print(f"Error processing predictions: {e}")
            print(f"Predictions shape: {[p.shape if hasattr(p, 'shape') else type(p) for p in decoded_preds]}")
            return

        # Get input size for scaling
        h, w = input_size

        # Prepare data for torchmetrics
        preds_for_map = []
        targets_for_map = []

        for i in range(batch_preds.shape[0]):
            # Filter predictions by confidence
            pred = batch_preds[i]
            conf_mask = pred[:, 4] >= self.conf_threshold
            pred = pred[conf_mask]

            if pred.shape[0] == 0:
                preds_for_map.append({
                    'boxes': torch.empty((0, 4), device=self.device),
                    'scores': torch.empty((0,), device=self.device),
                    'labels': torch.empty((0,), device=self.device, dtype=torch.int64)  # ✅ int64 pour les labels
                })
            else:
                # Scale prediction boxes from normalized [0,1] to pixel coordinates
                pred_boxes = pred[:, :4].clone()
                pred_boxes[:, 0] *= w # scale cx
                pred_boxes[:, 1] *= h # scale cy
                pred_boxes[:, 2] *= w # scale w
                pred_boxes[:, 3] *= h # scale h

                preds_for_map.append({
                    'boxes': pred_boxes,  # [N, 4] in cxcywh pixel format
                    'scores': pred[:, 4],
                    'labels': torch.zeros(pred.shape[0], device=self.device, dtype=torch.int64) # ✅ int64 pour labels
                })

            # Filter targets for the current image
            target = targets[targets[:, 0] == i]
            
            if target.shape[0] == 0:
                # Pas de targets pour cette image
                targets_for_map.append({
                    'boxes': torch.empty((0, 4), device=self.device),
                    'labels': torch.empty((0,), device=self.device, dtype=torch.int64)
                })
            else:
                # Clone and scale target boxes from normalized [0,1] to pixel coordinates
                target_boxes = target[:, 2:].clone() # [M, 4] in cxcywh format
                target_boxes[:, 0] *= w # scale cx
                target_boxes[:, 1] *= h # scale cy
                target_boxes[:, 2] *= w # scale w
                target_boxes[:, 3] *= h # scale h
                
                targets_for_map.append({
                    'boxes': target_boxes, # [M, 4] in cxcywh pixel format
                    'labels': torch.zeros(target.shape[0], device=self.device, dtype=torch.int64) # ✅ int64 pour labels
                })
        has_preds = any(len(p['boxes']) > 0 for p in preds_for_map)
        has_targets = any(len(t['boxes']) > 0 for t in targets_for_map)
        
        if not has_targets:
            print("Warning: No targets found in batch")
            return
            
        # Update the mAP metric
        try:
            self.map_metric.update(preds_for_map, targets_for_map)
        except Exception as e:
            print(f"Error updating mAP metric: {e}")
            print(f"Preds shapes: {[p['boxes'].shape for p in preds_for_map]}")
            print(f"Targets shapes: {[t['boxes'].shape for t in targets_for_map]}")
        
        if 'inference_time' in kwargs:
            self.inference_times.append(kwargs['inference_time'])
    
    def compute(self):
        """Compute the final metrics for the epoch."""
        if not self.map_metric:
            return {'mAP': 0.0, 'mAP@0.5': 0.0}

        try:
            map_results = self.map_metric.compute()
            
            map_value = map_results.get('map', torch.tensor(0.0))
            map_50_value = map_results.get('map_50', torch.tensor(0.0))
            map_75_value = map_results.get('map_75', torch.tensor(0.0))
            
            map_final = map_value.item() if not torch.isnan(map_value) else 0.0
            map_50_final = map_50_value.item() if not torch.isnan(map_50_value) else 0.0
            map_75_final = map_75_value.item() if not torch.isnan(map_75_value) else 0.0
            
            metrics = {
                'mAP': map_final,
                'mAP@0.5': map_50_final,
                'mAP@0.75': map_75_final,
                'avg_inference_time': sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0.0
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {
                'mAP': 0.0,
                'mAP@0.5': 0.0,
                'mAP@0.75': 0.0,
                'avg_inference_time': 0.0
            }
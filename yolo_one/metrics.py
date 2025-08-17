"""
IATrax Team - 2023 - https://iatrax.com

LICENSE: MIT

METRICS MODULE FOR YOLO-ONE

"""

import torch
from typing import List
from torchmetrics.detection import MeanAveragePrecision
from typing import Tuple

class YoloOneMetrics:
    """
    Comprehensive metrics for YOLO-One validation.
    Handles decoded predictions and computes mAP if torchmetrics is available.
    """
    
    def __init__(self, device='cuda', conf_threshold=0.001, iou_threshold=0.6):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.map_metric = MeanAveragePrecision(box_format='cxcywh').to(self.device)

        self.reset()
    
    def reset(self):
        """Resets the state of the metrics."""
        if self.map_metric:
            self.map_metric.reset()
        self.inference_times = []
    
    def update(self, predictions: List[torch.Tensor], targets: torch.Tensor, input_size: Tuple[int, int], **kwargs):
        """
        Update metrics with a batch of decoded predictions and targets.

        Args:
            predictions (List[torch.Tensor]): List of decoded prediction tensors from different FPN levels.
                                              Each tensor is [B, 5, H, W] on the correct device.
            targets (torch.Tensor): Ground truth targets [N, 6] (batch_idx, class, xc, yc, w, h) on the correct device.
            input_size (Tuple[int, int]): The (height, width) of the model input, for scaling targets.
        """
        if not self.map_metric:
            return

        # Concatenate predictions from all levels and reshape
        # [B, 5, H, W] -> [B, H, W, 5] -> [B, N, 5]
        # Predictions are already on the correct device
        all_preds = [p.permute(0, 2, 3, 1).reshape(p.shape[0], -1, 5) for p in predictions]
        batch_preds = torch.cat(all_preds, dim=1)  # [B, Total_N, 5]

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

            preds_for_map.append({
                'boxes': pred[:, :4],  # [N, 4] in cxcywh format
                'scores': pred[:, 4],
                'labels': torch.zeros(pred.shape[0], device=self.device, dtype=torch.int32) # Single class
            })

            # Filter targets for the current image
            target = targets[targets[:, 0] == i]
            # Clone and scale target boxes from normalized [0,1] to pixel coordinates
            target_boxes = target[:, 2:].clone() # [M, 4] in cxcywh format
            target_boxes[:, 0] *= w # scale cx
            target_boxes[:, 1] *= h # scale cy
            target_boxes[:, 2] *= w # scale w
            target_boxes[:, 3] *= h # scale h
            targets_for_map.append({
                'boxes': target_boxes, # [M, 4] in cxcywh pixel format
                'labels': torch.zeros(target.shape[0], device=self.device, dtype=torch.int32) # Single class
            })

        # Update the mAP metric
        self.map_metric.update(preds_for_map, targets_for_map)
        
        if 'inference_time' in kwargs:
            self.inference_times.append(kwargs['inference_time'])
    
    def compute(self):
        """Compute the final metrics for the epoch."""
        if not self.map_metric:
            return {'mAP': 0.0, 'mAP@0.5': 0.0}

        map_results = self.map_metric.compute()
        
        metrics = {
            'mAP': map_results['map'].item(),
            'mAP@0.5': map_results['map_50'].item(),
            'mAP@0.75': map_results['map_75'].item(),
        }
        return metrics
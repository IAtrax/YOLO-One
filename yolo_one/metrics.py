"""
IAtrax Team - 2025 - https://iatrax.com

LICENSE: MIT

Metrics module for YOLO-One single-class object detection
Optimized metrics for single-class evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Union
from collections import defaultdict
import time


class YoloOneMetrics:
    """
    Comprehensive metrics for YOLO-One single-class detection
    Optimized for single-class evaluation without per-class overhead
    """
    
    def __init__(
        self,
        iou_thresholds: List[float] = None,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        max_detections: int = 300,
        device: str = 'cuda'
    ):
        """
        Initialize YOLO-One metrics
        
        Args:
            iou_thresholds: IoU thresholds for mAP calculation
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS threshold for post-processing
            max_detections: Maximum detections per image
            device: Computation device
        """
        
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.device = torch.device(device)
        
        # Metrics storage
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.ground_truths = []
        self.inference_times = []
        self.nms_times = []
        
    def update(
        self,
        predictions: Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]],
        targets: torch.Tensor,
        inference_time: float = 0.0,
        nms_time: float = 0.0
    ):
        """
        Update metrics with batch predictions and targets
        
        Args:
            predictions: Model predictions (List[Tensor] or Dict with multi-head outputs)
            targets: Ground truth targets [batch_idx, class, x, y, w, h]
            inference_time: Model inference time in seconds
            nms_time: NMS post-processing time in seconds
        """
        
        # Convert predictions to detections
        detections = self._predictions_to_detections(predictions)
        
        # Store for evaluation
        self.predictions.extend(detections)
        self.ground_truths.append(targets.cpu())
        
        if inference_time > 0:
            self.inference_times.append(inference_time)
        if nms_time > 0:
            self.nms_times.append(nms_time)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary with computed metrics
        """
        
        if not self.predictions or not self.ground_truths:
            return self._empty_metrics()
        
        # Prepare data
        all_predictions = self._prepare_predictions()
        all_targets = self._prepare_targets()
        
        # Compute metrics
        metrics = {}
        
        # Detection metrics
        ap_metrics = self._compute_average_precision(all_predictions, all_targets)
        metrics.update(ap_metrics)
        
        # Precision/Recall at different confidence thresholds
        pr_metrics = self._compute_precision_recall_curves(all_predictions, all_targets)
        metrics.update(pr_metrics)
        
        # Speed metrics
        speed_metrics = self._compute_speed_metrics()
        metrics.update(speed_metrics)
        
        # Detection statistics
        stats_metrics = self._compute_detection_stats(all_predictions, all_targets)
        metrics.update(stats_metrics)
        
        return metrics
    
    def _predictions_to_detections(self, predictions: Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]]) -> List[Dict]:
        """
        Convert raw model predictions to detection format
        Support both legacy format and new multi-head format
        
        Args:
            predictions: Raw model outputs (List[Tensor] or Dict with multi-head)
            
        Returns:
            List of detection dictionaries per batch item
        """
        
        # Handle different prediction formats
        if isinstance(predictions, dict):
            # New multi-head format: extract main detections
            if 'detections' in predictions:
                detection_preds = predictions['detections']
            else:
                # Fallback: use first available key
                first_key = list(predictions.keys())[0]
                detection_preds = predictions[first_key]
        else:
            # Legacy format: direct list of tensors
            detection_preds = predictions
        
        # Ensure we have a list of tensors
        if not isinstance(detection_preds, list):
            detection_preds = [detection_preds]
        
        batch_size = detection_preds[0].shape[0]
        batch_detections = []
        
        for batch_idx in range(batch_size):
            # Extract predictions for current batch item
            batch_preds = [pred[batch_idx:batch_idx+1] for pred in detection_preds]
            
            # Decode predictions
            boxes, scores = self._decode_predictions(batch_preds)
            
            # Apply confidence threshold
            if len(scores) > 0:
                conf_mask = scores >= self.confidence_threshold
                boxes = boxes[conf_mask]
                scores = scores[conf_mask]
            
            # Apply NMS
            if len(boxes) > 0:
                keep_indices = self._nms(boxes, scores, self.nms_threshold)
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                
                # Limit max detections
                if len(boxes) > self.max_detections:
                    boxes = boxes[:self.max_detections]
                    scores = scores[:self.max_detections]
            
            batch_detections.append({
                'boxes': boxes.cpu() if len(boxes) > 0 else torch.empty(0, 4),
                'scores': scores.cpu() if len(scores) > 0 else torch.empty(0),
                'batch_idx': batch_idx
            })
        
        return batch_detections
    
    def _decode_predictions(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode raw predictions to boxes and scores
        Support both anchor-based and anchor-free formats
        
        Args:
            predictions: Raw predictions for single batch item
            
        Returns:
            Tuple of (boxes, scores) tensors
        """
        
        all_boxes = []
        all_scores = []
        
        # Scale information
        scales = [
            {'stride': 8, 'size': 80},   # P3
            {'stride': 16, 'size': 40},  # P4
            {'stride': 32, 'size': 20}   # P5
        ]
        
        for scale_idx, (pred, scale_info) in enumerate(zip(predictions, scales)):
            
            batch_size, channels, height, width = pred.shape
            
            # Detect format: anchor-free (5 channels) vs anchor-based (15 channels)
            if channels == 5:
                # Anchor-free format
                boxes, scores = self._decode_anchor_free(pred, scale_info, height, width)
            elif channels == 15:
                # Anchor-based format with 3 anchors
                boxes, scores = self._decode_anchor_based(pred, scale_info, height, width, num_anchors=3)
            elif channels == 10:
                # Anchor-based format with 2 anchors
                boxes, scores = self._decode_anchor_based(pred, scale_info, height, width, num_anchors=2)
            else:
                # Unknown format: skip
                print(f"Warning: Unknown prediction format with {channels} channels at scale {scale_idx}")
                continue
            
            all_boxes.append(boxes)
            all_scores.append(scores)
        
        # Concatenate all scales
        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
        else:
            all_boxes = torch.empty(0, 4, device=self.device)
            all_scores = torch.empty(0, device=self.device)
        
        return all_boxes, all_scores
    
    def _decode_anchor_free(self, pred: torch.Tensor, scale_info: Dict, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode anchor-free predictions"""
        
        # pred: [1, 5, H, W]
        pred = pred.squeeze(0)  # [5, H, W]
        pred = pred.permute(1, 2, 0)  # [H, W, 5]
        pred = pred.reshape(-1, 5)  # [H*W, 5]
        
        # Generate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing='ij'
        )
        
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.reshape(-1, 2)  # [H*W, 2]
        
        # Decode boxes
        xy = torch.sigmoid(pred[:, :2]) + grid
        wh = torch.exp(pred[:, 2:4])
        
        # Convert to absolute coordinates
        xy = xy / torch.tensor([width, height], device=self.device)
        wh = wh / torch.tensor([width, height], device=self.device)
        
        # Convert to x1, y1, x2, y2 format
        x1 = xy[:, 0] - wh[:, 0] / 2
        y1 = xy[:, 1] - wh[:, 1] / 2
        x2 = xy[:, 0] + wh[:, 0] / 2
        y2 = xy[:, 1] + wh[:, 1] / 2
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        scores = torch.sigmoid(pred[:, 4])
        
        return boxes, scores
    
    def _decode_anchor_based(self, pred: torch.Tensor, scale_info: Dict, height: int, width: int, num_anchors: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode anchor-based predictions"""
        
        # Anchors for each scale
        if num_anchors == 3:
            anchors = [
                [[10, 13], [16, 30], [33, 23]],      # P3
                [[30, 61], [62, 45], [59, 119]],     # P4
                [[116, 90], [156, 198], [373, 326]]  # P5
            ]
        else:  # num_anchors == 2
            anchors = [
                [[10, 13], [16, 30]],      # P3
                [[30, 61], [62, 45]],      # P4
                [[116, 90], [156, 198]]    # P5
            ]
        
        scale_idx = {8: 0, 16: 1, 32: 2}.get(scale_info['stride'], 0)
        scale_anchors = anchors[scale_idx]
        
        # Reshape predictions
        batch_size, channels, height, width = pred.shape
        pred = pred.view(batch_size, num_anchors, 5, height, width)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        pred = pred.view(-1, 5)  # [N, 5]
        
        # Generate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing='ij'
        )
        
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.unsqueeze(0).repeat(num_anchors, 1, 1, 1).view(-1, 2)
        
        # Decode boxes
        xy = torch.sigmoid(pred[:, :2]) + grid
        wh = torch.exp(pred[:, 2:4]) * torch.tensor(scale_anchors, device=self.device).repeat(height * width, 1)
        
        # Convert to absolute coordinates
        xy = xy / torch.tensor([width, height], device=self.device)
        wh = wh / torch.tensor([width, height], device=self.device)
        
        # Convert to x1, y1, x2, y2 format
        x1 = xy[:, 0] - wh[:, 0] / 2
        y1 = xy[:, 1] - wh[:, 1] / 2
        x2 = xy[:, 0] + wh[:, 0] / 2
        y2 = xy[:, 1] + wh[:, 1] / 2
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        scores = torch.sigmoid(pred[:, 4])
        
        return boxes, scores
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Non-Maximum Suppression
        
        Args:
            boxes: Boxes tensor [N, 4] in x1,y1,x2,y2 format
            scores: Scores tensor [N]
            threshold: IoU threshold
            
        Returns:
            Indices of kept boxes
        """
        
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Sort by scores
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep = []
        while len(sorted_indices) > 0:
            # Take highest scoring box
            current = sorted_indices[0]
            keep.append(current)
            
            if len(sorted_indices) == 1:
                break
            
            # Compute IoU with remaining boxes
            current_box = boxes[current:current+1]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            iou = self._compute_iou(current_box, remaining_boxes).squeeze(0)
            
            # Keep boxes with IoU below threshold
            mask = iou <= threshold
            sorted_indices = sorted_indices[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long)
    
    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes
        
        Args:
            boxes1: First set of boxes [N, 4]
            boxes2: Second set of boxes [M, 4]
            
        Returns:
            IoU matrix [N, M]
        """
        
        # Expand dimensions for broadcasting
        boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
        boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]
        
        # Intersection
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        
        return iou
    
    def _prepare_predictions(self) -> List[Dict]:
        """Prepare predictions for evaluation"""
        return self.predictions
    
    def _prepare_targets(self) -> torch.Tensor:
        """Prepare ground truth targets for evaluation"""
        return torch.cat(self.ground_truths, dim=0) if self.ground_truths else torch.empty(0, 6)
    
    def _compute_average_precision(self, predictions: List[Dict], targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute Average Precision metrics
        
        Args:
            predictions: List of prediction dictionaries
            targets: Ground truth targets
            
        Returns:
            Dictionary with AP metrics
        """
        
        metrics = {}
        
        # Compute AP for each IoU threshold
        aps = []
        for iou_threshold in self.iou_thresholds:
            ap = self._compute_ap_at_iou(predictions, targets, iou_threshold)
            aps.append(ap)
            metrics[f'AP@{iou_threshold:.2f}'] = ap
        
        # mAP (mean over IoU thresholds 0.5:0.95)
        metrics['mAP'] = np.mean(aps)
        metrics['mAP@0.5'] = aps[0] if aps else 0.0
        metrics['mAP@0.75'] = aps[5] if len(aps) > 5 else 0.0
        
        return metrics
    
    def _compute_ap_at_iou(self, predictions: List[Dict], targets: torch.Tensor, iou_threshold: float) -> float:
        """
        Compute AP at specific IoU threshold
        
        Args:
            predictions: Predictions list
            targets: Ground truth targets
            iou_threshold: IoU threshold
            
        Returns:
            Average Precision value
        """
        
        # Collect all detections and ground truths
        all_detections = []
        all_ground_truths = []
        
        # Group targets by batch
        targets_by_batch = defaultdict(list)
        for target in targets:
            batch_idx = int(target[0])
            targets_by_batch[batch_idx].append(target[2:6])  # [x, y, w, h]
        
        # Process each batch
        for batch_idx, pred_dict in enumerate(predictions):
            detections = pred_dict['boxes']  # [N, 4] in x1,y1,x2,y2 format
            scores = pred_dict['scores']     # [N]
            
            # Add to all detections with scores
            for det, score in zip(detections, scores):
                all_detections.append({
                    'box': det,
                    'score': score.item(),
                    'batch_idx': batch_idx
                })
            
            # Add ground truths for this batch
            if batch_idx in targets_by_batch:
                gt_boxes = targets_by_batch[batch_idx]
                for gt_box in gt_boxes:
                    # Convert from center format to corner format
                    x_center, y_center, width, height = gt_box
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    all_ground_truths.append({
                        'box': torch.tensor([x1, y1, x2, y2]),
                        'batch_idx': batch_idx,
                        'matched': False
                    })
        
        if not all_detections or not all_ground_truths:
            return 0.0
        
        # Sort detections by score (descending)
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Compute precision and recall
        tp = []
        fp = []
        
        for detection in all_detections:
            det_box = detection['box']
            det_batch = detection['batch_idx']
            
            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(all_ground_truths):
                if gt['batch_idx'] != det_batch or gt['matched']:
                    continue
                
                iou = self._compute_iou(det_box.unsqueeze(0), gt['box'].unsqueeze(0)).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if detection is true positive
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                all_ground_truths[best_gt_idx]['matched'] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # Compute cumulative precision and recall
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        
        recall = tp / len(all_ground_truths)
        precision = tp / (tp + fp + 1e-6)
        
        # Compute AP using interpolation
        ap = self._compute_ap_from_pr(precision, recall)
        
        return ap
    
    def _compute_ap_from_pr(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """
        Compute AP from precision-recall curve using 11-point interpolation
        
        Args:
            precision: Precision values
            recall: Recall values
            
        Returns:
            Average Precision
        """
        
        # Add endpoints
        precision = np.concatenate([[0], precision, [0]])
        recall = np.concatenate([[0], recall, [1]])
        
        # Compute maximum precision for each recall level
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            indices = recall >= t
            if np.any(indices):
                ap += np.max(precision[indices])
        
        return ap / 11.0
    
    def _compute_precision_recall_curves(self, predictions: List[Dict], targets: torch.Tensor) -> Dict[str, float]:
        """Compute precision-recall metrics at different confidence thresholds"""
        
        confidence_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        metrics = {}
        
        for conf_thresh in confidence_thresholds:
            # Filter predictions by confidence
            filtered_preds = []
            for pred_dict in predictions:
                conf_mask = pred_dict['scores'] >= conf_thresh
                if conf_mask.any():
                    filtered_preds.append({
                        'boxes': pred_dict['boxes'][conf_mask],
                        'scores': pred_dict['scores'][conf_mask],
                        'batch_idx': pred_dict['batch_idx']
                    })
                else:
                    filtered_preds.append({
                        'boxes': torch.empty(0, 4),
                        'scores': torch.empty(0),
                        'batch_idx': pred_dict['batch_idx']
                    })
            
            # Compute precision and recall
            ap = self._compute_ap_at_iou(filtered_preds, targets, 0.5)
            metrics[f'P@{conf_thresh}'] = ap
        
        return metrics
    
    def _compute_speed_metrics(self) -> Dict[str, float]:
        """Compute speed and performance metrics"""
        
        metrics = {}
        
        if self.inference_times:
            avg_inference = np.mean(self.inference_times) * 1000  # ms
            fps_inference = 1.0 / np.mean(self.inference_times)
            
            metrics['inference_time_ms'] = avg_inference
            metrics['inference_fps'] = fps_inference
        
        if self.nms_times:
            avg_nms = np.mean(self.nms_times) * 1000  # ms
            
            metrics['nms_time_ms'] = avg_nms
        
        if self.inference_times and self.nms_times:
            total_time = np.mean(self.inference_times) + np.mean(self.nms_times)
            metrics['total_time_ms'] = total_time * 1000
            metrics['total_fps'] = 1.0 / total_time
        
        return metrics
    
    def _compute_detection_stats(self, predictions: List[Dict], targets: torch.Tensor) -> Dict[str, float]:
        """Compute detection statistics"""
        
        metrics = {}
        
        # Count total detections and ground truths
        total_detections = sum(len(pred['boxes']) for pred in predictions)
        total_ground_truths = len(targets)
        
        metrics['total_detections'] = total_detections
        metrics['total_ground_truths'] = total_ground_truths
        metrics['avg_detections_per_image'] = total_detections / max(len(predictions), 1)
        
        # Confidence statistics
        all_scores = []
        for pred in predictions:
            all_scores.extend(pred['scores'].tolist())
        
        if all_scores:
            metrics['avg_confidence'] = np.mean(all_scores)
            metrics['max_confidence'] = np.max(all_scores)
            metrics['min_confidence'] = np.min(all_scores)
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary"""
        
        metrics = {}
        
        # AP metrics
        for iou_threshold in self.iou_thresholds:
            metrics[f'AP@{iou_threshold:.2f}'] = 0.0
        
        metrics.update({
            'mAP': 0.0,
            'mAP@0.5': 0.0,
            'mAP@0.75': 0.0,
            'P@0.1': 0.0,
            'P@0.25': 0.0,
            'P@0.5': 0.0,
            'P@0.75': 0.0,
            'P@0.9': 0.0,
            'total_detections': 0,
            'total_ground_truths': 0,
            'avg_detections_per_image': 0.0,
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 0.0
        })
        
        return metrics


def create_yolo_one_metrics(
    iou_thresholds: List[float] = None,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    max_detections: int = 300,
    device: str = 'cuda'
) -> YoloOneMetrics:
    """
    Factory function to create YOLO-One metrics
    
    Args:
        iou_thresholds: IoU thresholds for mAP calculation
        confidence_threshold: Minimum confidence for detections
        nms_threshold: NMS threshold for post-processing
        max_detections: Maximum detections per image
        device: Computation device
        
    Returns:
        Configured YoloOneMetrics instance
    """
    return YoloOneMetrics(
        iou_thresholds=iou_thresholds,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        max_detections=max_detections,
        device=device
    )
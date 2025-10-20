"""
IATrax Team - 2025 - https://iatrax.com

LICENSE: MIT

METRICS MODULE FOR YOLO-ONE
"""

import torch
from typing import List, Optional, Tuple, Union, Dict


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Calculate IoU between two sets of boxes in xyxy format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    iou = inter / union.clamp(min=1e-6)
    return iou


def compute_ap(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """Compute Average Precision using 11-point interpolation."""
    mrec = torch.cat([torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)])
    mpre = torch.cat([torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)])

    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    i = torch.where(mrec[1:] != mrec[:-1])[0]

    ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap.item()


class YoloOneMetrics:
    """Comprehensive metrics for YOLO-One validation."""

    def __init__(self, device='cuda', conf_threshold: float = 0.001, iou_thresholds: Optional[List[float]] = None, max_det: int = 300):
        self.device = device
        self.conf_threshold = conf_threshold
        self.max_det = max_det  # Limite le nombre de détections pour éviter les calculs trop longs
        self.iou_thresholds = iou_thresholds if iou_thresholds else [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.reset()

    def reset(self):
        """Resets the state of the metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.inference_times = []

    def update(self, predictions: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
               targets: torch.Tensor, input_size: Tuple[int, int], **kwargs):
        """Update metrics with a batch of decoded predictions and targets."""
        # Extract decoded predictions
        if isinstance(predictions, dict):
            decoded_preds = predictions.get('decoded')
            if decoded_preds is None:
                return
        elif isinstance(predictions, list):
            decoded_preds = predictions
        else:
            return

        if not decoded_preds:
            return

        # Concatenate multi-scale predictions
        all_preds = [p.permute(0, 2, 3, 1).reshape(p.shape[0], -1, 5) for p in decoded_preds]
        batch_preds = torch.cat(all_preds, dim=1)

        # Extract image dimensions
        h, w = (input_size[0], input_size[1]) if isinstance(input_size, torch.Size) else input_size

        # Process each image in batch
        for i in range(batch_preds.shape[0]):
            pred = batch_preds[i]

            # Filter by confidence
            conf_mask = pred[:, 4] >= self.conf_threshold
            pred = pred[conf_mask]

            # Store predictions
            if pred.shape[0] > 0:
                # Limit predictions to max_det (keep top scoring)
                if pred.shape[0] > self.max_det:
                    top_k_indices = torch.topk(pred[:, 4], self.max_det).indices
                    pred = pred[top_k_indices]

                pred_boxes = pred[:, :4].clone()

                # Scale to pixels and clamp
                x1 = (pred_boxes[:, 0] * w).clamp(min=0, max=w)
                y1 = (pred_boxes[:, 1] * h).clamp(min=0, max=h)
                x2 = (pred_boxes[:, 2] * w).clamp(min=0, max=w)
                y2 = (pred_boxes[:, 3] * h).clamp(min=0, max=h)

                pred_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

                self.all_predictions.append({
                    'boxes': pred_boxes_xyxy,
                    'scores': pred[:, 4],
                })

            # Process targets
            target = targets[targets[:, 0] == i]

            if target.shape[0] > 0:
                target_boxes = target[:, 1:].clone()

                # Convert from cxcywh to xyxy
                cx = target_boxes[:, 0] * w
                cy = target_boxes[:, 1] * h
                box_w = target_boxes[:, 2] * w
                box_h = target_boxes[:, 3] * h

                x1 = (cx - box_w / 2).clamp(min=0, max=w)
                y1 = (cy - box_h / 2).clamp(min=0, max=h)
                x2 = (cx + box_w / 2).clamp(min=0, max=w)
                y2 = (cy + box_h / 2).clamp(min=0, max=h)

                target_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

                self.all_targets.append({'boxes': target_boxes_xyxy})

        if 'inference_time' in kwargs:
            self.inference_times.append(kwargs['inference_time'])

    def compute(self):
        """Compute the final metrics for the epoch."""
        if len(self.all_targets) == 0:
            return {'mAP': 0.0, 'mAP@0.5': 0.0, 'mAP@0.75': 0.0, 'avg_inference_time': 0.0}

        # Gather all predictions and targets
        all_pred_boxes = [pred['boxes'] for pred in self.all_predictions if len(pred['boxes']) > 0]
        all_pred_scores = [pred['scores'] for pred in self.all_predictions if len(pred['boxes']) > 0]
        all_target_boxes = [target['boxes'] for target in self.all_targets if len(target['boxes']) > 0]

        if not all_pred_boxes or not all_target_boxes:
            return {
                'mAP': 0.0,
                'mAP@0.5': 0.0,
                'mAP@0.75': 0.0,
                'avg_inference_time': sum(self.inference_times) / max(len(self.inference_times), 1)
            }

        # Concatenate
        pred_boxes = torch.cat(all_pred_boxes, dim=0)
        pred_scores = torch.cat(all_pred_scores, dim=0)
        target_boxes = torch.cat(all_target_boxes, dim=0)

        # Sort by confidence
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        # OPTIMISATION: Calcul vectorisé des IoU pour toutes les prédictions d'un coup
        ious = box_iou_xyxy(pred_boxes, target_boxes)  # Shape: [num_preds, num_targets]

        # Compute AP for each IoU threshold
        aps = []
        for iou_threshold in self.iou_thresholds:
            tp = torch.zeros(len(pred_boxes), device=self.device)
            fp = torch.zeros(len(pred_boxes), device=self.device)
            matched_targets = torch.zeros(len(target_boxes), dtype=torch.bool, device=self.device)

            # OPTIMISATION: Traitement vectorisé
            max_ious, max_indices = ious.max(dim=1)  # Meilleur IoU pour chaque prédiction

            for i in range(len(pred_boxes)):
                max_iou = max_ious[i]
                max_idx = max_indices[i]

                if max_iou >= iou_threshold and not matched_targets[max_idx]:
                    tp[i] = 1
                    matched_targets[max_idx] = True
                else:
                    fp[i] = 1

            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)

            recall = tp_cumsum / len(target_boxes)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-6)

            ap = compute_ap(recall, precision)
            aps.append(ap)

        map_value = sum(aps) / len(aps)
        map_50 = aps[0] if aps else 0.0
        map_75_idx = self.iou_thresholds.index(0.75) if 0.75 in self.iou_thresholds else -1
        map_75 = aps[map_75_idx] if map_75_idx >= 0 else 0.0

        return {
            'mAP': map_value,
            'mAP@0.5': map_50,
            'mAP@0.75': map_75,
            'avg_inference_time': sum(self.inference_times) / max(len(self.inference_times), 1)
        }

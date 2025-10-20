"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

YOLO-ONE LOSS MODULE - ANCHOR-FREE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class YoloOneLoss(nn.Module):
    """
    Anchor-free loss function optimized for YOLO-One single-class detection
    Direct regression without predefined anchor boxes
    """

    def __init__(
        self,
        box_weight: float = 5.0,
        obj_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        obj_neg_weight: float = 0.1,
        iou_type: str = 'iou',
        label_smoothing: float = 0.0,
    ):
        """
        Initialize the YOLO-One loss function

        Args:
            box_weight: Weight for bounding box regression loss
            obj_weight: Weight for objectness loss
            focal_alpha: Alpha value for focal loss
            focal_gamma: Gamma value for focal loss
            obj_neg_weight: Weight for objectness loss of negative samples
            iou_type: Type of IoU loss ('iou', 'giou', 'diou', 'ciou')
            label_smoothing: Label smoothing value
        """
        super().__init__()

        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.obj_neg_weight = obj_neg_weight
        self.iou_type = iou_type.lower()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute anchor-free YOLO-One loss

        Args:
            predictions: Dict with 'detections' key containing list of tensors [B, 5, H, W]
            targets: Ground truth annotations [N, 5] -> (batch_idx, cx, cy, w, h) normalized

        Returns:
            Dictionary with loss components
        """
        device = predictions['detections'][0].device

        box_losses = []
        obj_losses = []

        # Process each detection scale
        for scale_idx, detections in enumerate(predictions['detections']):
            batch_size, _, height, width = detections.shape

            # Build targets for current scale
            scale_targets, obj_mask, box_mask = self._build_targets(
                targets, (batch_size, height, width)
            )

            # Extract predictions
            pred_boxes = detections[:, :4]  # [B, 4, H, W] (x1, y1, x2, y2)
            pred_conf = detections[:, 4]     # [B, H, W]

            # Extract targets
            target_boxes = scale_targets[:, :, :, :4]  # [B, H, W, 4]
            target_conf = scale_targets[:, :, :, 4]    # [B, H, W]

            # Box loss (only where objects exist)
            if box_mask.sum() > 0:
                # Permute predictions to [B, H, W, 4]
                pred_boxes_hwc = pred_boxes.permute(0, 2, 3, 1)

                # Extract masked predictions and targets
                pred_boxes_masked = pred_boxes_hwc[box_mask]
                target_boxes_masked = target_boxes[box_mask]

                # Compute box loss
                scale_box_loss = self._compute_box_loss(
                    pred_boxes_masked,
                    target_boxes_masked
                )
            else:
                scale_box_loss = torch.tensor(0.0, device=device)

            # Objectness loss
            scale_obj_loss = self._compute_objectness_loss(
                pred_conf,
                target_conf,
                obj_mask
            )

            # Store UNWEIGHTED losses
            box_losses.append(scale_box_loss)
            obj_losses.append(scale_obj_loss)

        # Average across scales (pas de somme!)
        avg_box_loss = torch.stack(box_losses).mean() if len(box_losses) > 0 else torch.tensor(0.0, device=device)
        avg_obj_loss = torch.stack(obj_losses).mean() if len(obj_losses) > 0 else torch.tensor(0.0, device=device)

        # Apply weights ONCE
        weighted_box_loss = avg_box_loss * self.box_weight
        weighted_obj_loss = avg_obj_loss * self.obj_weight

        # Total loss
        total_loss = weighted_box_loss + weighted_obj_loss

        return {
            'total_loss': total_loss,
            'box_loss': avg_box_loss,      # Unweighted for logging
            'obj_loss': avg_obj_loss,      # Unweighted for logging
            'avg_loss': total_loss.item()
        }

    def _build_targets(
        self,
        targets: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build targets for anchor-free detection

        Args:
            targets: [N, 5] (batch_idx, cx, cy, w, h) normalized [0, 1]
            grid_shape: (batch_size, grid_h, grid_w)

        Returns:
            target_tensor: [B, H, W, 5] (x1, y1, x2, y2, conf)
            obj_mask: [B, H, W] boolean mask for objectness
            box_mask: [B, H, W] boolean mask for box regression
        """
        batch_size, grid_h, grid_w = grid_shape
        device = targets.device

        # Initialize tensors
        target_tensor = torch.zeros(batch_size, grid_h, grid_w, 5, device=device)
        obj_mask = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.bool, device=device)
        box_mask = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.bool, device=device)

        if targets.size(0) == 0:
            return target_tensor, obj_mask, box_mask

        # Process each target
        for target in targets:
            batch_idx = int(target[0])
            if batch_idx >= batch_size:
                continue

            # Extract normalized cxcywh [0, 1]
            cx, cy, w, h = target[1:5]

            # Find responsible grid cell
            grid_x = (cx * grid_w).clamp(0, grid_w - 1)
            grid_y = (cy * grid_h).clamp(0, grid_h - 1)

            grid_i = int(grid_x)
            grid_j = int(grid_y)

            # Convert cxcywh to x1y1x2y2 (normalized)
            x1 = (cx - w / 2).clamp(0, 1)
            y1 = (cy - h / 2).clamp(0, 1)
            x2 = (cx + w / 2).clamp(0, 1)
            y2 = (cy + h / 2).clamp(0, 1)

            # Store target
            target_tensor[batch_idx, grid_j, grid_i, 0] = x1
            target_tensor[batch_idx, grid_j, grid_i, 1] = y1
            target_tensor[batch_idx, grid_j, grid_i, 2] = x2
            target_tensor[batch_idx, grid_j, grid_i, 3] = y2
            target_tensor[batch_idx, grid_j, grid_i, 4] = 1.0

            # Set masks
            obj_mask[batch_idx, grid_j, grid_i] = True
            box_mask[batch_idx, grid_j, grid_i] = True

        return target_tensor, obj_mask, box_mask

    def _compute_box_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bounding box regression loss

        Args:
            pred_boxes: [N, 4] raw predictions (x1, y1, x2, y2)
            target_boxes: [N, 4] targets (x1, y1, x2, y2) normalized [0, 1]

        Returns:
            Box loss value (scalar)
        """
        if pred_boxes.size(0) == 0:
            return torch.tensor(0.0, device=pred_boxes.device)

        # Normalize predictions with sigmoid
        pred_boxes_norm = torch.sigmoid(pred_boxes)

        # Ensure valid boxes
        pred_boxes_norm = self._ensure_valid_boxes(pred_boxes_norm)
        target_boxes = self._ensure_valid_boxes(target_boxes)

        # Compute IoU-based loss
        if self.iou_type == 'giou':
            iou = self._box_giou(pred_boxes_norm, target_boxes)
        elif self.iou_type == 'diou':
            iou = self._box_diou(pred_boxes_norm, target_boxes)
        elif self.iou_type == 'ciou':
            iou = self._box_ciou(pred_boxes_norm, target_boxes)
        else:  # 'iou'
            iou = self._box_iou(pred_boxes_norm, target_boxes)

        # IoU loss: 1 - IoU ∈ [0, 1] pour IoU, [0, 2] pour GIoU/DIoU/CIoU
        loss = 1.0 - iou

        return loss.mean()

    def _ensure_valid_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Ensure boxes have valid coordinates (x2 > x1, y2 > y1)

        Args:
            boxes: [N, 4] (x1, y1, x2, y2)

        Returns:
            Valid boxes
        """
        x1, y1, x2, y2 = boxes.unbind(dim=-1)

        # Ensure proper ordering
        x_min = torch.min(x1, x2)
        x_max = torch.max(x1, x2)
        y_min = torch.min(y1, y2)
        y_max = torch.max(y1, y2)

        # Add epsilon to ensure non-zero width/height
        x_max = torch.maximum(x_max, x_min + 1e-6)
        y_max = torch.maximum(y_max, y_min + 1e-6)

        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    def _compute_objectness_loss(
        self,
        pred_conf: torch.Tensor,
        target_conf: torch.Tensor,
        obj_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute objectness loss with focal loss

        Args:
            pred_conf: [B, H, W] predicted confidence (logits)
            target_conf: [B, H, W] target confidence [0, 1]
            obj_mask: [B, H, W] boolean mask

        Returns:
            Objectness loss (scalar)
        """
        # Apply label smoothing
        target_conf_smooth = target_conf.clone()
        if self.label_smoothing > 0:
            target_conf_smooth = target_conf * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Binary cross-entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_conf,
            target_conf_smooth,
            reduction='none'
        )

        # Apply focal loss modulation
        if self.focal_gamma > 0:
            with torch.no_grad():
                pred_prob = torch.sigmoid(pred_conf)
                p_t = target_conf * pred_prob + (1 - target_conf) * (1 - pred_prob)
                focal_weight = (1 - p_t.clamp(min=1e-7)) ** self.focal_gamma
                alpha_t = target_conf * self.focal_alpha + (1 - target_conf) * (1 - self.focal_alpha)
                modulation = alpha_t * focal_weight

            bce_loss = modulation * bce_loss

        # Separate positive and negative samples
        pos_mask = obj_mask
        neg_mask = ~pos_mask

        # Compute losses
        pos_loss = bce_loss[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=pred_conf.device)
        neg_loss = bce_loss[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=pred_conf.device)

        return pos_loss + self.obj_neg_weight * neg_loss

    # ==================== IoU Variants ====================

    def _box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calculate standard IoU

        Args:
            box1, box2: [N, 4] in (x1, y1, x2, y2) format

        Returns:
            IoU values [N] in range [0, 1]
        """
        # Intersection area
        lt = torch.max(box1[:, :2], box2[:, :2])
        rb = torch.min(box1[:, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        # Union area
        area1 = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])).clamp(min=1e-7)
        area2 = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])).clamp(min=1e-7)
        union = (area1 + area2 - inter).clamp(min=1e-7)

        iou = inter / union
        return iou.clamp(min=0.0, max=1.0)

    def _box_giou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Generalized IoU (GIoU)

        Returns:
            GIoU values [N] in range [-1, 1]
        """
        # Standard IoU
        iou = self._box_iou(box1, box2)

        # Smallest enclosing box
        enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
        enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
        enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
        enclose_y2 = torch.max(box1[:, 3], box2[:, 3])

        enclose_area = ((enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)).clamp(min=1e-7)

        # Union area
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        lt = torch.max(box1[:, :2], box2[:, :2])
        rb = torch.min(box1[:, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        union = (area1 + area2 - inter).clamp(min=1e-7)

        # GIoU
        giou = iou - (enclose_area - union) / enclose_area
        return giou.clamp(min=-1.0, max=1.0)

    def _box_diou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Distance IoU (DIoU)

        Returns:
            DIoU values [N] in range [-1, 1]
        """
        # Standard IoU
        iou = self._box_iou(box1, box2)

        # Center coordinates
        cx1 = (box1[:, 0] + box1[:, 2]) / 2
        cy1 = (box1[:, 1] + box1[:, 3]) / 2
        cx2 = (box2[:, 0] + box2[:, 2]) / 2
        cy2 = (box2[:, 1] + box2[:, 3]) / 2

        # Center distance squared
        center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

        # Diagonal of smallest enclosing box
        enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
        enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
        enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
        enclose_y2 = torch.max(box1[:, 3], box2[:, 3])

        diagonal_sq = ((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2).clamp(min=1e-7)

        # DIoU
        diou = iou - center_dist_sq / diagonal_sq
        return diou.clamp(min=-1.0, max=1.0)

    def _box_ciou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Complete IoU (CIoU)

        Returns:
            CIoU values [N] in range [-1, 1]
        """
        # DIoU components
        iou = self._box_iou(box1, box2)

        # Center coordinates
        cx1 = (box1[:, 0] + box1[:, 2]) / 2
        cy1 = (box1[:, 1] + box1[:, 3]) / 2
        cx2 = (box2[:, 0] + box2[:, 2]) / 2
        cy2 = (box2[:, 1] + box2[:, 3]) / 2

        center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

        # Enclosing box diagonal
        enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
        enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
        enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
        enclose_y2 = torch.max(box1[:, 3], box2[:, 3])

        diagonal_sq = ((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2).clamp(min=1e-7)

        # Aspect ratio consistency
        w1 = (box1[:, 2] - box1[:, 0]).clamp(min=1e-6)
        h1 = (box1[:, 3] - box1[:, 1]).clamp(min=1e-6)
        w2 = (box2[:, 2] - box2[:, 0]).clamp(min=1e-6)
        h2 = (box2[:, 3] - box2[:, 1]).clamp(min=1e-6)

        v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        # Alpha parameter
        with torch.no_grad():
            alpha = v / ((1 - iou + v).clamp(min=1e-7))

        # CIoU
        ciou = iou - center_dist_sq / diagonal_sq - alpha * v
        return ciou.clamp(min=-1.0, max=1.0)


# ==================== Factory Function ====================

def create_yolo_one_loss(
    box_weight: float = 5.0,
    obj_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    obj_neg_weight: float = 0.1,
    iou_type: str = 'iou',
    label_smoothing: float = 0.0,
) -> YoloOneLoss:
    """
    Factory function to create YOLO-One loss

    Args:
        box_weight: Weight for box loss (default: 5.0)
        obj_weight: Weight for objectness loss (default: 1.0)
        focal_alpha: Focal loss alpha (default: 0.25)
        focal_gamma: Focal loss gamma (default: 2.0)
        obj_neg_weight: Weight for negative objectness (default: 0.1)
        iou_type: Type of IoU ('iou', 'giou', 'diou', 'ciou') (default: 'iou')
        label_smoothing: Label smoothing factor (default: 0.0)

    Returns:
        YoloOneLoss instance
    """
    return YoloOneLoss(
        box_weight=box_weight,
        obj_weight=obj_weight,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        obj_neg_weight=obj_neg_weight,
        iou_type=iou_type,
        label_smoothing=label_smoothing,
    )

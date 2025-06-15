"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

YOLO-ONE LOSS MODULE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Optional
class YoloOneLoss(nn.Module):
    """
    Loss function optimized for YOLO-One single-class detection
    Compatible with backbone + spatial attention architecture
    """
    
    def __init__(
        self,
        box_weight: float = 7.5,
        obj_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 1.5,
        iou_type: str = 'ciou',
        label_smoothing: float = 0.0,
        p5_weight_boost: float = 1.2
    ):
        super().__init__()
        
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.iou_type = iou_type
        self.label_smoothing = label_smoothing
        self.p5_weight_boost = p5_weight_boost
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Anchors for 3 scales: P3, P4, P5
        self.register_buffer('anchors', torch.tensor([
            [[10, 13], [16, 30], [33, 23]],      # P3: 80x80, small objects
            [[30, 61], [62, 45], [59, 119]],     # P4: 40x40, medium objects
            [[116, 90], [156, 198], [373, 326]]  # P5: 20x20, large objects + spatial attention
        ], dtype=torch.float32))
    
    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute YOLO-One loss with spatial attention support
        
        Args:
            predictions: [P3, P4, P5] features from backbone
            targets: Ground truth annotations [batch_idx, class, x, y, w, h]
            model: Model for info extraction (optional)
            
        Returns:
            Dictionary with loss components
        """
        
        device = predictions[0].device
        
        # Initialize losses
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        
        # Scale information
        scales = [
            {'stride': 8, 'size': 80},   # P3
            {'stride': 16, 'size': 40},  # P4
            {'stride': 32, 'size': 20}   # P5 with spatial attention
        ]
        
        # Process each scale
        for scale_idx, (pred, scale_info) in enumerate(zip(predictions, scales)):
            
            # Reshape predictions to YOLO format
            batch_size, channels, height, width = pred.shape
            num_anchors = 3  # Always 3 anchors per scale
            
            # Reshape: [B, 3*5, H, W] -> [B, 3, H, W, 5]
            pred = pred.view(batch_size, num_anchors, 5, height, width)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Build targets for current scale
            scale_targets, obj_mask, box_mask = self._build_targets(
                targets, pred.shape, self.anchors[scale_idx], 
                height, width, scale_info['stride']
            )
            
            # Extract predictions
            pred_boxes = pred[..., :4]  # [x, y, w, h]
            pred_conf = pred[..., 4]    # fused confidence
            
            # Extract targets
            target_boxes = scale_targets[..., :4]
            target_conf = scale_targets[..., 4]
            
            # Box loss (only where objects exist)
            if box_mask.sum() > 0:
                pred_boxes_masked = pred_boxes[box_mask]
                target_boxes_masked = target_boxes[box_mask]
                
                box_loss = self._compute_box_loss(
                    pred_boxes_masked, target_boxes_masked, 
                    scale_info['stride'], height, width
                )
                
                # Boost P5 loss (spatial attention enhances P5)
                weight = self.box_weight
                if scale_idx == 2:  # P5 scale
                    weight *= self.p5_weight_boost
                    
                loss_box += box_loss * weight
            
            # Objectness loss
            obj_loss = self._compute_objectness_loss(pred_conf, target_conf, obj_mask)
            loss_obj += obj_loss * self.obj_weight
        
        # Total loss
        total_loss = loss_box + loss_obj
        
        return {
            'total_loss': total_loss,
            'box_loss': loss_box,
            'obj_loss': loss_obj,
            'avg_loss': total_loss.item()
        }
    
    def _build_targets(
        self, 
        targets: torch.Tensor,
        pred_shape: Tuple[int, ...],
        anchors: torch.Tensor,
        grid_h: int,
        grid_w: int,
        stride: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build targets for current scale
        Optimized for single-class YOLO-One
        """
        
        batch_size, num_anchors = pred_shape[:2]
        device = targets.device
        
        # Initialize target tensors
        target_tensor = torch.zeros(batch_size, num_anchors, grid_h, grid_w, 5, device=device)
        obj_mask = torch.zeros(batch_size, num_anchors, grid_h, grid_w, dtype=torch.bool, device=device)
        box_mask = torch.zeros(batch_size, num_anchors, grid_h, grid_w, dtype=torch.bool, device=device)
        
        if targets.size(0) == 0:
            return target_tensor, obj_mask, box_mask
        
        # Process each target
        for target in targets:
            batch_idx = int(target[0])
            if batch_idx >= batch_size:
                continue
            
            # Extract target info (class always 0 for single-class)
            x_center, y_center, width, height = target[2:6]
            
            # Convert to grid coordinates
            grid_x = x_center * grid_w
            grid_y = y_center * grid_h
            
            grid_i = int(grid_x.clamp(0, grid_w - 1))
            grid_j = int(grid_y.clamp(0, grid_h - 1))
            
            # Find best anchor match
            target_wh = torch.tensor([width, height], device=device) * torch.tensor([grid_w, grid_h], device=device)
            anchor_ious = self._compute_anchor_ious(target_wh, anchors)
            best_anchor = torch.argmax(anchor_ious)
            
            # Set target (relative to cell)
            target_tensor[batch_idx, best_anchor, grid_j, grid_i, 0] = grid_x - grid_i  # dx
            target_tensor[batch_idx, best_anchor, grid_j, grid_i, 1] = grid_y - grid_j  # dy
            target_tensor[batch_idx, best_anchor, grid_j, grid_i, 2] = width * grid_w   # dw
            target_tensor[batch_idx, best_anchor, grid_j, grid_i, 3] = height * grid_h  # dh
            target_tensor[batch_idx, best_anchor, grid_j, grid_i, 4] = 1.0  # confidence
            
            obj_mask[batch_idx, best_anchor, grid_j, grid_i] = True
            box_mask[batch_idx, best_anchor, grid_j, grid_i] = True
        
        return target_tensor, obj_mask, box_mask
    
    def _compute_box_loss(
        self, 
        pred_boxes: torch.Tensor, 
        target_boxes: torch.Tensor,
        stride: int,
        grid_h: int,
        grid_w: int
    ) -> torch.Tensor:
        """Box loss with CIoU optimized for YOLO-One"""
        
        # Convert predictions to absolute format
        pred_xy = torch.sigmoid(pred_boxes[..., :2])
        pred_wh = pred_boxes[..., 2:4]
        
        target_xy = target_boxes[..., :2]
        target_wh = target_boxes[..., 2:4]
        
        # CIoU loss
        if self.iou_type == 'ciou':
            loss = self._ciou_loss(
                torch.cat([pred_xy, pred_wh], dim=-1),
                torch.cat([target_xy, target_wh], dim=-1)
            )
        else:
            # Fallback to MSE
            loss = F.mse_loss(pred_xy, target_xy) + F.mse_loss(pred_wh, target_wh)
        
        return loss.mean()
    
    def _compute_objectness_loss(
        self,
        pred_conf: torch.Tensor,
        target_conf: torch.Tensor, 
        obj_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Objectness loss with focal loss for single-class
        No separate classification needed
        """
        
        # Label smoothing if specified
        if self.label_smoothing > 0:
            target_conf = target_conf * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # BCE loss
        bce_loss = self.bce_loss(pred_conf, target_conf)
        
        # Focal loss for hard examples
        if self.focal_gamma > 0:
            pred_prob = torch.sigmoid(pred_conf)
            pt = target_conf * pred_prob + (1 - target_conf) * (1 - pred_prob)
            focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma
            bce_loss = focal_weight * bce_loss
        
        # Balance positive/negative samples
        pos_mask = obj_mask
        neg_mask = ~obj_mask
        
        pos_loss = bce_loss[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=pred_conf.device)
        neg_loss = bce_loss[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=pred_conf.device)
        
        # Reduce negative loss weight (background)
        return pos_loss + 0.05 * neg_loss
    
    def _ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Complete IoU loss implementation"""
        
        # Convert to corner format
        pred_x1, pred_y1, pred_x2, pred_y2 = self._xywh_to_xyxy(pred_boxes)
        target_x1, target_y1, target_x2, target_y2 = self._xywh_to_xyxy(target_boxes)
        
        # Intersection area
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union area
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        
        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        enclose_c2 = enclose_w ** 2 + enclose_h ** 2
        
        # Center distance
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        target_cx = (target_x1 + target_x2) / 2
        target_cy = (target_y1 + target_y2) / 2
        
        rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Aspect ratio consistency
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1
        target_w = target_x2 - target_x1
        target_h = target_y2 - target_y1
        
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_w / torch.clamp(target_h, min=1e-6)) - 
            torch.atan(pred_w / torch.clamp(pred_h, min=1e-6)), 2
        )
        
        alpha = v / torch.clamp(1 - iou + v, min=1e-6)
        
        # CIoU
        ciou = iou - rho2 / torch.clamp(enclose_c2, min=1e-6) - alpha * v
        
        return 1 - ciou
    
    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Convert center format to corner format"""
        x, y, w, h = boxes.unbind(-1)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2
    
    def _compute_anchor_ious(self, target_wh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Compute IoU between target and anchors"""
        target_area = target_wh[0] * target_wh[1]
        anchor_areas = anchors[:, 0] * anchors[:, 1]
        
        inter_w = torch.min(target_wh[0], anchors[:, 0])
        inter_h = torch.min(target_wh[1], anchors[:, 1])
        inter_area = inter_w * inter_h
        
        union_area = target_area + anchor_areas - inter_area
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        
        return iou


def create_yolo_one_loss(
    box_weight: float = 7.5,
    obj_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 1.5,
    iou_type: str = 'ciou',
    label_smoothing: float = 0.0,
    p5_weight_boost: float = 1.2
) -> YoloOneLoss:
    """
    Factory function to create YOLO-One loss with specified parameters
    
    Args:
        box_weight: Weight for box regression loss
        obj_weight: Weight for objectness loss
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        iou_type: Type of IoU loss ('ciou', 'diou', 'iou')
        label_smoothing: Label smoothing factor
        p5_weight_boost: Weight boost for P5 scale (spatial attention)
        
    Returns:
        Configured YoloOneLoss instance
    """
    return YoloOneLoss(
        box_weight=box_weight,
        obj_weight=obj_weight,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        iou_type=iou_type,
        label_smoothing=label_smoothing,
        p5_weight_boost=p5_weight_boost
    )
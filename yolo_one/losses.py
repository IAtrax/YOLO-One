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
        box_weight: float = 7.5,
        obj_weight: float = 1.0,
        focal_alpha: float = 0.25,
        moe_balance_weight: float = 0.001,
        focal_gamma: float = 1.5,
        obj_neg_weight: float = 0.05,
        iou_type: str = 'iou',
        label_smoothing: float = 0.0,
        focal_loss: bool = True,
    ):
        """
        Initialize the YOLO-One loss function

        Args:
            box_weight (float, optional): Weight for bounding box regression loss. Defaults to 7.5.
            obj_weight (float, optional): Weight for objectness loss. Defaults to 1.0.
            focal_alpha (float, optional): Alpha value for focal loss. Defaults to 0.25.
            moe_balance_weight (float, optional): Weight for Mixture of Experts (MoE) balancing loss. Defaults to 0.001.
            focal_gamma (float, optional): Gamma value for focal loss. Defaults to 1.5.
            obj_neg_weight (float, optional): Weight for objectness loss of negative samples. Defaults to 0.05.
            iou_type (str, optional): Type of IoU loss to use, one of 'meiou', 'iou', 'giou', or 'diou'. Defaults to 'meiou'.
            label_smoothing (float, optional): Label smoothing value. Defaults to 0.0.
        """
        super().__init__()
        
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.moe_balance_weight = moe_balance_weight
        self.obj_neg_weight = obj_neg_weight
        self.iou_type = iou_type
        self.label_smoothing = label_smoothing
        self.focal_loss = focal_loss
        
    
    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute anchor-free YOLO-One loss
        
        Args:
            predictions: Dict with 'detections', 'aspects', 'shape_confidences'
            targets: Ground truth annotations [batch_idx, x1, y1, x2, y2]
            
        Returns:
            Dictionary with loss components
        """
        
        device = predictions['detections'][0].device
        
        # MoE Load Balancing Loss
        gate_scores = predictions.get('gate_scores')
        loss_moe_balance = torch.zeros(1, device=device)
        if self.moe_balance_weight > 0 and gate_scores is not None:
            num_experts = gate_scores.shape[1]
            load_per_expert = gate_scores.sum(dim=0)
            loss_moe_balance = (load_per_expert.var() / (load_per_expert.mean()**2 + 1e-8)) * num_experts

        # Per-Expert Loss Calculation
        box_losses_per_expert = []
        obj_losses_per_expert = []

        scales = [
            {'stride': 8, 'size': 80},   # P3
            {'stride': 16, 'size': 40},  # P4
            {'stride': 32, 'size': 20}   # P5
        ]
        
        # Process each scale
        for scale_idx, _ in enumerate(scales):
            
            detections = predictions['detections'][scale_idx]
            batch_size, _, height, width = detections.shape
            
            # Build targets for current scale
            scale_targets, obj_mask, box_mask = self._build_anchor_free_targets(
                targets, (batch_size, height, width)
            )
            
            # Extract predictions
            pred_boxes = detections[:, :4]  # [B, 4, H, W]
            pred_conf = detections[:, 4]    # [B, H, W]
            
            # Extract targets
            target_boxes = scale_targets[:, :, :, :4]  # [B, H, W, 4]
            target_conf = scale_targets[:, :, :, 4]    # [B, H, W]
            
            scale_box_loss = torch.zeros(1, device=device)
            scale_obj_loss = torch.zeros(1, device=device)

            # Box loss (only where objects exist)
            if box_mask.sum() > 0:
                pred_boxes_hwc = pred_boxes.permute(0, 2, 3, 1)  # [B, H, W, 4]
                
                pred_boxes_masked = pred_boxes_hwc[box_mask]  # [num_objects, 4]
                target_boxes_masked = target_boxes[box_mask]  # [num_objects, 4]
                
                scale_box_loss = self._compute_anchor_free_box_loss(
                    pred_boxes_masked, target_boxes_masked
                )
                
            # Objectness loss for this scale
            scale_obj_loss = self._compute_objectness_loss(pred_conf, target_conf, obj_mask)

            # Apply weights
            box_losses_per_expert.append(scale_box_loss )
            obj_losses_per_expert.append(scale_obj_loss)

        # Combine Losses
        if gate_scores is not None:
            expert_weights = gate_scores.mean(dim=0)  # [num_experts]
            
            total_box_losses = torch.stack(box_losses_per_expert)
            total_obj_losses = torch.stack(obj_losses_per_expert)
            
            # loss_box = (total_box_losses * expert_weights).sum()
            # loss_obj = (total_obj_losses * expert_weights).sum()
            loss_box = (total_box_losses).sum()
            loss_obj = (total_obj_losses).sum()
        else:
            loss_box = torch.sum(torch.stack(box_losses_per_expert))
            loss_obj = torch.sum(torch.stack(obj_losses_per_expert))

        total_loss = loss_box + loss_obj # + (loss_moe_balance * self.moe_balance_weight)
        
        return {
            'total_loss': total_loss,
            'box_loss': loss_box,
            'obj_loss': loss_obj,
            'moe_balance_loss': loss_moe_balance,
            'avg_loss': total_loss.item()
        }
    
    def _build_anchor_free_targets(
        self, 
        targets: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build targets for anchor-free detection with corner coordinates
        """
        
        batch_size, grid_h, grid_w = grid_shape
        device = targets.device
        
        # Initialize target tensors
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
                
            x1, y1, x2, y2 = target[1:]
            
            # Calculate center to determine which grid cell
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            
            # Convert to grid coordinates
            grid_x = x_center * grid_w
            grid_y = y_center * grid_h
            
            grid_i = int(grid_x.clamp(0, grid_w - 1))
            grid_j = int(grid_y.clamp(0, grid_h - 1))
            
            # Store corner coordinates directly (keeping them normalized)
            target_tensor[batch_idx, grid_j, grid_i, 0] = x1
            target_tensor[batch_idx, grid_j, grid_i, 1] = y1
            target_tensor[batch_idx, grid_j, grid_i, 2] = x2
            target_tensor[batch_idx, grid_j, grid_i, 3] = y2
            target_tensor[batch_idx, grid_j, grid_i, 4] = 1.0  # confidence
            
            # Set mask     
            obj_mask[batch_idx, grid_j, grid_i] = True
            box_mask[batch_idx, grid_j, grid_i] = True
        
        return target_tensor, obj_mask, box_mask
    
    def _compute_anchor_free_box_loss(self, pred_boxes, target_boxes):
       

        # Apply sigmoid to normalize to [0,1]
        """
        Compute the anchor-free box loss.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes [B, 4] (x1, y1, x2, y2)
            target_boxes (torch.Tensor): Target boxes [B, 4] (x1, y1, x2, y2)

        Returns:
            torch.Tensor: The loss value
        """
        pred_boxes_norm = torch.sigmoid(pred_boxes)
        if self.iou_type == 'iou':
            iou = self._iou_loss(pred_boxes_norm, target_boxes)
            loss = 1 - iou
    
        if self.iou_type == 'ciou':
            iou = self._ciou_loss(pred_boxes_norm, target_boxes)
            loss = 1 - iou
        elif self.iou_type == 'meiou':
            iou = self._meiou_loss(pred_boxes_norm, target_boxes)
            loss = 1 - iou
        else:
            # Simple MSE
            loss = F.mse_loss(pred_boxes_norm, target_boxes)

        return loss.mean()
    
    
    def _compute_objectness_loss(
        self,
        pred_conf: torch.Tensor,      # Preds [-inf, inf]
        target_conf: torch.Tensor,    # Labels [0,1]
        obj_mask: torch.Tensor        
    ) -> torch.Tensor:
        
        """
        Compute objectness loss (Binary Cross-Entropy with logits)
        with optional label smoothing, focal loss, and balance between positive/negative samples
        
        Args:
            pred_conf (torch.Tensor): Predicted confidence [-inf, inf]
            target_conf (torch.Tensor): Labels [0,1]
            obj_mask (torch.Tensor): Object mask
        
        Returns:
            torch.Tensor: Objectness loss
        """
        if self.label_smoothing > 0:
            target_conf = target_conf * (1 - self.label_smoothing) + \
                        0.5 * self.label_smoothing
        
        # BCE(with logits!)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_conf, target_conf, reduction='none'
        )
        
        if self.focal_gamma > 0:
            with torch.no_grad():
                pred_prob = torch.sigmoid(pred_conf)
                p_t = target_conf * pred_prob + (1 - target_conf) * (1 - pred_prob)
                # Focal weight
                focal_weight = (1 - p_t) ** self.focal_gamma                
                alpha_t = target_conf * self.focal_alpha + \
                        (1 - target_conf) * (1 - self.focal_alpha)
                
                modulation = alpha_t * focal_weight
            
            # apply modulation
            bce_loss = modulation * bce_loss
        
        # Balance pos/neg samples
        pos_mask = obj_mask.bool()
        neg_mask = ~pos_mask
        
        pos_loss = bce_loss[pos_mask].mean() if pos_mask.sum() > 0 \
                else torch.tensor(0.0, device=pred_conf.device)
        neg_loss = bce_loss[neg_mask].mean() if neg_mask.sum() > 0 \
                else torch.tensor(0.0, device=pred_conf.device)
        
        return pos_loss + self.obj_neg_weight * neg_loss
    
    def _iou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Simple IoU loss implementation for anchor-free"""
        # Intersection area
        inter_area = torch.min(pred_boxes[:, 2], target_boxes[:, 2]) * torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        # Union area
        union_area = pred_boxes[:, 2] * pred_boxes[:, 3] + target_boxes[:, 2] * target_boxes[:, 3] - inter_area
        
        # IoU
        iou = inter_area / union_area
        
        return iou
    

    def _ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Complete IoU loss implementation for anchor-free"""
        
        # Convert to corner format
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(-1)
        
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
    
    def _eiou_loss(
                    self,
                    pred_boxes: torch.Tensor, 
                    target_boxes: torch.Tensor, 
                    )-> torch.Tensor:
        
        px1, py1, px2, py2 = pred_boxes.unbind(-1)
        tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)

        inter_x1, inter_y1 = torch.max(px1, tx1), torch.max(py1, ty1)
        inter_x2, inter_y2 = torch.min(px2, tx2), torch.min(py2, ty2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        pred_area = torch.clamp(px2 - px1, min=0) * torch.clamp(py2 - py1, min=0)
        target_area = torch.clamp(tx2 - tx1, min=0) * torch.clamp(ty2 - ty1, min=0)
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-6)

        ex1, ey1 = torch.min(px1, tx1), torch.min(py1, ty1)
        ex2, ey2 = torch.max(px2, tx2), torch.max(py2, ty2)
        ew, eh = ex2 - ex1, ey2 - ey1
        c2 = ew**2 + eh**2 

        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        rho2 = (pcx - tcx)**2 + (pcy - tcy)**2

        pw, ph = px2 - px1, py2 - py1
        tw, th = tx2 - tx1, ty2 - ty1
        wc2, hc2 = ew**2, eh**2 
        rho2_w = (pw - tw) ** 2
        rho2_h = (ph - th) ** 2

        eiou = iou - rho2 / (c2+ 1e-6) - rho2_w / (wc2 + 1e-6) - rho2_h / (hc2 + 1e-6)
        return (iou**self.focal_gamma)*(1 - eiou)

    def _meiou_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        
        """More efficient IoU loss implementation for anchor-free"""

        # Pred and target boxes do not contain None 
        assert pred_boxes is not None, "pred_boxes must not be None"
        assert target_boxes is not None, "target_boxes must not be None" 


        # Pred and target boxes must be Tensor
        assert isinstance(pred_boxes, torch.Tensor), f"pred_boxes must be a torch.Tensor type, got {type(pred_boxes)}"
        assert isinstance(target_boxes, torch.Tensor), f"target_boxes must be a torch.Tensor type, got {type(target_boxes)}"

        # Pred and target boxes must have the same shape
        #assert len(pred_boxes) == len(target_boxes), f"Length mismatch: len(pred_boxes)={len(pred_boxes)}, len(target_boxes)={len(target_boxes)}"

        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(-1)

        # Intersection
        inter_x1, inter_y1 = torch.max(pred_x1, target_x1), torch.max(pred_y1, target_y1)
        inter_x2, inter_y2 = torch.min(pred_x2, target_x2), torch.min(pred_y2, target_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Union
        pred_area = torch.clamp(pred_x2 - pred_x1, min=0) * torch.clamp(pred_y2 - pred_y1, min=0)
        target_area = torch.clamp(target_x2 - target_x1, min=0) * torch.clamp(target_y2 - target_y1, min=0)
        union_area = pred_area + target_area - inter_area
        iou = inter_area / torch.clamp(union_area, min=1e-6)


        # Enclosing box 
        enclosed_x1, enclosed_y1 = torch.min(pred_x1, target_x1), torch.min(pred_y1, target_y1)
        enclosed_x2, enclosed_y2 = torch.max(pred_x2, target_x2), torch.max(pred_y2, target_y2)
        enclosed_w = enclosed_x2 - enclosed_x1
        enclosed_h = enclosed_y2 - enclosed_y1
        enclosed_2 = enclosed_w**2 + enclosed_h**2 

        # Center distance 
        pred_center_x, pred_center_y = (pred_x1 + pred_x2) / 2, (pred_y1 + pred_y2) / 2
        target_center_x, target_center_y = (target_x1 + target_x2) / 2, (target_y1 + target_y2) / 2
        dist_center_2 = (pred_center_x - target_center_x)**2 + (pred_center_y - target_center_y)**2


        # Width/Height loss
        pred_w, pred_h = pred_x2 - pred_x1, pred_y2 - pred_y1
        target_w, target_h = target_x2 - target_x1, target_y2 - target_y1
        # Fix box size : the box size practically does not exceed 99% of the entire image size
        #pred_w, pred_h = torch.clamp(pred_w, max=0.99*width_img), torch.clamp(pred_h, max=0.99*height_img)
        rho2_w = (pred_w - target_w) ** 2
        rho2_h = (pred_h - target_h) ** 2 
        height_width_loss = (rho2_h / enclosed_h)  +  (rho2_w / enclosed_w)

        # Angle cost
        ch = torch.max(pred_center_y, target_center_y) - torch.min(pred_center_y, target_center_y)
        sigma = torch.sqrt(torch.abs(dist_center_2))
        delta_angle_loss = torch.where(sigma > 1e-6, 
                                       1 - 2 * torch.pow(torch.sin(torch.arcsin(ch / sigma) - torch.pi / 4), 2),
                                        torch.zeros_like(sigma))

        # MEIoU Loss
        eiou_loss  =  1 - iou + (dist_center_2 / enclosed_2) + height_width_loss
        meiou_loss = eiou_loss + delta_angle_loss

        # Focal
        if self.focal_loss:
            meiou_loss = (iou**self.focal_gamma)*(meiou_loss)

        # Mask invalid boxes (sum==0)
        valid_mask = (target_boxes.sum(-1) > 0) & (pred_boxes.sum(-1) > 0)

        if valid_mask.any():
            meiou_loss = meiou_loss[valid_mask].mean()
        else:
            meiou_loss = torch.tensor(0., device=pred_boxes.device)

        return meiou_loss

    
    

def create_yolo_one_loss(
    box_weight: float = 7.5,
    obj_weight: float = 1.0,
    focal_alpha: float = 0.25,
    moe_balance_weight: float = 0.001,
    focal_gamma: float = 1.5,
    obj_neg_weight: float = 0.05,
    iou_type: str = 'meiou',
    label_smoothing: float = 0.0,
) -> YoloOneLoss:
    """Factory function to create anchor-free YOLO-One loss"""
    return YoloOneLoss(
        box_weight=box_weight,
        obj_weight=obj_weight,
        focal_alpha=focal_alpha,
        moe_balance_weight=moe_balance_weight,
        focal_gamma=focal_gamma,
        obj_neg_weight=obj_neg_weight,
        iou_type=iou_type,
        label_smoothing=label_smoothing,
    )

"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

YOLO-ONE LOSS MODULE - ANCHOR-FREE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from yolo_one.utils.general import box_cxcywh_to_xyxy
from typing import List, Tuple, Dict, Optional

class YoloOneLoss(nn.Module):
    """
    Anchor-free loss function optimized for YOLO-One single-class detection
    Direct regression without predefined anchor boxes
    """
    
    def __init__(
        self,
        box_weight: float = 7.5,
        obj_weight: float = 1.0,
        aspect_weight: float = 0.5,
        shape_conf_weight: float = 0.2,
        focal_alpha: float = 0.25,
        moe_balance_weight: float = 0.05,
        focal_gamma: float = 1.5,
        obj_neg_weight: float = 0.05,
        iou_type: str = 'meiou',
        label_smoothing: float = 0.0,
        p5_weight_boost: float = 1.2
    ):
        super().__init__()
        
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.aspect_weight = aspect_weight
        self.shape_conf_weight = shape_conf_weight
        self.focal_alpha = focal_alpha
        self.moe_balance_weight = moe_balance_weight
        self.focal_gamma = focal_gamma
        self.obj_neg_weight = obj_neg_weight
        self.iou_type = iou_type
        self.label_smoothing = label_smoothing
        self.p5_weight_boost = p5_weight_boost
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute anchor-free YOLO-One loss
        
        Args:
            predictions: Dict with 'detections', 'aspects', 'shape_confidences'
            targets: Ground truth annotations [batch_idx, class, x, y, w, h]
            model: Model for info extraction (optional)
            
        Returns:
            Dictionary with loss components
        """
        
        device = predictions['detections'][0].device
        
        # --- MoE Load Balancing Loss ---
        # This loss encourages the gating network to distribute load across experts,
        # preventing it from collapsing to using only one expert for all inputs.
        gate_scores = predictions.get('gate_scores')
        loss_moe_balance = torch.zeros(1, device=device)
        if self.moe_balance_weight > 0 and gate_scores is not None:
            num_experts = gate_scores.shape[1]
            load_per_expert = gate_scores.sum(dim=0)
            # Calculate the coefficient of variation squared, a common load balancing loss.
            # We multiply by num_experts as a heuristic to keep the loss magnitude stable.
            loss_moe_balance = (load_per_expert.var() / (load_per_expert.mean()**2 + 1e-8)) * num_experts

        # --- Per-Expert Loss Calculation ---
        # We calculate the loss for each expert (detection head) separately.
        # These will then be weighted by the gate_scores to train the Gating Network.
        box_losses_per_expert = []
        obj_losses_per_expert = []

        # The following losses are disabled as the current head does not produce these outputs
        # aspect_losses_per_expert = []
        # shape_conf_losses_per_expert = []

        scales = [
            {'stride': 8, 'size': 80},   # P3
            {'stride': 16, 'size': 40},  # P4
            {'stride': 32, 'size': 20}   # P5
        ]
        
        # Process each scale
        for scale_idx, scale_info in enumerate(scales): # Each scale corresponds to an expert
            
            detections = predictions['detections'][scale_idx]
            # aspects = predictions.get('aspects', [None]*3)[scale_idx]
            # shape_confs = predictions.get('shape_confidences', [None]*3)[scale_idx]
            
            batch_size, _, height, width = detections.shape
            
            # Build targets for current scale
            scale_targets, obj_mask, box_mask, aspect_targets = self._build_anchor_free_targets(
                targets, (batch_size, height, width), scale_info['stride']
            )
            
            # Extract predictions
            pred_boxes = detections[:, :4]  # [B, 4, H, W]
            pred_conf = detections[:, 4]    # [B, H, W]
            # pred_aspects = aspects[:, 0] if aspects is not None else None
            # pred_shape_conf = shape_confs[:, 0] if shape_confs is not None else None
            
            # Extract targets
            target_boxes = scale_targets[:, :, :, :4]  # [B, H, W, 4]
            target_conf = scale_targets[:, :, :, 4]    # [B, H, W]
            
            scale_box_loss = torch.zeros(1, device=device)
            scale_obj_loss = torch.zeros(1, device=device)

            # Box loss (only where objects exist)
            if box_mask.sum() > 0:
                # pred_boxes: [B, 4, H, W] -> [B, H, W, 4] for proper masking
                pred_boxes_hwc = pred_boxes.permute(0, 2, 3, 1)  # [B, H, W, 4]
                
                pred_boxes_masked = pred_boxes_hwc[box_mask]  # [num_objects, 4]
                target_boxes_masked = target_boxes[box_mask]  # [num_objects, 4]
                
                scale_box_loss = self._compute_anchor_free_box_loss(
                    pred_boxes_masked, target_boxes_masked, 
                    scale_info['stride'], height, width
                )
                
            # Objectness loss for this scale
            scale_obj_loss = self._compute_objectness_loss(pred_conf, target_conf, obj_mask)

            # Apply weights
            box_weight = self.box_weight * (self.p5_weight_boost if scale_idx == 2 else 1.0)
            box_losses_per_expert.append(scale_box_loss * box_weight)
            obj_losses_per_expert.append(scale_obj_loss * self.obj_weight)
            
            # # Aspect ratio loss (disabled)
            # if pred_aspects is not None and aspect_targets is not None and box_mask.sum() > 0:
            #     aspect_loss = self._compute_aspect_loss(
            #         pred_aspects[box_mask], aspect_targets[box_mask]
            #     )
            #     loss_aspect += aspect_loss * self.aspect_weight
            
            # # Shape confidence loss (disabled)
            # if pred_shape_conf is not None:
            #     shape_conf_targets = obj_mask.float()
            #     shape_conf_loss = self._compute_shape_confidence_loss(
            #         pred_shape_conf, shape_conf_targets
            #     )
            #     loss_shape_conf += shape_conf_loss * self.shape_conf_weight

        # --- Combine Losses ---
        # If MoE is active, weigh losses by gate scores. This teaches the Gating Network
        # to route inputs to the expert that is best at handling them.
        if gate_scores is not None:
            # Average scores across the batch to get a weight for each expert
            expert_weights = gate_scores.mean(dim=0)  # [num_experts]
            
            # Stack losses into a tensor: [num_experts]
            total_box_losses = torch.stack(box_losses_per_expert)
            total_obj_losses = torch.stack(obj_losses_per_expert)
            
            # Weighted sum
            loss_box = (total_box_losses * expert_weights).sum()
            loss_obj = (total_obj_losses * expert_weights).sum()
        else:
            # Standard summation if no gating (e.g., for a non-MoE model)
            loss_box = torch.sum(torch.stack(box_losses_per_expert))
            loss_obj = torch.sum(torch.stack(obj_losses_per_expert))

        total_loss = loss_box + loss_obj + (loss_moe_balance * self.moe_balance_weight)
        
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
        stride: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build targets for anchor-free detection
        """
        
        batch_size, grid_h, grid_w = grid_shape
        device = targets.device
        
        # Initialize target tensors
        target_tensor = torch.zeros(batch_size, grid_h, grid_w, 5, device=device)
        obj_mask = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.bool, device=device)
        box_mask = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.bool, device=device)
        aspect_targets = torch.zeros(batch_size, grid_h, grid_w, device=device)
        
        if targets.size(0) == 0:
            return target_tensor, obj_mask, box_mask, aspect_targets
        
        # Process each target
        for target in targets:
            batch_idx = int(target[0])
            if batch_idx >= batch_size:
                continue
            
            # Extract target info
            x_center, y_center, width, height = target[2:6]
            
            # Convert to grid coordinates
            grid_x = x_center * grid_w
            grid_y = y_center * grid_h
            
            grid_i = int(grid_x.clamp(0, grid_w - 1))
            grid_j = int(grid_y.clamp(0, grid_h - 1))
            
            # Direct regression targets (relative to cell)
            target_tensor[batch_idx, grid_j, grid_i, 0] = grid_x - grid_i  # dx
            target_tensor[batch_idx, grid_j, grid_i, 1] = grid_y - grid_j  # dy
            target_tensor[batch_idx, grid_j, grid_i, 2] = width * grid_w   # dw
            target_tensor[batch_idx, grid_j, grid_i, 3] = height * grid_h  # dh
            target_tensor[batch_idx, grid_j, grid_i, 4] = 1.0  # confidence
            
            # Aspect ratio target
            aspect_ratio = width / (height + 1e-6)
            normalized_aspect = aspect_ratio / (1 + aspect_ratio)  # Normalize to [0,1]
            aspect_targets[batch_idx, grid_j, grid_i] = normalized_aspect
            
            obj_mask[batch_idx, grid_j, grid_i] = True
            box_mask[batch_idx, grid_j, grid_i] = True
        
        return target_tensor, obj_mask, box_mask, aspect_targets
    
    def _compute_anchor_free_box_loss(
        self, 
        pred_boxes: torch.Tensor, 
        target_boxes: torch.Tensor,
        stride: int,
        grid_h: int,
        grid_w: int
    ) -> torch.Tensor:
        """Anchor-free box loss with direct regression"""
        
        # Apply sigmoid to position predictions
        pred_xy = torch.sigmoid(pred_boxes[:, :2])
        pred_wh = pred_boxes[:, 2:4]
        
        target_xy = target_boxes[:, :2]
        target_wh = target_boxes[:, 2:4]
        
        # CIoU loss for direct regression
        if self.iou_type == 'ciou':
            # Convert to absolute coordinates for IoU computation
            pred_boxes_abs = torch.cat([pred_xy, torch.exp(pred_wh)], dim=-1)
            target_boxes_abs = torch.cat([target_xy, target_wh], dim=-1)
            
            loss = self._ciou_loss(pred_boxes_abs, target_boxes_abs)
        elif self.iou_type == 'meiou':
            pred_boxes_abs = torch.cat([pred_xy, torch.exp(pred_wh)], dim=-1)
            target_boxes_abs = torch.cat([target_xy, target_wh], dim=-1)
            
            loss = self._meiou_loss(pred_boxes_abs, target_boxes_abs)
        elif self.iou_type == 'eiou':
            pred_boxes_abs = torch.cat([pred_xy, torch.exp(pred_wh)], dim=-1)
            target_boxes_abs = torch.cat([target_xy, target_wh], dim=-1)
            
            loss = self._eiou_loss(pred_boxes_abs, target_boxes_abs)
        else:
            # MSE fallback
            loss = F.mse_loss(pred_xy, target_xy) + F.mse_loss(pred_wh, target_wh)
        
        return loss.mean()
    
    def _compute_aspect_loss(
        self,
        pred_aspects: torch.Tensor,
        target_aspects: torch.Tensor
    ) -> torch.Tensor:
        """Loss for aspect ratio predictions"""
        return F.mse_loss(pred_aspects, target_aspects)
    
    def _compute_shape_confidence_loss(
        self,
        pred_shape_conf: torch.Tensor,
        target_shape_conf: torch.Tensor
    ) -> torch.Tensor:
        """Loss for shape confidence predictions"""
        bce_loss = self.bce_loss(pred_shape_conf, target_shape_conf)
        
        # Apply focal loss
        if self.focal_gamma > 0:
            pred_prob = torch.sigmoid(pred_shape_conf)
            pt = target_shape_conf * pred_prob + (1 - target_shape_conf) * (1 - pred_prob)
            focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma
            bce_loss = focal_weight * bce_loss
        
        return bce_loss.mean()
    
    def _compute_objectness_loss(
        self,
        pred_conf: torch.Tensor,
        target_conf: torch.Tensor, 
        obj_mask: torch.Tensor
    ) -> torch.Tensor:
        """Objectness loss with focal loss for single-class"""
        
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
        
        return pos_loss + self.obj_neg_weight * neg_loss
    
    def _ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Complete IoU loss implementation for anchor-free"""
        
        # Convert to corner format
        pred_x1, pred_y1, pred_x2, pred_y2 = box_cxcywh_to_xyxy(pred_boxes).unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = box_cxcywh_to_xyxy(target_boxes).unbind(-1)
        
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
        
        px1, py1, px2, py2 = box_cxcywh_to_xyxy(pred_boxes).unbind(-1)
        tx1, ty1, tx2, ty2 = box_cxcywh_to_xyxy(target_boxes).unbind(-1)

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
                    target_boxes: torch.Tensor, 
                    lambda1: float =0.3, 
                    lambda2: float =0.5, 
                    lambda3: float = 0.2
                    )-> torch.Tensor:
        

        px1, py1, px2, py2 = box_cxcywh_to_xyxy(pred_boxes).unbind(-1)
        tx1, ty1, tx2, ty2 = box_cxcywh_to_xyxy(target_boxes).unbind(-1)
        
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
        
        # Define pw, ph, tw, th before use
        pw, ph = px2 - px1, py2 - py1
        tw, th = tx2 - tx1, ty2 - ty1
        
        wc2, hc2 = ew**2, eh**2 
        rho2_w = (pw - tw) ** 2
        rho2_h = (ph - th) ** 2

        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        rho2_center = (pcx - tcx)**2 + (pcy - tcy)**2

        #v_aspect = (4 / (torch.pi**2)) * (torch.atan(tw / th) - torch.atan(pw / ph))**2
        v_absolute =  rho2_w / (wc2 + 1e-6) + rho2_h / (hc2 + 1e-6)
        #v_absolute = ((ph - th)**2 + (pw - tw)**2) / (c2+ 1e-6)
        
        ch = torch.max(ph, th) / 2
        sigma = torch.sqrt(ew**2 + eh**2)
        delta_angle = 1 - 2 * torch.sin(torch.arcsin(torch.clamp(ch / (sigma + 1e-6), -1.0, 1.0)) - (torch.pi / 4))**2

        #v_co = lambda1 * v_aspect + lambda2 * v_absolute + lambda3 * delta_angle
        #alpha = v_co / ((1 - iou) + v_co + 1e-6)

        meiou = iou - rho2_center / c2 - v_absolute - delta_angle #- 0.2 * delta_angle

        return (iou**self.focal_gamma)*(1 - meiou)
    
    

def create_yolo_one_loss(
    box_weight: float = 7.5,
    obj_weight: float = 1.0,
    aspect_weight: float = 0.5,
    shape_conf_weight: float = 0.2,
    focal_alpha: float = 0.25,
    moe_balance_weight: float = 0.05,
    focal_gamma: float = 1.5,
    obj_neg_weight: float = 0.05,
    iou_type: str = 'meiou',
    label_smoothing: float = 0.0,
    p5_weight_boost: float = 1.2
) -> YoloOneLoss:
    """Factory function to create anchor-free YOLO-One loss"""
    return YoloOneLoss(
        box_weight=box_weight,
        obj_weight=obj_weight,
        aspect_weight=aspect_weight,
        shape_conf_weight=shape_conf_weight,
        focal_alpha=focal_alpha,
        moe_balance_weight=moe_balance_weight,
        focal_gamma=focal_gamma,
        obj_neg_weight=obj_neg_weight,
        iou_type=iou_type,
        label_smoothing=label_smoothing,
        p5_weight_boost=p5_weight_boost
    )

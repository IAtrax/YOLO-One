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
        focal_gamma: float = 1.5,
        iou_type: str = 'meiou',
        label_smoothing: float = 0.0,
        p5_weight_boost: float = 1.2,
        theta: int = 4,
        focal_loss: bool= False
    ):
        super().__init__()
        
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.aspect_weight = aspect_weight
        self.shape_conf_weight = shape_conf_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.iou_type = iou_type
        self.label_smoothing = label_smoothing
        self.p5_weight_boost = p5_weight_boost
        self.theta = theta
        self.focal_loss = focal_loss
        
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
        
        device = predictions['preds'][0].device
        
        # Initialize losses
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        # loss_aspect = torch.zeros(1, device=device)
        # loss_shape_conf = torch.zeros(1, device=device)
        
        # Scale information
        scales = [
            {'stride': 8, 'size': 80},   # P3
            {'stride': 16, 'size': 40},  # P4
            {'stride': 32, 'size': 20}   # P5
        ]
        
        # Process each scale
        for scale_idx, scale_info in enumerate(scales):
            
            detections = predictions['preds'][scale_idx]
            # aspects = predictions['aspects'][scale_idx]
            # shape_confs = predictions['shape_confidences'][scale_idx]
            
            batch_size, _, height, width = detections.shape
            
            # Build targets for current scale
            scale_targets, obj_mask, box_mask, aspect_targets = self._build_anchor_free_targets(
                targets, (batch_size, height, width), scale_info['stride']
            )
            
            # Extract predictions
            pred_boxes = detections[:, :4]  # [B, 4, H, W]
            pred_conf = detections[:, 4]    # [B, H, W]
            # pred_aspects = aspects[:, 0]    # [B, H, W]
            # pred_shape_conf = shape_confs[:, 0]  # [B, H, W]
            
            # Extract targets
            target_boxes = scale_targets[:, :, :, :4]  # [B, H, W, 4]
            target_conf = scale_targets[:, :, :, 4]    # [B, H, W]
            
            # Box loss (only where objects exist)
            if box_mask.sum() > 0:
                # Fix: Proper indexing for box predictions and targets
                # pred_boxes: [B, 4, H, W] -> [B, H, W, 4] for proper masking
                pred_boxes_hwc = pred_boxes.permute(0, 2, 3, 1)  # [B, H, W, 4]
                
                pred_boxes_masked = pred_boxes_hwc[box_mask]  # [num_objects, 4]
                target_boxes_masked = target_boxes[box_mask]  # [num_objects, 4]
                
                box_loss = self._compute_anchor_free_box_loss(
                    pred_boxes_masked, target_boxes_masked, 
                    scale_info['stride'], height, width
                )
                
                # Boost P5 loss
                weight = self.box_weight
                if scale_idx == 2:  # P5 scale
                    weight *= self.p5_weight_boost
                    
                loss_box += box_loss * weight
            
            # Objectness loss
            obj_loss = self._compute_objectness_loss(pred_conf, target_conf, obj_mask)
            loss_obj += obj_loss * self.obj_weight
            
            # Aspect ratio loss
            # if aspect_targets is not None and box_mask.sum() > 0:
            #     aspect_loss = self._compute_aspect_loss(
            #         pred_aspects[box_mask], aspect_targets[box_mask]
            #     )
            #     loss_aspect += aspect_loss * self.aspect_weight
            
            # # Shape confidence loss
            # shape_conf_targets = obj_mask.float()
            # shape_conf_loss = self._compute_shape_confidence_loss(
            #     pred_shape_conf, shape_conf_targets
            # )
            # loss_shape_conf += shape_conf_loss * self.shape_conf_weight
        
        # Total loss
        total_loss = loss_box + loss_obj #+ loss_shape_conf #+ loss_aspect 
        
        return {
            'total_loss': total_loss,
            'box_loss': loss_box,
            'obj_loss': loss_obj,
            # 'aspect_loss': loss_aspect,
            # 'shape_conf_loss': loss_shape_conf,
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
        elif self.iou_type == 'siou':
            pred_boxes_abs = torch.cat([pred_xy, torch.exp(pred_wh)], dim=-1)
            target_boxes_abs = torch.cat([target_xy, target_wh], dim=-1)
            
            loss = self._siou_loss(pred_boxes_abs, target_boxes_abs)
        else:
            # MSE fallback
            loss = F.mse_loss(pred_xy, target_xy) + F.mse_loss(pred_wh, target_wh)
        
        return loss
    
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
        
        return pos_loss + 0.05 * neg_loss
    
    def _ciou_loss(
            self,
            pred_boxes: torch.Tensor,
            target_boxes: torch.Tensor
            ) -> torch.Tensor:

        """Complete IoU loss implementation for anchor-free"""
        
        # Pred and target boxes do not contain None 
        assert pred_boxes is not None, "pred_boxes must not be None"
        assert target_boxes is not None, "target_boxes must not be None" 


        # Pred and target boxes must be Tensor
        assert isinstance(pred_boxes, torch.cuda.FloatTensor), f"pred_boxes must be a torch.cuda.FloatTensor type, got {type(pred_boxes)}"
        assert isinstance(target_boxes, torch.cuda.FloatTensor), f"target_boxes must be a torch.cuda.FloatTensor type, got {type(target_boxes)}"

        # Pred and target boxes must have the same shape
        assert len(pred_boxes) == len(target_boxes), f"Length mismatch: len(pred_boxes)={len(pred_boxes)}, len(target_boxes)={len(target_boxes)}"

        pred_x1, pred_y1, pred_x2, pred_y2 = box_cxcywh_to_xyxy(pred_boxes).unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = box_cxcywh_to_xyxy(target_boxes).unbind(-1)

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
        
        # CIoU Loss
        ciou_loss = 1- iou + dist_center_2 / enclosed_2 + alpha * v

        # Focal 
        if self.focal_loss:
            ciou_loss= (iou**self.focal_gamma)*(ciou_loss)

        # Mask invalid boxes (sum==0)
        valid_mask = (target_boxes.sum(-1) > 0) & (pred_boxes.sum(-1) > 0)

        if valid_mask.any():
            ciou_loss = ciou_loss[valid_mask].mean()
        else:
            ciou_loss = torch.tensor(0., device=pred_boxes.device)

        return ciou_loss

    
    def _eiou_loss(
                    self,
                    pred_boxes: torch.Tensor, 
                    target_boxes: torch.Tensor, 
                    width_img: int = 640,
                    height_img: int = 640,
                    )-> torch.Tensor:
        
        """Efficient IoU loss implementation for anchor-free"""
        
        # Pred and target boxes do not contain None 
        assert pred_boxes is not None, "pred_boxes must not be None"
        assert target_boxes is not None, "target_boxes must not be None" 


        # Pred and target boxes must be Tensor
        assert isinstance(pred_boxes, torch.cuda.FloatTensor), f"pred_boxes must be a torch.cuda.FloatTensor type, got {type(pred_boxes)}"
        assert isinstance(target_boxes, torch.cuda.FloatTensor), f"target_boxes must be a torch.cuda.FloatTensor type, got {type(target_boxes)}"

        # Pred and target boxes must have the same shape
        assert len(pred_boxes) == len(target_boxes), f"Length mismatch: len(pred_boxes)={len(pred_boxes)}, len(target_boxes)={len(target_boxes)}"

        pred_x1, pred_y1, pred_x2, pred_y2 = box_cxcywh_to_xyxy(pred_boxes).unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = box_cxcywh_to_xyxy(target_boxes).unbind(-1)

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
        pred_w, pred_h = torch.clamp(pred_w, max=0.99*width_img), torch.clamp(pred_h, max=0.99*height_img)
        target_w, target_h = torch.clamp(target_w, max=0.99*width_img), torch.clamp(target_h, max=0.99*height_img)
        rho2_w = (pred_w - target_w) ** 2
        rho2_h = (pred_h - target_h) ** 2
        height_width_loss = (rho2_h / enclosed_h)  +  (rho2_w / enclosed_w)


        # EIoU Loss
        eiou_loss  =  1 - iou + dist_center_2 / enclosed_2  + height_width_loss
        
        # Focal
        if self.focal_loss:
            eiou_loss = (iou**self.focal_gamma)*(eiou_loss)

        # Mask invalid boxes (sum==0)
        valid_mask = (target_boxes.sum(-1) > 0) & (pred_boxes.sum(-1) > 0)

        if valid_mask.any():
            eiou_loss = eiou_loss[valid_mask].mean()
        else:
            eiou_loss = torch.tensor(0., device=pred_boxes.device)

        return eiou_loss

    

    def _meiou_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        width_img: int = 640,
        height_img: int = 640,
        
    ) -> torch.Tensor:
        
        """More efficient IoU loss implementation for anchor-free"""

        # Pred and target boxes do not contain None 
        assert pred_boxes is not None, "pred_boxes must not be None"
        assert target_boxes is not None, "target_boxes must not be None" 


        # Pred and target boxes must be Tensor
        assert isinstance(pred_boxes, torch.cuda.FloatTensor), f"pred_boxes must be a torch.cuda.FloatTensor type, got {type(pred_boxes)}"
        assert isinstance(target_boxes, torch.cuda.FloatTensor), f"target_boxes must be a torch.cuda.FloatTensor type, got {type(target_boxes)}"

        # Pred and target boxes must have the same shape
        assert len(pred_boxes) == len(target_boxes), f"Length mismatch: len(pred_boxes)={len(pred_boxes)}, len(target_boxes)={len(target_boxes)}"

        pred_x1, pred_y1, pred_x2, pred_y2 = box_cxcywh_to_xyxy(pred_boxes).unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = box_cxcywh_to_xyxy(target_boxes).unbind(-1)

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
        pred_w, pred_h = torch.clamp(pred_w, max=0.99*width_img), torch.clamp(pred_h, max=0.99*height_img)
        target_w, target_h = torch.clamp(target_w, max=0.99*width_img), torch.clamp(target_h, max=0.99*height_img)
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
        eiou_loss  =  1 - iou + dist_center_2 / enclosed_2 + height_width_loss
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



    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Convert center format to corner format"""
        x, y, w, h = boxes.unbind(-1)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2


def create_yolo_one_loss(
    box_weight: float = 7.5,
    obj_weight: float = 1.0,
    aspect_weight: float = 0.5,
    shape_conf_weight: float = 0.2,
    focal_alpha: float = 0.25,
    focal_gamma: float = 1.5,
    iou_type: str = 'meiou',
    label_smoothing: float = 0.0,
    p5_weight_boost: float = 1.2,
    theta: int = 4,
    focal_loss: bool= False
) -> YoloOneLoss:
    """Factory function to create anchor-free YOLO-One loss"""
    return YoloOneLoss(
        box_weight=box_weight,
        obj_weight=obj_weight,
        aspect_weight=aspect_weight,
        shape_conf_weight=shape_conf_weight,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        iou_type=iou_type,
        label_smoothing=label_smoothing,
        p5_weight_boost=p5_weight_boost,
        theta = theta,
        focal_loss=focal_loss
    )
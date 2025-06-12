"""
YOLO-One Post-Processing Module
Iatrax Team - 2025 - https://iatrax.com

Complete post-processing pipeline for YOLO-One detections
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional

class YoloOnePostProcessor:
    """
    Post-processor for YOLO-One model outputs
    Handles coordinate conversion, NMS, and filtering
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize post-processor
        
        Args:
            conf_threshold: Minimum confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to keep
            input_size: Model input size (H, W)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.input_size = input_size
        
        # Grid sizes for each scale
        self.grid_sizes = [
            (input_size[0] // 8, input_size[1] // 8),    # P3: 80x80
            (input_size[0] // 16, input_size[1] // 16),  # P4: 40x40  
            (input_size[0] // 32, input_size[1] // 32)   # P5: 20x20
        ]
        
        # Stride for each scale
        self.strides = [8, 16, 32]
    
    def process_batch(
        self, 
        predictions: List[torch.Tensor],
        original_shapes: Optional[List[Tuple[int, int]]] = None
    ) -> List[Dict]:
        """
        Process a batch of predictions
        
        Args:
            predictions: List of prediction tensors [P3, P4, P5]
            original_shapes: Original image shapes for rescaling
            
        Returns:
            List of detection dictionaries for each image in batch
        """
        batch_size = predictions[0].shape[0]
        batch_results = []
        
        for batch_idx in range(batch_size):
            # Extract predictions for this image
            image_predictions = [pred[batch_idx] for pred in predictions]
            
            # Process single image
            detections = self.process_single_image(image_predictions)
            
            # Rescale to original image size if provided
            if original_shapes is not None:
                detections = self.rescale_detections(
                    detections, 
                    self.input_size, 
                    original_shapes[batch_idx]
                )
            
            batch_results.append(detections)
        
        return batch_results
    
    def process_single_image(self, predictions: List[torch.Tensor]) -> Dict:
        """
        Process predictions for a single image
        
        Args:
            predictions: List of prediction tensors for one image
            
        Returns:
            Dictionary with processed detections
        """
        # Convert predictions to detection format
        all_detections = []
        
        for scale_idx, pred in enumerate(predictions):
            scale_detections = self.decode_predictions(pred, scale_idx)
            all_detections.append(scale_detections)
        
        # Concatenate all scales
        if all_detections:
            detections = torch.cat(all_detections, dim=0)
        else:
            detections = torch.empty(0, 6)
        
        # Apply confidence filtering
        detections = self.filter_by_confidence(detections)
        
        # Apply Non-Maximum Suppression
        detections = self.apply_nms(detections)
        
        # Format output
        return self.format_detections(detections)
    
    def decode_predictions(self, pred: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """
        Decode predictions from grid format to absolute coordinates
        
        Args:
            pred: Prediction tensor [6, H, W]
            scale_idx: Scale index (0=P3, 1=P4, 2=P5)
            
        Returns:
            Decoded detections [N, 6] (x1, y1, x2, y2, conf, class_prob)
        """
        device = pred.device
        channels, grid_h, grid_w = pred.shape
        stride = self.strides[scale_idx]
        
        # Create grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=device),
            torch.arange(grid_w, device=device),
            indexing='ij'
        )
        
        # Extract predictions
        class_prob = pred[0]      # [H, W]
        center_x = pred[1]        # [H, W] 
        center_y = pred[2]        # [H, W]
        width = pred[3]           # [H, W]
        height = pred[4]          # [H, W]
        objectness = pred[5]      # [H, W]
        
        # Calculate confidence
        confidence = class_prob * objectness  # [H, W]
        
        # Convert relative coordinates to absolute
        abs_center_x = (grid_x + center_x) * stride
        abs_center_y = (grid_y + center_y) * stride
        abs_width = width * self.input_size[1]
        abs_height = height * self.input_size[0]
        
        # Convert center format to corner format
        x1 = abs_center_x - abs_width / 2
        y1 = abs_center_y - abs_height / 2
        x2 = abs_center_x + abs_width / 2
        y2 = abs_center_y + abs_height / 2
        
        # Stack all predictions
        detections = torch.stack([
            x1.flatten(),
            y1.flatten(), 
            x2.flatten(),
            y2.flatten(),
            confidence.flatten(),
            class_prob.flatten()
        ], dim=1)
        
        return detections
    
    def filter_by_confidence(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Filter detections by confidence threshold
        
        Args:
            detections: Detection tensor [N, 6]
            
        Returns:
            Filtered detections
        """
        if detections.shape[0] == 0:
            return detections
            
        confidence_mask = detections[:, 4] >= self.conf_threshold
        return detections[confidence_mask]
    
    def apply_nms(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Apply Non-Maximum Suppression
        
        Args:
            detections: Detection tensor [N, 6]
            
        Returns:
            NMS filtered detections
        """
        if detections.shape[0] == 0:
            return detections
        
        # Sort by confidence
        conf_sort_idx = torch.argsort(detections[:, 4], descending=True)
        detections = detections[conf_sort_idx]
        
        # Limit to max detections before NMS for efficiency
        if detections.shape[0] > self.max_detections * 2:
            detections = detections[:self.max_detections * 2]
        
        # Extract boxes and scores
        boxes = detections[:, :4]
        scores = detections[:, 4]
        
        # Apply NMS
        keep_indices = self.nms(boxes, scores, self.iou_threshold)
        
        # Limit final detections
        keep_indices = keep_indices[:self.max_detections]
        
        return detections[keep_indices]
    
    def nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """
        Non-Maximum Suppression implementation
        
        Args:
            boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
            scores: Confidence scores [N]
            iou_threshold: IoU threshold
            
        Returns:
            Indices of boxes to keep
        """
        if boxes.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by scores
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            # Keep highest scoring box
            i = order[0].item()
            keep.append(i)
            
            if order.numel() == 1:
                break
            
            # Calculate IoU with remaining boxes
            rest_boxes = boxes[order[1:]]
            ious = self.calculate_iou(boxes[i:i+1], rest_boxes)
            
            # Keep boxes with IoU < threshold
            mask = ious[0] <= iou_threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def calculate_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU between boxes
        
        Args:
            boxes1: First set of boxes [N, 4]
            boxes2: Second set of boxes [M, 4]
            
        Returns:
            IoU matrix [N, M]
        """
        # Calculate intersection
        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate union
        union_area = area1[:, None] + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        
        return iou
    
    def rescale_detections(
        self, 
        detections: Dict, 
        model_size: Tuple[int, int], 
        original_size: Tuple[int, int]
    ) -> Dict:
        """
        Rescale detections to original image size
        
        Args:
            detections: Detection dictionary
            model_size: Model input size (H, W)
            original_size: Original image size (H, W)
            
        Returns:
            Rescaled detections
        """
        if len(detections['boxes']) == 0:
            return detections
        
        # Calculate scale factors
        scale_x = original_size[1] / model_size[1]
        scale_y = original_size[0] / model_size[0]
        
        # Rescale boxes
        boxes = detections['boxes'].clone()
        boxes[:, [0, 2]] *= scale_x  # x coordinates
        boxes[:, [1, 3]] *= scale_y  # y coordinates
        
        detections['boxes'] = boxes
        
        return detections
    
    def format_detections(self, detections: torch.Tensor) -> Dict:
        """
        Format detections into structured output
        
        Args:
            detections: Detection tensor [N, 6]
            
        Returns:
            Formatted detection dictionary
        """
        if detections.shape[0] == 0:
            return {
                'boxes': torch.empty(0, 4),
                'scores': torch.empty(0),
                'class_probs': torch.empty(0),
                'count': 0
            }
        
        return {
            'boxes': detections[:, :4],          # [N, 4] (x1, y1, x2, y2)
            'scores': detections[:, 4],          # [N] confidence scores
            'class_probs': detections[:, 5],     # [N] class probabilities
            'count': detections.shape[0]         # Number of detections
        }

# Convenience function
def postprocess_yolo_one_outputs(
    predictions: List[torch.Tensor],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
    input_size: Tuple[int, int] = (640, 640),
    original_shapes: Optional[List[Tuple[int, int]]] = None
) -> List[Dict]:
    """
    Convenience function for post-processing YOLO-One outputs
    
    Args:
        predictions: List of prediction tensors [P3, P4, P5]
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections to keep
        input_size: Model input size
        original_shapes: Original image shapes for rescaling
        
    Returns:
        List of detection dictionaries
    """
    processor = YoloOnePostProcessor(
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        input_size=input_size
    )
    
    return processor.process_batch(predictions, original_shapes)
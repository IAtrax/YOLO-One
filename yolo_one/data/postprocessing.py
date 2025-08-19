"""
YOLO-One Post-Processing Module
Iatrax Team - 2025 - https://iatrax.com

Complete post-processing pipeline for YOLO-One detections
"""
import torch
from typing import List, Tuple, Dict, Optional
try:
    from torchvision.ops import nms
except ImportError:
    nms = None

from yolo_one.utils.general import box_cxcywh_to_xyxy

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
    
    def process_batch(
        self, 
        predictions: List[torch.Tensor],
        original_shapes: List[Tuple[int, int]],
        scale_factors: List[float],
        paddings: List[Tuple[int, int]]
    ) -> List[Dict]:
        """
        Process a batch of predictions
        
        Args:
            predictions: List of DECODED prediction tensors from the model head for the whole batch.
                         Each tensor is [B, 5, H, W] -> (xc, yc, w, h, conf).
            original_shapes: Original image shapes for rescaling
            
        Returns:
            List of detection dictionaries for each image in batch
        """
        batch_size = predictions[0].shape[0]
        batch_results = []
        
        for batch_idx in range(batch_size):
            image_predictions = [pred[batch_idx:batch_idx+1] for pred in predictions]

            # Flatten and concatenate all predictions for the current image
            preds = [p.permute(0, 2, 3, 1).reshape(-1, 5) for p in image_predictions]
            all_preds = torch.cat(preds, dim=0)
            # Process single image
            detections = self.process_single_image(all_preds)
            
            # Rescale to original image size
            detections = self.rescale_detections(
                detections,
                scale_factors[batch_idx],
                paddings[batch_idx],
                original_shapes[batch_idx]
            )
            batch_results.append(detections)
        
        return batch_results
    
    def process_single_image(self, predictions: torch.Tensor) -> Dict:
        """
        Process predictions for a single image
        
        Args:
            predictions: A tensor of [N, 5] decoded predictions (xc, yc, w, h, conf)
            
        Returns:
            Dictionary with processed detections
        """
        # Predictions are normalized to image size [0, 1]
        # Apply confidence filtering
        confident_preds = self.filter_by_confidence(predictions)

        if confident_preds.shape[0] == 0:
            return self.format_detections(torch.empty(0, 5, device=predictions.device))

        # Convert to corner format (xyxy) for NMS
        boxes_xyxy = box_cxcywh_to_xyxy(confident_preds[:, :4])
        scores = confident_preds[:, 4]

        # Apply Non-Maximum Suppression
        final_detections = self.apply_nms(torch.cat([boxes_xyxy, scores.unsqueeze(1)], dim=1))

        # Format output
        return self.format_detections(final_detections)
    
    def filter_by_confidence(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Filter detections by confidence threshold
        
        Args:
            detections: Detection tensor [N, 5] (xc, yc, w, h, conf)
            
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
            detections: Detection tensor [N, 5] (x1, y1, x2, y2, score)
            
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
        
        if nms is None:
            raise ImportError("torchvision is not installed. Please install it to use NMS (`pip install torchvision`).")

        # Apply the highly optimized torchvision NMS
        keep_indices = nms(boxes, scores, self.iou_threshold)
        
        # Limit final detections after NMS
        keep_indices = keep_indices[:self.max_detections]
        
        return detections[keep_indices]

    def rescale_detections(
        self, 
        detections: Dict, 
        scale_factor: float,
        padding: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> Dict:
        """
        Rescale detections from model input size to original image size,
        accounting for letterbox padding.
        
        Args:
            detections: Detection dictionary
            scale_factor: The factor used to scale the image.
            padding: The padding added to the image (pad_left, pad_top).
            original_size: Original image size (H, W)
            
        Returns:
            Rescaled detections
        """
        if detections['count'] == 0:
            return detections
        
        boxes = detections['boxes'].clone() # Normalized [0,1] xyxy
        pad_left, pad_top = padding
        original_height, original_width = original_size

        boxes *= self.input_size[0]
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        boxes /= scale_factor
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, original_width)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, original_height)
        
        detections['boxes'] = boxes
        
        return detections
    
    def format_detections(self, detections: torch.Tensor) -> Dict:
        """
        Format detections into structured output
        
        Args:
            detections: Detection tensor [N, 5] (x1, y1, x2, y2, score)
            
        Returns:
            Formatted detection dictionary
        """
        if detections.shape[0] == 0:
            return {
                'boxes': torch.empty(0, 4),
                'scores': torch.empty(0),
                'count': 0
            }
        
        return {
            'boxes': detections[:, :4],          # [N, 4] (x1, y1, x2, y2)
            'scores': detections[:, 4],          # [N] confidence scores
            'count': detections.shape[0]         # Number of detections
        }
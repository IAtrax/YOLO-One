"""
YOLO-One Post-Processing Module
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

Complete post-processing pipeline for YOLO-One detections
"""
import torch
from typing import List, Tuple, Dict, Optional

try:
    from torchvision.ops import nms
except ImportError:
    nms = None


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

        if nms is None:
            raise ImportError(
                "torchvision is required for NMS. Install with: pip install torchvision"
            )

    def process_batch(
        self,
        predictions: List[torch.Tensor],
        original_shapes: Optional[List[Tuple[int, int]]] = None,
        scale_factors: Optional[List[float]] = None,
        paddings: Optional[List[Tuple[int, int]]] = None
    ) -> List[Dict]:
        """
        Process a batch of predictions

        Args:
            predictions: List of decoded prediction tensors [B, 5, H, W] (x1, y1, x2, y2, conf)
                        from multi-scale detection heads (P3, P4, P5)
            original_shapes: Original image shapes [(H, W), ...] for rescaling
            scale_factors: Scale factors used during preprocessing
            paddings: Padding added during preprocessing [(left, top), ...]

        Returns:
            List of detection dictionaries for each image in batch
        """
        batch_size = predictions[0].shape[0]
        batch_results = []

        for batch_idx in range(batch_size):
            # Gather all scale predictions for this image
            image_predictions = [pred[batch_idx] for pred in predictions]

            # Process single image
            detections = self._process_single_image(image_predictions)

            # Rescale to original image size if info provided
            if original_shapes is not None and scale_factors is not None and paddings is not None:
                detections = self._rescale_detections(
                    detections,
                    scale_factors[batch_idx],
                    paddings[batch_idx],
                    original_shapes[batch_idx]
                )

            batch_results.append(detections)

        return batch_results

    def _process_single_image(self, predictions: List[torch.Tensor]) -> Dict:
        """
        Process predictions for a single image from all scales

        Args:
            predictions: List of tensors [5, H, W] for each scale (P3, P4, P5)

        Returns:
            Dictionary with processed detections
        """
        # Flatten all multi-scale predictions
        all_preds = []
        for pred in predictions:
            # pred shape: [5, H, W] -> permute to [H, W, 5] -> flatten to [H*W, 5]
            flattened = pred.permute(1, 2, 0).reshape(-1, 5)
            all_preds.append(flattened)

        # Concatenate all scales: [N_total, 5]
        all_preds = torch.cat(all_preds, dim=0)

        # Apply sigmoid to get normalized coordinates and confidence
        all_preds_norm = all_preds.clone()
        all_preds_norm[:, :4] = torch.sigmoid(all_preds[:, :4])  # x1, y1, x2, y2 -> [0, 1]
        all_preds_norm[:, 4] = torch.sigmoid(all_preds[:, 4])    # conf -> [0, 1]

        # Filter by confidence
        conf_mask = all_preds_norm[:, 4] >= self.conf_threshold
        confident_preds = all_preds_norm[conf_mask]

        if confident_preds.shape[0] == 0:
            return self._format_detections(torch.empty(0, 5, device=all_preds.device))

        # Apply NMS
        final_detections = self._apply_nms(confident_preds)

        # Format output
        return self._format_detections(final_detections)

    def _apply_nms(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Apply Non-Maximum Suppression

        Args:
            detections: [N, 5] tensor (x1, y1, x2, y2, score) normalized [0, 1]

        Returns:
            NMS filtered detections
        """
        if detections.shape[0] == 0:
            return detections

        # Sort by confidence (highest first)
        sorted_indices = torch.argsort(detections[:, 4], descending=True)
        detections = detections[sorted_indices]

        # Limit before NMS for efficiency
        if detections.shape[0] > self.max_detections * 2:
            detections = detections[:self.max_detections * 2]

        # Extract boxes and scores
        boxes = detections[:, :4]
        scores = detections[:, 4]

        # Apply NMS
        keep_indices = nms(boxes, scores, self.iou_threshold)

        # Limit final detections
        keep_indices = keep_indices[:self.max_detections]

        return detections[keep_indices]

    def _rescale_detections(
        self,
        detections: Dict,
        scale_factor: float,
        padding: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> Dict:
        """
        Rescale detections from normalized [0,1] to original image coordinates

        Args:
            detections: Detection dictionary with normalized boxes [0, 1]
            scale_factor: Scale factor used during letterbox resize
            padding: Padding (left, top) added during letterbox
            original_size: Original image size (H, W)

        Returns:
            Rescaled detections in original image coordinates
        """
        if detections['count'] == 0:
            return detections

        boxes = detections['boxes'].clone()  # [N, 4] normalized [0, 1]
        pad_left, pad_top = padding
        original_height, original_width = original_size
        input_h, input_w = self.input_size

        # Step 1: Denormalize to input size
        boxes[:, [0, 2]] *= input_w
        boxes[:, [1, 3]] *= input_h

        # Step 2: Remove padding
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top

        # Step 3: Unscale to original size
        boxes /= scale_factor

        # Step 4: Clamp to image boundaries
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_height)

        detections['boxes'] = boxes

        return detections

    def _format_detections(self, detections: torch.Tensor) -> Dict:
        """
        Format detections into structured output

        Args:
            detections: [N, 5] tensor (x1, y1, x2, y2, score)

        Returns:
            Formatted detection dictionary
        """
        if detections.shape[0] == 0:
            return {
                'boxes': torch.empty(0, 4, device=detections.device),
                'scores': torch.empty(0, device=detections.device),
                'count': 0
            }

        return {
            'boxes': detections[:, :4],     # [N, 4] (x1, y1, x2, y2)
            'scores': detections[:, 4],     # [N] confidence scores
            'count': detections.shape[0]    # Number of detections
        }

    @torch.no_grad()
    def __call__(
        self,
        predictions: List[torch.Tensor],
        original_shapes: Optional[List[Tuple[int, int]]] = None,
        scale_factors: Optional[List[float]] = None,
        paddings: Optional[List[Tuple[int, int]]] = None
    ) -> List[Dict]:
        """
        Convenience method to process batch (same as process_batch)
        """
        return self.process_batch(predictions, original_shapes, scale_factors, paddings)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)

    Args:
        boxes: [N, 4] tensor in (cx, cy, w, h) format

    Returns:
        [N, 4] tensor in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from corner format (x1, y1, x2, y2) to center format (cx, cy, w, h)

    Args:
        boxes: [N, 4] tensor in (x1, y1, x2, y2) format

    Returns:
        [N, 4] tensor in (cx, cy, w, h) format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)

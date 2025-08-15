"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

YOLO-ONE INFERENCE ENGINE
Optimized inference for anchor-free single-class detection
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Dict, Tuple, Union
import time
import logging

# Import your YOLO-One model
from yolo_one.models.yolo_one_model import YoloOne

class YoloOneInference:
    """
    High-performance inference engine for YOLO-One anchor-free detection
    Designed specifically for your multi-head architecture
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        max_detections: int = 2,
        input_size: int = 640,
        half_precision: bool = True,
        warmup_iterations: int = 3
    ):
        """
        Initialize YOLO-One inference engine
        
        Args:
            model_path: Path to trained YOLO-One model weights
            device: Computation device ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS threshold for duplicate removal
            max_detections: Maximum detections per image
            input_size: Model input size (square)
            half_precision: Use FP16 for faster inference
            warmup_iterations: GPU warmup iterations
        """
        
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.input_size = input_size
        self.half_precision = half_precision and device == 'cuda'
        
        # Load YOLO-One model
        self.model = self._load_yolo_one_model(model_path)
        
        # GPU warmup for optimal performance
        if warmup_iterations > 0 and device == 'cuda':
            self._warmup_gpu(warmup_iterations)
        
        # Performance tracking
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
        
        logging.info(f"YOLO-One inference engine ready on {device}")
        logging.info(f"Confidence: {confidence_threshold}, NMS: {nms_threshold}")
    
    def _load_yolo_one_model(self, model_path: str) -> nn.Module:
        """Load and prepare YOLO-One model for inference"""
        
        model = YoloOne(model_size='nano')  # Adjust size as needed
        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        # Prepare for inference
        model = model.to(self.device)
        model.eval()
        print("USED DEVICE", self.device)
        # Enable half precision if requested
        if self.half_precision:
            model = model.half()
        
        # Model optimization
        for param in model.parameters():
            param.requires_grad = False
        
        logging.info(f"YOLO-One model loaded from {model_path}")
        return model
    
    def _warmup_gpu(self, iterations: int):
        """Warmup GPU for optimal inference performance"""
        
        logging.info(f"Warming up GPU for {iterations} iterations...")
        
        dummy_input = torch.randn(
            1, 3, self.input_size, self.input_size, 
            device=self.device
        )
        
        if self.half_precision:
            dummy_input = dummy_input.half()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model(dummy_input)
        
        torch.cuda.empty_cache()
        logging.info("GPU warmup completed")
    
    def predict_image(
        self, 
        image: Union[str, np.ndarray], 
        return_crops: bool = False,
        visualize: bool = False
    ) -> Dict:
        """
        Predict on single image with YOLO-One
        
        Args:
            image: Image path or numpy array (BGR format)
            return_crops: Return cropped detections
            visualize: Return visualization image
            
        Returns:
            Dictionary with detections and metadata
        """
        
        # Load image
        if isinstance(image, str):
            original_image = cv2.imread(image)
            if original_image is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            original_image = image.copy()
        
        # Preprocessing
        start_time = time.time()
        preprocessed_tensor, scale_factor, padding = self._preprocess_image(original_image)
        preprocessing_time = time.time() - start_time
        
        # YOLO-One inference
        start_time = time.time()
        with torch.no_grad():
            # Activate decoding directly in the model head
            predictions = self.model(preprocessed_tensor, decode=True, img_size=preprocessed_tensor.shape[2:])
        inference_time = time.time() - start_time
        
        # Postprocessing
        start_time = time.time()
        # Use the new simplified post-processor on the 'decoded' outputs
        detections = self._postprocess_decoded_predictions(
            predictions, scale_factor, padding, original_image.shape[:2]
        )
        postprocessing_time = time.time() - start_time
        
        # Store timing
        self.preprocessing_times.append(preprocessing_time)
        self.inference_times.append(inference_time)
        self.postprocessing_times.append(postprocessing_time)
        
        # Prepare results
        results = {
            'detections': detections,
            'image_shape': original_image.shape[:2],
            'num_detections': len(detections['boxes']),
            'inference_time': inference_time,
            'total_time': preprocessing_time + inference_time + postprocessing_time,
            'confidence_scores': detections['scores'],
            'aspect_ratios': detections.get('aspects', [])
        }
        
        # Optional features
        if return_crops and len(detections['boxes']) > 0:
            results['crops'] = self._extract_crops(original_image, detections)
        
        if visualize:
            results['visualization'] = self._visualize_detections(original_image, detections)
        
        return results
    
    def predict_batch(
        self, 
        images: List[Union[str, np.ndarray]], 
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Batch prediction with YOLO-One
        
        Args:
            images: List of image paths or numpy arrays
            batch_size: Processing batch size
            
        Returns:
            List of detection results
        """
        
        results = []
        total_batches = (len(images) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(images), batch_size):
            batch = images[batch_idx:batch_idx + batch_size]
            
            print(f"Processing batch {batch_idx//batch_size + 1}/{total_batches}")
            
            batch_results = self._predict_batch_internal(batch)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_internal(self, batch: List[Union[str, np.ndarray]]) -> List[Dict]:
        """Internal batch prediction for YOLO-One"""
        
        # Preprocess batch
        original_images = []
        preprocessed_tensors = []
        scale_factors = []
        paddings = []
        
        for image in batch:
            # Load image
            if isinstance(image, str):
                orig_img = cv2.imread(image)
                if orig_img is None:
                    continue
            else:
                orig_img = image.copy()
            
            # Preprocess
            prep_tensor, scale_factor, padding = self._preprocess_image(orig_img)
            
            original_images.append(orig_img)
            preprocessed_tensors.append(prep_tensor.squeeze(0))
            scale_factors.append(scale_factor)
            paddings.append(padding)
        
        if not preprocessed_tensors:
            return []
        
        # Stack batch tensors
        batch_tensor = torch.stack(preprocessed_tensors, dim=0)
        
        # YOLO-One batch inference
        start_time = time.time()
        with torch.no_grad():
            batch_predictions = self.model(batch_tensor, decode=True, img_size=batch_tensor.shape[2:])
        inference_time = time.time() - start_time
        
        # Process each image in batch
        results = []
        for i, (orig_img, scale_factor, padding) in enumerate(
            zip(original_images, scale_factors, paddings)
        ):
            # Extract predictions for current image
            img_predictions = self._extract_single_prediction(batch_predictions, i)
            
            # Postprocess
            detections = self._postprocess_decoded_predictions(
                img_predictions, scale_factor, padding, orig_img.shape[:2]
            )
            
            results.append({
                'detections': detections,
                'image_shape': orig_img.shape[:2],
                'num_detections': len(detections['boxes']),
                'inference_time': inference_time / len(original_images)
            })
        
        return results
    
    def _extract_single_prediction(self, batch_predictions: Dict, batch_idx: int) -> Dict:
        """Extract predictions for single image from batch"""
        
        single_predictions = {}
        
        for key, pred_list in batch_predictions.items():
            single_predictions[key] = [pred[batch_idx:batch_idx+1] for pred in pred_list]
        
        return single_predictions
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """
        Preprocess image for YOLO-One inference
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (preprocessed_tensor, scale_factor, padding)
        """
        
        original_height, original_width = image.shape[:2]
        
        # Calculate scale factor
        scale_factor = min(
            self.input_size / original_width,
            self.input_size / original_height
        )
        
        # Resize image
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        resized = cv2.resize(image, (new_width, new_height))
        
        # Calculate padding
        pad_x = self.input_size - new_width
        pad_y = self.input_size - new_height
        pad_left = pad_x // 2
        pad_top = pad_y // 2
        
        # Apply padding
        padded = cv2.copyMakeBorder(
            resized, 
            pad_top, pad_y - pad_top,
            pad_left, pad_x - pad_left,
            cv2.BORDER_CONSTANT, 
            value=(114, 114, 114)
        )
        
        # Convert to tensor
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_image).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = tensor.to(self.device)
        
        if self.half_precision:
            tensor = tensor.half()
        
        return tensor, scale_factor, (pad_left, pad_top)
    
    def _postprocess_decoded_predictions(
        self, 
        predictions: Dict[str, List[torch.Tensor]],
        scale_factor: float, 
        padding: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> Dict:
        """
        Postprocess the already decoded predictions from the model head.
        This function handles filtering, NMS, and rescaling.
        
        Args:
            predictions: Dictionary from the model, must contain the 'decoded' key.
            scale_factor: Image scale factor
            padding: Applied padding (left, top)
            original_shape: Original image shape (height, width)
            
        Returns:
            Dictionary with final processed detections.
        """
        # The 'decoded' key contains a list of tensors [B, 5, Hk, Wk]
        # where 5 is (x_c, y_c, w, h, conf), all normalized to image size.
        # Concatenate predictions from all levels
        # [B, 5, H, W] -> [B, H, W, 5] -> [B, N, 5]        
        all_preds = [p.permute(0, 2, 3, 1).reshape(p.shape[0], -1, 5) for p in predictions['decoded']]
        preds = torch.cat(all_preds, dim=1).squeeze(0)  # [N, 5] for a single image

        # Filter by confidence
        conf_mask = preds[:, 4] >= self.confidence_threshold
        preds = preds[conf_mask]

        if preds.shape[0] == 0:
            return {'boxes': np.empty((0, 4)), 'scores': np.empty(0), 'num_detections': 0}

        # Convert from (center_x, center_y, width, height) to (x1, y1, x2, y2)
        box_cxcywh = preds[:, :4]
        boxes_xyxy = torch.empty_like(box_cxcywh)
        boxes_xyxy[:, 0] = box_cxcywh[:, 0] - box_cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = box_cxcywh[:, 1] - box_cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = box_cxcywh[:, 0] + box_cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = box_cxcywh[:, 1] + box_cxcywh[:, 3] / 2
        scores = preds[:, 4]

        # Apply NMS
        keep_indices = self._apply_nms(boxes_xyxy, scores)
        final_boxes = boxes_xyxy[keep_indices]
        final_scores = scores[keep_indices]

        # Limit max detections
        if len(final_boxes) > self.max_detections:
            final_boxes = final_boxes[:self.max_detections]
            final_scores = final_scores[:self.max_detections]

        # Rescale boxes to original image coordinates
        if len(final_boxes) > 0:
            final_boxes = self._rescale_boxes(final_boxes, scale_factor, padding, original_shape)

        result = {
            'boxes': final_boxes.cpu().numpy(),
            'scores': final_scores.cpu().numpy(),
            'num_detections': len(final_boxes)
        }
        return result
    
    def _apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply Non-Maximum Suppression"""
        
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        
        # Use torchvision NMS if available
        try:
            from torchvision.ops import nms
            return nms(boxes, scores, self.nms_threshold)
        except ImportError:
            # Fallback to custom NMS
            return self._custom_nms(boxes, scores)
    
    def _custom_nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Custom NMS implementation"""
        
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep = []
        while len(sorted_indices) > 0:
            current = sorted_indices[0]
            keep.append(current)
            
            if len(sorted_indices) == 1:
                break
            
            current_box = boxes[current:current+1]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            iou = self._compute_iou(current_box, remaining_boxes).squeeze(0)
            mask = iou <= self.nms_threshold
            sorted_indices = sorted_indices[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long, device=self.device)
    
    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between boxes"""
        
        boxes1 = boxes1.unsqueeze(1)
        boxes2 = boxes2.unsqueeze(0)
        
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / torch.clamp(union_area, min=1e-6)
    
    def _rescale_boxes(
        self, 
        boxes: torch.Tensor, 
        scale_factor: float, 
        padding: Tuple[int, int], 
        original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Rescale boxes to original image coordinates"""
        
        if len(boxes) == 0:
            return boxes
        
        pad_left, pad_top = padding
        original_height, original_width = original_shape
        
        # Convert from normalized to input image coordinates
        boxes = boxes * self.input_size
        
        # Remove padding
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        
        # Scale back to original image
        boxes = boxes / scale_factor
        
        # Clamp to image boundaries
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, original_width)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, original_height)
        
        return boxes
    
    def _extract_crops(self, image: np.ndarray, detections: Dict) -> List[np.ndarray]:
        """Extract cropped regions from detections"""
        
        crops = []
        boxes = detections['boxes']
        
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                crops.append(crop)
        
        return crops
    
    def _visualize_detections(self, image: np.ndarray, detections: Dict) -> np.ndarray:
        """Create visualization with bounding boxes"""
        
        vis_image = image.copy()
        boxes = detections['boxes']
        scores = detections['scores']
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f'{score:.2f}'
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics"""
        
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000,
            'avg_preprocessing_time_ms': np.mean(self.preprocessing_times) * 1000,
            'avg_postprocessing_time_ms': np.mean(self.postprocessing_times) * 1000,
            'avg_total_time_ms': np.mean([
                p + i + post for p, i, post in zip(
                    self.preprocessing_times, 
                    self.inference_times, 
                    self.postprocessing_times
                )
            ]) * 1000,
            'fps': 1.0 / np.mean(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times) * 1000,
            'max_inference_time_ms': np.max(self.inference_times) * 1000,
            'total_predictions': len(self.inference_times)
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times.clear()
        self.preprocessing_times.clear()
        self.postprocessing_times.clear()


def create_yolo_one_inference(
    model_path: str,
    device: str = 'cuda',
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    input_size: int = 640,
    half_precision: bool = True
) -> YoloOneInference:
    """
    Factory function to create YOLO-One inference engine
    
    Args:
        model_path: Path to trained YOLO-One model
        device: Computation device
        confidence_threshold: Detection confidence threshold
        nms_threshold: NMS IoU threshold
        input_size: Model input size
        half_precision: Use FP16 for speed optimization
        
    Returns:
        Configured YOLO-One inference engine
    """
    
    return YoloOneInference(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        input_size=input_size,
        half_precision=half_precision
    )


# Example usage
if __name__ == "__main__":
    
    # Initialize YOLO-One inference engine
    inference_engine = create_yolo_one_inference(
        model_path="/home/ibra/Documents/iatrax/YOLO-One/runs/train_20250813_123730/final_model.pt",
        device="cuda",
        confidence_threshold=0.05,
        nms_threshold=0.15
    )
    
    # Single image prediction
    results = inference_engine.predict_image(
        "/home/ibra/Documents/iatrax/YOLO-One/datasets_test/images/train/img1.jpg", 
        return_crops=True,
        visualize=True
    )
    
    print(f"Detected {results['num_detections']} objects")
    print(f"Inference time: {results['inference_time']*1000:.2f}ms")
    img_save = results['visualization']
    cv2.imwrite('output1.jpg', img_save)
    # Performance statistics
    stats = inference_engine.get_performance_stats()
    print(f"Average FPS: {stats['fps']:.1f}")
    print(f"Average total time: {stats['avg_total_time_ms']:.2f}ms")
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
import rerun as rr
# Import your YOLO-One model
from yolo_one.models.yolo_one_model import YoloOne
from yolo_one.data.postprocessing import YoloOnePostProcessor

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
        
        # Centralized post-processor
        self.post_processor = YoloOnePostProcessor(
            conf_threshold=self.confidence_threshold,
            iou_threshold=self.nms_threshold,
            max_detections=self.max_detections,
            input_size=(self.input_size, self.input_size)
        )
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
        image: np.ndarray, 
        return_crops: bool = False,
        visualize: bool = False
    ) -> Dict:
        """
        Predict on single image with YOLO-One
        
        Args:
            image: Image numpy array (BGR format)
            return_crops: Return cropped detections
            visualize: Return visualization image
            
        Returns:
            Dictionary with detections and metadata
        """
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
        
        # Delegate post-processing to the centralized processor
        results_list = self.post_processor.process_batch(
            predictions=predictions['decoded'],
            original_shapes=[original_image.shape[:2]],
            scale_factors=[scale_factor],
            paddings=[padding]
        )
        processed_detections = results_list[0]
        postprocessing_time = time.time() - start_time
        
        detections = self._format_output_detections(processed_detections)
        
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
        images: List[np.ndarray], 
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Batch prediction with YOLO-One
        
        Args:
            images: List of image numpy arrays
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
    
    def _predict_batch_internal(self, batch: List[np.ndarray]) -> List[Dict]:
        """Internal batch prediction for YOLO-One"""
        
        # Preprocess batch
        original_images = []
        preprocessed_tensors = []
        scale_factors = []
        paddings = []
        
        for image in batch:
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
            zip(original_images, scale_factors, paddings) # This was missing paddings
        ):
            # Extract predictions for current image
            img_predictions = {key: [p[i:i+1] for p in pred_list] for key, pred_list in batch_predictions.items()}
            
            # Postprocess
            processed_detections_list = self.post_processor.process_batch(
                predictions=img_predictions['decoded'],
                original_shapes=[orig_img.shape[:2]],
                scale_factors=[scale_factor],
                paddings=[padding]
            )
            detections = self._format_output_detections(processed_detections_list[0])
            
            results.append({
                'detections': detections,
                'image_shape': orig_img.shape[:2],
                'num_detections': len(detections['boxes']),
                'inference_time': inference_time / len(original_images)
            })
        
        return results
    
    def _extract_single_prediction(self, batch_predictions: Dict, batch_idx: int) -> Dict:
        """No longer needed, post-processor handles batch logic."""
        pass

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
    
    def _format_output_detections(self, processed_detections: Dict) -> Dict:
        """Converts post-processor output (tensors) to numpy for the final API result."""
        return {
            'boxes': processed_detections['boxes'].cpu().numpy(),
            'scores': processed_detections['scores'].cpu().numpy(),
            'num_detections': processed_detections['count']
        }

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

def log_to_rerun(image: np.ndarray, detections: Dict, stream_name: str = "inference"):
    """
    Logger image + bounding boxes dans rerun
    Args:
        image: np.ndarray (BGR)
        detections: dict avec 'boxes' et 'scores'
        stream_name: chemin rerun (ex: "inference/cam1")
    """

    # Convertir BGR -> RGB (Rerun attend RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rr.log(f"{stream_name}/image", rr.Image(img_rgb))

    if detections["num_detections"] > 0:
        boxes = detections["boxes"]
        scores = detections["scores"]

        # Convertir (x1, y1, x2, y2) -> (x, y, w, h)
        boxes_xywh = []
        labels = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            labels.append(f"{scores[i]:.2f}")

        # Logger les bounding boxes dans rerun
        rr.log(
            f"{stream_name}/detections",
            rr.Boxes2D(
                boxes_xywh,
                array_id="image",  # associer aux images
                labels=labels
            )
        )


if __name__ == "__main__":
    # Initialiser rerun
    
    rr.init("YOLO-One Inference", spawn=True)

    # Initialiser YOLO-One inference engine
    inference_engine = create_yolo_one_inference(
        model_path="/YOLO-One/runs/train_20250818_192510/final_model.pt",
        device="cuda",
        confidence_threshold=0.25,
        nms_threshold=0.45
    )

    # ----------- MODE IMAGE UNIQUE ------------
    image_path = "/YOLO-One/datasets/images/test/The-Curve-Atrium_mp4-8_jpg.rf.d21b38d61f9f727604859cd2274761e2.jpg"
    results = inference_engine.predict_image(image_path, return_crops=True, visualize=False)

    print(f"Detected {results['num_detections']} objects")
    print(f"Inference time: {results['inference_time']*1000:.2f}ms")

    img_save = results['visualization']
    cv2.imwrite('output.jpg', img_save)

    # Logger l’image et les prédictions dans rerun
    original_img = cv2.imread(image_path)
    log_to_rerun(original_img, results["detections"], stream_name="image_test")

    # Décommente pour activer la webcam ou une vidéo
    """
    cap = cv2.VideoCapture(0)  # mettre chemin vidéo à la place de 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prédiction YOLO-One
        results = inference_engine.predict_image(frame, visualize=False)

        # Logger dans rerun
        log_to_rerun(frame, results["detections"], stream_name="video_stream")

        # (optionnel) affichage local OpenCV
        vis = inference_engine._visualize_detections(frame, results["detections"])
        cv2.imshow("YOLO-One", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    """

    # ----------- STATISTIQUES PERF ------------
    stats = inference_engine.get_performance_stats()
    print(f"Average FPS: {stats['fps']:.1f}")
    print(f"Average total time: {stats['avg_total_time_ms']:.2f}ms")

"""

# Example usage
if __name__ == "__main__":

    rr.init("yolo-one-inference", spawn=True)

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="YOLO-One Inference Example")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights file.')
    parser.add_argument('--source', type=str, required=True, help='Path to input image.')
    parser.add_argument('--output', type=str, default='output_inference.jpg', help='Path to save output image.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda, cpu).')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS.')
    args = parser.parse_args()

    
    # Initialize YOLO-One inference engine
    inference_engine = create_yolo_one_inference(
        model_path=args.weights,
        device=args.device,
        confidence_threshold=args.conf,
        nms_threshold=args.iou
    )
    
    # Load the image before passing it to the engine
    image_path = Path(args.source)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {args.source}")
    image_bgr = cv2.imread(str(image_path))
    
    # Single image prediction
    results = inference_engine.predict_image(
        image_bgr, 
        return_crops=True,
        visualize=True
    )
    
    print(f"Detected {results['num_detections']} objects")
    print(f"Inference time: {results['inference_time']*1000:.2f}ms")

    img_save = results['visualization']
    cv2.imwrite('output.jpg', img_save)

    img = cv2.cvtColor(results["visualization"], cv2.COLOR_BGR2RGB)  # rerun attend du RGB
    rr.log("image", rr.Image(img))
    
    # Batch prediction
    #image_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
    #batch_results = inference_engine.predict_batch(image_list, batch_size=4)

    
    if results['visualization'] is not None:
        cv2.imwrite(args.output, results['visualization'])
        print(f"Visualization saved to {args.output}")

    # Performance statistics
    stats = inference_engine.get_performance_stats()

    print(f"Average FPS: {stats['fps']:.1f}")
    print(f"Average total time: {stats['avg_total_time_ms']:.2f}ms")"""

    if stats:
        print(f"Average FPS: {stats['fps']:.1f}")
        print(f"Average total time: {stats['avg_total_time_ms']:.2f}ms")


import argparse
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Local imports
from yolo_one.models.yolo_one_model import YoloOne
from yolo_one.data.postprocessing import postprocess_yolo_one_outputs
from yolo_one.utils.general import get_device, load_checkpoint
from yolo_one.utils.visualize import draw_bounding_boxes, log_detections
import rerun as rr

def preprocess_image(image_path: Path, input_size: tuple = (640, 640)) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess an image for YOLO-One inference."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    original_shape = image.shape[:2]  # (height, width)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize and pad
    h, w, _ = image_rgb.shape
    scale = min(input_size[0] / h, input_size[1] / w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    padded_image = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    dw, dh = (input_size[1] - resized_w) // 2, (input_size[0] - resized_h) // 2
    padded_image[dh:resized_h + dh, dw:resized_w + dw, :] = resized_image

    # Normalize and convert to tensor
    tensor_image = torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0
    return tensor_image.unsqueeze(0), image



def detect(args: argparse.Namespace):
    """Main detection function."""
    # Initialize Rerun
    rr.init("yolo_one_detection", spawn=True)
    device = get_device(args.device)

    # Load Model
    print("Loading model...")
    model = YoloOne(model_size=args.model_size).to(device)
    if args.weights:
        if not Path(args.weights).exists():
            raise FileNotFoundError(f"Weights file not found: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        # Adjust for different checkpoint formats (full vs. state_dict)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Remove the '_orig_mod.' prefix from keys
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully.")

    # Preprocess Image
    print(f"Processing image: {args.source}")
    input_tensor, original_image = preprocess_image(args.source)
    input_tensor = input_tensor.to(device)

    # Run Inference
    print("Running inference...")
    with torch.no_grad():
        raw_outputs = model(input_tensor)

    # Post-process
    detections = postprocess_yolo_one_outputs(
        raw_outputs['detections'],
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    print(f"Found {detections[0]['count']} objects.")

    # Log to Rerun
    log_detections(
        cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
        detections[0]['boxes'].numpy(),
        [f"object" for _ in range(detections[0]['count'])],
        detections[0]['scores'].numpy()
    )

    # Draw boxes and save output
    output_image = draw_bounding_boxes(
        original_image,
        detections[0]['boxes'].numpy(),
        [f"object" for _ in range(detections[0]['count'])],
        detections[0]['scores'].numpy()
    )
    output_path = Path(args.output_dir) / args.source.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output_image)
    print(f"Output image saved to: {output_path}")
    time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-One Detection Script")
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights file.')
    parser.add_argument('--source', type=Path, required=True, help='Path to input image.')
    parser.add_argument('--model-size', type=str, default='nano', choices=['nano', 'small', 'medium', 'large'], help='Model size.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS.')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., cpu, cuda, cuda:0).')
    parser.add_argument('--output-dir', type=str, default='./runs/detect', help='Directory to save output images.')

    args = parser.parse_args()

    detect(args)
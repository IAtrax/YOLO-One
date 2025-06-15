"""
Test YOLO-One Post-Processing
Iatrax Team - 2025 - https://iatrax.com
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
from yolo_one.models.yolo_one_model import YoloOne
from yolo_one.data.postprocessing import postprocess_yolo_one_outputs

def test_postprocessing():
    """Test post-processing pipeline"""
    
    print("YOLO-One Post-Processing Test")
    print("=" * 50)
    
    # Create model and get predictions
    model = YoloOne(model_size='nano')
    model.eval()
    
    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 640, 640)
    
    with torch.no_grad():
        raw_outputs = model(test_input)
    
    print(f"Raw outputs shapes:")
    for i, output in enumerate(raw_outputs):
        print(f"  P{i+3}: {output.shape}")
    
    # Test post-processing
    print(f"\n⚡ Running post-processing...")
    
    detections = postprocess_yolo_one_outputs(
        raw_outputs,
        conf_threshold=0.1,  # Low threshold for testing
        iou_threshold=0.40,
        max_detections=100
    )
    
    print(f"✅ Post-processing completed!")
    print(f"Batch size: {len(detections)}")
    
    # Analyze results
    for batch_idx, detection in enumerate(detections):
        print(f"\nImage {batch_idx + 1} results:")
        print(f"  Detections found: {detection['count']}")
        
        if detection['count'] > 0:
            print(f"  Boxes shape: {detection['boxes'].shape}")
            print(f"  Score range: {detection['scores'].min():.4f} - {detection['scores'].max():.4f}")
            print(f"  Top 3 scores: {detection['scores'][:3].tolist()}")
            
            # Show first detection details
            if detection['count'] > 0:
                box = detection['boxes'][0]
                score = detection['scores'][0]
                print(f"  First detection: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] conf={score:.4f}")
    
    return True

if __name__ == "__main__":
    test_postprocessing()
"""
Test YOLO-One Post-Processing
Iatrax Team - 2025 - https://iatrax.com
"""
import torch
from yolo_one.models.yolo_one_model import YoloOne
from yolo_one.data.postprocessing import postprocess_yolo_one_outputs

def test_postprocessing():
    """Test post-processing pipeline"""
    model = YoloOne(model_size='nano')
    model.eval()
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 640, 640)
    with torch.no_grad():
        raw_outputs = model(test_input)
    assert isinstance(raw_outputs, dict)
    assert 'detections' in raw_outputs
    detections = postprocess_yolo_one_outputs(
        raw_outputs['detections'],
        conf_threshold=0.1,
        iou_threshold=0.40,
        max_detections=100
    )
    assert isinstance(detections, list)
    assert len(detections) == batch_size

if __name__ == "__main__":
    test_postprocessing()
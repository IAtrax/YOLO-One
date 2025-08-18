"""
Test YOLO-One Post-Processing
Iatrax Team - 2025 - https://iatrax.com

Comprehensive test suite for the YOLO-One post-processing pipeline.
This suite validates the entire post-processing workflow, including:
- Confidence and class probability filtering.
- Non-Maximum Suppression (NMS) for duplicate removal.
- Coordinate conversion and rescaling to original image dimensions.
- Handling of edge cases (e.g., no detections, empty batches).
"""

import torch
import pytest
from typing import List

from yolo_one.data.postprocessing import YoloOnePostProcessor

# --- Fixtures for Test Setup ---

@pytest.fixture
def post_processor() -> YoloOnePostProcessor:
    """Provides a default YoloOnePostProcessor instance."""
    return YoloOnePostProcessor(
        conf_threshold=0.5,
        iou_threshold=0.5,
        max_detections=100,
        input_size=(640, 640)
    )

@pytest.fixture
def mock_decoded_predictions() -> List[torch.Tensor]:
    """
    Generates mock DECODED model predictions for a batch of 2 images.
    The format is (xc, yc, w, h, conf) normalized to the input size (640x640).

    - Image 1: Contains two high-confidence, overlapping detections.
    - Image 2: Contains one low-confidence detection that should be filtered out.
    """
    # Batch size = 2, 1 scale for simplicity
    # Shape: [B, 5, H, W]
    preds = torch.zeros(2, 5, 1, 2) # 2 predictions per image

    # Image 1, Detection 1 (high confidence)
    preds[0, :, 0, 0] = torch.tensor([320, 320, 64, 64, 0.9]) # xc, yc, w, h, conf

    # Image 1, Detection 2 (high confidence, overlapping)
    preds[0, :, 0, 1] = torch.tensor([330, 330, 64, 64, 0.85])

    # Image 2, Detection 1 (low confidence)
    preds[1, :, 0, 0] = torch.tensor([100, 100, 50, 50, 0.3])

    return [preds]

# --- Test Cases ---

def test_initialization(post_processor: YoloOnePostProcessor):
    """Tests if the post-processor initializes with correct settings."""
    assert post_processor.conf_threshold == 0.5
    assert post_processor.iou_threshold == 0.5
    assert post_processor.max_detections == 100
    assert post_processor.input_size == (640, 640)

def test_confidence_filtering(post_processor: YoloOnePostProcessor):
    """Tests the confidence threshold filtering logic."""
    detections = torch.tensor([
        [10, 10, 20, 20, 0.9],  # Keep
        [10, 10, 20, 20, 0.4],  # Drop
        [10, 10, 20, 20, 0.6],  # Keep
    ])
    filtered = post_processor.filter_by_confidence(detections)
    assert filtered.shape[0] == 2

def test_nms(post_processor: YoloOnePostProcessor):
    """Tests the Non-Maximum Suppression implementation."""
    # xyxy format for boxes
    boxes = torch.tensor([
        [100, 100, 200, 200], # High score
        [110, 110, 210, 210], # High IoU, lower score
        [300, 300, 400, 400], # No overlap
    ])
    scores = torch.tensor([0.9, 0.85, 0.8])
    detections = torch.cat([boxes, scores.unsqueeze(1), scores.unsqueeze(1)], dim=1)

    keep = post_processor.apply_nms(detections)
    assert keep.shape[0] == 2 # Should keep the first and third boxes

def test_full_processing_pipeline(post_processor: YoloOnePostProcessor, mock_decoded_predictions: List[torch.Tensor]):
    """
    Tests the entire post-processing pipeline from decoded predictions to final detections.
    """
    batch_results = post_processor.process_batch(mock_decoded_predictions)

    assert isinstance(batch_results, list)
    assert len(batch_results) == 2  # Batch size

    # --- Image 1 Assertions ---
    detections1 = batch_results[0]
    assert detections1['count'] == 1  # NMS should remove one of the overlapping boxes
    assert detections1['boxes'].shape == (1, 4)
    assert detections1['scores'].shape == (1,)
    assert detections1['scores'][0] == 0.9  # The highest confidence detection should be kept

    # --- Image 2 Assertions ---
    detections2 = batch_results[1]
    assert detections2['count'] == 0  # Low confidence detection should be filtered
    assert detections2['boxes'].shape == (0, 4)

def test_rescaling(post_processor: YoloOnePostProcessor):
    """Tests the rescaling of bounding boxes to original image size."""
    detections = {
        'boxes': torch.tensor([[100., 100., 200., 200.]]),
        'scores': torch.tensor([0.9]),
        'count': 1
    }
    model_size = (640, 640)
    original_size = (1280, 1280)  # 2x scale

    rescaled = post_processor.rescale_detections(detections, model_size, original_size)

    expected_box = torch.tensor([[200., 200., 400., 400.]])
    torch.testing.assert_close(rescaled['boxes'], expected_box)

def test_empty_predictions(post_processor: YoloOnePostProcessor):
    """Tests that the pipeline handles empty prediction tensors gracefully."""
    empty_preds = [torch.zeros(1, 5, 10, 10)]
    results = post_processor.process_batch(empty_preds)

    assert len(results) == 1
    assert results[0]['count'] == 0

if __name__ == "__main__":
    pytest.main()
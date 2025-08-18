"""
YOLO-One Architecture Forward Pass Test
Iatrax Team - 2025 - https://iatrax.com
"""

import torch
import pytest
from yolo_one.models.yolo_one_model import YoloOne

def test_yolo_one_forward():
    """Test forward pass with random data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloOne(model_size='nano')
    model = model.to(device)
    model.eval()
    
    batch_size = 1
    channels = 3
    height = 640
    width = 640
    
    random_input = torch.randn(batch_size, channels, height, width).to(device)
    
    try:
        with torch.no_grad():
            outputs = model(random_input)
        
        assert 'detections' in outputs
        assert 'obj_logits' in outputs
        assert 'bbox' in outputs
        
    except Exception as e:
        pytest.fail(f"Forward pass failed with exception: {e}")

def test_individual_components():
    """Test individual components"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from yolo_one.models.yolo_one_backbone import create_yolo_one_backbone
        from yolo_one.models.yolo_one_neck import create_yolo_one_neck
        from yolo_one.models.yolo_one_head import create_yolo_one_head
        
        # Test Backbone
        backbone = create_yolo_one_backbone('nano').to(device)
        test_input = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            backbone_out = backbone(test_input)
        assert isinstance(backbone_out, list)
        assert len(backbone_out) > 0
        
        # Test Neck
        neck = create_yolo_one_neck('nano', in_channels=backbone.out_channels).to(device)
        with torch.no_grad():
            neck_out = neck(backbone_out)
        assert isinstance(neck_out, list)
        assert len(neck_out) > 0
        
        # Test Head
        head = create_yolo_one_head('nano', in_channels=neck.out_channels).to(device)
        with torch.no_grad():
            head_out = head(neck_out)
        assert isinstance(head_out, dict)
        assert 'detections' in head_out
        
    except Exception as e:
        pytest.fail(f"Component test failed with exception: {e}")

def test_benchmark_inference_speed():
    """Benchmark inference speed"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = YoloOne(model_size='nano').to(device)
        model.eval()
        
        # Warmup
        warmup_input = torch.randn(1, 3, 640, 640).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(warmup_input)
        
        # Benchmark
        import time
        num_runs = 50
        
        test_input = torch.randn(1, 3, 640, 640).to(device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(test_input)
                
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        assert avg_time > 0
        
    except Exception as e:
        pytest.fail(f"Benchmark failed with exception: {e}")

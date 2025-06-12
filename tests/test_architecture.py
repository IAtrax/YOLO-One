"""
YOLO-One Architecture Forward Pass Test
Iatrax Team - 2025 - https://iatrax.com
"""

import sys
import os

# Add root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
from yolo_one.models.yolo_one_model import YoloOne

def test_yolo_one_forward():
    """Test forward pass with random data"""
    
    print("🚀 YOLO-One Architecture Test - Iatrax")
    print("=" * 50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model instantiation
    print("\n📦 Creating model...")
    model = YoloOne(model_size='nano')
    model = model.to(device)
    model.eval()
    
    # Random input (batch_size=2 for testing)
    batch_size = 2
    channels = 3
    height = 640
    width = 640
    
    print(f"\n🎲 Generating random data...")
    print(f"Input shape: [{batch_size}, {channels}, {height}, {width}]")
    
    random_input = torch.randn(batch_size, channels, height, width).to(device)
    
    # Forward pass
    print(f"\n⚡ Forward pass...")
    
    try:
        with torch.no_grad():
            outputs = model(random_input)
        
        print("✅ Forward pass successful!")
        print("outputs:", outputs)
        print('outputs[0].shape:', outputs[0].shape)
        # Output analysis
        print(f"\n📊 Output analysis:")
        print(f"Number of scales: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            scale_name = ['P3', 'P4', 'P5'][i]
            print(f"{scale_name}: {output.shape}")
            
            # Channel verification
            expected_channels = 6  # 1 class + 4 bbox + 1 conf
            actual_channels = output.shape[1]
            
            if actual_channels == expected_channels:
                print(f"  ✅ Channels correct: {actual_channels}")
            else:
                print(f"  ❌ Channels incorrect: {actual_channels} (expected {expected_channels})")
        
        # Memory usage
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"\n💾 GPU Memory used: {memory_used:.1f} MB")
        
        # Model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📏 Model Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB (float32)")
        
        # Multi-resolution test
        print(f"\n🔄 Multi-resolution test...")
        test_sizes = [(320, 320), (416, 416), (480, 480), (640, 640)]
        
        for h, w in test_sizes:
            test_input = torch.randn(1, 3, h, w).to(device)
            try:
                with torch.no_grad():
                    test_outputs = model(test_input)
                print(f"  ✅ {h}x{w}: OK")
            except Exception as e:
                print(f"  ❌ {h}x{w}: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components"""
    
    print(f"\n🔧 Testing individual components...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from yolo_one.models.yolo_one_backbone import create_yolo_one_backbone
        from yolo_one.models.yolo_one_neck import PAFPN
        from yolo_one.models.yolo_one_head import YoloOneDetectionHead
        
        # Test Backbone
        backbone = create_yolo_one_backbone('nano').to(device)
        test_input = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            backbone_out = backbone(test_input)
        print(f"✅ Backbone: {[x.shape for x in backbone_out]}")
        
        # Test Neck
        neck = PAFPN(backbone.out_channels).to(device)
        with torch.no_grad():
            neck_out = neck(backbone_out)
        print(f"✅ Neck: {[x.shape for x in neck_out]}")
        
        # Test Head
        head = YoloOneDetectionHead(neck.out_channels, num_classes=1).to(device)
        with torch.no_grad():
            head_out = head(neck_out)
        print(f"✅ Head: {[x.shape for x in head_out]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_inference_speed():
    """Benchmark inference speed"""
    
    print(f"\n⏱️ Inference speed benchmark...")
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
        num_runs = 100
        total_time = 0
        
        test_input = torch.randn(1, 3, 640, 640).to(device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model(test_input)
                
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        fps = 1000 / avg_time
        
        print(f"✅ Average inference time: {avg_time:.2f} ms")
        print(f"✅ FPS: {fps:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 YOLO-One Architecture Test Suite")
    print("Iatrax Team - 2025")
    print("=" * 60)
    
    # Full test
    success1 = test_yolo_one_forward()
    
    # Component test
    success2 = test_individual_components()
    
    # Speed benchmark
    success3 = benchmark_inference_speed()
    
    print(f"\n" + "=" * 60)
    if success1 and success2 and success3:
        print("🎉 ALL TESTS PASSED! YOLO-One architecture fully functional!")
    else:
        print("❌ Some tests failed. Check errors above.")
    print("=" * 60)
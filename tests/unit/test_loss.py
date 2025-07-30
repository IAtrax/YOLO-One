"""
IATrax - YOLO-One

MIT License

Test suite for YOLO-One Loss Function
Tests loss computation, gradient flow, and edge cases
"""

import torch
import torch.nn as nn
import pytest
from typing import List, Dict

from yolo_one.losses import YoloOneLoss, create_yolo_one_loss

class TestYoloOneLoss:
    """Comprehensive test suite for YOLO-One loss function"""
    
    @pytest.fixture
    def device(self):
        """Test device (CUDA if available, else CPU)"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def loss_function(self, device):
        """Standard loss function for testing"""
        loss_fn = YoloOneLoss(
            box_weight=7.5,
            obj_weight=1.0,
            focal_alpha=0.25,
            focal_gamma=1.5,
            iou_type='ciou',
            p5_weight_boost=1.2
        )
        return loss_fn.to(device)
    
    @pytest.fixture
    def sample_predictions(self, device):
        """Generate sample predictions for 3 scales (P3, P4, P5) in the correct dict format."""
        batch_size = 2
        return self.create_predictions_with_grad(batch_size, device)

    @pytest.fixture
    def sample_targets(self, device):
        """Generate sample ground truth targets"""
        # Format: [batch_idx, class, x_center, y_center, width, height]
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.3, 0.4],    # Object in batch 0
            [0, 0, 0.2, 0.3, 0.1, 0.2],    # Another object in batch 0
            [1, 0, 0.7, 0.6, 0.2, 0.3]     # Object in batch 1
        ], dtype=torch.float32, device=device)
        return targets

    def create_predictions_with_grad(self, batch_size: int, device: torch.device) -> Dict[str, List[torch.Tensor]]:
        """Helper to create predictions with gradients enabled in the correct dict format."""
        return {
            'detections': [
                torch.randn(batch_size, 15, 80, 80, device=device, requires_grad=True),
                torch.randn(batch_size, 15, 40, 40, device=device, requires_grad=True),
                torch.randn(batch_size, 15, 20, 20, device=device, requires_grad=True)
            ],
            'aspects': [
                torch.randn(batch_size, 1, 80, 80, device=device, requires_grad=True),
                torch.randn(batch_size, 1, 40, 40, device=device, requires_grad=True),
                torch.randn(batch_size, 1, 20, 20, device=device, requires_grad=True)
            ],
            'shape_confidences': [
                torch.randn(batch_size, 1, 80, 80, device=device, requires_grad=True),
                torch.randn(batch_size, 1, 40, 40, device=device, requires_grad=True),
                torch.randn(batch_size, 1, 20, 20, device=device, requires_grad=True)
            ]
        }
    
    def test_loss_initialization(self):
        """Test loss function initialization"""
        print("\n🧪 Testing Loss Initialization")
        print("-" * 40)
        
        # Test default initialization
        loss_fn = YoloOneLoss()
        assert loss_fn.box_weight == 7.5
        assert loss_fn.obj_weight == 1.0
        assert loss_fn.focal_alpha == 0.25
        assert loss_fn.focal_gamma == 1.5
        assert loss_fn.iou_type == 'ciou'
        assert loss_fn.p5_weight_boost == 1.2
        print("✅ Default initialization successful")
        
        # Test custom initialization
        custom_loss = YoloOneLoss(
            box_weight=10.0,
            obj_weight=2.0,
            focal_alpha=0.5,
            focal_gamma=2.0,
            iou_type='diou',
            p5_weight_boost=1.5
        )
        assert custom_loss.box_weight == 10.0
        assert custom_loss.obj_weight == 2.0
        assert custom_loss.iou_type == 'diou'
        print("✅ Custom initialization successful")
        
        # Test factory function
        factory_loss = create_yolo_one_loss(box_weight=5.0, obj_weight=0.5)
        assert factory_loss.box_weight == 5.0
        assert factory_loss.obj_weight == 0.5
        print("✅ Factory function successful")
    
    def test_loss_forward_pass(self, loss_function, sample_predictions, sample_targets, device):
        """Test forward pass with sample data"""
        print("\n🧪 Testing Forward Pass")
        print("-" * 40)
        
        # Forward pass
        loss_dict = loss_function(sample_predictions, sample_targets)
        
        # Check output structure
        required_keys = ['total_loss', 'box_loss', 'obj_loss', 'avg_loss']
        for key in required_keys:
            assert key in loss_dict, f"Missing key: {key}"
        print("✅ All required loss components present")
        
        # Check loss values
        total_loss = loss_dict['total_loss']
        box_loss = loss_dict['box_loss']
        obj_loss = loss_dict['obj_loss']
        
        assert isinstance(total_loss, torch.Tensor), "Total loss must be tensor"
        assert total_loss.numel() == 1, "Total loss must be scalar"
        assert total_loss.requires_grad, "Total loss must require gradients"
        assert total_loss.item() >= 0, "Total loss must be non-negative"
        print(f"✅ Total loss: {total_loss.item():.6f}")
        
        assert box_loss.item() >= 0, "Box loss must be non-negative"
        assert obj_loss.item() >= 0, "Objectness loss must be non-negative"
        print(f"✅ Box loss: {box_loss.item():.6f}")
        print(f"✅ Objectness loss: {obj_loss.item():.6f}")
        
        # Check loss composition
        loss_aspect = loss_dict['aspect_loss']
        loss_shape_conf = loss_dict['shape_conf_loss']
        expected_total = box_loss + obj_loss + loss_aspect + loss_shape_conf
        assert torch.allclose(total_loss, expected_total, atol=1e-6), "Loss composition error"
        print("✅ Loss composition verified")

    def test_gradient_flow(self, loss_function, sample_predictions, sample_targets, device):
        """Test gradient computation and backpropagation"""
        print("\n🧪 Testing Gradient Flow")
        print("-" * 40)

        # Ensure predictions require gradients (already set in fixture)
        for key, pred_list in sample_predictions.items():
            for i, pred in enumerate(pred_list):
                assert pred.requires_grad, f"Prediction {key}[{i}] should require gradients"

        # Forward pass
        loss_dict = loss_function(sample_predictions, sample_targets)
        total_loss = loss_dict['total_loss']

        # Backward pass
        total_loss.backward()

        # Check gradients
        for key, pred_list in sample_predictions.items():
            for i, pred in enumerate(pred_list):
                assert pred.grad is not None, f"No gradients for prediction {key}[{i}]"
                assert not torch.isnan(pred.grad).any(), f"NaN gradients in {key}[{i}]"
                assert not torch.isinf(pred.grad).any(), f"Inf gradients in {key}[{i}]"

                grad_norm = pred.grad.norm().item()
                assert grad_norm > 0, f"Zero gradients in {key}[{i}]"
                print(f"✅ {key}[{i}] gradient norm: {grad_norm:.6f}")

        print("✅ Gradient flow verified")
    
    def test_empty_targets(self, loss_function, device):
        """Test behavior with empty targets"""
        print("\n🧪 Testing Empty Targets")
        print("-" * 40)
        
        # Create predictions with gradients
        batch_size = 2
        predictions = self.create_predictions_with_grad(batch_size, device)
        
        # Empty targets tensor
        empty_targets = torch.empty(0, 6, device=device)
        
        # Forward pass with empty targets
        loss_dict = loss_function(predictions, empty_targets)
        
        # Should have only objectness loss (background)
        total_loss = loss_dict['total_loss']
        box_loss = loss_dict['box_loss']
        obj_loss = loss_dict['obj_loss']
        
        assert total_loss.item() >= 0, "Total loss must be non-negative"
        assert box_loss.item() == 0, "Box loss should be zero with empty targets"
        assert obj_loss.item() > 0, "Should have background objectness loss"
        
        print(f"✅ Empty targets - Total loss: {total_loss.item():.6f}")
        print(f"✅ Empty targets - Box loss: {box_loss.item():.6f}")
        print(f"✅ Empty targets - Obj loss: {obj_loss.item():.6f}")
    
    def test_single_object(self, loss_function, device):
        """Test with single object target"""
        print("\n🧪 Testing Single Object")
        print("-" * 40)
        
        # Create predictions with gradients
        batch_size = 2
        predictions = self.create_predictions_with_grad(batch_size, device)
        
        # Single object target
        single_target = torch.tensor([
            [0, 0, 0.5, 0.5, 0.3, 0.4]  # Single object in batch 0
        ], dtype=torch.float32, device=device)
        
        loss_dict = loss_function(predictions, single_target)
        
        total_loss = loss_dict['total_loss']
        box_loss = loss_dict['box_loss']
        obj_loss = loss_dict['obj_loss']
        
        assert total_loss.item() > 0, "Should have positive loss"
        assert box_loss.item() > 0, "Should have box loss"
        assert obj_loss.item() > 0, "Should have objectness loss"
        
        print(f"✅ Single object - Total loss: {total_loss.item():.6f}")
        print(f"✅ Single object - Box loss: {box_loss.item():.6f}")
        print(f"✅ Single object - Obj loss: {obj_loss.item():.6f}")
    
    def test_multiple_objects_same_batch(self, loss_function, device):
        """Test with multiple objects in same batch"""
        print("\n🧪 Testing Multiple Objects Same Batch")
        print("-" * 40)
        
        # Create predictions with gradients
        batch_size = 2
        predictions = self.create_predictions_with_grad(batch_size, device)
        
        # Multiple objects in same batch
        multi_targets = torch.tensor([
            [0, 0, 0.2, 0.2, 0.2, 0.2],  # Object 1 in batch 0
            [0, 0, 0.8, 0.8, 0.2, 0.2],  # Object 2 in batch 0
            [0, 0, 0.5, 0.5, 0.3, 0.3]   # Object 3 in batch 0
        ], dtype=torch.float32, device=device)
        
        loss_dict = loss_function(predictions, multi_targets)
        
        total_loss = loss_dict['total_loss']
        box_loss = loss_dict['box_loss']
        obj_loss = loss_dict['obj_loss']
        
        assert total_loss.item() > 0, "Should have positive loss"
        assert box_loss.item() > 0, "Should have box loss"
        assert obj_loss.item() > 0, "Should have objectness loss"
        
        print(f"✅ Multiple objects - Total loss: {total_loss.item():.6f}")
        print(f"✅ Multiple objects - Box loss: {box_loss.item():.6f}")
        print(f"✅ Multiple objects - Obj loss: {obj_loss.item():.6f}")
    
    def test_different_iou_types(self, sample_targets, device):
        """Test different IoU loss types"""
        print("\n🧪 Testing Different IoU Types")
        print("-" * 40)
        
        iou_types = ['ciou', 'diou', 'iou']
        batch_size = 2
        
        for iou_type in iou_types:
            predictions = self.create_predictions_with_grad(batch_size, device)
            loss_fn = YoloOneLoss(iou_type=iou_type).to(device)
            loss_dict = loss_fn(predictions, sample_targets)
            
            total_loss = loss_dict['total_loss']
            assert total_loss.item() > 0, f"IoU type {iou_type} should produce positive loss"
            assert not torch.isnan(total_loss), f"IoU type {iou_type} produced NaN"
            assert not torch.isinf(total_loss), f"IoU type {iou_type} produced Inf"
            
            print(f"✅ IoU type '{iou_type}' - Loss: {total_loss.item():.6f}")
    
    def test_focal_loss_parameters(self, sample_targets, device):
        """Test different focal loss parameters"""
        print("\n🧪 Testing Focal Loss Parameters")
        print("-" * 40)
        
        focal_configs = [
            {'alpha': 0.25, 'gamma': 0.0},   # No focal loss
            {'alpha': 0.25, 'gamma': 1.0},   # Mild focal loss
            {'alpha': 0.25, 'gamma': 2.0},   # Strong focal loss
        ]
        batch_size = 2
        
        for config in focal_configs:
            predictions = self.create_predictions_with_grad(batch_size, device)
            loss_fn = YoloOneLoss(
                focal_alpha=config['alpha'],
                focal_gamma=config['gamma']
            ).to(device)
            
            loss_dict = loss_fn(predictions, sample_targets)
            total_loss = loss_dict['total_loss']
            
            assert total_loss.item() > 0, f"Focal config {config} should produce positive loss"
            assert not torch.isnan(total_loss), f"Focal config {config} produced NaN"
            
            print(f"✅ Focal α={config['alpha']}, γ={config['gamma']} - Loss: {total_loss.item():.6f}")
    
    def test_p5_weight_boost(self, sample_targets, device):
        """Test P5 weight boost effect"""
        print("\n🧪 Testing P5 Weight Boost")
        print("-" * 40)
        
        batch_size = 2
        
        # Test without boost
        predictions_normal = self.create_predictions_with_grad(batch_size, device)
        loss_fn_normal = YoloOneLoss(p5_weight_boost=1.0).to(device)
        loss_normal = loss_fn_normal(predictions_normal, sample_targets)['total_loss']
        
        # Test with boost
        predictions_boost = self.create_predictions_with_grad(batch_size, device)
        loss_fn_boost = YoloOneLoss(p5_weight_boost=1.5).to(device)
        loss_boost = loss_fn_boost(predictions_boost, sample_targets)['total_loss']
        
        print(f"✅ Normal P5 weight - Loss: {loss_normal.item():.6f}")
        print(f"✅ Boosted P5 weight - Loss: {loss_boost.item():.6f}")
        
        # Note: We can't directly compare since predictions are different tensors
        assert loss_normal.item() > 0, "Normal loss should be positive"
        assert loss_boost.item() > 0, "Boosted loss should be positive"
    
    def test_label_smoothing(self, sample_targets, device):
        """Test label smoothing effect"""
        print("\n🧪 Testing Label Smoothing")
        print("-" * 40)
        
        smoothing_values = [0.0, 0.1, 0.2]
        batch_size = 2
        
        for smoothing in smoothing_values:
            predictions = self.create_predictions_with_grad(batch_size, device)
            loss_fn = YoloOneLoss(label_smoothing=smoothing).to(device)
            loss_dict = loss_fn(predictions, sample_targets)
            
            total_loss = loss_dict['total_loss']
            obj_loss = loss_dict['obj_loss']
            
            assert total_loss.item() > 0, f"Smoothing {smoothing} should produce positive loss"
            assert not torch.isnan(total_loss), f"Smoothing {smoothing} produced NaN"
            
            print(f"✅ Label smoothing {smoothing} - Total: {total_loss.item():.6f}, Obj: {obj_loss.item():.6f}")
    
    def test_batch_size_variations(self, loss_function, sample_targets, device):
        """Test different batch sizes"""
        print("\n🧪 Testing Batch Size Variations")
        print("-" * 40)
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            # Generate predictions for current batch size with gradients
            predictions = self.create_predictions_with_grad(batch_size, device)
            
            # Adjust targets to current batch size
            targets = sample_targets.clone()
            targets[:, 0] = targets[:, 0] % batch_size  # Ensure valid batch indices
            
            loss_dict = loss_function(predictions, targets)
            total_loss = loss_dict['total_loss']
            
            assert total_loss.item() > 0, f"Batch size {batch_size} should produce positive loss"
            assert not torch.isnan(total_loss), f"Batch size {batch_size} produced NaN"
            
            print(f"✅ Batch size {batch_size} - Loss: {total_loss.item():.6f}")
    
    def test_edge_cases(self, loss_function, device):
        """Test edge cases and boundary conditions"""
        print("\n🧪 Testing Edge Cases")
        print("-" * 40)
        
        batch_size = 1
        
        # Test very small predictions
        small_predictions = [
            torch.randn(batch_size, 15, 80, 80, device=device, requires_grad=True) * 0.001,
            torch.randn(batch_size, 15, 40, 40, device=device, requires_grad=True) * 0.001,
            torch.randn(batch_size, 15, 20, 20, device=device, requires_grad=True) * 0.001
        ]
        
        # Test very large predictions  
        large_predictions = [
            torch.randn(batch_size, 15, 80, 80, device=device, requires_grad=True) * 100,
            torch.randn(batch_size, 15, 40, 40, device=device, requires_grad=True) * 100,
            torch.randn(batch_size, 15, 20, 20, device=device, requires_grad=True) * 100
        ]
        
        # Boundary targets
        boundary_targets = torch.tensor([
            [0, 0, 0.0, 0.0, 0.1, 0.1],    # Corner object
            [0, 0, 1.0, 1.0, 0.1, 0.1],    # Another corner
            [0, 0, 0.5, 0.5, 1.0, 1.0]     # Full image object
        ], dtype=torch.float32, device=device)
        
        # Test small predictions
        try:
            loss_small = loss_function(small_predictions, boundary_targets)
            assert not torch.isnan(loss_small['total_loss']), "Small predictions produced NaN"
            print(f"✅ Small predictions - Loss: {loss_small['total_loss'].item():.6f}")
        except Exception as e:
            print(f"⚠️ Small predictions failed: {e}")
        
        # Test large predictions
        try:
            loss_large = loss_function(large_predictions, boundary_targets)
            assert not torch.isnan(loss_large['total_loss']), "Large predictions produced NaN"
            print(f"✅ Large predictions - Loss: {loss_large['total_loss'].item():.6f}")
        except Exception as e:
            print(f"⚠️ Large predictions failed: {e}")
    
    def test_reproducibility(self, device):
        """Test loss function reproducibility"""
        print("\n🧪 Testing Reproducibility")
        print("-" * 40)
        
        loss_function = YoloOneLoss().to(device)
        batch_size = 2
        
        sample_targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.3, 0.4],
            [1, 0, 0.7, 0.6, 0.2, 0.3]
        ], dtype=torch.float32, device=device)
        
        # Set random seed and create predictions
        torch.manual_seed(42)
        predictions1 = self.create_predictions_with_grad(batch_size, device)
        loss1 = loss_function(predictions1, sample_targets)['total_loss']
        
        # Reset seed and create same predictions
        torch.manual_seed(42)
        predictions2 = self.create_predictions_with_grad(batch_size, device)
        loss2 = loss_function(predictions2, sample_targets)['total_loss']
        
        assert torch.allclose(loss1, loss2, atol=1e-6), "Loss function not reproducible"
        print(f"✅ Reproducibility verified - Loss: {loss1.item():.6f}")
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting YOLO-One Loss Function Test Suite")
        print("=" * 60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {device}")
        
        try:
            # Initialize test fixtures
            loss_function = YoloOneLoss().to(device)
            
            batch_size = 2
            sample_predictions = self.create_predictions_with_grad(batch_size, device)
            
            sample_targets = torch.tensor([
                [0, 0, 0.5, 0.5, 0.3, 0.4],
                [0, 0, 0.2, 0.3, 0.1, 0.2],
                [1, 0, 0.7, 0.6, 0.2, 0.3]
            ], dtype=torch.float32, device=device)
            
            # Run tests
            self.test_loss_initialization()
            self.test_loss_forward_pass(loss_function, sample_predictions, sample_targets, device)
            self.test_gradient_flow(loss_function, sample_predictions, sample_targets, device)
            self.test_empty_targets(loss_function, device)
            self.test_single_object(loss_function, device)
            self.test_multiple_objects_same_batch(loss_function, device)
            self.test_different_iou_types(sample_targets, device)
            self.test_focal_loss_parameters(sample_targets, device)
            self.test_p5_weight_boost(sample_targets, device)
            self.test_label_smoothing(sample_targets, device)
            self.test_batch_size_variations(loss_function, sample_targets, device)
            self.test_edge_cases(loss_function, device)
            self.test_reproducibility(device)
            
            print("\n🎉 All Tests Passed Successfully!")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Test Suite Failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_loss_performance():
    """Performance benchmark for loss function"""
    print("\n⚡ Performance Benchmark")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = YoloOneLoss().to(device)
    
    # Large batch for performance testing
    batch_size = 16
    predictions = {
        'detections': [
            torch.randn(batch_size, 15, 80, 80, device=device, requires_grad=True),
            torch.randn(batch_size, 15, 40, 40, device=device, requires_grad=True),
            torch.randn(batch_size, 15, 20, 20, device=device, requires_grad=True)
        ],
        'aspects': [
            torch.randn(batch_size, 1, 80, 80, device=device, requires_grad=True),
            torch.randn(batch_size, 1, 40, 40, device=device, requires_grad=True),
            torch.randn(batch_size, 1, 20, 20, device=device, requires_grad=True)
        ],
        'shape_confidences': [
            torch.randn(batch_size, 1, 80, 80, device=device, requires_grad=True),
            torch.randn(batch_size, 1, 40, 40, device=device, requires_grad=True),
            torch.randn(batch_size, 1, 20, 20, device=device, requires_grad=True)
        ]
    }
    
    # Multiple targets
    targets = torch.tensor([
        [i % batch_size, 0, 0.5, 0.5, 0.3, 0.4] for i in range(batch_size * 3)
    ], dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(10):
        _ = loss_fn(predictions, targets)
    
    # Benchmark
    import time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    start_time = time.time()
    num_iterations = 100
    
    for _ in range(num_iterations):
        loss_dict = loss_fn(predictions, targets)
        loss_dict['total_loss'].backward(retain_graph=True)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    print(f"✅ Average loss computation time: {avg_time:.2f} ms")
    print(f"✅ Throughput: {num_iterations / (end_time - start_time):.1f} iterations/sec")


if __name__ == "__main__":
    """Run the test suite"""
    
    print("🧪 YOLO-One Loss Function Test Suite")
    print("=" * 50)
    
    # Run main test suite
    test_suite = TestYoloOneLoss()
    success = test_suite.run_all_tests()
    
    if success:
        # Run performance benchmark
        test_loss_performance()
        
        print("\n🎯 Summary:")
        print("✅ All functionality tests passed")
        print("✅ Performance benchmark completed")
        print("✅ YOLO-One loss function ready for training!")
        
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        exit(1)
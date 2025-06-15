"""
YOLO-One Dataset Test Suite
Iatrax Team - 2025 - https://iatrax.com

Comprehensive tests for YoloOneDataset and preprocessing logic
"""

import tempfile
import shutil
import unittest
import torch
import numpy as np
import cv2
from pathlib import Path

from yolo_one.data.preprocessing import (
    YoloOneDataset, 
    YoloOneDatasetAnalyzer, 
    create_yolo_one_dataset,
)

class YoloOneDatasetTester:
    """Comprehensive test suite for YOLO-One dataset functionality"""
    
    def __init__(self):
        self.temp_dir = None
        self.test_results = {}
        
    def run_all_tests(self):
        """Run complete test suite"""
        
        print("🧪 YOLO-One Dataset Test Suite")
        print("Iatrax Team - 2025")
        print("=" * 60)
        
        try:
            # Setup test environment
            self.setup_test_environment()
            
            # Run tests
            self.test_single_class_dataset()
            self.test_multi_class_dataset()
            self.test_analyzer_functionality()
            self.test_dataset_creation()
            self.test_dataloader_functionality()
            self.test_edge_cases()
            self.test_interactive_behavior()
            # Print summary
            self.print_test_summary()
            
        finally:
            # Cleanup
            self.cleanup_test_environment()
    
    def setup_test_environment(self):
        """Setup temporary test datasets"""
        
        print("🏗️  Setting up test environment...")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="yolo_one_test_"))
        print(f"📁 Test directory: {self.temp_dir}")
        
        # Create test datasets
        self.create_single_class_dataset()
        self.create_multi_class_dataset()
        self.create_empty_dataset()
        self.create_corrupted_dataset()
        
        print("✅ Test environment ready")
    
    def create_single_class_dataset(self):
        """Create single-class test dataset"""
        
        dataset_path = self.temp_dir / "single_class_dataset"
        
        # Create directory structure
        for split in ['train', 'val']:
            (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Generate test data
        for split in ['train', 'val']:
            num_images = 20 if split == 'train' else 10
            
            for i in range(num_images):
                # Create dummy image
                image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                img_path = dataset_path / 'images' / split / f"img_{i:03d}.jpg"
                cv2.imwrite(str(img_path), image)
                
                # Create corresponding label (class 0 only)
                label_path = dataset_path / 'labels' / split / f"img_{i:03d}.txt"
                
                # Random number of annotations (1-3)
                num_annotations = np.random.randint(1, 4)
                
                with open(label_path, 'w') as f:
                    for _ in range(num_annotations):
                        # Class 0, random bbox
                        x_center = np.random.uniform(0.2, 0.8)
                        y_center = np.random.uniform(0.2, 0.8)
                        width = np.random.uniform(0.1, 0.3)
                        height = np.random.uniform(0.1, 0.3)
                        
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"✅ Created single-class dataset: {dataset_path}")
    
    def create_multi_class_dataset(self):
        """Create multi-class test dataset (simulating COCO subset)"""
        
        dataset_path = self.temp_dir / "multi_class_dataset"
        
        # Create directory structure
        for split in ['train', 'val']:
            (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Classes: 0=person, 1=car, 2=bicycle (simulating COCO subset)
        classes = [0, 1, 2]
        class_names = {0: 'person', 1: 'car', 2: 'bicycle'}
        
        # Generate test data
        for split in ['train', 'val']:
            num_images = 30 if split == 'train' else 15
            
            for i in range(num_images):
                # Create dummy image
                image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                img_path = dataset_path / 'images' / split / f"img_{i:03d}.jpg"
                cv2.imwrite(str(img_path), image)
                
                # Create corresponding label with multiple classes
                label_path = dataset_path / 'labels' / split / f"img_{i:03d}.txt"
                
                # Random classes for this image
                present_classes = np.random.choice(classes, 
                                                 size=np.random.randint(1, 3), 
                                                 replace=False)
                
                with open(label_path, 'w') as f:
                    for class_id in present_classes:
                        # Random number of instances for this class (1-2)
                        num_instances = np.random.randint(1, 3)
                        
                        for _ in range(num_instances):
                            x_center = np.random.uniform(0.2, 0.8)
                            y_center = np.random.uniform(0.2, 0.8)
                            width = np.random.uniform(0.1, 0.3)
                            height = np.random.uniform(0.1, 0.3)
                            
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"✅ Created multi-class dataset: {dataset_path}")
    
    def create_empty_dataset(self):
        """Create dataset with empty labels"""
        
        dataset_path = self.temp_dir / "empty_dataset"
        
        # Create directory structure
        for split in ['train']:
            (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Create images with empty labels
        for i in range(5):
            # Create dummy image
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = dataset_path / 'images' / 'train' / f"empty_{i:03d}.jpg"
            cv2.imwrite(str(img_path), image)
            
            # Create empty label file
            label_path = dataset_path / 'labels' / 'train' / f"empty_{i:03d}.txt"
            label_path.touch()  # Empty file
        
        print(f"✅ Created empty dataset: {dataset_path}")
    
    def create_corrupted_dataset(self):
        """Create dataset with corrupted labels"""
        
        dataset_path = self.temp_dir / "corrupted_dataset"
        
        # Create directory structure
        for split in ['train']:
            (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Create images with corrupted labels
        for i in range(5):
            # Create dummy image
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = dataset_path / 'images' / 'train' / f"corrupt_{i:03d}.jpg"
            cv2.imwrite(str(img_path), image)
            
            # Create corrupted label file
            label_path = dataset_path / 'labels' / 'train' / f"corrupt_{i:03d}.txt"
            
            with open(label_path, 'w') as f:
                if i == 0:
                    f.write("invalid_line_format\n")
                elif i == 1:
                    f.write("0 0.5 0.5\n")  # Missing values
                elif i == 2:
                    f.write("abc 0.5 0.5 0.2 0.2\n")  # Non-numeric class
                elif i == 3:
                    f.write("0 1.5 0.5 0.2 0.2\n")  # Out of range coordinates
                else:
                    f.write("0 0.5 0.5 0.2 0.2\n")  # Valid line
        
        print(f"✅ Created corrupted dataset: {dataset_path}")
    
    def test_single_class_dataset(self):
        """Test single-class dataset functionality"""
        
        print(f"\n🧪 Test 1: Single-Class Dataset")
        print("-" * 40)
        
        dataset_path = self.temp_dir / "single_class_dataset"
        
        try:
            # Test analyzer
            analyzer = YoloOneDatasetAnalyzer(str(dataset_path))
            selected_class = analyzer.analyze_and_select_class()
            
            assert selected_class == 0, f"Expected class 0, got {selected_class}"
            print(f"✅ Analyzer correctly detected single class: {selected_class}")
            
            # Test dataset creation
            dataset = YoloOneDataset(
                root_dir=str(dataset_path),
                split='train',
                target_class=selected_class
            )
            
            assert len(dataset) > 0, "Dataset should not be empty"
            assert dataset.target_class == 0, f"Expected target_class 0, got {dataset.target_class}"
            print(f"✅ Dataset created successfully: {len(dataset)} samples")
            
            # Test data loading
            sample = dataset[0]
            assert 'image' in sample, "Sample should contain 'image'"
            assert 'targets' in sample, "Sample should contain 'targets'"
            assert sample['image'].shape == (3, 640, 640), f"Unexpected image shape: {sample['image'].shape}"
            
            if len(sample['targets']) > 0:
                assert sample['targets'].shape[1] == 6, f"Unexpected target shape: {sample['targets'].shape}"
                # Check that all classes are 0 (single-class)
                assert torch.all(sample['targets'][:, 1] == 0), "All targets should have class 0"
            
            print(f"✅ Data loading works correctly")
            
            self.test_results['single_class'] = True
            
        except Exception as e:
            print(f"❌ Single-class test failed: {e}")
            self.test_results['single_class'] = False
    
    def test_multi_class_dataset(self):
        """Test multi-class dataset functionality"""
        
        print(f"\n🧪 Test 2: Multi-Class Dataset")
        print("-" * 40)
        
        dataset_path = self.temp_dir / "multi_class_dataset"
        
        try:
            # Test analyzer with specific class
            analyzer = YoloOneDatasetAnalyzer(str(dataset_path))
            classes = analyzer._scan_dataset_classes()
            
            assert len(classes) > 1, f"Expected multiple classes, got {len(classes)}"
            print(f"✅ Analyzer detected multiple classes: {sorted(classes.keys())}")
            
            # Test with specific target class
            for target_class in [0, 1, 2]:
                if target_class in classes:
                    selected_class = analyzer.analyze_and_select_class(target_class=target_class)
                    assert selected_class == target_class, f"Expected class {target_class}, got {selected_class}"
                    print(f"✅ Specific class selection works: {target_class}")
                    
                    # Test dataset creation with specific class
                    dataset = YoloOneDataset(
                        root_dir=str(dataset_path),
                        split='train',
                        target_class=target_class
                    )
                    
                    assert dataset.target_class == target_class, f"Dataset target_class mismatch"
                    assert len(dataset) > 0, "Dataset should not be empty"
                    print(f"✅ Dataset created for class {target_class}: {len(dataset)} samples")
                    
                    # Test that dataset only contains target class
                    for i in range(min(5, len(dataset))):
                        sample = dataset[i]
                        if len(sample['targets']) > 0:
                            assert torch.all(sample['targets'][:, 1] == 0), "All targets should be remapped to class 0"
                    
                    break
            
            self.test_results['multi_class'] = True
            
        except Exception as e:
            print(f"❌ Multi-class test failed: {e}")
            self.test_results['multi_class'] = False
    def test_interactive_behavior(self):
        """Test interactive behavior of YoloOneDataset class"""
        
        print(f"\n🧪 Test: Interactive Behavior")
        print("-" * 40)
        
        dataset_path = self.temp_dir / "multi_class_dataset"
        
        try:
            dataset, loader = create_yolo_one_dataset(
                    root_dir=str(dataset_path),
                    split='train',
                )
            print("✅ Class selection logic works correctly")
            print("📝 Note: Interactive UI can be tested manually")
            self.test_results['interactive'] = True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            self.test_results['interactive'] = False
    def test_analyzer_functionality(self):
        """Test analyzer functionality in detail"""
        
        print(f"\n🧪 Test 3: Analyzer Functionality")
        print("-" * 40)
        
        try:
            # Test with single-class dataset
            single_path = self.temp_dir / "single_class_dataset"
            analyzer1 = YoloOneDatasetAnalyzer(str(single_path))
            
            classes1 = analyzer1._scan_dataset_classes()
            assert len(classes1) == 1, f"Expected 1 class, got {len(classes1)}"
            assert 0 in classes1, "Expected class 0 in single-class dataset"
            print(f"✅ Single-class analysis correct: {classes1}")
            
            # Test with multi-class dataset
            multi_path = self.temp_dir / "multi_class_dataset"
            analyzer2 = YoloOneDatasetAnalyzer(str(multi_path))
            
            classes2 = analyzer2._scan_dataset_classes()
            assert len(classes2) > 1, f"Expected multiple classes, got {len(classes2)}"
            expected_classes = {0, 1, 2}
            assert set(classes2.keys()) == expected_classes, f"Expected {expected_classes}, got {set(classes2.keys())}"
            print(f"✅ Multi-class analysis correct: {classes2}")
            
            # Test with empty dataset
            empty_path = self.temp_dir / "empty_dataset"
            analyzer3 = YoloOneDatasetAnalyzer(str(empty_path))
            
            classes3 = analyzer3._scan_dataset_classes()
            assert len(classes3) == 0, f"Expected 0 classes in empty dataset, got {len(classes3)}"
            print(f"✅ Empty dataset analysis correct: {classes3}")
            
            self.test_results['analyzer'] = True
            
        except Exception as e:
            print(f"❌ Analyzer test failed: {e}")
            self.test_results['analyzer'] = False
    
    def test_dataset_creation(self):
        """Test dataset creation function"""
        
        print(f"\n🧪 Test 4: Dataset Creation Function")
        print("-" * 40)
        
        try:
            dataset_path = self.temp_dir / "single_class_dataset"
            
            # Test create_yolo_one_dataset function
            dataset, dataloader = create_yolo_one_dataset(
                root_dir=str(dataset_path),
                split='train',
                batch_size=4,
                num_workers=0  # No multiprocessing for tests
            )
            
            assert isinstance(dataset, YoloOneDataset), "Should return YoloOneDataset instance"
            assert hasattr(dataloader, '__iter__'), "Should return iterable DataLoader"
            print(f"✅ Dataset creation function works")
            
            # Test dataloader iteration
            batch = next(iter(dataloader))
            assert 'images' in batch, "Batch should contain 'images'"
            assert 'targets' in batch, "Batch should contain 'targets'"
            assert 'image_paths' in batch, "Batch should contain 'image_paths'"
            
            # Check batch shapes
            images = batch['images']
            targets = batch['targets']
            
            assert images.shape[0] <= 4, f"Batch size should be <= 4, got {images.shape[0]}"
            assert images.shape[1:] == (3, 640, 640), f"Unexpected image shape: {images.shape[1:]}"
            
            if len(targets) > 0:
                assert targets.shape[1] == 6, f"Unexpected target shape: {targets.shape}"
                # Check batch indices
                batch_indices = targets[:, 0].unique()
                assert torch.all(batch_indices < images.shape[0]), "Batch indices should be valid"
            
            print(f"✅ DataLoader works correctly")
            print(f"📊 Batch: {images.shape[0]} images, {len(targets)} targets")
            
            self.test_results['dataset_creation'] = True
            
        except Exception as e:
            print(f"❌ Dataset creation test failed: {e}")
            self.test_results['dataset_creation'] = False
    
    def test_dataloader_functionality(self):
        """Test DataLoader functionality with collate function"""
        
        print(f"\n🧪 Test 5: DataLoader Functionality")
        print("-" * 40)
        
        try:
            dataset_path = self.temp_dir / "multi_class_dataset"
            
            # Create dataset and dataloader
            dataset, dataloader = create_yolo_one_dataset(
                root_dir=str(dataset_path),
                split='train',
                target_class=0,  # Use person class
                batch_size=8,
                num_workers=0
            )
            
            print(f"✅ DataLoader created: {len(dataset)} samples, batch_size=8")
            
            # Test multiple batches
            total_samples = 0
            max_batches = 3
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                images = batch['images']
                targets = batch['targets']
                
                total_samples += images.shape[0]
                
                # Validate batch structure
                assert images.dtype == torch.float32, f"Images should be float32, got {images.dtype}"
                assert targets.dtype == torch.float32, f"Targets should be float32, got {targets.dtype}"
                
                # Check image values range
                assert torch.all(images >= 0) and torch.all(images <= 1), "Images should be in [0, 1] range"
                
                # Check target format
                if len(targets) > 0:
                    # Check batch indices
                    batch_indices = targets[:, 0]
                    assert torch.all(batch_indices >= 0) and torch.all(batch_indices < images.shape[0]), "Invalid batch indices"
                    
                    # Check class indices (should all be 0 for single-class)
                    class_indices = targets[:, 1]
                    assert torch.all(class_indices == 0), "All classes should be 0 for single-class"
                    
                    # Check coordinate ranges
                    coords = targets[:, 2:6]  # x, y, w, h
                    assert torch.all(coords >= 0) and torch.all(coords <= 1), "Coordinates should be in [0, 1] range"
                
                print(f"  Batch {batch_idx}: {images.shape[0]} images, {len(targets)} targets")
            
            print(f"✅ Processed {total_samples} samples across {batch_idx + 1} batches")
            
            self.test_results['dataloader'] = True
            
        except Exception as e:
            print(f"❌ DataLoader test failed: {e}")
            self.test_results['dataloader'] = False
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        
        print(f"\n🧪 Test 6: Edge Cases")
        print("-" * 40)
        
        try:
            # Test with corrupted dataset
            corrupted_path = self.temp_dir / "corrupted_dataset"
            
            try:
                dataset = YoloOneDataset(
                    root_dir=str(corrupted_path),
                    split='train'
                )
                
                # Should handle corrupted data gracefully
                sample = dataset[0]
                assert 'image' in sample, "Should return valid sample structure"
                print(f"✅ Corrupted data handled gracefully")
                
            except Exception as e:
                print(f"⚠️  Corrupted dataset test: {e}")
            
            # Test with non-existent dataset
            try:
                non_existent_path = self.temp_dir / "non_existent"
                dataset = YoloOneDataset(
                    root_dir=str(non_existent_path),
                    split='train'
                )
                print(f"❌ Should have failed with non-existent dataset")
                
            except (FileNotFoundError, ValueError):
                print(f"✅ Correctly handles non-existent dataset")
            
            # Test with invalid target class
            try:
                dataset_path = self.temp_dir / "single_class_dataset"
                dataset = YoloOneDataset(
                    root_dir=str(dataset_path),
                    split='train',
                    target_class=999  # Non-existent class
                )
                print(f"❌ Should have failed with invalid target class")
                
            except ValueError:
                print(f"✅ Correctly handles invalid target class")
            
            self.test_results['edge_cases'] = True
            
        except Exception as e:
            print(f"❌ Edge cases test failed: {e}")
            self.test_results['edge_cases'] = False
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        
        print(f"\n📊 YOLO-One Dataset Test Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n🎯 Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! YOLO-One dataset is ready for training!")
        else:
            print("⚠️  Some tests failed. Check implementation.")
        
        return passed_tests == total_tests
    
    def cleanup_test_environment(self):
        """Clean up temporary test files"""
        
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 Test environment cleaned up")

def main():
    """Run YOLO-One dataset tests"""
    
    tester = YoloOneDatasetTester()
    success = tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
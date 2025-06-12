"""
YOLO-One Dataset Module with Integrated Analysis
Iatrax Team - 2025 - https://iatrax.com

Complete dataset handling with automatic class analysis and selection
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import albumentations as A
from pathlib import Path
from collections import Counter

class YoloDatasetAnalyzer:
    """Integrated analyzer for YOLO format datasets"""
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.splits = ['train', 'val', 'test']
        self.analysis_results = {}
        
    def analyze_dataset(self, verbose: bool = True) -> Dict:
        """Analyze complete dataset for class distribution"""
        
        if verbose:
            print("🔍 Analyzing YOLO-ONE Dataset Classes...")
            print("=" * 50)
        
        total_stats = {
            'all_classes': set(),
            'class_counts': Counter(),
            'split_stats': {},
            'total_files': 0,
            'total_annotations': 0,
            'empty_files': 0,
            'invalid_files': []
        }
        
        for split in self.splits:
            split_path = self.dataset_root / 'labels' / split
            
            if not split_path.exists():
                if verbose:
                    print(f"⚠️  Split '{split}' not found, skipping...")
                continue
            
            if verbose:
                print(f"📂 Analyzing {split} split...")
                
            split_stats = self._analyze_split(split_path, verbose)
            
            if split_stats:
                total_stats['split_stats'][split] = split_stats
                total_stats['all_classes'].update(split_stats['classes'])
                total_stats['class_counts'].update(split_stats['class_counts'])
                total_stats['total_files'] += split_stats['total_files']
                total_stats['total_annotations'] += split_stats['total_annotations']
                total_stats['empty_files'] += split_stats['empty_files']
                total_stats['invalid_files'].extend(split_stats['invalid_files'])
                
                if verbose:
                    print(f"  ✅ {split_stats['total_files']} files analyzed")
                    print(f"  📊 {split_stats['total_annotations']} annotations found")
                    print(f"  🎯 Classes: {sorted(split_stats['classes'])}")
        
        total_stats['all_classes'] = sorted(list(total_stats['all_classes']))
        self.analysis_results = total_stats
        return total_stats
    
    def _analyze_split(self, split_path: Path, verbose: bool = True) -> Optional[Dict]:
        """Analyze single split"""
        
        label_files = list(split_path.glob('*.txt'))
        
        if not label_files:
            return None
        
        split_stats = {
            'classes': set(),
            'class_counts': Counter(),
            'total_files': len(label_files),
            'total_annotations': 0,
            'empty_files': 0,
            'invalid_files': []
        }
        
        for label_file in label_files:
            try:
                file_stats = self._analyze_label_file(label_file)
                
                if file_stats['annotations'] == 0:
                    split_stats['empty_files'] += 1
                else:
                    split_stats['classes'].update(file_stats['classes'])
                    split_stats['class_counts'].update(file_stats['class_counts'])
                    split_stats['total_annotations'] += file_stats['annotations']
                        
            except Exception as e:
                split_stats['invalid_files'].append({
                    'file': str(label_file),
                    'error': str(e)
                })
        
        return split_stats
    
    def _analyze_label_file(self, label_file: Path) -> Dict:
        """Analyze single label file"""
        
        file_stats = {
            'classes': set(),
            'class_counts': Counter(),
            'annotations': 0,
            'invalid_lines': []
        }
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            if len(parts) != 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                       0 < width <= 1 and 0 < height <= 1):
                    continue
                
                file_stats['classes'].add(class_id)
                file_stats['class_counts'][class_id] += 1
                file_stats['annotations'] += 1
                
            except ValueError:
                continue
        
        return file_stats
    
    def print_analysis_summary(self):
        """Print concise analysis summary"""
        
        if not self.analysis_results:
            return
        
        stats = self.analysis_results
        
        print(f"\n📊 DATASET ANALYSIS SUMMARY:")
        print(f"  Total files: {stats['total_files']:,}")
        print(f"  Total annotations: {stats['total_annotations']:,}")
        print(f"  Classes found: {stats['all_classes']}")
        
        if len(stats['all_classes']) > 1:
            print(f"\n📈 Class distribution:")
            for class_id, count in stats['class_counts'].most_common():
                percentage = (count / stats['total_annotations']) * 100
                print(f"    Class {class_id}: {count:,} ({percentage:.1f}%)")

class YoloFormatParser:
    """Parser for YOLO format annotations"""
    
    @staticmethod
    def parse_yolo_label(label_path: str, img_width: int, img_height: int) -> List[Dict]:
        """Parse YOLO format label file"""
        
        annotations = []
        
        if not os.path.exists(label_path):
            return annotations
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                continue
                
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert relative to absolute coordinates
                abs_x_center = x_center * img_width
                abs_y_center = y_center * img_height
                abs_width = width * img_width
                abs_height = height * img_height
                
                # Convert to corner format
                x1 = abs_x_center - abs_width / 2
                y1 = abs_y_center - abs_height / 2
                x2 = abs_x_center + abs_width / 2
                y2 = abs_y_center + abs_height / 2
                
                annotations.append({
                    'class_id': class_id,
                    'bbox': [x1, y1, x2, y2],
                    'bbox_rel': [x_center, y_center, width, height],
                    'area': abs_width * abs_height
                })
                
            except (ValueError, IndexError):
                continue
                
        return annotations

class YoloOneDataset(Dataset):
    """
    Dataset class for YOLO-One training with integrated analysis
    Automatically analyzes dataset and handles class selection
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (640, 640),
        augmentations: Optional[A.Compose] = None,
        target_class: Optional[int] = None,
        auto_select_class: bool = True,
        cache_images: bool = False,
        verbose: bool = True
    ):
        """
        Initialize dataset with automatic analysis
        
        Args:
            root_dir: Root directory containing images/ and labels/ folders
            split: Dataset split ('train', 'val', 'test')
            img_size: Target image size (H, W)
            augmentations: Albumentations transforms
            target_class: Specific class to use (None for auto-detection)
            auto_select_class: Auto-select class if multiple found
            cache_images: Cache images in memory for speed
            verbose: Print analysis details
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augmentations = augmentations
        self.cache_images = cache_images
        self.verbose = verbose
        
        # Setup paths
        self.images_dir = self.root_dir / 'images' / split
        self.labels_dir = self.root_dir / 'labels' / split
        
        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise ValueError(f"Dataset directories not found in {root_dir}")
        
        # STEP 1: Analyze dataset for classes
        if self.verbose:
            print(f"🚀 Initializing YOLO-One Dataset for {split} split")
            print("=" * 60)
        
        self.analyzer = YoloDatasetAnalyzer(root_dir)
        self.analysis_results = self.analyzer.analyze_dataset(verbose=verbose)
        
        if not self.analysis_results['all_classes']:
            raise ValueError("No valid classes found in dataset!")
        
        # STEP 2: Determine target class
        self.target_class = self._determine_target_class(
            target_class, auto_select_class, verbose
        )
        
        if self.target_class is None:
            raise ValueError("No target class selected for training!")
        
        # STEP 3: Load dataset with class filter
        self.image_paths = self._load_image_paths()
        self.label_paths = self._get_label_paths()
        
        # STEP 4: Filter dataset for target class
        self._filter_dataset_for_target_class()
        
        # Image cache
        self.image_cache = {} if cache_images else None
        
        # Final statistics
        self.stats = self._compute_filtered_stats()
        
        if self.verbose:
            print(f"\n✅ Dataset ready for YOLO-One training!")
            print(f"📊 Final dataset: {len(self.valid_indices)} images with class {self.target_class}")
            print(f"📈 Total annotations: {self.stats['target_class_annotations']}")
    
    def _determine_target_class(
        self, 
        target_class: Optional[int], 
        auto_select_class: bool,
        verbose: bool
    ) -> Optional[int]:
        """Determine which class to use for training"""
        
        available_classes = self.analysis_results['all_classes']
        class_counts = self.analysis_results['class_counts']
        
        # Print analysis summary
        if verbose:
            self.analyzer.print_analysis_summary()
        
        # Case 1: Single class dataset
        if len(available_classes) == 1:
            selected_class = available_classes[0]
            if verbose:
                print(f"\n🎯 Single class dataset detected: class {selected_class}")
                print("✅ Perfect for YOLO-One training!")
            return selected_class
        
        # Case 2: Specific class requested
        if target_class is not None:
            if target_class in available_classes:
                if verbose:
                    count = class_counts[target_class]
                    total = self.analysis_results['total_annotations']
                    percentage = (count / total) * 100
                    print(f"\n🎯 Using specified target class: {target_class}")
                    print(f"📊 {count:,} annotations ({percentage:.1f}% of dataset)")
                return target_class
            else:
                raise ValueError(f"Specified target_class {target_class} not found in dataset. Available: {available_classes}")
        
        # Case 3: Multiple classes - need selection
        if auto_select_class:
            # Auto-select most common class
            most_common_class = class_counts.most_common(1)[0][0]
            if verbose:
                count = class_counts[most_common_class]
                total = self.analysis_results['total_annotations']
                percentage = (count / total) * 100
                print(f"\n🤖 Auto-selected most common class: {most_common_class}")
                print(f"📊 {count:,} annotations ({percentage:.1f}% of dataset)")
                print("💡 Tip: Specify target_class parameter to choose different class")
            return most_common_class
        
        # Case 4: Interactive selection
        return self._interactive_class_selection(verbose)
    
    def _interactive_class_selection(self, verbose: bool) -> Optional[int]:
        """Interactive class selection"""
        
        classes = self.analysis_results['all_classes']
        class_counts = self.analysis_results['class_counts']
        
        print(f"\n🎯 MULTIPLE CLASSES DETECTED - SELECTION REQUIRED")
        print("=" * 50)
        print("Available classes:")
        
        for i, class_id in enumerate(classes):
            count = class_counts[class_id]
            percentage = (count / self.analysis_results['total_annotations']) * 100
            print(f"  {i+1}. Class {class_id}: {count:,} annotations ({percentage:.1f}%)")
        
        print(f"\n💡 YOLO-One is optimized for single-class detection")
        print(f"   Choose the class you want to detect:")
        
        while True:
            try:
                choice = input(f"\nSelect class (1-{len(classes)}) or enter class ID directly: ").strip()
                
                if choice.isdigit():
                    choice_int = int(choice)
                    
                    # Check if it's a valid index
                    if 1 <= choice_int <= len(classes):
                        selected_class = classes[choice_int - 1]
                        break
                    
                    # Check if it's a valid class ID
                    elif choice_int in classes:
                        selected_class = choice_int
                        break
                    
                    else:
                        print(f"❌ Invalid choice. Use 1-{len(classes)} or valid class ID")
                        continue
                
                else:
                    print("❌ Please enter a number")
                    continue
                    
            except (ValueError, KeyboardInterrupt):
                print("\n❌ Selection cancelled")
                return None
        
        if verbose:
            count = class_counts[selected_class]
            percentage = (count / self.analysis_results['total_annotations']) * 100
            print(f"\n✅ Selected class {selected_class}")
            print(f"📊 {count:,} annotations ({percentage:.1f}% of dataset)")
        
        return selected_class
    
    def _load_image_paths(self) -> List[Path]:
        """Load all image paths"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(self.images_dir.glob(ext))
            image_paths.extend(self.images_dir.glob(ext.upper()))
        
        return sorted(image_paths)
    
    def _get_label_paths(self) -> List[Path]:
        """Get corresponding label paths"""
        label_paths = []
        
        for img_path in self.image_paths:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            label_paths.append(label_path)
        
        return label_paths
    
    def _filter_dataset_for_target_class(self):
        """Filter dataset to only include images with target class"""
        
        if self.verbose:
            print(f"\n🔍 Filtering dataset for class {self.target_class}...")
        
        self.valid_indices = []
        
        for idx, label_path in enumerate(self.label_paths):
            if not label_path.exists():
                continue
            
            # Check if this file contains target class
            has_target_class = False
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            if class_id == self.target_class:
                                has_target_class = True
                                break
                        except ValueError:
                            continue
                
                if has_target_class:
                    self.valid_indices.append(idx)
                    
            except Exception:
                continue
        
        if self.verbose:
            print(f"✅ Found {len(self.valid_indices)} images containing class {self.target_class}")
    
    def _compute_filtered_stats(self) -> Dict:
        """Compute statistics for filtered dataset"""
        
        total_annotations = 0
        
        for idx in self.valid_indices:
            label_path = self.label_paths[idx]
            
            if not label_path.exists():
                continue
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) == 5:
                    try:
                        class_id = int(parts[0])
                        if class_id == self.target_class:
                            total_annotations += 1
                    except ValueError:
                        continue
        
        return {
            'filtered_images': len(self.valid_indices),
            'target_class_annotations': total_annotations,
            'target_class': self.target_class,
            'avg_annotations_per_image': total_annotations / max(1, len(self.valid_indices))
        }
    
    def _load_image(self, img_path: Path) -> Optional[np.ndarray]:
        """Load image from path or cache"""
        
        if self.image_cache is not None and str(img_path) in self.image_cache:
            return self.image_cache[str(img_path)]
        
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.image_cache is not None:
                self.image_cache[str(img_path)] = image
            
            return image
            
        except Exception:
            return None
    
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return empty sample for corrupted data"""
        
        return {
            'image': torch.zeros(3, *self.img_size),
            'targets': torch.zeros(0, 6),  # Empty targets
            'image_path': '',
            'original_size': (0, 0)
        }
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item"""
        
        actual_idx = self.valid_indices[idx]
        img_path = self.image_paths[actual_idx]
        label_path = self.label_paths[actual_idx]
        
        # Load image
        image = self._load_image(img_path)
        if image is None:
            return self._get_empty_sample()
        
        original_h, original_w = image.shape[:2]
        
        # Load annotations (only target class)
        annotations = YoloFormatParser.parse_yolo_label(
            str(label_path), original_w, original_h
        )
        
        # Filter for target class only
        annotations = [ann for ann in annotations if ann['class_id'] == self.target_class]
        
        # Prepare for augmentation
        bboxes = []
        class_labels = []
        
        for ann in annotations:
            bboxes.append(ann['bbox'])
            class_labels.append(0)  # Always 0 for single-class
        
        # Apply augmentations
        if self.augmentations and bboxes:
            try:
                augmented = self.augmentations(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                image = augmented['image']
                bboxes = augmented['bboxes']
                class_labels = augmented['class_labels']
                
            except Exception:
                # Fallback to original if augmentation fails
                pass
        
        # Resize image
        if not hasattr(image, 'shape') or len(image.shape) != 3:
            # Image is tensor from augmentation
            pass
        else:
            # Resize image
            image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Prepare targets tensor
        if bboxes:
            targets = []
            for i, bbox in enumerate(bboxes):
                # Convert to relative coordinates for model
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2 / self.img_size[1]
                y_center = (y1 + y2) / 2 / self.img_size[0]
                width = (x2 - x1) / self.img_size[1]
                height = (y2 - y1) / self.img_size[0]
                
                # Format: [batch_idx, class, x_center, y_center, width, height]
                targets.append([0, 0, x_center, y_center, width, height])
            
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = torch.zeros(0, 6)
        
        return {
            'image': image,
            'targets': targets,
            'image_path': str(img_path),
            'original_size': (original_h, original_w)
        }

# Convenience function for easy dataset creation
def create_yolo_one_dataset(
    root_dir: str,
    split: str = 'train',
    img_size: Tuple[int, int] = (640, 640),
    target_class: Optional[int] = None,
    auto_select_class: bool = True,
    batch_size: int = 16,
    num_workers: int = 4,
    augmentations: Optional[A.Compose] = None
) -> Tuple[YoloOneDataset, DataLoader]:
    """
    Create YOLO-One dataset and dataloader with automatic analysis
    
    Args:
        root_dir: Dataset root directory
        split: Dataset split to load
        img_size: Target image size
        target_class: Specific class to use (None for auto-detection)
        auto_select_class: Auto-select most common class if multiple found
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        augmentations: Augmentation pipeline
        
    Returns:
        Tuple of (dataset, dataloader)
    """
    
    # Create dataset with integrated analysis
    dataset = YoloOneDataset(
        root_dir=root_dir,
        split=split,
        img_size=img_size,
        target_class=target_class,
        auto_select_class=auto_select_class,
        augmentations=augmentations,
        verbose=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return dataset, dataloader

def collate_fn(batch):
    """Custom collate function for YOLO-One batches"""
    
    images = []
    targets = []
    image_paths = []
    original_sizes = []
    
    for i, sample in enumerate(batch):
        images.append(sample['image'])
        
        # Add batch index to targets
        if len(sample['targets']) > 0:
            batch_targets = sample['targets'].clone()
            batch_targets[:, 0] = i  # Set batch index
            targets.append(batch_targets)
        
        image_paths.append(sample['image_path'])
        original_sizes.append(sample['original_size'])
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Concatenate targets
    if targets:
        targets = torch.cat(targets, 0)
    else:
        targets = torch.zeros(0, 6)
    
    return {
        'images': images,
        'targets': targets,
        'image_paths': image_paths,
        'original_sizes': original_sizes
    }
"""
YOLO-One Dataset Preprocessing - Single-Class Selection
Iatrax Team - 2025 - https://iatrax.com

"""

import cv2
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Tuple, Dict, Optional
import albumentations as A
from pathlib import Path
from collections import Counter
from tqdm import tqdm
class YoloOneDatasetAnalyzer:
    """
    Dataset analyzer for class detection and selection
    Core logic: 1 class = auto-continue, multiple classes = user selection
    """
    def __init__(self, dataset_root: str, split: str = 'train'):
        self.dataset_root = Path(dataset_root)
        self.analysis_results = {}
        self.split = split

    def analyze_and_select_class(self, target_class: Optional[int] = None) -> int:
        """
        Main function: Analyze dataset and return selected class
        """
        print("↝ Analyzing dataset for YOLO-One...")
        print("=" * 50)

        # Step 1: Scan all splits for classes
        all_classes = self._scan_dataset_classes()

        if not all_classes:
            raise ValueError("❌ No valid classes found in dataset!")

        # Step 2: Apply selection logic
        return self._apply_selection_logic(all_classes, target_class)

    def _scan_dataset_classes(self) -> Dict[int, int]:
        """Scan dataset and count classes across all splits"""

        assert self.split in ['train', 'val', 'test'], "Invalid split"
        class_counts = Counter()
        total_files = 0
        split_path = self.dataset_root / 'labels' / self.split
        if not split_path.exists():
            raise FileNotFoundError(f"Labels directory not found: {split_path}")     
        print(f"Scanning {self.split} split...")
        label_files = list(split_path.glob('*.txt'))
        total_files += len(label_files)

        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                        except ValueError:
                            continue            
            except Exception:
                continue
        # Print scan results
        print(f"✅ Scanned {total_files} files")
        print(f"🏆 Found {len(class_counts)} unique classes: {sorted(class_counts.keys())}")

        # Store results
        self.analysis_results = {
            'class_counts': class_counts,
            'total_annotations': sum(class_counts.values()),
            'total_files': total_files
        }

        return dict(class_counts)

    def _apply_selection_logic(self, classes: Dict[int, int], target_class: Optional[int]) -> int:
        """
        Apply selection logic based on number of classes found
        """
        class_list = sorted(classes.keys())
        class_counts = self.analysis_results['class_counts']

        print(f"\n📊 Class distribution:")
        for class_id in class_list:
            count = classes[class_id]
            percentage = (count / self.analysis_results['total_annotations']) * 100
            print(f"   Class {class_id}: {count:,} annotations ({percentage:.1f}%)")

        # CASE 1: Single class detected → Perfect for YOLO-One!
        if len(class_list) == 1:
            selected_class = class_list[0]
            print(f"🎯 SINGLE CLASS DETECTED: {selected_class}")
            print("✅ Perfect for YOLO-One! Continuing automatically...")
            return selected_class

        # CASE 2: Multiple classes + target specified → Validate
        if target_class is not None:
            if target_class in class_list:
                count = classes[target_class]
                percentage = (count / self.analysis_results['total_annotations']) * 100
                print(f"\n🏆 USING SPECIFIED CLASS: {target_class}")
                print(f"📊 {count:,} annotations ({percentage:.1f}% of dataset)")
                print("✅ Valid class selected!")
                return target_class
            else:
                raise ValueError(f"❌ Specified class {target_class} not found! Available: {class_list}")

        # CASE 3: Multiple classes → User selection required
        return self._interactive_class_selection(class_list, classes)

    def _interactive_class_selection(self, class_list: List[int], classes: Dict[int, int]) -> int:
        """
        Interactive class selection for multiple classes
        """
        print("\nMULTIPLE CLASSES DETECTED - SELECTION REQUIRED")
        print("=" * 55)
        print("💡 YOLO-One requires one class per training set for object detection.")
        print("    Please select a class :")

        sorted_class_list = sorted(class_list)
        # show class list
        for i, class_id in enumerate(sorted_class_list, 1):
            count = classes[class_id]
            print(f"   {i}. Class {class_id} ({count} annotations)")

        print("\n📝 Enter your choice :")
        print(f"   • Menu option (1-{len(sorted_class_list)})")

        # Get user input
        while True:
            try:
                user_input = self._get_user_input()

                if not user_input:
                    print("Please enter a valid option.")
                    continue

                try:
                    choice = int(user_input)
                    if 1 <= choice <= len(sorted_class_list):
                        selected_class = sorted_class_list[choice - 1]
                        print(f"🏆 Selected Option {choice} → Class {selected_class}")
                        return selected_class
                    else:
                        print(f"❌ Option {choice} is not valid. Please enter a number between 1 and {len(sorted_class_list)}.")
                except ValueError:
                    print("Please enter a valid option.")
                    continue

            except Exception as e:
                print(f"❌ Error : {e}")
                continue

    def _get_user_input(self):
        # Get user input
        return input(">>> Enter your choice : ")

class YoloOneDataset(Dataset):
    """
    YOLO-One Dataset with caching for faster initialization
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (640, 640),
        target_class: Optional[int] = None,
        augmentations: Optional[A.Compose] = None,
        cache_images: bool = False
    ):
        """
        Initialize YOLO-One dataset with automatic class selection
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augmentations = augmentations
        self.cache_images = cache_images

        # Validate directories
        self.images_dir = self.root_dir / 'images' / split
        self.labels_dir = self.root_dir / 'labels' / split

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        print(f"🚀 Initializing YOLO-One Dataset ({split} split)")
        print(f"📁 Dataset: {root_dir}")

        # CORE LOGIC: Analyze and select class
        analyzer = YoloOneDatasetAnalyzer(root_dir, split=split)
        self.target_class = analyzer.analyze_and_select_class(target_class)

        # Bug fix: use caching for dataset building
        self.valid_samples = self._build_dataset_with_caching()

        # Initialize cache
        self.image_cache = {} if cache_images else None

        print(f"Dataset ready: {len(self.valid_samples)} samples with class {self.target_class}")

    def _build_dataset_with_caching(self):
        """
        Build dataset filtering for target class only, using a cache file.
        Sequential version optimized for Colab.
        """
        print(f"\nBuilding dataset for class {self.target_class}...")
        
        # Define cache file path
        cache_file = self.root_dir / f'{self.split}_class_{self.target_class}.cache'

        # Check if cache file exists
        if cache_file.exists():
            print(f"Loading dataset from cache: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    valid_samples = json.load(f)
                
                # Convert string paths back to Path objects
                for sample in valid_samples:
                    sample['image_path'] = Path(sample['image_path'])
                    sample['label_path'] = Path(sample['label_path'])
                    
                print(f"Cache loaded successfully! Found {len(valid_samples)} samples")
                return valid_samples
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading cache file: {e}. Rebuilding dataset...")

        print("No valid cache found. Building dataset from scratch...")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f'*{ext}'))
            image_files.extend(self.images_dir.glob(f'*{ext.upper()}'))
        image_files = sorted(image_files)
        print(f"Found {len(image_files)} total images")

        # Process files sequentially
        valid_samples = []
        target_class_annotations = 0
        processed_count = 0
        iterator = tqdm(image_files, desc=f"Filtering class {self.target_class}")
        
        for img_path in iterator:
            processed_count += 1
            
            # Show progress every 100 files if no tqdm
            if processed_count % 100 == 0 and 'tqdm' not in locals():
                print(f"Processed {processed_count}/{len(image_files)} files...")
            
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
                
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                class_annotations = 0
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) >= 5:  # YOLO format: class_id x y w h
                        try:
                            class_id = int(parts[0])
                            if class_id == self.target_class:
                                class_annotations += 1
                        except (ValueError, IndexError):
                            continue
                
                if class_annotations > 0:
                    valid_samples.append({
                        'image_path': img_path,
                        'label_path': label_path,
                        'annotations': class_annotations
                    })
                    target_class_annotations += class_annotations
                    
            except Exception as e:
                # Skip files that cannot be read
                continue
        
        # Save to cache if we found samples
        if valid_samples:
            print(f"Writing dataset to cache file: {cache_file}")
            try:
                # Ensure cache directory exists
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(cache_file, 'w') as f:
                    # Convert Path objects to strings for JSON serialization
                    serializable_samples = []
                    for sample in valid_samples:
                        serializable_samples.append({
                            'image_path': str(sample['image_path']),
                            'label_path': str(sample['label_path']),
                            'annotations': sample['annotations']
                        })
                    json.dump(serializable_samples, f, indent=2)
                print("Cache file saved successfully!")
            except Exception as e:
                print(f"Warning: Could not save cache file: {e}")
        else:
            print(f"Warning: No samples found for class {self.target_class}")

        # Print summary statistics
        print(f"\n--- Dataset Summary ---")
        print(f"Filtered dataset: {len(valid_samples)} images with class {self.target_class}")
        print(f"Total annotations: {target_class_annotations}")
        if valid_samples:
            print(f"Average annotations per image: {target_class_annotations/len(valid_samples):.1f}")
        
        return valid_samples

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item with target class only"""
        sample = self.valid_samples[idx]
        img_path = sample['image_path']
        label_path = sample['label_path']

        # Load image
        image = self._load_image(img_path)
        if image is None:
            return self._get_empty_sample()

        original_h, original_w = image.shape[:2]

        # Load and filter annotations for target class
        annotations = self._load_filtered_annotations(label_path, original_w, original_h)

        # Apply augmentations if enabled (using absolute coordinates)
        augmented_image = image.copy()
        augmented_annotations = [ann.copy() for ann in annotations] if annotations else []
        
        if self.augmentations and augmented_annotations:
            try:
                # Use absolute coordinates for augmentation
                bboxes = [ann['bbox'] for ann in augmented_annotations]
                class_labels = [ann['class_id'] for ann in augmented_annotations]

                augmented = self.augmentations(
                    image=augmented_image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']
                
                # Update annotations with augmented boxes
                for i, bbox in enumerate(augmented_bboxes):
                    augmented_annotations[i]['bbox'] = bbox
                    # Recalculate normalized coordinates after augmentation
                    h, w = augmented_image.shape[:2]
                    augmented_annotations[i]['bbox_norm'] = [
                        bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h
                    ]

            except Exception as e:
                # Keep original if augmentation fails
                pass

        # Use augmented data for training
        processed_image = augmented_image
        processed_annotations = augmented_annotations

        # Letterbox preprocessing
        pre_proc_h, pre_proc_w = processed_image.shape[:2]
        scale_factor = min(self.img_size[0] / pre_proc_h, self.img_size[1] / pre_proc_w)
        new_h, new_w = int(pre_proc_h * scale_factor), int(pre_proc_w * scale_factor)

        # Create padded image
        padded_image = np.full((self.img_size[0], self.img_size[1], 3), 114, dtype=np.uint8)
        resized_img = cv2.resize(processed_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate padding
        pad_top = (self.img_size[0] - new_h) // 2
        pad_left = (self.img_size[1] - new_w) // 2
        padded_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_img

        # Adjust bounding boxes for letterboxing
        final_annotations = []
        if processed_annotations:
            for ann in processed_annotations:
                # Use absolute coordinates for letterbox transformation
                x1, y1, x2, y2 = ann['bbox']
                
                # Apply letterbox transformation
                new_x1 = x1 * scale_factor + pad_left
                new_y1 = y1 * scale_factor + pad_top
                new_x2 = x2 * scale_factor + pad_left
                new_y2 = y2 * scale_factor + pad_top
                
                # Convert back to normalized coordinates for the padded image
                new_x1_norm = new_x1 / self.img_size[1]
                new_y1_norm = new_y1 / self.img_size[0]
                new_x2_norm = new_x2 / self.img_size[1]
                new_y2_norm = new_y2 / self.img_size[0]
                
                # Clip to [0, 1]
                new_x1_norm = max(0, min(1, new_x1_norm))
                new_y1_norm = max(0, min(1, new_y1_norm))
                new_x2_norm = max(0, min(1, new_x2_norm))
                new_y2_norm = max(0, min(1, new_y2_norm))
                
                # Only keep valid boxes
                if new_x2_norm > new_x1_norm and new_y2_norm > new_y1_norm:
                    final_annotations.append({
                        'bbox_norm': [new_x1_norm, new_y1_norm, new_x2_norm, new_y2_norm],
                        'batch_index': ann['batch_index']
                    })

        # Convert image to tensor
        image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0

        # Convert annotations to tensor (using normalized coordinates)
        targets = self._annotations_to_tensor_normalized(final_annotations)

        return {
            'image': image_tensor,
            'targets': targets,
            'image_path': str(img_path),
            'original_size': (original_h, original_w),
            'scale_factor': scale_factor,
            'padding': (pad_top, pad_left)
        }

    def _annotations_to_tensor_normalized(self, annotations: List[Dict]) -> torch.Tensor:
        """
        Convert annotations to tensor format with normalized coordinates.
        Format: [batch_idex, x1_norm, y1_norm, x2_norm, y2_norm]
        """
        if not annotations:
            return torch.zeros((0, 5), dtype=torch.float32)
        
        targets = []
        for ann in annotations:
            x1, y1, x2, y2 = ann['bbox_norm']
            batch_idex = ann['batch_index']
            targets.append([batch_idex, x1, y1, x2, y2])
        
        return torch.tensor(targets, dtype=torch.float32)
    
    def _load_image(self, img_path: Path) -> Optional[np.ndarray]:
            """Load image with optional caching"""
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

    def _load_filtered_annotations(self, label_path: Path, img_w: int, img_h: int) -> List[Dict]:
        """
        Load annotations filtered for target class only.
        Returns both normalized (YOLO format) and absolute pixel coordinates.
        """
        annotations = []

        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:  # YOLO format requires at least 5 values
                    continue

                try:
                    class_id = int(parts[0])

                    # # Only process target class
                    # if class_id != self.target_class:
                    #     continue

                    # Read normalized coordinates (0-1 range)
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])

                    # Validate normalized coordinates
                    if not (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1 and 
                            0 <= width_norm <= 1 and 0 <= height_norm <= 1):
                        continue

                    # Convert normalized center coordinates to normalized corner coordinates
                    x1_norm = x_center_norm - width_norm / 2
                    y1_norm = y_center_norm - height_norm / 2
                    x2_norm = x_center_norm + width_norm / 2
                    y2_norm = y_center_norm + height_norm / 2

                    # Clip to [0, 1] range
                    x1_norm = max(0, min(1, x1_norm))
                    y1_norm = max(0, min(1, y1_norm))
                    x2_norm = max(0, min(1, x2_norm))
                    y2_norm = max(0, min(1, y2_norm))

                    # Convert to absolute pixel coordinates for augmentation libraries
                    x1_abs = x1_norm * img_w
                    y1_abs = y1_norm * img_h
                    x2_abs = x2_norm * img_w
                    y2_abs = y2_norm * img_h

                    # Store both formats
                    annotations.append({
                        # Absolute coordinates for augmentation (albumentation expects pixels)
                        'bbox': [x1_abs, y1_abs, x2_abs, y2_abs],
                        # Normalized coordinates for model training
                        'bbox_norm': [x1_norm, y1_norm, x2_norm, y2_norm],
                        # Original YOLO format (center, width, height) normalized
                        'yolo_format': [x_center_norm, y_center_norm, width_norm, height_norm],
                        'batch_index': 0,
                    })

                except (ValueError, IndexError):
                    continue

        except Exception as e:
            # Log error if needed for debugging
            # print(f"Error reading label file {label_path}: {e}")
            pass

        return annotations
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return empty sample for corrupted data"""
        return {
            'image': torch.zeros(3, *self.img_size),
            'targets': torch.zeros(0, 5),
            'image_path': '',
            'original_size': (0, 0)
        }

def get_augmentations(aug_config: Dict[str, Any]) -> A.Compose:
   

    """
    Construct an albumentations.Compose object from a dictionary of augmentation parameters.

    The dictionary should contain the following keys:
        - fliplr: probability of horizontal flip
        - flipud: probability of vertical flip
        - translate: range of translation
        - scale: range of scale
        - degrees: range of rotation
        - shear: range of shear
        - perspective: range of perspective distortion
        - hsv_h, hsv_s, hsv_v: range of hue, saturation, value shift

    The function returns a Compose object which can be used to augment images.

    Args:
        aug_config (Dict[str, Any]): dictionary of augmentation parameters

    Returns:
        A.Compose: an albumentations compose object
    """
    return A.Compose([
        A.HorizontalFlip(p=aug_config.get('fliplr', 0.0)),
        A.VerticalFlip(p=aug_config.get('flipud', 0.0)),
        A.ShiftScaleRotate(
            shift_limit=aug_config.get('translate', 0.0),
            scale_limit=aug_config.get('scale', 0.0),
            rotate_limit=aug_config.get('degrees', 0.0),
            shear_limit=aug_config.get('shear', 0.0),
            perspective_limit=aug_config.get('perspective', 0.0),
            p=0.7, # probility for this block
            border_mode=cv2.BORDER_CONSTANT,
            value=114
        ),

        A.HueSaturationValue(
            hue_shift_limit=aug_config.get('hsv_h', 0.0) * 100, 
            sat_shift_limit=aug_config.get('hsv_s', 0.0) * 100,
            val_shift_limit=aug_config.get('hsv_v', 0.0) * 100,
            p=0.7 
        ),
        A.RandomBrightnessContrast(p=0.5),

    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.1
    ))
def create_yolo_one_dataset(
    root_dir: str,
    split: str = 'train',
    img_size: Tuple[int, int] = (640, 640),
    target_class: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    augmentations: Optional[A.Compose] = None,
    use_augmentation: bool = False
) -> Tuple[YoloOneDataset, DataLoader]:
    """
    Create YOLO-One dataset and dataloader with intelligent class selection
    """
    augmentations_pipeline = None
    if split == 'train' and use_augmentation:
        print("✅ Augmentations enabled for training...")
        augmentations_pipeline = get_augmentations(augmentations)
    dataset = YoloOneDataset(
        root_dir=root_dir,
        split=split,
        img_size=img_size,
        target_class=target_class,
        augmentations=augmentations_pipeline
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=yolo_one_collate_fn
    )

    return dataset, dataloader

def yolo_one_collate_fn(batch):
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
        targets = torch.zeros(0, 5, device=images.device)

    return {
        'images': images,
        'targets': targets,
        'image_paths': image_paths,
        'original_sizes': original_sizes
    }

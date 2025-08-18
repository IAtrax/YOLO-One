"""
YOLO-One Dataset Preprocessing - Single-Class Selection
Iatrax Team - 2025 - https://iatrax.com
Bug fix: Added caching for dataset initialization to drastically reduce startup time.
"""

import cv2
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import albumentations as A
from pathlib import Path
from collections import Counter
import concurrent.futures

class YoloOneDatasetAnalyzer:
    """
    Dataset analyzer for class detection and selection
    Core logic: 1 class = auto-continue, multiple classes = user selection
    """
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.analysis_results = {}

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
        class_counts = Counter()
        total_files = 0

        # Scan train, val, test splits
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_root / 'labels' / split

            if not split_path.exists():
                continue

            print(f"Scanning {split} split...")

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
        analyzer = YoloOneDatasetAnalyzer(root_dir)
        self.target_class = analyzer.analyze_and_select_class(target_class)

        # Bug fix: use caching for dataset building
        self.valid_samples = self._build_dataset_with_caching()

        # Initialize cache
        self.image_cache = {} if cache_images else None

        print(f"Dataset ready: {len(self.valid_samples)} samples with class {self.target_class}")

    def _build_dataset_with_caching(self):
        """
        Build dataset filtering for target class only, using a cache file.
        This is the main performance fix.
        """
        print(f"\nBuilding dataset for class {self.target_class}...")
        
        # Define cache file path
        cache_file = self.root_dir / f'{self.split}_class_{self.target_class}.cache'

        # Check if cache file exists
        if cache_file.exists():
            print(f"⏳ Loading dataset from cache: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    valid_samples = json.load(f)
                
                # Convert string paths back to Path objects
                for sample in valid_samples:
                    sample['image_path'] = Path(sample['image_path'])
                    sample['label_path'] = Path(sample['label_path'])
                    
                print("✅ Cache loaded successfully!")
                return valid_samples
            except (IOError, json.JSONDecodeError) as e:
                print(f"❌ Error loading cache file: {e}. Rebuilding dataset...")

        print("👷 No valid cache found. Rebuilding dataset from scratch...")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f'*{ext}'))
            image_files.extend(self.images_dir.glob(f'*{ext.upper()}'))
        image_files = sorted(image_files)
        print(f"📂 Found {len(image_files)} total images")

        # Function to process a single label file
        def process_label_file(label_path, image_path):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                class_annotations = 0
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            if class_id == self.target_class:
                                class_annotations += 1
                        except ValueError:
                            continue
                
                if class_annotations > 0:
                    return {
                        'image_path': str(image_path),  # Store as string for JSON
                        'label_path': str(label_path),  # Store as string for JSON
                        'annotations': class_annotations
                    }
            except Exception:
                pass
            return None

        # Filter for target class using multithreading for speed
        valid_samples = []
        target_class_annotations = 0
        
        # Create a list of (label_path, image_path) tuples
        tasks = []
        for img_path in image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                tasks.append((label_path, img_path))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_label_file, label_path, img_path): (label_path, img_path) for label_path, img_path in tasks}
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                if result:
                    valid_samples.append(result)
                    target_class_annotations += result['annotations']
        
        # Convert string paths back to Path objects
        for sample in valid_samples:
            sample['image_path'] = Path(sample['image_path'])
            sample['label_path'] = Path(sample['label_path'])

        # Save to cache
        print(f"Writing dataset list to cache file: {cache_file}")
        with open(cache_file, 'w') as f:
            # Revert to string for serialization
            serializable_samples = [{k: str(v) if isinstance(v, Path) else v for k, v in s.items()} for s in valid_samples]
            json.dump(serializable_samples, f)

        print(f"✅ Filtered dataset: {len(valid_samples)} images")
        print(f"📊 Total annotations: {target_class_annotations}")
        print(f"Avg annotations/image: {target_class_annotations/max(1, len(valid_samples)):.1f}")
        
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

        # Apply augmentations
        if self.augmentations and annotations:
            try:
                bboxes = [ann['bbox'] for ann in annotations]
                class_labels = [0] * len(annotations)  # Always 0 for single-class

                augmented = self.augmentations(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                image = augmented['image']
                bboxes = augmented['bboxes']
                annotations = [{'bbox': bbox} for bbox in bboxes]

            except Exception:
                # Keep original if augmentation fails
                pass

        # --- Letterbox Preprocessing (to match inference pipeline) ---
        # This ensures the model trains on data with the same aspect ratio preservation as in production.
        pre_proc_h, pre_proc_w = image.shape[:2]
        scale_factor = min(self.img_size[0] / pre_proc_h, self.img_size[1] / pre_proc_w)
        new_h, new_w = int(pre_proc_h * scale_factor), int(pre_proc_w * scale_factor)

        # Create a padded image with a constant color (114)
        padded_image = np.full((self.img_size[0], self.img_size[1], 3), 114, dtype=np.uint8)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate padding and place the resized image
        pad_top = (self.img_size[0] - new_h) // 2
        pad_left = (self.img_size[1] - new_w) // 2
        padded_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_img

        # Adjust bounding boxes for letterboxing
        if annotations:
            new_annotations = []
            for ann in annotations:
                x1, y1, x2, y2 = ann['bbox']
                # Scale and pad the box coordinates to match the padded image
                new_bbox = [
                    x1 * scale_factor + pad_left, y1 * scale_factor + pad_top,
                    x2 * scale_factor + pad_left, y2 * scale_factor + pad_top
                ]
                new_annotations.append({'bbox': new_bbox})
            annotations = new_annotations

        # Convert image to tensor
        image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0

        # Convert annotations to tensor
        targets = self._annotations_to_tensor(annotations)

        return {
            'image': image_tensor,
            'targets': targets,
            'image_path': str(img_path),
            'original_size': (original_h, original_w)
        }

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
        """Load annotations filtered for target class only"""
        annotations = []

        try:
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

                    # Only process target class
                    if class_id != self.target_class:
                        continue

                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert to absolute coordinates
                    abs_x_center = x_center * img_w
                    abs_y_center = y_center * img_h
                    abs_width = width * img_w
                    abs_height = height * img_h

                    # Convert to corner format
                    x1 = abs_x_center - abs_width / 2
                    y1 = abs_y_center - abs_height / 2
                    x2 = abs_x_center + abs_width / 2
                    y2 = abs_y_center + abs_height / 2

                    annotations.append({
                        'bbox': [x1, y1, x2, y2],
                        'bbox_rel': [x_center, y_center, width, height]
                    })

                except ValueError:
                    continue

        except Exception:
            pass

        return annotations

    def _annotations_to_tensor(self, annotations: List[Dict]) -> torch.Tensor:
        """Convert annotations to YOLO-One tensor format"""
        if not annotations:
            return torch.zeros(0, 6)  # Empty tensor

        targets = []

        for ann in annotations:
            bbox = ann['bbox']

            # Convert to relative coordinates
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2 / self.img_size[1]
            y_center = (y1 + y2) / 2 / self.img_size[0]
            width = (x2 - x1) / self.img_size[1]
            height = (y2 - y1) / self.img_size[0]

            # Format: [batch_idx, class, x_center, y_center, width, height]
            # Note: batch_idx will be set in collate_fn, class is always 0
            targets.append([0, 0, x_center, y_center, width, height])

        return torch.tensor(targets, dtype=torch.float32)

    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return empty sample for corrupted data"""
        return {
            'image': torch.zeros(3, *self.img_size),
            'targets': torch.zeros(0, 6),
            'image_path': '',
            'original_size': (0, 0)
        }

def create_yolo_one_dataset(
    root_dir: str,
    split: str = 'train',
    img_size: Tuple[int, int] = (640, 640),
    target_class: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    augmentations: Optional[A.Compose] = None
) -> Tuple[YoloOneDataset, DataLoader]:
    """
    Create YOLO-One dataset and dataloader with intelligent class selection
    """
    dataset = YoloOneDataset(
        root_dir=root_dir,
        split=split,
        img_size=img_size,
        target_class=target_class,
        augmentations=augmentations
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
        targets = torch.zeros(0, 6)

    return {
        'images': images,
        'targets': targets,
        'image_paths': image_paths,
        'original_sizes': original_sizes
    }

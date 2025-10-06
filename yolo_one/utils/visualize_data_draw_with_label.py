"""
YOLO Annotations Visualizer - Iatrax 2025
Draw bounding boxes on images and save to runs/visualizations/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys


class YOLOVisualizer:
    """Visualizer for YOLO annotations with bounding boxes"""
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        output_dir: str = "runs/visualizations",
        box_color: Tuple[int, int, int] = (0, 255, 0),  # Green BGR
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2,
    ):
        """
        Initialize the visualizer
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO labels (.txt files)
            output_dir: Output directory (auto-created if not exists)
            box_color: Box color in BGR format (default: green)
            box_thickness: Box border thickness
            font_scale: Font scale for text
            font_thickness: Text thickness
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        
        self.box_color = box_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        
        # Validation
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        # Create output dir if not exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory: {self.output_dir.absolute()}")
        
    
    def parse_yolo_label(self, label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """
        Parse YOLO label file
        
        YOLO format (per line):
        class_id x_center y_center width height
        
        All coordinates normalized [0, 1]
        
        Args:
            label_path: Path to .txt file
            
        Returns:
            List of (class_id, x_center, y_center, width, height)
        """
        if not label_path.exists():
            return []
        
        annotations = []
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    print(f"⚠️  Invalid line in {label_path.name}: {line}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate [0, 1] range
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                            0 <= width <= 1 and 0 <= height <= 1):
                        print(f"⚠️  Coords out of range [0,1] in {label_path.name}: {line}")
                        continue
                    
                    annotations.append((class_id, x_center, y_center, width, height))
                    
                except ValueError as e:
                    print(f"⚠️  Parse error in {label_path.name}: {line} - {e}")
                    continue
        
        return annotations
    
    
    def yolo_to_corners(
        self,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        img_width: int,
        img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Convert YOLO normalized coords to corner pixels
        
        Args:
            x_center, y_center, width, height: YOLO coords [0,1]
            img_width, img_height: Image dimensions in pixels
            
        Returns:
            (x1, y1, x2, y2) in pixels
        """
        # Denormalize
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Corners
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        # Clamp to image bounds
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        return x1, y1, x2, y2
    
    
    def draw_boxes(
        self,
        image: np.ndarray,
        annotations: List[Tuple[int, float, float, float, float]],
        class_names: List[str] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: BGR image (numpy array)
            annotations: List of YOLO annotations
            class_names: Class names (optional)
            
        Returns:
            Annotated image (copy)
        """
        img_annotated = image.copy()
        img_height, img_width = image.shape[:2]
        
        for class_id, x_center, y_center, width, height in annotations:
            # Convert to corner pixels
            x1, y1, x2, y2 = self.yolo_to_corners(
                x_center, y_center, width, height,
                img_width, img_height
            )
            
            # Draw rectangle
            cv2.rectangle(
                img_annotated,
                (x1, y1),
                (x2, y2),
                self.box_color,
                self.box_thickness
            )
            
            # Label text
            if class_names and class_id < len(class_names):
                label = class_names[class_id]
            else:
                label = f"Class {class_id}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.font_thickness
            )
            
            # Background rectangle for text
            cv2.rectangle(
                img_annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                self.box_color,
                -1  # Filled
            )
            
            # Text
            cv2.putText(
                img_annotated,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),  # White
                self.font_thickness,
                cv2.LINE_AA
            )
        
        return img_annotated
    
    
    def process_single_image(
        self,
        image_path: Path,
        class_names: List[str] = None,
        save: bool = True
    ) -> np.ndarray:
        """
        Process a single image
        
        Args:
            image_path: Path to image
            class_names: Class names (optional)
            save: Save annotated image
            
        Returns:
            Annotated image
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Error reading image: {image_path}")
            return None
        
        # Find corresponding label
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        
        # Parse annotations
        annotations = self.parse_yolo_label(label_path)
        
        if not annotations:
            print(f"⚠️  No annotations for {image_path.name}")
        
        # Draw boxes
        img_annotated = self.draw_boxes(image, annotations, class_names)
        
        # Save
        if save:
            output_path = self.output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
            cv2.imwrite(str(output_path), img_annotated)
            print(f"✅ Saved: {output_path.name} ({len(annotations)} boxes)")
        
        return img_annotated
    
    
    def process_all(self, class_names: List[str] = None, extensions: List[str] = None):
        """
        Process all images in directory
        
        Args:
            class_names: Class names (optional)
            extensions: Image extensions to process (default: jpg, png, jpeg)
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # List images
        image_files = []
        for ext in extensions:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No images found in {self.images_dir}")
            return
        
        print(f"\n🔍 Found {len(image_files)} images\n")
        
        # Process each image
        total_boxes = 0
        processed = 0
        
        for image_path in sorted(image_files):
            result = self.process_single_image(image_path, class_names, save=True)
            if result is not None:
                processed += 1
                # Count boxes
                label_path = self.labels_dir / f"{image_path.stem}.txt"
                annotations = self.parse_yolo_label(label_path)
                total_boxes += len(annotations)
        
        # Final stats
        print(f"\n{'='*60}")
        print(f"✅ Processing complete!")
        print(f"   Images processed: {processed}/{len(image_files)}")
        print(f"   Total boxes drawn: {total_boxes}")
        print(f"   Output directory: {self.output_dir.absolute()}")
        print(f"{'='*60}\n")


def main():
    """Script entry point"""
    
    parser = argparse.ArgumentParser(
        description="Visualize YOLO annotations by drawing bounding boxes on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python visualize_yolo.py --images data/images --labels data/labels
  
  # Custom output directory
  python visualize_yolo.py --images data/images --labels data/labels --output runs/my_viz
  
  # With class names
  python visualize_yolo.py --images data/images --labels data/labels --classes person car dog
  
  # Custom colors (BGR format)
  python visualize_yolo.py --images data/images --labels data/labels --color 255 0 0  # Red
  
  # Thick boxes
  python visualize_yolo.py --images data/images --labels data/labels --thickness 4
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--images', '-i',
        type=str,
        required=True,
        help='Path to images directory'
    )
    
    parser.add_argument(
        '--labels', '-l',
        type=str,
        required=True,
        help='Path to YOLO labels directory (.txt files)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='runs/visualizations',
        help='Output directory for annotated images (default: runs/visualizations)'
    )
    
    parser.add_argument(
        '--classes', '-c',
        type=str,
        nargs='+',
        default=None,
        help='Class names (e.g., --classes person car dog)'
    )
    
    parser.add_argument(
        '--color',
        type=int,
        nargs=3,
        default=[0, 255, 0],
        metavar=('B', 'G', 'R'),
        help='Box color in BGR format (default: 0 255 0 = green)'
    )
    
    parser.add_argument(
        '--thickness', '-t',
        type=int,
        default=2,
        help='Box line thickness (default: 2)'
    )
    
    parser.add_argument(
        '--font-scale', '-fs',
        type=float,
        default=0.6,
        help='Font scale for labels (default: 0.6)'
    )
    
    parser.add_argument(
        '--extensions', '-e',
        type=str,
        nargs='+',
        default=['.jpg', '.jpeg', '.png'],
        help='Image extensions to process (default: .jpg .jpeg .png)'
    )
    
    args = parser.parse_args()
    
    # Display config
    print("\n" + "="*60)
    print("🎨 YOLO Annotations Visualizer - Iatrax 2025")
    print("="*60)
    print(f"Images directory : {args.images}")
    print(f"Labels directory : {args.labels}")
    print(f"Output directory : {args.output}")
    print(f"Box color (BGR)  : {tuple(args.color)}")
    print(f"Box thickness    : {args.thickness}")
    print(f"Font scale       : {args.font_scale}")
    if args.classes:
        print(f"Class names      : {args.classes}")
    print("="*60 + "\n")
    
    try:
        # Create visualizer
        visualizer = YOLOVisualizer(
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            box_color=tuple(args.color),
            box_thickness=args.thickness,
            font_scale=args.font_scale,
        )
        
        # Process all images
        visualizer.process_all(
            class_names=args.classes,
            extensions=args.extensions
        )
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
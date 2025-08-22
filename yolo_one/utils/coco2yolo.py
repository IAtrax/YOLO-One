"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

COCO to YOLO Conversion Script
Converts COCO JSON annotations to YOLO .txt format, specifically for single-class detection.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def convert_coco_to_yolo(json_path: Path, output_dir: Path, target_coco_id: int):
    """
    Converts COCO bounding box annotations to YOLO .txt format for a single target class.

    Args:
        json_path (Path): Path to the input COCO JSON file.
        output_dir (Path): Directory where the YOLO .txt label files will be saved.
        target_coco_id (int): The COCO category ID of the class to extract.
    """
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO annotations from: {json_path}")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping from image_id to image information
    image_info_map = {img['id']: {'width': img['width'], 'height': img['height'], 'file_name': img['file_name']}
                      for img in coco_data['images']}

    # Group annotations by image_id, but only for the target class
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann['category_id'] == target_coco_id:
            annotations_by_image[ann['image_id']].append(ann)

    print(f"Found {len(annotations_by_image)} images with the target class ID {target_coco_id}.")
    print(f"Converting annotations to YOLO format in: {output_dir}")

    # Process each image's annotations
    for image_id, annotations in tqdm(annotations_by_image.items(), desc="Converting images"):
        img_info = image_info_map.get(image_id)
        if not img_info:
            print(f"Warning: Image info not found for image_id {image_id}. Skipping.")
            continue

        img_width = img_info['width']
        img_height = img_info['height']
        base_file_name = Path(img_info['file_name']).stem
        output_txt_path = output_dir / f"{base_file_name}.txt"

        yolo_lines = []
        for ann in annotations:
            # The output class_id is always 0 for YOLO-One
            class_id = 0
            x_min, y_min, bbox_width, bbox_height = ann['bbox']

            # Calculate YOLO format: x_center, y_center, width, height (normalized)
            x_center = (x_min + bbox_width / 2) / img_width
            y_center = (y_min + bbox_height / 2) / img_height
            norm_width = bbox_width / img_width
            norm_height = bbox_height / img_height

            # Format to 6 decimal places for precision
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            yolo_lines.append(yolo_line)

        # Write annotations to the .txt file
        with open(output_txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    print(f"\n✅ Conversion complete. {len(annotations_by_image)} YOLO label files are saved in '{output_dir}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO JSON annotations to YOLO .txt format for a single class.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--json-path', type=Path, required=True, help="Path to the input COCO JSON file (e.g., instances_val2017.json).")
    parser.add_argument('--output-dir', type=Path, default=Path('./labels'), help="Directory to save the output YOLO .txt label files. Defaults to './labels'.")
    parser.add_argument('--target-id', type=int, required=True, help="The COCO category ID of the single class you want to extract (e.g., 1 for 'person' in COCO).")
    args = parser.parse_args()

    convert_coco_to_yolo(args.json_path, args.output_dir, args.target_id)

if __name__ == "__main__":
    main()

 
"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

Dataset Cleaning Utility
Removes image files that do not have a corresponding label file.
"""

import argparse
from pathlib import Path
from typing import Set


def clean_dataset(image_dir: Path, label_dir: Path, delete_files: bool = False):
    """
     Removes image files from a directory that do not have a corresponding label file
    in another directory.

    Args:
       image_dir (Path): Path to the directory containing image files.
        label_dir (Path): Path to the directory containing YOLO .txt label files.
        delete_files (bool): If True, deletes the image files. If False, performs a dry run.
    """
    print(f"🔍 Scanning for unlabeled images in '{image_dir}'...")
    print(f"   Using labels from '{label_dir}'")
    print(f"   Mode: {'DELETE' if delete_files else 'DRY RUN'}")
    print("-" * 50)
     # 1. Get all image paths and their stems (filenames without extension)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    image_paths = {p.stem: p for ext in image_extensions for p in image_dir.glob(f'*{ext.lower()}')}
    image_paths.update({p.stem: p for ext in image_extensions for p in image_dir.glob(f'*{ext.upper()}')})
    # 2. Get all label stems
    label_stems: Set[str] = {p.stem for p in label_dir.glob('*.txt')}
    if not image_paths:
        print("⚠️ No images found in the specified directory. Exiting.")
        return
    # 3. Find the difference
    unlabeled_image_stems = set(image_paths.keys()) - label_stems

    if not unlabeled_image_stems:
        print("✅ No unlabeled images found. Your dataset is clean!")
        return

    print(f"Found {len(unlabeled_image_stems)} unlabeled images:")
    deleted_count = 0
    for stem in sorted(list(unlabeled_image_stems)):
        image_to_remove = image_paths[stem]
        if delete_files:
            try:
                image_to_remove.unlink()
                print(f"  - 🗑️ Deleted: {image_to_remove}")
                deleted_count += 1
            except OSError as e:
                print(f"  - ❌ Error deleting {image_to_remove}: {e}")
        else:
            print(f"  - ❓ Would delete: {image_to_remove}")

    print("-" * 50)
    if delete_files:
        print(f"✅ Deletion complete. Removed {deleted_count} image(s).")
    else:
        print(f"ℹ️ This was a dry run. To delete these files, run again with the --delete flag.")
def main():
    """Main function to parse arguments and run the cleaning script."""
    parser = argparse.ArgumentParser(
        description="Clean a dataset by removing images that lack corresponding label files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_dir', type=Path, help="Path to the directory containing image files.")
    parser.add_argument('label_dir', type=Path, help="Path to the directory containing YOLO .txt label files.")
    parser.add_argument('--delete', action='store_true', help="Actually delete the files. Without this flag, it's a dry run.")
    args = parser.parse_args()     
    if not args.image_dir.is_dir():
        print(f"❌ Error: Image directory not found at '{args.image_dir}'")
        return
    if not args.label_dir.is_dir():
        print(f"❌ Error: Label directory not found at '{args.label_dir}'")
        return
      
    clean_dataset(args.image_dir, args.label_dir, delete_files=args.delete) 


if __name__ == "__main__":

   main()
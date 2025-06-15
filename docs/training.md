# 📊 Dataset Setup for YOLO-One

YOLO-One requires a properly organized dataset in YOLO format. Follow this guide to prepare your data for training.

## 🗂️ Required Directory Structure

your_dataset/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images (optional)
├── labels/
│   ├── train/          # Training labels (.txt files)
│   ├── val/            # Validation labels (.txt files)
│   └── test/           # Test labels (optional)
└── data.yaml           # Dataset configuration

## 📝 Label Format

Each image needs a corresponding `.txt` file with the same name containing bounding box annotations:

```
# Format: class_id x_center y_center width height
# For YOLO-One: class_id is always 0 (single class)
# All coordinates are normalized (0.0 to 1.0)

0 0.5 0.5 0.3 0.4      # Object 1
0 0.2 0.3 0.1 0.2      # Object 2
```

## ⚙️ Configuration File (data.yaml)

Create a `data.yaml` file in your dataset root:

```yaml
# Paths to image directories
train: images/train
val: images/val
test: images/test  # optional

# Number of classes (always 1 for YOLO-One)
nc: 1

# Class names
names:
  0: your_object_name
```

## 🚀 Quick Start Examples

### Using Recommended Datasets

```bash
# Download and convert COCO Person subset
python scripts/convert_coco_person.py --output ./datasets/coco_person

# Download WIDER FACE dataset
python scripts/download_wider_face.py --output ./datasets/wider_face

# Use your custom dataset
python scripts/validate_dataset.py ./datasets/my_dataset
```

### Validate Your Dataset

```bash
# Validate dataset structure and annotations
python scripts/validate_dataset.py /path/to/your/dataset

# Expected output:
# ✅ Dataset is ready for YOLO-One training!
# 📊 Total Images: 1,000
# 📊 Total Annotations: 2,500
```

## 📊 Dataset Requirements

### **Minimum Requirements:**

- **Images:** At least 500 images per class
- **Annotations:** YOLO format with normalized coordinates
- **Splits:** 80% train, 20% validation recommended
- **Quality:** Clear, diverse, representative images

### **Recommended:**

- **Images:** 1,000+ images for good performance
- **Resolution:** 640x640 or higher
- **Diversity:** Various lighting, angles, scales
- **Balance:** Even distribution of object sizes

## 🛠️ Dataset Preparation Tools

We provide several tools to help you prepare your dataset:

```bash
# Validate existing dataset
python tools/validate_dataset.py /path/to/dataset

# Convert from COCO format
python tools/convert_coco.py \
    --annotations /path/to/annotations.json \
    --images /path/to/images \
    --class-name "person" \
    --output /path/to/yolo_dataset

# Convert from Pascal VOC
python tools/convert_voc.py \
    --input /path/to/voc/dataset \
    --class-name "car" \
    --output /path/to/yolo_dataset

# Visualize annotations
python tools/visualize_dataset.py /path/to/dataset --num-samples 10
```

## ❓ Common Issues

### **Missing Label Files**

```bash
# Check for missing labels
python tools/check_missing_labels.py /path/to/dataset
```

### **Invalid Coordinates**

```bash
# Validate coordinate ranges
python tools/validate_coordinates.py /path/to/dataset
```

### **Class ID Issues**

For YOLO-One, all class IDs must be `0`. Use our conversion tools to fix this automatically.

## 📈 Dataset Statistics

After preparing your dataset, generate statistics:

```bash
python tools/dataset_stats.py /path/to/dataset

# Output:
# 📊 Dataset Statistics
# ├── Total Images: 1,000
# ├── Training Images: 800
# ├── Validation Images: 200
# ├── Average Objects/Image: 2.5
# └── Object Size Distribution: [small: 30%, medium: 50%, large: 20%]
```

## 🎯 Ready to Train

Once your dataset is validated:

```bash
python train.py \
    --data /path/to/your/dataset/data.yaml \
    --model-size nano \
    --epochs 300 \
    --batch-size 64
```

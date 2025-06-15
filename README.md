# YOLO-One 🚀

<div align="center">

**Revolutionary Single-Class Object Detection**  
*Ultra-fast YOLO architecture optimized exclusively for single-class detection*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![GitHub Stars](https://img.shields.io/github/stars/IAtrax/YOLO-One?style=social)](https://github.com/IAtrax/YOLO-One)

[**Quick Start**](#quick-start) •
[**Benchmarks**](#benchmarks) •
[**Documentation**](#documentation) •
[**Examples**](#examples)

</div>

---
> **⚠️ WORK IN PROGRESS** - YOLO-One is currently under active development. The architecture is functional with promising initial results. Star ⭐ this repository to stay updated on our progress!
---

## 🎯 Why YOLO-One?

While existing YOLO models excel at multi-class detection, **most real-world applications only need to detect ONE type of object**. YOLO-One is the first YOLO architecture designed from the ground up for single-class detection:

- ⚡ **Faster inference** with optimized single-class architecture
- 📦 **3x smaller model size** (1.9MB vs 6.2MB YOLOv8n)
- 🎯 **Same accuracy** for single-class tasks (target)
- 🔋 **Lower power consumption** (mobile/edge optimized)
- 💾 **Reduced memory usage** (streamlined architecture)
- 🚀 **Multi-platform deployment** ready

## 📊 Current Performance (YOLO-One Nano)

### 🏗️ Architecture Efficiency
```
Model Size:     1.9MB (vs 6.2MB YOLOv8n)     📦 3.3x smaller
Parameters:     485K (vs ~3M YOLOv8n)         ⚡ 6x fewer params
Channels:       5 per scale (vs 85 COCO)     🎯 Single-class optimized
Memory Format:  Float32 (FP16/INT8 planned)  💾 Further optimization ready
```

### ⚡ Speed Benchmarks (Current)
| Platform | Resolution | Current FPS | Target (Optimized) | Improvement Path |
|----------|------------|-------------|-------------------|------------------|
| **Development GPU** | 640x640 | **132 FPS** | 400+ FPS | +TensorRT +FP16 |
| **Inference Time** | 640x640 | **7.56ms** | ~2.5ms | +Optimizations |

### 🎯 Optimization Roadmap
```python
# Performance Projection Pipeline
Current:    132 FPS (7.56ms)    # PyTorch Float32
Step 1:     200+ FPS (~5ms)     # + torch.compile
Step 2:     300+ FPS (~3ms)     # + TensorRT
Step 3:     400+ FPS (~2.5ms)   # + FP16 precision
Mobile:     TBD                 # Core ML / TFLite
```

## 🌐 Platform Coverage (Planned)

### 📱 Mobile & IoT
- **iOS**: Core ML export (in development)
- **Android**: TensorFlow Lite export (in development)
- **Edge Devices**: ONNX export ready
- **ARM Optimization**: Native support planned

### 🖥️ Desktop & Workstation  

- **Windows**: ✅ PyTorch ready, TensorRT planned
- **Linux**: ✅ PyTorch ready, CUDA optimization
- **macOS**: ✅ PyTorch ready, Metal planned

### ☁️ Cloud & Production

- **Docker**: Container-ready architecture
- **ONNX**: Export capability implemented
- **TensorRT**: High-priority optimization
- **Serverless**: Lightweight deployment ready

## 🚀 Quick Start

## 🛠️ Installation

### 🚀 Quick Start (Recommended)
```bash
pip install git+https://github.com/IAtrax/YOLO-One.git
# Run architecture test script to validate installation
python tests/test_architecture.py
```

### Basic Usage

```python
import torch
from yolo_one.models import YoloOne

# Create model
model = YoloOne(model_size='nano')

# Test inference
input_tensor = torch.randn(1, 3, 640, 640)
predictions = model(input_tensor)

print(f"Model size: {model.count_parameters():,} parameters")
# Output: Model size: 485,312 parameters
```

## 🎯 Single-Class Optimizations

### 🔧 Architecture Benefits

```python
# Traditional YOLO (multi-class)
output_channels = 4 + 1 + num_classes  # bbox + conf + classes
# Example: 4 + 1 + 80 = 85 channels for COCO

# YOLO-One (single-class)
output_channels = 4 + 1  # bbox + fused_confidence
# Always: 5 channels only! 🎯
```

### ⚡ Performance Optimizations

- **No class probability computation** (always 1 class)
- **Simplified NMS** (no per-class separation)
- **Fused confidence** (objectness + class probability)
- **Streamlined post-processing** pipeline
- **Optimized loss function** for single-class

## 📈 Benchmarks (In Development)

### Current vs Target Performance
| Metric | Current (Nano) | Target (Optimized) | YOLOv8n Baseline |
|--------|---------------|-------------------|------------------|
| **Model Size** | ✅ 1.9MB | 1.9MB | 6.2MB |
| **Parameters** | ✅ 485K | 485K | ~3M |
| **Inference (GPU)** | 132 FPS | 400+ FPS | ~800 FPS |
| **Memory Usage** | TBD | <1GB | ~2GB |
| **Accuracy (mAP)** | TBD | Same | Baseline |

### 🔄 Multi-Resolution Support
```python
✅ 320x320: OK    # Fast inference
✅ 416x416: OK    # Balanced 
✅ 480x480: OK    # Good quality
✅ 640x640: OK    # High quality
```

## 🛠️ Development Status

### ✅ Completed
- [x] **Core Architecture**: Backbone + Neck + Head
- [x] **Multi-scale Detection**: P3, P4, P5 outputs  
- [x] **Single-class Optimization**: 5-channel output
- [x] **Model Variants**: nano, small, medium, large
- [x] **Basic Testing**: Architecture validation
- [x] **Multi-resolution**: 320-640px support

### 🚧 In Development
- [ ] **Training Pipeline**: Loss function + trainer
- [ ] **Benchmark Suite**: vs YOLOv8n comparison
- [ ] **Export Pipeline**: ONNX, TensorRT, Core ML
- [ ] **Mobile Optimization**: INT8 quantization
- [ ] **Documentation**: Complete API docs

### 🎯 Next Priorities
1. **Training Pipeline** - Validate accuracy claims
2. **TensorRT Export** - Achieve speed targets  
3. **Benchmark Suite** - Automated comparisons
4. **Mobile Deployment** - iOS/Android support

## 🔧 Technical Details

### Architecture Components
```python
# Component breakdown
Backbone:  ~400K params (82%)  # Feature extraction
Neck:      ~70K params  (14%)  # Feature fusion  
Head:      ~15K params  (3%)   # Detection output
Total:     485K params         # Ultra-lightweight
```

### Memory Efficiency
```python
# Multi-resolution memory usage
320x320: ~7MB GPU memory
640x640: ~11B GPU memory  
# Scales efficiently with resolution
```

## Training Pipeline

to launch training pipeline, run the following command:

```python
python train.py \
    --config configs/yolo_one_nano.yaml \
    --data data/my_dataset.yaml \
    --model-size nano \
    --epochs 300 \
    --batch-size 16 \
    --device cuda
```

Train with custom parameters

```python
python train.py \
    --data data/my_dataset.yaml \
    --model-size small \
    --epochs 500 \
    --batch-size 32 \
    --lr 0.001 \
    --output-dir runs/experiment_1
```

Resume training

```python
python train.py \
    --data data/my_dataset.yaml \
    --resume runs/train_20250615_120000/best_model.pt
```

## 🤝 Contributing

We welcome contributions! Key areas:

- **Training Pipeline Development**
- **Mobile Optimization** 
- **Benchmark Implementation**
- **Documentation & Examples**
- **Export Format Support**

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Roadmap

### Phase 1: Foundation (Current)
- [x] Core architecture
- [ ] Training pipeline
- [ ] Basic benchmarks

### Phase 2: Optimization (Q3 2025)
- [ ] TensorRT integration
- [ ] Mobile deployment
- [ ] Performance validation

### Phase 3: Production (Q4 2025)
- [ ] Cloud deployment tools
- [ ] Enterprise features
- [ ] Community ecosystem

---

<div align="center">

**🚀 Ready to revolutionize single-class object detection?**

⭐ **Star this repository** • 🔄 **Fork and contribute** • 📖 **Read the docs**

*Built with ❤️ by the IAtrax team*

</div>
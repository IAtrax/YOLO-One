#!/usr/bin/env python3
"""
YOLO-One Project Structure Builder
Creates the complete directory structure for YOLO-One project
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the complete YOLO-One project structure"""

    # Base project structure
    structure = {
        # Root files
        ".": [
            "README.md",
            "LICENSE",
            ".gitignore",
            "CMakeLists.txt",
            "requirements.txt",
            "CONTRIBUTING.md",
            "CHANGELOG.md"
        ],

        # C++ Core Implementation
        "cpp": [],
        "cpp/src": [],
        "cpp/src/core": [
            "yolo_one.hpp",
            "yolo_one.cpp",
            "architecture.hpp",
            "architecture.cpp",
            "backbone.hpp",
            "backbone.cpp",
            "neck.hpp",
            "neck.cpp",
            "head.hpp",
            "head.cpp"
        ],
        "cpp/src/inference": [
            "inference_engine.hpp",
            "inference_engine.cpp",
            "tensor_processor.hpp",
            "tensor_processor.cpp",
            "memory_manager.hpp",
            "memory_manager.cpp"
        ],
        "cpp/src/optimizations": [
            "simd_ops.hpp",
            "simd_ops.cpp",
            "threading.hpp",
            "threading.cpp",
            "quantization.hpp",
            "quantization.cpp"
        ],
        "cpp/src/exports": [
            "onnx_exporter.hpp",
            "onnx_exporter.cpp",
            "tensorrt_exporter.hpp",
            "tensorrt_exporter.cpp",
            "openvino_exporter.hpp",
            "openvino_exporter.cpp"
        ],
        "cpp/src/utils": [
            "image_utils.hpp",
            "image_utils.cpp",
            "math_utils.hpp",
            "math_utils.cpp",
            "logger.hpp",
            "logger.cpp"
        ],

        # C++ Headers
        "cpp/include": [],
        "cpp/include/yolo_one": [
            "detector.hpp",
            "types.hpp",
            "config.hpp",
            "version.hpp"
        ],

        # C++ Tests
        "cpp/tests": [
            "test_core.cpp",
            "test_inference.cpp",
            "test_optimizations.cpp",
            "benchmark_performance.cpp",
            "CMakeLists.txt"
        ],

        # Python Interface
        "python": [],
        "python/yolo_one": [
            "__init__.py",
            "models.py",
            "training.py",
            "inference.py",
            "utils.py",
            "datasets.py",
            "losses.py",
            "metrics.py",
            "callbacks.py",
            "exports.py"
        ],
        "python/yolo_one/nn": [
            "__init__.py",
            "modules.py",
            "blocks.py",
            "activations.py",
            "attention.py"
        ],
        "python/yolo_one/data": [
            "__init__.py",
            "dataloaders.py",
            "augmentations.py",
            "preprocessing.py",
            "postprocessing.py"
        ],
        "python/yolo_one/utils": [
            "__init__.py",
            "general.py",
            "metrics.py",
            "plots.py",
            "benchmarks.py"
        ],

        # Python Setup
        "python": [
            "setup.py",
            "setup.cfg",
            "pyproject.toml",
            "MANIFEST.in"
        ],

        # Bindings (PyBind11)
        "bindings": [
            "yolo_one_bind.cpp",
            "detector_bind.cpp",
            "utils_bind.cpp",
            "CMakeLists.txt"
        ],

        # Configuration Files
        "configs": [
            "yolo_one_nano.yaml",
            "yolo_one_tiny.yaml",
            "yolo_one_small.yaml",
            "yolo_one_medium.yaml",
            "yolo_one_large.yaml"
        ],
        "configs/datasets": [
            "coco_single.yaml",
            "voc_single.yaml",
            "custom_template.yaml"
        ],
        "configs/training": [
            "default.yaml",
            "mobile_optimized.yaml",
            "accuracy_focused.yaml",
            "speed_focused.yaml"
        ],

        # Scripts & Tools
        "scripts": [
            "train.py",
            "eval.py",
            "detect.py",
            "export.py",
            "benchmark.py",
            "download_models.py"
        ],
        "scripts/tools": [
            "convert_dataset.py",
            "analyze_model.py",
            "profile_performance.py",
            "validate_installation.py"
        ],

        # Examples & Demos
        "examples": [],
        "examples/basic": [
            "simple_detection.py",
            "batch_inference.py",
            "video_detection.py",
            "webcam_demo.py"
        ],
        "examples/advanced": [
            "custom_training.py",
            "model_optimization.py",
            "multi_gpu_training.py",
            "deployment_example.py"
        ],
        "examples/mobile": [
            "ios_coreml_example.py",
            "android_tflite_example.py",
            "raspberry_pi_example.py"
        ],

        # Benchmarks & Tests
        "benchmarks": [
            "performance_suite.py",
            "accuracy_evaluation.py",
            "memory_profiling.py",
            "mobile_benchmarks.py"
        ],
        "benchmarks/datasets": [
            "download_benchmarks.py",
            "prepare_datasets.py"
        ],
        "benchmarks/results": [
            "README.md"
        ],

        # Documentation
        "docs": [
            "index.md",
            "installation.md",
            "quickstart.md",
            "training.md",
            "deployment.md",
            "api_reference.md",
            "benchmarks.md",
            "contributing.md"
        ],
        "docs/tutorials": [
            "getting_started.md",
            "custom_dataset.md",
            "model_optimization.md",
            "mobile_deployment.md"
        ],
        "docs/assets": [
            "README.md"
        ],

        # Models & Weights
        "models": [
            "README.md",
            "download_weights.py"
        ],
        "models/configs": [
            "architectures.yaml"
        ],

        # Tests
        "tests": [],
        "tests/unit": [
            "__init__.py",
            "test_models.py",
            "test_training.py",
            "test_inference.py",
            "test_utils.py"
        ],
        "tests/integration": [
            "__init__.py",
            "test_end_to_end.py",
            "test_exports.py",
            "test_benchmarks.py"
        ],
        "tests/fixtures": [
            "sample_image.jpg",
            "test_config.yaml"
        ],

        # CI/CD & DevOps
        ".github": [],
        ".github/workflows": [
            "ci.yml",
            "build_wheels.yml",
            "docs.yml",
            "benchmarks.yml"
        ],
        ".github/ISSUE_TEMPLATE": [
            "bug_report.md",
            "feature_request.md",
            "question.md"
        ],

        # Docker
        "docker": [
            "Dockerfile",
            "Dockerfile.cpu",
            "Dockerfile.gpu",
            "docker-compose.yml",
            "entrypoint.sh"
        ],

        # Data & Assets
        "data": [
            "README.md",
            ".gitkeep"
        ],
        "data/samples": [
            "README.md"
        ],

        # Research & Paper
        "paper": [
            "README.md",
            "yolo_one_paper.tex",
            "bibliography.bib"
        ],
        "paper/figures": [
            "README.md"
        ],
        "paper/tables": [
            "benchmarks.tex"
        ]
    }

    # Get current directory or create new project directory
    base_path = Path(".")

    print("🚀 Creating YOLO-One project structure...")
    print(f"📁 Base directory: {base_path.absolute()}")

    # Create directories and files
    for dir_path, files in structure.items():
        # Create directory
        full_dir_path = base_path / dir_path
        full_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📂 Created: {dir_path}/")

        # Create files in directory
        for file_name in files:
            file_path = full_dir_path / file_name
            if not file_path.exists():
                # Create file with basic content based on extension
                content = get_file_template(file_name, dir_path)
                file_path.write_text(content, encoding='utf-8')
                print(f"📄 Created: {dir_path}/{file_name}")

    print("\n✅ YOLO-One project structure created successfully!")
    print("\n🎯 Next steps:")
    print("1. cd into your project directory")
    print("2. Initialize git: git init")
    print("3. Start with C++ core implementation in cpp/src/core/")
    print("4. Set up Python environment: python -m venv venv")
    print("5. Install dependencies: pip install -r requirements.txt")

    return True

def get_file_template(filename, dir_path):
    """Get appropriate template content for different file types"""

    # File extension templates
    if filename.endswith('.py'):
        return get_python_template(filename, dir_path)
    elif filename.endswith('.cpp'):
        return get_cpp_template(filename, dir_path)
    elif filename.endswith('.hpp'):
        return get_hpp_template(filename, dir_path)
    elif filename.endswith('.md'):
        return get_markdown_template(filename, dir_path)
    elif filename.endswith('.yaml') or filename.endswith('.yml'):
        return get_yaml_template(filename, dir_path)
    elif filename == 'CMakeLists.txt':
        return get_cmake_template(dir_path)
    elif filename == 'requirements.txt':
        return get_requirements_template()
    elif filename == '.gitignore':
        return get_gitignore_template()
    elif filename == 'LICENSE':
        return get_license_template()
    else:
        return f"# {filename}\n# TODO: Implement content for {filename}\n"

def get_python_template(filename, dir_path):
    """Python file templates"""
    if filename == '__init__.py':
        if 'yolo_one' in dir_path:
            return '''"""YOLO-One: Ultra-fast single-class object detection"""

__version__ = "0.1.0"
__author__ = "IAtrax Team"
__email__ = "contact@iatrax.com"

from .models import YOLOOne
from .inference import detect, load_model
from .training import train

__all__ = ['YOLOOne', 'detect', 'load_model', 'train']
'''
        else:
            return f'"""Package: {dir_path}"""\n'

    elif filename == 'setup.py':
        return '''"""Setup script for YOLO-One"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

# C++ extension modules
ext_modules = [
    Pybind11Extension(
        "yolo_one_core",
        [
            "bindings/yolo_one_bind.cpp",
            "bindings/detector_bind.cpp",
        ],
        include_dirs=["cpp/include"],
        cxx_std=17,
    ),
]

setup(
    name="yolo-one",
    version="0.1.0",
    author="IAtrax Team",
    description="Ultra-fast single-class object detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "pyyaml>=5.4.0",
        "pillow>=8.3.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "export": ["onnx", "onnxruntime"],
    },
)
'''

    else:
        return f'''"""
{filename.replace('.py', '').title()} module for YOLO-One

TODO: Implement {filename} functionality
"""

# TODO: Add imports
# TODO: Add implementation
'''

def get_cpp_template(filename, dir_path):
    """C++ source file templates"""
    header_name = filename.replace('.cpp', '.hpp')
    class_name = filename.replace('.cpp', '').replace('_', ' ').title().replace(' ', '')

    return f'''/**
 * @file {filename}
 * @brief {class_name} implementation for YOLO-One
 * @author IAtrax Team
 */

#include "{header_name}"
#include <iostream>
#include <memory>

namespace yolo_one {{

// TODO: Implement {class_name} methods

}} // namespace yolo_one
'''

def get_hpp_template(filename, dir_path):
    """C++ header file templates"""
    guard_name = f"YOLO_ONE_{filename.replace('.hpp', '').upper()}_HPP"
    class_name = filename.replace('.hpp', '').replace('_', ' ').title().replace(' ', '')

    return f'''/**
 * @file {filename}
 * @brief {class_name} header for YOLO-One
 * @author IAtrax Team
 */

#ifndef {guard_name}
#define {guard_name}

#include <vector>
#include <memory>

namespace yolo_one {{

/**
 * @class {class_name}
 * @brief TODO: Brief description of {class_name}
 */
class {class_name} {{
public:
    {class_name}();
    ~{class_name}();

    // TODO: Add public methods

private:
    // TODO: Add private members
}};

}} // namespace yolo_one

#endif // {guard_name}
'''

def get_markdown_template(filename, dir_path):
    """Markdown file templates"""
    if filename == 'README.md':
        if dir_path == '.':
            return open('README.md', 'r').read() if os.path.exists('README.md') else "# YOLO-One\n\nUltra-fast single-class object detection\n"
        else:
            return f"# {dir_path.split('/')[-1].title()}\n\nTODO: Add documentation for {dir_path}\n"
    else:
        title = filename.replace('.md', '').replace('_', ' ').title()
        return f"# {title}\n\nTODO: Add content for {title}\n"

def get_yaml_template(filename, dir_path):
    """YAML configuration templates"""
    if 'yolo_one' in filename:
        model_size = filename.replace('yolo_one_', '').replace('.yaml', '')
        return f'''# YOLO-One {model_size.title()} Configuration

# Model Architecture
backbone:
  type: "EfficientNet"
  variant: "b0"
  pretrained: true

neck:
  type: "PAFPN"
  channels: [64, 128, 256]

head:
  type: "SingleClassHead"
  channels: 64
  num_classes: 1

# Training Parameters
epochs: 100
batch_size: 32
learning_rate: 0.001
optimizer: "AdamW"
scheduler: "CosineAnnealingLR"

# Data Augmentation
augmentation:
  mosaic: 0.5
  mixup: 0.1
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
'''
    else:
        return f"# {filename}\n# TODO: Add YAML configuration\n"

def get_cmake_template(dir_path):
    """CMake templates"""
    if dir_path == '.':
        return '''cmake_minimum_required(VERSION 3.15)
project(YOLOOne VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(cpp/include)

# Add subdirectories
add_subdirectory(cpp)
add_subdirectory(bindings)

# Build options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(USE_CUDA "Enable CUDA support" OFF)

if(BUILD_TESTS)
    add_subdirectory(cpp/tests)
endif()
'''
    else:
        return f'''# CMakeLists.txt for {dir_path}

# TODO: Add CMake configuration for {dir_path}
'''

def get_requirements_template():
    """Python requirements template"""
    return '''# Core dependencies
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.5.0
numpy>=1.21.0
pyyaml>=5.4.0
pillow>=8.3.0
tqdm>=4.62.0

# Development dependencies
pytest>=6.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950

# Export dependencies (optional)
onnx>=1.12.0
onnxruntime>=1.12.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
requests>=2.28.0
scipy>=1.8.0
'''

def get_gitignore_template():
    """Git ignore template"""
    return '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
out/

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Model weights and data
*.pt
*.pth
*.onnx
*.trt
*.engine
data/
datasets/
runs/

# Logs
logs/
*.log

# CMake
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
Makefile
'''

def get_license_template():
    """MIT License template"""
    return '''MIT License

Copyright (c) 2025 IAtrax

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

if __name__ == "__main__":
    print("🚀 YOLO-One Project Structure Builder")
    print("=" * 50)

    try:
        success = create_directory_structure()
        if success:
            print("\n🎉 Project structure created successfully!")
            print("💡 Run this script in your desired project directory")
            print("🔧 Ready to start developing YOLO-One!")
        else:
            print("❌ Failed to create project structure")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error creating project structure: {e}")
        sys.exit(1)

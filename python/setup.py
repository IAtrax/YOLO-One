"""Setup script for YOLO-One"""

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

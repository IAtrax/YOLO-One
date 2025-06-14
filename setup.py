from setuptools import setup, find_packages

setup(
    name="yolo-one",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "opencv-python",
        "albumentations",
        "numpy",
        "pathlib",
    ],
    author="Iatrax Team",
    description="Revolutionary Single-Class Object Detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IAtrax/YOLO-One",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
    ],
)
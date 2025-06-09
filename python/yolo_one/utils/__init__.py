"""YOLO-One: Ultra-fast single-class object detection"""

__version__ = "0.1.0"
__author__ = "IAtrax Team"
__email__ = "contact@iatrax.com"

from .models import YOLOOne
from .inference import detect, load_model
from .training import train

__all__ = ['YOLOOne', 'detect', 'load_model', 'train']

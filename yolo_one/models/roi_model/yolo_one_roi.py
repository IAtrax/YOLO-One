"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

BACKBONE MODULE FOR YOLO-ONE

This module contains the backbone for YOLO-ONE based on ROI detection
Principe: 
    We propose a backbone with 2 large main blocks: 
       1) A coarse area of interest detection block
       2) A fine box detection block based on the areas of interest found (ROI).
"""
import torch
import torch.nn as nn
from yolo_one.models.common import Conv
from yolo_one.models.yolo_one_model import YoloOne
from typing import List, Dict, Any

class RoiDetectionBlock(nn.Module):
    def __init__(self, num_rois: int = 5):
        """
        Initializes the Region of Interest (ROI) detection block.

        This neural network takes an image as input and predicts a fixed number
        of coarse bounding boxes (ROIs).

        Args:
            num_rois (int): The number of regions of interest to predict.
        """
        super().__init__()
        self.num_rois = num_rois

        # 1. Backbone: A custom, ultra-lightweight CNN.
        self.backbone = nn.Sequential(
            # Input: [batch, 3, H, W]
            Conv(3, 16, kernel_size=3, stride=2),    # -> [batch, 16, H/2, W/2]
            Conv(16, 32, kernel_size=3, stride=2),   # -> [batch, 32, H/4, W/4]
            Conv(32, 64, kernel_size=3, stride=2),   # -> [batch, 64, H/8, W/8]
        )
        # The output of the backbone has 64 channels.
        num_output_features = 64

        # 2. Head: Predicts box coordinates from features
        self.head = nn.Sequential(
            # Pool spatial features into a single vector per image
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), # -> [batch, num_output_features]
            nn.Linear(num_output_features, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_rois * 4), # 4 coordinates (x,y,w,h) per ROI
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ROI detection block.

        Args:
            x (torch.Tensor): Input tensor (batch of images), e.g., [16, 3, 640, 640].

        Returns:
            torch.Tensor: Output tensor containing the ROI coordinates.
                          Shape: [batch_size, num_rois, 4]
                          Each ROI is (x_center, y_center, width, height), normalized.
        """
        features = self.backbone(x)
        predictions = self.head(features)
        # Reshape the output to have the shape [batch, num_rois, 4]
        return predictions.view(-1, self.num_rois, 4)


class BoxDetectionBlock(nn.Module):
    def __init__(self, model_size: str = 'nano'):
        """
        Initialize the BoxDetectionBlock using composition.
        This block contains the main components of YOLO: backbone, neck, and head.
        """
        super().__init__()
        self.model = YoloOne(model_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BoxDetectionBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)
        
class YoloOneRoi(nn.Module):
    def __init__(self):
        """
        Initialize the YoloOneRoi model.

        This method initializes the YoloOneRoi model by calling the __init__ method of the parent class
        and creating two child modules: RoiDetectionBlock and BoxDetectionBlock.

        The RoiDetectionBlock is the first block of the YoloOneRoi model, responsible for detecting areas of interest
        in the input image. The BoxDetectionBlock is the second block, responsible for detecting the boxes inside
        the areas of interest detected by the RoiDetectionBlock.

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        self.roi_detection_block = RoiDetectionBlock()
        self.box_detection_block = BoxDetectionBlock(model_size='nano')
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the YoloOneRoi model.

        Args:
            x (torch.Tensor): Input tensor.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Output tensor.
        """
        # 1. Detect coarse Regions of Interest (ROI)
        # rois -> shape: [batch, num_rois, 4]
        rois = self.roi_detection_block(x)

        # --- ROI processing logic ---
        # Ideally, here we would use the 'rois' to crop the image 'x'
        # into several smaller images, one for each ROI.
        # An operation like `torchvision.ops.roi_align` would be used.
        # For now, we pass the full image to the next block for simplicity.
        # cropped_images = crop_and_resize(x, rois)
        
        # 2. Detect fine-grained boxes within the regions (or the full image for now)
        # final_predictions = self.box_detection_block(cropped_images)
        final_predictions = self.box_detection_block(x)

        # 3. Loss calculation
        # The total loss would be a combination of the ROI loss and the final loss.
        # loss_roi = compute_roi_loss(rois, ground_truth_rois)
        # loss_final = self.box_detection_block.model.loss(final_predictions, labels)
        # total_loss = loss_roi + loss_final
        
        return final_predictions # During inference, return the predictions
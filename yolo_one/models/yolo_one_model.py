"""
Iatrax Team - 2025 - https://iatrax.com

COMPLETE YOLO-ONE MODEL ASSEMBLY
"""
from .yolo_one_backbone import create_yolo_one_backbone
from .yolo_one_neck import create_yolo_one_neck
from .yolo_one_head import create_yolo_one_head
import torch.nn as nn
from .common import GatingNetwork

class YoloOne(nn.Module):
    def __init__(self, model_size='nano', use_moe: bool = False):
        super().__init__()
        self.use_moe = use_moe

        # Create each component using its factory function
        self.backbone = create_yolo_one_backbone(model_size)
        self.neck = create_yolo_one_neck(model_size, self.backbone.out_channels)
        self.head = create_yolo_one_head(model_size, in_channels=self.neck.out_channels)

        if self.use_moe:
            num_experts = len(self.neck.out_channels)
            # Use channels from the last neck output (P5) for the gating network
            gating_in_channels = self.neck.out_channels[-1]
            self.gating_network = GatingNetwork(in_channels=gating_in_channels, num_experts=num_experts)
    
    def forward(self, x, decode=False, img_size=None):
        features_from_backbone = self.backbone(x)
        features_from_neck = self.neck(features_from_backbone)

        # --- MoE Logic ---
        if self.use_moe and hasattr(self, 'gating_network'):
            # Use the P5 feature map (last in the list, lowest resolution) for the gating decision
            gate_scores = self.gating_network(features_from_neck[-1])  # Shape: [B, num_experts]

            # During inference, we perform "hard" gating to save computation
            if not self.training:
                features_for_head = []
                for i, feat in enumerate(features_from_neck):
                    # Create a mask for the batch for this expert: [B, 1, 1, 1]
                    # Cast mask to feature dtype to avoid mixed-precision errors
                    score_mask = (gate_scores[:, i] > 0.5).to(feat.dtype).view(-1, 1, 1, 1)
                    # Multiply the feature map by the mask. Inactive items in batch become zero.
                    features_for_head.append(feat * score_mask)
            else:
                # During training, we pass all features to the head for "soft" gating on the loss
                features_for_head = features_from_neck

            # The head processes the (potentially partially zeroed-out) features
            outputs = self.head(features_for_head, decode=decode, img_size=img_size)
            # Rename 'preds' to 'detections' for compatibility with the loss function
            if 'preds' in outputs:
                outputs['detections'] = outputs.pop('preds')

            outputs['gate_scores'] = gate_scores

            # During training, we apply the continuous gate scores to the loss-relevant outputs
            if self.training:
                for key in ['preds', 'obj_logits', 'bbox']:
                    if key in outputs:
                        gated_tensors = []
                        for i, tensor in enumerate(outputs.get(key, [])):
                            # Reshape score for broadcasting: [B, num_experts] -> [B, 1, 1, 1]
                            score = gate_scores[:, i].view(-1, 1, 1, 1)
                            gated_tensors.append(tensor * score)
                        outputs[key] = gated_tensors
        else:
            # Original forward pass without MoE
            outputs = self.head(features_from_neck, decode=decode, img_size=img_size)
            # Rename 'preds' to 'detections' for compatibility with the loss function
            if 'preds' in outputs:
                outputs['detections'] = outputs.pop('preds')

        return outputs

        
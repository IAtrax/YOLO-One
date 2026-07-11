"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

YOLO-One Detection Head
Mono-class, 5-channel head (obj, x, y, w, h) with decoupled towers and optional feature refinement.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from yolo_one.models.common import Conv

class Scale(nn.Module):
    """Learnable scalar (per level) to stabilize bbox regression magnitude."""
    def __init__(self, init_value: float = 1.0) -> None:
        """
        Initialize the learnable scalar for stabilizing bbox regression magnitude.

        Args:
            init_value (float, optional): Initial value of the learnable scalar. Defaults to 1.0.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DecoupledHeadPerLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: Optional[int] = None,
        num_convs: int = 2,
        obj_prior: float = 0.01,
    ) -> None:
        """
        Initialize a DecoupledHeadPerLevel module.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (Optional[int], optional): Number of middle channels. Defaults to None.
            num_convs (int, optional): Number of convolution blocks in each tower. Defaults to 2.
            obj_prior (float, optional): Initial objectness prior. Defaults to 0.01.
        """
        super().__init__()
        c_mid = mid_channels or in_channels

        obj_tower_layers = [Conv(in_channels, c_mid, kernel_size=3, stride=1)]
        reg_tower_layers = [Conv(in_channels, c_mid, kernel_size=3, stride=1)]
        for _ in range(num_convs - 1):
            obj_tower_layers.append(Conv(c_mid, c_mid, kernel_size=3, stride=1))
            reg_tower_layers.append(Conv(c_mid, c_mid, kernel_size=3, stride=1))
        self.obj_tower = nn.Sequential(*obj_tower_layers)
        self.reg_tower = nn.Sequential(*reg_tower_layers)

        self.obj_out = nn.Conv2d(c_mid, 1, kernel_size=1)
        self.bbox_out = nn.Conv2d(c_mid, 4, kernel_size=1)
        self.reg_scale = Scale(2.0)  # 1.0 to 4.0


        self._init_weights(obj_prior)

    def _init_weights(self, prior: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        bias = -math.log((1.0 - prior) / max(prior, 1e-6))
        nn.init.constant_(self.obj_out.bias, bias)

        if self.bbox_out.bias is not None:
            nn.init.constant_(self.bbox_out.bias[2], 1.2)  # w
            nn.init.constant_(self.bbox_out.bias[3], 1.2)  # h

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_obj = self.obj_tower(x)
        h_reg = self.reg_tower(x)

        obj_logits = self.obj_out(h_obj)
        bbox = self.reg_scale(self.bbox_out(h_reg))

        detections = torch.cat([obj_logits, bbox], dim=1)
        return {"detections": detections, "obj_logits": obj_logits, "bbox": bbox}


class YoloOneDetectionHead(nn.Module):
    """
    YOLO-One mono-class head with 5 channels per level (obj, x, y, w, h).

    Inputs:
        - x: list of [P3, P4, P5], each tensor [B, Ck, Hk, Wk]
    Outputs (dict):
        - detections:   list of [B, 5, Hk, Wk]
        - obj_logits:   list of [B, 1, Hk, Wk]
        - bbox:         list of [B, 4, Hk, Wk]
        - features:     list of [B, Ck, Hk, Wk] (refined or pass-through), if enabled
        - decoded:      list of [B, 5, Hk, Wk] with xyxy normalized to image and conf=sigmoid(obj), if decode=True
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        
        super().__init__()
        self.config = config

        self.in_channels: List[int] = list(config["in_channels"])
        self.strides: List[int] = list(config.get("strides", [8, 16, 32]))
        if len(self.in_channels) != len(self.strides):
            raise ValueError("in_channels and strides must have the same length")

        head_mid: Optional[Union[int, Sequence[int]]] = config.get("head_channels", None)
        num_head_convs: int = int(config.get("num_head_convs", 2))
        obj_prior: float = float(config.get("obj_prior", 0.01))

        self.level_heads = nn.ModuleList()
        for i, c_in in enumerate(self.in_channels):
            if isinstance(head_mid, int):
                c_mid = head_mid
            elif isinstance(head_mid, (list, tuple)):
                if len(head_mid) != len(self.in_channels):
                    raise ValueError("When head_channels is a list/tuple, its length must match in_channels")
                c_mid = int(head_mid[i])
            else:
                c_mid = None

            self.level_heads.append(
                DecoupledHeadPerLevel(
                    in_channels=c_in,
                    mid_channels=c_mid,
                    num_convs=num_head_convs,
                    obj_prior=obj_prior,
                )
            )

    @torch.no_grad()
    def _make_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Create a 2xHxW grid with (x, y) indices on the given device."""
        gy, gx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        return torch.stack((gx, gy), dim=0).float()

    def forward(
        self,
        x: List[torch.Tensor],
        decode: bool = False,
        img_size: Optional[Sequence[int]] = None,
        gate_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        if not isinstance(x, (list, tuple)) or len(x) != len(self.in_channels):
            raise ValueError("Expected a list [P3, P4, P5] matching configured in_channels")

        detections: List[torch.Tensor] = []
        obj_logits: List[torch.Tensor] = []
        bboxs: List[torch.Tensor] = []    
        for i, feat in enumerate(x):
            out = self.level_heads[i](feat)
            detections.append(out["detections"])
            obj_logits.append(out["obj_logits"])
            bboxs.append(out["bbox"])
                
        outputs: Dict[str, Any] = {
            "detections": detections,
            "obj_logits": obj_logits,
            "bbox": bboxs,
        }

        if decode:
            if img_size is None or len(img_size) != 2:
                raise ValueError("img_size=[H_img, W_img] is required when decode=True")
            h_img, w_img = int(img_size[0]), int(img_size[1])

            activated = torch.ones(len(self.level_heads), dtype=torch.bool, device=x[0].device)

            decoded: List[torch.Tensor] = []
            for i, (pred, stride) in enumerate(zip(detections, self.strides)):
                if not activated[i]:
                    continue

                b, _, hk, wk = pred.shape
                obj = torch.sigmoid(pred[:, :1])
                xy = torch.sigmoid(pred[:, 1:3])
                wh = torch.sigmoid(pred[:, 3:5]).clamp(max=1E4)

                grid = self._make_grid(hk, wk, pred.device)
                xy_center_pix = (grid.unsqueeze(0) + xy) * float(stride)
                wh_pix = wh * float(stride)

                # CONVERT TO XYXY FORMAT
                x1 = xy_center_pix[:, 0:1] - wh_pix[:, 0:1] / 2
                y1 = xy_center_pix[:, 1:2] - wh_pix[:, 1:2] / 2
                x2 = xy_center_pix[:, 0:1] + wh_pix[:, 0:1] / 2
                y2 = xy_center_pix[:, 1:2] + wh_pix[:, 1:2] / 2

                xyxy_pix = torch.cat([x1, y1, x2, y2], dim=1)

                # NORMALIZE TO [0, 1]
                norm_tensor = torch.tensor([w_img, h_img, w_img, h_img], device=pred.device, dtype=pred.dtype).view(1, 4, 1, 1)
                xyxy_norm = xyxy_pix / norm_tensor

                # [B, 5, Hk, Wk] -> (x1, y1, x2, y2, conf) in xyxy format
                decoded.append(torch.cat([xyxy_norm, obj], dim=1))
            outputs["decoded"] = detections

        return outputs


def create_yolo_one_head(
    model_size: str,
    backbone: Optional[nn.Module] = None,
    in_channels: Optional[List[int]] = None,
    strides: Optional[List[int]] = None,
    **kwargs: Any,
) -> YoloOneDetectionHead:
    """
    Factory for the YOLO-One detection head.

    Args:
        model_size: Model scale id ('nano' | 'small' | 'medium' | 'large'). Currently informational.
        backbone:   Backbone instance exposing .out_channels (optional).
        in_channels:List of [C3, C4, C5] if backbone is not provided.
        strides:    List of strides for [P3, P4, P5], defaults to [8, 16, 32].
        kwargs:
            - head_channels: int | list[int] | None
            - return_features: bool
            - refine_features: bool
            - obj_prior: float

    Returns:
        YoloOneDetectionHead
    """
    if backbone is None and in_channels is None:
        raise ValueError("Provide either 'backbone' or 'in_channels'.")

    if in_channels is None:
        if not hasattr(backbone, "out_channels"):
            raise ValueError("Provided backbone does not expose 'out_channels'.")
        in_channels = list(getattr(backbone, "out_channels"))

    num_head_convs = 1 if model_size in ['nano', 'small'] else 2

    config: Dict[str, Any] = {
        "in_channels": in_channels,
        "strides": strides or [8, 16, 32],
        "head_channels": kwargs.get("head_channels", None),
        "num_head_convs": kwargs.get("num_head_convs", num_head_convs),
        "return_features": kwargs.get("return_features", True),
        "refine_features": kwargs.get("refine_features", False),
        "obj_prior": kwargs.get("obj_prior", 0.01),
    }
    return YoloOneDetectionHead(config)

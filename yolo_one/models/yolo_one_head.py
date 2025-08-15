"""
Iatrax Team - 2025 - https://iatrax.com

LICENSE: MIT

YOLO-One Detection Head
Mono-class, 5-channel head (obj, x, y, w, h) with decoupled towers and optional feature refinement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reusable Conv block from the repo (Conv-BN-Act, mobile-friendly)
from yolo_one.models.common import Conv


class Scale(nn.Module):
    """Learnable scalar (per level) to stabilize bbox regression magnitude."""
    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DecoupledHeadPerLevel(nn.Module):
    """
    Per-level head with decoupled towers for objectness and regression.

    Outputs:
        - obj_logits: [B, 1, H, W]
        - bbox:       [B, 4, H, W]  (xywh in cell/stride space; decoding handles conversion)
        - pred:       [B, 5, H, W]  (concat of obj_logits and bbox)
        - feat:       [B, C, H, W]  (optionally refined feature)
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: Optional[int] = None,
        obj_prior: float = 0.01,
        use_refine: bool = False,
    ) -> None:
        super().__init__()
        c_mid = mid_channels or in_channels

        # Lightweight decoupled towers: two 3x3 Conv blocks each
        self.obj_tower = nn.Sequential(
            Conv(in_channels, c_mid, kernel_size=3, stride=1),
            Conv(c_mid, c_mid, kernel_size=3, stride=1),
        )
        self.reg_tower = nn.Sequential(
            Conv(in_channels, c_mid, kernel_size=3, stride=1),
            Conv(c_mid, c_mid, kernel_size=3, stride=1),
        )

        # Prediction heads
        self.obj_out = nn.Conv2d(c_mid, 1, kernel_size=1)
        self.bbox_out = nn.Conv2d(c_mid, 4, kernel_size=1)
        self.reg_scale = Scale(1.0)

        # Optional light feature refinement (pass-through if disabled)
        self.refine = Conv(in_channels, in_channels, kernel_size=3, stride=1) if use_refine else nn.Identity()

        # Initialize weights and set objectness bias prior
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
        # Bias prior for objectness: bias = -log((1 - p) / p)
        bias = -math.log((1.0 - prior) / max(prior, 1e-6))
        nn.init.constant_(self.obj_out.bias, bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_obj = self.obj_tower(x)
        h_reg = self.reg_tower(x)

        obj_logits = self.obj_out(h_obj)                 # [B, 1, H, W]
        bbox = self.reg_scale(self.bbox_out(h_reg))      # [B, 4, H, W]

        pred = torch.cat([obj_logits, bbox], dim=1)      # [B, 5, H, W]
        feat_out = self.refine(x)
        return {"pred": pred, "obj_logits": obj_logits, "bbox": bbox, "feat": feat_out}


class YoloOneDetectionHead(nn.Module):
    """
    YOLO-One mono-class head with 5 channels per level (obj, x, y, w, h).

    Inputs:
        - x: list of [P3, P4, P5], each tensor [B, Ck, Hk, Wk]
    Outputs (dict):
        - preds:        list of [B, 5, Hk, Wk]
        - obj_logits:   list of [B, 1, Hk, Wk]
        - bbox:         list of [B, 4, Hk, Wk]
        - features:     list of [B, Ck, Hk, Wk] (refined or pass-through), if enabled
        - decoded:      list of [B, 5, Hk, Wk] with xywh normalized to image and conf=sigmoid(obj), if decode=True
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        self.in_channels: List[int] = list(config["in_channels"])
        self.strides: List[int] = list(config.get("strides", [8, 16, 32]))
        if len(self.in_channels) != len(self.strides):
            raise ValueError("in_channels and strides must have the same length")

        head_mid: Optional[Union[int, Sequence[int]]] = config.get("head_channels", None)
        self.return_features: bool = bool(config.get("return_features", True))
        self.refine_features: bool = bool(config.get("refine_features", False))
        obj_prior: float = float(config.get("obj_prior", 0.01))

        # Build per-level heads
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
                    obj_prior=obj_prior,
                    use_refine=self.refine_features,
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
        return torch.stack((gx, gy), dim=0).float()  # [2, H, W]

    def forward(
        self,
        x: List[torch.Tensor],
        decode: bool = False,
        img_size: Optional[Sequence[int]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        if not isinstance(x, (list, tuple)) or len(x) != len(self.in_channels):
            raise ValueError("Expected a list [P3, P4, P5] matching configured in_channels")

        preds: List[torch.Tensor] = []
        obj_logits: List[torch.Tensor] = []
        bboxs: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []

        for i, feat in enumerate(x):
            out = self.level_heads[i](feat)
            preds.append(out["pred"])
            obj_logits.append(out["obj_logits"])
            bboxs.append(out["bbox"])
            if self.return_features:
                feats.append(out["feat"])

        outputs: Dict[str, List[torch.Tensor]] = {
            "preds": preds,
            "obj_logits": obj_logits,
            "bbox": bboxs,
        }
        if self.return_features:
            outputs["features"] = feats

        if decode:
            if img_size is None or len(img_size) != 2:
                raise ValueError("img_size=[H_img, W_img] is required when decode=True")
            h_img, w_img = int(img_size[0]), int(img_size[1])

            decoded: List[torch.Tensor] = []
            for pred, stride in zip(preds, self.strides):
                b, _, hk, wk = pred.shape
                obj = torch.sigmoid(pred[:, :1])     # [B, 1, Hk, Wk]
                xy = torch.sigmoid(pred[:, 1:3])     # [B, 2, Hk, Wk] cell offsets in [0,1]
                wh = F.softplus(pred[:, 3:5])        # [B, 2, Hk, Wk] positive sizes

                grid = self._make_grid(hk, wk, pred.device)   # [2, Hk, Wk]
                xy_pix = (grid.unsqueeze(0) + xy) * float(stride)  # [B, 2, Hk, Wk]
                wh_pix = wh * float(stride)                      # [B, 2, Hk, Wk]

                x_c = xy_pix[:, 0] / w_img
                y_c = xy_pix[:, 1] / h_img
                w_n = wh_pix[:, 0] / w_img
                h_n = wh_pix[:, 1] / h_img

                # [B, 5, Hk, Wk] → (x, y, w, h, conf) with conf=sigmoid(obj)
                decoded.append(torch.stack([x_c, y_c, w_n, h_n, obj.squeeze(1)], dim=1))

            outputs["decoded"] = decoded

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

    config: Dict[str, Any] = {
        "in_channels": in_channels,
        "strides": strides or [8, 16, 32],
        "head_channels": kwargs.get("head_channels", None),
        "return_features": kwargs.get("return_features", True),
        "refine_features": kwargs.get("refine_features", False),
        "obj_prior": kwargs.get("obj_prior", 0.01),
    }
    return YoloOneDetectionHead(config)
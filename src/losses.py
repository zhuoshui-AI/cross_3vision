"""Loss / label-format helpers.

The DETR loss (Hungarian matcher + CE + L1 + GIoU) and auxiliary losses are
reused verbatim from HuggingFace — we do NOT reimplement them. The only thing
this module provides is:

- `build_detr_config(cfg)` re-exported from `src.model` for callers that import
  from `src.losses` (kept here so the loss/config surface lives in one place).
- box-format converters used by `eval.py` / `inference.py` to translate between
  the model's cxcywh predictions and the xyxy_pixel format torchmetrics expects
  / the spec's output format.
"""
from .model import build_detr_config  # noqa: F401  (re-export)


def box_cxcywh_to_xyxy(boxes):
    """(..., 4) cxcywh → xyxy. Works for tensors and numpy arrays."""
    import torch
    if isinstance(boxes, torch.Tensor):
        cx, cy, w, h = boxes.unbind(-1)
        out = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        return out
    # numpy path
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    import numpy as np
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def xyxy_to_cxcywh(boxes):
    import torch
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
    import numpy as np
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)


def cxcywh_to_xyxy_pixel(boxes_cxcywh_norm, img_size):
    """cxcywh normalized → xyxy pixel coords for torchmetrics / spec output."""
    import torch
    H, W = img_size
    xyxy = box_cxcywh_to_xyxy(boxes_cxcywh_norm)
    if isinstance(xyxy, torch.Tensor):
        scale = torch.tensor([W, H, W, H], device=xyxy.device, dtype=xyxy.dtype)
        return (xyxy * scale).round()
    import numpy as np
    scale = np.array([W, H, W, H], dtype=xyxy.dtype)
    return (xyxy * scale).round()

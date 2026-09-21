"""Diagnose mAP~0: T1 loss breakdown at init, T2 oracle-loss (pred=GT) test,
T3 GT box stats + init prediction stats. Runs on CPU with example_data only.

Usage:  python -m scripts.diag_dataflow  (from lab/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types as _types
import numpy as np
import torch
import cv2
from PIL import Image

# Stub albumentations so src.data imports without the package (we never call
# transforms here; only the static normalizers and pure helpers are used).
if _types.SimpleNamespace:
    try:
        import albumentations  # noqa: F401
    except ModuleNotFoundError:
        _alb = _types.ModuleType("albumentations")
        _alb.BboxParams = object
        sys.modules["albumentations"] = _alb

from src.data import MultiModalDataset, collate_fn, load_label, find_image
from src.model import build_model
from src.utils import load_yaml_config
from src.losses import cxcywh_to_xyxy_pixel

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_data")
CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")


def make_batch(cfg, stems, img_size=512):
    """Hand-rolled loader: same normalization as data.py, NO augmentation."""
    rgb_b, ir_b, dep_b, dmask_b, pmask_b, labels = [], [], [], [], [], []
    for stem in stems:
        rgb = np.array(Image.open(find_image(stem, os.path.join(ROOT, "visible"))).convert("RGB"))
        ir3 = np.array(Image.open(find_image(stem, os.path.join(ROOT, "infrared"))).convert("RGB"))
        ir = ir3[..., :1]
        dimg = Image.open(find_image(stem, os.path.join(ROOT, "depth")))
        if dimg.mode in ("I;16", "I"):
            depth = np.array(dimg).astype(np.float32)
            depth = depth[..., None] if depth.ndim == 2 else depth[..., :1]
            is_mm = True
        else:
            if dimg.mode != "L":
                dimg = dimg.convert("L")
            depth = np.array(dimg).astype(np.float32)
            depth = depth[..., None] if depth.ndim == 2 else depth[..., :1]
            is_mm = False
        H, W = rgb.shape[:2]
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        ir = cv2.resize(ir, (img_size, img_size), interpolation=cv2.INTER_LINEAR)[..., None]
        depth = cv2.resize(depth, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        depth = depth[..., None] if depth.ndim == 2 else depth[..., :1]

        boxes, cls = load_label(os.path.join(ROOT, "labels", stem + ".txt"), 12)
        # normalized cxcywh targets are scale-invariant; keep them as-is
        px_rgb = MultiModalDataset._normalize_rgb(rgb)
        px_ir = MultiModalDataset._normalize_ir(ir)
        px_dep, dmask = MultiModalDataset._normalize_depth(depth, is_mm)
        rgb_b.append(px_rgb); ir_b.append(px_ir); dep_b.append(px_dep); dmask_b.append(dmask)
        pmask_b.append(torch.ones(img_size, img_size, dtype=torch.bool))
        labels.append({"class_labels": torch.from_numpy(cls).long(),
                       "boxes": torch.from_numpy(boxes.astype(np.float32))})
    return {
        "pixel_values_rgb": torch.stack(rgb_b),
        "pixel_values_ir": torch.stack(ir_b),
        "pixel_values_depth": torch.stack(dep_b),
        "depth_mask": torch.stack(dmask_b),
        "pixel_mask": torch.stack(pmask_b),
        "labels": labels,
    }


def oracle_loss(model, batch, device):
    """Feed pred_boxes = GT boxes and one-hot GT logits; loss should be ~0."""
    labels = [{"class_labels": t["class_labels"].to(device), "boxes": t["boxes"].to(device)}
              for t in batch["labels"]]
    B, Q, C = len(labels), 100, 12
    logits = torch.zeros(B, Q, C + 1, device=device)
    boxes = torch.full((B, Q, 4), 0.5, device=device)
    for b, tgt in enumerate(labels):
        n = tgt["boxes"].shape[0]
        for i in range(n):
            logits[b, i, int(tgt["class_labels"][i])] = 10.0
            boxes[b, i] = tgt["boxes"][i]
        logits[b, n:, C] = 10.0  # no-object
    # aux_loss=True requires real aux tensors; repeat the oracle across 6 layers
    outputs_class = logits.unsqueeze(0).repeat(6, 1, 1, 1)
    outputs_coord = boxes.unsqueeze(0).repeat(6, 1, 1, 1)
    loss, loss_dict, _ = model.loss_function(
        logits, labels, device, boxes, model.config, outputs_class, outputs_coord)
    return loss, loss_dict


def main():
    torch.manual_seed(0)
    device = "cpu"
    cfg = load_yaml_config(CFG_PATH)
    cfg["model"]["swin_pretrained"] = False  # random init for T1 clarity

    stems = sorted({os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "labels"))})
    stems = [s for s in stems if os.path.exists(os.path.join(ROOT, "visible", s + ".png"))
             or os.path.exists(os.path.join(ROOT, "visible", s + ".jpg"))]

    # ---------- T3: GT stats ----------
    all_boxes = np.concatenate(
        [load_label(os.path.join(ROOT, "labels", s + ".txt"), 12)[0] for s in stems], axis=0)
    print(f"[T3] {len(stems)} images, {len(all_boxes)} GT boxes")
    wh = all_boxes[:, 2:4] - all_boxes[:, 0:2]  # xyxy -> wh
    print(f"     wh  mean={wh.mean(0).round(3)} min={wh.min(0).round(4)} max={wh.max(0).round(3)}")
    px = torch.from_numpy(all_boxes) * 512.0  # xyxy norm -> 512px xyxy
    wh_px = (px[:, 2:] - px[:, :2])
    print(f"     at 512px: mean wh = {[round(v,1) for v in wh_px.mean(0).tolist()]}, "
          f"<16px boxes: {(wh_px.min(1).values < 16).sum().item()}/{len(px)}")

    # ---------- build model ----------
    print("building model (random init)...")
    model = build_model(cfg).to(device).eval()

    # ---------- T1: init loss breakdown on real data ----------
    batch = make_batch(cfg, stems[:3])
    with torch.no_grad():
        out = model(**batch)
    print("\n[T1] init loss_dict (batch=3 real samples, random init):")
    for k, v in sorted(out.loss_dict.items()):
        print(f"     {k:24s} {float(v):8.4f}")
    print(f"     {'TOTAL':24s} {float(out.loss):8.4f}")

    # init prediction stats: are scores above conf 0.05? box spread?
    probs = out.logits.softmax(-1)
    s_nonobj, c_nonobj = probs[..., :-1].max(-1)
    print(f"     max non-eos prob: median={s_nonobj.median():.4f} "
          f"p95={s_nonobj.flatten().kthvalue(int(s_nonobj.numel()*0.95)).values:.4f}")
    print(f"     pred box wh: mean={out.pred_boxes[..., 2:].mean(0).mean():.4f} "
          f"std={out.pred_boxes[..., 2:].std():.4f}")

    # ---------- T2: oracle loss ----------
    loss_o, dict_o = oracle_loss(model, batch, device)
    print("\n[T2] ORACLE loss (pred_boxes=GT, one-hot logits) — expect ~0:")
    for k, v in sorted(dict_o.items()):
        print(f"     {k:24s} {float(v):8.4f}")
    print(f"     {'ORACLE TOTAL':24s} {float(loss_o):8.4f}")
    if float(loss_o) > 0.05:
        print("     >>> ORACLE LOSS NOT ZERO => supervision<->prediction format mismatch!")
    else:
        print("     >>> oracle ~0: labels<->loss format chain is CONSISTENT.")


if __name__ == "__main__":
    main()

"""Comprehensive validation visualization tool (Ultralytics-style).

Generates:
  1. Training curves (train/val loss, mAP, LR over epochs)
  2. GT data distribution (class counts, box size histogram)
  3. Per-class TP/FP/FN bar chart + precision/recall/F1 table
  4. Per-class PR curves @ IoU 0.5
  5. Confusion matrix (predicted vs true class)
  6. Box-size distribution (small/medium/large)
  7. Error case visualization (GT green / TP blue / FP red / FN yellow on worst images)
  8. HTML report combining everything

Usage (on server with data + checkpoint):
    python -m scripts.visualize --config configs/default.yaml \
        --ckpt outputs/ckpt/best.pth --data-root /path/to/AIC2026_Train_2000

    # Only parse training log (no GPU/data needed):
    python -m scripts.visualize --config configs/default.yaml --log-only

    # Use a specific log file (default: outputs/logs/log.txt):
    python -m scripts.visualize --log-file outputs/logs/log.txt --log-only

Output directory: outputs/vis/
"""
import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

# --- Try to use a CJK-capable font so Chinese labels don't render as boxes ---
def _setup_cjk_font():
    candidates = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "PingFang SC", "Heiti SC", "STHeiti",
        "Source Han Sans SC", "Source Han Sans CN", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    # No CJK font found; switch labels to English to avoid tofu boxes
    global _USE_ENGLISH
    _USE_ENGLISH = True

_USE_ENGLISH = False
_setup_cjk_font()

# -----------------------------------------------------------------------
#  Config / label helpers
# -----------------------------------------------------------------------

def _load_cfg(config_path):
    from src.utils import load_yaml_config
    return load_yaml_config(config_path)


def _class_names(cfg):
    id2label = cfg.get("id2label", {})
    n = int(cfg["model"]["num_labels"])
    names = []
    for i in range(n):
        names.append(id2label.get(str(i), id2label.get(i, f"cls{i}")))
    return names


# Color palette for 12 classes (tab20 subset, high contrast)
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324",
    "#800000", "#808000",
]


def _color(idx):
    return _PALETTE[idx % len(_PALETTE)]


# -----------------------------------------------------------------------
#  1. Training log parsing + curves
# -----------------------------------------------------------------------

def parse_log(log_path):
    """Parse DETR-style JSONL log (one JSON per line) + human-readable lines.

    Returns dict with keys:
        epochs: list[int]
        train_loss: list[float]
        test_loss: list[float]
        map_5095: list[float]   (NaN where not evaluated)
        map_50: list[float]
        lr: list[float]
        coco12: list[list[float]]  (per-epoch 12-item COCO eval)
        loss_components: dict[str, list[float]]  (train_loss_ce etc.)
    """
    records = []
    if not os.path.isfile(log_path):
        # Try plain-text log fallback
        return _parse_text_log(log_path)

    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Skip human-readable [INFO] lines
            if line.startswith("[") or "INFO" in line[:20]:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue

    if not records:
        return _parse_text_log(log_path)

    epochs = sorted(set(r.get("epoch", -1) for r in records if "epoch" in r))
    by_epoch = {}
    for r in records:
        ep = r.get("epoch", -1)
        if ep not in by_epoch:
            by_epoch[ep] = r

    train_loss, test_loss, lr_list = [], [], []
    map_5095, map_50 = [], []
    coco12_list = []

    for ep in epochs:
        r = by_epoch[ep]
        train_loss.append(r.get("train_loss", float("nan")))
        test_loss.append(r.get("test_loss", float("nan")))
        lr_list.append(r.get("train_lr", float("nan")))
        coco12 = r.get("test_coco_eval_bbox")
        if coco12 and len(coco12) >= 2:
            map_5095.append(coco12[0] if coco12[0] is not None else float("nan"))
            map_50.append(coco12[1] if coco12[1] is not None else float("nan"))
        else:
            map_5095.append(float("nan"))
            map_50.append(float("nan"))
        coco12_list.append(coco12 or [])

    # Loss components
    comp_keys = set()
    for r in records:
        for k in r:
            if k.startswith(("train_loss_", "test_loss_")) and not k.endswith("_unscaled"):
                comp_keys.add(k)
    components = {k: [] for k in comp_keys}
    for ep in epochs:
        r = by_epoch[ep]
        for k in comp_keys:
            components[k].append(r.get(k, float("nan")))

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "map_5095": map_5095,
        "map_50": map_50,
        "lr": lr_list,
        "coco12": coco12_list,
        "loss_components": components,
    }


def _parse_text_log(log_path):
    """Fallback parser for human-readable log (regex-based).

    Extracts:
      Epoch N mean loss = X          → train_loss
      Val epN: mAP@50-95=X mAP@50=Y  → map values
      epN it0/... lr=X               → lr (first per epoch)
    """
    import re
    epochs_set = set()
    train_loss_map, val_map5095_map, val_map50_map, lr_map = {}, {}, {}, {}

    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            # Epoch N mean loss = X
            m = re.search(r"Epoch\s+(\d+)\s+mean loss\s*=\s*([\d.]+)", line)
            if m:
                ep, loss = int(m.group(1)), float(m.group(2))
                train_loss_map[ep] = loss
                epochs_set.add(ep)
                continue
            # Val epN: mAP@50-95=X mAP@50=Y
            m = re.search(r"Val\s+ep(\d+):\s+mAP@50-95=([\d.]+)\s+mAP@50=([\d.]+)", line)
            if m:
                ep = int(m.group(1))
                val_map5095_map[ep] = float(m.group(2))
                val_map50_map[ep] = float(m.group(3))
                epochs_set.add(ep)
                continue
            # epN it0/... lr=X  (capture first LR per epoch)
            m = re.search(r"ep(\d+)\s+it0/.*\s+lr=([\d.eE+-]+)", line)
            if m:
                ep = int(m.group(1))
                if ep not in lr_map:
                    lr_map[ep] = float(m.group(2))
                epochs_set.add(ep)

    epochs = sorted(epochs_set)
    train_loss = [train_loss_map.get(ep, float("nan")) for ep in epochs]
    map_5095 = [val_map5095_map.get(ep, float("nan")) for ep in epochs]
    map_50 = [val_map50_map.get(ep, float("nan")) for ep in epochs]
    lr = [lr_map.get(ep, float("nan")) for ep in epochs]

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "test_loss": [float("nan")] * len(epochs),
        "map_5095": map_5095,
        "map_50": map_50,
        "lr": lr,
        "coco12": [],
        "loss_components": {},
    }


def plot_training_curves(log_data, out_dir, class_names):
    """Generate training curve plots (loss, mAP, LR, loss components)."""
    epochs = log_data["epochs"]
    n = len(epochs)
    if n == 0:
        print("  [WARN] No epoch data found in log, skipping training curves.")
        return []

    saved = []
    E = "Epoch" if _USE_ENGLISH else "Epoch"

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    tl = log_data["train_loss"]
    vl = log_data["test_loss"]
    has_train = any(not np.isnan(x) for x in tl)
    has_val = any(not np.isnan(x) for x in vl)
    if has_train:
        ax.plot(epochs, tl, color="#2196F3", linewidth=1.5, label="train loss", zorder=3)
    if has_val:
        ax.plot(epochs, vl, color="#FF5722", linewidth=1.5, label="val loss", zorder=3)
    ax.set_xlabel(E)
    ax.set_ylabel("Loss")
    title = "Training / Validation Loss" if _USE_ENGLISH else "训练 / 验证 损失曲线"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "curve_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # --- mAP curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    m5095 = log_data["map_5095"]
    m50 = log_data["map_50"]
    has_5095 = any(not np.isnan(x) for x in m5095)
    has_50 = any(not np.isnan(x) for x in m50)
    if has_5095:
        valid = [(e, v) for e, v in zip(epochs, m5095) if not np.isnan(v)]
        if valid:
            ve, vv = zip(*valid)
            ax.plot(ve, vv, "o-", color="#4CAF50", linewidth=1.5,
                    markersize=4, label="mAP@50-95", zorder=3)
    if has_50:
        valid = [(e, v) for e, v in zip(epochs, m50) if not np.isnan(v)]
        if valid:
            ve, vv = zip(*valid)
            ax.plot(ve, vv, "s-", color="#FF9800", linewidth=1.5,
                    markersize=4, label="mAP@50", zorder=3)
    ax.set_xlabel(E)
    ax.set_ylabel("mAP")
    title = "Validation mAP" if _USE_ENGLISH else "验证集 mAP 曲线"
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    if has_5095:
        vals = [v for v in m5095 if not np.isnan(v)]
        if vals:
            ax.set_ylim(bottom=max(-0.01, min(vals) - 0.02),
                        top=max(vals) * 1.15 + 0.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "curve_map.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # --- LR curve ---
    fig, ax = plt.subplots(figsize=(10, 4))
    lr = log_data["lr"]
    has_lr = any(not np.isnan(x) for x in lr)
    if has_lr:
        ax.plot(epochs, lr, color="#9C27B0", linewidth=1.5, label="lr", zorder=3)
    ax.set_xlabel(E)
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule" if _USE_ENGLISH else "学习率调度曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))
    fig.tight_layout()
    path = os.path.join(out_dir, "curve_lr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # --- Loss components (if available from JSONL) ---
    comps = log_data.get("loss_components", {})
    if comps:
        fig, ax = plt.subplots(figsize=(12, 6))
        for key, vals in sorted(comps.items()):
            if any(not np.isnan(v) for v in vals):
                ax.plot(epochs, vals, linewidth=1, label=key, alpha=0.8)
        ax.set_xlabel(E)
        ax.set_ylabel("Loss Component")
        ax.set_title("Loss Components" if _USE_ENGLISH else "损失分量曲线")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, "curve_loss_components.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # --- COCO 12-item bar chart (latest epoch) ---
    coco12_list = log_data.get("coco12", [])
    if coco12_list:
        last = None
        for c in reversed(coco12_list):
            if c and len(c) == 12:
                last = c
                break
        if last:
            labels12 = ["mAP", "mAP@50", "mAP@75",
                       "mAP_s", "mAP_m", "mAP_l",
                       "AR@1", "AR@10", "AR@100",
                       "AR_s", "AR_m", "AR_l"]
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(labels12))
            bars = ax.bar(x, last, color=[_color(i) for i in range(12)],
                          edgecolor="white", width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels12, rotation=30, ha="right")
            ax.set_ylabel("Score")
            ax.set_title("COCO Metrics (Best Epoch)" if _USE_ENGLISH
                         else "COCO 12 项指标 (最优 epoch)")
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, last):
                if val is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            fig.tight_layout()
            path = os.path.join(out_dir, "coco12_bar.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

    return saved


# -----------------------------------------------------------------------
#  2. Validation run — collect per-image predictions
# -----------------------------------------------------------------------

def run_validation(cfg, ckpt_path, device=None):
    """Run model on validation set, return per-image results.

    Returns list of dicts:
        {
            "stem": str,
            "gt_boxes": np.ndarray (N, 4) xyxy pixel,
            "gt_labels": np.ndarray (N,) int,
            "pred_boxes": np.ndarray (M, 4) xyxy pixel,
            "pred_scores": np.ndarray (M,),
            "pred_labels": np.ndarray (M,) int,
            "image_size": (H, W),
        }
    """
    import torch
    from torch.utils.data import DataLoader
    from src.data import MultiModalDataset, collate_fn
    from src.model import build_model
    from src.losses import cxcywh_to_xyxy_pixel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build val dataset (same logic as train.py)
    data_root = cfg["data"].get("data_root")
    train_split = cfg["data"].get("train_split")
    val_split = cfg["data"].get("val_split")
    val_ratio = float(cfg["data"].get("val_ratio", 0.1))
    img_size = int(cfg["data"]["img_size"])
    ncls = int(cfg["model"]["num_labels"])

    if val_split and os.path.isabs(val_split) and os.path.isdir(val_split):
        ds = MultiModalDataset(val_split, img_size=img_size, train=False,
                               num_classes=ncls)
    else:
        # Split from train set
        if train_split and os.path.isabs(train_split) and os.path.isdir(train_split):
            data_root = train_split
        full_ds = MultiModalDataset(data_root, img_size=img_size, train=False,
                                    num_classes=ncls)
        n_total = len(full_ds)
        n_val = max(1, int(n_total * val_ratio))
        # Deterministic split (same seed as training)
        import random
        rng = random.Random(42)
        indices = list(range(n_total))
        rng.shuffle(indices)
        val_indices = sorted(indices[:n_val])
        ds = torch.utils.data.Subset(full_ds, val_indices)

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=int(cfg["data"].get("num_workers", 4)),
                        collate_fn=collate_fn,
                        pin_memory=(device == "cuda"))

    model = build_model(cfg).to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  [WARN] No checkpoint found at {ckpt_path}, using random weights!")
    model.eval()

    conf_th = 0.0  # keep all predictions for PR curves
    results = []
    with torch.no_grad():
        for batch in loader:
            rgb = batch["pixel_values_rgb"].to(device)
            ir = batch["pixel_values_ir"].to(device)
            depth = batch["pixel_values_depth"].to(device)
            depth_mask = batch["depth_mask"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            img_size_hw = (rgb.shape[2], rgb.shape[3])

            out = model(pixel_values_rgb=rgb, pixel_values_ir=ir,
                        pixel_values_depth=depth, depth_mask=depth_mask,
                        pixel_mask=pixel_mask, labels=None)
            probs = out.logits.softmax(-1)
            scores, labels_pred = probs[..., :-1].max(-1)
            boxes_cxcywh = out.pred_boxes

            b = 0  # batch_size=1
            keep = scores[b] >= conf_th
            pred_boxes = cxcywh_to_xyxy_pixel(
                boxes_cxcywh[b][keep], img_size_hw).cpu().numpy()
            pred_scores = scores[b][keep].cpu().numpy()
            pred_labels = labels_pred[b][keep].cpu().numpy()

            # GT
            tgt = batch["labels"][b]
            gt_boxes = cxcywh_to_xyxy_pixel(
                tgt["boxes"], img_size_hw).cpu().numpy()
            gt_labels = tgt["class_labels"].cpu().numpy()

            results.append({
                "stem": batch["stems"][b],
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels,
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
                "pred_labels": pred_labels,
                "image_size": img_size_hw,
            })

    return results, ds


# -----------------------------------------------------------------------
#  3. Data distribution charts
# -----------------------------------------------------------------------

def plot_gt_distribution(results, class_names, out_dir):
    """GT class distribution + box size histogram."""
    n_cls = len(class_names)
    gt_counts = np.zeros(n_cls, dtype=int)
    gt_areas = []  # normalized areas (w*h / (H*W))
    gt_area_by_cls = defaultdict(list)

    for r in results:
        H, W = r["image_size"]
        for label in r["gt_labels"]:
            gt_counts[int(label)] += 1
        for box, label in zip(r["gt_boxes"], r["gt_labels"]):
            x1, y1, x2, y2 = box
            area = ((x2 - x1) * (y2 - y1)) / (W * H)
            gt_areas.append(area)
            gt_area_by_cls[int(label)].append(area)

    saved = []

    # --- Bar chart: GT class counts ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_cls)
    bars = ax.bar(x, gt_counts, color=[_color(i) for i in range(n_cls)],
                  edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("GT Class Distribution" if _USE_ENGLISH else "标注 类别分布")
    ax.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, gt_counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(cnt), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "gt_class_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # --- Histogram: GT box size distribution (COCO area bands) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    if gt_areas:
        areas = np.array(gt_areas)
        # COCO: small < 32² / img² ≈ 32²/512² ≈ 0.0039
        # medium 32²–96², large > 96²
        # But our boxes are normalized to img area, so:
        small_th = (32 / 512) ** 2
        large_th = (96 / 512) ** 2
        small = np.sum(areas < small_th)
        medium = np.sum((areas >= small_th) & (areas < large_th))
        large = np.sum(areas >= large_th)
        cats = ["small\n(<32px²)", "medium\n(32-96px²)", "large\n(>96px²)"]
        counts = [small, medium, large]
        colors = ["#f44336", "#FF9800", "#4CAF50"]
        bars = ax.bar(cats, counts, color=colors, edgecolor="white", width=0.5)
        ax.set_ylabel("Count")
        ax.set_title("GT Box Size Distribution (COCO)" if _USE_ENGLISH
                     else "标注 框大小分布 (COCO 标准)")
        ax.grid(axis="y", alpha=0.3)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(cnt), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "gt_box_size.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # --- Per-class box area violin/box plot ---
    fig, ax = plt.subplots(figsize=(12, 5))
    data_to_plot = []
    for i in range(n_cls):
        vals = gt_area_by_cls.get(i, [])
        if len(vals) > 0:
            data_to_plot.append(vals)
        else:
            data_to_plot.append([0])
    bp = ax.boxplot(data_to_plot, positions=range(n_cls),
                    widths=0.5, patch_artist=True, showfliers=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_color(i))
        patch.set_alpha(0.6)
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Normalized Area (w×h / img)")
    ax.set_yscale("log")
    ax.set_title("GT Box Area by Class" if _USE_ENGLISH else "各类别 框面积分布")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "gt_box_area_by_cls.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    return saved, gt_counts


# -----------------------------------------------------------------------
#  4. Per-class TP/FP/FN + PR curves
# -----------------------------------------------------------------------

def _iou_matrix(boxes1, boxes2):
    """Compute IoU between all pairs. boxes: (N,4) xyxy, (M,4) xyxy → (N,M)."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # (N,M,2)
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N,M,2)
    wh = np.clip(rb - lt, 0, None)  # (N,M,2)
    inter = wh[..., 0] * wh[..., 1]  # (N,M)
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_tp_fp_fn(results, iou_th=0.5, n_cls=12):
    """Match predictions to GT per-class using greedy IoU matching.

    Returns:
        per_class: dict keyed by class id with:
            tp, fp, fn: int counts
            precision, recall, f1: float
            scores: list of scores for TP and FP (for PR curve)
            tp_scores, fp_scores: lists
    """
    per_class = {i: {"tp": 0, "fp": 0, "fn": 0,
                     "tp_scores": [], "fp_scores": [],
                     "all_scores": [], "all_tp": []}
                 for i in range(n_cls)}

    for r in results:
        gt_boxes = r["gt_boxes"]
        gt_labels = r["gt_labels"]
        pred_boxes = r["pred_boxes"]
        pred_scores = r["pred_scores"]
        pred_labels = r["pred_labels"]

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        # Sort predictions by score descending (greedy matching)
        if len(pred_boxes) > 0:
            order = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_labels = pred_labels[order]

        for pi in range(len(pred_boxes)):
            pbox = pred_boxes[pi:pi+1]
            pscore = pred_scores[pi]
            plabel = int(pred_labels[pi])

            # Find best unmatched GT of the same class
            best_iou = 0
            best_gi = -1
            for gi in range(len(gt_boxes)):
                if gt_matched[gi]:
                    continue
                if int(gt_labels[gi]) != plabel:
                    continue
                iou = _iou_matrix(pbox, gt_boxes[gi:gi+1])[0, 0]
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_th and best_gi >= 0:
                gt_matched[best_gi] = True
                per_class[plabel]["tp"] += 1
                per_class[plabel]["tp_scores"].append(float(pscore))
                per_class[plabel]["all_scores"].append(float(pscore))
                per_class[plabel]["all_tp"].append(1)
            else:
                per_class[plabel]["fp"] += 1
                per_class[plabel]["fp_scores"].append(float(pscore))
                per_class[plabel]["all_scores"].append(float(pscore))
                per_class[plabel]["all_tp"].append(0)

        # FN = unmatched GT
        for gi in range(len(gt_boxes)):
            if not gt_matched[gi]:
                per_class[int(gt_labels[gi])]["fn"] += 1

    # Compute precision/recall/f1
    for i in range(n_cls):
        c = per_class[i]
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        c["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        c["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        c["f1"] = 2 * c["precision"] * c["recall"] / (
            c["precision"] + c["recall"]) if (c["precision"] + c["recall"]) > 0 else 0.0

    return per_class


def plot_tp_fp_fn(per_class, class_names, out_dir):
    """Grouped bar chart: TP/FP/FN per class."""
    n = len(class_names)
    tp = [per_class[i]["tp"] for i in range(n)]
    fp = [per_class[i]["fp"] for i in range(n)]
    fn = [per_class[i]["fn"] for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n)
    w = 0.25
    b1 = ax.bar(x - w, tp, w, label="TP", color="#4CAF50", edgecolor="white")
    b2 = ax.bar(x, fp, w, label="FP", color="#f44336", edgecolor="white")
    b3 = ax.bar(x + w, fn, w, label="FN", color="#FF9800", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Per-Class TP / FP / FN @ IoU 0.5" if _USE_ENGLISH
                 else "各类别 TP / FP / FN (IoU=0.5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        str(int(h)), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "tp_fp_fn.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pr_curves(per_class, class_names, out_dir):
    """Per-class Precision-Recall curves at IoU 0.5."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    has_any = False
    for i in range(n):
        c = per_class[i]
        scores = np.array(c["all_scores"])
        tps = np.array(c["all_tp"])
        n_gt = c["tp"] + c["fn"]
        if n_gt == 0 or len(scores) == 0:
            continue
        # Sort by score desc
        order = np.argsort(-scores)
        scores = scores[order]
        tps = tps[order]
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(1 - tps)
        precision = cum_tp / (cum_tp + cum_fp + 1e-12)
        recall = cum_tp / n_gt
        # Prepend (1, 0) and append (0, final_recall) for nice curve
        precision = np.concatenate([[1.0], precision])
        recall = np.concatenate([[0.0], recall])
        ax.plot(recall, precision, linewidth=1.5, color=_color(i),
                label=f"{class_names[i]} ({c['precision']:.2f}/{c['recall']:.2f})")
        has_any = True

    if has_any:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Per-Class PR Curves @ IoU 0.5" if _USE_ENGLISH
                     else "各类别 PR 曲线 (IoU=0.5)")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No detections to plot", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "pr_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_precision_recall_f1(per_class, class_names, out_dir):
    """Bar chart: precision/recall/F1 per class."""
    n = len(class_names)
    prec = [per_class[i]["precision"] for i in range(n)]
    rec = [per_class[i]["recall"] for i in range(n)]
    f1 = [per_class[i]["f1"] for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, prec, w, label="Precision", color="#2196F3", edgecolor="white")
    ax.bar(x, rec, w, label="Recall", color="#4CAF50", edgecolor="white")
    ax.bar(x + w, f1, w, label="F1", color="#FF9800", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class Precision / Recall / F1 @ IoU 0.5" if _USE_ENGLISH
                 else "各类别 精度 / 召回 / F1 (IoU=0.5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "precision_recall_f1.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
#  5. Confusion matrix
# -----------------------------------------------------------------------

def plot_confusion_matrix(results, class_names, out_dir, iou_th=0.5):
    """12×12 (or 13×13 with background) confusion matrix.

    Rows = true class, Columns = predicted class.
    Last row/col = background (FN goes to background col, FP from background row).
    """
    n = len(class_names)
    cm = np.zeros((n + 1, n + 1), dtype=int)  # +1 for background

    for r in results:
        gt_boxes = r["gt_boxes"]
        gt_labels = r["gt_labels"]
        pred_boxes = r["pred_boxes"]
        pred_labels = r["pred_labels"]
        pred_scores = r["pred_scores"]

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.zeros(len(pred_boxes), dtype=bool)

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            order = np.argsort(-pred_scores)
            for pi in order:
                pbox = pred_boxes[pi:pi+1]
                plabel = int(pred_labels[pi])
                best_iou, best_gi = 0, -1
                for gi in range(len(gt_boxes)):
                    if gt_matched[gi]:
                        continue
                    iou = _iou_matrix(pbox, gt_boxes[gi:gi+1])[0, 0]
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_iou >= iou_th and best_gi >= 0:
                    gt_matched[best_gi] = True
                    pred_matched[pi] = True
                    tl = int(gt_labels[best_gi])
                    cm[tl, plabel] += 1
                else:
                    cm[n, plabel] += 1  # FP → background row
            for gi in range(len(gt_boxes)):
                if not gt_matched[gi]:
                    tl = int(gt_labels[gi])
                    cm[tl, n] += 1  # FN → background col
        elif len(gt_boxes) > 0:
            for gi in range(len(gt_boxes)):
                tl = int(gt_labels[gi])
                cm[tl, n] += 1
        elif len(pred_boxes) > 0:
            for pi in range(len(pred_boxes)):
                plabel = int(pred_labels[pi])
                cm[n, plabel] += 1

    labels = list(class_names) + ["bg"]
    fig, ax = plt.subplots(figsize=(10, 9))
    # Normalize by row (per true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized)" if _USE_ENGLISH
                 else "混淆矩阵 (行归一化)")
    # Annotate cells
    for i in range(n + 1):
        for j in range(n + 1):
            val = cm[i, j]
            if val > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        color=color, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
#  6. Error case visualization (draw bboxes on worst images)
# -----------------------------------------------------------------------

def _find_image_file(stem, data_root):
    """Locate the RGB image for a stem in the visible/ directory."""
    vis_dir = os.path.join(data_root, "visible")
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(vis_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def draw_error_cases(results, class_names, out_dir, data_root,
                     top_k=12, iou_th=0.5):
    """Draw bounding boxes on images with the most errors (FP + FN).

    Colors: GT = green, TP = blue, FP = red, FN = yellow.
    Saves individual images + a grid summary.
    """
    import cv2

    n_cls = len(class_names)
    # Score each image by (FP + FN) count
    scored = []
    for idx, r in enumerate(results):
        gt_boxes = r["gt_boxes"]
        gt_labels = r["gt_labels"]
        pred_boxes = r["pred_boxes"]
        pred_labels = r["pred_labels"]
        pred_scores = r["pred_scores"]

        n_tp, n_fp, n_fn = 0, 0, 0
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        if len(pred_boxes) > 0:
            order = np.argsort(-pred_scores)
            for pi in order:
                pbox = pred_boxes[pi:pi+1]
                plabel = int(pred_labels[pi])
                best_iou, best_gi = 0, -1
                for gi in range(len(gt_boxes)):
                    if gt_matched[gi]:
                        continue
                    if int(gt_labels[gi]) != plabel:
                        continue
                    iou = _iou_matrix(pbox, gt_boxes[gi:gi+1])[0, 0]
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_iou >= iou_th and best_gi >= 0:
                    gt_matched[best_gi] = True
                    n_tp += 1
                else:
                    n_fp += 1
            n_fn = int(np.sum(~gt_matched))
        else:
            n_fn = len(gt_boxes)

        error_count = n_fp + n_fn
        scored.append((error_count, n_tp, n_fp, n_fn, idx))

    scored.sort(reverse=True, key=lambda x: x[0])
    worst = [s for s in scored if s[0] > 0][:top_k]

    if not worst:
        print("  No error cases found (all predictions correct or no GT).")
        return []

    saved = []
    err_dir = os.path.join(out_dir, "error_cases")
    os.makedirs(err_dir, exist_ok=True)

    for rank, (err, n_tp, n_fp, n_fn, idx) in enumerate(worst):
        r = results[idx]
        img_path = _find_image_file(r["stem"], data_root)
        if img_path is None:
            print(f"  [WARN] Image not found for stem {r['stem']}, skipping.")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Failed to read {img_path}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to model input size for consistent box coords
        H, W = r["image_size"]
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H))

        gt_boxes = r["gt_boxes"]
        gt_labels = r["gt_labels"]
        pred_boxes = r["pred_boxes"]
        pred_labels = r["pred_labels"]
        pred_scores = r["pred_scores"]

        # Determine TP/FP/FN
        gt_status = ["fn"] * len(gt_boxes)  # default
        pred_status = ["fp"] * len(pred_boxes)

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            order = np.argsort(-pred_scores)
            for pi in order:
                pbox = pred_boxes[pi:pi+1]
                plabel = int(pred_labels[pi])
                best_iou, best_gi = 0, -1
                for gi in range(len(gt_boxes)):
                    if gt_status[gi] == "tp":
                        continue
                    if int(gt_labels[gi]) != plabel:
                        continue
                    iou = _iou_matrix(pbox, gt_boxes[gi:gi+1])[0, 0]
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_iou >= iou_th and best_gi >= 0:
                    gt_status[best_gi] = "tp"
                    pred_status[pi] = "tp"

        # Draw GT boxes (green=TP, yellow=FN)
        for gi in range(len(gt_boxes)):
            x1, y1, x2, y2 = gt_boxes[gi].astype(int)
            label = int(gt_labels[gi])
            if gt_status[gi] == "tp":
                color = (0, 200, 0)  # green
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"GT:{class_names[label]}", (x1, max(y1-5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                color = (255, 220, 0)  # yellow
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"FN:{class_names[label]}", (x1, max(y1-5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw pred boxes (blue=TP, red=FP)
        for pi in range(len(pred_boxes)):
            x1, y1, x2, y2 = pred_boxes[pi].astype(int)
            label = int(pred_labels[pi])
            score = pred_scores[pi]
            if pred_status[pi] == "tp":
                color = (0, 150, 255)  # blue
                tag = "TP"
            else:
                color = (255, 0, 0)  # red
                tag = "FP"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            txt = f"{tag}:{class_names[label]} {score:.2f}"
            cv2.putText(img, txt, (x1, min(y2+15, H-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Add header info
        hdr = f"#{rank+1} {r['stem']} | TP={n_tp} FP={n_fp} FN={n_fn} err={err}"
        cv2.putText(img, hdr, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)

        path = os.path.join(err_dir, f"error_{rank+1:02d}_{r['stem']}.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        ax.set_title(hdr, fontsize=8)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # --- Grid summary (up to 12 images) ---
    n_grid = min(len(saved), 12)
    cols = 3
    rows = (n_grid + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if rows == 1:
        axes = [axes] if cols > 1 else [[axes]]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i]
        if i < n_grid:
            img = plt.imread(saved[i])
            ax.imshow(img)
            ax.set_title(os.path.basename(saved[i]), fontsize=7)
        ax.axis("off")
    fig.suptitle("Worst Error Cases (FP+FN sorted)" if _USE_ENGLISH
                 else "错误最多的图片 (按 FP+FN 排序)", fontsize=12)
    fig.tight_layout()
    grid_path = os.path.join(out_dir, "error_cases_grid.png")
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)
    saved.append(grid_path)

    return saved


# -----------------------------------------------------------------------
#  7. Summary table as image
# -----------------------------------------------------------------------

def plot_summary_table(per_class, class_names, out_dir, gt_counts):
    """Render per-class metrics summary as a table image."""
    n = len(class_names)
    col_labels = ["Class", "GT", "TP", "FP", "FN", "Precision", "Recall", "F1"]
    rows = []
    for i in range(n):
        c = per_class[i]
        rows.append([
            class_names[i], gt_counts[i],
            c["tp"], c["fp"], c["fn"],
            f"{c['precision']:.3f}", f"{c['recall']:.3f}", f"{c['f1']:.3f}",
        ])
    # Totals
    total_tp = sum(per_class[i]["tp"] for i in range(n))
    total_fp = sum(per_class[i]["fp"] for i in range(n))
    total_fn = sum(per_class[i]["fn"] for i in range(n))
    total_gt = sum(gt_counts)
    total_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_p * total_r / (total_p + total_r) if (total_p + total_r) > 0 else 0
    rows.append(["TOTAL", total_gt, total_tp, total_fp, total_fn,
                 f"{total_p:.3f}", f"{total_r:.3f}", f"{total_f1:.3f}"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", weight="bold")
    # Color total row
    for j in range(len(col_labels)):
        table[len(rows), j].set_facecolor("#ECEFF1")
        table[len(rows), j].set_text_props(weight="bold")
    # Alternate row colors
    for i in range(1, len(rows)):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#F5F5F5")
    title = "各类别检测指标汇总 (IoU=0.5)" if not _USE_ENGLISH \
            else "Per-Class Detection Summary @ IoU 0.5"
    ax.set_title(title, fontsize=13, pad=15)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
#  8. HTML report
# -----------------------------------------------------------------------

def generate_html_report(out_dir, sections, log_data=None):
    """Generate a self-contained HTML report combining all visualizations."""
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>"]
    html.append("<style>")
    html.append("""
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }
    h2 { color: #283593; margin-top: 30px; border-left: 4px solid #3f51b5; padding-left: 10px; }
    .img-container { background: white; padding: 15px; border-radius: 8px;
                     box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: 15px 0; }
    .img-container img { max-width: 100%; display: block; margin: 0 auto; }
    .img-container .caption { text-align: center; color: #666; font-size: 13px; margin-top: 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 15px; }
    .stats { background: white; padding: 15px; border-radius: 8px; margin: 10px 0;
             box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .stats table { border-collapse: collapse; width: 100%; }
    .stats th, .stats td { border: 1px solid #ddd; padding: 8px; text-align: center; }
    .stats th { background: #37474F; color: white; }
    .stats tr:nth-child(even) { background: #f5f5f5; }
    """)
    html.append("</style></head><body>")
    html.append("<h1>Validation Report - Multimodal DETR</h1>")

    for section in sections:
        if section.get("type") == "text":
            html.append(f"<p>{section['content']}</p>")
        elif section.get("type") == "heading":
            html.append(f"<h2>{section['content']}</h2>")
        elif section.get("type") == "image":
            title = section.get("title", "")
            img_path = section["path"]
            rel = os.path.relpath(img_path, out_dir)
            html.append(f"<div class='img-container'>")
            if title:
                html.append(f"<div class='caption'><b>{title}</b></div>")
            html.append(f"<img src='{rel}' />")
            html.append("</div>")
        elif section.get("type") == "grid":
            html.append("<div class='grid'>")
            for item in section["items"]:
                rel = os.path.relpath(item["path"], out_dir)
                html.append(f"<div class='img-container'>")
                html.append(f"<div class='caption'>{item.get('title','')}</div>")
                html.append(f"<img src='{rel}' />")
                html.append("</div>")
            html.append("</div>")
        elif section.get("type") == "table":
            html.append("<div class='stats'>")
            html.append(section["content"])
            html.append("</div>")

    # Log summary stats
    if log_data and log_data.get("epochs"):
        eps = log_data["epochs"]
        m5095 = [v for v in log_data["map_5095"] if not np.isnan(v)]
        m50 = [v for v in log_data["map_50"] if not np.isnan(v)]
        tl = [v for v in log_data["train_loss"] if not np.isnan(v)]
        html.append("<h2>Training Summary</h2>")
        html.append(f"<div class='stats'><table>")
        html.append(f"<tr><th>Total Epochs</th><th>Best mAP@50-95</th>"
                    f"<th>Best mAP@50</th><th>Final Train Loss</th></tr>")
        html.append(f"<tr><td>{len(eps)}</td>"
                    f"<td>{max(m5095):.4f}</td>"
                    f"<td>{max(m50):.4f}</td>"
                    f"<td>{tl[-1]:.4f}</td></tr>")
        html.append("</table></div>")

    html.append("</body></html>")

    report_path = os.path.join(out_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))
    return report_path


# -----------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Ultralytics-style validation visualization for Multimodal DETR")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="outputs/ckpt/best.pth",
                    help="checkpoint path (default: outputs/ckpt/best.pth)")
    ap.add_argument("--data-root", default=None,
                    help="override data root for validation")
    ap.add_argument("--log-file", default="outputs/logs/log.txt",
                    help="training log file (JSONL or text)")
    ap.add_argument("--log-only", action="store_true",
                    help="only parse log and plot training curves (no model/data)")
    ap.add_argument("--out-dir", default="outputs/vis",
                    help="output directory for visualizations")
    ap.add_argument("--iou-th", type=float, default=0.5,
                    help="IoU threshold for TP/FP matching")
    ap.add_argument("--top-k-errors", type=int, default=12,
                    help="number of worst error-case images to save")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    cfg = _load_cfg(args.config)
    if args.data_root:
        cfg["data"]["train_split"] = args.data_root
        cfg["data"]["data_root"] = args.data_root

    class_names = _class_names(cfg)
    print(f"Classes: {class_names}")

    sections = []
    all_images = []

    # ---- 1. Training curves ----
    print("\n=== 1. Parsing training log ===")
    log_data = parse_log(args.log_file)
    n_eps = len(log_data.get("epochs", []))
    print(f"  Found {n_eps} epochs in {args.log_file}")
    curve_imgs = plot_training_curves(log_data, out_dir, class_names)
    all_images.extend(curve_imgs)
    if curve_imgs:
        sections.append({"type": "heading", "content": "1. Training Curves"})
        for img in curve_imgs:
            sections.append({"type": "image", "path": img,
                             "title": os.path.basename(img)})

    if args.log_only:
        print("\n--log-only mode: skipping model validation.")
        report = generate_html_report(out_dir, sections, log_data)
        print(f"\nReport: {report}")
        print(f"Training curves saved to: {out_dir}")
        return

    # ---- 2. Run validation ----
    print("\n=== 2. Running validation ===")
    try:
        results, val_ds = run_validation(cfg, args.ckpt)
    except Exception as e:
        print(f"  [ERROR] Validation failed: {e}")
        print("  Falling back to log-only mode.")
        report = generate_html_report(out_dir, sections, log_data)
        print(f"\nReport: {report}")
        return

    print(f"  Validated {len(results)} images")

    # ---- 3. GT data distribution ----
    print("\n=== 3. GT data distribution ===")
    dist_imgs, gt_counts = plot_gt_distribution(results, class_names, out_dir)
    all_images.extend(dist_imgs)
    sections.append({"type": "heading", "content": "2. GT Data Distribution"})
    for img in dist_imgs:
        sections.append({"type": "image", "path": img,
                         "title": os.path.basename(img)})

    # ---- 4. Per-class TP/FP/FN ----
    print("\n=== 4. Per-class TP/FP/FN ===")
    ncls = int(cfg["model"]["num_labels"])
    per_class = compute_tp_fp_fn(results, iou_th=args.iou_th, n_cls=ncls)
    tp_fp_img = plot_tp_fp_fn(per_class, class_names, out_dir)
    prf1_img = plot_precision_recall_f1(per_class, class_names, out_dir)
    all_images.extend([tp_fp_img, prf1_img])
    sections.append({"type": "heading", "content": "3. Per-Class TP/FP/FN"})
    sections.append({"type": "image", "path": tp_fp_img,
                     "title": "TP/FP/FN per class"})
    sections.append({"type": "image", "path": prf1_img,
                     "title": "Precision/Recall/F1 per class"})

    # ---- 5. PR curves ----
    print("\n=== 5. PR curves ===")
    pr_img = plot_pr_curves(per_class, class_names, out_dir)
    all_images.append(pr_img)
    sections.append({"type": "heading", "content": "4. PR Curves"})
    sections.append({"type": "image", "path": pr_img,
                     "title": "Per-class PR curves @ IoU 0.5"})

    # ---- 6. Confusion matrix ----
    print("\n=== 6. Confusion matrix ===")
    cm_img = plot_confusion_matrix(results, class_names, out_dir,
                                   iou_th=args.iou_th)
    all_images.append(cm_img)
    sections.append({"type": "heading", "content": "5. Confusion Matrix"})
    sections.append({"type": "image", "path": cm_img,
                     "title": "Confusion matrix (row-normalized)"})

    # ---- 7. Summary table ----
    print("\n=== 7. Summary table ===")
    table_img = plot_summary_table(per_class, class_names, out_dir, gt_counts)
    all_images.append(table_img)
    sections.append({"type": "heading", "content": "6. Summary Table"})
    sections.append({"type": "image", "path": table_img,
                     "title": "Per-class detection summary"})

    # ---- 8. Error cases ----
    print("\n=== 8. Error case visualization ===")
    data_root = cfg["data"].get("train_split") or cfg["data"].get("data_root")
    if data_root and os.path.isdir(data_root):
        err_imgs = draw_error_cases(results, class_names, out_dir,
                                    data_root, top_k=args.top_k_errors,
                                    iou_th=args.iou_th)
        if err_imgs:
            grid_img = err_imgs[-1]  # last one is the grid
            individual = err_imgs[:-1]
            sections.append({"type": "heading",
                             "content": "7. Error Case Visualization"})
            sections.append({"type": "image", "path": grid_img,
                             "title": "Worst error cases grid (FP+FN sorted)"})
            if len(individual) > 0:
                grid_items = [{"path": p,
                               "title": os.path.basename(p)}
                              for p in individual]
                sections.append({"type": "grid", "items": grid_items})
    else:
        print(f"  [WARN] Data root not found: {data_root}, skipping error cases.")

    # ---- 9. HTML report ----
    print("\n=== 9. Generating HTML report ===")
    report = generate_html_report(out_dir, sections, log_data)
    print(f"\n{'='*50}")
    print(f"Done! {len(all_images)} images + 1 HTML report generated.")
    print(f"Output: {out_dir}")
    print(f"Report: {report}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

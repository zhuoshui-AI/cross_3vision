"""Inference: write per-stem prediction txt files matching the spec.

Output line format (per spec): [cls, cx, cy, w, h, confidence]
- boxes are cxcywh normalized (DETR native), so we can emit them directly.
- ≤ max_dets_per_image boxes, sorted by score desc, conf_threshold filter.
- Empty file written when no detections survive the threshold.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from .data import MultiModalDataset, collate_fn
from .model import build_model
from .utils import load_yaml_config


def write_predictions_txt(path, boxes_cxcywh_norm, labels, scores):
    """One box per line: `cls cx cy w h conf` with 6 space-separated floats."""
    with open(path, "w", encoding="utf-8") as fh:
        for cls, (cx, cy, w, h), s in zip(labels.tolist(),
                                          boxes_cxcywh_norm.tolist(),
                                          scores.tolist()):
            fh.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {s:.6f}\n")


@torch.no_grad()
def run_inference(cfg, ckpt_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    # Same resolution logic as train.py: test_split may be an absolute dir
    # (the AIC2026 layout) or a relative split-file path under data_root.
    data_root = cfg["data"].get("data_root")
    test_split = cfg["data"].get("test_split")
    if test_split and os.path.isabs(test_split) and os.path.isdir(test_split):
        data_root, split_file = test_split, None
    elif test_split:
        split_file = os.path.join(data_root, test_split) if data_root else test_split
    else:
        split_file = None
    if data_root is None:
        raise ValueError(
            "data.data_root or data.test_split (absolute dir) must be set")

    ds = MultiModalDataset(
        data_root=data_root, split_file=split_file, ids=None,
        img_size=int(cfg["data"]["img_size"]), train=False,
        num_classes=int(cfg["model"]["num_labels"]),
    )
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        collate_fn=collate_fn, pin_memory=(device == "cuda"),
    )
    conf_th = float(cfg["inference"]["conf_threshold"])
    max_dets = int(cfg["inference"]["max_dets_per_image"])
    out_dir = cfg["inference"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    n_written = 0
    for batch in loader:
        rgb = batch["pixel_values_rgb"].to(device)
        ir = batch["pixel_values_ir"].to(device)
        depth = batch["pixel_values_depth"].to(device)
        depth_mask = batch["depth_mask"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)

        out = model(pixel_values_rgb=rgb, pixel_values_ir=ir,
                    pixel_values_depth=depth, depth_mask=depth_mask,
                    pixel_mask=pixel_mask, labels=None)
        probs = out.logits.softmax(-1)            # (1, Q, C+1)
        scores, labels_pred = probs[..., :-1].max(-1)  # drop no-object
        boxes = out.pred_boxes[0]                  # (Q, 4) cxcywh norm

        keep = scores[0] >= conf_th
        bx = boxes[keep]
        sc = scores[0][keep]
        lb = labels_pred[0][keep]
        # sort by score desc, keep top max_dets
        order = torch.argsort(sc, descending=True)
        bx = bx[order][:max_dets]
        sc = sc[order][:max_dets]
        lb = lb[order][:max_dets]

        stem = batch["stems"][0]
        write_predictions_txt(
            os.path.join(out_dir, stem + ".txt"), bx.cpu(), lb.cpu(), sc.cpu())
        n_written += 1
    print(f"Inference done: {n_written} images → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None, help="checkpoint path")
    ap.add_argument("--data-root", default=None,
                    help="override cfg data.data_root")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    run_inference(cfg, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()

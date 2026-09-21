"""Decisive test: tiny-overfit the REAL pipeline on a few example_data images.

If the code (data -> model -> loss -> eval) is consistent, mAP@50 on the SAME
images must climb well above 0.1 within a few hundred steps. If it stays ~0,
the pipeline has a reproducible train/eval inconsistency.

On the server (GPU), 300 steps takes ~1 min.

Usage:
    # use your training data dir (must have labels + 3 modalities)
    python -u -m scripts.diag_overfit --data-root /path/to/AIC2026_Train_2000

    # or use the bundled example_data
    python -u -m scripts.diag_overfit --data-root ./example_data
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data import MultiModalDataset, collate_fn
from src.model import build_model
from src.utils import load_yaml_config


def _iou(b, g):
    ix = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
    iy = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
    inter = ix * iy
    a1 = (b[2] - b[0]) * (b[3] - b[1])
    a2 = (g[2] - g[0]) * (g[3] - g[1])
    return float(inter / max(a1 + a2 - inter, 1e-9))


def mini_ap(preds, targets, iou_th):
    """preds/targets: list of dict(boxes xyxy px np[N,4], labels np[M], scores)."""
    cls_set = sorted({int(l) for t in targets for l in t["labels"]})
    aps = []
    for c in cls_set:
        n_gt, rows = 0, []
        for img, (p, t) in enumerate(zip(preds, targets)):
            pm = p["labels"] == c
            tm = t["labels"] == c
            n_gt += int(tm.sum())
            for b, s in zip(p["boxes"][pm], p["scores"][pm]):
                rows.append((float(s), img, b, t["boxes"][tm]))
        if n_gt == 0:
            continue
        rows.sort(key=lambda r: -r[0])
        matched = [set() for _ in targets]
        tp, fp = [], []
        for s, img, b, gtb in rows:
            best, bi = 0.0, -1
            for gi, gb in enumerate(gtb):
                if gi in matched[img]:
                    continue
                iou = _iou(b, gb)
                if iou > best:
                    best, bi = iou, gi
            if best >= iou_th:
                matched[img].add(bi)
                tp.append(1); fp.append(0)
            else:
                tp.append(0); fp.append(1)
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / max(n_gt, 1)
        prec = ctp / np.maximum(ctp + cfp, 1)
        mrec = np.concatenate([[0], rec, [1]])
        mpre = np.concatenate([[1], prec, [0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))
    return float(np.mean(aps)) if aps else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--img-size", type=int, default=512)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_yaml_config(args.config)
    cfg["data"]["img_size"] = args.img_size

    ds = MultiModalDataset(data_root=args.data_root, split_file=None,
                           img_size=args.img_size, train=True,
                           num_classes=int(cfg["model"]["num_labels"]))
    ds.ids = ds.ids[:args.n_imgs]
    print(f"[overfit] {len(ds)} imgs @ {args.img_size}px on {device}", flush=True)

    model = build_model(cfg).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[overfit] trainable params: {n_p/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20))
    t0 = time.time()
    for step in range(args.steps):
        # rebuild one big batch from all images each step (no augmentation in
        # train transform here, but let's be explicit: use train=True so the
        # real training pipeline is exercised, including RandomResizedCrop).
        items = [ds[i] for i in range(len(ds))]
        batch = collate_fn(items)
        batch = {k: (v.to(device) if torch.is_tensor(v) else
                     [{"class_labels": t["class_labels"].to(device),
                       "boxes": t["boxes"].to(device)} for t in v])
                 for k, v in batch.items() if k in ("pixel_values_rgb",
                                                     "pixel_values_ir",
                                                     "pixel_values_depth",
                                                     "depth_mask", "pixel_mask",
                                                     "labels")}
        out = model(**batch)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        opt.step(); sched.step()

        with torch.no_grad():
            probs = out.logits.softmax(-1)
            s_ne, _ = probs[..., :-1].max(-1)
            wh = out.pred_boxes[..., 2:]
            if step % 50 == 0 or step == args.steps - 1:
                ld = out.loss_dict
                print(f"step {step:4d} total={float(out.loss):7.3f} "
                      f"bbox={float(ld['loss_bbox']):.3f} "
                      f"giou={float(ld['loss_giou']):.3f} "
                      f"ce={float(ld['loss_ce']):.3f} "
                      f"wh_std={float(wh.std()):.4f} "
                      f"p50_non_eos={float(s_ne.median()):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    with torch.no_grad():
        items = [ds[i] for i in range(len(ds))]
        batch = collate_fn(items)
        batch = {k: (v.to(device) if torch.is_tensor(v) else
                     [{"class_labels": t["class_labels"].to(device),
                       "boxes": t["boxes"].to(device)} for t in v])
                 for k, v in batch.items() if k in ("pixel_values_rgb",
                                                     "pixel_values_ir",
                                                     "pixel_values_depth",
                                                     "depth_mask", "pixel_mask",
                                                     "labels")}
        out = model(**batch)
    probs = out.logits.softmax(-1)
    scores, labels_pred = probs[..., :-1].max(-1)
    preds, targets = [], []
    SZ = args.img_size
    for b in range(len(ds)):
        keep = scores[b] >= 0.05
        bx = out.pred_boxes[b][keep].cpu().numpy()
        cx, cy, w, h = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * SZ
        preds.append({"boxes": xyxy, "scores": scores[b][keep].cpu().numpy(),
                      "labels": labels_pred[b][keep].cpu().numpy()})
        tbox = batch["labels"][b]["boxes"].cpu().numpy()
        cx, cy, w, h = tbox[:, 0], tbox[:, 1], tbox[:, 2], tbox[:, 3]
        txyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * SZ
        targets.append({"boxes": txyxy,
                        "labels": batch["labels"][b]["class_labels"].cpu().numpy()})
    ap50 = mini_ap(preds, targets, 0.5)
    ap75 = mini_ap(preds, targets, 0.75)
    n_pred = sum(len(p["scores"]) for p in preds)
    n_gt = sum(len(t["labels"]) for t in targets)
    print(f"\n[overfit] preds kept={n_pred} (gt={n_gt})  "
          f"mini-mAP@50={ap50:.4f}  mini-mAP@75={ap75:.4f}", flush=True)
    if ap50 < 0.05:
        print("[overfit] >>> mAP stays ~0 even overfitting => REPRODUCIBLE bug", flush=True)
    else:
        print("[overfit] >>> pipeline CAN learn => server issue is data/config scale", flush=True)


if __name__ == "__main__":
    main()

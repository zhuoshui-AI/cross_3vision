"""Run on the TRAINING SERVER (has best.pth + full dataset):

    python -m scripts.diag_server --ckpt outputs/ckpt/best.pth \
        --data-root <train_root>   # same root train.py used

Prints an autopsy of why mAP~0:
  [A] val loss_dict breakdown          -> which loss still dominates after 150 ep
  [B] prediction score distribution    -> do object queries clear conf 0.05?
  [C] predicted class histogram vs GT  -> classifier collapse?
  [D] per-GT best-IoU stats            -> do boxes even land near objects?
  [E] preds-per-image after filtering  -> is evaluate() fed empty predictions?
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data import MultiModalDataset, collate_fn
from src.model import build_model
from src.utils import load_yaml_config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-imgs", type=int, default=100)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    root = args.data_root or cfg["data"]["train_split"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"  WARNING missing keys: {missing[:5]} ...")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected[:5]} ...")
    model.eval()
    print(f"loaded {args.ckpt} (epoch {state.get('epoch')}, best {state.get('best_metric')})")

    ds = MultiModalDataset(data_root=root, split_file=None, img_size=512,
                           train=False, num_classes=int(cfg["model"]["num_labels"]))
    # deterministic subset
    ds.ids = ds.ids[:args.max_imgs]
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                         collate_fn=collate_fn, num_workers=2)
    print(f"diag over {len(ds)} val-distribution images")

    conf_th = float(cfg["inference"]["conf_threshold"])
    C = int(cfg["model"]["num_labels"])
    from src.losses import cxcywh_to_xyxy_pixel

    loss_sum, n_batches = {}, 0
    score_all, kept_all, pred_cls_hist, gt_cls_hist = [], 0, np.zeros(C, np.int64), np.zeros(C, np.int64)
    best_iou_all, n_gt_total = [], 0
    boxes_per_img = []

    for batch in loader:
        rgb = batch["pixel_values_rgb"].to(device)
        ir = batch["pixel_values_ir"].to(device)
        dep = batch["pixel_values_depth"].to(device)
        dm = batch["depth_mask"].to(device)
        pm = batch["pixel_mask"].to(device)
        labels = [{"class_labels": t["class_labels"].to(device), "boxes": t["boxes"].to(device)}
                  for t in batch["labels"]]
        out = model(pixel_values_rgb=rgb, pixel_values_ir=ir, pixel_values_depth=dep,
                    depth_mask=dm, pixel_mask=pm, labels=labels)
        n_batches += 1
        for k, v in out.loss_dict.items():
            if k.split("_")[-1].isdigit():
                continue  # skip aux-layer copies
            loss_sum[k] = loss_sum.get(k, 0.0) + float(v)

        probs = out.logits.softmax(-1)
        scores, labels_pred = probs[..., :-1].max(-1)
        for b in range(rgb.shape[0]):
            sb = scores[b]
            score_all.append(sb.cpu().numpy())
            keep = sb >= conf_th
            kept_all += int(keep.sum())
            boxes_per_img.append(int(keep.sum()))
            for c in labels_pred[b][keep].flatten().cpu().numpy():
                pred_cls_hist[int(c)] += 1
            tgt = labels[b]
            for c in tgt["class_labels"].cpu().numpy():
                gt_cls_hist[int(c)] += 1
            n_gt_total += int(tgt["boxes"].shape[0])
            # best IoU per GT over ALL kept preds (class-agnostic localization check)
            pb = cxcywh_to_xyxy_pixel(out.pred_boxes[b][keep], (512, 512)).cpu().numpy()
            tb = cxcywh_to_xyxy_pixel(tgt["boxes"], (512, 512)).cpu().numpy()
            for gb in tb:
                best = 0.0
                for pbox in pb:
                    ix = max(0.0, min(pbox[2], gb[2]) - max(pbox[0], gb[0]))
                    iy = max(0.0, min(pbox[3], gb[3]) - max(pbox[1], gb[1]))
                    inter = ix * iy
                    u = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]) + \
                        (gb[2] - gb[0]) * (gb[3] - gb[1]) - inter
                    best = max(best, inter / max(u, 1e-9))
                best_iou_all.append(best)

    s = np.concatenate(score_all)
    print(f"\n[A] val loss (mean over {n_batches} batches):")
    for k in sorted(loss_sum):
        print(f"    {k:22s} {loss_sum[k]/n_batches:8.4f}")

    print(f"\n[B] max non-eos prob per query: p10={np.percentile(s,10):.4f} "
          f"p50={np.percentile(s,50):.4f} p90={np.percentile(s,90):.4f} "
          f"p99={np.percentile(s,99):.4f} max={s.max():.4f}")
    print(f"    preds surviving conf {conf_th}: {kept_all} "
          f"({kept_all/max(len(ds),1):.1f}/img, gt={n_gt_total/max(len(ds),1):.1f}/img)")

    print(f"\n[C] pred class hist: {pred_cls_hist.tolist()}")
    print(f"    GT   class hist: {gt_cls_hist.tolist()}")

    bi = np.array(best_iou_all)
    print(f"\n[D] per-GT best-IoU (class-agnostic): p25={np.percentile(bi,25):.3f} "
          f"p50={np.percentile(bi,50):.3f} p75={np.percentile(bi,75):.3f} "
          f"mean={bi.mean():.3f}  IoU>0.5: {(bi>0.5).mean()*100:.1f}%  IoU>0.1: {(bi>0.1).mean()*100:.1f}%")
    print(f"\n[E] preds/img after filter: p50={np.percentile(boxes_per_img,50):.0f} "
          f"max={max(boxes_per_img)} zero-imgs={sum(1 for x in boxes_per_img if x==0)}")


if __name__ == "__main__":
    main()

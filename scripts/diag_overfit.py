"""Decisive test: tiny-overfit the REAL pipeline on a few example_data images.

If the code (data -> model -> loss -> eval) is consistent, mAP@50 on the SAME
images must climb well above 0.1 within a few hundred steps. If it stays ~0,
the pipeline has a reproducible train/eval inconsistency.

On the server (GPU), 1500 steps takes ~5-8 min.

AP uses the competition's 101-point interpolation (spec §6) so the numbers
are directly comparable to the official metric logic.

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
        # Competition rule (spec §6): 101-point interpolated AP — average, over
        # r = 0, 0.01, ..., 1, of the max precision at recall >= r.
        q = np.zeros(101, dtype=np.float64)
        for i, r in enumerate(np.linspace(0.0, 1.0, 101)):
            mask = rec >= r
            q[i] = float(prec[mask].max()) if mask.any() else 0.0
        aps.append(float(q.mean()))
    return float(np.mean(aps)) if aps else float("nan")


def decode_preds(logits, pred_boxes, img_size, conf=0.05):
    """Softmax (EOS excluded) -> per-image kept preds in pixel xyxy."""
    probs = logits.softmax(-1)
    scores, labels = probs[..., :-1].max(-1)
    preds = []
    for b in range(logits.shape[0]):
        keep = scores[b] >= conf
        bx = pred_boxes[b][keep].cpu().numpy()
        cx, cy, w, h = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * img_size
        preds.append({"boxes": xyxy, "scores": scores[b][keep].cpu().numpy(),
                      "labels": labels[b][keep].cpu().numpy()})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--img-size", type=int, default=512)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_yaml_config(args.config)
    cfg["data"]["img_size"] = args.img_size

    ds = MultiModalDataset(data_root=args.data_root, split_file=None,
                           img_size=args.img_size, train=False,
                           num_classes=int(cfg["model"]["num_labels"]))
    ds.ids = ds.ids[:args.n_imgs]
    print(f"[overfit] {len(ds)} imgs @ {args.img_size}px on {device} (train=False, fixed data)", flush=True)

    model = build_model(cfg).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[overfit] trainable params: {n_p/1e6:.1f}M", flush=True)

    # Pre-cache a FIXED batch (train=False => deterministic Resize only) so
    # the model can truly memorize these 8 images.
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

    SZ = args.img_size
    targets = []
    for b in range(len(ds)):
        tbox = batch["labels"][b]["boxes"].cpu().numpy()
        cx, cy, w, h = tbox[:, 0], tbox[:, 1], tbox[:, 2], tbox[:, 3]
        txyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * SZ
        targets.append({"boxes": txyxy,
                        "labels": batch["labels"][b]["class_labels"].cpu().numpy()})

    # Two-param-group AdamW mirroring src/train.py, but with a cosine schedule
    # (constant LR after warmup caused repeated loss spikes at steps ~300/1000
    # and prevented convergence on this tiny 8-image batch). Head LR is halved
    # to 5e-5 to tame the oscillation; backbone stays at 1e-5.
    import math
    tcfg = cfg.get("train") or {}
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = any(s in n for s in (
            "rgb_backbone.", "ir_backbone.", "depth_encoder."))
        (backbone_params if is_backbone else head_params).append(p)
    base_lr = float(tcfg.get("lr", 1e-4)) * 0.5
    bb_lr = float(tcfg.get("backbone_lr", 1e-5))
    opt = torch.optim.AdamW(
        [{"params": head_params, "lr": base_lr},
         {"params": backbone_params, "lr": bb_lr}],
        weight_decay=float(tcfg.get("weight_decay", 1e-4)), betas=(0.9, 0.999))
    warmup = 50
    cosine_steps = max(1, args.steps - warmup)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / cosine_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    t0 = time.time()
    for step in range(args.steps):
        out = model(**batch)
        opt.zero_grad()
        out.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
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
                      f"gnorm={float(grad_norm):.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if step % 100 == 0 or step == args.steps - 1:
                with torch.no_grad():
                    p_now = decode_preds(out.logits, out.pred_boxes, SZ)
                print(f"         -> mini-AP@50 = "
                      f"{mini_ap(p_now, targets, 0.5):.4f} (in-loop)", flush=True)

    # ---- Mode-divergence probe: compare TRAIN vs EVAL on the same batch.
    # If eval queries collapse but train queries don't, some module behaves
    # differently by mode (BatchNorm/Dropout/DropPath). Hook the fused
    # backbone feature to see whether collapse starts in the backbone.
    captured = {}

    def _hook(module, inp, outp):
        # MultiModalBackbone returns [(f_enh, mask)]
        captured["f"] = outp[0][0].detach()

    handle = model.model.backbone.conv_encoder.register_forward_hook(_hook)

    def _mode_stats(mode):
        model.train(mode)
        with torch.no_grad():
            o = model(**batch)
        f = captured["f"]  # (B, C, h, w)
        # spatial variation of backbone features (per-image mean over channels
        # of the std across HW tokens): ~0 means the feature map collapsed.
        f_spatial_std = float(f.flatten(2).std(dim=2).mean())
        # query diversity: std of predicted boxes across the 100 queries
        box_std = float(o.pred_boxes.std(dim=1).mean())
        # number of distinct box modes (rounded to 2px) on image 0
        b0 = (o.pred_boxes[0] * args.img_size).round().tolist()
        n_modes = len({tuple(round(v, 0) for v in row) for row in b0})
        return f_spatial_std, box_std, n_modes, o

    f_tr, b_tr, m_tr, _ = _mode_stats(True)
    f_ev, b_ev, m_ev, out = _mode_stats(False)
    handle.remove()
    print(f"[probe] TRAIN: feat_spatial_std={f_tr:.4f} box_std={b_tr:.4f} "
          f"unique_box_modes(img0)={m_tr}", flush=True)
    print(f"[probe] EVAL : feat_spatial_std={f_ev:.4f} box_std={b_ev:.4f} "
          f"unique_box_modes(img0)={m_ev}", flush=True)
    if f_ev < 1e-3 and f_tr >= 1e-3:
        print("[probe] >>> backbone FEATURES collapse only in EVAL "
              "=> norm/mode bug in a backbone branch", flush=True)
    elif f_ev >= 1e-3 and m_ev <= 2:
        print("[probe] >>> features fine but DECODER queries collapse "
              "=> check decoder/query wiring", flush=True)

    model.eval()
    preds = decode_preds(out.logits, out.pred_boxes, SZ)
    ap50 = mini_ap(preds, targets, 0.5)
    ap75 = mini_ap(preds, targets, 0.75)
    n_pred = sum(len(p["scores"]) for p in preds)
    n_gt = sum(len(t["labels"]) for t in targets)
    if n_gt == 0:
        print(f"\n[overfit] >>> selected {len(ds)} images contain ZERO GT boxes; "
              f"the overfit test is meaningless (check labels/ dir / id list).",
              flush=True)
        sys.exit(1)
    print(f"\n[overfit] preds kept={n_pred} (gt={n_gt})  "
          f"mini-AP@50={ap50:.4f}  mini-AP@75={ap75:.4f}  "
          f"(101-pt interp, {len({int(l) for t in targets for l in t['labels']})} "
          f"classes present)", flush=True)
    # print first image boxes for visual sanity check
    if preds and targets:
        print(f"[overfit] img0: {len(preds[0]['boxes'])} preds, {len(targets[0]['boxes'])} gts", flush=True)
        print(f"  GT   boxes(xyxy px): {np.round(targets[0]['boxes'][:5], 1).tolist()}", flush=True)
        print(f"  PRED boxes(xyxy px): {np.round(preds[0]['boxes'][:5], 1).tolist()}", flush=True)
        print(f"  GT   classes: {targets[0]['labels'].tolist()}", flush=True)
        print(f"  PRED classes: {preds[0]['labels'][:5].tolist()}", flush=True)
    if ap50 < 0.05:
        print("[overfit] >>> mAP stays ~0 even overfitting => REPRODUCIBLE bug", flush=True)
    else:
        print("[overfit] >>> pipeline CAN learn => server issue is data/config scale", flush=True)


if __name__ == "__main__":
    main()

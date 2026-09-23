"""Training entry: manual loop with AMP, cosine + linear warmup, grad accum,
two-param-group AdamW (backbone vs. head), periodic val + best-ckpt save.

We bypass HF Trainer for explicit control over the multi-modal batch dict
(per-modality tensors + depth_mask). Loss / matcher / auxiliary losses are
reused from HF via MultiModalSwinDETR.loss_function (ForObjectDetectionLoss).
"""
import argparse
import math
import os

import torch
from torch.utils.data import DataLoader

from .data import MultiModalDataset, collate_fn
from .eval import evaluate
from .model import build_model
from .utils import (EMA, load_yaml_config, save_ckpt, set_seed, setup_logging)


def _move_batch(batch, device):
    return {
        "pixel_values_rgb": batch["pixel_values_rgb"].to(device, non_blocking=True),
        "pixel_values_ir": batch["pixel_values_ir"].to(device, non_blocking=True),
        "pixel_values_depth": batch["pixel_values_depth"].to(device, non_blocking=True),
        "depth_mask": batch["depth_mask"].to(device, non_blocking=True),
        "pixel_mask": batch["pixel_mask"].to(device, non_blocking=True),
        "labels": [
            {"class_labels": t["class_labels"].to(device),
             "boxes": t["boxes"].to(device)}
            for t in batch["labels"]
        ],
    }


def build_optimizer(model, cfg):
    """AdamW with two param groups: backbones at backbone_lr, rest at lr.

    Works for both architectures: cssa_detr params live under
    `conv_encoder.{rgb_backbone,ir_backbone,depth_encoder}`, tri_swin params
    under `tri_backbone.{rgb_backbone,ir_backbone,depth_encoder}` — the
    substring matching below covers both.
    """
    tcfg = cfg["train"]
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Anything under our MultiModalBackbone's rgb/ir/depth sub-backbones
        # gets the backbone learning rate; fusion + DETR heads get the head lr.
        is_backbone = any(s in n for s in (
            "rgb_backbone.", "ir_backbone.", "depth_encoder."))
        (backbone_params if is_backbone else head_params).append(p)
    groups = [
        {"params": head_params, "lr": float(tcfg["lr"])},
        {"params": backbone_params, "lr": float(tcfg["backbone_lr"])},
    ]
    opt = torch.optim.AdamW(
        groups, weight_decay=float(tcfg["weight_decay"]), betas=(0.9, 0.999))
    return opt


def build_scheduler(optimizer, cfg, total_steps):
    tcfg = cfg["train"]
    warmup = int(tcfg["warmup_steps"])
    cosine_steps = max(1, total_steps - warmup)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)  # linear warmup
        s = step - warmup
        return 0.5 * (1 + math.cos(math.pi * s / cosine_steps))  # cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg, resume=None):
    tcfg = cfg["train"]
    log = setup_logging(tcfg.get("log_dir"), "multimodal_swin_detr")
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    img_size = int(cfg["data"]["img_size"])
    ncls = int(cfg["model"]["num_labels"])

    def _resolve(split_key):
        """Resolve (data_root, split_file) for a split.

        Supports two config styles:
          1) data_root + split_file:  split_key = "splits/train.txt" (relative
             to data_root); data_root holds visible/infrared/depth/labels subdirs.
          2) split_key = absolute directory path: the directory itself holds
             visible/infrared/depth/labels; data_root = that dir, split_file = None
             (auto-scan labels/). This is how AIC2026 train/test sets are laid
             out — two independent dataset directories.
        """
        data_root = cfg["data"].get("data_root")
        sp = cfg["data"].get(split_key)
        if sp and os.path.isabs(sp) and os.path.isdir(sp):
            return sp, None
        if sp:
            return data_root, os.path.join(data_root, sp) if data_root else sp
        return data_root, None

    train_root, train_split = _resolve("train_split")
    val_root, val_split = _resolve("val_split")
    if train_root is None:
        raise ValueError(
            "data.data_root or data.train_split (absolute dir) must be set")

    # ---- Split strategy ----
    # If val_split points to a *different* directory, use it as an independent
    # val set. Otherwise (val_split is null, missing, or the same dir as train)
    # carve out a val_ratio fraction of the training IDs as the val set.
    # This is the recommended setup for AIC2026: the test set has no labels, so
    # we hold out part of the training set for mAP validation.
    use_separate_val = (val_root is not None
                        and val_split is not None
                        and os.path.abspath(val_root) != os.path.abspath(train_root))

    if use_separate_val:
        train_ds = MultiModalDataset(
            data_root=train_root, split_file=train_split,
            img_size=img_size, train=True, num_classes=ncls)
        val_ds = MultiModalDataset(
            data_root=val_root, split_file=val_split,
            img_size=img_size, train=False, num_classes=ncls)
        log.info(f"Val set: independent dir {val_root} ({len(val_ds)} imgs)")
    else:
        # Get the full ID list once, then split by val_ratio.
        full_ds = MultiModalDataset(
            data_root=train_root, split_file=train_split,
            img_size=img_size, train=True, num_classes=ncls)
        n_total = len(full_ds)
        val_ratio = float(cfg["data"].get("val_ratio", 0.1))
        n_val = max(1, int(round(n_total * val_ratio)))
        n_train = n_total - n_val
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_total, generator=g).tolist()
        train_ids = [full_ds.ids[i] for i in perm[:n_train]]
        val_ids = [full_ds.ids[i] for i in perm[n_train:]]
        train_ds = MultiModalDataset(
            data_root=train_root, ids=train_ids,
            img_size=img_size, train=True, num_classes=ncls)
        val_ds = MultiModalDataset(
            data_root=train_root, ids=val_ids,
            img_size=img_size, train=False, num_classes=ncls)
        log.info(f"Split train set {train_root}: {n_train} train / {n_val} val "
                 f"(val_ratio={val_ratio})")

    nw = int(cfg["data"].get("num_workers", 4))
    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True,
        num_workers=nw, collate_fn=collate_fn,
        pin_memory=(device == "cuda"), drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=nw,
        collate_fn=collate_fn, pin_memory=(device == "cuda"))

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train_loader) // int(tcfg["grad_accum"]))
    total_steps = steps_per_epoch * int(tcfg["epochs"])
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    amp = bool(tcfg.get("amp", True)) and device == "cuda"
    amp_dtype = torch.bfloat16 if (amp and torch.cuda.is_bf16_supported()) \
        else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and amp_dtype == torch.float16))
    ema_decay = float(tcfg.get("ema_decay", 0.0))
    ema = EMA(model, ema_decay) if ema_decay > 0 else None

    start_epoch = 0
    best = -1.0
    if resume and os.path.isfile(resume):
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        if state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", -1)) + 1
        best = float(state.get("best_metric", -1.0))
        log.info(f"Resumed from {resume} at epoch {start_epoch}, best={best}")

    ckpt_dir = tcfg["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    val_interval = int(tcfg["val_interval"])
    grad_clip = float(tcfg["grad_clip"])

    for epoch in range(start_epoch, int(tcfg["epochs"])):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for it, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.split(":")[0],
                                dtype=amp_dtype, enabled=amp):
                out = model(**batch)
                loss = out.loss / int(tcfg["grad_accum"])
            scaler.scale(loss).backward()
            if (it + 1) % int(tcfg["grad_accum"]) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                if ema:
                    ema.update(model)
            running += loss.item() * int(tcfg["grad_accum"])
            if it % 50 == 0:
                log.info(f"ep{epoch} it{it}/{len(train_loader)} "
                         f"loss={loss.item()*int(tcfg['grad_accum']):.4f} "
                         f"lr={scheduler.get_last_lr()[0]:.2e}")
        log.info(f"Epoch {epoch} mean loss = {running/len(train_loader):.4f}")

        if (epoch + 1) % val_interval == 0 or epoch + 1 == int(tcfg["epochs"]):
            metrics = evaluate(model, val_loader, device, cfg)
            score = metrics.get("map_5095", float("nan"))
            log.info(f"Val ep{epoch}: mAP@50-95={score:.4f} "
                     f"mAP@50={metrics.get('map_50', float('nan')):.4f}")
            if score == score and score > best:  # not NaN & improved
                best = score
                save_ckpt(os.path.join(ckpt_dir, "best.pth"),
                          model, optimizer, scheduler, epoch, best, cfg)
                log.info(f"  ↑ new best {best:.4f} → best.pth")
        save_ckpt(os.path.join(ckpt_dir, "last.pth"),
                  model, optimizer, scheduler, epoch, best, cfg)

    log.info(f"Training done. Best mAP@50-95 = {best:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()

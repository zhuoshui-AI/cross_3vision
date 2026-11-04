"""Training entry: manual loop with AMP, cosine + linear warmup, grad accum,
two-param-group AdamW (backbone vs. head), periodic val + best-ckpt save.

We bypass HF Trainer for explicit control over the multi-modal batch dict
(per-modality tensors + depth_mask). Loss / matcher / auxiliary losses are
reused from HF via the DETR model's loss_function (ForObjectDetectionLoss).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# 兼容两种运行方式：
#   python -m src.train          → 包模式，用相对导入
#   python src/train.py          → 脚本模式，手动加父目录到 sys.path
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import MultiModalDataset, collate_fn
    from src.eval import evaluate
    from src.model import build_model
    from src.utils import (EMA, load_yaml_config, save_ckpt, set_seed, setup_logging)
else:
    from .data import MultiModalDataset, collate_fn
    from .eval import evaluate
    from .model import build_model
    from .utils import (EMA, load_yaml_config, save_ckpt, set_seed, setup_logging)


def _get_git_info(project_root):
    """获取当前仓库的 git 信息，返回 (sha, branch, subject, dirty)。

    任一步失败都回退到占位字符串，保证训练不因为 git 不可用而中断。
    """
    sha = "nogit"
    branch = "unknown"
    subject = "no-commit"
    dirty = False
    try:
        def _run(*args):
            return subprocess.check_output(
                ["git", "-C", str(project_root), *args],
                stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
        sha = _run("rev-parse", "--short=7", "HEAD") or sha
        branch = _run("rev-parse", "--abbrev-ref", "HEAD") or branch
        subject = _run("log", "-1", "--format=%s") or subject
        dirty = bool(_run("status", "--porcelain"))
    except Exception:
        pass
    return sha, branch, subject, dirty


def _sanitize_name(text, max_len=40):
    """把任意文本转成安全的文件/文件夹名。

    - 只保留字母、数字、下划线、连字符
    - 连续多个分隔符折叠成一个
    - 截断到 max_len 字符
    - 转小写
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\-]+", "_", text)  # 非字母数字下划线连字符 → _
    text = re.sub(r"_+", "_", text).strip("_-")  # 连续 _ → 单个
    return text[:max_len] or "untitled"


def _build_run_name(project_root):
    """生成本次运行的唯一标识名：{sha7}_{safe_subject}_{timestamp}。

    示例：`a1b2c3d_fix_depth_encoding_20261104_143025`
    """
    sha, _branch, subject, dirty = _get_git_info(project_root)
    safe_subj = _sanitize_name(subject)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{sha}_{safe_subj}_{ts}"
    if dirty:
        name += "_dirty"
    return name, sha, subject, dirty


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

    rgbir_swin params live under `conv_encoder.rgbir_backbone` (4ch Swin) /
    `conv_encoder.depth_encoder`, tri_swin params under
    `tri_backbone.{rgb_backbone,ir_backbone,depth_encoder}` — the substring
    matching below covers both.
    """
    tcfg = cfg["train"]
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Anything under the per-modality sub-backbones gets the backbone
        # learning rate; fusion + DETR heads get the head lr.
        # rgbir_swin 架构的 4 通道 Swin 主干命名为 rgbir_backbone，同样归入
        # backbone 参数组（微调 lr）。
        is_backbone = any(s in n for s in (
            "rgb_backbone.", "ir_backbone.", "depth_encoder.",
            "rgbir_backbone."))
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
        # cosine; clamp ≥0 so overshooting total_steps can't go negative
        return max(0.0, 0.5 * (1 + math.cos(math.pi * s / cosine_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _scale_loss_stats(raw_stats, loss_weights, prefix):
    """按 DETR log.txt 约定把 loss_dict 均值展开成 缩放/未缩放 两版字段。

    raw_stats: {loss_dict key: 均值}（HF 返回的均为未乘权重的原始值，
    辅助层键形如 loss_ce_0）。输出键示例：
    train_loss_ce / train_loss_ce_unscaled；
    class_error / cardinality_error 不加权（与原仓一致）。
    """
    stats = {}
    for k, v in raw_stats.items():
        base = re.sub(r"_\d+$", "", k)  # 剥掉辅助层后缀 _0.._4
        is_error = base in ("class_error", "cardinality_error")
        scale = 1.0 if is_error else float(loss_weights.get(base, 1.0))
        stats[f"{prefix}_{k}"] = v * scale
        stats[f"{prefix}_{k}_unscaled"] = v
    return stats


@torch.no_grad()
def _val_loss_pass(model, val_loader, device, cfg):
    """在验证集上计算 DETR 损失分量的均值（no_grad，不改 BN/EMA 状态）。"""
    was_training = model.training
    model.eval()
    sums, n = {}, 0
    for batch in val_loader:
        batch = _move_batch(batch, device)
        out = model(**batch)
        for k, v in out.loss_dict.items():
            sums[k] = sums.get(k, 0.0) + float(v.item())
        n += 1
    if was_training:
        model.train()
    return {k: v / max(1, n) for k, v in sums.items()}


def train(cfg, resume=None, restart=None):
    tcfg = cfg["train"]
    arch = cfg["model"].get("arch", "unknown")

    # ---- 生成唯一运行标识（git commit + 时间戳）----
    # 每次训练都在独立子文件夹下，互不覆盖；日志文件名也与之对齐。
    project_root = Path(__file__).resolve().parent.parent
    run_name, git_sha, git_subject, git_dirty = _build_run_name(project_root)

    # 手动覆盖：如果用户通过 --set 指定了完整路径，尊重其选择；
    # 否则按 base/arch/run_name 的层级自动隔离。
    force_run_dir = tcfg.get("force_run_dir", False)
    for key in ("ckpt_dir", "log_dir"):
        raw = tcfg.get(key) or f"./outputs/{key.replace('_dir','')}"
        # 如果用户已手动指定到 arch/run_name 级别，就不再追加
        if force_run_dir or run_name in raw.split(os.sep):
            tcfg[key] = raw
            continue
        # 先确保有 /{arch}，再加 /{run_name}
        if f"/{arch}" not in raw and f"\\{arch}" not in raw:
            raw = raw.rstrip("/\\") + f"/{arch}"
        tcfg[key] = raw.rstrip("/\\") + f"/{run_name}"

    # setup_logging 用 run_name 作 run_id，日志文件名 train_{run_name}.log
    log = setup_logging(tcfg.get("log_dir"), "multimodal_swin_detr", run_id=run_name)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 打印清晰的启动 banner
    log.info("=" * 60)
    log.info(f"  Run name              : {run_name}")
    log.info(f"  Git commit            : {git_sha}  {'(dirty)' if git_dirty else ''}")
    log.info(f"  Git subject           : {git_subject}")
    log.info(f"  Network architecture : {arch}")
    log.info(f"  Device                : {device}")
    log.info(f"  Image size            : {cfg['data']['img_size']}")
    log.info(f"  Classes / queries     : {cfg['model']['num_labels']} / {cfg['model']['num_queries']}")
    log.info(f"  Epochs / batch / accum: {cfg['train']['epochs']} / {cfg['train']['batch_size']} / {cfg['train']['grad_accum']}")
    log.info(f"  CKPT dir              : {tcfg['ckpt_dir']}")
    log.info(f"  Log dir               : {tcfg['log_dir']}")
    if cfg["model"].get("detr_warm_start"):
        log.info(f"  DETR warm start       : {cfg['model']['detr_warm_start']}")
    log.info("=" * 60)

    # ---- 完整有效配置写入日志（--set / --data-root 覆盖后的最终值）----
    # 便于事后从日志完整复现本次训练的全部超参（含 CLI 覆盖项）。
    log.info("Effective config (含 CLI --set / --data-root 覆盖):")
    log.info(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False,
                            default_flow_style=False).rstrip())
    log.info("=" * 60)

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
    depth_enc = cfg["model"].get("depth_encoding", "linear")

    if use_separate_val:
        train_ds = MultiModalDataset(
            data_root=train_root, split_file=train_split,
            img_size=img_size, train=True, num_classes=ncls,
            depth_encoding=depth_enc)
        val_ds = MultiModalDataset(
            data_root=val_root, split_file=val_split,
            img_size=img_size, train=False, num_classes=ncls,
            depth_encoding=depth_enc)
        log.info(f"Val set: independent dir {val_root} ({len(val_ds)} imgs)")
    else:
        # Get the full ID list once, then split by val_ratio.
        full_ds = MultiModalDataset(
            data_root=train_root, split_file=train_split,
            img_size=img_size, train=True, num_classes=ncls,
            depth_encoding=depth_enc)
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
            img_size=img_size, train=True, num_classes=ncls,
            depth_encoding=depth_enc)
        val_ds = MultiModalDataset(
            data_root=train_root, ids=val_ids,
            img_size=img_size, train=False, num_classes=ncls,
            depth_encoding=depth_enc)
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
    # ceil: the epoch-end flush below steps on leftover partial groups too
    steps_per_epoch = max(1, math.ceil(len(train_loader) / int(tcfg["grad_accum"])))
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
    if restart and os.path.isfile(restart):
        # 热重启（SGDR 式新余弦周期）：只继承模型权重，优化器/调度器全新。
        # 调度器按当前 config 的 epochs 从 warmup + 峰值 lr 重新退火；
        # EMA 状态未入 checkpoint，自动从载入权重重新累积。
        state = torch.load(restart, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        log.info(f"Restart: 载入 {restart} 权重，optimizer/scheduler 全新 "
                 f"（新 cosine 周期 {tcfg['epochs']} epochs）")
    elif resume and os.path.isfile(resume):
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
    # ---- iter 级日志间隔（防大 epoch 刷屏 / 小 epoch 不欠打）----
    #   "auto" = ceil(iters/epoch ÷ log_lines_per_epoch)，每 epoch 固定约 N
    #            行；iters/epoch = 样本数/batch_size，随 batch 与数据规模
    #            自动伸缩（batch 越大/数据越小 → 间隔越小）
    #   数字 N = 固定间隔；0 = 关闭。epoch 级汇总不受此影响。
    n_iters_per_epoch = len(train_loader)
    li_cfg = tcfg.get("log_interval", "auto")
    lines_per_epoch = int(tcfg.get("log_lines_per_epoch", 10))
    if isinstance(li_cfg, str) and li_cfg.lower() == "auto":
        log_interval = max(
            1, math.ceil(n_iters_per_epoch / max(1, lines_per_epoch)))
        li_desc = (f"auto (interval={log_interval}, {n_iters_per_epoch} "
                   f"iters/epoch ÷ {lines_per_epoch} lines)")
    else:
        log_interval = max(0, int(li_cfg))
        li_desc = f"fixed (interval={log_interval or 'off'})"
    log.info(f"Log: iter {li_desc}, val every {val_interval} epochs")

    # ---- 早停（以验证 mAP@50-95 为监控指标）----
    es_patience = int(tcfg.get("early_stop_patience", 0))
    es_min_delta = float(tcfg.get("early_stop_min_delta", 0.0))
    es_bad_evals = 0          # 连续无提升的验证次数
    early_stop = False
    if es_patience > 0:
        log.info(f"Early stopping ON : patience={es_patience} 次验证, "
                 f"min_delta={es_min_delta}")

    # ---- DETR 风格 per-epoch JSONL 日志（对齐 facebook/detr 的 log.txt）----
    # 与文本日志共用同一 run_id 时间戳 → 每次训练一对独立文件，互不混淆
    log_dir = tcfg.get("log_dir") or "./outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    jsonl_path = os.path.join(log_dir, f"log_{log.run_id}.txt")
    loss_weights = {k: float(v) for k, v in cfg.get("loss", {}).items()}
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"DETR-style epoch log → {jsonl_path} "
             f"(trainable params: {n_parameters/1e6:.1f}M)")

    for epoch in range(start_epoch, int(tcfg["epochs"])):
        model.train()
        running = 0.0
        train_sums, train_n = {}, 0   # 本 epoch 各损失分量的累计
        optimizer.zero_grad()
        accum = int(tcfg["grad_accum"])
        n_batches = len(train_loader)
        for it, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.split(":")[0],
                                dtype=amp_dtype, enabled=amp):
                out = model(**batch)
                loss = out.loss / accum
            scaler.scale(loss).backward()
            # Step on full groups OR at the epoch's last batch, so a
            # trailing partial group's gradients aren't silently dropped
            # by the next epoch's optimizer.zero_grad().
            if (it + 1) % accum == 0 or (it + 1) == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                if ema:
                    ema.update(model)
            running += loss.item() * accum
            with torch.no_grad():  # 损失分量累计（用于 epoch 末的 JSONL）
                for k, v in out.loss_dict.items():
                    train_sums[k] = train_sums.get(k, 0.0) + float(v.item())
                train_n += 1
            if log_interval > 0 and (it == 0 or it % log_interval == 0):
                log.info(f"ep{epoch} it{it}/{len(train_loader)} "
                         f"loss={loss.item()*accum:.4f} "
                         f"lr={scheduler.get_last_lr()[0]:.2e}")

        # ---- 组装本 epoch 的 JSONL 记录 ----
        train_means = {k: v / max(1, train_n) for k, v in train_sums.items()}
        stats = {"train_lr": float(scheduler.get_last_lr()[0])}
        stats.update(_scale_loss_stats(train_means, loss_weights, "train"))
        # 缩放后的总损失（与 HF 的 out.loss 同口径：sum(分量 × 权重)）
        loss_keys = {"loss_ce", "loss_bbox", "loss_giou"}
        stats["train_loss"] = sum(
            v * (1.0 if re.sub(r"_\d+$", "", k)
                 in ("class_error", "cardinality_error")
                 else float(loss_weights.get(re.sub(r"_\d+$", "", k), 1.0)))
            for k, v in train_means.items()
            if re.sub(r"_\d+$", "", k) in loss_keys)

        do_val = ((epoch + 1) % val_interval == 0
                  or epoch + 1 == int(tcfg["epochs"]))

        # 每个 epoch 都算 val 损失（EMA 权重，快速 no_grad 前向）；
        # mAP 只在 val_interval 时算（慢）。
        ema_backup = ema.apply_to(model) if ema else None
        val_means = _val_loss_pass(model, val_loader, device, cfg)
        stats.update(_scale_loss_stats(val_means, loss_weights, "test"))
        stats["test_loss"] = sum(
            v * float(loss_weights.get(re.sub(r"_\d+$", "", k), 1.0))
            for k, v in val_means.items()
            if re.sub(r"_\d+$", "", k) in loss_keys)

        # ---- epoch 汇总（控制台唯一一行：trend 一眼可读）----
        # 两组 lr 都打：LambdaLR 为每个参数组各返回一个值，[0]=head 组，
        # [-1]=backbone 组（各乘同一个 lambda，量级差 = lr/backbone_lr）。
        lrs = scheduler.get_last_lr()
        log.info(f"Epoch {epoch} train_loss={running/max(1, n_batches):.4f} "
                 f"test_loss={stats['test_loss']:.4f} "
                 f"lr(head/backbone)={lrs[0]:.2e}/{lrs[-1]:.2e}")

        if do_val:
            # Evaluate (and save best) with EMA weights when EMA is on: the
            # shadow params are what we'd ship, so the metric and best.pth
            # must reflect them. Raw weights resume training afterwards;
            # last.pth always stores the raw training weights.
            metrics = evaluate(model, val_loader, device, cfg)
            score = metrics.get("map_5095", float("nan"))
            log.info(f"Val ep{epoch}: mAP@50-95={score:.4f} "
                     f"mAP@50={metrics.get('map_50', float('nan')):.4f}"
                     + (" (EMA)" if ema else ""))
            # 逐类别 AP@50-95（无 GT 的类别显示 n/a），同时写入 JSONL
            pc_ap = metrics.get("per_class_ap")
            if pc_ap:
                stats["test_per_class_ap"] = pc_ap
                log.info(f"  Per-class AP@50-95 (ep{epoch}):")
                for _name, _ap in pc_ap.items():
                    log.info(f"    {_name:12s}: "
                             + ("n/a (no GT)" if _ap is None else f"{_ap:.4f}"))
            # COCO 标准 12 项顺序；本数据集只有小目标，s/m/l 槽位重复填
            # 全量值（对应原 DETR log 的 test_coco_eval_bbox 字段）
            coco12 = [
                metrics.get("map_5095"), metrics.get("map_50"),
                metrics.get("map_75"), metrics.get("map_small"),
                metrics.get("map_medium"), metrics.get("map_large"),
                metrics.get("mar_1"), metrics.get("mar_10"),
                metrics.get("mar_100"), metrics.get("mar_small"),
                metrics.get("mar_medium"), metrics.get("mar_large"),
            ]
            stats["test_coco_eval_bbox"] = [
                float(x) if x is not None else None for x in coco12]
            prev_best = best
            if score == score and score > best:  # not NaN & improved
                best = score
                save_ckpt(os.path.join(ckpt_dir, "best.pth"),
                          model, optimizer, scheduler, epoch, best, cfg)
                log.info(f"  ↑ new best {best:.4f} → best.pth")
            # ---- 早停计数：提升幅度 ≤ min_delta 视为无提升 ----
            if es_patience > 0:
                if score == score and score > prev_best + es_min_delta:
                    es_bad_evals = 0
                else:
                    es_bad_evals += 1
                    log.info(f"  early-stop: {es_bad_evals}/{es_patience} "
                             f"次验证无显著提升")
                if es_bad_evals >= es_patience:
                    early_stop = True
        if ema is not None:
            ema.restore(model, ema_backup)
            del ema_backup
            torch.cuda.empty_cache() if device == "cuda" else None

        stats["epoch"] = epoch
        stats["n_parameters"] = n_parameters
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(stats) + "\n")

        save_ckpt(os.path.join(ckpt_dir, "last.pth"),
                  model, optimizer, scheduler, epoch, best, cfg)

        if early_stop:
            log.info(f"Early stopping triggered at epoch {epoch}: "
                     f"连续 {es_patience} 次验证 mAP@50-95 提升 "
                     f"≤ {es_min_delta}。best={best:.4f}")
            break

    log.info(f"Training done. Best mAP@50-95 = {best:.4f}")
    log.info(f"Run saved to: {tcfg['ckpt_dir']}")
    log.info(f"Run log     : {log.log_file}")


def _apply_overrides(cfg, overrides):
    """把 `model.arch=tri_swin` 形式的点路径覆盖进 cfg dict.

    值用 yaml.safe_load 解析，所以 `model.tri_fuse_stages=[1,2,3,4]` 会变成
    真正的 list；`model.img_size=640` 会是 int；字符串加不加引号都可以。
    """
    import yaml as _yaml
    for expr in overrides or []:
        if "=" not in expr:
            raise ValueError(f"--set 必须是 key=value 形式，得到: {expr!r}")
        key, val = expr.split("=", 1)
        key = key.strip()
        try:
            parsed = _yaml.safe_load(val)
        except Exception:
            parsed = val  # 解析失败就当纯字符串
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--resume", default=None,
                    help="断点续训：恢复权重+optimizer+scheduler+epoch")
    ap.add_argument("--restart", default=None,
                    help="热重启：只载入权重，新 cosine 周期从头退火")
    ap.add_argument("--set", action="append", default=None,
                    metavar="KEY=VALUE",
                    help="点路径覆盖，可多次指定；值用 yaml 解析："
                         "model.arch=tri_swin, model.tri_fuse_stages=[1,2,3,4]")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    _apply_overrides(cfg, args.set)
    train(cfg, resume=args.resume, restart=args.restart)


if __name__ == "__main__":
    main()

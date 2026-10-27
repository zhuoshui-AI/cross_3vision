"""Misc utilities: seed, config loading, checkpointing, logging."""
import logging
import os
import random
import time
import yaml
import torch


def set_seed(seed=42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging(log_dir=None, name="multimodal_swin_detr", run_id=None):
    """每次运行写独立的时间戳日志文件（train_<run_id>.log），避免多次
    训练的日志混在同一文件里。run_id 挂在 logger.run_id 上，供调用方
    给其他产物（如 JSONL 训练日志）取同名。"""
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True) if log_dir else None
    logger = logging.getLogger(name)
    logger.run_id = run_id
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_dir:
        log_path = os.path.join(log_dir, f"train_{run_id}.log")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.log_file = log_path
    return logger


def save_ckpt(path, model, optimizer=None, scheduler=None, epoch=-1,
              best_metric=None, config=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict() if hasattr(model, "state_dict") else model,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
    }
    torch.save(state, path)


def load_ckpt(path, model=None, optimizer=None, scheduler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    if model is not None and state.get("model") is not None:
        (model.module if hasattr(model, "module") else model).load_state_dict(
            state["model"], strict=False)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


class EMA:
    """Simple parameter EMA. `ema_decay=0` (set in config) disables it."""

    def __init__(self, model, decay=0.9997):
        self.decay = decay
        self.updates = 0
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # Ramp the effective decay up from ~0 (timm-style warmup) so early
        # EMA evals track the training weights instead of the random init.
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    def apply_to(self, model):
        """Swap EMA weights into model for eval. Only shadow params need backup
        — frozen params aren't touched and won't accumulate epoch-over-epoch."""
        backup = {n: p.clone()
                  for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

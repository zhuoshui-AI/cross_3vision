"""Misc utilities: seed, config loading, checkpointing, logging."""
import logging
import os
import random
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


def setup_logging(log_dir=None, name="multimodal_swin_detr"):
    os.makedirs(log_dir, exist_ok=True) if log_dir else None
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_dir:
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
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
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        """Return a state-dict copy with EMA weights swapped in (for eval)."""
        backup = {n: p.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

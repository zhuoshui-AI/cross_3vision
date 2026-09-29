"""rgbir_resnet 架构失败诊断：8 图真实数据过拟合 + 消融对照（服务器 GPU）。

背景（2026-09-28 完整训练 150 epoch 后 mAP@50≈0.0006）：
本地对照实验已定位——原生 HF DETR 在 200 步内 loss 66→17.8、query 多样性
0.11（健康）；我们的模型 500 步仍卡在 28+，100 个 query 塌缩成同一预测
（qstd≈0）。排除项：grad_clip（关掉无效）、GroupNorm 换 BN（换回预训练 BN
无效）、前景先验偏置（清零无效）。头号嫌疑：DepthLateFusion 默认初始化
尺度过大 → 已在 fusion.py 修复（out_proj/FFN 末层零初始化，起点=恒等）。

本脚本在服务器真实数据上做最终验证，一次跑 4 组对照（各 --steps 步）：

  [fixed]    修复后模型（零初始化融合）       —— 期望：loss 快速下降，mini-AP@50 → 1.0
  [nozinit]  旧默认初始化（复现失败）         —— 期望：复现 query 塌缩
  [bypass]   完全旁路深度融合                 —— 期望：与 fixed 接近（假设验证）
  [reference]原生 HF DETR（ResNet50 单尺度）   —— 健康基线

用法（服务器，项目根目录）：
    python -u -m scripts.diag_rgbir_overfit --data-root /path/to/AIC2026_Train_2000

    # 只跑关键的一组（先跑这个，5 分钟内出结论）
    python -u -m scripts.diag_rgbir_overfit --data-root /path/to/train --only fixed
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from scripts.diag_overfit import decode_preds, mini_ap
from src.data import MultiModalDataset, collate_fn
from src.model import build_model
from src.utils import load_yaml_config

from transformers import DetrConfig, DetrForObjectDetection


def make_batch(ds, n, device):
    """取前 n 张图固定成 batch（train=False → 仅确定性 Resize，可真正记忆）。"""
    ds.ids = ds.ids[:n]
    items = [ds[i] for i in range(len(ds))]
    batch = collate_fn(items)
    out = {
        "rgb": batch["pixel_values_rgb"].to(device),
        "ir": batch["pixel_values_ir"].to(device),
        "depth": batch["pixel_values_depth"].to(device),
        "depth_mask": batch["depth_mask"].to(device),
        "pixel_mask": batch["pixel_mask"].to(device),
    }
    labels = [{"class_labels": t["class_labels"].to(device),
               "boxes": t["boxes"].to(device)} for t in batch["labels"]]
    # mini-AP 的 GT（像素 xyxy）
    tgt = []
    sz = out["rgb"].shape[-1]
    for t in labels:
        cx, cy, w, h = t["boxes"][:, 0], t["boxes"][:, 1], t["boxes"][:, 2], t["boxes"][:, 3]
        tgt.append({"boxes": torch.stack(
            [(cx - w/2)*sz, (cy - h/2)*sz, (cx + w/2)*sz, (cy + h/2)*sz], 1
        ).cpu().numpy(), "labels": t["class_labels"].cpu().numpy()})
    return out, labels, tgt


def build_group(mode, cfg, device):
    """按消融模式构建模型。返回 (model, forward_fn(batch, labels), 参数组)。"""
    if mode == "reference":
        config = DetrConfig(
            num_labels=int(cfg["model"]["num_labels"]),
            num_queries=int(cfg["model"]["num_queries"]),
            d_model=int(cfg["model"]["d_model"]),
            encoder_layers=int(cfg["model"]["encoder_layers"]),
            decoder_layers=int(cfg["model"]["decoder_layers"]),
            auxiliary_loss=True,
            id2label={int(k): v for k, v in cfg["id2label"].items()},
            label2id={v: int(k) for k, v in cfg["id2label"].items()},
            use_timm_backbone=False, backbone="resnet50",
            use_pretrained_backbone=False)
        model = DetrForObjectDetection(config).to(device)

        def fwd(b, labels):
            return model(pixel_values=b["rgb"], labels=labels)
    else:
        model = build_model(cfg).to(device)
        enc = model.model.backbone.conv_encoder
        if mode == "nozinit":
            # 复现旧行为：融合模块恢复 PyTorch 默认初始化
            fuse = enc.depth_fuse
            nn.init.xavier_uniform_(fuse.cross_attn.out_proj.weight)
            nn.init.constant_(fuse.cross_attn.out_proj.bias, 0.0)
            for m in fuse.ffn.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
        elif mode == "bypass":
            class _Bypass(nn.Module):
                def forward(self, fused, depth_feat, depth_mask=None):
                    return fused
            enc.depth_fuse = _Bypass()

        def fwd(b, labels):
            return model(pixel_values_rgb=b["rgb"], pixel_values_ir=b["ir"],
                         pixel_values_depth=b["depth"], depth_mask=b["depth_mask"],
                         pixel_mask=b["pixel_mask"], labels=labels)

    backbone_p, head_p = [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = any(s in n_ for s in ("rgbir_backbone.", "depth_encoder."))
        (backbone_p if is_bb else head_p).append(p)
    if not backbone_p:  # reference 模式没有分组
        groups = [{"params": head_p, "lr": 1e-4}]
    else:
        groups = [{"params": head_p, "lr": 1e-4},
                  {"params": backbone_p, "lr": 1e-5}]
    return model, fwd, groups


def run_group(mode, cfg, batch, labels, tgt, device, args):
    print(f"\n{'='*66}\n[{mode}] steps={args.steps}", flush=True)
    torch.manual_seed(42)
    model, fwd, groups = build_group(mode, cfg, device)
    model.train()
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sz = batch["rgb"].shape[-1]
    t0 = time.time()

    for step in range(args.steps + 1):
        out = fwd(batch, labels)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % max(1, args.steps // 15) == 0 or step == args.steps:
            d = out.loss_dict
            qstd = float(out.pred_boxes.detach().std(dim=1).mean())
            with torch.no_grad():
                model.eval()
                o = fwd(batch, None)
                preds = decode_preds(o.logits, o.pred_boxes, sz, conf=0.0)
                ap50 = mini_ap(preds, tgt, 0.5)
                model.train()
            print(f"  step {step:5d}  loss={out.loss.item():8.3f}  "
                  f"ce={d['loss_ce'].item():.3f}  bbox={d['loss_bbox'].item():.3f}  "
                  f"giou={d['loss_giou'].item():.3f}  qstd={qstd:.4f}  "
                  f"mini-AP@50={ap50:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--grad-clip", type=float, default=0.1)
    ap.add_argument("--only", default=None,
                    choices=["fixed", "nozinit", "bypass", "reference"],
                    help="只跑指定组（省时间）；默认四组全跑")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diag] device={device}  data={args.data_root}", flush=True)
    cfg = load_yaml_config(args.config)
    cfg["data"]["img_size"] = args.img_size

    ds = MultiModalDataset(data_root=args.data_root, split_file=None,
                           img_size=args.img_size, train=False,
                           num_classes=int(cfg["model"]["num_labels"]))
    print(f"[diag] 数据集共 {len(ds)} 张，取前 {args.n_imgs} 张过拟合", flush=True)
    batch, labels, tgt = make_batch(ds, args.n_imgs, device)

    modes = (["fixed", "nozinit", "bypass", "reference"] if args.only is None
             else [args.only])
    for mode in modes:
        run_group(mode, cfg, batch, labels, tgt, device, args)

    print("\n判读：")
    print("  fixed 应在几百步内 loss 显著下降、mini-AP@50 → 1.0（则可开完整训练）")
    print("  nozinit 应复现 query 塌缩（qstd≈0、loss 卡平台）→ 证实修复有效")
    print("  bypass ≈ fixed → 证实 DepthLateFusion 默认初始化就是元凶")
    print("  reference = 原生 HF DETR 健康基线，用于对照收敛速度")


if __name__ == "__main__":
    main()

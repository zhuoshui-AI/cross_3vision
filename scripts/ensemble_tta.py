"""P2：TTA + WBF 多 checkpoint 融合评测 / 预测。

能力：
  1) 融合任意多个 ckpt（不同 IR 塔 / 不同训练分辨率 / 不同深度编码均可，
     按 (depth_encoding, img_size) 分组建 loader，框坐标归一化后在同一空间融合）。
  2) TTA：水平翻转（默认开；训练本身有 flip 增强，是最安全的 TTA）+
     可选多尺度（--tta-scales 960,1120）。
  3) WBF（Weighted Box Fusion, Solovyev et al. 2021）逐类别聚类融合：
     每个来源（模型×TTA 变体）每簇至多贡献 1 框；簇内
     box = Σ(w_i·s_i·b_i)/Σ(w_i·s_i)，score = Σ(w_i·s_i)/Σ(w_i)。
     纯 numpy 手工实现，不依赖 ensemble-boxes（服务器免装包）。
  4) eval 模式：在磁盘固定 val 名单（P0 协议）上同时报告每个 ckpt
     单独 mAP 与融合 mAP —— 一眼看出融合增益。
  5) predict 模式：对无标注测试集输出融合预测 txt（提交格式 / 供
     pseudo_label.py 复用）。

用法示例：
  # 双 ckpt 融合评测（固定 val 名单，P0 协议）
  python scripts/ensemble_tta.py \
      --ckpt outputs/ckpt/tri_swin/936e197_.../best.pth \
      --ckpt outputs/ckpt/tri_swin/e048ccf_.../best.pth \
      --splits outputs/ckpt/tri_swin/936e197_.../splits \
      --extra-split outputs/ckpt/tri_swin/<random_run>/splits/val_random.txt \
      --data-root /HUBU-AI027/AIdemo/训练集/AIC2026_Train_2000

  # 融合预测 phase_2（提交件）
  python scripts/ensemble_tta.py --mode predict \
      --ckpt ... --ckpt ... \
      --test-root /HUBU-AI027/AI_competition/data/AIC2026_PHASE_2_1000 \
      --out-dir outputs/predictions_phase2_fused

口径说明：
  - 评测与 eval_splits.py 完全一致：conf_th=0.0（保留全部 query），
    val 名单来自磁盘固定文件，不重跑切分。
  - 各 ckpt 默认用自己的训练分辨率评测；--img-size 可全局覆盖
    （FPN+DETR 对输入尺寸不敏感）。
  - --pre-conf（默认 0.01）只滤掉接近 0 分的 query 再融合：对 mAP 无感
    （这些框成不了 TP），只省融合与 metric 的算力。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# scripts/ 独立入口：把项目根加到 sys.path（与 eval_splits.py 同款）
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import scripts.eval_splits as _es  # noqa: E402  复用 P0 的 ckpt/splits 机制
from src.data import MultiModalDataset, collate_fn  # noqa: E402
from src.inference import write_predictions_txt  # noqa: E402
from src.losses import cxcywh_to_xyxy_pixel  # noqa: E402
from src.model import build_model  # noqa: E402

try:
    from torchmetrics.detection import MeanAveragePrecision
    _HAS_TM = True
except Exception:  # pragma: no cover
    _HAS_TM = False


# ---------------------------------------------------------------------------
# WBF 融合（单图，numpy 实现）
# ---------------------------------------------------------------------------

def _pairwise_iou_xyxy(a, b):
    """(N,4) vs (M,4) xyxy → (N,M) IoU 矩阵。"""
    ax1, ay1 = a[:, 0][:, None], a[:, 1][:, None]
    ax2, ay2 = a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1 = b[:, 0][None, :], b[:, 1][None, :]
    bx2, by2 = b[:, 2][None, :], b[:, 3][None, :]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
    inter = iw * ih
    area_a = np.clip((ax2 - ax1) * (ay2 - ay1), 0.0, None)
    area_b = np.clip((bx2 - bx1) * (by2 - by1), 0.0, None)
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def wbf_fuse(pred_list, weights, iou_thr=0.55):
    """单图 WBF。

    pred_list: [(boxes_xyxy_norm [K,4], scores [K], labels [K])] —
               每个元素是一个"来源"（模型 × TTA 变体）的预测。
    weights:   与 pred_list 对齐的来源权重（模型权重 × 变体权重）。
    返回 (boxes_xyxy_norm [F,4] f32, scores [F] f32, labels [F] i64)，
    按分数降序。
    """
    boxes_all, scores_all, labels_all, src_all, w_all = [], [], [], [], []
    for pi, (b, s, l) in enumerate(pred_list):
        if b is None or len(b) == 0:
            continue
        boxes_all.append(np.asarray(b, dtype=np.float32).reshape(-1, 4))
        scores_all.append(np.asarray(s, dtype=np.float32).reshape(-1))
        labels_all.append(np.asarray(l).reshape(-1).astype(np.int64))
        src_all.append(np.full(len(b), pi, dtype=np.int64))
        w_all.append(np.full(len(b), float(weights[pi]), dtype=np.float32))
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
             np.zeros(0, np.int64))
    if not boxes_all:
        return empty

    B = np.clip(np.concatenate(boxes_all), 0.0, 1.0)
    S = np.concatenate(scores_all)
    L = np.concatenate(labels_all)
    SRC = np.concatenate(src_all)
    W = np.concatenate(w_all)

    out_b, out_s, out_l = [], [], []
    for cls in np.unique(L):
        idx = np.where(L == cls)[0]
        idx = idx[np.argsort(-S[idx])]          # 分数降序 → 贪心簇种子
        K = len(idx)
        if K == 1:
            out_b.append(B[idx[0]])
            out_s.append(float(S[idx[0]]))
            out_l.append(cls)
            continue
        iou = _pairwise_iou_xyxy(B[idx], B[idx])
        used = np.zeros(K, dtype=bool)
        for ai in range(K):
            if used[ai]:
                continue
            cluster = [ai]
            used[ai] = True
            srcs = {int(SRC[idx[ai]])}
            for bj in range(ai + 1, K):
                if used[bj] or iou[ai, bj] < iou_thr:
                    continue
                src = int(SRC[idx[bj]])
                if src in srcs:                 # 每来源每簇至多 1 框
                    continue
                cluster.append(bj)
                used[bj] = True
                srcs.add(src)
            cb, cs, cw = B[idx[cluster]], S[idx[cluster]], W[idx[cluster]]
            wts = cw * cs                       # 坐标按（模型权重×分数）加权
            fb = (cb * wts[:, None]).sum(0) / max(float(wts.sum()), 1e-9)
            fs = float((cw * cs).sum() / max(float(cw.sum()), 1e-9))
            out_b.append(fb)
            out_s.append(fs)
            out_l.append(cls)

    if not out_b:
        return empty
    fb = np.stack(out_b).astype(np.float32)
    fs = np.stack(out_s).astype(np.float32)
    fl = np.stack(out_l).astype(np.int64)
    order = np.argsort(-fs)
    return fb[order], fs[order], fl[order]


# ---------------------------------------------------------------------------
# 模型加载与分组
# ---------------------------------------------------------------------------

class _Entry:
    """一个待融合的 ckpt：模型 + 它自己的有效 config + 融合权重。"""

    def __init__(self, name, model, cfg, weight, device):
        self.name = name
        self.model = model
        self.cfg = cfg
        self.weight = weight
        self.depth_enc = cfg["model"].get("depth_encoding", "linear")
        self.img_size = int(cfg["data"].get("img_size", 640))
        self.model.to(device)


def load_entries(ckpt_paths, cli_cfg, weights, device, img_size_override=None):
    ents = []
    for i, p in enumerate(ckpt_paths):
        state, ckpt_cfg, epoch, best = _es._load_ckpt(p)
        cfg = _es._resolve_cfg(ckpt_cfg, cli_cfg)
        cfg = _es._fill_missing(cfg)
        if img_size_override:
            cfg["data"]["img_size"] = int(img_size_override)
        model = build_model(cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [{Path(p).name}] state_dict mismatch — "
                  f"missing={len(missing)} unexpected={len(unexpected)} "
                  f"(前几个 missing: {list(missing)[:3]})")
        model.eval()
        run_dir = Path(p).parent.name[:28]
        name = f"#{i}:{run_dir}/{Path(p).name}"
        w = float(weights[i]) if i < len(weights) else 1.0
        ent = _Entry(name, model, cfg, w, device)
        ents.append(ent)
        print(f"  loaded [{name}] epoch={epoch} best={best:.4f} "
              f"enc={ent.depth_enc} size={ent.img_size} w={w}")
    return ents


def _make_loader(data_root, ids, img_size, depth_enc, ncls, batch_size,
                 device, scan_source):
    ds = MultiModalDataset(
        data_root=data_root, ids=ids,
        img_size=img_size, train=False, num_classes=ncls,
        depth_encoding=depth_enc, scan_source=scan_source)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=(device == "cuda"))


def _group_loaders(ents, data_root, ids, ncls, batch_size, device,
                   tta_scales, scan_source):
    """按 (depth_enc, img_size) 分组；组内每个尺度建一个 loader。

    返回 [(group_key, scale, loader, [entry...])]：同组同尺度的多个
    ckpt 共享同一个 loader（省 IO）。
    """
    groups = {}
    for e in ents:
        groups.setdefault((e.depth_enc, e.img_size), []).append(e)
    flat = []
    for (enc, size), members in groups.items():
        scales = [size]
        for s in tta_scales:
            if int(s) != size:
                scales.append(int(s))
        for s in scales:
            loader = _make_loader(data_root, ids, s, enc, ncls, batch_size,
                                  device, scan_source)
            flat.append(((enc, size), s, loader, members))
    return flat


# ---------------------------------------------------------------------------
# 前向（含 TTA 翻转）
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward(model, batch, device, flip, amp_dtype, amp):
    rgb = batch["pixel_values_rgb"].to(device)
    ir = batch["pixel_values_ir"].to(device)
    depth = batch["pixel_values_depth"].to(device)
    dmask = batch["depth_mask"].to(device)
    pmask = batch["pixel_mask"].to(device)
    if flip:                                    # 三模态 + 两个 mask 同步翻转
        rgb, ir, depth = rgb.flip(-1), ir.flip(-1), depth.flip(-1)
        dmask, pmask = dmask.flip(-1), pmask.flip(-1)
    with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                        enabled=amp):
        out = model(pixel_values_rgb=rgb, pixel_values_ir=ir,
                    pixel_values_depth=depth, depth_mask=dmask,
                    pixel_mask=pmask, labels=None)
    probs = out.logits.float().softmax(-1)      # (B,Q,C+1)
    scores, labels = probs[..., :-1].max(-1)     # 丢 no-object
    boxes = out.pred_boxes.float()               # (B,Q,4) cxcywh 归一化
    if flip:                                    # 翻回原图坐标系
        cx = boxes[..., 0]
        boxes = torch.stack([1.0 - cx, boxes[..., 1], boxes[..., 2],
                             boxes[..., 3]], dim=-1)
    return boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()


def _pred_xyxy_np(pred, pre_conf, b):
    """取 batch 第 b 张图的预测 → (xyxy_norm, scores, labels)，低分过滤。"""
    bx, sc, lb = pred[0][b], pred[1][b], pred[2][b]
    keep = sc >= pre_conf
    bx, sc, lb = bx[keep], sc[keep], lb[keep]
    if len(bx) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int64))
    xyxy = np.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2,
                     bx[:, 0] + bx[:, 2] / 2, bx[:, 1] + bx[:, 3] / 2], axis=1)
    return xyxy, sc, lb


def _to_pixel_xyxy(boxes_xyxy_norm_np, hw):
    """归一化 xyxy → 像素 xyxy（torch），供 torchmetrics。"""
    H, W = hw
    scale = np.array([W, H, W, H], dtype=np.float32)
    return torch.from_numpy(
        np.clip(boxes_xyxy_norm_np * scale, 0, None).round()
    ).float()


# ---------------------------------------------------------------------------
# 评测 / 预测 共用的批量前向收集
# ---------------------------------------------------------------------------

def _collect_variant_preds(flat, batch_stack, ents, device, args, amp_dtype):
    """对一批（多 loader 对齐的）batch 做全部 (entry, scale, flip) 前向。

    返回 (variant_keys, preds)：
      variant_keys: [(entry_name, entry_weight, scale, flip)]
      preds:       {key: np (boxes_cxcywh, scores, labels)}
    """
    variant_keys, preds = [], {}
    flips = [False, True] if args.tta == "hflip" else [False]
    for (_, scale, _, members), batch in zip(flat, batch_stack):
        for e in members:
            for flip in flips:
                variant_keys.append((e.name, e.weight, scale, flip))
                preds[(e.name, scale, flip)] = _forward(
                    e.model, batch, device, flip, amp_dtype, args.amp)
    return variant_keys, preds


def _targets_from_batch(batch):
    return [{"boxes": t["boxes"], "labels": t["class_labels"]}
            for t in batch["labels"]]


def _summarize_metric(metric, cfg):
    """torchmetrics 结果 → 精简 dict（mAP + per-class AP）。"""
    if not _HAS_TM:
        return {"map_5095": float("nan"), "note": "torchmetrics unavailable"}
    res = metric.compute()

    def _f(key):
        try:
            return float(res[key].item())
        except (KeyError, AttributeError):
            return -1.0

    per_class_ap = None
    try:
        id2label = {int(k): str(v) for k, v in (cfg.get("id2label") or {}).items()}
        classes = torch.atleast_1d(res["classes"]).tolist()
        ap_list = torch.atleast_1d(res["map_per_class"]).tolist()
        per_class_ap = {}
        for cid, ap in zip(classes, ap_list):
            name = id2label.get(int(cid), f"class_{int(cid)}")
            per_class_ap[name] = float(ap) if ap >= 0 else None
    except (KeyError, AttributeError):
        per_class_ap = None
    return {"map_5095": _f("map"), "map_50": _f("map_50"),
            "per_class_ap": per_class_ap}


def _new_metric():
    if not _HAS_TM:
        return None
    return MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox",
        iou_thresholds=[round(v, 2) for v in
                        torch.linspace(0.5, 0.95, 10).tolist()],
        class_metrics=True)


# ---------------------------------------------------------------------------
# eval 模式
# ---------------------------------------------------------------------------

def run_eval(ents, splits, data_root, args, device):
    ncls = int(ents[0].cfg["model"]["num_labels"])
    amp_dtype = torch.bfloat16 if (args.amp and torch.cuda.is_bf16_supported()) \
        else torch.float16
    all_results = {}

    for label, val_ids, kind in splits:
        print(f"\n{'─' * 60}")
        print(f"[fused-eval] split={label} ({kind})  #val={len(val_ids)}  "
              f"data={data_root}")
        flat = _group_loaders(ents, data_root, val_ids, ncls, args.batch_size,
                              device, args.tta_scales, scan_source="labels")
        for _, scale, loader, members in flat:
            print(f"    loader size={loader.dataset.img_size} "
                  f"enc={loader.dataset.depth_encoding} "
                  f"models={[m.name for m in members]}")

        if not _HAS_TM:
            print("[FATAL] 缺 torchmetrics，无法计算 mAP")
            sys.exit(2)

        metric_fused = _new_metric()
        metric_solo = {e.name: _new_metric() for e in ents}

        for batch_stack in zip(*[fl[2] for fl in flat]):
            stems = batch_stack[0]["stems"]
            for bs in batch_stack[1:]:
                if bs["stems"] != stems:
                    raise RuntimeError("loader 之间 stem 顺序不一致（bug）")
            hw = (batch_stack[0]["pixel_values_rgb"].shape[2],
                  batch_stack[0]["pixel_values_rgb"].shape[3])
            B = len(stems)

            variant_keys, preds = _collect_variant_preds(
                flat, batch_stack, ents, device, args, amp_dtype)

            # ---- 融合（全部来源 × TTA 变体）----
            src_weights = [w for (_, w, _, _) in variant_keys]
            fused_preds, targets_px = [], _targets_from_batch(batch_stack[0])
            for b in range(B):
                pl = [_pred_xyxy_np(preds[(n, s, f)], args.pre_conf, b)
                      for (n, _, s, f) in variant_keys]
                fb, fs, fl_ = wbf_fuse(pl, src_weights, args.iou_thr)
                fused_preds.append({
                    "boxes": _to_pixel_xyxy(fb, hw),
                    "scores": torch.from_numpy(fs).float(),
                    "labels": torch.from_numpy(fl_).long(),
                })
            metric_fused.update(fused_preds, [
                {"boxes": cxcywh_to_xyxy_pixel(t["boxes"], hw),
                 "labels": t["labels"]} for t in targets_px])

            # ---- 单模型口径（各自 native 尺度、无翻转）----
            for e in ents:
                pred = preds[(e.name, e.img_size, False)]
                solo = []
                for b in range(B):
                    bx, sc, lb = _pred_xyxy_np(pred, args.pre_conf, b)
                    solo.append({
                        "boxes": _to_pixel_xyxy(bx, hw),
                        "scores": torch.from_numpy(sc).float(),
                        "labels": torch.from_numpy(lb).long(),
                    })
                metric_solo[e.name].update(solo, [
                    {"boxes": cxcywh_to_xyxy_pixel(t["boxes"], hw),
                     "labels": t["labels"]} for t in targets_px])

        # ---- 汇报 ----
        fused = _summarize_metric(metric_fused, ents[0].cfg)
        print(f"\n  == split={label} 结果 ==")
        for e in ents:
            s = _summarize_metric(metric_solo[e.name], e.cfg)
            print(f"  solo  {e.name:<40s} mAP@50-95={s['map_5095']:.4f}  "
                  f"mAP@50={s['map_50']:.4f}")
        print(f"  FUSED {'(WBF over %d sources)' % len(variant_keys):<40s}"
              f" mAP@50-95={fused['map_5095']:.4f}  "
              f"mAP@50={fused['map_50']:.4f}")
        if fused.get("per_class_ap"):
            print("  FUSED Per-class AP@50-95:")
            for name, ap in sorted(fused["per_class_ap"].items(),
                                   key=lambda kv: kv[1] if kv[1] is not None else -1.0):
                v = f"{ap:.4f}" if ap is not None else "—"
                print(f"    {name:<14s}: {v}")

        all_results[label] = {
            "split": label, "kind": kind, "n_val": len(val_ids),
            "fused": fused,
            "solo": {e.name: _summarize_metric(metric_solo[e.name], e.cfg)
                     for e in ents},
        }

    if args.json_out:
        out = {"ckpt": [str(p) for p in args.ckpt],
               "weights": args.weights, "tta": args.tta,
               "tta_scales": args.tta_scales, "iou_thr": args.iou_thr,
               "pre_conf": args.pre_conf, "results": all_results}
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"\n结果已写 → {args.json_out}")


# ---------------------------------------------------------------------------
# predict 模式
# ---------------------------------------------------------------------------

def fused_predict_all(ents, test_root, args, device):
    """对无标注目录做融合推理，返回 {stem: (boxes_cxcywh np[F,4],
    scores np[F], labels np[F])}。pseudo_label.py 复用此接口打伪标签。"""
    ncls = int(ents[0].cfg["model"]["num_labels"])
    amp_dtype = torch.bfloat16 if (args.amp and torch.cuda.is_bf16_supported()) \
        else torch.float16

    # 扫描测试集 stem（visible/ 全有 + 三模态齐全者），各 loader 复用同一份 ids
    scan_loader = _make_loader(test_root, None, ents[0].img_size,
                               ents[0].depth_enc, ncls, 1, device,
                               scan_source="images")
    ids = list(scan_loader.dataset.ids)
    print(f"Test set: {len(ids)} images from {test_root}")
    flat = _group_loaders(ents, test_root, ids, ncls, args.batch_size, device,
                          args.tta_scales, scan_source="images")
    for _, scale, loader, members in flat:
        print(f"    loader size={loader.dataset.img_size} "
              f"enc={loader.dataset.depth_encoding} "
              f"models={[m.name for m in members]}")

    results = {}
    for batch_stack in zip(*[fl[2] for fl in flat]):
        stems = batch_stack[0]["stems"]
        for bs in batch_stack[1:]:
            if bs["stems"] != stems:
                raise RuntimeError("loader 之间 stem 顺序不一致（bug）")
        B = len(stems)
        variant_keys, preds = _collect_variant_preds(
            flat, batch_stack, ents, device, args, amp_dtype)
        src_weights = [w for (_, w, _, _) in variant_keys]
        for b in range(B):
            pl = [_pred_xyxy_np(preds[(n, s, f)], args.pre_conf, b)
                  for (n, _, s, f) in variant_keys]
            fb, fs, fl_ = wbf_fuse(pl, src_weights, args.iou_thr)
            cxcywh = xyxy_to_cxcywh_np(fb)
            results[stems[b]] = (cxcywh, fs, fl_)
    return results


def run_predict(ents, test_root, out_dir, args, device):
    os.makedirs(out_dir, exist_ok=True)
    results = fused_predict_all(ents, test_root, args, device)
    n_boxes = 0
    for stem, (cxcywh, fs, fl_) in results.items():
        top = min(args.max_dets, len(fs))
        cxcywh, fs, fl_ = cxcywh[:top], fs[:top], fl_[:top]
        write_predictions_txt(os.path.join(out_dir, stem + ".txt"),
                              torch.from_numpy(cxcywh).float(),
                              torch.from_numpy(fl_).long(),
                              torch.from_numpy(fs).float())
        n_boxes += int(len(fs))
    print(f"\nFused predict done: {len(results)} images, "
          f"{n_boxes} boxes (avg {n_boxes / max(1, len(results)):.1f}/img) → {out_dir}")


def xyxy_to_cxcywh_np(b):
    """xyxy (N,4) numpy → cxcywh (N,4) numpy。"""
    if len(b) == 0:
        return np.zeros((0, 4), np.float32)
    cx = (b[:, 0] + b[:, 2]) / 2.0
    cy = (b[:, 1] + b[:, 3]) / 2.0
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_splits(args, ckpt_paths):
    """与 eval_splits.py 相同的名单解析：--splits / 自动发现 / --extra-split。"""
    val_ids_by_label, kind_by_label = {}, {}
    splits_dir = None
    meta = None
    if args.splits:
        p = Path(args.splits).resolve()
        if p.is_dir():
            splits_dir = p
        else:
            val_ids_by_label[p.stem] = _es._read_ids(p)
            kind_by_label[p.stem] = "external"
    else:
        splits_dir = _es._discover_splits_dir(ckpt_paths[0])
        if splits_dir is None:
            print("[FATAL] 第一个 ckpt 旁没有 splits/，请用 --splits 指定 val 名单")
            sys.exit(2)
    if splits_dir is not None:
        meta_path = splits_dir / "split_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        main_txt = splits_dir / (args.split_name or "val.txt")
        if not main_txt.exists():
            cands = sorted(splits_dir.glob("val*.txt"))
            if not cands:
                print(f"[FATAL] {splits_dir} 里找不到 val*.txt")
                sys.exit(2)
            main_txt = cands[-1]
            print(f"  未指定 --split-name，自动选了: {main_txt.name}")
        val_ids_by_label[main_txt.stem] = _es._read_ids(main_txt)
        kind_by_label[main_txt.stem] = (meta or {}).get("split_kind", "unknown")
    for extra in args.extra_split:
        q = Path(extra).resolve()
        val_ids_by_label[q.stem] = _es._read_ids(q)
        kind_by_label[q.stem] = "external"
    splits = [(lbl, ids, kind_by_label[lbl])
              for lbl, ids in val_ids_by_label.items()]
    return splits, meta


def main():
    ap = argparse.ArgumentParser(
        description="TTA + WBF 多 ckpt 融合评测 / 预测（P2）")
    ap.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint 路径，可传多次（融合成员）")
    ap.add_argument("--mode", choices=["eval", "predict"], default="eval")
    ap.add_argument("--cfg", default=None,
                    help="yaml 配置（仅当 ckpt 不带 config 时需要；作用于所有 ckpt）")
    ap.add_argument("--weights", default=None,
                    help="各 ckpt 融合权重，逗号分隔，如 1.0,1.2（默认全 1.0）")
    ap.add_argument("--img-size", type=int, default=None,
                    help="全局覆盖评测分辨率（默认各 ckpt 用自己的训练分辨率）")
    ap.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    ap.add_argument("--tta-scales", default="",
                    help="额外 TTA 尺度，逗号分隔（如 960,1120；默认关）")
    # eval 相关
    ap.add_argument("--splits", default=None,
                    help="splits 目录或 val 名单 txt（默认从第一个 ckpt 自动发现）")
    ap.add_argument("--split-name", default=None)
    ap.add_argument("--extra-split", action="append", default=[])
    ap.add_argument("--data-root", default=None,
                    help="覆盖 split_meta.json 里的 data_root")
    ap.add_argument("--json-out", default=None)
    # predict 相关
    ap.add_argument("--test-root", default=None,
                    help="predict 模式：无标注测试集目录（含 visible/...）")
    ap.add_argument("--out-dir", default=None,
                    help="predict 模式：输出 txt 目录")
    ap.add_argument("--max-dets", type=int, default=100)
    # 通用
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--pre-conf", type=float, default=0.01,
                    help="融合前过滤低分 query（不影响 mAP，只省算力）")
    ap.add_argument("--iou-thr", type=float, default=0.55,
                    help="WBF 聚类 IoU 阈值")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    args.amp = not args.no_amp
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) \
        else (args.device if args.device != "auto" else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] 没 GPU，退回 CPU")
        device = "cpu"
    args.amp = args.amp and (device == "cuda")
    args.tta_scales = [int(s) for s in args.tta_scales.split(",") if s.strip()]

    weights = [float(x) for x in args.weights.split(",")] if args.weights else []
    print(f"Device={device}  mode={args.mode}  tta={args.tta}  "
          f"scales={args.tta_scales or '-'}  iou_thr={args.iou_thr}")
    print("Loading checkpoints:")
    ents = load_entries(args.ckpt, args.cfg, weights, device, args.img_size)
    if not ents:
        print("[FATAL] 没有可用的 ckpt")
        sys.exit(2)

    if args.mode == "eval":
        splits, meta = _resolve_splits(args, args.ckpt)
        data_root = _es._pick_data_root(meta, args.data_root)
        if data_root is None:
            print("[FATAL] 请给 --data-root，或让 splits/ 里带 split_meta.json")
            sys.exit(2)
        if not Path(data_root).is_dir():
            print(f"[FATAL] data_root 不存在: {data_root}")
            sys.exit(2)
        run_eval(ents, splits, data_root, args, device)
    else:
        if not args.test_root or not args.out_dir:
            print("[FATAL] predict 模式需要 --test-root 和 --out-dir")
            sys.exit(2)
        if not Path(args.test_root).is_dir():
            print(f"[FATAL] test_root 不存在: {args.test_root}")
            sys.exit(2)
        run_predict(ents, args.test_root, args.out_dir, args, device)


if __name__ == "__main__":
    main()

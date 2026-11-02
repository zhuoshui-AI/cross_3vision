"""模型可解释性与缺陷分析可视化 (DETR cross-attention CAM + Grad-CAM + 错误解剖).

针对本项目 DETR 系模型(推理无全局分类输出, 传统 CNN Grad-CAM 不直接适用),
用三种互补方法回答"模型学到了什么 / GT 框在哪 / 缺陷在哪":

  A. 检测 query 的 cross-attention CAM (DETR 论文原版可视化)
     - decoder 最后一层起的 cross-attention, 头/层平均 → 每个 query 一张
       "模型看哪里"的空间热图; 按 P2~P5 尺度上采样回原图叠加.
     - 附带每个 query 的层级注意力质量分布(小目标是否真的用上了 stride-4).
  B. Grad-CAM 变体(梯度 × 激活)
     - target = top-K query 的类别 logit 之和, 反传到 encoder last_hidden_state,
       逐 token 做 relu(Σ_c grad·act) → 联合敏感图.
  C. 错误解剖(直接量化能力缺陷)
     - FN 细分: 类别混淆 / 低分漏检(有框但分数低) / 定位不准 / 完全漏检(无 query 关注)
     - FP 细分: 重复检测 / 类别混淆 / 定位漂移 / 背景幻觉
     - GT 注意力覆盖率: GT 框内 top-50 query 的最大 attention —— 漏检目标模型是否"看过"
     - 按 COCO 尺寸桶(small/medium/large)交叉统计

产出(默认 outputs/cam_vis/):
  {stem}.png            每图 12 面板诊断图(GT/预测/IR/Depth/CAM/GradCAM/能量图/尺度分布/FN 诊断)
  defect_summary.png    汇总: FN 原因×尺寸桶 / FP 类型 / 每类召回 / GT 覆盖率散点
  summary.json          全部数值明细

用法(服务器, 真实 checkpoint; val 集与训练时完全一致, 同 seed=42 切分):
  python scripts/visualize_cam.py --config configs/default.yaml
  python scripts/visualize_cam.py --ckpt outputs/ckpt/rgbir_swin/best.pth --top-k 12
  python scripts/visualize_cam.py --stems 000003_013_00000085,000016_001_00000001

本地无 checkpoint 时可用 example_data + COCO 热启动权重验证管线(CPU 可跑):
  python scripts/visualize_cam.py --data-root example_data --img-size 256 \
      --set model.rgbir_encoder_scales=[16,32] --val-ratio 1.0 --limit 6 \
      --top-k 4 --no-ckpt --num-workers 0
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# CJK 字体(中文标题防豆腐块; 找不到则回退英文)
# ---------------------------------------------------------------------------
_USE_EN = False

def _setup_cjk_font():
    global _USE_EN
    for name in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
                 "PingFang SC", "Source Han Sans SC", "WenQuanYi Micro Hei"):
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    _USE_EN = True

_setup_cjk_font()

_T = {  # 图表文案(中/英)
    "gt": "GT 标注框 (绿=检出, 黄=漏检)" if not _USE_EN else "GT boxes",
    "pred": "预测框 (蓝=TP, 红=FP)" if not _USE_EN else "Predictions",
    "cam": "query 注意力 CAM" if not _USE_EN else "Query cross-attn CAM",
    "gradcam": "联合 Grad-CAM (top query)" if not _USE_EN else "Joint Grad-CAM",
    "energy": "encoder 特征能量图" if not _USE_EN else "Encoder feature energy",
    "union": "top-50 query 注意力并集 + GT" if not _USE_EN else "Union attention + GT",
    "scales": "层级注意力质量分布" if not _USE_EN else "Per-scale attention mass",
}

_SIZE_BUCKETS = ("small", "medium", "large")  # COCO: <32², 32²~96², >96² px² (输入分辨率下)
_FN_CAUSES = ("类别混淆", "低分漏检", "定位不准", "完全漏检")
_FP_TYPES = ("重复检测", "类别混淆", "定位漂移", "背景幻觉")
_STRIDE_TO_LEVEL = {4: "P2", 8: "P3", 16: "P4", 32: "P5"}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _apply_overrides(cfg, overrides):
    """`model.rgbir_encoder_scales=[16,32]` 形式的点路径覆盖(与 train.py --set 一致)."""
    import yaml
    for expr in overrides or []:
        if "=" not in expr:
            raise ValueError(f"--set 必须是 key=value 形式: {expr!r}")
        key, val = expr.split("=", 1)
        try:
            parsed = yaml.safe_load(val)
        except Exception:
            parsed = val
        node = cfg
        for p in key.strip().split(".")[:-1]:
            node = node.setdefault(p, {})
        node[key.strip().split(".")[-1]] = parsed


def _iou_matrix(a, b):
    """xyxy (N,4)×(M,4) → IoU (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    A = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    B = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = A[:, None] + B[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _size_bucket(box):
    """COCO 尺寸桶(按模型输入分辨率下的像素面积)."""
    area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def _norm_map(x, p99=True):
    """归一化到 [0,1]; p99=True 用 99 分位截断避免单点尖峰吞掉全图."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    hi = np.percentile(x, 99) if p99 else x.max()
    hi = hi if hi > 0 else x.max()
    if hi <= 0:
        return np.zeros_like(x)
    return np.clip(x / hi, 0.0, 1.0)


def _overlay(ax, rgb, cam, title, alpha=0.45, cmap="jet"):
    ax.imshow(rgb)
    if cam is not None:
        ax.imshow(cam, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _draw_boxes(ax, boxes, labels, scores, class_names, status, H, W,
                style="gt"):
    """在 ax 上画框. style='gt': status∈{'tp','fn'}; style='pred': status∈{'tp','fp'}."""
    for i, (box, lab) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        ok = status[i] == "tp"
        if style == "gt":
            ec = "#00c853" if ok else "#ffd600"
            ls, tag = ("-", "") if ok else ("--", " FN")
            lw = 1.8
        else:
            ec = "#2979ff" if ok else "#ff1744"
            ls, tag = "-", (" TP" if ok else " FP")
            lw = 1.5
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, ec=ec, ls=ls, lw=lw))
        txt = class_names[int(lab)] + tag
        if style == "pred":
            txt = f"{class_names[int(lab)]}{tag} {scores[i]:.2f}"
        ax.text(x1, max(8, y1 - 3), txt, fontsize=6.5, color=ec,
                ha="left", va="bottom",
                bbox=dict(fc="black", alpha=0.45, ec="none", pad=0.8))


# ---------------------------------------------------------------------------
# 模型加载 + 探针
# ---------------------------------------------------------------------------
def load_model(cfg, ckpt_path, device, no_ckpt=False):
    from src.model import build_model
    model = build_model(cfg).to(device)
    model.eval()
    if no_ckpt or ckpt_path is None:
        print("  [WARN] 未加载训练 checkpoint —— 使用 COCO 热启动权重演示管线"
              "(定位/注意力是 COCO 先验, 不是本数据集学到的)")
        return model
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  加载 checkpoint: {ckpt_path}")
    print(f"    缺失 {len(missing)} / 多余 {len(unexpected)} 个张量(strict=False)")
    if missing:
        print(f"    缺失示例: {missing[:3]}")
    if len(missing) > 20:
        print("    [WARN] 缺失键过多 —— checkpoint 可能与本 config 架构不匹配!")
    carch = (state.get("config") or {}).get("model", {}).get("arch") \
        if isinstance(state.get("config"), dict) else None
    if carch and carch != cfg["model"].get("arch"):
        print(f"    [WARN] ckpt 内 config arch={carch} 与当前 config "
              f"arch={cfg['model'].get('arch')} 不一致, 结果可能无效")
    return model


class _EncoderProbe:
    """monkey-patch encoder.forward:
    1) 强制 output_attentions=False —— encoder self-attn 对 N=33800 token 走 eager
       会物化 N² 注意力矩阵(单层 ~26GB), 必须保持 SDPA; decoder cross-attn
       (100×N) 走 eager 提取注意力权重, 供 CAM 使用.
    2) stash last_hidden_state 并 retain_grad, 供 Grad-CAM 反传.
    """

    def __init__(self, model):
        self.enc = model.model.encoder
        self._orig = self.enc.forward
        self.enc_out = None

        def fwd(*args, **kwargs):
            kwargs["output_attentions"] = False
            kwargs["output_hidden_states"] = False
            out = self._orig(*args, **kwargs)
            t = out.last_hidden_state
            self.enc_out = t
            if torch.is_grad_enabled() and t.requires_grad:
                t.retain_grad()
            return out

        self.enc.forward = fwd


def _install_layout_probe(model, arch):
    """hook 主干, 记录送入 encoder 的每个尺度的 (stride, h, w) —— 按 token 拼接顺序.
    用于把 (Q, N) 注意力/梯度切回各尺度并上采样到原图."""
    state = {"layout": None}
    if arch == "tri_swin":
        target = model.tri_backbone
        sel = set(model.encoder_stride_set)

        def hook(_m, _i, out):
            pyramids, masks, strides = out
            state["layout"] = [
                (int(s), p.shape[2], p.shape[3])
                for p, m, s in zip(pyramids, masks, strides) if int(s) in sel]
    else:
        # rgbir_swin: hook 融合主干，返回 [(p2,m),...,(p5,m)]
        target = model.model.backbone.conv_encoder
        sel = set(model.encoder_strides)
        all_strides = target.fpn_strides

        def hook(_m, _i, out):
            state["layout"] = [
                (int(s), f.shape[2], f.shape[3])
                for (f, _mask), s in zip(out, all_strides) if int(s) in sel]

    target.register_forward_hook(hook)
    return state


def _scale_split(vec, layout):
    """(N,) → 各尺度 (h,w) 列表."""
    out, i = [], 0
    for _s, h, w in layout:
        n = h * w
        out.append(vec[i:i + n].reshape(h, w))
        i += n
    return out


def _to_image_map(vec, layout, H, W, agg="sum"):
    """(N,) token 向量 → 各尺度上采样到原图后聚合."""
    maps = [cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            for m in _scale_split(vec, layout)]
    maps = np.stack(maps)
    return maps.max(0) if agg == "max" else maps.sum(0)


def _box_token_indices(box, layout):
    """GT 框覆盖的 token 全局索引(各尺度 token 中心落在框内)."""
    x1, y1, x2, y2 = box
    idx, off = [], 0
    for stride, h, w in layout:
        xs = (np.arange(w) + 0.5) * stride
        ys = (np.arange(h) + 0.5) * stride
        gx = np.nonzero((xs >= x1) & (xs < x2))[0]
        gy = np.nonzero((ys >= y1) & (ys < y2))[0]
        if len(gx) and len(gy):
            idx.append(off + (gy[:, None] * w + gx[None, :]).ravel())
        off += h * w
    return np.concatenate(idx) if idx else np.array([], dtype=int)


# ---------------------------------------------------------------------------
# 检测结果与错误解剖
# ---------------------------------------------------------------------------
def analyze_image(stem, gt_boxes, gt_labels, all_boxes, all_labels, all_scores,
                  conf_th, iou_th=0.5):
    """对一张图做匹配 + 错误解剖.

    all_* 为全部 100 个 query 的结果(无阈值), 阈值内预测 = score>=conf_th 的子集.
    返回 dict: 计数、FP 类型、每个 GT 的状态与 FN 原因(在全 query 空间诊断)。
    """
    n_gt = len(gt_boxes)
    order = np.argsort(-all_scores)
    pred_mask = all_scores >= conf_th
    p_idx = [i for i in order if pred_mask[i]]          # 阈值内, 按分数降序
    pred_boxes = all_boxes[p_idx] if len(p_idx) else np.zeros((0, 4))
    pred_labels = all_labels[p_idx]
    pred_scores = all_scores[p_idx]

    # ---- 同类贪心匹配 (IoU>=iou_th) ----
    gt_status = ["fn"] * n_gt
    pred_status = ["fp"] * len(p_idx)
    if n_gt and len(p_idx):
        ious = _iou_matrix(pred_boxes, gt_boxes)
        for pi in range(len(p_idx)):
            best, bg = 0.0, -1
            for gi in range(n_gt):
                if gt_status[gi] == "tp" or gt_labels[gi] != pred_labels[pi]:
                    continue
                if ious[pi, gi] > best:
                    best, bg = ious[pi, gi], gi
            if best >= iou_th:
                gt_status[bg] = "tp"
                pred_status[pi] = "tp"

    # ---- FP 细分 ----
    fp_types = []
    if n_gt and len(p_idx):
        ious_all = _iou_matrix(pred_boxes, gt_boxes)
        for pi in range(len(p_idx)):
            if pred_status[pi] == "tp":
                continue
            best_any_g = int(ious_all[pi].argmax())
            best_any = float(ious_all[pi, best_any_g])
            same = [ious_all[pi, g] for g in range(n_gt)
                    if gt_labels[g] == pred_labels[pi]]
            best_same = max(same) if same else 0.0
            if best_same >= iou_th:          # 框对了但同类 GT 已被占用
                fp_types.append("重复检测")
            elif best_any >= iou_th:         # 框住了别的类的目标
                fp_types.append("类别混淆")
            elif best_any >= 0.3:
                fp_types.append("定位漂移")
            else:
                fp_types.append("背景幻觉")
    else:
        fp_types = ["背景幻觉"] * int((len(p_idx)) - sum(
            s == "tp" for s in pred_status))

    # ---- FN 细分(在全 query 空间: 低分/错类/定位差都能暴露出来) ----
    fn_info = [None] * n_gt
    if n_gt:
        ious_q = _iou_matrix(gt_boxes, all_boxes) if len(all_boxes) \
            else np.zeros((n_gt, 0))
        for gi in range(n_gt):
            if gt_status[gi] == "tp":
                fn_info[gi] = {"cause": "tp"}
                continue
            same_q = [q for q in range(len(all_boxes))
                      if all_labels[q] == gt_labels[gi]]
            best_same_iou = max((ious_q[gi, q] for q in same_q), default=0.0)
            best_same_q = max(same_q, key=lambda q: ious_q[gi, q],
                              default=-1) if same_q else -1
            best_any_iou = float(ious_q[gi].max()) if ious_q[gi].size else 0.0
            if best_any_iou >= iou_th:
                cause = "类别混淆"          # 有 query 框住它但类别错了
            elif best_same_iou >= iou_th:
                cause = "低分漏检"          # 有同类好框但分数不过阈值
            elif best_same_iou >= 0.3:
                cause = "定位不准"
            else:
                cause = "完全漏检"
            fn_info[gi] = {
                "cause": cause,
                "best_same_iou": round(float(best_same_iou), 4),
                "best_any_iou": round(best_any_iou, 4),
                "best_query_score": round(float(all_scores[best_same_q]), 4)
                if best_same_q >= 0 else 0.0,
            }

    rec = {
        "stem": stem,
        "gt_boxes": gt_boxes, "gt_labels": gt_labels, "gt_status": gt_status,
        "pred_boxes": pred_boxes, "pred_labels": pred_labels,
        "pred_scores": pred_scores, "pred_status": pred_status,
        "n_tp": gt_status.count("tp"),
        "n_fn": gt_status.count("fn"),
        "n_fp": pred_status.count("fp"),
        "fp_types": fp_types,
        "fn_info": fn_info,
        "gt_buckets": [_size_bucket(b) for b in gt_boxes],
    }
    return rec


# ---------------------------------------------------------------------------
# 前向
# ---------------------------------------------------------------------------
def _forward_batch(model, batch, device, need_attn=False):
    keys = ("pixel_values_rgb", "pixel_values_ir", "pixel_values_depth",
            "depth_mask", "pixel_mask")
    tensors = {k: batch[k].to(device) for k in keys}
    ctx = torch.enable_grad() if need_attn else torch.no_grad()
    with ctx:
        out = model(**tensors, labels=None,
                    output_attentions=need_attn,
                    output_hidden_states=False)
    H, W = tensors["pixel_values_rgb"].shape[-2:]
    probs = out.logits.softmax(-1)
    scores, labels = probs[..., :-1].max(-1)     # (1,Q) 掉 no-object
    boxes = out.pred_boxes[0]                    # (Q,4) cxcywh norm
    from src.losses import cxcywh_to_xyxy_pixel
    boxes_xyxy = cxcywh_to_xyxy_pixel(boxes.detach(), (H, W)).cpu().numpy()
    res = {
        "scores": scores[0].detach().cpu().numpy(),
        "labels": labels[0].detach().cpu().numpy(),
        "boxes": boxes_xyxy,
        "H": H, "W": W,
        "cross_attn": None,
    }
    if need_attn:
        # (L,1,heads,Q,N) → 头/层平均 → (Q,N)
        attn = torch.stack([c[0].mean(0) for c in out.cross_attentions])
        res["cross_attn"] = attn.mean(0).detach().cpu().numpy()
        res["enc_out"] = out.encoder_last_hidden_state
        res["logits_raw"] = out.logits  # Grad-CAM 的 target 从同一张计算图取
    gt = batch["labels"][0]
    res["gt_boxes"] = cxcywh_to_xyxy_pixel(
        gt["boxes"].detach(), (H, W)).cpu().numpy()
    res["gt_labels"] = gt["class_labels"].cpu().numpy()
    return res


# ---------------------------------------------------------------------------
# 全量面板绘制
# ---------------------------------------------------------------------------
def visualize_full(model, probe, layout_state, batch, device, class_names,
                   out_dir, args):
    stem = batch["stems"][0]
    need_attn = True
    fwd = _forward_batch(model, batch, device, need_attn=need_attn)
    layout = layout_state["layout"]
    assert layout is not None, "layout probe 未触发(架构不支持?)"
    H, W = fwd["H"], fwd["W"]

    rec = analyze_image(stem, fwd["gt_boxes"], fwd["gt_labels"],
                        fwd["boxes"], fwd["labels"], fwd["scores"],
                        args.conf_th)

    attn = fwd["cross_attn"]               # (Q,N)
    Q, N = attn.shape
    order = np.argsort(-fwd["scores"])

    # ---- CAM query 选择: 阈值内优先, 不足补 top 分数 ----
    cam_idx = [i for i in order if fwd["scores"][i] >= args.conf_th][:args.n_cam]
    for i in order:
        if len(cam_idx) >= args.n_cam:
            break
        if i not in cam_idx:
            cam_idx.append(i)

    # ---- Grad-CAM: top query 的类别 logit 之和 → 反传到 encoder token ----
    # forward 在 enable_grad 下执行, logits_raw 与 probe.enc_out 同属一张
    # 计算图, 单次 backward 即可拿到逐 token 的 梯度×激活。
    grad_tok = None
    if not args.no_gradcam and len(cam_idx):
        logits = fwd["logits_raw"]
        target = logits[0, cam_idx, fwd["labels"][cam_idx]].sum()
        probe.enc_out.grad = None
        target.backward()
        g = probe.enc_out.grad[0]                 # (N, d) 梯度
        a = probe.enc_out.detach()[0]              # (N, d) 激活
        grad_tok = torch.relu((g * a).sum(-1)).cpu().numpy()

    # ---- 图像底片 ----
    rgb_t = batch["pixel_values_rgb"][0]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    rgb = (rgb_t * torch.from_numpy(std) + torch.from_numpy(mean)).numpy()
    rgb = (np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    ir = batch["pixel_values_ir"][0, 0].numpy()
    depth = batch["pixel_values_depth"][0, 0].numpy()
    dmask = batch["depth_mask"][0].numpy()
    depth_show = np.where(dmask, depth, np.nan)

    # ---- 注意力派生图 ----
    top50 = order[:50]
    union = attn[top50].max(0)                       # (N,) 全 query 并集
    union_map = _norm_map(_to_image_map(union, layout, H, W, agg="max"))
    energy = fwd["enc_out"][0].detach().pow(2).sum(-1).cpu().numpy()
    energy_map = _norm_map(_to_image_map(energy, layout, H, W, agg="sum"))
    gradcam_map = _norm_map(_to_image_map(grad_tok, layout, H, W)) \
        if grad_tok is not None else None

    # ---- GT 覆盖率 ----
    coverage = []
    for gi, box in enumerate(fwd["gt_boxes"]):
        tidx = _box_token_indices(box, layout)
        cov = float(attn[top50][:, tidx].max()) if len(tidx) else 0.0
        coverage.append(cov)
        if rec["fn_info"][gi]:
            rec["fn_info"][gi]["attn_coverage"] = round(cov, 6)
    rec["attn_coverage"] = coverage

    # ---- 每尺度注意力质量(top query) ----
    scale_mass = {q: [] for q in cam_idx}
    counts = [h * w for _s, h, w in layout]
    for q in cam_idx:
        seg, off = [], 0
        for c in counts:
            seg.append(float(attn[q, off:off + c].sum()))
            off += c
        tot = sum(seg) + 1e-12
        scale_mass[q] = [s / tot for s in seg]

    # ---- 画布 ----
    fig, axes = plt.subplots(3, 4, figsize=(19, 14.5))
    n_tp, n_fp, n_fn = rec["n_tp"], rec["n_fp"], rec["n_fn"]
    fig.suptitle(
        f"{stem}   TP={n_tp}  FP={n_fp}  FN={n_fn}   "
        f"(conf≥{args.conf_th}, IoU≥0.5; 输入 {W}×{H}; 尺度 "
        f"{'/'.join(str(s) for s, _h, _w in layout)})",
        fontsize=13)

    ax = axes[0][0]
    ax.imshow(rgb)
    _draw_boxes(ax, rec["gt_boxes"], rec["gt_labels"], None, class_names,
                rec["gt_status"], H, W, "gt")
    ax.set_title(_T["gt"], fontsize=10)
    ax.axis("off")

    ax = axes[0][1]
    ax.imshow(rgb)
    _draw_boxes(ax, rec["pred_boxes"], rec["pred_labels"], rec["pred_scores"],
                class_names, rec["pred_status"], H, W, "pred")
    ax.set_title(_T["pred"], fontsize=10)
    ax.axis("off")

    ax = axes[0][2]
    ax.imshow(ir, cmap="gray")
    ax.set_title("红外输入 (单通道)" if not _USE_EN else "IR")
    ax.axis("off")

    ax = axes[0][3]
    cmap_d = plt.get_cmap("gray").copy()
    cmap_d.set_bad("#c62828")
    ax.imshow(depth_show, cmap=cmap_d, vmin=0, vmax=1)
    ax.set_title("深度输入 (红=无效)" if not _USE_EN else "Depth (red=invalid)")
    ax.axis("off")

    # Row 2: per-query CAM(最多 3 个)+ Grad-CAM
    for k in range(4):
        ax = axes[1][k]
        if k < min(len(cam_idx), 3):
            q = int(cam_idx[k])
            cam_map = _norm_map(_to_image_map(attn[q], layout, H, W))
            _overlay(ax, rgb, cam_map,
                     f"{_T['cam']} q{q}: {class_names[fwd['labels'][q]]} "
                     f"{fwd['scores'][q]:.2f}")
            box = fwd["boxes"][q]
            ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0],
                                   box[3] - box[1], fill=False,
                                   ec="#00e5ff", lw=1.6))
        elif k == 3 and gradcam_map is not None:
            _overlay(ax, rgb, gradcam_map, _T["gradcam"], cmap="inferno")
        else:
            ax.imshow(rgb)
            ax.set_title("(无更多高分 query)" if not _USE_EN else "(none)")
            ax.axis("off")

    # Row 3: 尺度分布 / 能量图 / 并集图 / FN 诊断
    ax = axes[2][0]
    if len(layout) > 1:
        x = np.arange(len(layout))
        w = 0.8 / max(1, len(cam_idx))
        for j, q in enumerate(cam_idx):
            ax.bar(x + j * w, scale_mass[q], w,
                   label=f"q{q}:{class_names[fwd['labels'][q]][:6]} "
                         f"{fwd['scores'][q]:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{_STRIDE_TO_LEVEL.get(s, f'x{s}')}\nstride {s}"
             for s, _h, _w in layout], fontsize=8)
        ax.set_ylabel("注意力质量占比" if not _USE_EN else "attn mass")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "单尺度架构\n(无层级分布)", ha="center", va="center",
                fontsize=12)
    ax.set_title(_T["scales"], fontsize=10)

    _overlay(axes[2][1], rgb, energy_map, _T["energy"], cmap="inferno")
    ax = axes[2][2]
    _overlay(ax, rgb, union_map, _T["union"])
    _draw_boxes(ax, rec["gt_boxes"], rec["gt_labels"], None, class_names,
                rec["gt_status"], H, W, "gt")

    ax = axes[2][3]
    ax.axis("off")
    lines = [f"FN 诊断 (GT {len(fwd['gt_boxes'])} 个):"]
    for gi, info in enumerate(rec["fn_info"]):
        b = rec["gt_buckets"][gi]
        nm = class_names[int(fwd["gt_labels"][gi])]
        if info["cause"] == "tp":
            lines.append(f"  {nm:<10s}[{b:<6s}] 检出")
        else:
            lines.append(
                f"  {nm:<10s}[{b:<6s}] {info['cause']}  "
                f"bestIoU={info['best_same_iou']:.2f} "
                f"q分={info['best_query_score']:.2f} "
                f"cov={info.get('attn_coverage', 0):.3f}")
    if rec["fp_types"]:
        cnt = defaultdict(int)
        for t in rec["fp_types"]:
            cnt[t] += 1
        lines.append("FP 诊断: " + ", ".join(f"{k}×{v}" for k, v in cnt.items()))
    if len(fwd["gt_boxes"]) == 0:
        lines.append("  (无标注)")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)

    # numpy → json 可序列化
    rec_j = dict(rec)
    for k in ("gt_boxes", "pred_boxes"):
        rec_j[k] = np.asarray(rec_j[k]).round(1).tolist()
    for k in ("gt_labels", "pred_labels", "pred_scores"):
        rec_j[k] = np.asarray(rec_j[k]).tolist()
    rec_j["top_cam_queries"] = [
        {"query": int(q), "label": int(fwd["labels"][q]),
         "score": round(float(fwd["scores"][q]), 4),
         "scale_mass": [round(m, 4) for m in scale_mass[q]]}
        for q in cam_idx]
    return path, rec_j


# ---------------------------------------------------------------------------
# 数据集构建(与 train.py 完全一致的 val 切分: torch.Generator seed 42)
# ---------------------------------------------------------------------------
def build_val_dataset(cfg):
    from src.data import MultiModalDataset
    img_size = int(cfg["data"]["img_size"])
    ncls = int(cfg["model"]["num_labels"])

    def resolve(split_key):
        data_root = cfg["data"].get("data_root")
        sp = cfg["data"].get(split_key)
        if sp and os.path.isabs(sp) and os.path.isdir(sp):
            return sp, None                      # split 本身就是数据目录
        if sp:
            cand = sp if os.path.isabs(sp) else os.path.join(data_root, sp)
            if os.path.isfile(cand):
                return data_root, cand           # 正常 root + split 文件
        return data_root, None                   # split 无效 → 整目录

    train_root, train_split = resolve("train_split")
    val_root, val_split = resolve("val_split")
    if (val_root and val_split
            and os.path.abspath(val_root) != os.path.abspath(train_root)):
        ds = MultiModalDataset(val_root, split_file=val_split,
                               img_size=img_size, train=False, num_classes=ncls)
        print(f"  独立 val 目录: {val_root} ({len(ds)} 张)")
        return ds
    if train_root is None:
        raise ValueError("data.train_split / data.data_root 未配置或不存在")
    full = MultiModalDataset(train_root, split_file=train_split,
                             img_size=img_size, train=True, num_classes=ncls)
    n_total = len(full)
    ratio = float(cfg["data"].get("val_ratio", 0.1))
    n_val = max(1, int(round(n_total * ratio)))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(42)      # 与 train.py 完全一致
    perm = torch.randperm(n_total, generator=g).tolist()
    val_ids = [full.ids[i] for i in perm[n_train:]]
    ds = MultiModalDataset(train_root, ids=val_ids, img_size=img_size,
                           train=False, num_classes=ncls)
    print(f"  从训练集切出 val: {n_total} 张 → val {len(ds)} 张 "
          f"(val_ratio={ratio}, seed=42, 与训练时一致)")
    return ds


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def plot_summary(quick_recs, full_recs, class_names, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # (1) FN 原因 × 尺寸桶
    ax = axes[0][0]
    mat = np.zeros((len(_FN_CAUSES), len(_SIZE_BUCKETS)), dtype=int)
    for rec in quick_recs:
        for info, bucket in zip(rec["fn_info"], rec["gt_buckets"]):
            if info and info["cause"] in _FN_CAUSES:
                mat[_FN_CAUSES.index(info["cause"]),
                   _SIZE_BUCKETS.index(bucket)] += 1
    bottom = np.zeros(len(_SIZE_BUCKETS))
    colors = ["#e53935", "#fb8c00", "#fdd835", "#8e24aa"]
    for r, name in enumerate(_FN_CAUSES):
        ax.bar(_SIZE_BUCKETS, mat[r], bottom=bottom, label=name,
               color=colors[r])
        bottom += mat[r]
    ax.set_title("FN 原因 × 目标尺寸桶" if not _USE_EN
                else "FN cause by size")
    ax.set_xlabel("COCO 尺寸桶 (输入分辨率下)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (2) FP 类型
    ax = axes[0][1]
    cnt = defaultdict(int)
    for rec in quick_recs:
        for t in rec["fp_types"]:
            cnt[t] += 1
    vals = [cnt.get(t, 0) for t in _FP_TYPES]
    bars = ax.bar(_FP_TYPES, vals,
                  color=["#5e35b1", "#00897b", "#f4511e", "#37474f"])
    ax.set_ylim(0, max(1, max(vals) * 1.15))
    for b, v in zip(bars, vals):
        if v:
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center",
                    va="bottom", fontsize=9)
    if not any(vals):
        ax.text(0.5, 0.55, "无 FP (模型极保守或召回塌缩)" if not _USE_EN
                else "no FP", ha="center", fontsize=10, color="gray")
    ax.set_title("FP 类型构成" if not _USE_EN else "FP composition")
    ax.grid(axis="y", alpha=0.3)

    # (3) 每类召回率
    ax = axes[1][0]
    n = len(class_names)
    total = np.zeros(n, dtype=int)
    det = np.zeros(n, dtype=int)
    for rec in quick_recs:
        for lab, st in zip(rec["gt_labels"], rec["gt_status"]):
            total[int(lab)] += 1
            det[int(lab)] += st == "tp"
    recall = np.where(total > 0, det / np.maximum(total, 1), np.nan)
    x = np.arange(n)
    ax.bar(x, np.nan_to_num(recall), color="#43a047",
           label="检出率@conf阈值" if not _USE_EN else "recall")
    for i, (t, d) in enumerate(zip(total, det)):
        if t:
            ax.text(i, recall[i] + 0.02, f"{d}/{t}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("每类召回率 (IoU≥0.5)" if not _USE_EN
                 else "Per-class recall")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (4) GT 覆盖率散点(仅全量可视化的图)
    ax = axes[1][1]
    xs, ys, cs = [], [], []
    for rec in full_recs:
        for box, st, cov, bucket in zip(rec["gt_boxes"], rec["gt_status"],
                                        rec["attn_coverage"], rec["gt_buckets"]):
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
            xs.append(area)
            ys.append(cov)
            cs.append("#2e7d32" if st == "tp" else "#c62828")
    if xs:
        ax.scatter(xs, ys, c=cs, s=26, alpha=0.75)
        ax.set_xscale("log")
        ax.set_xlabel("GT 面积 px^2 (log)")
        ax.set_ylabel("框内最大注意力 (top-50 query)")
        ax.set_title("模型是否“看过”每个 GT: 绿=检出 红=漏检" if not _USE_EN
                     else "Attention coverage per GT")
        ax.axhline(np.median(ys), ls="--", c="gray", lw=0.8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "无 GT", ha="center", va="center")

    fig.tight_layout()
    path = os.path.join(out_dir, "defect_summary.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="DETR cross-attention CAM / Grad-CAM / 错误解剖可视化")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint 路径; 默认自动找 "
                         "outputs/ckpt/{arch}/best.pth → last.pth")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="允许无 checkpoint(COCO 热启动权重演示管线)")
    ap.add_argument("--data-root", default=None,
                    help="数据集目录(visible/infrared/depth/labels), "
                         "覆盖 config")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--val-ratio", type=float, default=None,
                    help="覆盖 val 切分比例(默认与训练一致 0.1)")
    ap.add_argument("--set", action="append", default=None,
                    metavar="KEY=VALUE", help="同 train.py 的点路径覆盖")
    ap.add_argument("--stems", default=None,
                    help="逗号分隔的图片 stem 列表; 指定后跳过 quick-pass 直接全量可视化")
    ap.add_argument("--top-k", type=int, default=12,
                    help="quick-pass 后选错误最多的 K 张做全量可视化")
    ap.add_argument("--include-best", type=int, default=2,
                    help="额外加入错误最少的 N 张作对照")
    ap.add_argument("--limit", type=int, default=0,
                    help="quick-pass 最多评估多少张(0=全部 val)")
    ap.add_argument("--conf-th", type=float, default=0.35)
    ap.add_argument("--n-cam", type=int, default=3,
                    help="每图画几个 query 的 attention CAM")
    ap.add_argument("--no-gradcam", action="store_true")
    ap.add_argument("--out-dir", default="outputs/cam_vis")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from src.utils import load_yaml_config
    from src.data import collate_fn

    cfg = load_yaml_config(args.config)
    if args.set:
        _apply_overrides(cfg, args.set)
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root   # 与 train.py --data-root 一致
    if args.img_size:
        cfg["data"]["img_size"] = args.img_size
    if args.val_ratio is not None:
        cfg["data"]["val_ratio"] = args.val_ratio

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    arch = cfg["model"].get("arch", "rgbir_swin")
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    class_names = [id2label[i] for i in range(int(cfg["model"]["num_labels"]))]
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- checkpoint 自动定位 ----
    ckpt = args.ckpt
    if ckpt is None and not args.no_ckpt:
        base = (cfg.get("train") or {}).get("ckpt_dir") or "./outputs/ckpt"
        for cand in ("best.pth", "last.pth"):
            p = os.path.join(base, arch, cand)
            if os.path.isfile(p):
                ckpt = p
                break
        if ckpt is None:
            raise FileNotFoundError(
                f"在 {base}/{arch}/ 下未找到 best.pth / last.pth。"
                f"请用 --ckpt 指定, 或 --no-ckpt 走演示管线")

    print(f"架构: {arch} | 设备: {device} | 阈值 conf≥{args.conf_th}")
    model = load_model(cfg, ckpt, device, no_ckpt=args.no_ckpt)

    probe = _EncoderProbe(model)
    layout_state = _install_layout_probe(model, arch)

    ds = build_val_dataset(cfg)
    idx_map = {s: i for i, s in enumerate(ds.ids)}

    # ---- 选择目标图: --stems 直接指定; 否则 quick-pass 全 val 排序 ----
    if args.stems:
        wanted = {s.strip() for s in args.stems.split(",") if s.strip()}
        missing = wanted - set(ds.ids)
        if missing:
            print(f"  [WARN] 以下 stem 不在 val 集中: {sorted(missing)}")
        stems = [s for s in ds.ids if s in wanted]
    else:
        print(f"\n[1/3] quick-pass: 评估 {min(len(ds), args.limit or len(ds))} 张 "
              f"(无注意力, 仅预测+错误解剖)...")
        quick_recs = []
        for i, stem in enumerate(ds.ids):
            if args.limit and i >= args.limit:
                break
            batch = collate_fn([ds[i]])
            fwd = _forward_batch(model, batch, device, need_attn=False)
            rec = analyze_image(
                stem, fwd["gt_boxes"], fwd["gt_labels"],
                fwd["boxes"], fwd["labels"], fwd["scores"], args.conf_th)
            quick_recs.append(rec)
        if not quick_recs:
            raise RuntimeError("val 集为空, 无可评估图片")
        ranked = sorted(
            quick_recs,
            key=lambda r: (2 * r["n_fn"] + r["n_fp"], sum(
                0 if info["cause"] == "tp" else 1 for info in r["fn_info"])),
            reverse=True)
        stems = [r["stem"] for r in ranked[:args.top_k]]
        best = [r["stem"] for r in ranked[-args.include_best:]] \
            if args.include_best > 0 else []
        stems += [s for s in best if s not in stems]
        print(f"  错误最多: {[(r['stem'], r['n_tp'], r['n_fp'], r['n_fn']) for r in ranked[:3]]}")
        total_gt = sum(len(r["gt_boxes"]) for r in quick_recs)
        total_tp = sum(r["n_tp"] for r in quick_recs)
        print(f"  val 总体: GT {total_gt} 个, 检出 {total_tp} "
              f"({100*total_tp/max(total_gt,1):.1f}%), "
              f"FP 共 {sum(r['n_fp'] for r in quick_recs)}")

    # ---- 全量可视化 ----
    print(f"\n[2/3] 全量可视化 {len(stems)} 张 "
          f"(cross-attention CAM + Grad-CAM + 覆盖率)...")
    full_recs, saved = [], []
    for stem in stems:
        batch = collate_fn([ds[idx_map[stem]]])
        path, rec_j = visualize_full(model, probe, layout_state, batch,
                                     device, class_names, args.out_dir, args)
        saved.append(path)
        full_recs.append(rec_j)
        print(f"  {stem}: TP={rec_j['n_tp']} FP={rec_j['n_fp']} "
              f"FN={rec_j['n_fn']} → {os.path.basename(path)}")

    # ---- 汇总 ----
    print(f"\n[3/3] 生成汇总报告...")
    if args.stems:
        plot_recs = full_recs
    else:
        by_stem = {r["stem"]: r for r in full_recs}
        plot_recs = [by_stem.get(r["stem"], r) for r in quick_recs]
    sum_path = plot_summary(plot_recs, full_recs, class_names, args.out_dir)

    summary = {
        "config": {
            "ckpt": ckpt, "arch": arch, "conf_th": args.conf_th,
            "img_size": int(cfg["data"]["img_size"]),
            "iou_th": 0.5,
            "encoder_scales": [s for s, _h, _w in (layout_state["layout"] or [])],
        },
        "quick_pass": [
            {"stem": r["stem"], "n_tp": r["n_tp"], "n_fp": r["n_fp"],
             "n_fn": r["n_fn"], "fp_types": r["fp_types"],
             "gt_buckets": r["gt_buckets"],
             "fn_causes": [i["cause"] for i in r["fn_info"]]}
            for r in (quick_recs if not args.stems else full_recs)],
        "full_visualization": full_recs,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    print(f"\n{'=' * 62}")
    print(f"完成: {len(saved)} 张诊断图 + 汇总")
    print(f"  目录: {args.out_dir}/")
    print(f"  汇总: {os.path.basename(sum_path)} + summary.json")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()

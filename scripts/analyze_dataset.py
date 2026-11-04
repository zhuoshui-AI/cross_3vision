"""数据集异质性分析脚本 — 纯 CPU，只需 labels/*.txt，不需要图像。

用途：
  1. 类别分布 + 不均衡程度（熵、max/min 比）
  2. 框尺寸分布 + COCO 尺度分桶（tiny/small/medium/large）
  3. 每图目标数分布（检测 empty image 比例）
  4. val 集 bootstrap 置信区间 → 量化 min_delta 该设多大

用法（在有数据的机器上跑）：
  python scripts/analyze_dataset.py --data-root /path/to/AIC2026_Train_2000
  python scripts/analyze_dataset.py --data-root /path/to/AIC2026_Train_2000 \
      --val-ratio 0.1 --num-bootstrap 2000 --num-classes 12
"""
import argparse
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def parse_label_file(txt_path):
    """YOLO 格式: cls cx cy w h (归一化, 每行一个框)。"""
    boxes, labels = [], []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = float(parts[1]), float(parts[2]), \
                    float(parts[3]), float(parts[4])
                labels.append(cls)
                boxes.append((cx, cy, w, h))
    except Exception:
        pass
    return boxes, labels


def load_all_labels(data_root, num_classes):
    """扫描 labels/ 下所有 .txt，返回 per-image 列表。"""
    label_dir = os.path.join(data_root, "labels")
    if not os.path.isdir(label_dir):
        print(f"[ERROR] labels dir not found: {label_dir}")
        sys.exit(1)
    results = []  # [(stem, [(cx,cy,w,h), ...], [cls, ...])]
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".txt"):
            continue
        stem = fname[:-4]
        boxes, labels = parse_label_file(os.path.join(label_dir, fname))
        # 过滤越界
        valid = []
        for (cx, cy, w, h), cls in zip(boxes, labels):
            if 0 <= cls < num_classes and w > 0 and h > 0 \
                    and 0 < cx < 1 and 0 < cy < 1:
                valid.append(((cx, cy, w, h), cls))
        results.append((stem,
                        [v[0] for v in valid],
                        [v[1] for v in valid]))
    return results


def analyze_class_distribution(all_data, num_classes):
    """统计类别频次 + 不均衡指标。"""
    counter = Counter()
    for _, _, labels in all_data:
        counter.update(labels)
    total = sum(counter.values())
    if total == 0:
        return counter, total, {}

    # 熵：分布越均匀越接近 log(K)，越集中越接近 0
    probs = np.array([counter.get(c, 0) / total for c in range(num_classes)])
    probs = probs[probs > 0]
    entropy = -(probs * np.log(probs)).sum()
    max_entropy = math.log(num_classes)

    # max/min 比 + median/max 比
    freqs = sorted(counter.values(), reverse=True)
    max_freq = freqs[0]
    min_freq = freqs[-1] if freqs else 0
    median_freq = np.median(freqs) if freqs else 0

    # Head / Tail 分拆（按频次排序，前 20% 为 head）
    n_head = max(1, int(num_classes * 0.2))
    head_sum = sum(freqs[:n_head])
    tail_sum = sum(freqs[n_head:])

    metrics = {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "entropy_ratio": entropy / max_entropy,
        "max_min_ratio": max_freq / max(1, min_freq),
        "max_median_ratio": max_freq / max(1, median_freq),
        "head_ratio": head_sum / total,
        "tail_ratio": tail_sum / total,
    }
    return counter, total, metrics


def analyze_box_sizes(all_data):
    """框尺寸 + COCO 尺度分桶。"""
    ws, hs, areas = [], [], []
    sizes = {}  # (w, h) pairs
    for _, boxes, _ in all_data:
        for (cx, cy, w, h) in boxes:
            ws.append(w); hs.append(h)
            areas.append(w * h)
            sizes[(w, h)] = sizes.get((w, h), 0) + 1

    if not areas:
        return {}, {}

    arr = np.array(areas)
    w_arr = np.array(ws)
    h_arr = np.array(hs)

    tiny = int((arr < 0.01).sum())
    small = int(((arr >= 0.01) & (arr < 0.05)).sum())
    medium = int(((arr >= 0.05) & (arr < 0.15)).sum())
    large = int((arr >= 0.15).sum())

    # 长宽比分布
    ratios = []
    for w, h in zip(ws, hs):
        ratios.append(max(w, h) / max(1e-6, min(w, h)))
    ratio_buckets = {
        "<2:1": sum(1 for r in ratios if r < 2),
        "2-4:1": sum(1 for r in ratios if 2 <= r < 4),
        "4-8:1": sum(1 for r in ratios if 4 <= r < 8),
        ">8:1": sum(1 for r in ratios if r >= 8),
    }

    size_stats = {
        "w_median": float(np.median(w_arr)),
        "w_p10": float(np.percentile(w_arr, 10)),
        "w_p90": float(np.percentile(w_arr, 90)),
        "h_median": float(np.median(h_arr)),
        "h_p10": float(np.percentile(h_arr, 10)),
        "h_p90": float(np.percentile(h_arr, 90)),
        "area_median": float(np.median(arr)),
        "area_p10": float(np.percentile(arr, 10)),
        "area_p90": float(np.percentile(arr, 90)),
        "tiny": tiny, "small": small, "medium": medium, "large": large,
    }
    return size_stats, ratio_buckets


def analyze_boxes_per_image(all_data):
    """每图目标数分布。"""
    counts = [len(lbls) for _, _, lbls in all_data]
    arr = np.array(counts)
    stats = {
        "total_images": len(counts),
        "empty_images": int((arr == 0).sum()),
        "non_empty": int((arr > 0).sum()),
        "avg": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": int(arr.max()),
        "min": int(arr.min()),
        "p90": float(np.percentile(arr, 90)),
    }
    return stats, counts


def bootstrap_val_variance(all_data, val_ratio, num_bootstrap, seed=42):
    """估计 val 集按 val_ratio 切分时，mAP 的采样波动上界。

    做法：重复 N 次随机切分 train/val，每次 val 集内部算"mAP 代理指标"
    （这里用每个类别的 recall 方差 + 目标密度差异来近似），
    输出均值 ± 标准差，用来指导 early_stop_min_delta 该设多大。

    注意：这是**采样波动估计**，不是真实 mAP — 真实 mAP 必须跑推理。
    但如果 val=200 图 vs train=1800 图，采样本身引入的方差下限
    可以用这个方法估出来。
    """
    rng = random.Random(seed)
    n = len(all_data)
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val

    # 代理指标：每次切分后，val 的"类别分布与全量的 L1 距离"
    # + val 的"目标密度（每图框数）与全量的偏差"
    class_distances = []
    density_deviations = []

    # 全量类别分布
    full_counts = Counter()
    for _, _, labels in all_data:
        full_counts.update(labels)
    total_full = sum(full_counts.values())

    for _ in range(num_bootstrap):
        perm = rng.sample(range(n), n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        val_data = [all_data[i] for i in val_idx]

        # val 类分布
        val_counts = Counter()
        val_boxes = 0
        for _, _, labels in val_data:
            val_counts.update(labels)
            val_boxes += len(labels)

        # L1 距离（相对频率差异）
        l1 = 0.0
        for c in range(max(max(full_counts.keys()), max(val_counts.keys())) + 1):
            l1 += abs(val_counts.get(c, 0) / max(1, val_boxes)
                      - full_counts.get(c, 0) / max(1, total_full))
        class_distances.append(l1)

        # 目标密度偏差
        val_density = val_boxes / n_val
        full_density = total_full / n
        density_deviations.append(abs(val_density - full_density))

    result = {
        "num_bootstrap": num_bootstrap,
        "n_total": n,
        "n_val_per_split": n_val,
        "n_train_per_split": n_train,
        "class_dist_l1_mean": float(np.mean(class_distances)),
        "class_dist_l1_std": float(np.std(class_distances)),
        "class_dist_l1_p95": float(np.percentile(class_distances, 95)),
        "density_dev_mean": float(np.mean(density_deviations)),
        "density_dev_std": float(np.std(density_deviations)),
        "density_dev_p95": float(np.percentile(density_deviations, 95)),
    }

    # 建议的 min_delta 粗估：
    #   类别不均衡 L1 距离的 95 分位 × 0.1 作为波动贡献
    #   实际 mAP 波动通常 ~ val_size^(-0.5) * baseline_mAP
    #   baseline_mAP 从过往实验取 ~0.05 作为早期训练基线
    baseline_map = 0.05
    std_from_sample = baseline_map / math.sqrt(n_val) * 1.96  # 95% CI 半宽
    result["suggested_min_delta_lower"] = round(std_from_sample * 0.5, 4)
    result["suggested_min_delta_upper"] = round(std_from_sample * 1.5, 4)
    result["note"] = (
        f"val={n_val}张, baseline_mAP~{baseline_map}, "
        f"采样波动上界~±{std_from_sample:.4f}. min_delta 建议在此区间内"
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="数据集根目录（含 labels/ 子目录）")
    ap.add_argument("--num-classes", type=int, default=12)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--num-bootstrap", type=int, default=2000)
    ap.add_argument("--id2label", default=None,
                    help="类别名映射（可选，JSON 格式或 dict 路径）")
    args = ap.parse_args()

    print("=" * 60)
    print(f"数据集分析: {args.data_root}")
    print(f"  num_classes={args.num_classes}, val_ratio={args.val_ratio}")
    print(f"  bootstrap={args.num_bootstrap}")
    print("=" * 60)

    # ---- 加载 ----
    all_data = load_all_labels(args.data_root, args.num_classes)
    if not all_data:
        print("没有找到任何标签文件，退出")
        sys.exit(1)
    print(f"\n✓ 已加载 {len(all_data)} 张图的标签\n")

    # ---- 1. 类别分布 ----
    counter, total, cm = analyze_class_distribution(all_data, args.num_classes)
    print("【1. 类别分布】")
    print(f"  总框数: {total}")
    print(f"  {'class':>6s}  {'count':>7s}  {'ratio':>7s}  {'cum_ratio':>10s}")
    cum = 0.0
    for c in sorted(range(args.num_classes)):
        cnt = counter.get(c, 0)
        cum += cnt
        print(f"  {c:>6d}  {cnt:>7d}  {cnt/total:>6.1%}  {cum/total:>9.1%}")
    print(f"\n  [不均衡指标]")
    print(f"  熵: {cm['entropy']:.3f} / max={cm['max_entropy']:.3f}  "
          f"(ratio={cm['entropy_ratio']:.3f}, 1.0=完全均匀)")
    print(f"  max/min: {cm['max_min_ratio']:.1f}x  "
          f"max/median: {cm['max_median_ratio']:.2f}x")
    print(f"  head 20% 类占比: {cm['head_ratio']:.1%}  "
          f"tail 80% 类占比: {cm['tail_ratio']:.1%}")

    # ---- 2. 框尺寸 ----
    size_stats, ratio_buckets = analyze_box_sizes(all_data)
    print("\n【2. 框尺寸分布】")
    print(f"  w:  median={size_stats['w_median']:.4f}  "
          f"p10={size_stats['w_p10']:.4f}  p90={size_stats['w_p90']:.4f}")
    print(f"  h:  median={size_stats['h_median']:.4f}  "
          f"p10={size_stats['h_p10']:.4f}  p90={size_stats['h_p90']:.4f}")
    print(f"  area: median={size_stats['area_median']:.6f}  "
          f"p10={size_stats['area_p10']:.6f}  p90={size_stats['area_p90']:.6f}")
    total_boxes = size_stats["tiny"] + size_stats["small"] + \
                  size_stats["medium"] + size_stats["large"]
    print(f"\n  COCO 尺度分桶 (total={total_boxes}):")
    for name, key in [("tiny   (<0.01)", "tiny"),
                      ("small  (0.01-0.05)", "small"),
                      ("medium (0.05-0.15)", "medium"),
                      ("large  (>0.15)", "large")]:
        cnt = size_stats[key]
        print(f"    {name:>22s}: {cnt:>5d}  ({cnt/total_boxes:>5.1f}%)")
    print(f"\n  长宽比分布:")
    for k, v in ratio_buckets.items():
        print(f"    {k:>10s}: {v:>5d} ({v/total_boxes:>5.1f}%)")

    # ---- 3. 每图目标数 ----
    bp_stats, _ = analyze_boxes_per_image(all_data)
    print("\n【3. 每图目标数】")
    print(f"  total:   {bp_stats['total_images']}")
    print(f"  empty:   {bp_stats['empty_images']} ({bp_stats['empty_images']/bp_stats['total_images']:.1%})")
    print(f"  non-empty: {bp_stats['non_empty']} ({bp_stats['non_empty']/bp_stats['total_images']:.1%})")
    print(f"  boxes/img:  avg={bp_stats['avg']:.1f}  median={bp_stats['median']:.1f}  "
          f"p90={bp_stats['p90']:.1f}  min={bp_stats['min']}  max={bp_stats['max']}")

    # ---- 4. Bootstrap val 波动估计 ----
    print("\n【4. val 集采样波动 bootstrap 估计】")
    result = bootstrap_val_variance(
        all_data, args.val_ratio, args.num_bootstrap)
    print(f"  采样 {result['num_bootstrap']} 次，每次 val={result['n_val_per_split']} 张, "
          f"train={result['n_train_per_split']} 张")
    print(f"  类别分布 L1 距离（val vs 全量）:")
    print(f"    mean={result['class_dist_l1_mean']:.4f}  "
          f"std={result['class_dist_l1_std']:.4f}  "
          f"p95={result['class_dist_l1_p95']:.4f}")
    print(f"  目标密度偏差（每图框数差）:")
    print(f"    mean={result['density_dev_mean']:.2f}  "
          f"std={result['density_dev_std']:.2f}  "
          f"p95={result['density_dev_p95']:.2f}")
    print(f"\n  【早停 min_delta 建议区间】: "
          f"{result['suggested_min_delta_lower']} ~ {result['suggested_min_delta_upper']}")
    print(f"  {result['note']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

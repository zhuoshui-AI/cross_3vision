"""P3 冒烟：三模态 copy-paste 增强的正确性自检（example_data 可跑）。

验证五件事（不用肉眼盯蒙太奇也能放心上训练）：
  1) 管线不炸：train 模式反复取样，张量形状/值域/标注框全合法；
  2) 真的在贴：目标类每轮实例数显著多于关闭 copy-paste 的基线
     （目标类自动选数据里"实例最少但非零"的类——example_data 没有 ball，
      脚本会自己挑；正式数据可 --class 7 指定 ball）；
  3) 三模态同区一致：框内 IR 有内容（非全平）、框内深度有效占比不塌。
     对比口径是与基线的违规率（真实小目标本来就常落在深度空洞里，
     绝对零违规不成立）；粘贴框不应比原始框更差。
  4) 关闭路径零污染：copy_paste_prob=0 行为与旧代码一致；val 模式
     （train=False）禁用增强且两次取样逐像素一致（确定性不破坏）；
  5) 产一张人眼复核蒙太奇：RGB/IR/Depth 三联图 + GT 框
     → outputs/copy_paste_preview/preview.png

用法：
    python scripts/smoke_copy_paste.py --data-root example_data
    # 服务器上跑正式冒烟（真实分布，指定弱类 ball）：
    python scripts/smoke_copy_paste.py \
        --data-root /HUBU-AI027/AIdemo/训练集/AIC2026_Train_2000 --class 7
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.data import MultiModalDataset, _RGB_MEAN, _RGB_STD


def pick_rare_class(data_root, num_classes=12):
    """扫 labels/*.txt，返回"实例数最少但非零"的类（贴近稀有类粘贴场景）。"""
    counts = np.zeros(num_classes, dtype=np.int64)
    label_dir = Path(data_root) / "labels"
    for txt in sorted(label_dir.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                c = int(float(parts[0]))
                if 0 <= c < num_classes:
                    counts[c] += 1
    present = np.where(counts > 0)[0]
    if len(present) == 0:
        raise RuntimeError(f"{label_dir} 里没有任何标注")
    cls = int(present[np.argmin(counts[present])])
    print(f"[auto] per-class instances: {dict(enumerate(counts.tolist()))}")
    print(f"[auto] 选定目标类 = {cls}（实例最少且非零：{counts[cls]} 个）")
    return cls


def denorm_rgb(t):
    x = t.numpy().transpose(1, 2, 0) * _RGB_STD + _RGB_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def region_violations(sample, encoding):
    """对每个 GT 框做'三模态同区一致'检查，返回 (违规框数, 总框数)。"""
    ir = sample["pixel_values_ir"].numpy()[0]
    boxes = sample["labels"]["boxes"].numpy()
    H, W = ir.shape
    bad, total = 0, 0
    for (cx, cy, w, h) in boxes:
        x1 = int(max(0, round((cx - w / 2) * W)))
        x2 = int(min(W, round((cx + w / 2) * W)))
        y1 = int(max(0, round((cy - h / 2) * H)))
        y2 = int(min(H, round((cy + h / 2) * H)))
        total += 1
        if x2 - x1 < 2 or y2 - y1 < 2:
            bad += 1
            continue
        if float(ir[y1:y2, x1:x2].std()) < 0.01:      # 框内 IR 全平
            bad += 1
        if encoding == "geo3":
            valid = sample["pixel_values_depth"].numpy()[2, y1:y2, x1:x2]
        else:
            valid = sample["depth_mask"].numpy()[y1:y2, x1:x2].astype(np.float32)
        if float(valid.mean()) < 0.3:                 # 深度全无效塌陷
            bad += 1
    return bad, total


def iterate(ds, cls, epochs, encoding, do_checks):
    total_cls, bad, n_boxes, max_labels = 0, 0, 0, 0
    for _ep in range(epochs):
        for i in range(len(ds)):
            s = ds[i]
            labs = s["labels"]["class_labels"].numpy()
            total_cls += int((labs == cls).sum())
            max_labels = max(max_labels, int(len(labs)))
            for k in ("pixel_values_rgb", "pixel_values_ir",
                      "pixel_values_depth"):
                assert np.isfinite(s[k].numpy()).all(), f"{k} 出现 NaN/Inf"
            b = s["labels"]["boxes"].numpy()
            if len(b):
                assert (b >= 0).all() and (b <= 1).all(), "标注框越界 [0,1]"
                assert (b[:, 2] > 0).all() and (b[:, 3] > 0).all(), "零尺寸框"
            if do_checks:
                nb, nt = region_violations(s, encoding)
                bad += nb
                n_boxes += nt
    return total_cls, bad, n_boxes, max_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="example_data")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--encoding", default="geo3",
                    choices=["geo3", "linear"])
    ap.add_argument("--class", dest="target_class", type=int, default=None,
                    help="粘贴目标类 id（默认自动选数据里最稀有的非零类）")
    ap.add_argument("--crop-max", dest="crop_max", type=int, default=200,
                    help="收割实例最长边上限（example_data 640 分辨率需 200；"
                         "真实 1920 数据默认 96）")
    args = ap.parse_args()

    ncls = 12
    cls = args.target_class if args.target_class is not None \
        else pick_rare_class(args.data_root, ncls)

    kw = dict(img_size=args.img_size, train=True, num_classes=ncls,
              depth_encoding=args.encoding)

    # example_data 是 640 分辨率、实例普遍 100px+；收割上限要放宽才能冒烟。
    # 真实训练数据 1920 分辨率下弱类目标很小，默认 96 即可（--crop-max 可调）。
    crop_max = int(getattr(args, "crop_max", 200) or 200)

    # ---- 基线：关闭 copy-paste（同样做区域检查，得到基线违规率）----
    np.random.seed(0)
    ds_off = MultiModalDataset(data_root=args.data_root,
                                copy_paste_prob=0.0, copy_paste_classes=[cls],
                                copy_paste_crop_max=crop_max, **kw)
    base_total, base_bad, base_boxes, base_max = iterate(
        ds_off, cls, args.epochs, args.encoding, do_checks=True)
    base_rate = base_bad / max(1, base_boxes)
    print(f"[baseline off] cls={cls} instances={base_total}  "
          f"violations={base_bad}/{base_boxes} (rate={base_rate:.3f})  "
          f"max-labels={base_max}")

    # ---- 开启 copy-paste（prob=1 全开；=1.0 时不消耗 RNG，两流对齐）----
    np.random.seed(0)
    ds_on = MultiModalDataset(data_root=args.data_root,
                              copy_paste_prob=1.0, copy_paste_max=3,
                              copy_paste_classes=[cls],
                              copy_paste_bank_cap=200,
                              copy_paste_crop_max=crop_max, **kw)
    on_total, on_bad, on_boxes, on_max = iterate(
        ds_on, cls, args.epochs, args.encoding, do_checks=True)
    on_rate = on_bad / max(1, on_boxes)
    print(f"[copy-paste on] cls={cls} instances={on_total}  "
          f"violations={on_bad}/{on_boxes} (rate={on_rate:.3f})  "
          f"max-labels={on_max}")
    print(f"  bank={ {k: len(v) for k, v in ds_on._cp_bank.items()} }  "
          f"stats={ds_on._cp_stats}")

    ok = True
    if on_total > base_total:
        print(f"[PASS] 实例数 {base_total} → {on_total} "
              f"(+{on_total - base_total})")
    else:
        print(f"[FAIL] 实例数没有增加（{base_total} → {on_total}）——粘贴未生效")
        ok = False
    # 粘贴框不应比原始框更不"一致"：违规率允许 2 个百分点的抽样噪声
    if on_rate <= base_rate + 0.02:
        print(f"[PASS] 三模态同区一致性未劣化 "
              f"(rate {base_rate:.3f} → {on_rate:.3f})")
    else:
        print(f"[FAIL] 违规率上升 {base_rate:.3f} → {on_rate:.3f}，粘贴疑似污染区域")
        ok = False

    # ---- val 模式：不粘贴、确定性 ----
    ds_v = MultiModalDataset(data_root=args.data_root, train=False,
                             num_classes=ncls, depth_encoding=args.encoding,
                             copy_paste_prob=1.0, copy_paste_classes=[cls])
    assert not ds_v._cp_enabled, "val 数据集不应启用 copy-paste"
    a = ds_v[0]
    b = MultiModalDataset(data_root=args.data_root, train=False,
                          num_classes=ncls,
                          depth_encoding=args.encoding)[0]
    same = all(np.array_equal(a[k].numpy(), b[k].numpy())
               for k in ("pixel_values_rgb", "pixel_values_ir",
                         "pixel_values_depth", "depth_mask"))
    print(f"[{'PASS' if same else 'FAIL'}] val 模式确定性（两次取样逐像素一致）")
    ok = ok and same

    # ---- 蒙太奇：前 3 个样本 RGB | IR | Depth 三联图（红框=目标类）----
    try:
        import cv2
        rows = []
        for i in range(min(3, len(ds_on))):
            s = ds_on[i]
            rgb = denorm_rgb(s["pixel_values_rgb"])
            ir = cv2.cvtColor((np.clip(s["pixel_values_ir"].numpy()[0], 0, 1)
                               * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if args.encoding == "geo3":
                d = (np.clip(s["pixel_values_depth"].numpy()[1], 0, 1)
                     * 255).astype(np.uint8)
            else:
                d = (np.clip(s["pixel_values_depth"].numpy()[0], 0, 1)
                     * 255).astype(np.uint8)
            dp = cv2.applyColorMap(d, cv2.COLORMAP_JET)
            row = np.concatenate([rgb, ir, dp], axis=1)   # (H, 3W, 3) BGR 混排
            Hh, Ww = rgb.shape[:2]
            for (cx, cy, w, h), lb in zip(s["labels"]["boxes"].numpy(),
                                          s["labels"]["class_labels"].numpy()):
                x1 = int(max(0, (cx - w / 2) * Ww))
                x2 = int(min(Ww, (cx + w / 2) * Ww))
                y1 = int(max(0, (cy - h / 2) * Hh))
                y2 = int(min(Hh, (cy + h / 2) * Hh))
                for x0 in (0, Ww, 2 * Ww):                  # 三个面板画同一框
                    cv2.rectangle(row, (x0 + x1, y1), (x0 + max(x1 + 1, x2),
                                                      max(y1 + 1, y2)),
                                  (0, 0, 255) if int(lb) == cls else (0, 255, 0),
                                  2 if int(lb) == cls else 1)
            rows.append(row)
        out_dir = _PROJ_ROOT / "outputs" / "copy_paste_preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = str(out_dir / "preview.png")
        cv2.imwrite(out, np.concatenate(rows, axis=0))
        print(f"[PASS] 蒙太奇已存 → {out}（红框=目标类实例，含粘贴；RGB|IR|Depth）")
    except Exception as e:
        print(f"[WARN] 蒙太奇生成失败（不影响冒烟判定）: {e}")

    print(f"\nsmoke_copy_paste: {'ALL PASS' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

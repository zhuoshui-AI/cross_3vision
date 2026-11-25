"""P4：测试集高置信伪标签生成 + 训练用数据目录物化。

流程：
  1) 用给定的打标模型（1 个 ckpt，或多个 ckpt 融合 = ensemble 伪标签，
     复用 P2 的 TTA+WBF 机器）推理全部 --test-root；
  2) 只保留 conf ≥ --conf 的框（默认 0.5：小数据集上高置信伪标签的
     精度通常 >90%，低置信框引入的噪声标签会反噬训练）；
  3) 物化两个可直接训练的数据目录（symlink，服务器零拷贝；--copy 退化为复制）：
       <out>/pseudo_data/   训练用：原训练集去掉 val 部分 + 带伪标签的测试图
       <out>/val_data/      固定的 200 张 val（P0 协议原封不动）
  4) 打印下一条训练命令（利用 train.py 的"独立 val 目录"机制，
     val 完全来自 val_data/，杜绝伪标签混入验证集）。

训练侧无需改任何代码（val 走 train.py 的"独立 val 目录"分支，伪标签
只进训练侧、val 仍是 P0 固定名单）：
  python -m src.train \
      --set data.data_root=<out>/val_data \
      --set data.train_split=<out>/pseudo_data \
      --set data.val_split=<out>/val.txt

注意三个键必须同时给：train.py 的 _resolve 只有在 val_split 解析出
"非空 split 文件"时才启用独立 val 分支；data_root=val_data 提供验证图
的根目录，val.txt 是固定 200 名单（脚本已写好）。

用法（服务器，用 960 主力 ckpt 打标）：
  python scripts/pseudo_label.py \
      --ckpt outputs/ckpt/tri_swin/<best_960_run>/best.pth \
      --train-root /HUBU-AI027/AIdemo/训练集/AIC2026_Train_2000 \
      --val-list outputs/ckpt/tri_swin/<ref_run>/splits/val.txt \
      --test-root /HUBU-AI027/AI_competition/data/AIC2026_PHASE_1_1000 \
      --test-root /HUBU-AI027/AI_competition/data/AIC2026_PHASE_2_1000 \
      --out-root outputs/pseudo

注意事项：
  - 测试集无标注图里"零个高置信框"的默认不物化（--keep-empty 可选保留
    作纯背景/负样本，有漏检风险，默认关）。
  - 测试 stem 与训练 stem 撞名时该图跳过（防止覆盖真标注）。
  - 伪标签文件是 5 列 YOLO 格式（cls cx cy w h）——data.load_label 只认
    5 列，conf 已经过滤完不需要再写进文件。
  - 伪标签质量上限 = 打标模型质量：务必用目前最好的 ckpt（960 主力
    run 的 best.pth），别用旧 640 模型。
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# scripts/ 独立入口：把项目根加到 sys.path
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import scripts.eval_splits as _es  # noqa: E402
import scripts.ensemble_tta as _ens  # noqa: E402
from src.data import IMG_EXTS, find_image  # noqa: E402


def link_or_copy(src, dst, use_copy=False):
    """优先 symlink（服务器零拷贝），失败（如 Windows 无权限）退化为复制。"""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        return          # 幂等：重复运行不重复物化
    if use_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(str(src.resolve()), str(dst))
    except OSError:
        shutil.copy2(src, dst)


def materialize_dir(data_root, stems, label_src, out_dir, use_copy):
    """物化一个可训练目录：三模态图 + labels 全部来自原数据。"""
    for sub in ("visible", "infrared", "depth", "labels"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for stem in stems:
        for sub, d in (("visible", data_root), ("infrared", data_root),
                       ("depth", data_root)):
            try:
                src = find_image(stem, os.path.join(d, sub))
            except FileNotFoundError:
                break
            link_or_copy(src, out_dir / sub / Path(src).name, use_copy)
        else:
            lbl_src = label_src(stem)
            if lbl_src is not None:
                link_or_copy(lbl_src, out_dir / "labels" / f"{stem}.txt",
                             use_copy)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(
        description="测试集高置信伪标签 + 训练目录物化（P4）")
    ap.add_argument("--ckpt", action="append", required=True,
                    help="打标模型 ckpt（传多个 = ensemble 伪标签）")
    ap.add_argument("--cfg", default=None,
                    help="yaml 配置（仅当 ckpt 不带 config 时需要）")
    ap.add_argument("--weights", default=None, help="多 ckpt 融合权重，逗号分隔")
    ap.add_argument("--train-root", required=True,
                    help="原训练集目录（labels/ 真标注所在）")
    ap.add_argument("--val-list", required=True,
                    help="P0 协议的固定 val 名单 txt（这些图只进 val_data）")
    ap.add_argument("--test-root", action="append", required=True,
                    help="无标注测试集目录，可传多次（phase_1 + phase_2）")
    ap.add_argument("--out-root", default="outputs/pseudo")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    ap.add_argument("--tta-scales", default="")
    ap.add_argument("--iou-thr", type=float, default=0.55)
    ap.add_argument("--pre-conf", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--keep-empty", action="store_true",
                    help="零检出测试图也物化为纯背景（默认跳过）")
    ap.add_argument("--copy", action="store_true",
                    help="用复制代替 symlink（Windows/跨盘符时用）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只推理 + 统计，不物化任何文件")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    args.amp = not args.no_amp
    args.max_dets = 100       # 打标上限，对齐推理协议
    device = "cuda" if (args.device == "auto" and __import__("torch").cuda.is_available()) \
        else (args.device if args.device != "auto" else "cpu")
    args.amp = args.amp and (device == "cuda")
    args.tta_scales = [int(s) for s in args.tta_scales.split(",") if s.strip()]

    train_root = Path(args.train_root)
    val_ids = _es._read_ids(args.val_list)
    val_set = set(val_ids)
    print(f"train_root={train_root}  fixed val={len(val_ids)} 张 "
          f"(来自 {args.val_list})")

    # ---- 原训练集 stem（labels 扫描，与 MultiModalDataset 同口径）----
    label_dir = train_root / "labels"
    train_stems = sorted(p.stem for p in label_dir.glob("*.txt"))
    train_only = [s for s in train_stems if s not in val_set]
    print(f"训练集共 {len(train_stems)} 张，其中 {len(train_only)} 张进 "
          f"pseudo_data，{len(train_stems) - len(train_only)} 张固定 val")

    # ---- 加载打标模型 ----
    import torch
    weights = [float(x) for x in args.weights.split(",")] if args.weights else []
    print("Loading labeler checkpoint(s):")
    ents = _ens.load_entries(args.ckpt, args.cfg, weights, device,
                             args.img_size)
    if not ents:
        print("[FATAL] 没有可用的 ckpt")
        sys.exit(2)

    # ---- 逐个测试集推理 + 过滤 ----
    all_pseudo = {}          # stem -> (boxes cxcywh, scores, labels)
    cls_counter = Counter()
    for tr in args.test_root:
        results = _ens.fused_predict_all(ents, tr, args, device)
        n_img, n_box = 0, 0
        for stem, (cxcywh, fs, fl_) in results.items():
            keep = fs >= args.conf
            bx, sc, lb = cxcywh[keep], fs[keep], fl_[keep]
            # 坐标夹回 [0,1]，退化的丢掉
            if len(bx):
                bx = np.clip(bx, 0.0, 1.0)
                good = (bx[:, 2] > 1e-4) & (bx[:, 3] > 1e-4) \
                       & (bx[:, 0] >= 0) & (bx[:, 1] >= 0) \
                       & (bx[:, 0] + bx[:, 2] / 2 <= 1.0001) \
                       & (bx[:, 1] + bx[:, 3] / 2 <= 1.0001)
                bx, sc, lb = bx[good], sc[good], lb[good]
            if len(bx) > 0:
                all_pseudo[stem] = (bx, sc, lb)
                cls_counter.update(int(c) for c in lb)
                n_img += 1
                n_box += int(len(bx))
            elif args.keep_empty:
                all_pseudo[stem] = (bx, sc, lb)
                n_img += 1
        print(f"  {tr}: {len(results)} 图 → {n_img} 图有伪标签，"
              f"共 {n_box} 框 (conf≥{args.conf})")

    # ---- stem 撞名检查（测试 stem 不得覆盖真标注）----
    collision = [s for s in all_pseudo if s in set(train_stems)]
    for s in collision:
        print(f"[WARN] 测试集 stem 与训练集撞名，跳过伪标签: {s}")
        all_pseudo.pop(s)

    print(f"\n伪标签类分布: {dict(sorted(cls_counter.items()))}")

    if args.dry_run:
        print("\n[dry-run] 未物化任何文件。确认分布合理后去掉 --dry-run 重跑。")
        return

    # ---- 物化 ----
    out_root = Path(args.out_root)
    pseudo_dir = out_root / "pseudo_data"
    val_dir = out_root / "val_data"

    # 0) 固定 val 名单落盘（train.py 独立 val 分支要求 val_split 是文件）
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "val.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(val_ids) + "\n")

    # 1) val_data：固定 200 张，标注 symlink 原文件
    n_val = materialize_dir(
        train_root, val_ids,
        label_src=lambda s: label_dir / f"{s}.txt", out_dir=val_dir,
        use_copy=args.copy)
    print(f"\nval_data: {n_val} 张 → {val_dir}")

    # 2) pseudo_data：train∖val 原图原标注
    n_train = materialize_dir(
        train_root, train_only,
        label_src=lambda s: label_dir / f"{s}.txt", out_dir=pseudo_dir,
        use_copy=args.copy)

    # 3) pseudo_data：测试图 + 写 5 列伪标签（load_label 只认 5 列）
    n_pseudo = 0
    for stem, (bx, sc, lb) in all_pseudo.items():
        # 找原图所在 test_root（三模态都要 symlink）
        src_root = None
        for tr in args.test_root:
            try:
                find_image(stem, os.path.join(tr, "visible"))
                src_root = Path(tr)
                break
            except FileNotFoundError:
                continue
        if src_root is None:
            continue
        for sub in ("visible", "infrared", "depth"):
            src = find_image(stem, src_root / sub)
            link_or_copy(src, pseudo_dir / sub / Path(src).name, args.copy)
        with open(pseudo_dir / "labels" / f"{stem}.txt", "w",
                  encoding="utf-8") as fh:
            for (cx, cy, w, h), c in zip(bx.tolist(), lb.tolist()):
                fh.write(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        n_pseudo += 1

    meta = {
        "ckpt": [str(p) for p in args.ckpt],
        "conf": args.conf,
        "tta": args.tta,
        "tta_scales": args.tta_scales,
        "train_root": str(train_root),
        "val_list": str(args.val_list),
        "test_roots": [str(t) for t in args.test_root],
        "n_train_kept": n_train,
        "n_val": n_val,
        "n_pseudo_images": n_pseudo,
        "n_pseudo_boxes": int(sum(cls_counter.values())),
        "pseudo_class_dist": dict(cls_counter),
        "skipped_collisions": collision,
    }
    (out_root / "pseudo_meta.json").parent.mkdir(parents=True, exist_ok=True)
    with open(out_root / "pseudo_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"pseudo_data: {n_train} 原训练图 + {n_pseudo} 伪标签测试图 "
          f"→ {pseudo_dir}")
    print(f"meta → {out_root / 'pseudo_meta.json'}")
    print(f"""
下一步训练命令（val 完全来自 val_data 固定名单，伪标签只进训练侧；
data_root/train_split/val_split 三个键必须同时给，缺一不可）：
  python -m src.train \\
      --set data.data_root={val_dir} \\
      --set data.train_split={pseudo_dir} \\
      --set data.val_split={out_root / 'val.txt'}
提示：先跑 150ep 对照确认伪标签增益为正，再上 400ep 全量。
""")


if __name__ == "__main__":
    main()

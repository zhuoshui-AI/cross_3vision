"""独立评测脚本：基于磁盘上固定的 val 名单评测任意 checkpoint。

设计目标：
- 训练跑完自动落盘 splits/train.txt / val.txt / split_meta.json，
  之后任何对比评测都用这同一份 val 集，不再重跑切分逻辑。
- 老 ckpt 没有 splits/ 目录时，可显式 --splits PATH 传入任何一份
  val 名单（例如从新 run 复制过来的 val.txt），实现跨 run 统一协议。

用法示例：
    # 新 run（有 splits/ 目录）——自动发现
    python scripts/eval_splits.py --ckpt outputs/ckpt/tri_swin/xxx/best.pth

    # 老 ckpt（无 splits/）——显式指定 val 名单 + 数据根
    python scripts/eval_splits.py --ckpt outputs/ckpt/tri_swin/best.pth \
        --splits outputs/ckpt/tri_swin/xxx/splits/val.txt \
        --data-root /HUBU-AI027/AIdemo/训练集/AIC2026_Train_2000

    # 同一个 ckpt 跑两套 val（分层 + 随机 双报告）
    python scripts/eval_splits.py --ckpt .../best.pth \\
        --splits outputs/ckpt/run_a/splits/val.txt --split-name val.txt \\
        --extra-split outputs/ckpt/run_b/splits/val_random.txt
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

# scripts/ 独立入口：把项目根加到 sys.path，就能用 `from src.xxx import`
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.data import MultiModalDataset, collate_fn
from src.eval import evaluate
from src.model import build_model


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _load_ckpt(ckpt_path):
    """返回 (state_dict, config_dict, epoch, best_metric)。

    兼容两种 ckpt 格式：
      - 本项目 pth：{"model": state_dict, "config": dict, "epoch": int, "best_metric": float}
      - 原始 state_dict（没有 model/config 键）：把整个 dict 当 state_dict，config=None
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
        cfg = raw.get("config")
        epoch = raw.get("epoch", -1)
        best = raw.get("best_metric") or raw.get("best_map") or -1.0
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
        cfg = raw.get("config")
        epoch = raw.get("epoch", -1)
        best = -1.0
    else:
        state = raw
        cfg = None
        epoch = -1
        best = -1.0
    return state, cfg, epoch, best


def _resolve_cfg(ckpt_cfg, cli_cfg_path):
    """优先级：CLI --cfg > ckpt 自带 > 致命报错。"""
    if cli_cfg_path:
        import yaml
        with open(cli_cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    if ckpt_cfg is not None:
        return ckpt_cfg
    raise RuntimeError(
        "Checkpoint 里没有 config 字段，也没传 --cfg，请提供训练用的 yaml 配置文件。")


def _discover_splits_dir(ckpt_path):
    """从 ckpt 位置向上找到同目录下的 splits/。

    best.pth / last.pth 常见存放位置：
      run_dir/best.pth        → splits/ 同级
      run_dir/ckpt/best.pth   → splits/ 在 ../splits/
    只要父层级或同级有 splits/ 就返回其路径。
    """
    here = Path(ckpt_path).resolve().parent
    for p in (here, here.parent):
        cand = p / "splits"
        if cand.is_dir():
            return cand
    return None


def _read_ids(txt_path):
    """读 split txt，每行一个 stem（允许空行 / # 注释）。

    用 utf-8-sig 解码自动剥掉 Windows PowerShell 默认写进去的 BOM，
    否则 stem 前会挂一个 \ufeff 导致找不到图片（No image for stem）。
    """
    ids = []
    with open(txt_path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def _pick_data_root(meta, cli_data_root):
    """优先级：CLI > split_meta.json（新 ckpt 落盘了绝对路径）。"""
    if cli_data_root:
        return cli_data_root
    if meta and meta.get("data_root"):
        return meta["data_root"]
    return None


def _guess_img_size(cfg, meta, default=640):
    if cfg and cfg.get("data", {}).get("img_size"):
        return int(cfg["data"]["img_size"])
    if meta and meta.get("img_size"):
        return int(meta["img_size"])
    return default


# ---------------------------------------------------------------------------
# 单次评测
# ---------------------------------------------------------------------------

def run_one_eval(ckpt_state, cfg, device, val_ids, data_root, img_size,
                 depth_enc, ncls, arch, batch_size, conf_th, amp):
    """构造 loader + model → evaluate()，返回 result dict。"""
    val_ds = MultiModalDataset(
        data_root=data_root, ids=val_ids,
        img_size=img_size, train=False, num_classes=ncls,
        depth_encoding=depth_enc)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
        pin_memory=(device == "cuda"))

    print(f"  Val IDs: {len(val_ids)}  Data root: {data_root}")

    # build_model 吃完整 cfg；部分老 ckpt 可能缺少个别新字段，缺了就兜底补默认
    cfg_eff = _fill_missing(cfg)
    cfg_eff.setdefault("model", {}).setdefault("arch", arch)
    model = build_model(cfg_eff).to(device)

    # 严格匹配：如果有任意 key 缺了，torch.load 的 strict 默认就会报；
    # 但有些 key 只是新增（如 pregate），对应权重全零 init，跳过即可
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    if missing or unexpected:
        print(f"  state_dict mismatch — missing={missing or []}, "
              f"unexpected={unexpected or []}")
    model.eval()

    res = evaluate(model, val_loader, device, cfg_eff,
                   amp_enabled=amp, conf_th=conf_th)
    return res


def _fill_missing(cfg):
    """给 cfg 补几个 build_model 可能用到、老 ckpt config 里缺的默认项。"""
    import copy
    c = copy.deepcopy(cfg)
    m = c.setdefault("model", {})
    m.setdefault("arch", "tri_swin")
    m.setdefault("num_labels", 12)
    m.setdefault("depth_encoding", "linear")
    m.setdefault("detr_warm_start", None)
    m.setdefault("tri_enable_pre_gate", False)
    m.setdefault("tri_ir_backbone", "swin")
    return c


# ---------------------------------------------------------------------------
# 打印 + CLI
# ---------------------------------------------------------------------------

def _print_header(label, kind, n_val, mAP5095, mAP50, per_class_ap):
    print(f"\n{'='*60}")
    print(f"  Split      : {label}")
    print(f"  Kind       : {kind}")
    print(f"  #val       : {n_val}")
    print(f"  mAP@50-95  : {mAP5095:.4f}")
    print(f"  mAP@50     : {mAP50:.4f}")
    print(f"{'='*60}")
    if per_class_ap:
        print("  Per-class AP@50-95:")
        for name, ap in sorted(per_class_ap.items(),
                               key=lambda kv: kv[1] if kv[1] is not None else -1.0):
            v = f"{ap:.4f}" if ap is not None else "—"
            print(f"    {name:<14s}: {v}")


def main():
    ap = argparse.ArgumentParser(
        description="独立评测：用磁盘固定名单 (splits/val.txt) 评测任意 ckpt。")
    ap.add_argument("--ckpt", required=True, help="checkpoint 路径 (.pth)")
    ap.add_argument("--cfg", default=None,
                    help="可选：yaml 配置（当 ckpt 不带 config 时必填）")
    ap.add_argument("--splits", default=None,
                    help="splits 目录路径，或直接指向 val 名单 txt；"
                         "不传则尝试从 ckpt 同目录自动发现 splits/")
    ap.add_argument("--split-name", default=None,
                    help="splits 目录里要评测的文件（默认 val.txt）")
    ap.add_argument("--extra-split", action="append", default=[],
                    help="额外评测的 val 名单 txt（可多次传入，实现多协议同时出分）")
    ap.add_argument("--data-root", default=None,
                    help="覆盖 split_meta.json 里记录的 data_root（数据位置变了时用）")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--conf-th", type=float, default=0.0,
                    help="DETR 评测约定用 0.0 保留全部 query；提交流水线才应该阈值过滤")
    ap.add_argument("--no-amp", action="store_true",
                    help="关闭 autocast（调试精度或 CPU 评测时）")
    ap.add_argument("--json-out", default=None,
                    help="把所有评测结果（含各 split）写成 json")
    args = ap.parse_args()

    # ---------- ckpt + cfg ----------
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        print(f"[FATAL] ckpt 不存在: {ckpt_path}")
        sys.exit(2)

    state, ckpt_cfg, epoch, best = _load_ckpt(ckpt_path)
    cfg = _resolve_cfg(ckpt_cfg, args.cfg)

    print(f"Checkpoint: {ckpt_path}")
    print(f"  epoch={epoch}  best_metric={best:.4f}")

    # ---------- splits ----------
    # splits 可以是目录，也可以是单个 txt 文件（老 ckpt 从别处拷名单过来时）
    meta = None
    val_ids_by_label = {}   # label -> list of ids
    kind_by_label = {}      # label -> kind 字符串，header 打印

    if args.splits:
        p = Path(args.splits).resolve()
        if p.is_dir():
            splits_dir = p
        else:
            # 把单 txt 包装成 {label: (ids, kind)}，跳过下面的自动发现
            label = p.stem
            val_ids_by_label[label] = _read_ids(p)
            kind_by_label[label] = "external"
            splits_dir = None
    else:
        splits_dir = _discover_splits_dir(ckpt_path)

    if splits_dir is not None:
        meta_path = splits_dir / "split_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # 读主 val.txt，--split-name 可覆盖
        main_txt_name = args.split_name or "val.txt"
        main_txt = splits_dir / main_txt_name
        if not main_txt.exists():
            # 兼容 splits/ 只有 val_random.txt / val_stratified.txt 这类命名
            candidates = list(splits_dir.glob("val*.txt"))
            if not candidates:
                print(f"[FATAL] {splits_dir} 里找不到 val*.txt")
                sys.exit(2)
            main_txt = sorted(candidates)[-1]
            print(f"  未指定 --split-name，自动选了: {main_txt.name}")
        label = main_txt.stem
        val_ids_by_label[label] = _read_ids(main_txt)
        kind_by_label[label] = (meta or {}).get("split_kind", "unknown")

    # 额外 split
    for extra in args.extra_split:
        q = Path(extra).resolve()
        val_ids_by_label[q.stem] = _read_ids(q)
        kind_by_label[q.stem] = "external"

    if not val_ids_by_label:
        print("[FATAL] 没找到任何 val 名单：请用 --splits 指定。")
        sys.exit(2)

    # ---------- 全局参数 ----------
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] 没 GPU，退回 CPU")
        device = "cpu"

    arch = (cfg.get("model") or {}).get("arch", "tri_swin")
    ncls = int((cfg.get("model") or {}).get("num_labels", 12))
    depth_enc = (cfg.get("model") or {}).get("depth_encoding", "linear")
    img_size = _guess_img_size(cfg, meta, default=640)
    data_root = _pick_data_root(meta, args.data_root)
    if data_root is None:
        print("[FATAL] 不知道数据在哪：请给 --data-root，或让 splits/ 里带 split_meta.json")
        sys.exit(2)
    if not Path(data_root).is_dir():
        print(f"[FATAL] data_root 指向的目录不存在: {data_root}")
        sys.exit(2)

    amp = (device == "cuda") and (not args.no_amp)

    print(f"Device={device}  arch={arch}  img_size={img_size}  "
          f"depth_enc={depth_enc}  ncls={ncls}  amp={amp}")
    print(f"Data root: {data_root}")

    # ---------- 逐个 split 评测 ----------
    all_results = {}
    try:
        for label, val_ids in val_ids_by_label.items():
            print(f"\n{'─'*50}")
            print(f"[eval] split={label} ({kind_by_label[label]})  #val={len(val_ids)}")
            res = run_one_eval(
                ckpt_state=state, cfg=cfg, device=device,
                val_ids=val_ids, data_root=data_root,
                img_size=img_size, depth_enc=depth_enc, ncls=ncls, arch=arch,
                batch_size=args.batch_size, conf_th=args.conf_th, amp=amp)
            all_results[label] = {
                "split": label,
                "kind": kind_by_label[label],
                "n_val": len(val_ids),
                "map_5095": res.get("map_5095"),
                "map_50": res.get("map_50"),
                "per_class_ap": res.get("per_class_ap"),
            }
            _print_header(
                label, kind_by_label[label], len(val_ids),
                res.get("map_5095", float("nan")),
                res.get("map_50", float("nan")),
                res.get("per_class_ap"),
            )
    except KeyboardInterrupt:
        print("\n[interrupted] 已中断，当前结果保留。")

    # ---------- 写出 ----------
    if args.json_out:
        out = {
            "ckpt": str(ckpt_path),
            "epoch": epoch,
            "best_metric_ckpt": best,
            "device": device,
            "img_size": img_size,
            "depth_encoding": depth_enc,
            "arch": arch,
            "data_root": data_root,
            "results": all_results,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"\n结果已写 → {args.json_out}")


if __name__ == "__main__":
    main()

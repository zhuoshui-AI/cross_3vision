"""depth_encoding=geo3 实现验收（四组测试，CPU 可跑）。

用法（在 lab 根目录）：
    python -u -m scripts.verify_depth_geo3

  [T1] 数值单测：geo3 编码公式（log 单调增 / inverse 单调减 / 无效全零 /
       值域 [0,1]）+ linear 模式与改动前公式逐元素回归
  [T2] 前向 shape：DepthEncoder in_ch=1/3；SwinModalityBackbone
       adapt_mode="depth3" 的权重适配与「不冻结」耦合修正
  [T3] 端到端冒烟：rgbir_swin / tri_swin × geo3 在 example_data 上各训
       3 步（train=True，连带走 raw 域噪声/脱落增强），loss 有限无 NaN
  [T4] A/B 回归：depth_encoding=linear 下 build_model 的 state_dict
       关键层形状与改动前一致（旧 checkpoint 结构兼容）+ linear 前向
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(LAB)  # config 内 weights/... 为相对路径

import numpy as np
import torch
import torch.nn as nn

from src.data import MultiModalDataset, collate_fn
from src.backbones import DepthEncoder, SwinModalityBackbone
from src.model import build_model
from src.utils import load_yaml_config

EXAMPLE_DATA = os.path.join(LAB, "example_data")
PASS, FAIL = "PASS", "FAIL"
_failures = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        _failures.append(f"{name}  {detail}")


# ---------------------------------------------------------------- T1 数值单测
def t1_encoding_math():
    print("\n[T1] 数值单测：geo3 公式 + linear 回归", flush=True)
    H, W = 8, 10
    d = np.zeros((H, W), dtype=np.float32)
    vals = [0, 5, 10, 1000, 19999, 20000, 25000]  # 含无效哨兵与边界
    for i, v in enumerate(vals):
        d[i, :len(vals)] = v
    d[7, :] = 500.0          # 一个"洞"行
    d[7, 0] = 0.0            # 0 = 无回波 → 无效
    depth_hwc = d[..., None]
    valid = (d >= 10.0) & (d <= 20000.0)
    # 模拟 _fill_depth_holes 之后的真实生产输入形态：洞行(500mm)值已填充
    # 但掩码仍标无效；>20000 哨兵保持原值。即无效像素带非零 raw 值
    # （500mm → logc≈0.38 / inv=0.975；25000 → clip 后 logc=1.0），
    # 实证 geo[~valid]=0.0 把三通道全部压零、无值逃逸进最终 tensor。
    valid[7, :] = False

    # ---- linear 回归：与改动前公式逐元素完全相同 ----
    t, m = MultiModalDataset._normalize_depth(depth_hwc, True, valid, "linear")
    ref = np.clip(depth_hwc.astype(np.float32), 0, 20000.0) / 20000.0
    ref[~valid] = 0.0
    ref_t = torch.from_numpy(np.ascontiguousarray(ref.transpose(2, 0, 1))).float()
    check("linear 输出与改动前公式逐元素相同",
          t.shape == (1, H, W) and torch.equal(t, ref_t) and torch.equal(m, torch.from_numpy(valid)))
    # 兼容旧两位置参数调用（diag_dataflow.py 风格）：valid=None 的值域推断
    # 必须与显式传入"推断掩码"逐元素一致
    t2, m2 = MultiModalDataset._normalize_depth(depth_hwc, True)
    valid_inf = (d >= 10.0) & (d <= 20000.0)
    t3, m3 = MultiModalDataset._normalize_depth(depth_hwc, True, valid_inf, "linear")
    check("linear 两位置参数调用兼容（valid=None 推断 == 显式推断掩码）",
          torch.equal(t2, t3) and torch.equal(m2, m3))

    # ---- geo3 ----
    g, gm = MultiModalDataset._normalize_depth(
        depth_hwc, True, valid, "geo3")
    check("geo3 shape=(3,H,W) 且 mask=(H,W) bool",
          g.shape == (3, H, W) and gm.dtype == torch.bool and gm.shape == (H, W))
    check("geo3 值域 ⊂ [0,1]",
          float(g.min()) >= 0.0 and float(g.max()) <= 1.0,
          f"min={float(g.min()):.6f} max={float(g.max()):.6f}")
    # 元验证：测试输入非平凡 —— 无效像素的 logc 压零前必须非零
    # （洞 500mm → 0.628；25000 哨兵 → clip 后 1.0），否则本断言退化。
    pre_logc = np.log1p(np.clip(d, 0.0, 20000.0)) / np.log1p(20000.0)
    hole_px, sent_px = float(pre_logc[7, 1]), float(pre_logc[6, 0])
    assert hole_px > 0.5 and abs(sent_px - 1.0) < 1e-6, (hole_px, sent_px)
    check("geo3 无效像素三通道严格全 0（含填充洞 500mm 与 >20000 哨兵）",
          bool((g[:, ~valid] == 0).all()),
          f"压零前 logc: 洞={hole_px:.3f} 哨兵={sent_px:.3f}")
    check("geo3 valid 通道 == mask", bool((g[2] == gm.float()).all()))

    c0, c1 = g[0].numpy(), g[1].numpy()
    # 逐像素取有效值断言单调性（对每个有效像素，比较 10 < 1000 < 20000 的行）
    r10, r1000, r20k = c0[2, 2], c0[3, 2], c0[5, 2]     # log 通道 @ d=10/1000/20000
    check("log 通道随 d 单调递增（10<1000<20000 → c0 递增）",
          r10 < r1000 < r20k and abs(r20k - 1.0) < 1e-6,
          f"c0(10)={r10:.4f} c0(1000)={r1000:.4f} c0(20000)={r20k:.6f}")
    i10, i1000, i20k = c1[2, 2], c1[3, 2], c1[5, 2]     # inverse 通道
    check("inverse 通道随 d 单调递减（线性 (max-d)/max）",
          abs(i10 - 0.9995) < 1e-6 and abs(i1000 - 0.95) < 1e-6 and abs(i20k) < 1e-6,
          f"inv(10)={i10:.4f} inv(1000)={i1000:.4f} inv(20000)={i20k:.6f}")
    # 满图单调性：对所有有效像素对 (d_a < d_b) 验证 c0(a)<c0(b)、c1(a)>c1(b)
    dv, c0v, c1v = d[valid], c0[valid], c1[valid]
    order = np.argsort(dv)
    dv, c0v, c1v = dv[order], c0v[order], c1v[order]
    inc = np.diff(dv) > 0          # len = N-1，与 diff(c0v) 对齐
    check("geo3 全图单调（c0 增 / c1 减，随 d）",
          bool((np.diff(c0v)[inc] > 0).all() and (np.diff(c1v)[inc] < 0).all())
          and bool((np.diff(c0v) >= -1e-7).all() and (np.diff(c1v) <= 1e-7).all()))

    # ---- 8bit 路径（先映射 mm 域，共用公式）----
    d8 = np.array([[0, 1, 128, 255]] * 4, dtype=np.float32)
    v8 = d8 > 0
    g8, gm8 = MultiModalDataset._normalize_depth(
        d8[..., None], False, v8, "geo3")
    check("geo3 8bit：d=255 → log 通道 1.0；d=0 无效全零",
          abs(float(g8[0, 0, 3]) - 1.0) < 1e-6 and bool((g8[:, :, 0] == 0).all())
          and bool((g8[2] == gm8.float()).all()))

    # ---- 8bit 越界鲁棒性：损坏值 >255 不破坏 [0,1] 输出契约 ----
    d_bad = np.array([[300.0, -5.0, 128.0, 255.0]] * 2, dtype=np.float32)
    v_bad = d_bad > 0  # 300 会被当"有效"，输出必须仍被 clip 约束
    l_bad, _ = MultiModalDataset._normalize_depth(
        d_bad[..., None], False, v_bad, "linear")
    g_bad, _ = MultiModalDataset._normalize_depth(
        d_bad[..., None], False, v_bad, "geo3")
    check("越界 8bit：linear 输出仍 ⊆ [0,1]（clip 255）",
          float(l_bad.max()) <= 1.0 and float(l_bad.min()) >= 0.0
          and abs(float(l_bad[0, 0, 0]) - 1.0) < 1e-6,   # 300 → 255/255 = 1.0
          f"max={float(l_bad.max())}")
    check("越界 8bit：geo3 三通道仍 ⊆ [0,1]（log ≤1 / inv ≥0）",
          float(g_bad.max()) <= 1.0 and float(g_bad.min()) >= 0.0
          and abs(float(g_bad[0, 0, 0]) - 1.0) < 1e-6,   # 300 → d=20000 → log=1
          f"max={float(g_bad.max())} min={float(g_bad.min())}")

    # ---- raw 域增强（强制触发）：噪声仅有效像素、脱落只扩掩码 ----
    ds = MultiModalDataset.__new__(MultiModalDataset)
    ds._p_depth_noise, ds._p_depth_dropout = 1.0, 1.0
    raw = np.full((H, W, 1), 500.0, dtype=np.float32)
    raw[0, 0] = 0.0
    inv_m = np.zeros((H, W), dtype=bool)
    inv_m[0, 0] = True
    np.random.seed(0)
    out, out_inv = ds._augment_depth_train_raw(raw.copy(), inv_m.copy(), True)
    check("raw 噪声：无效像素值保持不变", float(out[0, 0, 0]) == 0.0)
    check("raw 噪声：有效像素全部被扰动（p=1）",
          bool((np.abs(out[..., 0] - 500.0)[~out_inv] > 0).all()))
    check("raw 噪声：clamp 后仍在 [10,20000] 有效区",
          bool((out[..., 0][~out_inv] >= 10).all() and (out[..., 0][~out_inv] <= 20000).all()))
    check("raw 脱落：掩码扩张且 raw 值不被清零",
          bool(out_inv.sum() > inv_m.sum()) and float(np.abs(out).max()) > 0)
    # 脱落区域编码后三通道全零（掩码兜底）
    g_drop, _ = MultiModalDataset._normalize_depth(out, True, ~out_inv, "geo3")
    check("脱落后 geo3 编码区域全零（mask 兜底）",
          bool((g_drop[:, out_inv] == 0).all()))


# ---------------------------------------------------------------- T2 前向 shape
def t2_forward_shapes():
    print("\n[T2] 前向 shape 与适配/冻结耦合", flush=True)
    B, Hsz, Wsz = 2, 256, 320
    mask = torch.rand(B, Hsz, Wsz) > 0.2

    enc3 = DepthEncoder(out_dim=64, in_ch=3)
    o3 = enc3(torch.rand(B, 3, Hsz, Wsz), mask)
    check("DepthEncoder in_ch=3：(B,3,H,W) → (B,64,H/32,W/32)",
          o3.shape == (B, 64, Hsz // 32, Wsz // 32), f"{tuple(o3.shape)}")
    check("DepthEncoder in_ch=3 首层卷积 3 通道",
          enc3.conv[0][0].weight.shape == (16, 3, 3, 3))

    enc1 = DepthEncoder(out_dim=64, in_ch=1)
    o1 = enc1(torch.rand(B, 1, Hsz, Wsz), mask)
    check("DepthEncoder in_ch=1（旧行为）：2 通道拼接 → 同输出 shape",
          o1.shape == (B, 64, Hsz // 32, Wsz // 32)
          and enc1.conv[0][0].weight.shape == (16, 2, 3, 3))

    # ---- tri_swin 深度塔：depth3 适配 + 不冻结耦合 ----
    torch.manual_seed(0)
    bb3 = SwinModalityBackbone(
        variant="swin_tiny_patch4_window7_224", in_chans=3,
        pretrained=False, freeze_stages=2, img_size=224,
        adapt_mode="depth3")
    w = bb3.swin.patch_embed.proj.weight
    check("depth3 塔首层 (96,3,4,4)", w.shape == (96, 3, 4, 4))
    check("depth3：c0 == c1（同为预训练三组均值）",
          bool((w[:, 0] == w[:, 1]).all()))
    check("depth3：c2 零初始化", bool((w[:, 2] == 0).all()))
    pe_trainable = all(p.requires_grad for p in bb3.swin.patch_embed.parameters())
    check("depth3：patch_embed 不被 freeze_stages 冻结（c2 需要梯度）",
          pe_trainable)
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        m0 = bb3.stem_maps(x)
        maps, feats = m0, []
        for i in range(4):
            maps = bb3.run_stage(i, maps)
            feats.append(maps)
    shapes = [tuple(f.shape[1:3]) for f in feats]
    chans = [f.shape[-1] for f in feats]
    check("depth3 塔 C1~C4 出图（56/28/14/7 @ 96/192/384/768）",
          shapes == [(56, 56), (28, 28), (14, 14), (7, 7)]
          and chans == [96, 192, 384, 768], f"{shapes}")

    # ---- 回归：原生 3ch（RGB 塔）仍冻结 stem；2ch 深度塔行为不变 ----
    bb_rgb = SwinModalityBackbone(
        variant="swin_tiny_patch4_window7_224", in_chans=3,
        pretrained=False, freeze_stages=2, img_size=224)
    rgb_frozen = all(not p.requires_grad
                     for p in bb_rgb.swin.patch_embed.parameters())
    check("回归：adapt_mode=None 的 3ch 塔仍冻结 patch_embed", rgb_frozen)
    bb2 = SwinModalityBackbone(
        variant="swin_tiny_patch4_window7_224", in_chans=2,
        pretrained=False, freeze_stages=2, img_size=224)
    w2 = bb2.swin.patch_embed.proj.weight
    check("回归：2ch 塔首层 (96,2,4,4)，c1 零初始化",
          w2.shape == (96, 2, 4, 4) and bool((w2[:, 1] == 0).all()))
    try:
        SwinModalityBackbone(variant="swin_tiny_patch4_window7_224",
                             in_chans=2, pretrained=False, img_size=224,
                             adapt_mode="depth3")
        check("depth3 + in_chans!=3 显式报错", False)
    except ValueError:
        check("depth3 + in_chans!=3 显式报错", True)


# ---------------------------------------------------------------- T3 端到端冒烟
def _make_batch(img_size, encoding):
    ds = MultiModalDataset(
        data_root=EXAMPLE_DATA, split_file=None, ids=None,
        img_size=img_size, train=True, num_classes=12,
        scan_source="labels", depth_encoding=encoding)
    ds.ids = ds.ids[:2]
    batch = collate_fn([ds[i] for i in range(len(ds))])
    # collate_fn 已把 labels 整理为逐样本 dict 列表
    labels = batch["labels"]
    return batch, labels


def _smoke_cfg(base, arch, img_size):
    """冒烟配置：CPU 上 SDPA 回退 math 路，全 4 层 encoder（如 256 输入
    B=2 → 5120 token）注意力矩阵 ~840MB/层×12 层会 OOM——encoder scales
    降级为 [8,16,32]（项目认可的降级档），与深度编码正交。"""
    cfg = copy.deepcopy(base)
    cfg["model"]["arch"] = arch
    cfg["model"]["depth_encoding"] = "geo3"
    cfg["model"]["detr_warm_start"] = None      # 冒烟只验证深度通路
    cfg["model"]["rgbir_encoder_scales"] = [8, 16, 32]
    cfg["model"]["tri_encoder_scales"] = [8, 16, 32]
    cfg["data"]["img_size"] = img_size
    return cfg


def t3_e2e_smoke(img_size, steps):
    print(f"\n[T3] 端到端冒烟（example_data, img={img_size}, "
          f"train=True 含 raw 域增强）", flush=True)
    base = load_yaml_config(os.path.join(LAB, "configs", "default.yaml"))
    for arch in ("rgbir_swin", "tri_swin"):
        cfg = _smoke_cfg(base, arch, img_size)
        batch, labels = _make_batch(img_size, "geo3")
        check(f"[{arch}] 数据侧 geo3 输出 (B,3,H,W)",
              batch["pixel_values_depth"].shape[1] == 3,
              f"{tuple(batch['pixel_values_depth'].shape)}")
        # valid 通道与 depth_mask 逐元素一致（几何变换后仍对齐）
        vch = batch["pixel_values_depth"][:, 2]
        check(f"[{arch}] valid 通道 == depth_mask（几何对齐）",
              bool((vch == batch["depth_mask"].float()).all()))
        t0 = time.time()
        torch.manual_seed(42)
        model = build_model(cfg)
        model.train()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4)
        losses, ok_finite = [], True
        for step in range(steps):
            out = model(pixel_values_rgb=batch["pixel_values_rgb"],
                        pixel_values_ir=batch["pixel_values_ir"],
                        pixel_values_depth=batch["pixel_values_depth"],
                        depth_mask=batch["depth_mask"],
                        pixel_mask=batch["pixel_mask"], labels=labels)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            losses.append(float(out.loss))
            ok_finite = ok_finite and np.isfinite(losses[-1])
        check(f"[{arch}] geo3 训 {steps} 步 loss 有限无 NaN",
              ok_finite and len(losses) == steps,
              f"losses={['%.2f' % v for v in losses]} ({time.time()-t0:.0f}s)")
        del model, opt


# ---------------------------------------------------------------- T4 A/B 回归
def t4_linear_compat(img_size):
    print(f"\n[T4] A/B 回归：depth_encoding=linear 与改动前结构兼容", flush=True)
    base = load_yaml_config(os.path.join(LAB, "configs", "default.yaml"))
    cfg = _smoke_cfg(base, "rgbir_swin", img_size)
    cfg["model"]["depth_encoding"] = "linear"   # 默认值，旧 ckpt 兼容前提
    batch, labels = _make_batch(img_size, "linear")
    check("[linear] 数据侧仍为 (B,1,H,W) + mask",
          batch["pixel_values_depth"].shape[1] == 1)
    torch.manual_seed(0)
    model = build_model(cfg)
    sd = model.state_dict()
    # 旧结构关键层形状（改动前硬编码值）
    dep_keys = {k: tuple(v.shape) for k, v in sd.items()
                if "depth_encoder" in k and "conv.0.0" in k}
    check("[linear] DepthEncoder 首层仍为 (16,2,3,3)（旧 ckpt 形状兼容）",
          any(s == (16, 2, 3, 3) for s in dep_keys.values()), f"{dep_keys}")
    # state_dict 保存 → strict 加载往返
    import io
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    model.load_state_dict(torch.load(buf, weights_only=True), strict=True)
    check("[linear] state_dict strict 往返加载通过", True)
    # linear 前向 1 步（eval 免随机性）
    model.eval()
    with torch.no_grad():
        out = model(pixel_values_rgb=batch["pixel_values_rgb"],
                    pixel_values_ir=batch["pixel_values_ir"],
                    pixel_values_depth=batch["pixel_values_depth"],
                    depth_mask=batch["depth_mask"],
                    pixel_mask=batch["pixel_mask"], labels=labels)
    check("[linear] 前向 loss 有限（推理路径不受影响）",
          bool(np.isfinite(float(out.loss))), f"loss={float(out.loss):.3f}")
    # tri_swin 的 linear 深度塔首层形状（旧 ckpt 兼容）在 T2 已验证 (96,2,4,4)
    del model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--only", choices=["t1", "t2", "t3", "t4"], default=None,
                    help="只跑指定组（t3/t4 耗时较长，可分组跑）")
    args = ap.parse_args()
    if args.only in (None, "t1"):
        t1_encoding_math()
    if args.only in (None, "t2"):
        t2_forward_shapes()
    if args.only in (None, "t3"):
        t3_e2e_smoke(args.img_size, args.steps)
    if args.only in (None, "t4"):
        t4_linear_compat(args.img_size)
    print("\n" + "=" * 60, flush=True)
    if _failures:
        print(f"FAILED ({len(_failures)}):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

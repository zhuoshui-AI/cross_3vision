"""Feature-flow autopsy: print per-layer stats to find where mode collapse starts.

On server:
    python -u -m scripts.diag_features --data-root /path/to/AIC2026_Train_2000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data import MultiModalDataset, collate_fn
from src.model import build_model
from src.utils import load_yaml_config


def stats(name, t):
    if not torch.is_tensor(t):
        print(f"  {name}: not a tensor ({type(t)})", flush=True)
        return
    t = t.detach().float()
    print(f"  {name:40s} shape={tuple(t.shape)} "
          f"mean={t.mean():.4f} std={t.std():.4f} "
          f"min={t.min():.4f} max={t.max():.4f} "
          f"spatial_std={t.std(dim=(0,1) if t.ndim==4 else (0,)).mean():.4f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None, help="if set, load trained weights")
    ap.add_argument("--n-imgs", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_yaml_config(args.config)
    cfg["data"]["img_size"] = 512

    ds = MultiModalDataset(data_root=args.data_root, split_file=None,
                           img_size=512, train=False,
                           num_classes=int(cfg["model"]["num_labels"]))
    ds.ids = ds.ids[:args.n_imgs]
    items = [ds[i] for i in range(len(ds))]
    batch = collate_fn(items)
    batch = {k: (v.to(device) if torch.is_tensor(v) else
                 [{"class_labels": t["class_labels"].to(device),
                   "boxes": t["boxes"].to(device)} for t in v])
             for k, v in batch.items() if k in ("pixel_values_rgb",
                                                 "pixel_values_ir",
                                                 "pixel_values_depth",
                                                 "depth_mask", "pixel_mask",
                                                 "labels")}

    model = build_model(cfg).to(device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=False)
    model.eval()

    print(f"=== feature-flow diagnosis on {'trained' if args.ckpt else 'random-init'} model ===", flush=True)

    # ---- backbone internals ----
    enc = model.model.backbone.conv_encoder
    pv = {
        "rgb": batch["pixel_values_rgb"],
        "ir": batch["pixel_values_ir"],
        "depth": batch["pixel_values_depth"],
        "depth_mask": batch["depth_mask"],
    }
    with torch.no_grad():
        f_rgb = enc.rgb_backbone(pv["rgb"])
        f_ir = enc.ir_backbone(pv["ir"])
        f_fused = enc.cssa(f_rgb, f_ir)
        f_depth = enc.depth_encoder(pv["depth"], pv["depth_mask"])
        depth_mask_ds = enc.depth_encoder.downsample_mask(pv["depth_mask"], f_fused.shape[-2:])
        f_enh = enc.depth_fuse(f_fused, f_depth, depth_mask_ds)

        feature_map = f_enh
        mask = torch.ones((f_enh.shape[0], f_enh.shape[2], f_enh.shape[3]),
                          dtype=torch.bool, device=device)
        projected = model.model.input_projection(feature_map)
        flattened = projected.flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)

        # pos embeddings
        _, pos_list = model.model.backbone.position_embedding if hasattr(model.model.backbone, 'position_embedding') else (None, None)
        # DetrConvModel computes pos inside forward; replicate via the conv_model path
        features, pos_list_full = model.model.backbone(pv, batch["pixel_mask"])
        object_queries = pos_list_full[-1].flatten(2).permute(0, 2, 1)

        encoder_outputs = model.model.encoder(
            inputs_embeds=flattened,
            attention_mask=flattened_mask,
            object_queries=object_queries,
        )
        enc_out = encoder_outputs[0]

        B = flattened.shape[0]
        query_pos = model.model.query_position_embeddings.weight.unsqueeze(0).repeat(B, 1, 1)
        queries = torch.zeros_like(query_pos)
        decoder_outputs = model.model.decoder(
            inputs_embeds=queries,
            object_queries=object_queries,
            query_position_embeddings=query_pos,
            encoder_hidden_states=enc_out,
            encoder_attention_mask=flattened_mask,
        )
        seq_out = decoder_outputs[0]
        logits = model.class_labels_classifier(seq_out)
        pred_boxes = model.bbox_predictor(seq_out).sigmoid()

    print("\n[backbone outputs]", flush=True)
    stats("f_rgb (swin)", f_rgb)
    stats("f_ir (resnet+adapter)", f_ir)
    stats("f_fused (cssa)", f_fused)
    stats("f_depth", f_depth)
    stats("f_enh (depth_fuse out)", f_enh)

    print("\n[projection + encoder]", flush=True)
    stats("input_projection weight", model.model.input_projection.weight)
    stats("projected", projected)
    stats("flattened (encoder input)", flattened)
    stats("object_queries (2D pos)", object_queries)
    stats("encoder output", enc_out)

    print("\n[decoder + heads]", flush=True)
    stats("query_position_embeddings", model.model.query_position_embeddings.weight)
    stats("decoder output (seq)", seq_out)
    stats("logits", logits)
    stats("pred_boxes", pred_boxes)

    # per-query diversity: how different are the 100 queries?
    pb = pred_boxes[0]  # (Q, 4)
    lg = logits[0]       # (Q, C+1)
    qbox_std = pb.std(dim=0)
    qlogit_std = lg.std(dim=0)
    print(f"\n[query diversity] (sample 0 of batch)", flush=True)
    print(f"  pred_boxes std across 100 queries: {qbox_std.tolist()}", flush=True)
    print(f"  logits std across 100 queries (mean over classes): {float(qlogit_std.mean()):.4f}", flush=True)
    print(f"  num unique pred_boxes (rounded to 2dp): "
          f"{len(torch.unique(pb.round(decimals=2), dim=0))}/100", flush=True)
    print(f"  pred_boxes[0]: {pb[0].tolist()}", flush=True)
    print(f"  pred_boxes[50]: {pb[50].tolist()}", flush=True)
    print(f"  pred_boxes[99]: {pb[99].tolist()}", flush=True)

    # encoder spatial diversity: is each spatial token the same?
    eo = enc_out[0]  # (L, C)
    print(f"\n[encoder spatial diversity] (sample 0)", flush=True)
    print(f"  encoder output std across spatial tokens (mean over C): "
          f"{float(eo.std(dim=0).mean()):.4f}", flush=True)
    print(f"  num unique spatial tokens (rounded to 2dp): "
          f"{len(torch.unique(eo.round(decimals=2), dim=0))}/{eo.shape[0]}", flush=True)

    if float(eo.std(dim=0).mean()) < 1e-3:
        print("\n  >>> ENCODER OUTPUT IS (NEARLY) CONSTANT across space! "
              "This causes all decoder queries to attend to the same value -> mode collapse.",
              flush=True)


if __name__ == "__main__":
    main()

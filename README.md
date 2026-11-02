# Multimodal Swin-DETR for Urban-Scene Object Detection (RGB + Infrared + Depth)

Three-modality object detector that fuses **RGB**, **infrared (IR)**, and **depth**
via attention, on top of a HuggingFace DETR head. Built for the 12-class urban
scene detection benchmark (person, boat, animal, seat, sign, bicycle, car,
ball, light, garbage_can, uav, tricycle).

## Architecture

Two architectures, selected by `model.arch`:

### rgbir_swin (default)

RGB + IR early-concat (4 channels) flow through a single ImageNet-22k Swin-T
tower, producing multi-scale C1~C4 features. A SimpleFPN (top-down + PAN
bottom-up) builds P2~P5 at `d_model` channels. A lightweight depth CNN
produces stride-32 depth features injected into P5 via a single
cross-attention layer (Q = fused RGB+IR, K=V = depth). Selected FPN levels
(default P2~P5) are flattened into tokens and fed to the DETR encoder.

### tri_swin (TMSC-Det)

Three per-modality Swin towers (RGB / IR / Depth), each capacity-tuned to its
modality's information content. Per-scale HCMAF (Hierarchical Cross-Modal
Attention Fusion) blocks jointly attend over all modalities at each stage and
feed refined features back into each tower's next stage. The fused stage
outputs go through SimpleFPN → P2~P5 and on to the DETR encoder. The IR tower
can swap to AnyThermal (DINOv2-distilled ViT-B/14).

Common to both:
- **RGB+IR backbone**: `swin_tiny_patch4_window7_224` (ImageNet-22k). For
  rgbir_swin the patch_embed is adapted to 4 channels (channels 0-2 keep RGB
  weights, channel 3 = mean of the 3). First 2 stages frozen.
- **Depth branch**: lightweight 5-stride-2 CNN, 2-channel input
  (depth + validity mask), no pretraining (depth is geometric).
- **Depth late fusion**: single cross-attention layer (Q = fused RGB+IR,
  K=V = depth). Zero-initialized so it starts as identity and depth is
  injected gradually (avoids query collapse).
- **DETR head**: `DetrForObjectDetection` (100 queries, d_model=256,
  6+6 encoder/decoder layers, auxiliary losses). Hungarian matcher +
  CE/L1/GIoU losses are reused verbatim; only the conv-encoder is replaced
  and `forward` overridden to accept a dict of per-modality tensors.
- **COCO warm-start**: encoder / decoder / bbox_predictor / query_position
  inherited from `facebook/detr-resnet-50` (small-data convergence recipe).
- **SDPA attention**: O(N) memory, lets P2-level (~33k-token) training fit on
  24 GB GPUs.

## File layout

```
lab/
├─ configs/default.yaml         # single source of truth (data/model/train/loss/infer)
├─ requirements.txt
├─ scripts/
│  ├─ download_pretrained.sh   # Swin-22k + optional AnyThermal
│  └─ make_submission.sh       # inference + zip predictions
├─ src/
│  ├─ __init__.py
│  ├─ data.py                  # MultiModalDataset + collate_fn (cxcywh labels)
│  ├─ backbones.py             # SwinModalityBackbone / DepthEncoder / AnyThermalBackbone
│  ├─ fusion.py                # DepthLateFusion + HCMAFBlock + SimpleFPN
│  ├─ model.py                 # RGBIRSwinDETR + TriModalSwinDETR + build_model
│  ├─ losses.py                # build_detr_config re-export + box converters
│  ├─ eval.py                  # torchmetrics mAP@50-95
│  ├─ inference.py             # write per-stem `<stem>.txt` per spec
│  ├─ train.py                 # manual loop + AMP + cosine + warmup + EMA
│  └─ utils.py                 # seed / yaml / ckpt / logging / EMA
└─ outputs/                    # logs / ckpt / predictions (created at runtime)
```

## Data layout

Set `data.data_root` in the config (or pass `--data-root`). Expected under it:

```
<data_root>/
├─ visible/<stem>.{jpg|png}    # 3-ch 8-bit visible (RGB)
├─ infrared/<stem>.{jpg|png}   # 3-ch 8-bit thermal (the 1st channel is taken)
├─ depth/<stem>.png            # 1-ch 16-bit mm (0 or <10 mm → invalid)
├─ labels/<stem>.txt          # `cls cx cy w h` normalized
└─ splits/{train,val,test}.txt # one stem per line (optional)
```

Labels are auto-converted to **cxcywh normalized** internally (the DETR matcher
and `loss_boxes` call `center_to_corners_format`, so cxcywh is required).

## Install

```bash
pip install -r requirements.txt
# Swin-22k weights (optional offline copy; timm hub fetch also works):
bash scripts/download_pretrained.sh
```

## Train

```bash
python -m src.train --config configs/default.yaml --data-root /path/to/data
```

- Manual loop (not HF Trainer) for explicit control over the multimodal batch.
- Two param groups: backbones at `train.backbone_lr` (1e-5), fusion + DETR
  heads at `train.lr` (1e-4). AdamW, cosine schedule with 1k-step linear
  warmup, grad accum 2 (effective batch 16), bf16/fp16 AMP, grad clip 0.1.
- EMA (`ema_decay: 0.9997`; set 0 to disable). Validation every `val_interval`
  epochs; best mAP@50-95 saved to `outputs/ckpt/best.pth`.
- Resume: `--resume outputs/ckpt/last.pth`.

## Inference / submission

```bash
python -m src.inference --config configs/default.yaml \
    --ckpt outputs/ckpt/best.pth --data-root /path/to/test_data
# or:
bash scripts/make_submission.sh
```

Writes one `<stem>.txt` per image to `inference.output_dir` (default
`./outputs/predictions`). Each line: `cls cx cy w h conf`
(cxcywh normalized + confidence), sorted by score desc, ≤100 boxes/image,
filtered at `inference.conf_threshold` (0.05). Empty file written when no
detections survive.

## Notes / known limitations

- **Swin resolution lock**: timm's Swin precomputes its window-attention mask
  for `data.img_size` (512). Feed other sizes only if you rebuild the model
  (the `patch_embed.strict_img_size=False` flag tolerates input sizes divisible
  by the patch size, but the mask mismatch still requires a fixed size).
- **AnyThermal** (tri_swin IR tower): switch `model.tri_ir_backbone` to
  `"anythermal"` and provide weights (`scripts/download_pretrained.sh
  WITH_ANYTHERMAL=1`) to enable the DINOv2-distilled ViT-B/14 thermal backbone.
- **Label format**: spec labels are `[cls, cx, cy, w, h]`; the dataset converts
  to cxcywh normalized for DETR. Inference output is also cxcywh (native DETR
  `pred_boxes`), so no re-conversion is needed on the spec side.

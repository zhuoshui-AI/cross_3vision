# Multimodal Swin-DETR for Urban-Scene Object Detection (RGB + Infrared + Depth)

Three-modality object detector that fuses **RGB**, **infrared (IR)**, and **depth**
via attention, on top of a HuggingFace DETR head. Built for the 12-class urban
scene detection benchmark (person, boat, animal, seat, sign, bicycle, car,
ball, light, garbage_can, uav, tricycle).

## Architecture

```
RGB ─► Swin-S (ImageNet-22k) ─► F_rgb (768ch, /32)
                                    │
IR  ─► ResNet50-1ch (ImageNet, conv1 summed) ─► F_ir (768ch, /32)
                                    │
                       CSSA Fusion (channel-switching + spatial attn)
                                    │  ◄── front-half: RGB↔IR info exchange
                              F_fused (256ch, /32)
                                    │
Depth ─► DepthEncoder (depth+mask CNN) ─► F_depth (256ch, /32)
                                    │
                       Depth Late Fusion (cross-attn: Q=fused, K=V=depth)
                                    │  ◄── back-half: depth injection
                              F_enh (256ch, /32)
                                    │
                       DETR encoder + decoder + heads (reused from HF)
                                    │
                            logits (100×13) + pred_boxes (100×4 cxcywh)
```

- **RGB branch**: `swin_small_patch4_window7_224` with ImageNet-22k weights
  (`timm` tag `ms_in22k`). First 2 stages frozen. Outputs stride-32 features
  at 768 channels.
- **IR branch**: ImageNet ResNet50-V2 with `conv1` adapted to 1-channel by
  summing weights across the original 3 input channels (equivalent to feeding
  the IR grayscale through all 3 RGB weights). A 1×1 adapter projects layer4
  (2048-dim) to the Swin dim (768). AnyThermal swap-in is stubbed.
- **CSSA Fusion** (Cao et al., CVPRW 2023): per-modality SE channel scorers
  switch low-scoring channels between RGB and IR (threshold τ=0.5), a CBAM
  spatial gate blends the switched streams, and a 1×1 conv projects to d_model.
- **Depth branch**: lightweight 5-stride-2 CNN, 2-channel input (depth + mask),
  no pretraining (depth is geometric). `DepthEncoder.downsample_mask` produces
  the cross-attn key_padding_mask.
- **Depth Late Fusion**: single cross-attention layer; fused RGB+IR = queries,
  depth = keys/values; the depth validity mask gates attention.
- **DETR head**: `DetrForObjectDetection` (100 queries, d_model=256, 6+6
  encoder/decoder layers, auxiliary losses). Hungarian matcher + CE/L1/GIoU
  losses are reused verbatim — `MultiModalSwinDETR` only replaces the
  conv-encoder and overrides `forward` to feed a dict of per-modality tensors.

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
│  ├─ backbones.py             # SwinRGBBackbone / IRBackbone / DepthEncoder
│  ├─ fusion.py                # CSSAFusion (front) + DepthLateFusion (back)
│  ├─ model.py                 # MultiModalBackbone + MultiModalSwinDETR + build_model
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
- **AnyThermal**: stubbed via `_AnyThermalAdapter` (raises cleanly). Switch
  `model.ir_backbone` to `"anythermal"` and provide weights to enable.
- **Label format**: spec labels are `[cls, cx, cy, w, h]`; the dataset converts
  to cxcywh normalized for DETR. Inference output is also cxcywh (native DETR
  `pred_boxes`), so no re-conversion is needed on the spec side.

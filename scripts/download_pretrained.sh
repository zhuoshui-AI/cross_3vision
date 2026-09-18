#!/usr/bin/env bash
# Download pretrained backbones: Swin (ImageNet-22k) + AnyThermal (optional IR).
#
# New timm exposes ImageNet-22k weights via the tag `swin_small_patch4_window7_224.ms_in22k`;
# the build code fetches that automatically when model.swin_pretrained=true. This
# script additionally drops the original 22k checkpoint locally for offline use
# (the code prefers the local file when model.swin_pretrained_path points to it).
set -e

mkdir -p ./weights
cd ./weights

# ---- 1. Swin Transformer (RGB branch, ImageNet-22k) ----
# Official 22k checkpoint; mapped into timm's state-dict via strict=False.
SWIN_URL="https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth"
if [ ! -f swin_small_patch4_window7_224_22k.pth ]; then
  echo "Downloading Swin-S ImageNet-22k (offline copy)..."
  wget -q "$SWIN_URL" -O swin_small_patch4_window7_224_22k.pth || \
    curl -sSL "$SWIN_URL" -o swin_small_patch4_window7_224_22k.pth
fi
# Also let timm cache its own 22k tag so train-time load is network-free:
python -c "import timm; timm.create_model('swin_small_patch4_window7_224', pretrained=True, pretrained_cfg='ms_in22k')" || true

# ---- 2. AnyThermal (IR branch, optional) ----
# CMU ICRA 2026 DINOv2-distilled ViT-B/14 thermal backbone.
# Requires `pip install huggingface-hub`.
if [ "${WITH_ANYTHERMAL:-0}" = "1" ]; then
  echo "Downloading AnyThermal from HuggingFace..."
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='theairlabcmu/AnyThermal', local_dir='./AnyThermal')"
fi

echo "Pretrained weights ready in ./weights/"

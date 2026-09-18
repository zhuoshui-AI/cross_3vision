#!/usr/bin/env bash
# Run inference on test set and package predictions into submission zip.
set -e

CONFIG="${1:-configs/default.yaml}"
CKPT="${2:-outputs/ckpt/best.pt}"

python -m src.inference --config "$CONFIG" --ckpt "$CKPT" --split test

cd ./outputs/predictions
zip -r ../submission.zip .
echo "Submission: outputs/submission.zip"

#!/usr/bin/env bash
set -euo pipefail

# ===== 宏参数（可直接修改） =====
DEFAULT_WEIGHTS_ROOT="/root/autodl-tmp/MM-MOE/runs/final_MMMOE"
DEFAULT_OUT_ROOT="/root/autodl-tmp/MM-MOE/runs/val/final_test"
DEFAULT_DATASET_YAML="/root/autodl-tmp/MM-MOE/ultralytics/cfg/datasets/myDualDataV.yaml"
DEFAULT_DEVICE="0"

# 支持命令行覆盖：
# bash val_all_bash.sh <weights_root> <out_root> <dataset_yaml> [device]
WEIGHTS_ROOT="${1:-$DEFAULT_WEIGHTS_ROOT}"
OUT_ROOT="${2:-$DEFAULT_OUT_ROOT}"
DATASET_YAML="${3:-$DEFAULT_DATASET_YAML}"
DEVICE="${4:-$DEFAULT_DEVICE}"

echo "[INFO] WEIGHTS_ROOT=$WEIGHTS_ROOT"
echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] DATASET_YAML=$DATASET_YAML"
echo "[INFO] DEVICE=$DEVICE"

python val_all.py \
  --weights_root "$WEIGHTS_ROOT" \
  --out_root "$OUT_ROOT" \
  --dataset_yaml "$DATASET_YAML" \
  --pattern "best.pt" \
  --imgsz 640 \
  --channels 6 \
  --use_simotm RGBRGB6C \
  --eval_batch 16 \
  --device "$DEVICE" \
  --out_name all_weights_joint_eval
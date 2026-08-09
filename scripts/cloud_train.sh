#!/usr/bin/env bash
# Lever C cloud training: train a stronger 16-token base on synthetic ARC data.
# Provider-agnostic: run on ONE rented 80GB GPU (A100/H100). ~$1.5-3/h.
#
#   BASE=Qwen/Qwen3-8B  KAGGLE_JSON=~/.kaggle/kaggle.json  bash scripts/cloud_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=${BASE:-Qwen/Qwen3-8B}
OUT=${OUT:-models/$(basename "$BASE")_grids16}
SFT_OUT=${SFT_OUT:-models/$(basename "$BASE")_arc_sft}
MAX_EX=${MAX_EX:-200000}
EPOCHS=${EPOCHS:-1}

echo "== 1. deps =="
pip install -q "torch" "transformers>=5" "peft>=0.13" accelerate datasets numpy kaggle

echo "== 2. data (needs ~/.kaggle/kaggle.json) =="
[ -d data/arc-agi-2/training ] || bash scripts/download_data.sh || true
if [ ! -d data/nvarc/synthetic ]; then
  mkdir -p data/nvarc/synthetic
  kaggle datasets download -d sorokin/nvarc-synthetic-puzzles -p data/nvarc/synthetic --unzip
fi
# (optional, larger) augmented 3.2M:
# kaggle datasets download -d sorokin/nvarc-augmented-puzzles -p data/nvarc/augmented --unzip

echo "== 3. vocab-cut $BASE -> 16 tokens =="
[ -d "$OUT" ] || python3 -m train.cut_vocab --base "$BASE" --out "$OUT"

echo "== 4. SFT on synthetic + ARC train =="
# LoRA (fits one 80GB GPU). Add --full-ft for full fine-tune (needs ~80GB, tight).
python3 -m train.sft_train --base "$OUT" \
  --sources data/nvarc/synthetic data/arc-agi-2/training \
  --max-examples "$MAX_EX" --epochs "$EPOCHS" \
  --lr 1e-4 --grad-accum 16 --lora-r 256 --lora-alpha 256 \
  --out "$SFT_OUT"

echo "== 5. (optional) merge LoRA for a standalone checkpoint =="
python3 - <<PY
import sys; sys.path.insert(0,"src")
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
b=AutoModelForCausalLM.from_pretrained("$OUT", dtype=torch.bfloat16)
m=PeftModel.from_pretrained(b, "$SFT_OUT").merge_and_unload()
m.save_pretrained("${SFT_OUT}_merged")
print("merged ->", "${SFT_OUT}_merged")
PY

echo "== done. Download ${SFT_OUT}_merged, then locally:"
echo "   python scripts/repro_nvarc.py --model ${SFT_OUT}_merged --limit 60 --aug-n 16 --eval-aug-n 3 --dfs-prob 0.1"
echo "   (compare pass@2 to the 4B baseline; if better, upload as a Kaggle model + point kaggle_submit at it)"

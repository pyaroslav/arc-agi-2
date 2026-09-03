#!/bin/bash
# Option-A probe: waits until the GPU is free of other compute jobs, then
# launches the 12-day corpus-scale continued-FT (550k rows ~ 2.5B tokens).
# NEVER includes arc2_evaluation6 (eval split) in training sources.
cd "$(dirname "$0")/.." || exit 1
echo "$(date -u) waiting for GPU to be free..."
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 300; done
echo "$(date -u) GPU free; power limit: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  nohup python3 -m train.sft_train --base data/nvarc/qwen3_4b_sft \
  --arrow-sources data/nvarc/augmented/nvarc_full data/nvarc/augmented/nvarc_training \
                  data/nvarc/augmented/arc2_training data/nvarc/augmented/rearc \
                  data/nvarc/augmented/concept data/nvarc/augmented/mini \
  --max-examples 550000 --epochs 1 --batch-size 1 --grad-accum 16 \
  --lr 5e-5 --lora-r 256 --lora-alpha 256 --save-every 50 --offload --resume \
  --out models/Qwen3-4B_corpus_lora > outputs/sft_4b_corpus.log 2>&1 < /dev/null &
sleep 5
pgrep -f "train.sft_train" | head -1 | awk '{print "trainer PID:", $1}'

#!/bin/bash
# Option-A probe launcher v2: waits for (a) packed data cache built and
# (b) GPU free, then trains with instant data load (seconds-wide GPU race
# window instead of hours). Retries on OOM/crash; --resume makes it idempotent.
cd "$(dirname "$0")/.." || exit 1
CACHE=data/nvarc/augmented/packed_550k.npz
while [ ! -f "$CACHE" ]; do sleep 120; done
echo "$(date -u) cache ready"
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 300; done
  echo "$(date -u) attempt $attempt: GPU free (limit $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)); launching"
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
    python3 -m train.sft_train --base data/nvarc/qwen3_4b_sft \
    --arrow-sources data/nvarc/augmented/nvarc_full \
    --data-cache "$CACHE" \
    --max-examples 550000 --epochs 1 --batch-size 1 --grad-accum 16 \
    --lr 5e-5 --lora-r 256 --lora-alpha 256 --save-every 50 --offload --resume \
    --out models/Qwen3-4B_corpus_lora >> outputs/sft_4b_corpus.log 2>&1
  rc=$?
  echo "$(date -u) trainer exited rc=$rc"
  [ $rc -eq 0 ] && break
  sleep 600
done
echo "$(date -u) probe launcher done"

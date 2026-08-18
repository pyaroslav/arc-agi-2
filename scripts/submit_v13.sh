#!/bin/bash
# One-shot: v13 (baseline + never-[[0]] fallback) — submits after 00:00 UTC Aug 13
while [ "$(date -u +%Y%m%d)" -le "20260812" ]; do sleep 300; done
kaggle competitions submit -c arc-prize-2026-arc-agi-2 \
  -k yaroslavperventsev/arc-agi2-baseline-fallback -v 1 -f submission.json \
  -m "v13: mother baseline + our never-[[0]] fallback (identity/mode guesses for unreached tasks; strictly nonnegative patch)"
sleep 60
kaggle competitions submissions -c arc-prize-2026-arc-agi-2 | head -3

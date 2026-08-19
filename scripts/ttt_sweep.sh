#!/bin/bash
# TTT-recipe sweep: 3 configs x 30 tasks, sequential, vs baseline 22.8% (6.83/30).
cd "$(dirname "$0")/.." || exit 1
run() {
  name=$1; shift
  echo "=== TTT config $name: $* ==="
  python scripts/repro_nvarc.py --model data/nvarc/qwen3_4b_sft --limit 30 \
    --aug-n 8 --dfs-prob 0.2 "$@" --pkl-out "outputs/cache_ttt_${name}.pkl" \
    > "outputs/eval_ttt_${name}.log" 2>&1
  grep "pass@2 (score_kgmon)" "outputs/eval_ttt_${name}.log" | tail -1 | sed "s/^/[$name] /"
}
run ep2       --ttt-epochs 2
run lr1e4     --lr 1e-4
run ep2lr25   --ttt-epochs 2 --lr 2.5e-5
echo "TTT_SWEEP_DONE (baseline: 6.83/30 = 22.8%)"

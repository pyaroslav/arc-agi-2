# Phase-3 Improvement Roadmap — ARC-AGI-2 (24% → ~27% target)

Produced by an adversarially-verified multi-agent research workflow (33 ideas → 6 kept),
grounded in our actual code + the candidate cache. Raw output: `docs/roadmap_raw.json`.

> **Honest ceiling:** ~24% → **~26–28% pass@2 high-confidence; ~27% ±2pp**. Reaching
> **30% requires the big bet (induction) landing net-new coverage** (~30% chance it adds ~0)
> *and* selection gains stacking without saturating. Frontier-API methods are banned (offline).

## The load-bearing insight
The dominant ARC-AGI-2 failure is **coverage** (the correct grid is never generated), **not
mis-ranking**. First local oracle reading (n=6): generation ceiling **33%**, achieved pass@2
**17%** → both coverage *and* selection lose points. The **#1 gating measurement** is
oracle-in-set vs achieved pass@2 on a 60–120 task cache (`scripts/analyze_selection.py`):
- big oracle−achieved gap → invest in **selection** (cheap), or
- low oracle → invest in **generation/coverage** (TTFT, data, induction — harder).

## Budget reality
L4×4 ≈ **360–540 s/task** realized (5090 ≈ 180 s/task, L4 ~2–3× slower) → only **~1.3–2× headroom**
over the current pipeline. Higher `m` and a second 8B model are *real* budget trades, not free.

## Prioritized backlog

| # | Idea | Effort | Δpp (realistic) | Notes |
|---|------|--------|-----------------|-------|
| 1 | **FREE selection sweep** on cached 8-aug scores (count-weight 0/1/2 × sum/arith/geom-mean) + oracle measurement | S (CPU) | +0.3–1.0 | **Gates everything.** `scripts/sweep_selection.py` |
| 2 | Raise re-scoring `m` 8→16 (recompute `calc_scores`) | S–M (GPU) | +0.3–1.0 | **Budget-gated**; only if #1 shows recoverable mis-ranking |
| 3 | Per-task **LoRA EMA** / last-K averaging during TTFT | M | 0–+0.7 | near-free variance reduction; cosine-to-0 schedule shrinks benefit |
| 4 | **LoRA rank/alpha hygiene** sweep (r{128,256}×α{16,32}, keep embed+lm_head) | S | 0–+0.5 | main value = iso-accuracy at lower s/task → frees budget for #2 |
| ~~5~~ | ~~Guarded symbolic verifier~~ **DROPPED — measured 0%** | — | **0** | A compact ~15-rule DSL (D4, colormap, tiling, scale, crop, depth-2) finds **0 consistent programs on all 120 eval tasks**. ARC-AGI-2 defeats brute-force DSL. `scripts/eval_symbolic.py` |
| 6 | **BIG BET: offline BARC induction-8B** → sample/execute-verify programs → inject verified grids, gated to low-confidence tasks | L | +0.5–1.5 (≈30% chance ~0) | only path to **net-new coverage**; 2nd model + sandbox + budget pressure; run LAST; only item justifying cloud GPU |

## Quick wins (do first on the 5090, no base re-SFT)
1. **Decode a 60–120 task candidate cache** (one decode pass) — prerequisite for all cache experiments.
2. **#1 free selection sweep** + oracle gap (gates the rest).
3. **#4 LoRA r/α hygiene** sweep (free up per-task time).
4. **#3 LoRA EMA** (composes with everything).
5. `fix-background` color perms A/B (`permute_rnd_keep0_` = `np.random.permutation(9)+1`) — unproven vs NVARC's full 10-color perms; small re-decode.

## Big bets (run last)
- **#6 BARC induction-8B** parallel program-synthesis branch (net-new coverage).
- **#2 m→16/24** conditional escalation (only if oracle gap proves it).

## Execution order
1. Finish the 30-task reproduction → confirm baseline ≈ NVARC.
2. Decode a 60–120 task cache.
3. Run `analyze_selection.py` + `sweep_selection.py` → **decide coverage-bound vs selection-bound**.
4. Ship the best free selector; then #4→#3 on GPU; then gate #2.
5. Evaluate #5 (symbolic) on the cache; ship only if net-new ≥3/120 or strictly non-negative.
6. Only if the gap is coverage-bound: invest in #6 induction (the real shot at 30%).

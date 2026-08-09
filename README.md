# ARC-AGI-2 — Competitive Offline Solver (ARC Prize 2026)

A from-scratch, prize-eligible offline pipeline for the
[ARC Prize 2026 – ARC-AGI-2 Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2),
built by **forking and improving the 2025 winner (NVARC)**.

Recipe (proven, NVARC 1st place / 24% private LB): fine-tune a small open LLM on
synthetic ARC data → **test-time train** a per-task LoRA → **batched-DFS decode**
→ **augment-and-vote** select 2 answers (pass@2). Inference fits Kaggle's L4×4,
<12h, no internet.

> **Realistic target:** top of the *offline* leaderboard (~25–35%). The 85% bonus
> is currently unreachable offline (best ever = 24%); the 77–95% public numbers
> use frontier APIs banned in the Kaggle sandbox. See `plans/…goblet.md`.

## Status
- [x] Repo scaffold + official ARC-AGI-2 data (1000 train / 120 eval); core modules + tests (9/9 green)
- [x] GPU toolchain validated on Blackwell/CUDA13 (bf16 + SDPA + peft LoRA; no unsloth/flash-attn needed)
- [x] **Checkpoint gate → Path A:** NVARC's winning 4B checkpoint `sorokin/qwen3_4b_grids15_sft139` is public; downloaded to `data/nvarc/qwen3_4b_sft/`
- [x] **NVARC ported to transformers+peft** (`src/nvarc/`): exact 16-token tokenizer, TTFT (LoRA r=256 rslora), `turbo_dfs`, NLL voting. Caught the transformers-5.9 tokenizer trap (drops `user`/`assistant` → custom encoder)
- [x] **Reproduced the winner: 22.8% pass@2** (30 eval tasks, `aug_n=8`) ≈ NVARC's 24% private; ~5.4h/240 projected on L4×4
- [x] **Phase-3 roadmap** (`docs/ROADMAP.md`): ~27% realistic ceiling; **coverage-bound** (oracle 37% vs achieved 29%); free selection sweep = no gain; **symbolic DSL = 0% on ARC-AGI-2 (dropped)**
- [x] **Kaggle submission notebook + driver** (`src/nvarc/kaggle_submit.py`, `notebooks/kaggle_submission.ipynb`) — schema-validated locally
- [~] 120-task definitive baseline + full cache (running)
- [ ] Coverage levers: `aug_n=16` test (next), then induction big-bet (`barc0/Llama-3.1-ARC-Potpourri-Induction-8B`) for net-new coverage → 30%
- [ ] Kaggle dry-run + leaderboard submission (**user step**) + open-source writeup

### What's working right now
```bash
python -m pytest tests/ -q                                          # 9/9 green, no GPU
python scripts/repro_nvarc.py --limit 5 --aug-n 8                   # reproduce on 5090 (needs checkpoint)
python scripts/analyze_selection.py && python scripts/sweep_selection.py   # oracle/selection gap (CPU)
```

### To submit on Kaggle (user)
Attach `sorokin/qwen3_4b_grids15_sft139` (Model) + this repo's `src/` (Dataset) + the
competition data to `notebooks/kaggle_submission.ipynb`, internet off, then run/submit.

## Layout
```
src/arc/    grids.py tokenizer.py data.py  (+ ttft.py decode.py vote.py model.py to come)
src/eval/   metrics.py  run_local_eval.py
configs/    *.yaml      scripts/  notebooks/  tests/  data/ (gitignored)
```

## Quickstart
```bash
pip install -r requirements.txt          # core; GPU stack: scripts/setup_gpu_env.sh
python -m pytest tests/ -q               # core foundation (no GPU needed)
bash scripts/download_data.sh            # NVARC artifacts (needs ~/.kaggle/kaggle.json)
```

## What you (the human) need to provide
1. **Kaggle API token** at `~/.kaggle/kaggle.json` — for NVARC datasets + the
   checkpoint gate (whether the SFT'd Qwen3-4B weights are public → Path A vs B).
2. A Kaggle account joined to the competition (for L4×4 + submission).

See `plans/do-deep-research-and-memoized-goblet.md` for the full plan, sources,
and the concrete NVARC hyperparameters.

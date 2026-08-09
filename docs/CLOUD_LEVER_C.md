# Lever C — Cloud training plan (stronger base → more coverage)

**Goal:** raise the *generation coverage* that all inference-time levers cannot (we
proved that at ~33%), by training a stronger 16-token ARC base on synthetic data —
the 2025 winner's explicit path ("bigger/newer models will take over").

## What's built & validated (local, no cloud)
- `src/train/sft_data.py` — SFT corpus from the **103k NVARC synthetic puzzles**
  (30 pairs each) + ARC-AGI-2 train, in the 16-token completion-only format. ✓
- `src/train/cut_vocab.py` — reduce a fresh base (e.g. Qwen3-8B) to **16 tokens**
  (validated: cut model runs with our `ArcTokenizer`). ✓
- `src/train/sft_train.py` — NVARC-recipe SFT (LoRA or full-FT, bf16, packing). ✓
- `scripts/cloud_train.sh` — one-command run on a single rented 80 GB GPU. ✓
- Result merges and drops straight into `kaggle_submit.py` (same `ArcTokenizer`). ✓

## The honest finding that shapes this
Continued-FT of NVARC's **4B on its own synthetic data → loss ≈ 0.02 (no gain)**: the
4B already mastered that corpus. So the bet must be a **bigger base** (more capacity)
or **genuinely harder/new data** — not more FT of the 4B on the same data.

## Budget options (one 80 GB A100/H100 ≈ $1.5–3/h)

| Option | What | Cloud cost | Odds of beating the 4B (~33%) |
|---|---|---|---|
| **A. LoRA-SFT cut Qwen3-8B** | bigger base, LoRA on synthetic | **~$50–200** (1 GPU, ~1 day) | **low–moderate** — LoRA may not teach a fresh base the grid format well enough |
| **B. Full-FT cut Qwen3-8B** | bigger base, full FT on full corpus | **~$800–2500** (multi-GPU, days) | **moderate–good** — the real "bigger model" attempt; matches the winner's compute scale |
| **C. Harder-data + continued-FT 4B** | generate compositional tasks the 4B fails, FT on those | **~$0–100** (mostly the 5090) | **uncertain** — targets coverage directly; needs a data-gen research step |

**Honest reality:** the winner spent ~$1.7k+ on the 4B and hints at a stronger model
coming. A *serious* shot at the winning ~42% is **Option B-scale compute (~$1k+)**.
Options A/C (~$100s) are worthwhile experiments but are gambles that may not beat the
already-strong 4B.

## Recommended sequence
1. **Cheap probe first (Option A, ~$100):** rent one 80 GB GPU, run `cloud_train.sh`
   with `BASE=Qwen/Qwen3-8B` (LoRA). Eval the merged model locally:
   `python scripts/repro_nvarc.py --model <merged> --limit 60 --aug-n 16 --dfs-prob 0.1`.
   If it beats ~33% → scale to Option B. If not → we've learned cheaply.
2. **If committing to win (Option B):** full-FT the cut 8B on the full corpus (+ the
   3.2M augmented set + harder generated data) on a multi-GPU node; budget ~$1k+.
3. Upload the winning weights as a Kaggle model; point `kaggle_submit.py` at it; submit.

## To run (on your rented instance)
```bash
git clone <this repo> && cd ARC-AGI-2
cp <your kaggle.json> ~/.kaggle/kaggle.json
BASE=Qwen/Qwen3-8B MAX_EX=200000 bash scripts/cloud_train.sh
# downloads data, cuts vocab, LoRA-SFTs, merges -> models/Qwen3-8B_arc_sft_merged
```

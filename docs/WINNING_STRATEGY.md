# How to actually WIN ARC Prize 2026 — strategy from the leaderboard + discussions

Grounded in: all 44 Kaggle discussion threads (`data/discussions/`), the 2025 winner's
own posts (cpmpml / J-F Puget), the 2025 technical report, and our own measurements.

## 1. The leaderboard reality (June 2026)
- **Top public LB ≈ 42.6%.** It was "stuck at ~33.9%" for weeks, now climbing as teams optimize.
- **Public NVARC-rehash notebooks score ~28–33%** (the winner's 2025 model, reused). Submissions are **non-deterministic** — the *same* code resubmitted scores 30.3% → 32.2%.
- **Our port ≈ 29% local → ~26–27% expected LB.** Mid-pack, but on the most LB-robust architecture.

## 2. What the 2025 WINNER says (cpmpml, posting directly)
- "**Simple tweaks to our last-year winning solution won't be enough to win this time.** Show me you can improve over what we shared, say with a **35+** score."
- The NVARC/Qwen approach **plateaus ~34–35%**; **winning needs ~42%**.
- "This year's environment fits **significantly larger models** … bigger and newer models will take over."
- Their 2025 numbers: **30.5% pass@2 but 40% pass@10 on eval → 27% public LB → 24% private. "Our scorer missed 10%."**
- On data: "ARC-AGI-2's hard puzzles should be **augmented in a different manner** than just rotation/colour permutation."
- On DSL: "**I never believed in the DSL approach for ARC-AGI-2.**" (Matches our measured 0% symbolic.)

## 3. The eval→LB gap (calibrate to THIS, not local eval)
| approach | eval | LB | drop |
|---|---|---|---|
| **NVARC (LLM+TTT)** | 30.5% pass@2 | **27%** | small ✅ |
| TRM (recursive) | 22% | 10.8% | ~50% ❌ |
| TOPAS (deep-recursive ~100M) | 36% local | 11–19% | huge ❌ (+ timeouts) |

**The LLM+TTT family generalizes best.** Recursive/small-model approaches overfit the public set
badly. → Our architecture choice is validated; optimize it, don't abandon it for TRM/TOPAS hype.

## 4. The winning levers (in priority order)

### Lever A — Close the scorer/pool gap  (+~10pp, the winner's own number)
The correct answer is often generated but **not selected** (pass@10 40% vs pass@2 30.5%). This
*corrects an earlier wrong conclusion of ours*: our local "selection sweep = 0" was on a **too-shallow
pool** (aug_n=8, DFS prob>0.2). At full pool depth the gap is real and worth ~10pp. Two parts:
1. **Widen the candidate pool** → raise pass@N: full `aug_n=16`, more eval views, **lower DFS threshold (prob>0.1–0.05)**, more sampling. (We measured these add *oracle* coverage only modestly per-setting, but the winner's 40% pass@10 shows depth matters — push it.)
2. **Build a better selector** → convert pass@N→pass@2: a **learned verifier** (train a small model to rank candidates by correctness), execution/consistency re-scoring, or a 2nd-model cross-score. This is where the winner says 10pp is hiding.

### Lever B — Bulletproof time management  (recovers the LB drop that kills teams)
Teams lose **huge** LB score to **timeouts → `[[0]]` placeholders** (one reported the algo "bailed on 56% of second attempts").
- **Use a true DYNAMIC shared queue across the 4 GPUs — NOT static/round-robin partitioning** (a top competitor "watched 12h evaporate because one GPU got the hard puzzles"). ⚠️ **Our current `kaggle_submit.py` uses round-robin = static partitioning — FIX THIS** to a shared `mp.Queue` workers pull from.
- **Never emit `[[0]]`** — always fall back to a real guess (base-model greedy / most-frequent train output) so every task scores something.
- Separate `MAX_TTT_TOKENS` (training, logit-bound) from `MAX_INFERENCE_TOKENS` (KV-cache-bound).
- **Submit repeatedly** (non-deterministic; resubmits score higher). Save-then-submit (accelerator auto-off bug burns the 1/day limit).

### Lever C — A bigger/better base + smarter synthetic data  (the path to actually WIN, needs cloud)
The winner's explicit path: 96GB L4x4 fits larger models; bigger+newer bases will win.
- **Fine-tune a stronger base** (7–8B, or a strong 2026 open reasoning model) on **harder, compositional synthetic data** (NVARC's SDG style + input-side augmentation that keeps outputs valid). Cloud cost ≈ **$100–500 one-time** (training off-Kaggle is allowed; upload weights; 12h = inference only).
- This is the difference between ~35% (rehash ceiling) and ~40%+ (winning).

## 5. Tiered plan for OUR situation (single RTX 5090 + Kaggle L4x4)

| Tier | What | Hardware | Expected LB | Effort |
|---|---|---|---|---|
| **1 — Maximize the port** | Lever A (deep pool + learned verifier) + Lever B (dynamic queue, no `[[0]]`, repeated submits) + swap to best base that fits L4x4 | 5090 + Kaggle | **~32–36%** | weeks, doable now |
| **2 — Win** | Tier 1 + Lever C (cloud-train a stronger 7–8B base on improved synthetic data) | + cloud $100–500 | **~38–42%** | months + budget |
| **3 — Research bet** | Novel/ensemble (diverse 2nd model for complementary coverage); avoid pure TRM/TOPAS (overfit) | + cloud | high variance | research |

## 6. Honest odds
- **Progress prize (top-8, $15–75K):** plausible with **Tier 1 + repeated submits** if the field stays ~35–42%; realistically needs Tier 2 to be safe.
- **1st place ($75K) / 85% bonus ($150K):** Tier 2 minimum for 1st; **85% is not reachable offline** (top is ~42%).
- **Grand Prize ($275K, best WRITEUP — guaranteed payout this year):** a *genuinely realistic target independent of LB rank.* Our exhaustive, honest experimental map (what works/fails + why, with measurements) is strong material. **Recommend targeting this explicitly.**

## 7. Concrete next actions (ranked)
1. **Fix the submission notebook's GPU scheduling** → true dynamic shared queue + `[[0]]`-elimination fallback (Lever B). *Biggest cheap LB recovery.*
2. **Deepen the candidate pool** (aug_n=16, DFS prob 0.1, more views) and **prototype a learned verifier/selector** to capture the ~10pp scorer gap (Lever A).
3. **Submit the hardened baseline** (repeatedly) to measure the real LB number and the eval→LB gap for OUR pipeline.
4. **If pursuing 1st:** rent cloud, build NVARC-style harder synthetic data, fine-tune a stronger base (Lever C).
5. **In parallel:** write the technical paper (Grand Prize) — we already have `docs/FINAL_REPORT.md` as the spine.

# What Actually Moves an Offline ARC-AGI-2 Score — A Measured Study

**Subtitle:** We reproduced the 2025 winner on one consumer GPU, measured every lever practitioners reach for — with sample sizes, paired comparisons, and 79 first-party leaderboard submissions — and found that almost nothing moves the score except time, that local gains reliably *invert* on the hidden set, and that same-code reruns alone span nearly 7pp.

*Track: ARC-AGI-2. Linked submission: 33.06 (submission ID 57502566), the best of 79 scored first-party submissions through Oct 26, 2026. Full paper (PDF) in project links; all code public (MIT-0; vendored files keep their upstream licenses, see NOTICE).*
<!-- Score line is date-anchored (through Oct 26, 2026): no edit is needed after later draws. The linked ID stays 57502566 regardless of later scores. Optional post-Nov 2 refresh is repo-only (docs/lb_draws.csv, PAPER.pdf). -->

---

## What we did

We did not build a new architecture. We rebuilt the 2025 winning recipe (NVARC: per-task test-time-trained LoRA on a grid-tokenized Qwen3-4B, DFS decoding, augmentation voting) as a dependency-light `transformers`+`peft` pipeline on a single RTX 5090, and used it as an **instrument**: every lever the community reaches for, measured with explicit N, paired where possible, and — the part that changed our conclusions — **validated against the real leaderboard**: fifteen submissions across five configurations, then a long series of byte-identical reruns that measures the leaderboard's own noise.

Three results organize everything we saw.

## Result 1: The score is coverage-bound, not selection-bound

Measuring the **oracle** (is the correct grid *anywhere* in the candidate pool?) against achieved pass@2 splits the problem: on our deepest pools the oracle is 37.2% and the achieved score 32.1% (n=78 outputs). The ~5pp residual is *selection*; the other ~63% of tasks fail because the model **never generates the correct grid at all**.

Selection is nearly saturated: a sweep over scoring variants recovers ≤1 output; a learned verifier trained on 29 positives loses 1.3pp, and retrained on 1,037 positives from a 1,000-task cache gains exactly one output — its top feature turns out to be the augmentation-NLL rank the baseline scorer already uses. Coverage is the wall. And coverage refuses to move: doubling augmentations, halving the decode threshold, adding views — each measured ≈ +0pp oracle.

**The one thing that adds coverage is independent runs.** Merging the candidate pools of three independent local runs (the submission config, aug16, and a two-epoch TTT variant) lifts the oracle from 35.5% to 48.6% on 138 paired outputs: TTT stochasticity reaches transformations that wider decoding of one run never does, so run-to-run variance is itself a coverage source. But the same selector converts only about a third of it — pass@2 rises 31.9% → 36.2% while the oracle↔pass@2 gap widens from 3.6 to 12.4pp. Once pools are deep, **selection becomes the bottleneck**, and neither our fixed nor our learned selectors close it. We did not submit pooling: one pass already uses ~10.7 of the 12h, and a pool-widening change had already inverted on the leaderboard once (next).

## Result 2: Local gains invert on the leaderboard (twice, replicated)

This is the finding we most want other teams to have.

**Deep pool:** aug16 + threshold 0.1 + 3 views measured **+2.4pp paired** locally — and scored **19.17 vs 25.00** on the leaderboard: a 5.8pp *regression*, far outside our measured same-code rerun noise (identical resubmit: 25.00 → 24.17).

**TTT depth:** the strongest local effect of the entire study. Re-spending the same token budget as 8 augmentations × **2 epochs** instead of 16 × 1 measured **+3.6pp paired on 138 outputs** (+8/−3 discordant), with a clean dose-response across configurations behind it. Deployed runtime-neutrally, four independent leaderboard runs averaged **1pp *below*** the unmodified engine.

The largest paired local gains we ever measured transferred as −5.8pp and ≈−1pp. Both interventions widen or deepen per-task adaptation; both were tuned against public-eval pairs. Our conclusion, now replicated: **recipe changes tuned on the public evaluation set overfit it.** The hidden set is not just harder; it is *differently distributed enough* that public-eval hill-climbing points downhill. This mechanism plausibly caps the "knob-fitting" pack (33-37 on the current board).

## Result 3: What actually transfers is time

Across our configuration submissions, exactly two interventions survived contact with the leaderboard:

**Engine/budget efficiency, +4-5pp.** The same model checkpoint scores 24.17-25.00 through our stock-`transformers` port (which used 3.9h of the 12h budget) and 26.4-33.06 (71 draws, median 29.3) through the community's unsloth-based engine, which converts the full budget into per-task compute. Nothing about the model changed. This is the largest single effect we measured anywhere — and it is engineering, not modeling.

**Placeholder hygiene, ≥0 by construction.** Never emit `[[0]]`; any real guess (test-input identity, modal training output) dominates.

**Time is spent, not wasted.** An instrumented run over all 120 public-eval tasks shows the engine's limits almost never bind: 0 of 667 decode calls hit the 540s cap, 8 of 120 tasks hit the 1,200s per-task cap, and those 8 solve at the same rate as the rest (25.0% vs 26.5%). A shortest-job-first task queue — the only scheduling change in our submitted kernel — is neutral on the leaderboard. The +4-5pp came from using the whole budget; there is no budget left to reallocate.

**Rerun noise is the rest of the public board.** Our linked score, 33.06, is the *maximum* of a series of byte-identical reruns of one attributed public kernel (advertised 33.89; our only change is the queue order). Over 61 such draws (Aug 25 to Oct 26; `docs/lb_draws.csv` lists every submission): median 29.3, interquartile 28.7-30.6, range 26.4-33.06, six draws ≥31, one ≥32. A kernel's advertised score is the max order statistic of such a series, not its expectation: expect ~4pp below the advertised number, and read the ~33-37 pack on the board as the upper tail of rerun lotteries. We report the whole distribution because the best draw alone would misstate what the method does.

## Training-time interventions: four nulls and a hypothesis

Because inference was saturated, we attacked the weights — four escalating attempts, all measured against the same paired protocol:

1. **Harder procedural data** (compositional generator, 2× the base's loss): continued-FT → **exactly 0.0pp** (identical 22.8%).
2. **A bigger base** (vocab-cut Qwen3-8B, LoRA r=256, 526M tokens, 74 GPU-h): **−11.7pp**. Scale does not substitute for corpus depth. (A first attempt at lr 2e-4 diverged at step 200 and produced a model that decoded *nothing* — the post-mortem is in the paper; treat loss spikes above ln(16)≈2.77 as fatal, not recoverable.)
3. **Frontier-generated OOD data:** we streamed 95k GPT-4o-generated BARC-HEAVY tasks through the base and kept the 24k it measurably does not know (teacher-forced loss >0.10, ~3× its own synthetic corpus). Clean training, a +1.6pp flash at n=30 — and the paired extension to 133 outputs landed **exactly equal: 44/133 vs 44/133**, four discordant outputs each way.
4. **The winner's own corpus:** continued-FT of the released 4B on 2.5B tokens of its public 3.2M-task set (19,637 steps, clean convergence) → **−4.6pp paired on 129 outputs** (4 vs 10 discordant). The released checkpoint sits at a selected optimum; more optimization on its own data *subtracts*.

The third pattern — *shuffling* the solved set without *growing* it — suggests our unifying hypothesis: **per-task test-time training washes out base-model deltas.** TTT re-adapts any nearby base onto the same per-task solutions. It explains why the entire public pack rides one checkpoint, why our four training interventions changed nothing or hurt, and why the leaders' advantage (the top team's runs finish in 8h41m with three hours of budget *unused* — it is a better base, not better search) required training a model far enough away that TTT cannot erase the difference.

Program induction, for completeness, is a triple zero on the eval distribution: a symbolic DSL (0/167), BARC-Llama-8B execution-verified sampling (0/166 net-new; 23% on *training* tasks — the distribution shift is total), and Qwen2.5-Coder-7B with a repair round (0/120).

## What we think winning requires

Reconciling our measurements with the board: the public frontier is ~34 advertised / ~29 rerun-median; the knob-fitting ceiling is ~37; and everything above ~40 is unpublished model training at industrial scale on frontier-grade synthetic data — the one lever our study shows TTT cannot wash out and public-eval tuning cannot fake. For everyone else, the honest playbook is: use the efficient engine, never emit placeholders, treat local paired gains under ~5pp as noise until a submission confirms them, treat a single leaderboard draw the same way, and spend your remaining effort on the base model or on a different competition.

## Reproducibility

Everything is released: the single-GPU reproduction (with a fix for a silent tokenizer bug in `transformers` ≥5 that scores exactly 0 with no error), every submission configuration and the leaderboard draw series, every cache and analysis script behind every number here, the hardened submission engine (dynamic work-queue, no-placeholder policy, atomic-checkpoint trainer with pause/resume), and the full paper with per-claim N and confidence intervals. Repository: github.com/pyaroslav/arc-agi-2 (MIT-0). Attached notebook: our submission kernel. We believe the measured map of dead ends — and especially the two replicated leaderboard inversions — will save other teams months.

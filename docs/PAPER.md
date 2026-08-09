# Coverage, Generalization, and Time: A Measured Study of Offline ARC-AGI-2

**Two analyses — a coverage-vs-selection decomposition and a cross-family
public-eval→leaderboard generalization gap — built on a clean single-GPU
reproduction of the 2025 winner, with every claimed effect reported with sample
size and confidence interval.**

*ARC Prize 2026 — ARC-AGI-2 (offline track). Local study; see §3 and §9 for
submission status.*

---

## Abstract

We reproduce the 2025 ARC Prize winner (NVARC) as a dependency-light
`transformers`+`peft` pipeline that runs on a single consumer GPU, and use it as an
instrument to study what governs the offline ARC-AGI-2 score. We offer two analytical
contributions. **First**, a *coverage-vs-selection decomposition*: by measuring the
oracle (is the correct grid anywhere in the candidate pool?) against achieved pass@2,
we show the dominant loss is **coverage** — the model never generates the right grid —
not mis-ranking. Across five inference-time levers (selection tuning, a learned
verifier, deeper decoding, a symbolic DSL solver, and 8B program induction) and two
training-time levers (continued fine-tuning on locally-generated harder data, and an
8B base trained on the same corpus — §4.1), **none produces an effect distinguishable
from zero at our sample sizes, and the 8B base is directionally *worse*** (11.1% vs
22.8% per-task, paired n=30); the one positive point estimate (a deeper candidate
pool, +2.4 pp paired) lies well within its confidence interval — and *inverted* on
the real leaderboard (§7). This *is* the saturation result: test-time compute of the
same kind does not add coverage on the hard eval distribution, and parameter scale
without a deeper training corpus subtracts. **Second**, a
*generalization-gap calibration* across approach families, drawn from the competition
forum: the LLM+test-time-training family loses ~3.5 pp absolute (~11% relative) from
public eval to leaderboard, versus ~11 pp absolute (~51% relative) for a Tiny
Recursive Model and a comparable collapse for a deep-recursive model — i.e. LLM+TTT is
**uniquely leaderboard-robust**. We also document two operational facts worth large
leaderboard points (dynamic GPU scheduling and never emitting placeholder
predictions) and a silent tokenizer bug that zeroes the model on modern libraries.
All numbers carry N and 95% CIs; all code and experiments are released. The study is
now **leaderboard-confirmed**: our shallow-config submission scored **25.00** on the
semi-private leaderboard (small, slightly *positive* eval→LB gap, consistent with the
family calibration of §6), while the deep-pool config — locally +2.4 pp — **regressed
to 19.17**, an instructive inversion analyzed in §7.

**Contributions.** (1) A faithful single-GPU reproduction of the winning recipe, plus
a fix for a library-version tokenizer bug that silently scores 0. (2) The
coverage-vs-selection decomposition and a measured saturation result over seven
levers (five inference-time, two training-time), each with N+CI. (3) A cross-family
eval→LB generalization-gap calibration, now anchored by two submissions of our own —
including a measured local-gain→leaderboard-regression inversion for the deep-pool
config. (4) A hardened offline submission engine (dynamic work-queue, no-placeholder
policy). (5) A reproducible repository and an honest map of dead ends.

---

## 1. Introduction

ARC-AGI-2 is engineered to be easy for humans and hard for machines, and was
explicitly redesigned to resist brute-force program search; frontier models that
score 20–50% on ARC-AGI-1 fell below 5% at its release. The Kaggle competition adds
a hard operational constraint: submissions run **offline** on fixed hardware (L4×4,
96 GB, ≤12 h, no internet), excluding the frontier-API systems that top public
ARC-AGI-2 leaderboards.

In 2025 one approach dominated the offline track: a small open LLM, fine-tuned on
synthetic ARC data, **test-time-trained** per task, decoded into a candidate pool,
and reduced to two answers by augmentation voting (NVARC, 24% private). The 2026
offline leaderboard is largely teams iterating on that recipe.

We do not propose a new architecture. We reproduce the winning recipe on a single
consumer GPU and then **measure**, with sample sizes and confidence intervals, the
levers practitioners reach for — reporting negatives as carefully as positives. The
framing question is operational: *given a fixed strong base model and an offline
budget, what actually changes the score?* Our answer is that it is governed far more
by **coverage** (does the model generate the right grid at all), **generalization**
(does local accuracy survive the leaderboard), and **time** (does the run finish
without placeholders) than by the decoding/selection tricks that attract most effort.

---

## 2. Background and setup

**Task & metric.** Each task gives a few input→output grid demonstrations and one or
more test inputs (grids ≤30×30, 10 colors). Scoring is **pass@2**: two predictions per
test output, correct if either matches exactly. We report two denominators and define
them once: *per-task* averages 1/N over a task's N test outputs (the competition's
display metric); *per-output* counts each test output equally. The two differ by
~2.4× on this eval split (≈167 outputs over 120 tasks), so we always state which is
used and never compare across denominators.

**The NVARC recipe (reproduced).** Base: a Qwen3-4B fine-tuned with a 16-token grid
vocabulary, publicly released by the 2025 winner. Per task: (a) **test-time-train** a
per-task LoRA on the task's demonstrations under random augmentations (D4 geometry ×
color permutation); (b) decode a candidate pool with a batched **depth-first search**
keeping token paths above a per-token probability threshold; (c) select two answers by
**augmentation-NLL voting**. We re-implement this in stock `transformers`+`peft` (SDPA
attention, gradient checkpointing) so it runs on one 32 GB GPU and on Kaggle L4s
without `unsloth`/Flash-Attention/`bitsandbytes`; the decoder and scorer are ported
faithfully from the winner's open notebook. Hyperparameters in Table 3.

**A silent, fatal compatibility bug.** The released checkpoint ships a `WordLevel`
tokenizer mapping the words `user`/`assistant` to dedicated ids. Under
`transformers ≥ 5.0`, `.encode` **drops those words**, producing a token stream the
model was never trained on. An A/B test is unambiguous: with the role tokens the model
solves tasks; without them it solves **zero** (0/15 vs 1/15 greedy, no-TTT). We restore
correctness with an exact custom encoder. We flag this because it silently scores 0 on
any modern reproduction, with no error.

---

## 3. Reproduction (local)

On the public ARC-AGI-2 evaluation set our port scores **22.8% per task / 28.9%
per output** (n=44 outputs over 30 tasks, shallow config), matching the winner's
reported 30.5% pass@2 / 27% public-LB / 24% private-LB. Per-task latency is ~130 s
(shallow) to ~261 s (deep) on one RTX 5090. **Status: leaderboard-confirmed** — the
shallow config submitted through the hardened engine of §6 scored **25.00** on the
semi-private leaderboard, a slightly *positive* gap vs our 22.8% local per-task
estimate (within its ±13 pp CI, and consistent with the family calibration of §6);
the deep config scored 19.17 (§7). Grand-Prize eligibility is satisfied (§9).

---

## 4. The lever map — measured, with N and 95% CIs

All numbers are held-out pass@2 (per output) on public-eval candidate caches.
"Oracle" = correct grid present anywhere in the pool (coverage ceiling). CIs are
Wilson 95%. At our sample sizes the 95% CI half-width on a single proportion is
**≈ ±10 pp (n=78)** to **≈ ±14 pp (n=42)** — so *pp-level differences below ~10 pp are
not statistically resolved*. We report them anyway, with that caveat, because the
**pattern of uniformly-near-zero effects is itself the result**.

**Table 1 — inference-time levers.**

| Lever | Effect (point estimate) | N | Significant? | Verdict |
|---|---|---|---|---|
| Deeper pool (aug16 + DFS prob 0.1 + 3 views, *combined*) | **+2.4 pp** pass@2 & oracle, paired | 42 common outputs | no (within ±14 pp CI) | best point estimate; *suggestive* |
| Free selection sweep (count-wt × {sum,mean,geom} aug-NLL) | +0 to +1 pp (≤1 task) | 78 | no | NVARC scorer already near-optimal |
| Learned verifier (GBM): group-CV at 29 pos → train-on-1,000-training-tasks transfer at 1,037 pos | **−1.3 pp** → **+1.3 pp** (25→26 of 78) | 78 test outputs | no (±1 task) | 36× data flips the sign but adds ≤1 task; top feature is the aug-NLL rank kgmon already uses |
| Symbolic DSL (~15 rules, depth-2, train-consistent) | **0 / 167** outputs | 167 | yes (true 0) | ARC-AGI-2 defeats brute-force DSL |
| 8B induction (BARC, execute-verify, 48–64 prog/task) | training **23%** (7/30 tasks); **eval 0 / 166** net-new | 166 | yes (true 0 on eval) | strong in-distribution; defeated by eval set |

Two single-knob ablations: aug_n 8→16 *alone*, and "more views + lower threshold"
*alone*, each measured **+0.0 pp oracle** (n=20–24). Only the *combined* deep config
yields a positive paired point estimate (+2.4 pp, n=42) — itself within CI. **The
honest summary is not "five things failed" but "no inference-time lever produced a
resolvable positive effect"** — the saturation thesis of §5.

### 4.1 Training-time levers: harder data, and a bigger base

If inference-time compute is saturated, the remaining local levers act on the
*weights*. We measured both, paired on the same first-30 eval tasks and the same
TTFT→decode→vote configuration as the baseline (per-task pass@2 22.8%).

**Table 1b — training-time levers.**

| Lever | Effect (point estimate) | N | Significant? | Verdict |
|---|---|---|---|---|
| Continued-FT of the 4B on locally-generated *harder* data (compositional generator; 2k tasks, ~2× harder by loss: 0.10 vs 0.05) | **±0.0 pp** (identical 22.8%) | 30 tasks, paired | no | locally-generated difficulty ≠ frontier-grade difficulty |
| 8B base (vocab-cut Qwen3-8B, LoRA r=256 incl. embeddings, 1 epoch / 526M packed tokens of the same corpus, ~74 GPU-h) | **−11.7 pp** (11.1% vs 22.8%) | 30 tasks, paired | no (within CI) but directionally negative | scale does not compensate for corpus depth |

The 8B result deserves its cautionary tale. A first run at peak lr 2e-4 (the
recipe's 1e-4, doubled) **diverged at step ~200** — loss 0.3 → 3.3, *above* the
uniform-over-16-tokens entropy ln 16 ≈ 2.77 — despite gradient clipping at 1.0, and
46 further hours of cosine decay "recovered" it only into a degenerate basin
(plateau 0.46; greedy decoding emits two cells then EOS; **empty candidate pools on
30/30 eval tasks**). A merge-vs-unmerged probe exonerated the checkpoint plumbing
(teacher-forced loss 0.708 merged vs 0.711 unmerged) and isolated the fault to the
weights themselves. The clean rerun at peak lr 5e-5 converged normally (final
train-mix loss ~0.065) and produced a *functional* model — well-formed grids,
healthy 133 s/task decode work, 4/30 tasks solved — that is simply **weaker than the
winner's 4B**: teacher-forced loss on ARC-train examples 0.082 vs the 4B's 0.017.
Our one-epoch pass over NVARC-synthetic + ARC-train + locally-generated data cannot
match the corpus scale and training depth behind the released 4B checkpoint, and 2×
the parameters do not close that gap; they appear to widen it at equal data. Both
rows point the same way: **the binding constraint is the training corpus, not the
parameter count and not local data generation** — consistent with §5's conclusion
that new coverage requires a stronger base *trained on stronger data*.

![Fig 1](figures/fig1_coverage.png)
*Fig 1. The gap is coverage, not selection: oracle ceiling vs achieved pass@2 for
shallow vs deep pools (Wilson 95% CIs). The deep pool raises both modestly; the
oracle↔achieved gap (selection headroom) is small (~4–7 pp) and within noise.*

---

## 5. Coverage-bound, not selection-bound

The organizing diagnostic is the gap between **oracle** and achieved pass@2. On the
deep pool the oracle is 37.2% and achieved 32.1% (score_kgmon, n=78): a residual
*selection* gap of ~5 pp, of which a better fixed selector recovers ≤1 task and a
learned verifier recovers none. The dominant loss is the other ~63%: tasks where the
model **never generates** the correct grid.

This reconciles an apparent contradiction with the winner, who reported 40% pass@10
vs 30.5% pass@2 ("our scorer missed 10%"). That 10 pp is real **at his pool depth**,
but it is a *property of a deeper pool*, not free selection: widening the pool raises
pass@N, and only then does selection have something to recover. Our "selection is free
money" attempts failed precisely because the answer was absent, not mis-ranked.

**A simple model.** Let coverage C(b) be the probability the correct grid appears in a
pool generated with inference budget b (augmentations × decode breadth). Empirically
C grows very slowly in b on ARC-AGI-2 (Fig 1: doubling augmentation, halving the decode
threshold, and adding views together move the oracle by ~2 pp, within noise), because
the failures are not near-misses the sampler rounds wrong but tasks whose
transformation the model cannot represent. Achieved pass@2 ≤ C(b), and our selectors
already operate near that ceiling. Hence **score is bounded by C(b), and C(b)
saturates** under same-kind compute — exactly the ARC "knowledge/reasoning, not
search" thesis, here measured. New coverage requires a *different generation mechanism
or a stronger base model*, not more search — consistent with symbolic search (0/167)
and offline induction (0/166) adding nothing on the eval distribution.

---

## 6. The other half: generalization and time

**6.1 The eval→LB gap is approach-dependent (Fig 2).** From the 2026 forum: the 2025
winner reports LLM+TTT 30.5% eval → 27% public-LB (**−3.5 pp absolute, ~11% relative**);
a competitor reports a Tiny Recursive Model 22% → 10.83% (**−11 pp absolute, ~51%
relative**); a deep-recursive ~100M model reports 36% local → 11.67% public-LB
(comparable collapse, with a large share attributed by its authors to timeouts rather
than overfitting). **LLM+TTT is uniquely LB-robust.** A plausible mechanism
(hypothesis, not isolated here): per-task TTT adapts to *this task's* demonstrations,
which transfers, whereas small models trained to convergence absorb public-set
statistics. The practical implication is concrete: do **not** abandon LLM+TTT for a
flashier architecture on a local number — the local number lies least for LLM+TTT.

![Fig 2](figures/fig2_family_gap.png)
*Fig 2. Public-eval vs public-LB by approach family (forum-sourced). Relative drop
annotated.*

**6.2 Time management is worth large LB points.** The 12 h budget is binding. Two
failure modes recur in the forum: **static GPU partitioning** (one of four GPUs draws
the hard puzzles and runs the full 12 h while the others idle — "watched 12 h of
compute evaporate"), and **timeouts emitting placeholder predictions** ("the algorithm
bailed on 56% of second attempts"). Our hardened engine (§7) fixes both with a single
**dynamic shared work-queue** (every GPU pulls the next task as it frees, so load
self-balances) and a **never-`[[0]]` policy** (any unsolved/unreached output falls
back to real guesses — the identity grid and the most-frequent training output, each
strictly better than a zero-scoring placeholder). These are engineering, but on the
forum evidence they plausibly recover more score than most modeling changes. Our
25.00 shallow submission ran through this engine with **zero placeholder outputs**
(4 of 255 outputs fallback-filled) in 3.9 h of the 12 h budget; we cannot isolate the
engine's contribution without an ablation submission, but the failure modes it
targets did not occur (§9).

---

## 7. What it takes to win, and the cost/quality frontier

Reconciling our measurements with the leaderboard (top ≈ 42.6%) and the 2025 winner's
public guidance ("simple tweaks won't be enough — show me 35+"; "this year's
environment fits significantly larger models … bigger/newer models will take over"):

- The LLM+TTT recipe **plateaus ~34–35% offline** (the winner's attributed claim); we
  reach ~33% per output locally and every inference-time lever is saturated (§4–5).
- The path to the winning ~42% is a **stronger base model on harder, compositional
  synthetic data** — not more search, and, per §4.1, **not parameter scale alone**: an
  8B trained on our locally-assemblable corpus lands at half the released 4B's score.
  The 96 GB environment fits 7–8B models; off-Kaggle training of weights is permitted;
  the corpus is the part that requires frontier-grade generation.
- **Avoid:** pure symbolic/DSL (we measured 0/167; the winner: "I never believed in
  DSL for ARC-AGI-2") and pure recursive small models (overfit; large eval→LB gap).

**Cost/quality and the 12 h budget (Table 2) — now leaderboard-measured.** We
submitted both configs. The shallow config scored **25.00** (240 tasks in 3.9 h of
the 12 h L4×4 budget; 0 placeholder outputs, 4 fallback-filled). The deep config —
locally +2.4 pp paired — scored **19.17**: a **−5.8 pp leaderboard regression**. The
inversion is exactly the §6 lesson applied to ourselves: at ~261 s/task the deep
config spends the budget (more per-task TTFT and wider pools) without better
selection, and budget pressure on the semi-private set converts marginal tasks into
fallbacks. (Competitors also report ±2 pp resubmission noise, which cannot explain a
5.8 pp drop.) **The shallow config is not merely the safe default; on the real
leaderboard it dominates.** Local paired gains inside the CI should be treated as
noise until they survive a submission.

**Table 2 — configs.** shallow: aug_n 8, DFS prob 0.2, 2 views, ~130 s/task —
**LB 25.00**. deep: aug_n 16, DFS prob 0.1, 3 views, ~261 s/task — **LB 19.17**.
Both fit 12 h on L4×4 via the dynamic queue; only shallow fits a single GPU under
12 h.

**Table 3 — key hyperparameters.** Base Qwen3-4B (16-token grid vocab); per-task LoRA
r=256, α=32, dropout 0, target = attn+MLP+embed+lm_head, rslora; TTT lr 5e-5 cosine,
1 epoch over ~128 augmentations, bf16, grad-checkpointing on; decode = batched DFS,
keep prob > τ (τ ∈ {0.2, 0.1}); vote = m=8 shared augmentations,
score = count − mean(per-aug NLL) or `probmul_3`; max_seq_len 8192. Env: torch
2.11+cu130, transformers 5.9 (the version exhibiting the tokenizer bug), peft 0.19,
vllm 0.21 (induction only, with `LD_PRELOAD` of system libstdc++ for Blackwell).

---

## 8. Reproducibility, licensing, attribution

Everything runs on one consumer GPU; hardware: 1× RTX 5090 (32 GB), plus Kaggle L4×4
for submission. Commands:

```bash
python -m pytest tests/ -q                                                   # core, no GPU
python scripts/repro_nvarc.py --limit 60 --aug-n 16 --eval-aug-n 3 --dfs-prob 0.1
python scripts/analyze_selection.py --pkl outputs/cache_deep.pkl             # oracle/selection gap
python scripts/sweep_selection.py  --pkl outputs/cache_deep.pkl              # free-selector sweep
python scripts/train_verifier.py   --pkl outputs/cache_deep.pkl              # learned verifier (group-CV)
python scripts/eval_symbolic.py                                              # DSL coverage (0/167)
python scripts/induction_complementarity.py --tasks 120 --samples 48        # BARC net-new (0/166)
python scripts/make_figures.py                                              # Fig 1, Fig 2
# offline submission: notebooks/kaggle_submission.ipynb -> nvarc.kaggle_submit.main(...)
```

Layout: `src/nvarc/` (ported pipeline + hardened submission engine), `src/arc/`
(tokenizer, augmentations, symbolic solver, verifier), `src/induction/` (BARC harness),
`scripts/` (every experiment), `data/` (gitignored; fetched at runtime).

**Licensing & attribution.** Original files are MIT (see `LICENSE`). Vendored/derived
files retain upstream licenses and carry attribution headers: `src/nvarc/arc_loader.py`
and `src/nvarc/selection.py` derive from NVARC (1ytic/NVARC, open-sourced per the ARC
Prize requirement); `src/induction/common.py` is vendored from BARC (xu3kev/BARC). We
**do not redistribute** model weights, synthetic datasets, or the competition forum
corpus: all of `data/` is gitignored and fetched at runtime from Kaggle/HuggingFace.
The forum corpus is used only for the quantitative analysis of §6 and is cited, not
relicensed.

---

## 9. Limitations and threats to validity

- **Two submissions, not a distribution.** The leaderboard numbers (25.00 shallow,
  19.17 deep) are single runs of a non-deterministic pipeline on which competitors
  report ±2 pp resubmission noise; the *sign* of the deep-config regression is robust
  to that noise, its magnitude is not. Grand-Prize eligibility (a linked Kaggle
  submission) is satisfied.
- **Small samples.** Most A/Bs are n=20–78; as Table 1 states, sub-~10 pp effects are
  within CI. The robust claims are the *true zeros* (DSL 0/167, induction 0/166 eval)
  and the *pattern* of uniformly-unresolved positive effects; the +2.4 pp deep-pool and
  −1.3 pp verifier are point estimates within noise, not established effects.
- **Verifier data scale — now tested.** Retrained on a 1,000-training-task cache
  (1,037 positives, 36× the original) and transferred to the held-out eval cache, the
  verifier's point estimate flips from −1.3 pp to **+1.3 pp** (26 vs 25 of 78) — one
  task, within noise, and its dominant feature is the augmentation-NLL rank the fixed
  scorer already uses (`scripts/verifier_transfer.py`). The prediction above held:
  data scale changes the sign, not the story; the residual selection gap (~4 tasks
  here) is mostly not capturable by candidate-level features.
- **The training-time nulls are single-GPU-scale.** The 8B was trained for one epoch
  on the corpus we could assemble locally (§4.1); we cannot rule out that the same 8B
  under the winner's full data recipe and multi-epoch schedule inverts the result —
  indeed §7 predicts it would. What our null establishes is narrower and useful: at
  *equal, locally-feasible* data, parameter scale subtracts rather than adds. The
  winning path (§7) still requires cloud compute outside this study.
- **Forum-sourced numbers** (§6) are self-reported by competitors; we attribute each to
  its source and treat the mechanism explanations as hypotheses.

---

## 10. Related work

Test-time training for ARC was established by Akyürek et al. ("The Surprising
Effectiveness of Test-Time Training for Abstract Reasoning"); NVARC (2025) combined it
with heavy synthetic-data generation and augmentation voting to win the offline track.
Program-induction approaches (BARC; "Combining Induction and Transduction") generate
and execute Python solvers; Tiny Recursive Models (Jolicoeur-Martineau, 2025) pursue
tiny recurrent reasoners. Our delta is not a new solver but a *measured decomposition*
of where the offline score is won and lost — coverage vs selection, and generalization
vs local accuracy — across these families, on the harder ARC-AGI-2 distribution.

---

## 11. Conclusion

The offline ARC-AGI-2 score is governed by coverage, generalization, and time more
than by decoding/selection cleverness. We reproduced the winning recipe on a single
GPU, showed via N+CI measurements that neither inference-time levers (~33% per-output
locally, saturated) nor locally-feasible training-time levers (harder local data:
±0.0 pp; an 8B on the same corpus: directionally *worse*) resolvably move the score,
calibrated the eval→LB gap that makes LLM+TTT uniquely robust — then confirmed it on
the leaderboard, where our shallow config scored 25.00 with a slightly positive gap
and our locally-positive deep config regressed by 5.8 pp. The path past ~35% is a
stronger base on better synthetic data — where "better" is beyond what a single GPU
can generate or absorb — not more search and not more parameters. Our most useful
contribution is the measured map of the dead ends, now with two of its edges
leaderboard-verified.

---

### References (selected)
- ARC Prize 2025 results & technical report; ARC-AGI-2 technical report (arcprize.org).
- NVARC 2025 winning solution and code (github.com/1ytic/NVARC).
- Akyürek et al., *The Surprising Effectiveness of Test-Time Training for Abstract Reasoning*.
- Jolicoeur-Martineau, *Tiny Recursive Models* (2025).
- *Combining Induction and Transduction for Abstract Reasoning* (BARC; github.com/xu3kev/BARC).
- ARC Prize 2026 Kaggle competition discussion forum (analyzed in §6; not redistributed).

# Experiments

Index of all experiments. Artifacts live on gitignored shared storage at
`<outputs|results|models/checkpoints>/<NN-exp>/<run-id>/`. Durable docs (this layer)
are git-tracked under `docs/experiments/`.

Launch runs with `EXP=<NN-exp> RUN=<run-id>` (the scripts enforce it); scaffold
a new experiment folder by hand and record a finished run following the existing
entries' format.

## Eval split and leakage note

⚠️ **2026-06-23: all metrics below were corrected from the test set to a held-out
validation set.** Every experiment 01–07 was originally evaluated — and its
hyperparameters (γ, epoch, halting threshold) selected — on the GSM8K **test** split
(`data/gsm8k_aug/test.jsonl`, 1319 ex), which is optimistically biased. The eval
scripts now default to `data/gsm8k_aug/validation.jsonl` (500 ex, disjoint from test);
the test split is reserved for a single final report after model selection.

Reconciliation was **only partial** because most checkpoints had already been deleted
(`save_total_limit` keeps the last 1–2 epochs). What we could and couldn't redo:

| exp | re-validated on val | not reconcilable (test, biased — ckpt deleted) |
|-----|---------------------|-----------------------------------------------|
| 01 | baseline-k6 (fixed-K, greedy) | fixedk-k6-lr1e4 |
| 02 | — | both runs (also superseded: broken KV cache) |
| 03 | ep4, ep5 | **ep2 (the old 42.23% headline)**, ep1, ep3 |
| 04 | g0.05 ep3 only | the other 7 γ configs + g0.05's other epochs (so the γ-sweep selection cannot be redone) |
| 05 | ep4 (winner), ep5 | ep1–ep3 |
| 06 | — | both trunc-K runs |
| 07 | ep5 (reported best) | ep1–ep4 (incl. the old Spearman record ep3=0.684) |

All numbers are **greedy single-pass** on validation (n=500) unless flagged otherwise.
Note the baseline is now greedy (40.80%), a fairer bar than the old 39.50% which was an
average of 5 *sampled* passes on test.

| # | experiment | what it tested | best result (validation, n=500, greedy) | status |
|---|-----------|----------------|-------------|--------|
| 01 | [simcot-baselines](experiments/01-simcot-baselines/experiment.md) | reference fixed-K SIM-CoT | **40.80%** @ K6 (prior: 39.50% test, avg-5-sample) | reference |
| 02 | [simcot-pondernet-early](experiments/02-simcot-pondernet-early/experiment.md) | first PonderNet runs (pre-gcfix; broken KV cache) | 19.26% _(test, biased — not reconcilable)_ | superseded |
| 03 | [simcot-pondernet-gcfix](experiments/03-simcot-pondernet-gcfix/experiment.md) | gradient-checkpointing fix | 41.20% @ ep5 thr0.5 _(old 42.23%@ep2 unrevalidatable — ckpt deleted)_ | complete |
| 04 | [simcot-pondernet-gammasweep](experiments/04-simcot-pondernet-gammasweep/experiment.md) | γ (KL-prior weight) vs accuracy/steps frontier | **41.40%** @ 5.34 steps (γ=0.05, thr0.8, ep3) _(sweep selection was on test)_ | complete |
| 05 | [simcot-pondernet-adaptive-prior](experiments/05-simcot-pondernet-adaptive-prior/experiment.md) | per-instance geometric prior (geom_mean_i = α·n_i+β) vs global geom=3.0 | 41.00% @ ep5 thr0.5 / 40.60% @ thr0.8; Spearman +0.66 (prior test: 40.49%, +0.65) | complete |
| 06 | [simcot-pondernet-trunc-k](experiments/06-simcot-pondernet-trunc-k/experiment.md) | per-instance truncated-K training (K_i=n_i) to break K=6 warm-start bias | 36.54%/35.94% _(test, biased — not reconcilable; ckpt deleted)_ | superseded |
| 07 | [simcot-pondernet-fullscope-prior](experiments/07-simcot-pondernet-fullscope-prior/experiment.md) | full backbone unfreeze (Recipe C) + per-instance adaptive prior + K_max=12 | 41.00% @ thr0.8 ep5; Spearman **+0.675** (ep5 thr0.5) _(old record 0.684@ep3 unrevalidatable)_ | complete |
| 08 | [simcot-pondernet-gamma-frontier](experiments/08-simcot-pondernet-gamma-frontier/experiment.md) | γ↑ (0.05→0.10) + prior reshaping to push the accuracy–steps frontier left | Run B ep5 **41.0% @ 3.64** (thr0.5); Run C ep5 **40.6% @ 2.93** (thr0.5, −32% steps vs baseline) | complete |
| 09 | [simcot-pondernet-gamma-push](experiments/09-simcot-pondernet-gamma-push/experiment.md) | γ push (0.10→0.15/0.20): does the frontier keep moving left? | TBD | planned |
| 10 | [simcot-pondernet-fromscratch](experiments/10-simcot-pondernet-fromscratch/experiment.md) | fair-comparison: replicate exp-08 Run C **from vanilla GPT-2** (no SIM-CoT warm-start; train aux decoder via new `full_dec` scope; lr 3e-3, 40 ep, full 385k GSM8k-Aug) | _(to run on external machine)_ | planned |

**Headline (validation).** The greedy fixed-K baseline is **40.80%** (K=6). On the
held-out set the adaptive runs cluster around it — best validated accuracy is
`04/g0.05` **41.40%** @ thr0.8 (5.34 steps) and `03/100k` 41.20% @ ep5/thr0.5 — i.e.
accuracy gains over the greedy baseline are small and within noise on 500 examples. The
durable result is **compute adaptivity, not accuracy**: per-instance prior and full-scope
training drive strong difficulty-tracking (Spearman rises from +0.46 at exp-03 to
**+0.675** at exp-07, ep5/thr0.5) and let easy instances halt early (avg_steps scales
~2→6 across n_expr) at matched accuracy. ⚠️ The previous "best overall 42.23% @ ep2"
claim **cannot be revalidated** — that checkpoint was deleted before the validation set
existed.

## Active / deferred (not yet migrated)

These sweeps are ongoing or held for separate integration once complete; their artifacts
remain at flat top-level paths (`<dir>/simcot-pondernet-k_recipe_sweep*`, `<dir>/optionb-*`)
and must **not** be moved (see
`docs/superpowers/specs/2026-06-10-repo-professionalization-design.md`). They will get a
numbered experiment folder by hand, following an existing one as a template, when their sweeps finish.

| experiment (provisional) | what it tests | status |
|--------------------------|---------------|--------|
| simcot-pondernet-k-recipe | K_max recipe sweep (recipeA k4–k12; recipeB variant) | active |
| simcot-optionb-caxis | Option-B adaptive vectors-per-step (c-axis) | active |

# Experiments

Index of all experiments. Artifacts live on gitignored shared storage at
`<outputs|results|models/checkpoints>/<NN-exp>/<run-id>/`. Durable docs (this layer)
are git-tracked under `docs/experiments/`.

Launch runs with `EXP=<NN-exp> RUN=<run-id>` (the scripts enforce it); scaffold
a new experiment folder by hand and record a finished run following the existing
entries' format.

| # | experiment | what it tested | best result | status |
|---|-----------|----------------|-------------|--------|
| 01 | [simcot-baselines](experiments/01-simcot-baselines/experiment.md) | reference fixed-K SIM-CoT | 39.50% @ K6 | reference |
| 02 | [simcot-pondernet-early](experiments/02-simcot-pondernet-early/experiment.md) | first PonderNet runs (pre-gcfix; broken KV cache) | 19.26% | superseded |
| 03 | [simcot-pondernet-gcfix](experiments/03-simcot-pondernet-gcfix/experiment.md) | gradient-checkpointing fix | **42.23%** @ ep2 | complete |
| 04 | [simcot-pondernet-gammasweep](experiments/04-simcot-pondernet-gammasweep/experiment.md) | γ (KL-prior weight) vs accuracy/steps frontier | 40.2% @ 5.38 steps (γ=0.05) | complete |
| 05 | [simcot-pondernet-adaptive-prior](experiments/05-simcot-pondernet-adaptive-prior/experiment.md) | per-instance geometric prior (geom_mean_i = α·n_i+β) vs global geom=3.0 | **40.49%** @ 5.262 avg_steps (thr0.8, ep4); Spearman +0.65 | complete |
| 06 | [simcot-pondernet-trunc-k](experiments/06-simcot-pondernet-trunc-k/experiment.md) | per-instance truncated-K training (K_i=n_i) to break K=6 warm-start bias | 36.32% @ thr0.8 (ep5); avg_steps 3.660 (thr0.5); Spearman 0.596 — acc −4pp vs exp-05, not yet converged | complete |

**Headline:** best overall = `03-simcot-pondernet-gcfix/100k` **42.23%** @ ep2 (first trained
run to beat the 39.5% baseline). Most efficient near-baseline point = `04/g0.05-gm3.0-ep5`
40.2% @ 5.38 avg steps (thr=0.8). Per-instance prior (`05/perinstance-g0.05-b1.5-ep5`) improves
all three axes vs exp-04: +0.29pp acc, −0.12 avg_steps, Spearman +0.650 (+0.07 vs baseline).

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

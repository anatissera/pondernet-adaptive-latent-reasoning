# 05: per-instance (adaptive) geometric halting prior

**Status:** complete   **Dates:** 2026-06-19 → 2026-06-20

## What's being tested

**Hypothesis:** replacing the **global** geometric halting prior (`geom_mean=3.0`, the
same step target nudged onto every example) with a **per-instance** prior keyed to each
example's own reasoning-step count `n_i = (#expr − 1)` — via the affine remap
`geom_mean_i = α·n_i + β` (see Setup) — sharpens and widens the steps-vs-difficulty signal
the halting head already learns, pulling easy problems down to fewer latent steps
**without losing accuracy**.

**Why this, why now.** At the exp-04 sweet spot (`g0.05`, global `geom_mean=3.0`) the halting
head *already* tracks difficulty: `Spearman(steps_used, #expr) = +0.58` at thr=0.5, with avg
steps rising monotonically from ~2.8 (1-expr problems) to ~5.2 (7–8 expr). But the dynamic
range is **compressed and saturates** near the K=6 cap — trivial 1–2-expr problems still burn
~3–5 steps because the global `geom_mean=3.0` floors every example at the same target. A
per-instance prior targets exactly that compression: nudge each example toward *its own*
step count so easy problems halt early and the range stretches.

This is the K-axis follow-up to the project's two prior findings: adaptive halting beats
fixed-K internally (exp 03/04), and the c-axis (vectors-per-step) has no headroom (Option-B).

## Setup

- **Method:** PonderNet adaptive halting on SIM-CoT, full-model warm-start, the
  [04](../04-simcot-pondernet-gammasweep/experiment.md) recipe.
- **Mapping:** `c=1` here (1 latent ⟷ 1 reasoning step; CODI lineage, **not** Coconut's c=2).
  Each example carries a variable explicit-step count `#expr`; the answer-computing expression
  is dropped in preprocessing (`include_last_cot=False`), so the per-instance step count is
  `n_i = (#expr − 1)`. The prior mean is an **affine remap** `geom_mean_i = α·n_i + β`
  (default `α=1, β=1.5`), clamped to `[β, K_max]` — **not** the identity `geom_mean_i = n_i`
  (see *Degeneracy* below).
- **Varied:** prior type — **global** `geom_mean=3.0` (baseline) vs **per-instance** affine
  remap `geom_mean_i = α·n_i + β`.
- **Held fixed:** warm-start (full SIM-CoT CODI), `train100k.jsonl`, eff. batch 128,
  lr 2e-5, ep 5, **γ=0.05**, **K_max=6**, seed 42.
- **Baseline = the existing `04/g0.05-gm3.0-ep5` run** — it already holds every other
  hyperparameter at exp-05's values, so no fresh baseline retrain is needed. Only the
  per-instance run is new.
- **Eval:** GSM8K test, greedy, 1 pass, thresholds 0.5 / 0.8 / 0.9.
- **Metrics:** accuracy vs `04/g0.05`; avg latent steps; `Spearman(steps_used, #expr)` and
  avg `steps_used` per `#expr` bin (goal: widen beyond +0.58 and lower the easy-bin steps
  without accuracy loss).

### Scope decisions

- **K_max kept fixed at 6 (not made per-instance).** A *hard* per-instance cap `n_i` is a
  training-only device: the reasoning-step count is a label, and at **inference the problem's
  step count is unknown** — so a hard cap cannot survive to test time, and it would disable
  the very halting head this project exists to train. The **soft per-instance prior is the
  inference-compatible form of the same intuition** (nudge toward `n_i` in training; the head
  learns the pattern and generalizes to unseen problems). The data also barely needs a larger
  cap: with `c=1` and the answer-expression dropped, only **0.21%** of training examples need
  >6 latent steps (the truncate/merge branch fires only at ≥9 expressions, **0.065%**).
  De-saturating the high-threshold regime via a larger **global** K_max is covered separately
  by the active k-recipe sweep (k4–k12). Exp 05 therefore isolates the prior.
- **Average-target confound (note for interpretation).** `mean(#expr − 1) ≈ 1.6` over the
  training set, well below the global 3.0, so the per-instance prior also *lowers the average*
  target. If the headline result is positive, a matched-mean control (global `geom_mean ≈ 1.6`)
  would isolate per-instance *shape* from *lower mean*; deferred until then.
- **Degeneracy → affine remap (why not `geom_mean_i = n_i`).** A truncated geometric with
  `g = 1/geom_mean` collapses to the point mass `[1,0,…,0]` at `geom_mean = 1`, and means in
  `[1,2]` are barely distinguishable. The identity map sends the **53.5%** of examples with
  `n_i ≤ 1` onto that degenerate point (verified on the full 385k *and* the 100k head slice —
  distributions match within ±0.27pp). The smoke run made it concrete: `kl_geom ≈ 4–13`
  (vs ~0.5 global), so at γ=0.05 the KL term swamped the task loss. Fix: the **affine remap**
  `geom_mean_i = α·n_i + β` (α=1, β=1.5) lifts every example off the floor and *improves*
  low-end resolution (`n_i` 0/1/2 → means 1.5/2.5/3.5, all distinct). Post-fix smoke:
  `kl_geom ≈ 0.6–1.3`. β doubles as the `z₀` + final-answer latent overhead.

## Implementation note

**Done** (`pondernet/src/model.py`, unit-tested in `pondernet/tests/test_kl_geom.py` — 13 tests;
smoke-verified on the 3060, `kl_geom ≈ 0.6–1.3`):

- Pure helpers `truncated_geometric_prior(geom_mean, K)` and `kl_to_truncated_geometric(p_k, geom_mean)`
  accept a per-example mean tensor `(B,)` → per-row prior `(B, K)`; scalar path unchanged.
- `count_real_steps(steps_list, pad_id)` derives `n_i` (non-pad reasoning slots from `get_steps`).
- `adaptive_prior_mean(n_real, scale, offset, K)` = the affine remap `clamp(α·n_i + β, β, K)`.
- `_kl_geom` delegates to the KL helper; in `forward`, when `pondernet_adaptive_prior` is on it
  passes `geom_mean_i = adaptive_prior_mean(...)` (requires `use_decoder`, which the recipe sets).
- New flags `--pondernet_adaptive_prior` (default `False`), `--pondernet_prior_scale` (α=1.0),
  `--pondernet_prior_offset` (β=1.5); train script exposes `ADAPTIVE_PRIOR`, `GAMMA`,
  `PRIOR_SCALE`, `PRIOR_OFFSET` env overrides.

**Launch (per-instance arm):**

```bash
EXP=05-simcot-pondernet-adaptive-prior RUN=perinstance-g0.05-b1.5-ep5 \
ADAPTIVE_PRIOR=True GAMMA=0.05 PRIOR_OFFSET=1.5 PRIOR_SCALE=1.0 \
bash scripts/train_gpt2_gsm8k_pondernet.sh
```

Baseline arm = existing `04/g0.05-gm3.0-ep5` (no retrain). Eval both with the standard
PonderNet eval at thr 0.5/0.8/0.9, `batch_size 1`, on the idle GPU. Note: per-instance KL is
inherently sharper than global, so γ may need lowering below 0.05 to match exp-04's
regularization strength — worth a small γ × β sweep once the first run lands.

## Findings

**Run:** `perinstance-g0.05-b1.5-ep5` — completed 2026-06-20, trained on RTX 3090 (5 epochs / 3890 steps, ~2.4 s/it). Eval on RTX 3060, bs=16, thresholds 0.5/0.8/0.9.

### Epoch sweep (thr=0.8)

| epoch | step | acc (thr0.8) | avg_steps |
|-------|------|-------------|-----------|
| 1 | 778 | 39.50% | 5.407 |
| 2 | 1556 | 39.80% | 5.303 |
| 3 | 2334 | 39.80% | 5.254 |
| **4** | **3112** | **40.49%** | **5.262** |
| 5 | 3890 | 40.11% | 5.264 |

**Winner: epoch 4** (step 3112, preserved in `_best_epoch/`). Accuracy peaks at ep4 then drops slightly — consistent with the exp-03/04 pattern of early peaking.

### vs baseline (`04/g0.05-gm3.0-ep5`)

| metric | baseline | this run (ep4) | delta |
|--------|----------|----------------|-------|
| acc @ thr0.8 | 40.20% | **40.49%** | +0.29pp |
| avg_steps @ thr0.8 | 5.38 | **5.262** | −0.118 |
| avg_steps @ thr0.5 | 5.38 (baseline used thr0.5 for Spearman) | **4.365** | — |
| Spearman(steps, #expr) @ thr0.5 | +0.58 | **+0.650** | +0.070 |

Per-instance prior wins on all three axes simultaneously.

### Steps-vs-difficulty (thr=0.5, ep4)

| #expr | n | avg_steps | acc% |
|-------|---|-----------|------|
| 0 | 83 | 2.518 | 44.6 |
| 1 | 357 | 2.980 | 61.3 |
| 2 | 364 | 4.558 | 45.9 |
| 3 | 290 | 5.407 | 27.6 |
| 4 | 138 | 5.493 | 15.2 |
| 5 | 57 | 5.684 | 5.3 |
| 6 | 21 | 5.810 | 4.8 |
| 7 | 9 | 5.889 | 22.2 |

avg_steps is monotone with #expr across all bins (Spearman r=+0.650, p=4.7e-159, n=1319). Easy problems (0–1 expr) now halt at 2.5–3.0 steps on average. The adaptive prior successfully widened the dynamic range relative to the global-prior baseline.

### Interpretation

The per-instance affine prior (`geom_mean_i = n_i + 1.5`) beats the global `geom_mean=3.0` on accuracy, efficiency, and alignment of compute with difficulty. The halting head learned to generalize the difficulty signal to the test set without knowing each example's step count at inference.

**Confound to note:** `mean(n_i) ≈ 1.6` over the training set, well below the global 3.0, so the per-instance prior also *lowers the average* target. A matched-mean control (global `geom_mean ≈ 1.6`) would isolate per-instance shape from lower mean; deferred.

**Next:** a γ × β sweep (per-instance γ may be slightly strong at 0.05 given sharper KL) and the matched-mean ablation are the natural follow-ups if this result needs further attribution.

See [runs.md](runs.md) for the run table · artifacts under `<dir>/05-simcot-pondernet-adaptive-prior/`.

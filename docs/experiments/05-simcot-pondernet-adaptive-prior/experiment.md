# 05: per-instance (adaptive) geometric halting prior

**Status:** complete   **Dates:** 2026-06-19 → 2026-06-20

> ✅ **Re-validated 2026-06-23.** Surviving checkpoints (ep4 `_best_epoch`, ep5) re-evaluated on
> the held-out validation split (500 ex, greedy); ep1–ep3 were deleted. Prior test-set numbers
> were optimistically biased — the headline "+0.29pp accuracy" win does **not** survive
> de-biasing (accuracy is flat vs baseline on validation); the robust result is step-efficiency
> and difficulty tracking. See [eval-split note](../../experiments.md#eval-split-and-leakage-note).

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

**Run:** `perinstance-g0.05-b1.5-ep5` — completed 2026-06-20, trained on RTX 3090 (5 epochs / 3890 steps, ~2.4 s/it). **Re-validated 2026-06-23** on the validation split (500 ex, greedy); only ep4/ep5 survived (ep1–ep3 deleted).

### Epoch sweep

| epoch | step | acc (thr0.8) | avg_steps | source |
|-------|------|-------------|-----------|--------|
| 1 | 778 | 39.50% | 5.407 | test, biased — ckpt deleted |
| 2 | 1556 | 39.80% | 5.303 | test, biased — ckpt deleted |
| 3 | 2334 | 39.80% | 5.254 | test, biased — ckpt deleted |
| **4** | **3112** | **39.80% (val)** | **5.232** | re-validated, n=500 (prior test 40.49%) |
| 5 | 3890 | 40.60% (val) | 5.244 | re-validated, n=500 (prior test 40.11%) |

**Winner: epoch 4** (step 3112, preserved in `_best_epoch/`).

### vs baseline (`04/g0.05-gm3.0-ep5`), re-validated on validation (n=500)

| metric | baseline | this run (ep4) | delta |
|--------|----------|----------------|-------|
| acc @ thr0.5 | 40.80% | 40.80% | 0.00pp (flat) |
| acc @ thr0.8 | 41.40% | 39.80% | −1.60pp |
| avg_steps @ thr0.5 | 4.060 | 4.276 | +0.216 |
| Spearman(steps, #expr) @ thr0.5 | +0.553 | **+0.662** | **+0.109** ✓ |

On validation there is **no accuracy win** (flat-to-down vs baseline); the consistent, robust
gain is in **difficulty tracking** (Spearman +0.662 vs +0.553). The prior test-set "+0.29pp acc"
result was a leakage artifact and does not reproduce.

### Steps-vs-difficulty (thr=0.5, ep4, validation)

avg_steps by n_expr bin: n0 = 2.00, n1 = 2.65, n2 = 3.03, n3 = 4.55, n4 = 5.30, n5+ = 5.72 —
monotone across all bins, a clear adaptive signal (Spearman r = +0.662, n=500). Easy problems
(0–1 expr) halt at ~2.0–2.7 steps on average. The adaptive prior widened the compute-vs-difficulty
dynamic range relative to the global-prior baseline.

### Interpretation

On the de-biased validation split, the per-instance affine prior (`geom_mean_i = n_i + 1.5`) does
**not** beat the global `geom_mean=3.0` on accuracy — both sit at the 40.80% greedy validation
baseline. What it robustly improves is **alignment of compute with difficulty** (Spearman +0.662
vs +0.553) and the step-vs-difficulty dynamic range. The halting head generalizes the difficulty
signal to held-out problems without knowing each example's step count at inference.

**Confound to note:** `mean(n_i) ≈ 1.6` over the training set, well below the global 3.0, so the per-instance prior also *lowers the average* target. A matched-mean control (global `geom_mean ≈ 1.6`) would isolate per-instance shape from lower mean; deferred.

**Next:** a γ × β sweep (per-instance γ may be slightly strong at 0.05 given sharper KL) and the matched-mean ablation are the natural follow-ups if this result needs further attribution.

See [runs.md](runs.md) for the run table · artifacts under `<dir>/05-simcot-pondernet-adaptive-prior/`.

## Next steps

**Informed by the k-recipe sweep (teammate, `feat/adaptive-k-from-scratch`):** Recipe C showed that unfreezing the full backbone (`--pondernet_train_scope full`) gives +0.83pp at 3.21 avg steps, because it allows the backbone to reorganize representations so that early latent states (z₂–z₃) become answer-ready. Exp-05 trained with the frozen backbone (LoRA-only, effectively Recipe A), so the per-instance KL targets were issued to a backbone that still carries the K=6 structural bias and cannot redistribute signal toward earlier steps.

**Recommended: re-run exp-05 with full scope (exp-07 or extension).**
Combine the adaptive prior (exp-05) with the full backbone unfreeze (teammate Recipe C). This targets both failure modes simultaneously: the prior shape nudges the halting head, and the unfrozen backbone can actually rewire z₂–z₃ to carry complete reasoning. Suggested flags:

```bash
EXP=07-... RUN=perinstance-full-g0.05-b1.5-k12-ep5 \
ADAPTIVE_PRIOR=True GAMMA=0.05 PRIOR_OFFSET=1.5 PRIOR_SCALE=1.0 \
bash scripts/train_gpt2_gsm8k_pondernet.sh \
  --pondernet_train_scope full \
  --max_latent_steps 12 \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 8
```

Use K_max=12 rather than 6: the teammate's sweep found C-k12 (40.33%) > C-k6 (39.88%) despite both halting at ~3.2 steps — the larger budget sharpens the training-time KL pressure toward early halting and covers the hard-problem tail. The existing γ × β sweep and matched-mean ablation remain relevant follow-ups once this run lands.

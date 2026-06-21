# 06: per-instance truncated-K training (breaking the K=6 warm-start bias)

**Status:** active   **Dates:** 2026-06-20 → —

## What's being tested

**Hypothesis:** The SIM-CoT CODI warm-start encodes a structural K=6 bias: the auxiliary
decoder was trained to align z_k → reasoning step k for exactly k=1…6, so latent states
beyond step 6 carry no meaningful reasoning content, and the halting head can only learn
*early-exit from a K=6 model* rather than *genuine adaptive step-count*. The K sweep
(exp-03/04/05 + k_recipe_sweep) confirms this: avg_steps stabilises at ~5.5 regardless of
whether K_max=8, 10, or 12, because z_7+ are structurally OOD.

**Fix:** truncate the forward pass at K_i = max(1, n_i) per example during training.
For a 2-step example, the latent loop runs for exactly 2 steps; L_step, L_ans, and L_kl
are computed only over k=1…K_i. The backbone receives a direct gradient signal saying
"you must answer this problem correctly from {z_1, z_2} — there is no z_3 to fall back on."
Over training the backbone learns to pack complete reasoning into each example's natural
step count, giving the halting head genuine per-instance variation to learn from.

**What's varied:** truncated-K training (K_i = max(1, n_i)) vs current fixed-K=6.

**What's held fixed:** warm-start (full SIM-CoT CODI), train100k.jsonl, eff. batch 128,
lr 2e-5, 5 epochs, γ=0.05, adaptive prior β=1.5 α=1.0 (exp-05 recipe), seed 42,
full training scope (decoder unfrozen, RecipeC — decoder must adapt to variable-length
latent contexts).

## Setup

- **Backbone:** GPT-2, full warm-start from SIM-CoT CODI checkpoint.
- **K_i mapping:** `K_i = clamp(n_i, 1, K_max)` where `n_i = count_real_steps(example)`
  (i.e. `#expr − 1` with `include_last_cot=False`). K_max=8 covers the tail (0.21% of
  training examples need >6 steps).
- **Batching:** the latent loop runs to `max(K_i)` in the batch; per-example loss masks
  zero out L_step^(k) and L_ans^(k) for k > K_i[i]. This keeps the forward pass fully
  batched with no dynamic graph branching.
- **Halting supervision:** L_pondernet = Σ_{k=1}^{K_i} p_k · L_ans^(k) per example
  (masked sum). L_kl uses the per-instance adaptive prior from exp-05
  (`geom_mean_i = α·n_i + β`, clamped to [β, K_max]) — now doubly motivated: it both
  provides a soft halting target and matches the truncation boundary.
- **Training scope:** full (backbone LoRA + decoder + prj + halt head), same as RecipeC.
- **Baseline:** `05/perinstance-g0.05-b1.5-ep5` (40.49% @ 5.26 avg_steps, thr=0.8);
  fixed-K=6 recipeC (40.18%). No fresh baseline retrain needed.

### Implementation note

The key change is in `pondernet/src/model.py` `forward()`:

1. Compute `K_i` per example in the batch using the existing `count_real_steps()` helper.
2. Run the latent loop to `max(K_i_in_batch)` steps.
3. Build a boolean mask `active[B, K]` where `active[i, k] = (k <= K_i[i])`.
4. Apply mask to `L_step`, `L_ans^(k)`, and the halting distribution before summing.

No architecture changes required. The halting head, adaptive prior, and answer decoder are
all unchanged; only the loss aggregation and loop depth become per-example.

### Launch

```bash
EXP=06-simcot-pondernet-trunc-k RUN=trunc-ki-fullscope-g0.05-b1.5-ep5 \
ADAPTIVE_PRIOR=True GAMMA=0.05 PRIOR_OFFSET=1.5 PRIOR_SCALE=1.0 \
TRUNC_K=True \
bash scripts/train_gpt2_gsm8k_pondernet.sh --max_latent_steps 8 \
  --pondernet_train_scope full --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8
```

(`TRUNC_K=True` is the new env flag that activates per-instance K_i truncation in forward().)

## Findings

**Run:** `trunc-ki-fullscope-g0.05-b1.5-ep5` — completed 2026-06-21, trained on RTX 3090 (5 epochs / 5185 steps, bs=24 ga=4 eff-batch=96, ~2.0 s/it). Eval on RTX 3060, bs=16, thresholds 0.5/0.8/0.9.

Note: effective batch was 96 (not the planned 128) due to OOM at bs=32 and bs=32-resume; cold-start at bs=24 cleared the OOM. Steps per epoch = 1037 (not 778 for eff-batch 128).

### Epoch sweep (thr=0.5)

| epoch | step | acc (thr0.5) | acc (thr0.8) | avg_steps (thr0.5) |
|-------|------|-----------  |-------------|-------------------|
| 1 | 1037 | 34.65% | 34.19% | 4.531 |
| 2 | 2074 | 34.80% | 34.19% | 3.887 |
| 3 | 3111 | 35.78% | 35.78% | 3.792 |
| 4 | 4148 | 36.24% | 36.24% | 3.669 |
| **5** | **5185** | **36.39%** | **36.32%** | **3.660** |

**Winner: epoch 5** (step 5185; accuracy still rising — did not peak within 5 epochs).

### vs baseline (`05/perinstance-g0.05-b1.5-ep5`)

| metric | exp-05 (ep4) | this run (ep5) | delta |
|--------|-------------|----------------|-------|
| acc @ thr0.8 | 40.49% | 36.32% | **−4.17pp** |
| avg_steps @ thr0.5 | 4.365 | **3.660** | **−0.705** |
| avg_steps @ thr0.8 | 5.262 | 5.066 | −0.196 |
| Spearman @ thr0.5 | 0.650 | 0.596 | −0.054 |
| Spearman @ thr0.8 | 0.586 | 0.501 | −0.085 |

Trunc-K reduces average steps substantially but accuracy fell ~4pp below exp-05 and ~4pp below the exp-04 baseline.

### Steps-vs-difficulty (thr=0.5, ep5)

| #expr | n | avg_steps |
|-------|---|-----------|
| 0 | 18 | 2.333 |
| 1 | 65 | 2.077 |
| 2 | 357 | 2.588 |
| 3 | 364 | 3.412 |
| 4 | 290 | 4.697 |
| 5 | 138 | 4.877 |
| 6 | 57 | 4.930 |
| 7 | 21 | 5.524 |
| 8 | 9 | 5.778 |

Spearman r=+0.596 (thr0.5), p=8.2e-128. Step variation by difficulty is clearly present and the dynamic range is wide (2.08 to 5.78 steps). Easy problems (n_expr ≤ 2) now use 2–2.6 steps on average, dramatically lower than exp-05 (2.52–4.56). Hard problems (n_expr ≥ 6) top out near 5.5–5.8 steps.

### Interpretation

Trunc-K works mechanically: it forces the backbone to pack complete reasoning into each example's natural step count, and the halting head learns this pattern. The per-n_expr step distribution is the widest and most linear we have measured.

However, accuracy regressed ~4pp. Two confounds make attribution uncertain:

1. **Full-scope training converges slowly.** Backbone LoRA + prj unfrozen means more parameters moving simultaneously; the accuracy curve was still rising monotonically at ep5, suggesting the model had not converged. More epochs or a higher LR may close the gap.
2. **Eff-batch regressed from 128 → 96.** The OOM forced bs=24 instead of the intended bs=32; a smaller eff-batch can slow convergence and shift the optimal LR.
3. **Truncation may be too aggressive for hard examples.** With K_i = n_i, the model only sees `n_i` latent steps during training for examples where the answer requires more reasoning than the raw expression count suggests. A relaxed truncation (e.g. K_i = n_i + 2) might preserve accuracy better.

**Next steps:** (a) run more epochs (ep6–10) from the ep5 checkpoint to see if accuracy recovers; (b) try a relaxed truncation `K_i = n_i + 2` to give the model more breathing room; (c) match eff-batch=128 with bs=32 ga=4 on a larger GPU once available.

See [runs.md](runs.md) for the run table · artifacts under `<dir>/06-simcot-pondernet-trunc-k/`.

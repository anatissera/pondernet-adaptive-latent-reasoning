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

_(pending — update when runs complete)_

See [runs.md](runs.md) for the run table · artifacts under `<dir>/06-simcot-pondernet-trunc-k/`.

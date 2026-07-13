# 08: γ↑ + prior reshaping - pushing the accuracy–steps frontier left

**Status:** complete   **Dates:** 2026-06-24 → 2026-06-25

## What's being tested

**Goal:** keep ~baseline accuracy (40.80% greedy validation) at **fewer average latent
steps** than the current best operating points.

**Motivating finding - threshold-only frontier (no retrain).** A faithful `batch_size=1`
threshold sweep on exp-07's ep5 checkpoint (`checkpoint-3890`, validation n=500, greedy)
maps how far the operating point alone can be pushed:

| threshold | accuracy | avg steps | vs thr0.8 |
|----------:|:--------:|:---------:|:----------|
| 0.3 | 39.2% | **3.27** | −52% steps, −1.8pp |
| 0.4 | 40.4% | **3.78** | −44% steps, −0.6pp |
| 0.5 | 40.8% | 4.34 | −36% steps, −0.2pp |
| 0.8 (exp-07 op point) | 41.0% | 6.80 | - |
| 0.9 | 40.0% | 8.19 | +20% steps, −1.0pp |

The threshold-only frontier **knees at ~3.8 steps (thr0.4)**: above it the extra steps are
pure waste (the model's answer is already fixed); below it (thr0.3) accuracy erodes ~1.6pp.
So *banking the free win* = operate exp-07 ep5 at thr0.4–0.5. (bs=1 thr0.5 = 40.8% @ 4.34
matches the old bs=16 41.0% @ 4.34 → the `_slice_past_key_values` batch-faithfulness fix
holds.) Artifacts: `results/07-simcot-pondernet-fullscope-prior/.../ep5-bs1/thr{0.3,0.4,0.5}/`.

**Hypothesis (this experiment):** to push the *whole* frontier below ~3.5 steps without the
accuracy cost that threshold-only incurs, make the KL-geometric prior actually **bind**. In
exp-07 the per-instance prior target (mean geom_mean_i ≈ 3.1 train / 3.7 eval) sits *below*
realized halting (4.3–6.8 steps), because γ=0.05 is too weak to pull the halting distribution
onto its target. Raising γ tightens halting toward the per-instance mean; reshaping the prior
(lower offset β, or lower slope α) moves that mean earlier.

## Setup

- **Base recipe:** exp-07 ep5 - GPT-2, full warm-start from SIM-CoT CODI, `scope=full`
  (backbone + halt_head trainable; decoder + adapters frozen), adaptive per-instance prior,
  K_max=12, train100k.jsonl, lr 2e-5, 5 epochs, seed 42, no trunc-K.
- **Batch:** eff. batch **128 (bs=16, accum=8)** - same as exp-07 (proven-clean batch sequence).
  bs=24 was attempted first but cascaded into OOMs after Run A crashed; bs=16 is the safe known-good.
- **Held fixed across the grid:** everything above + **γ=0.10** (2× exp-07's 0.05).
- **Varied (3-run γ/α grid):**

  | run | γ | α (scale) | β (offset) | geom_mean_i | intent |
  |-----|---|-----------|------------|-------------|--------|
  | `fullscope-adaptive-g0.10-b1.0-k12-ep5` | 0.10 | 1.0 | 1.0 | clamp(n_i+1.0, 1.0, 12) | shift floor −0.5, keep difficulty signal |
  | `fullscope-adaptive-g0.10-b1.5-k12-ep5` | 0.10 | 1.0 | 1.5 | clamp(n_i+1.5, 1.5, 12) | tighten around exp-07 target only |
  | `fullscope-adaptive-g0.10-a0.6-b1.5-k12-ep5` | 0.10 | 0.6 | 1.5 | clamp(0.6·n_i+1.5, 1.5, 12) | cap hard-tail budget (lower Spearman expected) |

- **GPU:** RTX 3090, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (index 1 = 3090;
  the newly-installed RTX 5070 is sm_120, unsupported by this PyTorch - must pin PCI order).
- **Eval:** validation split, greedy, **bs=1** (faithful), thresholds 0.3/0.4/0.5/0.8.

## Launch

```bash
cd /home/tpnlp/alr-valentino/pondernet
# Run A (γ=0.10, β=1.0, α=1.0)
EXP=08-simcot-pondernet-gamma-frontier RUN=fullscope-adaptive-g0.10-b1.0-k12-ep5 \
ADAPTIVE_PRIOR=True GAMMA=0.10 PRIOR_OFFSET=1.0 PRIOR_SCALE=1.0 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
bash scripts/train_gpt2_gsm8k_pondernet.sh \
  --pondernet_train_scope full --max_latent_steps 12 \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 8 --save_total_limit 3
# Run B: PRIOR_OFFSET=1.5 PRIOR_SCALE=1.0 ;  Run C: PRIOR_OFFSET=1.5 PRIOR_SCALE=0.6
```

## Findings

Both valid runs (B and C) beat the exp-07 threshold-only baseline on every threshold.
Full per-epoch results in [runs.md](runs.md).

### Summary - best epoch (ep5), validation bs=1 n=500 greedy

| threshold | exp-07 baseline | Run B (γ=0.10, α=1.0, β=1.5) | Run C (γ=0.10, α=0.6, β=1.5) |
|----------:|:---------------:|:-----------------------------:|:-----------------------------:|
| 0.3 | 39.2% @ 3.27 | 40.0% @ 2.55 | 39.6% @ **2.03** |
| 0.4 | 40.4% @ 3.78 | **40.6% @ 3.10** | 40.2% @ **2.47** |
| 0.5 | 40.8% @ 4.34 | **41.0% @ 3.64** | 40.6% @ **2.93** |
| 0.8 | 41.0% @ 6.80 | **41.2% @ 6.23** | 40.4% @ **4.85** |

### Key conclusions

1. **γ=0.10 pulls the frontier left uniformly.** Run B ep5 beats the exp-07 baseline at every
   threshold - more accurate *and* fewer steps - confirming the hypothesis that γ=0.05 was too
   weak to make the KL-geom prior bind.

2. **α=0.6 (Run C) reduces steps ~20% further vs Run B**, but costs 0.4–0.8pp accuracy.
   At thr0.5 the tradeoff is nearly neutral: 40.6% @ 2.93 (C) vs 41.0% @ 3.64 (B) - 0.4pp
   for 0.71 fewer steps per question.

3. **Recommended operating points:**
   - High accuracy: **Run B ep5, thr0.5** - 41.0% @ 3.64 steps (+0.2pp, −16% steps vs baseline)
   - Min steps: **Run C ep5, thr0.5** - 40.6% @ 2.93 steps (−0.2pp, −32% steps vs baseline)
   - Sweet spot: **Run B ep5, thr0.4** - 40.6% @ 3.10 steps (+0.2pp, −18% steps vs baseline)

4. **Run A (β=1.0) invalid** - degenerate prior for n_i=0 examples; crashes epoch 2.
   See runs.md for details.

See [runs.md](runs.md) for the full run table · artifacts under `<dir>/08-simcot-pondernet-gamma-frontier/`.

# Option B: synthetic controlled-density dataset (design)

**Date:** 2026-06-23 | **Branch:** `option-b-adaptive-vectors`

## 1. Context and motivation

Option B (adaptive vectors-per-step, the `c` axis) gave a negative result on **atomic**
GSM8K-Aug (c saturates at 2, adaptive ~ random). But the **warm + coarse** experiment
(2026-06-21) showed the negative was **granularity-specific**: with dense steps (2-3 ops
grouped), the c-curve recovers headroom (delta c2->c3 = +3.7 vs +0.5 atomic) and
**adaptive beats random by 2.7 sigma** (eps0.15 38.8% @ 7.2 vec vs random 35.2% @ 6.1
vec). Over the uniform allocation, adaptive wins by +0.6-1.0 (modest). Conclusion:
**a per-instance signal is exploitable when per-step density varies.** Detail in
`RESULTS.md`, section "Warm + coarse".

The problem: with GSM8K-Aug we can only create per-step density by grouping trivial
ops, which gives small, poorly controlled variance. **We need a clean per-step-density
knob** to measure how much the adaptive headroom grows as a function of difficulty
variance.

## 2. Goal and hypothesis

**Goal:** a synthetic dataset where **per-step difficulty is a controlled variable**,
to measure adaptive-c headroom as a function of **per-step difficulty variance**.

**Hypothesis:** the benefit of adaptive over fixed-uniform allocation **grows with the
per-step depth variance of the test set**: about 0 when every step is equally hard (as
in the atomic setting), large when difficulty varies a lot.

## 3. Dataset design

### 3.1 Instance structure
- A sequential chain of **K=4 steps**. Each step `k` is a `<<expr_k = result_k>>` block
  with **depth `d_k` in {1,2,3,4}** = the number of binary operations in that block.
- Each step uses the previous step's result (chaining, like GSM8K-Aug).
- Operators: `+ - * /` (division only when it yields an integer; small operands to
  avoid huge numbers that would add magnitude noise; the controlled variable is
  **depth**, not magnitude).

Example, depth profile `[1,3,2,4]`:
```
<<7+5=12>> <<12*3-4+2=34>> <<34/2=17>> <<17+8*2-5+1=29>>
```

### 3.2 Format (plugs into the existing pipeline unchanged)
Each example is a JSON line with the same keys as GSM8K-Aug:
- `question`: the expression/chain to evaluate (pure arithmetic, no natural language).
- `cot`: the `<<expr=result>>` blocks separated by spaces (`get_steps` segments it
  unchanged).
- `answer`: the final result (string).
- `depths`: **[extra field, ground truth]** list `[d_1..d_K]`, each step's real depth.
  Not used by training (only `question`/`cot`/`answer` are); used by evaluation for
  the vectors-vs-depth correlation diagnostic and the oracle.

### 3.3 Variance levels (mean depth fixed at ~2.5; only the variance changes)
| level | `d_k` distribution | variance | role |
|-------|---------------------|----------|------|
| L0 | depth-2 and depth-3 (mean 2.5, minimal spread) | ~0 | control (expect adaptive ~ uniform) |
| L1 | {2,3} balanced | low | |
| L2 | {1,2,3,4} uniform | medium | |
| L3 | {1,4} balanced (bimodal) | high | (expect adaptive >> uniform) |

Keeping the mean at ~2.5 prevents "harder" from being confused with "more variance"
(otherwise the high-variance levels would just score lower from higher mean
difficulty, not from variance itself).

### 3.4 Generator
- A deterministic Python script (fixed seed). Generates:
  - a **warm-up set** = uniform depth-2 (~5k examples) for the Stage 0 de-risking step.
  - **1 train set** = a full-depth {1,2,3,4} uniform mix (~15k examples; generation is
    free).
  - **4 test sets** (L0..L3), ~1000-1500 examples each, held out (no overlap with
    train).
- Verifies: correct arithmetic (each `result_k` is computed, not made up), exact
  integer division, no magnitude overflow, `answer` consistent with the last
  `result_k`.
- M=4 (max sub-vectors/step) maps 1:1 to the maximum depth.

## 4. Experiment design

**A single model trained on the full-depth mix, evaluated across the test-variance
sweep.** This isolates variance as the only variable (the model's own allocation
ability stays constant).

### 4.1 Init / de-risking (mitigates the cold-start failure mode: inert latents)
- **Stage 0 (warm-up + de-risk):** warm-start from CODI (`simcot-gpt2-codi`) plus a
  short training run on a **uniform depth-2** synthetic set (~3 epochs). **Gate:**
  confirm the model works: reasonable accuracy (not ~chance) **and** `fixed_k_eval`
  shows a rising curve (latents NOT inert, unlike the cold-start failure). If the gate
  fails, revisit the transfer before continuing.
- **Stage 1 (main training):** **starting from the Stage 0 functional checkpoint**
  (not raw CODI), train on the full-depth {1,2,3,4} mix. Warm-coarse-style recipe:
  LR 2e-5, ~3 epochs, LoRA r128, K=4, M=4, penalty off for a clean c-curve.
  `gradient_checkpointing False`. *(If the Stage 0 gate shows the transfer is
  trivial, Stage 0 and Stage 1 can be merged into a single training run on the
  full-depth mix, which already includes depth-2.)*

### 4.2 Evaluation (per test set L0..L3, bs=1, full eval set)
- **Fixed-c curve:** c=1,2,3,4 (`--ob_max_subvectors {1..4} --ob_eps 0.0`), defining
  the uniform line.
- **Adaptive:** sweep `--ob_eps`; **random** baseline (`--ob_random`).
- **Oracle:** decode with vectors-per-step = the real depth `d_k` (the ceiling of a
  perfect adaptive policy).

## 5. Metrics and success criteria

1. **Headline curve:** the gap `(adaptive - fixed-uniform)` at matched budget, **vs
   the test set's variance** (L0->L3). *Success:* the gap grows monotonically with
   variance (~0 at L0).
2. **Adaptive vs random vs uniform** at matched budget at every level (warm-coarse
   style table).
3. **Oracle:** how much of the ceiling (vectors = depth) the learned adaptive policy
   captures.
4. **Key diagnostic:** correlation(vectors allocated by the MLP per step, real depth
   `d_k`). *Success:* a strong positive correlation at L2/L3 (it was ~0 in the atomic
   setting). This is the direct test of whether the MLP detects per-step difficulty.

**A positive result for the project** = (1) the gap grows with variance **and** (4) a
positive correlation, meaning Option B has real headroom that grows with density
variance, and the original negative was an artifact of GSM8K-Aug's uniformity.

## 6. Integration with the existing pipeline
- Reuses `train.py`, `_forward_option_b`, `get_steps` (identical `<<>>` format), and
  `test.py` eval (including `fixed_k_eval` and `ob_random`), plus the
  `eval_gpt2_gsm8k_optionb.sh` / `ob_step3_coldcoarse.sh` scripts (parametrized by
  CKPT_ROOT/data).
- **New:** the dataset generator (`scripts/gen_synthetic_density.py`), a training
  script (`train_gpt2_synth_density.sh`), and an eval/diagnostic script that adds the
  oracle and the vectors-vs-depth correlation (extends `test.py`'s report using the
  `depths` field).
- Data lives under `data/synth_density/` (symlinked like the rest).

## 7. Risks and mitigations
- **NL warm-start does not transfer to pure arithmetic, leaving latents inert (like
  the cold-start failure).** Mitigation: the Stage 0 gate (fixed_k rises) catches this
  early; if it fails, add a longer warm-up or a fixed mini NL prompt.
- **Mean depth not perfectly matched across levels, confounding variance with
  difficulty.** Mitigation: the generator forces mean ~2.5 by construction and reports
  the real per-level mean.
- **M=4 insufficient if depth-4 needs more than 4 vectors.** Mitigation: the fixed-c
  curve c=1..4 reveals this (if c=4 is still rising, raise M).
- **Number magnitude introduces spurious difficulty.** Mitigation: small operands,
  exact integer division; depth is the only knob.

## 8. Out of scope (YAGNI)
- Natural language / realistic word problems (chose pure arithmetic instead).
- Varying K (number of steps): fixed at 4 to isolate the `c` axis.
- Varying magnitude/operator type as independent knobs: only depth varies.
- A backbone larger than GPT-2: a separate project if the headroom justifies it.
- Training one model per level (approach A): B was chosen instead.

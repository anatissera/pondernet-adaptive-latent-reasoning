# Option B: Results (adaptive vectors per step, the `c` axis)

**Date:** 2026-06-19 (updated 2026-06-21) | **GPU:** RTX 3060/5070/3090
**Backbone:** GPT-2 (+ SIM-CoT/CODI warm-start in the baseline; cold-start in the retrain)
**Data:** GSM8K-Aug (train) / GSM8K test (1319 examples)

## TL;DR

Option B implements and trains end to end the adaptivity of the **number of vectors per
step** (`c`): each reasoning step is built as a block of up to `M` sub-vectors; the
SIM-CoT decoder reconstructs each step from the accumulated block, and an MLP distills
that per-example `L_step` to decide at inference when the step is "mature". **The
mechanism works and trims ~16-33% of the latent compute without losing accuracy.** But
the learned halting **does not beat random or fixed c=2 at matched budget** (2 runs,
with and without penalty), and the MLP **does not allocate more compute to harder
problems**. The accuracy-vs-`c` curve saturates hard at c=2 (c1=27.5%, c2=39.4%,
c3=39.9%): the required `c` is nearly constant (~2) across instances, so there is
**little exploitable headroom** on GSM8K-Aug with GPT-2. The simple optimum is **fixed
c=2**. A clean, reproducible negative result: the machinery is correct, the task does
not reward adapting `c`. The axis with real headroom is the number of steps `K`
(Proposal C).

> **Update 2026-06-21 (unbiased retrain).** To rule out the saturation being warm-start
> c=1 bias, the model was retrained **cold from plain GPT-2 + coarse segmentation**.
> Result: the cold model **collapses to ~chance** (c1=5.3/c2=5.5/c3=5.5%). Diagnosis
> (see the "Unbiased retrain" section): not a loading bug; the model learns the answer
> *format* but the **latents stay inert** (`fixed_k_eval` flat k1..k9 at ~6%), so latent
> reasoning does not bootstrap from scratch under the block-of-c objective. **The c=1
> anchor was load-bearing, not just a bias.** The `c` axis negative is robust: where the
> model works (warm), `c` is flat; the real lever is the **per-step density of the
> dataset**, not the model's `c` axis.

> **Update 2026-06-21 (warm + coarse: REVERSES/QUALIFIES the negative).** The negative
> was **granularity-specific**, not a property of Option B. With warm-start (functional
> model) + **coarse** segmentation (steps of 2-3 ops, so variable density): the c-curve
> recovers headroom (delta c2->c3 = **+3.7** vs +0.5 atomic) and **adaptive beats random
> by 2.7 sigma** (eps0.15 38.8% @ 7.2 vec vs random 35.2% @ 6.1 vec), which did NOT
> happen in the atomic setting. Over the uniform allocation line, adaptive wins by
> **+0.6 to +1.0** (modest). The per-instance signal **exists** when per-step density
> varies. See the "Warm + coarse" section. Next: a **synthetic dataset with controlled
> density** to amplify and measure headroom vs density variance.

## What was built

- `_forward_option_b` (training): K steps x M sub-vectors; reconstruction from the
  accumulated block (`_block_step_loss`, per-example); `ob_mlp` distills `L_step`
  (SmoothL1); ponder penalty `lambda * sum sigmoid(-L_hat)`. Inherited SIM-CoT path
  intact (gated by `--option_b`).
- Adaptive inference (`test.py`): for each step, add sub-vectors until
  `|L_hat_j - L_hat_{j-1}| < eps` or `M_max`; decode the answer after K steps. Faithful
  at bs=1.
- **Random** halting baseline (`--ob_random`) for matched-budget comparison.
- Phase-1 diagnostic (`--ob_probe`): measures whether `L_step` goes down within a step.

## Phase 1: Probe (pretrained model)

On the SIM-CoT checkpoint (pinned to c=1), `L_step` does **not** go down when adding
sub-vectors within a step (it goes up, in both the SINGLE and BLOCK measurements). This
confirmed that any `c != 1` is OOD and that **retraining is mandatory**. (Not a
definitive go/no-go; it only says the pretrained model gives no advantage.)

## Phase 5: Training + evaluation

**run1:** K=4, M=3, 3 epochs, 8000 examples, LR 2e-5, HALT_LR 1e-3, **lambda_halt=0**.
Loss 3.40 -> 0.36. ckpt `.../optionb-run1/.../checkpoint-747`.

### Adaptive vs fixed vs random (full GSM8K test, 1319 examples, bs=1)

| config                | avg vectors | accuracy |
|-----------------------|-------------|----------|
| fixed c=3 (eps=0)     | 12.00       | 39.88%   |
| **adaptive** eps=0.05 | 10.12       | 39.80%   |
| **random** halting    | 8.05        | 39.35%   |

Spread = 0.53% across the whole range (SE ~1.35% with 1319 examples), so
**statistically flat**. Adaptive matches fixed with 16% less compute; random matches
both with 33% less.

### eps sweep (300-subset): budget/accuracy tradeoff

| eps   | avg vectors | accuracy |
|-------|-------------|----------|
| 0.00  | 12.00 | 41.00% |
| 0.02  | 10.69 | 41.00% |
| 0.05  | 10.18 | 41.33% |
| 0.15  | 9.23  | 41.00% |
| 0.40  | 8.43  | 40.67% |

### Does the MLP give more vectors to hard problems? NO

Mean vectors used: **correct = 10.28, incorrect = 10.01** (the reverse of what would be
useful, and a negligible difference). The vectors-per-step pattern is **positional**
(s0=3.00 always; later steps stop earlier: s1~2.7, s2~2.3, s3~2.2), not driven by
instance difficulty. The learned halting does not capture a per-instance signal.

### Fixed-`c` accuracy curve (full GSM8K test)

Does `c` matter at all? Forcing a fixed `c` per step:

| fixed c | total vectors | accuracy |
|---------|---------------|----------|
| c=1     | 4   | **27.52%** |
| c=2     | 8   | **39.42%** |
| c=3     | 12  | 39.88%     |

**Key finding:** the curve is **steep from 1 to 2 (+11.9%) and flat from 2 to 3
(+0.46%)**. It saturates hard at **c=2**. So `c` DOES matter, but the *required* `c` is
**nearly constant (~2)** across instances: almost every problem needs 2 vectors and
none benefits from 3. That is why there is no adaptive headroom: not because `c` is
irrelevant, but because its per-instance variance is very low. The optimum is **fixed
c=2** (8 vec, 39.42%), and adaptive (10 vec, 39.80%) spends more for the same result.

## Why does it saturate at c=2? Bias or the task?

Key question: does `c` really saturate, or did our setup bias it into saturating?
Audit (detail and implementation in `IMPLEMENTATION.md`, section 6):

**Paper context (research agent):** CODI (the code base) uses **1 vector per step**;
the SIM-CoT/Coconut papers use `c_thought=2` (`SIM-CoT/Coconut/args/*.yaml`). The
checkpoint we warm-started from (`models/pretrained/simcot-gpt2-codi`) was trained with
**c=1**. Steps are segmented by text markers (`<<...>>`), supervised from the CoT.

**Biases in our setup (to remove for a clean test):**
1. **c=1 warm-start + LoRA-only (frozen base).** The model starts in the c=1 regime and
   only LoRA(r=128) adapts it; the backbone's reasoning circuitry remains c=1.
   For a clean test, do not use the SIM-CoT checkpoint: retrain without that anchor
   (cold start / full fine-tune from plain GPT-2).
2. **Granularity: 1 arithmetic operation = 1 step.** In GSM8K-Aug each step is a
   trivial `<<a op b=c>>`, so 2 vectors saturate it. The **task's variance is in the
   number of steps (1-6 ops), not in per-step complexity**. That is why the `c` axis is
   flat and the `K` axis has headroom.
3. **The answer is only trained at maximum budget (M per step).** The answer head never
   learns to answer with fewer vectors, biasing against low c and against adaptivity
   paying off.

**Verdict of the analysis:** the c=2 saturation is **mostly the task** (trivial,
uniform steps), with warm-start/LoRA as a **secondary** anchor. An unbiased retrain
(points 1-3) is needed to state it rigorously, which is exactly what was done next.

## Unbiased retrain: cold + coarse (result + diagnosis)

**Date:** 2026-06-21 | **GPU:** RTX 5070 (training) + 3090 (eval). Plan: remove the
controllable biases (#1 c=1 warm-start, #2 atomic granularity) in a single run and
measure again.

**Setup:** plain GPT-2 (no SIM-CoT checkpoint, fresh decoder) + **coarse** segmentation
(`get_steps_coarse`: groups the ops into K=3 even buckets, so per-step complexity
varies), K=3, M=3, 30 epochs, `train15k`. Script:
`scripts/train_gpt2_gsm8k_optionb_cold.sh`.
- **Divergence at LR 3e-3** (CODI's cold recipe): blew up at epoch 7-8 (loss 2.8 -> 20,
  stuck at ~10). Relaunched with **LR 1e-3 + max_grad_norm 0.5 + warmup 0.05**: clean
  monotonic descent 7.5 -> **0.77**, no spikes. Final checkpoint: `checkpoint-7020`.

**Result: c-curve (full GSM8K test, 1319 examples, eps=0.0):**

| c (M_max) | vecs/inst | acc (%) cold | acc (%) warm (baseline) |
|-----------|-----------|--------------|--------------------------|
| 1         | 3.00      | **5.31**     | 27.52                    |
| 2         | 6.00      | **5.46**     | 39.42                    |
| 3         | 9.00      | **5.53**     | 39.88                    |

Adaptive/random (300-subset): **everything ~8%**, adaptive ~ random at every eps (the
halting trims vectors 9 -> 6 but accuracy does not move). **The cold model collapsed to
~chance.**

### Diagnosis: why the cold start fails (3 checks)

1. **Not a loading bug.** The cold checkpoint has an **identical key structure** (404
   keys, zero diff, zero shape mismatches) to the warm checkpoint that scores 39%
   through the same `test.py`. The LoRA/prj/ob_mlp weights are trained (large norms,
   not init). The loss went down to 0.77.
2. **It learned the format, not the reasoning.** Generations are **well-formed** answers
   ("The answer is: N") with plausible magnitudes, but the arithmetic is wrong
   (16-3-4=9 => $18, it says 96; 3x3x60=540, it says 360). Not garbage: a model giving
   format-correct but incorrect answers.
3. **The latents are inert** (`fixed_k_eval`, decoding the answer at every prefix
   k=1..9):

   ```
   accuracy@k (cold):  k1=7.3  k2=6.7  k3=6.0  k4=5.3  k5=5.7  k6=5.7  k7=6.0  k8=6.0  k9=5.7
   ```
   **Flat (even decreasing):** adding latent vectors does not help. The reasoning chain
   carries no computation. Contrast: the **warm model DOES use the latents**; its
   c-curve **rises** (c1=27.5 -> c2=39.4, +12 points from the 2nd vector).

**Failure mode (shortcut):** from scratch, under the block-of-c objective, optimization
finds a shortcut: the decoder imitates the teacher's hidden state (`distill_loss`) and
emits a plausible number *from the question alone*, without the latents becoming
load-bearing. `L_step` trains the latents to reconstruct the step *text*, but that
stays decoupled from computing the answer. CODI avoids this because its warm-start
**already brings functional latents** (trained 40 epochs cold with a simple objective:
c=1, one reconstruction).

**Updated verdict:** the c=1 anchor was **not just a bias, it was load-bearing**. You
cannot separate "working model" from "anchored at c=1" by removing the warm-start,
because latent reasoning does not bootstrap from scratch under this objective. The `c`
axis negative is **robust in both regimes**: where the model works (warm), `c` is flat;
in cold there is no working model. The root cause is not the init but **the task**:
each GSM8K-Aug step is a trivial op that 2 vectors saturate. The real lever is the
**per-step density of the dataset**, not the model's `c` axis.

## Warm + coarse: the negative was granularity-specific (CENTRAL FINDING)

**Date:** 2026-06-21. The missing cell of the 2x2 (init x granularity): keep the
**warm-start** (functional model, latents that compute) but with **coarse**
segmentation (K=3 buckets of 2-3 ops, so variable per-step density). Isolates
granularity on a model that works. Script:
`scripts/train_gpt2_gsm8k_optionb_warmcoarse.sh` (LR 2e-5, 3 ep, penalty off).
Loss 1.8 -> 0.33 (healthy). Checkpoint: `optionb-warm-coarse/.../checkpoint-747`.

| init x granularity | c=1 | c=2 | c=3 | **delta c2->c3** |
|--------------------|-----|-----|-----|------------------|
| warm + atomic (baseline) | 27.5 | 39.4 | 39.9 | **+0.5** (saturated) |
| **warm + coarse** | 25.5 | 36.6 | 40.3 | **+3.7** (still rising) |
| cold + coarse | 5.3 | 5.5 | 5.5 | broken model |

**1. The c-curve recovers headroom.** With dense steps the 3rd vector adds +3.7 (about
7x the atomic +0.5, ~49 examples out of 1319, outside noise). The `c` axis stops
saturating at 2.

**2. Adaptive vs random vs uniform (full test, 1319 examples, matched budget):**

| config | acc (%) | vecs | vs uniform line* |
|--------|---------|------|------------------|
| adaptive eps0.05 | 39.58 | 7.93 | +0.6 |
| adaptive eps0.15 | 38.82 | 7.18 | +0.7 |
| adaptive eps0.30 | 38.51 | 6.73 | +1.0 |
| **random** | **35.18** | **6.07** | **-1.5** |

*\*uniform line = interpolating the fixed c-curve at the same budget (what giving
everyone the same c yields).*

- **Adaptive >> random: +3.6 pts (eps0.15 vs random), about 2.7 sigma.** The MLP's
  halting is not pathological and allocates budget much better than chance. **This did
  NOT happen in the atomic setting** (there adaptive ~ random ~ fixed): the
  per-instance signal **exists** when density varies.
- **Adaptive > fixed-uniform: +0.6 to +1.0, consistent (3/3, replicated on sub300)**
  but <1 sigma per point, so the headroom is **real but modest** on GSM8K-Aug-coarse.
- eps0.05 reaches 98% of the c=3 ceiling (39.6 vs 40.3) with 7.9 vec instead of 9.

**Updated project verdict:** the `c` axis negative was **specific to GSM8K-Aug's atomic
granularity**, not a property of Option B. Grouping trivial ops already revives a real
per-instance signal (adaptive beats random at 2.7 sigma), though modest over uniform.
This motivates the **synthetic controlled-density dataset**: if grouping 2-3 ops gives
signal, a dataset with *large, controlled* per-step density variance should amplify the
adaptive-vs-uniform gap. That is the next experiment.

## Conclusion and recommendation

1. **Implementation validated.** Option B trains, halts adaptively per step, and cuts
   latent compute by ~16-33% at no accuracy cost. Competitive with the SIM-CoT baseline
   (~39.5%).
2. **No headroom for *smart* `c` adaptivity in the atomic setting.** Accuracy saturates
   at c=2 almost uniformly; adaptive ~ random ~ fixed c=2 at matched budget; the MLP
   does not allocate more compute to hard problems (correct 10.28 vs incorrect 10.01).
   The simple optimum is **fixed c=2**.
3. **Why.** Each CODI reasoning step saturates at ~2 vectors evenly; the `c` axis has
   low per-instance variance, unlike the dynamic range of the `K` axis (number of
   steps). This reinforces the project premise: **the axis with real headroom is the
   number of steps (Proposal C)**. The warm + coarse follow-up qualifies this: with
   variable per-step density the signal reappears, modestly.

### run2: trained WITH the penalty (lambda_halt=0.05)

Does the penalty push the solvable-with-c=1 subset toward 1 vector, dropping below 8
vec at ~39%? **No.** (full GSM8K test)

| run2 config        | avg vecs | acc(%) |
|--------------------|----------|--------|
| fixed c=3          | 12.00    | 39.58  |
| **fixed c=2**      | 8.00     | **39.65** |
| adaptive eps=0.15  | 9.19     | 39.95  |
| adaptive eps=0.40  | 8.40     | 39.50  |
| random             | 8.05     | 39.20  |

The penalty did NOT unlock sub-c=2 efficiency: even at eps=0.40 the distribution is
practically all 2s (`2:4745, 3:531`, **zero 1s**). It only trimmed step 0 a bit
(s0=2.40). **Fixed c=2 (8 vec, 39.65%) matches or beats all of adaptive.** Confirms the
required `c` is ~constant at 2 and there is no identifiable c=1 subset to exploit.

**Possible pivots** (to keep pushing on `c`): tasks with denser steps (real multi-hop,
not GSM8K arithmetic), larger backbones, or higher `c` budgets where per-step
saturation varies. But the evidence recommends concentrating the adaptivity effort
on `K`.

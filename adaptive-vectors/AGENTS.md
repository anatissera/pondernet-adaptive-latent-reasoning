# Option B: traceability log (decisions and changes)

Phase-by-phase log of Option B (the `c` axis: adaptive vectors per step). Each phase
records **what was decided** and **what changed**, with dates. Read together with
`README.md` (the idea) and the approved plan.

> Convention: this file is updated as part of **every** phase's commit.

---

## NEW PHASE (2026-06-19): UNBIASED retrain: cold-start + coarse steps

**Why.** The negative result (c saturates at c=2) could have been biased by: (1)
warm-start from the c=1 SIM-CoT checkpoint + LoRA-only (frozen base) + only 3 epochs,
so the model never left the c=1 regime; (2) atomic granularity: 1 op = 1 step, each
step trivial, so 2 vectors saturate. Full audit in `IMPLEMENTATION.md` section 6.
Decision with the user: **cold-start from plain GPT-2 (no SIM-CoT checkpoint, fresh
decoder, ~30 epochs) + coarse segmentation**, in a single run. Work in a dedicated
worktree `/home/tpnlp/alr-anapaula-optionb` (data/models/outputs/results symlinked to
the canonical copy; `.venv` symlinked to the canonical one, `UV_NO_SYNC=1` so the
shared venv is untouched). All on the 3060.

**Code changes.**
- `get_steps_coarse` (src/model.py): finds the per-op segments (like `get_steps`) and
  distributes them into K **even** buckets (front-loaded) instead of 1-op-per-step +
  merge-into-the-last. Each step = concatenation of its bucket (one final eot).
  Inherited `get_steps` intact. Verified: 4 ops K=3 -> [op0 op1][op2][op3] (even) vs
  atomic [op0][op1][op2 op3].
- `ob_coarse_steps` flag (default False); `_forward_option_b` and `_ob_probe` pick
  `get_steps_coarse` when it is on. Inference untouched (it does not reconstruct text).
- `scripts/train_gpt2_gsm8k_optionb_cold.sh`: plain GPT-2, **without `--simcot_ckpt` or
  `--decoder_path`**, LR 3e-3, LoRA r128, 30 epochs, K=3 M=3, coarse, BS8xACCUM8, 15k.
- `scripts/ob_smoke.sh`: `COARSE` env for a coarse-steps smoke test.

**Smoke (coarse, overfit 8, warm-start, 3060):** ce 0.039->0.002, l_step 0.92->0.001,
l_dist 0.14->0.04, L_hat tracks L_step, so the coarse segmentation wiring is correct.

**Cold sanity (64 examples):** no warm-start (does not load simcot), no OOM,
~1.4 batch/s.

**Run in progress:** `optionb-cold-coarse` (tmux `ob-cold`), 7020 opt-steps (~11h),
lambda_halt=0. Eval pending on completion (c-curve + adaptive vs random, compare with
the warm+atomic baseline).

---

## Context (why this branch exists)

The project makes SIM-CoT's latent budget adaptive along two orthogonal axes:

- **The `K` axis: number of reasoning steps.** Option C works on this in `pondernet/`
  with a PonderNet-style halting head that decides *when to stop reasoning*.
- **The `c` axis: number of vectors per step (THIS branch, Option B).** Keeps the
  number of steps fixed and adapts *how many sub-vectors* build each step, distilling
  the per-step reconstruction loss `L_step` of SIM-CoT's auxiliary decoder into an MLP
  that survives inference.

`pondernet/` and Option B are **different**: they share only the SIM-CoT/CODI
*harness*, not the adaptation logic.

---

## Design decisions (agreed with the user)

- **D1: Code base.** Copy the proven `pondernet/` harness into `adaptive-vectors/`, remove the
  PonderNet halting path (`K` axis), and add the Option B logic (`c` axis) behind the
  `--option_b` flag. Reason: the harness already has the data wiring
  (`data/gsm8k_aug/train15k.jsonl`), the decoder fetch, and the
  `gradient_checkpointing` gotcha handling. `k-classifier/`, `pondernet/`, and the original
  SIM-CoT path are **not touched**.
- **D2: Branch.** `option-b` created from `main` (sibling of `option-a`), not from the
  proposal-C branches, to avoid dragging in the `K` axis work.
- **D3: Sequence: probe first.** Before building everything, a read-only diagnostic
  phase verifies the central assumption (that `L_step` goes down when adding
  sub-vectors within the same step). It is a GO/NO-GO gate.
- **D4: Documentation.** Keep `README.md` (idea/context) and this `AGENTS.md` (log of
  decisions and changes) updated phase by phase for traceability.
- **D5: Confirmation by diff.** Every change to the forward pass or the loss is shown
  as a diff and waits for the user's confirmation before being applied, even after the
  plan was approved. After each change: an overfit smoke test (4-8 examples).

## Accepted risks (to watch)

- **R1: The intra-step descent assumption is not guaranteed.** Re-feeding the hidden
  state may drift toward the *next* step instead of refining the current one.
  Mitigation: the Phase 1 gate; and in training, supervising the decoder to
  reconstruct step `k`'s text from *each* sub-vector of that step, creating pressure
  toward a decreasing curve.
- **R2: Weak train/inference coupling.** Training generates `M` fixed sub-vectors (to
  compute the regression targets in batch); the halting threshold only acts at
  inference. The penalty is a regularizer, not an expected-loss-over-the-halt like
  PonderNet's. Acceptable for v1, noted.
- **R3: Target stability.** `L_step` is a scalar CE in bf16 with variable scale, so the
  regression target is detached and (configurable, `ob_detach_hk`) `h_k` is
  stop-gradiented toward the MLP in v1.

---

## Change log

### Phase 0: Scaffold + docs (no behavior change)
**Date:** 2026-06-19

- `option-b` branch created from `main`.
- Copied the `pondernet/` harness into `adaptive-vectors/`: `src/model.py`, `train.py`,
  `test.py`, `smoke_optionb.py` (formerly `smoke_pondernet.py`) and `scripts/` (only
  generic helpers: `fetch_simcot_decoder.py`, `gcp_setup.sh`,
  `profile_batch_size.py`, `train_gpt2_gsm8k_pondernet.sh`,
  `eval_gpt2_gsm8k_pondernet.sh`). The `K`-axis-specific scripts were **discarded**
  (`sweep_k_recipe.sh`, `sweep_pondernet_gamma.sh`, `eval_gpt2_gsm8k_fixedk.sh`, etc.).
- Added to `TrainingArguments` (in `src/model.py`) the flags `option_b`,
  `ob_subvectors_per_step`, `ob_mlp_hidden`, `ob_detach_hk`, `ob_lambda_ans`,
  `ob_lambda_step`, `ob_lambda_dist`, `ob_lambda_halt`, `ob_eps`, `ob_max_subvectors`,
  `ob_probe`. **All inert by default** (`option_b=False`): the inherited path does not
  change until `--option_b` is passed.
- Wrote `README.md` and this `AGENTS.md`.
- No changes to the forward pass or the loss yet.

### Phase 1: Feasibility probe -> **NO-GO** (on the pretrained checkpoint)
**Date:** 2026-06-19 | GPU: RTX 3060 (PCI idx 2) | log: `outputs/optionb-probe/probe.log`

**Changes (read-only, gated by `--ob_probe`):** added to `src/model.py`
`_explain_loss_for` (a faithful extraction of `forward`'s decoder block), `_ob_probe`
(generates M sub-vectors per step by re-feeding the hidden and measures `L_step` of
THAT step's text from each sub-vector), and a hook at the start of `forward`. Script:
`scripts/ob_probe.sh` (SIM-CoT warm-start, M=4, 10 batches on the 3060).

**Result (consistent across the 10 batches):** `L_step` does **NOT** go down when
adding sub-vectors within a step; the mean curve rises monotonically:

```
mean over non-pad steps (10 batches): subvec  0     1     2     3
                                              0.08-0.89 -> ~1.0-1.7 (always rises)
step 0 (typical):  0.19 -> 2.35 -> 3.17 -> 3.18   (the 1st vector reconstructs almost perfectly, then collapses)
steps 1-3:         ~0.8-1.2 FLAT (e.g. 0.95 -> 0.88 -> 0.88 -> 0.95)
```

**Interpretation.** The SIM-CoT/CODI checkpoint was trained with **exactly 1 latent
vector per step**. Re-feeding the hidden does NOT refine the current step: it moves
toward the **next** step, so reconstructing the current step's text gets worse. The
first vector is already "mature" (step 0 ~0.2); later steps stay flat. **The central
assumption of proposal B (that `L_step` goes down across sub-vectors, so the MLP can
detect maturity when it "stops going down") does not hold on the pretrained model.**
Risk **R1** materialized.

**Important nuance (does not bury the idea, it recharacterizes it).** The probe
measures the *pretrained* model. Proposal B inherently requires **retraining** with
per-sub-vector supervision (Phase 2: the decoder reconstructs step `k`'s text from
*each* sub-vector). So B is **not** "distilling for free a signal that already
exists": the multi-vector dynamic must be **induced** by training. That changes the
cost/risk and was escalated to the user before investing in Phase 2.

**Additional observation (the `c` axis vs CODI).** The proposal's context describes
SIM-CoT with `c=2` vectors per step, but **this** code (CODI) uses 1 vector per step.
The paper's "c axis" does not map 1:1 onto CODI; in real SIM-CoT a step is a block of
`c` contiguous latent tokens before the decoder supervises that step. Making `c`
adaptive there is different from "re-feed and watch `L_step`".

**Decision:** PAUSE at the GO/NO-GO gate (as the plan anticipated). Options escalated
to the user: (A) continue to Phase 2 and *induce* multi-vector refinement by training
(a research bet; re-run the probe afterwards to see if the curve inverts); (B) rethink
the maturity signal / question whether the `c` axis has headroom (the data suggests
c=1 is already near-optimal per step); (C) reformulate what a "sub-vector" is to get
closer to SIM-CoT's real block-of-`c` design.

#### BLOCK (accumulation) probe variant: second run
To rule out the result being an artifact of measuring only the *last* sub-vector,
`_explain_loss_block` was added: it reconstructs the step text from the **accumulated
block** of sub-vectors (B,j,dim), the Option-B-style measurement. The probe now prints
SINGLE and BLOCK side by side (10 batches, 3060).

```
                subvec  0      1      2      3
SINGLE (mean 10b):     ~0.51 -> 1.20 -> 1.34 -> 1.37   (rises)
BLOCK  (mean 10b):     ~0.50 -> 1.07 -> 1.16 -> 1.37   (rises; ~ SINGLE)
```

BLOCK ~ SINGLE: the decoder (trained with 1 latent) does **not** exploit the extra
vectors of the OOD block. Confirms the pretrained model gives no advantage under any
measurement.

**Key conclusion (validated with the user):** since the checkpoint is pinned to
**c=1**, any `c != 1` is OOD and the probe on the pretrained model is at most a weak
hint; **retraining is mandatory** whether A or B is chosen. The probe cannot predict
whether it will work; only training can.

**Chosen direction:** **Option B (block-of-`c`, faithful to SIM-CoT) with retraining.**
The user prefers the B rebuild and accepts the retraining cost. Honest risk to watch:
even after retraining, the first vector already reconstructs each step well
(subvec0 ~0.5), so there may be little *headroom* and the model could learn c=1 for
everything; the ponder penalty and the acc-vs-budget evaluation will measure it. This
was re-planned outside the original Phase 2/3 scaffold (see the Option B plan below).

### Phases 2 + 3: MLP head + L_dist + penalty (full Option B objective)
**Date:** 2026-06-19 | GPU: RTX 3060

**Chosen design (self-contained rebuild).** Instead of surgery on `forward()`,
`_forward_option_b()` was added and is entered early (`if self.option_b: return
_forward_option_b(...)`). The inherited SIM-CoT path is **never** executed with
`--option_b`. Nested loop: K steps x M sub-vectors; at each `(k,j)` the decoder
reconstructs the step text from the **accumulated block** (`_block_step_loss`,
per-example). The `ob_mlp` (2xLinear+ReLU) predicts per-example `L_step` from `h_k`.
Answer decoded once after the whole block (the `c` axis focus, no per-prefix answer
loss; that is the K axis).

Objective: `L = lambda_ans*CE + distill + ref_ce + lambda_step*L_step +
lambda_dist*(L_dist + lambda_halt*sum sigmoid(-L_hat))`.
New flags: `ob_num_steps` (K). MLP in float32, `h_k` detached by default
(`ob_detach_hk`). `train.py`: warm-start allows `ob_mlp.*` newly-init; the fast-LR
group (`HALT_HEAD_LR`) now covers `ob_mlp` in addition to `halt_head`.

**Smoke (overfit 8 examples, 3060):**
- ce 0.22->0.003, ref_ce 0.32->0.03, l_step 1.57->~0, so architecture and block OK.
- **Bug found:** with MSE, `L_dist` did not converge (heavy/non-stationary CE target,
  so the MLP collapses to the mean, spikes of 12.4). **Fix:** SmoothL1 (Huber) for
  `L_dist` (like CODI's distill). Spikes eliminated.
- **Second issue:** the MLP (a critical head at inference) learned slowly in the
  base-LR group. **Fix:** add `ob_mlp` to the fast-LR group. With `HALT_HEAD_LR=2e-3`:
  `L_dist` 0.42->0.03, `L_hat` tracks `L_step`, `halt_pen` rises 0.27->0.45
  (sigmoid(-L_hat) registers maturity). Penalty-on (lambda_halt=0.05) stable.

**Note (pending rethink):** the overfit collapses all `L_step` to ~0, so it does NOT
test whether there is *headroom* (real variation in how many vectors each step needs).
That only shows with real training. Next: train short and measure whether at inference
the vectors/step VARY across instances while keeping accuracy. If it comes out flat
(everything matures at j=1), there is nothing to adapt: rethink/report (the user asked
to rethink if there are no results).

### Phase 4: Adaptive inference
**Date:** 2026-06-19

`test.py`: `option_b` branch (before pondernet). For each of the K fixed steps, add
sub-vectors (re-feeding the hidden) until `|L_hat_j - L_hat_{j-1}| < ob_eps` or
`ob_max_subvectors` is reached; then move to the next step; decode the answer after
the K steps. Latent generation mirrors training (the current sub-vector is added to
the cache when the next one is generated; the last is never added, the same
off-by-one). Faithful at `batch_size=1` (with bs>1 the shared cache inflates the
compute of rows that already stopped, the same caveat as PonderNet). Reports:
accuracy, avg total vectors/instance, mean vectors per step position, `n_k`
distribution, and an accuracy-vs-budget table. Saves per-instance detail in
`results/`. Script: `scripts/eval_gpt2_gsm8k_optionb.sh` (bs=1, 3060). `__init__` now
exposes `ob_eps`, `ob_max_subvectors`.

### Phase 5: Training + evaluation

**run1** (tmux `optionb-train`): K=4, M=3, BS=8xACCUM4, 3 epochs, 8000 examples,
LR 2e-5, HALT_LR 1e-3, **lambda_halt=0**. 3060, ~36 min. Loss 3.40->0.36. ckpt:
`models/checkpoints/optionb-run1/default/gpt2/ep_3/lr_2e-05/seed_42/checkpoint-747`.

**Eval bug:** `model.to(bf16)` casts `ob_mlp` to bf16 but we passed it `.float()`,
a dtype mismatch. Fix: cast the input to the MLP's dtype (`ob_mlp[0].weight.dtype`).

**Sweep results (300-subset, 3060):**

| config            | avg vecs | acc(%) |
|-------------------|----------|--------|
| fixed c=3 (eps=0) | 12.00    | 41.00  |
| adapt eps=0.02    | 10.69    | 41.00  |
| adapt eps=0.05    | 10.18    | 41.33  |
| adapt eps=0.15    | 9.23     | 41.00  |
| adapt eps=0.40    | 8.43     | 40.67  |
| **random**        | **8.01** | **41.00** |

Vectors-per-step pattern (adapt eps=0.05): s0=3.00, s1=2.66, s2=2.30, s3=2.22; early
steps use more vectors, later ones stop sooner. But it is mostly a **positional**
pattern (step 0 = more), not per-instance.

**Conclusion (FINDING):**
1. The Option B mechanism works: it trains, halts adaptively, trims compute
   12 -> ~9 vectors **with no accuracy loss** (~41% vs the ~39.5% SIM-CoT baseline).
2. **Accuracy is FLAT (~41%) across the whole 8-12 vector range, and random at a LOWER
   budget (8.0) matches adaptive.** The learned halting does NOT beat random at matched
   budget, so **the `c` axis has little exploitable headroom** on GSM8K-Aug with GPT-2.
   Each step is already nearly saturated with 1 vector (the Phase 1 probe anticipated
   this). The 40.67-41.33% differences are within noise (300 examples, SE ~2.8%).

This is exactly the "rethink if there are no results" case the user asked for: the
machinery is correct but the task does not reward `c` adaptivity.

**Full-set confirmation (GSM8K test, 1319 examples, bs=1):**

| config            | avg vecs | acc(%) |
|-------------------|----------|--------|
| fixed c=3 (eps=0) | 12.00    | 39.88  |
| adaptive eps=0.05 | 10.12    | 39.80  |
| random            | 8.05     | 39.35  |

Spread 0.53% (SE ~1.35%), flat. Adaptive = fixed with 16% less compute; random = both
with 33% less.

**Does the MLP give more vectors to hard problems? NO.** Mean vectors:
correct=10.28, incorrect=10.01 (the reverse of useful). Positional pattern (s0=3
always, later steps stop), not difficulty-driven. The learned halting captures no
per-instance signal.

**Fixed-`c` accuracy curve (full set):** c=1(4v)=27.52%, c=2(8v)=39.42%,
c=3(12v)=39.88%. Steep 1->2, saturates at c=2. So `c` matters but its required value
is ~constant (2); low per-instance variance = no adaptive headroom. Optimum: fixed c=2.

**run2 (lambda_halt=0.05, full set):** fixed c=2 = 39.65% (8v); adaptive eps0.15 =
39.95% (9.2v); adaptive eps0.40 = 39.50% (8.4v); random = 39.20% (8v). The penalty did
NOT unlock c=1 (eps0.40 -> almost all 2s, zero 1s). Fixed c=2 matches/beats everything.

**FINAL CONCLUSION.** Option B: correct, validated implementation (trains, halts,
trims compute 16-33% without losing accuracy). But **no headroom** for smart `c`
adaptivity on GSM8K-Aug/GPT-2: the required `c` saturates uniformly at 2; adaptive ~
random ~ fixed c=2 (all 39.2-40.0%, within noise); the MLP does not detect per-instance
difficulty; the penalty changes nothing. Recommendation: **fix c=2** and concentrate
the adaptivity effort on the `K` axis (Proposal C). Full summary in `RESULTS.md`.
Investigation closed (2 training runs + full-set sweeps).

### Phase 6: Unbiased retrain (cold + coarse) + diagnosis (2026-06-21)

To rule out the c=2 saturation being warm-start bias, retrained **cold from plain
GPT-2 + coarse segmentation** (`get_steps_coarse`, K=3 M=3, 30 ep, `train15k`).
- **Divergence at LR 3e-3** (the CODI recipe): blew up ep7->8 (loss 2.8->20). Fix:
  **LR 1e-3 + grad_norm 0.5 + warmup 0.05** -> clean descent 7.5->0.77, no spikes.
  `checkpoint-7020`.
- **Result: collapsed to ~chance.** Full-set c-curve: c1=5.31, c2=5.46, c3=5.53%.
  Adaptive/random (sub300) all ~8%, adaptive ~ random.

**Diagnosis (3 checks):**
1. **Not a loading bug**: keys identical to the warm ckpt that scores 39% (404 keys,
   zero diff); LoRA/prj/ob_mlp have trained weights.
2. **Learned format, not reasoning**: generates well-formed "The answer is: N" but with
   wrong arithmetic (3x3x60=540 -> says 360).
3. **Inert latents**: `fixed_k_eval` flat k1..k9 (~6%); adding vectors does not help.
   Contrast: warm DOES use the latents (c-curve rises 27.5->39.4).

**Failure mode:** a shortcut; the decoder imitates the teacher's hidden and emits a
plausible number from the question alone, without the latents becoming load-bearing.
CODI avoids this because its warm-start already brings functional latents (40 ep cold
with a simple objective, c=1).

**Verdict:** the c=1 anchor was **load-bearing, not just bias**. The `c` axis negative
is **robust in both regimes** cold/warm-atomic. Root cause = the task (trivial steps),
not the init. **Next lever to explore: per-step density of the dataset.**
Full detail in `RESULTS.md`, section "Unbiased retrain".

### Phase 7: Warm + coarse: the negative was granularity-specific (2026-06-21)

The missing cell of the 2x2 (init x granularity): **warm-start + coarse segmentation**
(K=3 buckets of 2-3 ops). Keeps the functional model, adds variable per-step density.
Script `train_gpt2_gsm8k_optionb_warmcoarse.sh` (LR 2e-5, 3 ep, penalty off;
loss 1.8->0.33).

**Result: the `c` axis negative was specific to the atomic granularity:**
- Full-test c-curve: c1=25.5, c2=36.6, **c3=40.3** -> delta c2->c3 = **+3.7** (vs +0.5
  atomic). The 3rd vector contributes again; `c` stops saturating at 2.
- **Adaptive vs random (full test, matched budget):** adaptive eps0.15 = 38.8% @ 7.2
  vec; random = 35.2% @ 6.1 vec -> **+3.6 pts, about 2.7 sigma**. In atomic, adaptive ~
  random; here NOT. The per-instance signal exists when density varies.
- Over the uniform line (interpolating the fixed c-curve): adaptive wins +0.6 to +1.0,
  consistent (3/3 eps, replicated on sub300) but <1 sigma, so real but **modest**
  headroom.

**Project verdict:** the negative was granularity-specific, not Option B's. Grouping
trivial ops already revives per-instance signal (adaptive>random at 2.7 sigma). Next:
a **synthetic controlled-density dataset** to amplify the adaptive-vs-uniform gap and
measure headroom vs density variance. Detail in `RESULTS.md`, section "Warm + coarse".

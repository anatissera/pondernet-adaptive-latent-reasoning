# Option B: Implementation of non-fixed `c` (adaptive vectors per step)

Reference document to **trace** the implementation of `c` adaptivity and compare
results if something changes. All the code lives in `adaptive-vectors/` and is gated by
`--option_b` (off by default, so the inherited SIM-CoT path is intact).

> **Glossary.** `K` = number of reasoning steps (fixed). `M` = maximum sub-vectors per
> step (`ob_subvectors_per_step` in train; `ob_max_subvectors` at inference).
> `c` = number of sub-vectors actually used in a step (n_k). `L_step` = SIM-CoT's
> auxiliary decoder loss for reconstructing the step text. `L_hat` = the MLP's
> prediction of `L_step`. `h_k` = the backbone's last hidden state at the latent
> position.

---

## 1. The idea in one sentence

Each reasoning step is built as a **block of up to M sub-vectors** generated one at a
time (re-feeding the hidden state). SIM-CoT's decoder reconstructs the step text from
the **accumulated block**, giving `L_step`. An MLP **distills** `L_step` from `h_k` and
survives inference (the decoder does not). At inference, sub-vectors are added to a
step until `L_hat` stops going down, meaning the step is "mature", and the model moves
to the next step.

---

## 2. Training: `CODI._forward_option_b` (`src/model.py`)

Self-contained method; entered through the hook at the start of `forward()`:
`if self.option_b: return self._forward_option_b(...)`.

**Flow:**
1. **Encode** the question -> `past_key_values`, `latent_hidden` = last hidden
   (post-`prj`). This first hidden is sub-vector 0 of step 0.
2. **Per-step texts:** `get_steps(ref_input_ids, K, ...)` segments the CoT into K steps
   using the `<<` markers (start_ids=(16791,9959)) and `>>` (end_id=4211). `pad_steps`
   aligns them. *(Each GSM8K-Aug `<<op>>` = one step; with >K ops they merge into the
   last one; with <K they are padded. See section 6 on the granularity bias.)*
3. **Teacher pass** (ref) for `distill_loss` + `ref_ce` (inherited from CODI, kept).
4. **Nested loop** `for k in range(K): for j in range(M):`
   - If not the first sub-vector: `codi(inputs_embeds=latent_hidden,
     past_key_values=...)` adds the previous sub-vector to the cache and produces the
     new `latent_hidden` (post-`prj`).
   - `block.append(latent_hidden)`; `latent_block = cat(block)` (B, j+1, dim).
   - `per_ex, valid = _block_step_loss(latent_block, steps_pad_list, k)` -> per-example
     `L_step` reconstructing step `k`'s text from the **accumulated block** (decoder).
   - `L_hat = ob_mlp(h_k)` (h_k detached if `ob_detach_hk`).
   - Accumulate: `l_step_sum += per_ex[valid].mean()`; `dist_terms +=
     SmoothL1(L_hat, sg(L_step))`; `halt_terms += sigmoid(-L_hat)`.
5. **Answer:** decoded **once** after the full block (K*M sub-vectors) -> `ce_loss`
   (answer CE) + `distill_loss` (student hidden vs teacher at the answer position, all
   layers). Warning: **the answer is only trained at the MAXIMUM budget (M per step).**
   See section 6 (methodological bias).

**Objective:**
```
L = ob_lambda_ans*ce_loss + distill_loss + ref_ce
    + ob_lambda_step*L_step
    + ob_lambda_dist*( L_dist + ob_lambda_halt*sum_{k,j} sigmoid(-L_hat) )
L_step = mean_{k,j} per_example_CE(reconstruction of step k from a j-vector block)
L_dist = mean_{k,j} SmoothL1( L_hat_{k,j} , stopgrad(L_step_{k,j}) )
```
- **SmoothL1 (not MSE)** for `L_dist`: `L_step` is a heavy-tailed CE with a
  non-stationary target; MSE collapsed to the mean / blew up (bug found in the smoke
  test; see AGENTS.md Phase 2+3).
- `ob_mlp` = `Linear(dim,256)->ReLU->Linear(256,1)`, float32, last layer init to 0 /
  bias 1.0.
- The MLP regresses to `L_step` **per-example** (not batch-mean) so it can discriminate
  instances.

## 3. Adaptive inference (`test.py`, `option_b` branch)

For each of the K steps: add sub-vectors (re-feeding, mirroring training) and evaluate
`L_hat` after each one. **Stop the step** when `|L_hat_j - L_hat_{j-1}| < ob_eps` or
`ob_max_subvectors` is reached; move to the next step. After the K steps, decode the
answer from the full latent prefix. Reports accuracy, total vectors/instance, mean per
step position, `n_k` distribution, and an accuracy-vs-budget table. **Faithful at
`batch_size=1`** (with bs>1 the shared cache inflates the compute of rows that already
stopped, the same caveat as PonderNet).

`--ob_random` baseline: stops each step at `n_k ~ Uniform[1, M_max]` (a control at a
similar budget).

## 4. Flags (all in `TrainingArguments`, `src/model.py`)

| flag | default | what it does |
|------|---------|--------------|
| `option_b` | False | enables the Option B path |
| `ob_num_steps` (K) | 4 | number of reasoning steps |
| `ob_subvectors_per_step` (M, train) | 4 | fixed sub-vectors per step in training |
| `ob_max_subvectors` (M_max, infer) | 4 | cap on sub-vectors per step at inference |
| `ob_eps` | 0.01 | "stopped going down" threshold to stop a step |
| `ob_mlp_hidden` | 256 | MLP width |
| `ob_detach_hk` | True | stop-grad of h_k toward the MLP (does not corrupt the backbone) |
| `ob_lambda_ans/step/dist/halt` | 1/1/1/0.01 | objective weights |
| `ob_probe` | False | Phase-1 diagnostic (L_step curve per sub-vector) |
| `ob_random` | False | random halting baseline (eval) |
| `ob_coarse_steps` | False | **coarse** segmentation: groups the ops into `ob_num_steps` even buckets (via `get_steps_coarse`) instead of 1-op-per-step, giving variable per-step complexity for the unbiased c-axis test. Training/probe only; inference is unaffected. |

## 5. Scripts and artifacts (where to look)

- `scripts/train_gpt2_gsm8k_optionb.sh`: training (3060). Env:
  `K,M,BS,ACCUM,EPOCHS,MAXSAMPLES,LR,LAMBDA_HALT,HALT_HEAD_LR`.
- `scripts/eval_gpt2_gsm8k_optionb.sh`: eval (bs=1, 3060). Env: `K,MMAX,EPS`,
  `--ob_random`.
- `scripts/ob_eval_sweep.sh` / `ob_smoke.sh` / `ob_probe.sh`: sweep / smoke / probe.
- Checkpoints:
  `models/checkpoints/optionb-run{1,2}/default/gpt2/ep_3/lr_2e-05/seed_42/checkpoint-747`.
- Logs/results: `outputs/optionb-*`, `results/optionb-*`.
- Results + conclusions: `RESULTS.md`. Per-phase log: `AGENTS.md`.

## 6. Decisions that can bias the result (READ before comparing)

Aspects of the current implementation/setup that could push toward "c saturates at 2"
and that an **unbiased** test should change:

1. **Warm-start from a c=1 checkpoint + LoRA-only (frozen base).** In `option_b` there
   is no freezing block (only `pondernet` has one), so: GPT-2 base **frozen**, only
   LoRA(r=128) adapts it; decoder/prj/ob_mlp trainable. The model starts in the c=1
   regime and moves little. *Unbiased test:* full fine-tune (or cold start without
   warm-start), more data/epochs.
2. **Step granularity = 1 arithmetic operation per step.** `get_steps` splits on
   `<<...>>`. Each GSM8K-Aug step is a trivial `<<a op b=c>>`, so 2 vectors saturate
   it. The dataset's variation is in the **number of steps** (1-6 ops), not per-step
   complexity. *Unbiased test:* **coarser** segmentation (smaller K, each step grouping
   several ops for variable complexity), or a dataset with variable per-step difficulty.
3. **The answer is trained only at maximum budget (M per step).** The answer head never
   learns to answer with fewer vectors, biasing against low-c configs and against
   adaptivity paying off. *Unbiased test:* supervise the answer at variable budgets
   (PonderNet-style on the c axis), not only at M.
4. **`L_step` = reconstruction of the step TEXT, not "answer-readiness".** Maturity is
   measured by text reconstruction (short/trivial in GSM8K-Aug), which saturates
   regardless of whether the answer needs more compute. `L_step` may not be the right
   signal for saving *answer* compute.

The "unbiased retrain" plan attacks (1)-(3); (4) is conceptual and discussed in the
plan.

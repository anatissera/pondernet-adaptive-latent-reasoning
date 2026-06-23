# 03: Gradient-checkpointing fix

**Status:** complete   **Dates:** 2026-06-12 → 2026-06-12

> ✅ **Re-validated 2026-06-23.** Surviving checkpoints (ep4/ep5) re-evaluated on the held-out
> validation split (500 ex, greedy); prior test-set numbers were optimistically biased. The
> headline ep2 = 42.23% checkpoint was **deleted and cannot be re-validated**. See
> [eval-split note](../../experiments.md#eval-split-and-leakage-note).

## What's being tested
The single change that turned PonderNet from "half the baseline" into "beats the baseline":
disabling gradient checkpointing so the latent loop's KV cache works during training.
One keeper run (`100k`) trained on 100k examples with the fix.

## Setup
- Method: PonderNet adaptive halting on SIM-CoT, warm-started from the SIM-CoT CODI checkpoint.
- ep=5, lr=2e-5, eff. batch=128 (`per_device 32` × `accum 4`), K_max=6, seed=42.
- The one change vs [02](../02-simcot-pondernet-early/experiment.md): `--gradient_checkpointing False`.
- Data: `train100k.jsonl` (`--max_train_samples 100000`). Eval: re-validated on the validation
  split (500 ex, greedy, 1 pass) at thr 0.5/0.8/0.9; prior numbers were on GSM8K test (biased).

## Findings
- **~41% on validation** for the surviving epochs (ep4 41.00%, ep5 41.20% @ thr0.5; n=500,
  greedy) — essentially on par with the 40.80% greedy validation baseline, confirming the fix
  produces a clean training objective rather than a collapsed one. The fix's qualitative effect
  (latent loop's KV cache works during training → loss no longer flat at ~2.5) is the durable
  result; the accuracy gain over baseline is small/within-noise on validation.
- The originally-reported **42.23%** @ ep2 (`checkpoint-1556`) was on the **test** set
  (biased; data leakage) and its checkpoint has been **deleted, so it cannot be re-validated**.
  Treat it as unreconcilable, not as a validated win.
- Adaptivity is emerging but mild (avg ~5.9/6 steps at thr0.8); difficulty tracking on
  validation is positive but weak: Spearman(n_expr, steps) = +0.456 @ thr0.5. Pushing real
  adaptivity is what [04](../04-simcot-pondernet-gammasweep/experiment.md) does.

## Root Cause: why every pre-`gcfix` trained run failed

**All of our own trained runs ([02](../02-simcot-pondernet-early/experiment.md):
`lr1e4` 19.26%, `warmstart-lr1e4` 15.24%; [01](../01-simcot-baselines/experiment.md):
`fixedk-k6-lr1e4` 17.82%) collapsed to ~15–19% — roughly half the 39.5% baseline — because
of a single bug: `--gradient_checkpointing True` silently disabled the latent loop's KV cache
during training.**

The latent reasoning loop is built on `self.codi(inputs_embeds=latent_hidden, use_cache=True,
past_key_values=...)` — each latent step and the per-step answer decode see the question +
accumulated latent prefix *only* through the KV cache. Gradient checkpointing is incompatible
with `use_cache=True`, so HF forces `use_cache=False` (it prints `"use_cache=True is
incompatible with gradient checkpointing. Setting use_cache=False"`). With the cache off,
`past_key_values` becomes `None`: every latent step runs context-free and the answer is
decoded with no reasoning prefix. The model trains against a broken objective and never learns.

Evidence (CE_PROBE, same warm-started batch, toggling `gradient_checkpointing_enable()`):

| | answer CE (`l_pondernet`) |
|---|---|
| grad-ckpt **OFF** | **0.243** |
| grad-ckpt **ON** | **2.517** |

The loss is ~2.5 and flat **from step 0** regardless of LR / data size / batch / attention-mask
/ dropout / γ — all of which were ruled out before finding the cache bug. The bug is invisible
at eval: `test.py` uses `--batch_size 1` and no gradient checkpointing, so `use_cache=True`
works there — but the weights were trained blind, hence ~19%.

**Fix:** `--gradient_checkpointing False` (the two are fundamentally incompatible with this
cache-based latent loop). Costs VRAM, so batch was re-profiled: `per_device 16` underfills the
GPU (~25% util), `32` ≈ 55% util (~2.6 h / 5 ep on the 3090), `64` OOMs. Eval must run on the
**idle 3060** (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`) — sharing the 3090 with
training OOMs it (which is what crashed `100k` mid-run; it was resumed from the epoch-2 checkpoint).

See [runs.md](runs.md) for the run table · artifacts under `<dir>/03-simcot-pondernet-gcfix/`.

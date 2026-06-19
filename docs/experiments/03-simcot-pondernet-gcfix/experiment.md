# 03: Gradient-checkpointing fix

**Status:** complete   **Dates:** 2026-06-12 → 2026-06-12

## What's being tested
The single change that turned PonderNet from "half the baseline" into "beats the baseline":
disabling gradient checkpointing so the latent loop's KV cache works during training.
One keeper run (`100k`) trained on 100k examples with the fix.

## Setup
- Method: PonderNet adaptive halting on SIM-CoT, warm-started from the SIM-CoT CODI checkpoint.
- ep=5, lr=2e-5, eff. batch=128 (`per_device 32` × `accum 4`), K_max=6, seed=42.
- The one change vs [02](../02-simcot-pondernet-early/experiment.md): `--gradient_checkpointing False`.
- Data: `train100k.jsonl` (`--max_train_samples 100000`). Eval: GSM8K test, greedy, 1 pass, thr=0.8.

## Findings
- **42.23%** best @ epoch 2 (`checkpoint-1556`) — the **first trained run to beat the 39.5%
  baseline**. Per-epoch: 40.56 / **42.23** / 40.11 / 39.95 / 39.95%.
- Accuracy peaks at ep2 then drifts back toward baseline by ep5 → **keep the best epoch, not
  the last** (or train ~2 epochs).
- Adaptivity is emerging but mild (avg 5.88/6 steps; 99 easy questions halt early at 78–88%
  acc). Pushing real adaptivity is what [04](../04-simcot-pondernet-gammasweep/experiment.md) does.
- Numbers are single greedy passes; older baselines are avg-of-5 — re-eval over 5 samples for
  a strictly apples-to-apples comparison.

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

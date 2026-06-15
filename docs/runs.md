# Run Manifest

**Migrated to the run-id layout on the valen/repo-cleanup branch.**

## Run-ID Convention

Run IDs follow the pattern `<base-model>-<method>[-<variant>]-<hparams>`:

- `simcot` = SIM-CoT (ICLR 2026) backbone (GPT-2 + CODI LoRA encoder + separate decoder)
- `baseline` = fixed-K decoding with no PonderNet halt head, used as reference
- `fixedk` = fixed latent-step count (no adaptive halting)
- `pondernet` = PonderNet-style adaptive halting head added on top of SIM-CoT decoder
- `joint` = joint training of halt head + backbone together (as opposed to halt-head-only fine-tuning)
- `warmstart` = decoder initialized from pre-trained SIM-CoT weights before PonderNet training
- `halthead` = halt-head-only fine-tuning (backbone frozen)

## Artifact Layout

Each run's artifacts are mirrored across three top-level directories:

| Directory | Contents |
|-----------|----------|
| `models/checkpoints/<run-id>/` | Final `pytorch_model.bin` + tokenizer files; rescued from outputs after training |
| `outputs/<run-id>/` | `train.log`, TensorBoard events, intermediate resume checkpoints |
| `results/<run-id>/` | `eval.log`, `gsm8k.json` (gold answers), `gsm8k_pondernet_detail.json` (per-instance steps + correct flag); threshold sub-variants appear as `results/<run-id>/thr<value>/` |

Rescued checkpoints:
- `models/checkpoints/simcot-pondernet-lr1e4/` — best PonderNet run (19.26%), rescued from `outputs/pondernet/`
- `models/checkpoints/simcot-pondernet-warmstart-lr1e4/` — warm-start experiment (15.24%), rescued from `outputs/simcot_warmstart_lr1e4/`

---

## Run Table

| run-id | date | method | key hparams | GSM8K accuracy | checkpoint? | notes |
|--------|------|--------|-------------|---------------|-------------|-------|
| `simcot-baseline-k6` | 2026-06-08 | SIM-CoT fixed-K (reference) | K=6, lr=1e-3 (upstream default) | **39.50%** (avg 5 samples) | no | Upstream SIM-CoT eval of the CODI checkpoint; results at `results/simcot-baseline-k6/` |
| `simcot-fixedk-k6-lr1e4` | 2026-06-08 | SIM-CoT fixed-K retrain | K=6, lr=1e-4 | **17.82%** (avg 5 samples) | no | Lower LR re-run of fixed-K; results at `results/simcot-fixedk-k6-lr1e4/` |
| `simcot-pondernet-halthead-ep40` | 2026-06-08 | PonderNet halt-head only (backbone frozen) | ep=40, lr=? (see trainer_state.json) | not evaluated | YES — `models/checkpoints/simcot-pondernet-halthead-ep40/` | Halt head trained on top of the frozen backbone; no eval results directory found; accuracy not evaluated |
| `simcot-pondernet-lr1e4` (thr=0.8) | 2026-06-09 | PonderNet adaptive halting | ep=40, lr=1e-4, halt_thr=0.8, seed=42 | **19.26%** (avg 5 samples) | YES — `models/checkpoints/simcot-pondernet-lr1e4/` | Best PonderNet result; results at `results/simcot-pondernet-lr1e4/thr0.8/` |
| `simcot-pondernet-lr1e4` (thr=0.9) | 2026-06-09 | PonderNet adaptive halting | ep=40, lr=1e-4, halt_thr=0.9, seed=42 | **17.82%** (avg 5 samples) | (same checkpoint as thr=0.8) | Sub-variant of same run evaluated at higher halt threshold; results at `results/simcot-pondernet-lr1e4/thr0.9/` |
| `simcot-pondernet-joint-ep40` | 2026-06-08 | PonderNet joint training (halt head + backbone) | ep=40, lr=3e-3, seed=42 | ~0% (eval aborted after 24/1319 Qs, all Prediction=inf) | YES — `models/checkpoints/simcot-pondernet-joint-ep40/` | ⚠ KNOWN-BAD: the CODI checkpoint was loaded as bare GPT-2 (random backbone); results at `results/simcot-pondernet-joint-ep40/` (truncated log only) |
| `simcot-pondernet-warmstart-lr1e4` | 2026-06-09 | PonderNet warm-started from SIM-CoT | ep=40, lr=1e-4, seed=42; warm-started 400 tensors | **15.24%** (avg 5 samples) | YES — `models/checkpoints/simcot-pondernet-warmstart-lr1e4/` | Warm-start experiment; lower than thr=0.8 pondernet; results at `results/simcot-pondernet-warmstart-lr1e4/` |
| `simcot-pondernet-joint-warmstart` | 2026-06-09 | PonderNet joint warm-started (crashed) | ep=40, lr=? | N/A — crashed at epoch 0.69 | NONE | ⚠ FAILED: RuntimeError `masked_scatter_: expected self and source to have same dtypes (BFloat16 vs Float)`; TB events + `train.log` deleted (approved cleanup) |
| `simcot-pondernet-gcfix-100k` (thr=0.8) | 2026-06-12 | PonderNet adaptive halting — **gradient-checkpointing fix** | ep=5, lr=2e-5, eff batch=128 (bs32×accum4), `--gradient_checkpointing False`, warm-started, seed=42 | **42.23%** best @ ep2 / 39.95% final @ ep5 (greedy, 1 pass) | YES — `models/checkpoints/simcot-pondernet-gcfix-100k/` (ep2 = `checkpoint-1556`, final = `checkpoint-3890`) | ✅ **First trained run to beat the 39.5% baseline.** Fixes the gradient-checkpointing bug that crippled every earlier run (see Root Cause below). Per-epoch: 40.56 / **42.23** / 40.11 / 39.95 / 39.95%. Adaptivity emerging (avg 5.88/6 steps, 99 easy Qs halt early at 78–88% acc). OOM-crashed mid-run when an eval shared the 3090; resumed cleanly from ep2 via `pondernet/scripts/_resume_gcfix.sh`. Results at `results/simcot-pondernet-gcfix-100k/ep{1..5}-thr0.8/` |

---

## Accuracy Summary

| run-id | accuracy (GSM8K test, avg 5 samples) | source |
|--------|--------------------------------------|--------|
| `simcot-pondernet-gcfix-100k` (ep2, thr=0.8) | **42.23%** (greedy, 1 pass; best) | `results/simcot-pondernet-gcfix-100k/ep2-thr0.8/` |
| `simcot-pondernet-gcfix-100k` (ep5/final, thr=0.8) | 39.95% (greedy, 1 pass) | `results/simcot-pondernet-gcfix-100k/ep5-thr0.8/` |
| `simcot-baseline-k6` | 39.50% | `results/simcot-baseline-k6/eval.log` |
| `simcot-pondernet-lr1e4` (thr=0.8) | 19.26% | `results/simcot-pondernet-lr1e4/thr0.8/eval.log` + confirmed via `gsm8k_pondernet_detail.json` (254/1319) |
| `simcot-pondernet-lr1e4` (thr=0.9) | 17.82% | `results/simcot-pondernet-lr1e4/thr0.9/eval.log` + confirmed via detail JSON (235/1319) |
| `simcot-fixedk-k6-lr1e4` | 17.82% | `results/simcot-fixedk-k6-lr1e4/eval.log` |
| `simcot-pondernet-warmstart-lr1e4` | 15.24% | `results/simcot-pondernet-warmstart-lr1e4/eval.log` |
| `simcot-pondernet-joint-ep40` | ? | eval log truncated (24/1319 Qs, all inf predictions, random backbone) |
| `simcot-pondernet-halthead-ep40` | ? | no eval results directory found |
| `simcot-pondernet-joint-warmstart` | N/A | training crashed before completion |

---

## Root Cause: why every pre-`gcfix` trained run failed (2026-06-12)

**All of our own trained runs (`fixedk-k6-lr1e4` 17.82%, `pondernet-lr1e4` 19.26%, `warmstart-lr1e4` 15.24%, etc.) collapsed to ~15–19% — roughly half the 39.5% baseline — because of a single bug: `--gradient_checkpointing True` silently disabled the latent loop's KV cache during training.**

The latent reasoning loop is built on `self.codi(inputs_embeds=latent_hidden, use_cache=True, past_key_values=...)` — each latent step and the per-step answer decode see the question + accumulated latent prefix *only* through the KV cache. Gradient checkpointing is incompatible with `use_cache=True`, so HF forces `use_cache=False` (it prints `"use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False"`). With the cache off, `past_key_values` becomes `None`: every latent step runs context-free and the answer is decoded with no reasoning prefix. The model trains against a broken objective and never learns.

Evidence (CE_PROBE, same warm-started batch, toggling `gradient_checkpointing_enable()`):

| | answer CE (`l_pondernet`) |
|---|---|
| grad-ckpt **OFF** | **0.243** |
| grad-ckpt **ON** | **2.517** |

The loss is ~2.5 and flat **from step 0** regardless of LR / data size / batch / attention-mask / dropout / γ — all of which were ruled out before finding the cache bug. The bug is invisible at eval: `test.py` uses `--batch_size 1` and no gradient checkpointing, so `use_cache=True` works there — but the weights were trained blind, hence ~19%.

**Fix:** `--gradient_checkpointing False` (the two are fundamentally incompatible with this cache-based latent loop). Costs VRAM, so batch was re-profiled: `per_device 16` underfills the GPU (~25% util), `32` ≈ 55% util (~2.6 h / 5 ep on the 3090), `64` OOMs. Eval must run on the **idle 3060** (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`) — sharing the 3090 with training OOMs it (which is what crashed `gcfix-100k` mid-run; it was resumed from the epoch-2 checkpoint).

## Next Steps

1. **Push real adaptivity (primary goal).** `gcfix-100k` recovers accuracy but barely reduces steps (avg 5.88/6) because `--pondernet_gamma 0.01` makes the KL-geometric halting pressure negligible (γ·KL ≈ 0.003 of the loss). Sweep `pondernet_gamma` upward (e.g. 0.1, 0.3) and/or tune `pondernet_geom_mean` to deliberately trade accuracy for fewer steps, then plot the accuracy-vs-steps curve.
2. **Discriminative learning rates.** `CustomTrainer.create_optimizer` now supports a hot halt-head LR via the `HALT_HEAD_LR` env var (gentle backbone, fast-learning halt head) — untested; use it so the halt head learns to halt earlier without disturbing the warm-started backbone.
3. **Use the best checkpoint, not the last.** Accuracy peaked at epoch 2 (42.23%) then drifted to baseline by epoch 5 at lr 2e-5 — keep/eval the best epoch, or shorten training to ~2 epochs.
4. **Multi-sample eval.** `gcfix-100k` numbers are single greedy passes; the older baselines are avg-of-5 — re-eval the best checkpoint over 5 samples for an apples-to-apples comparison.
5. **(stale) `simcot-pondernet-halthead-ep40`** was never evaluated; it predates the gcfix fix and was trained with the broken cache, so it is not worth evaluating — supersede with a fresh halt-head experiment if needed.

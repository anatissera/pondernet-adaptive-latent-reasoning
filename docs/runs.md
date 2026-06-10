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

---

## Accuracy Summary

| run-id | accuracy (GSM8K test, avg 5 samples) | source |
|--------|--------------------------------------|--------|
| `simcot-baseline-k6` | 39.50% | `results/simcot-baseline-k6/eval.log` |
| `simcot-pondernet-lr1e4` (thr=0.8) | 19.26% | `results/simcot-pondernet-lr1e4/thr0.8/eval.log` + confirmed via `gsm8k_pondernet_detail.json` (254/1319) |
| `simcot-pondernet-lr1e4` (thr=0.9) | 17.82% | `results/simcot-pondernet-lr1e4/thr0.9/eval.log` + confirmed via detail JSON (235/1319) |
| `simcot-fixedk-k6-lr1e4` | 17.82% | `results/simcot-fixedk-k6-lr1e4/eval.log` |
| `simcot-pondernet-warmstart-lr1e4` | 15.24% | `results/simcot-pondernet-warmstart-lr1e4/eval.log` |
| `simcot-pondernet-joint-ep40` | ? | eval log truncated (24/1319 Qs, all inf predictions, random backbone) |
| `simcot-pondernet-halthead-ep40` | ? | no eval results directory found |
| `simcot-pondernet-joint-warmstart` | N/A | training crashed before completion |

---

## Open Items

1. **`simcot-pondernet-halthead-ep40`**: No `results/` directory found. The model at `models/checkpoints/simcot-pondernet-halthead-ep40/` completed 40 epochs (`trainer_state.json` confirms `epoch: 40.0`), but was never evaluated. Accuracy is not evaluated — run eval or leave as unknown.

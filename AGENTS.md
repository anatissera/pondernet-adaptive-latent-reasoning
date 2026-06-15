# adaptive-latent-reasoning

NLP final project (5-person team, 2 subgroups). Goal: make the number of latent reasoning steps **adaptive** in a latent CoT model, building on SIM-CoT (ICLR 2026).

## Repo Structure

- `baselines/` — upstream reference implementations; treat as read-only
  - `Coconut/` — Coconut latent CoT (GSM8K, ProntoQA tasks)
  - `CODI/` — CODI implicit CoT baseline (upstream reference)
- `pondernet/` — PonderNet adaptive halting built on CODI; Subgroup 1's active working area
- `models/` — model weights (gitignored), split by provenance:
  - `pretrained/` — downloaded backbones + decoder: `gpt2`, `simcot-gpt2-codi`, `simcot-gpt2-coconut`, `simcot-gpt2-decoder` (fetch the decoder with `pondernet/scripts/fetch_simcot_decoder.py`)
  - `checkpoints/<run-id>/` — our trained runs
- `outputs/` — training logs and TensorBoard events (gitignored); one subdir per run-id
- `results/` — evaluation outputs (gitignored); one subdir per run-id
- `data/` — datasets (gitignored); `gsm8k_aug/` holds the GSM8k-Aug jsonl, training is pinned to `train15k.jsonl` by default
- A single **run-id** names the same experiment across `models/checkpoints/`, `outputs/`, and `results/` (see `docs/runs.md`)
- `docs/pipeline.md` — end-to-end training/eval workflow + diagram; **read this first to train a model**
- `docs/parameters.md` — CLI flag reference, warm-start recipes, and the kept-name glossary
- `docs/runs.md` — run manifest (run-id, date, hparams, accuracy)
- `docs/methods-comparison.md` — cross-paper comparison table and chain-of-influence narrative; read this for method context
- `docs/papers/` — full paper content (raw); only read if you need deeper detail beyond the comparison doc

## Two Subgroups

| Branch | Strategy | Active Working Area |
|--------|----------|---------------------|
| `pondernet` | PonderNet-style adaptive halting (phases 1–6 complete) | `pondernet/` |

## Method: adaptive halting (PonderNet on SIM-CoT)

SIM-CoT runs a **fixed** number of latent reasoning steps K: each step feeds the
model's own hidden state `z_k` back as the next input embedding (bypassing the
vocab projection); after K steps it switches to text and decodes the answer. An
auxiliary decoder reconstructs each step's text during training (`L_step`) and is
discarded at inference. We make K **adaptive per instance** following PonderNet:

- **Halting head** (`halt_head`, `nn.Linear(dim, 1)`, bias init `-2.0`): maps each
  latent state `h_k` → conditional halting prob `lambda_k = sigmoid(.)`.
- The answer is decoded after **every** prefix `k = 1…K`, not only at K.
- Halting distribution `p_k = (∏_{j<k}(1 - lambda_j)) · lambda_k`, with an absorbing
  boundary at K (`lambda_K → 1`) so rows sum to 1 (`_halting_distribution`).
- Loss: `L = L_pondernet + β·L_step + γ·KL_geom`, where
  `L_pondernet = Σ_k p_k · L_ans^(k)` is the expected answer loss and `KL_geom`
  regularizes `p_k` toward a truncated geometric prior (`_kl_geom`; defaults
  `γ=0.01`, `geom_mean=3.0`). CODI's `distill_loss` + `ref_ce` are also kept.
- **Inference** (`test.py`): accumulate halting mass and stop the latent loop once
  cumulative halt prob crosses a threshold (default `0.5`); `K_max = max_latent_steps`
  is a hard cap.

## Gotchas (read before training / evaluating)

- **`--gradient_checkpointing` MUST stay `False`.** It is incompatible with the
  latent loop's KV cache (`use_cache=True` + `past_key_values`); HF silently forces
  `use_cache=False`, so every latent step runs context-free and the model trains
  against a broken objective. This bug capped every early run at ~15–19% (half the
  39.5% baseline); the fix first beat baseline at **42.23%**. See `docs/runs.md` →
  *Root Cause*. Costs VRAM — `per_device_train_batch_size 32` fits the 3090.
- **Run eval at `--batch_size 1` for faithful adaptive halting.** In `test.py` the
  latent loop only breaks when **all** examples in the batch have halted, and the
  answer is decoded from the batch-termination prefix — so with batch > 1 an example
  that halts early still gets its answer computed from *more* steps than `steps_used`
  reports. The avg-steps metric stays correct; per-example accuracy does not reflect
  compute-at-halt. Exact only at `batch_size = 1` (see `pondernet/test.py`).
- **Eval on the idle GPU**, not the one training (`CUDA_VISIBLE_DEVICES`), or the
  shared card OOMs mid-run.

## Running Baselines

```bash
# Coconut (from baselines/Coconut/)
python run.py args/gsm_coconut.yaml

# PonderNet — adaptive halting variant (from pondernet/)
# Trains on the pinned data/gsm8k_aug/train15k.jsonl by default.
SAVE_DIR=../models/checkpoints/<run-id> LOG_DIR=../outputs/<run-id> \
  bash scripts/train_gpt2_gsm8k_pondernet.sh
```

See `docs/pipeline.md` for the full train → eval → record workflow and `docs/parameters.md` for every flag.

## Setup

```bash
uv sync
```

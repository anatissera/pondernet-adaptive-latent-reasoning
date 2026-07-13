# adaptive-latent-reasoning

NLP final project (5-person team, 2 subgroups). Goal: make the number of latent reasoning steps **adaptive** in a latent CoT model, building on SIM-CoT (ICLR 2026).

## Repo Structure

- `baselines/` - upstream reference implementations; treat as read-only
  - `Coconut/` - Coconut latent CoT (GSM8K, ProntoQA tasks)
  - `CODI/` - CODI implicit CoT baseline (upstream reference)
- `pondernet/` - PonderNet adaptive halting built on CODI; Subgroup 1's active working area
- `models/` - model weights (gitignored), split by provenance:
  - `pretrained/` - downloaded backbones + decoder: `gpt2`, `simcot-gpt2-codi`, `simcot-gpt2-coconut`, `simcot-gpt2-decoder` (fetch the decoder with `pondernet/scripts/fetch_simcot_decoder.py`)
  - `checkpoints/<NN-exp>/<run-id>/` - our trained runs, grouped under a numbered experiment
- `outputs/` - training logs and TensorBoard events (gitignored); `<NN-exp>/<run-id>/` per run
- `results/` - evaluation outputs (gitignored); `<NN-exp>/<run-id>/` per run
- `data/` - datasets (gitignored), **except the eval split `data/gsm8k_aug/test.jsonl`, which is tracked in git** (1319 ex - the GSM8K test set; this experiment's headline metric). `gsm8k_aug/` holds the GSM8k-Aug jsonl; materialize training data with `pondernet/scripts/prep_gsm8k_aug.py`. There is no `validation.jsonl` for this experiment and none should be added: the 500-ex validation split used in exps 01–07 was sampled from `train.jsonl`, and this experiment trains on the *entire* train set, so those examples would be train-contaminated here - see `docs/experiments/10-simcot-pondernet-fromscratch/experiment.md` → "Evaluation protocol".
- Runs are grouped into numbered **experiments**: a `<NN-exp>/<run-id>` pair names the same run across `models/checkpoints/`, `outputs/`, and `results/` (see `docs/experiments.md`). Dead/scratch runs live under `<dir>/archive/`.
- `docs/pipeline.md` - end-to-end training/eval workflow + diagram; **read this first to train a model**
- `docs/parameters.md` - CLI flag reference, warm-start recipes, and the kept-name glossary
- `docs/experiments.md` - experiment index → per-experiment `experiment.md`/`runs.md` → per-run `<run-id>.md`
- `docs/methods-comparison.md` - cross-paper comparison table and chain-of-influence narrative; read this for method context
- `docs/papers/` - full paper content (raw); only read if you need deeper detail beyond the comparison doc

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
  39.5% baseline); the fix first beat baseline at **42.23%**. See
  `docs/experiments/03-simcot-pondernet-gcfix/experiment.md` → *Root Cause*. Costs
  VRAM - `per_device_train_batch_size 32` fits the 3090.
- **Run eval at `--batch_size 1` for faithful adaptive halting.** In `test.py` the
  latent loop only breaks when **all** examples in the batch have halted, and the
  answer is decoded from the batch-termination prefix - so with batch > 1 an example
  that halts early still gets its answer computed from *more* steps than `steps_used`
  reports. The avg-steps metric stays correct; per-example accuracy does not reflect
  compute-at-halt. Exact only at `batch_size = 1` (see `pondernet/test.py`).
- **Eval on the idle GPU**, not the one training (`CUDA_VISIBLE_DEVICES`), or the
  shared card OOMs mid-run.

## Logging a run

- Launch train/eval with `EXP=<NN-exp> RUN=<run-id>` (the scripts refuse to run without
  a valid `EXP`/`RUN`; `EXP` must match `^[0-9]{2}-[a-z0-9.-]+$`).
  Artifacts land in `<outputs|results|models/checkpoints>/<NN-exp>/<run-id>/`. Explicit
  `SAVE_DIR`/`LOG_DIR`/`RESULTS_DIR` still override the derivation (back-compat).
- Eval self-writes `command.sh`, `eval.log`, and `summary.json` into the results dir.
- Before the first run of a new investigation, create
  `docs/experiments/<NN-exp>/` by hand, following an existing experiment as a template.
  After a run finishes, write the per-run `.md`, update the experiment table, and
  refresh the index. Docs live in git under `docs/experiments/`.

## Running Baselines

```bash
# Coconut (from baselines/Coconut/)
python run.py args/gsm_coconut.yaml

# PonderNet - adaptive halting variant (from pondernet/)
# Trains on the pinned data/gsm8k_aug/subsamples/train100k.jsonl (--max_train_samples 100000) by default.
EXP=04-simcot-pondernet-gammasweep RUN=g0.05-gm3.0-ep5 \
  bash scripts/train_gpt2_gsm8k_pondernet.sh
```

See `docs/pipeline.md` for the full train → eval → record workflow and `docs/parameters.md` for every flag.

## Setup

```bash
uv sync
```

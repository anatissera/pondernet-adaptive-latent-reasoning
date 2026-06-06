# Option-A: K Sweep Multi-Backend

This stage builds the exploratory pipeline for measuring how outputs and scores
change when the latent reasoning budget `k` changes. It does not train a
classifier, add soft labels, add RL, add ACT/PonderNet, or implement dynamic
halting.

## What `k` Means

The pipeline supports two backends:

- `coconut`: `k` is the number of latent stages. The runner inserts
  `k * c_thought` `<|latent|>` tokens between `<|start-latent|>` and
  `<|end-latent|>`, then calls `model.generate(...)`.
- `codi`: `k` is the number of latent iterations. The runner encodes the
  prompt, appends CODI's `bot_id`, runs `k` hidden-state latent iterations,
  appends `eot_id`, and decodes the answer.

The tokenizer is required for both backends.

## Input Format

`data/examples.jsonl` expects one JSON object per line:

```json
{"id": "ex_001", "input": "question text", "gold": "answer"}
```


## Environment Setup

Create a dedicated environment before running real model loads or sweeps:

```bash
python3 -m venv .venv-option-a
source .venv-option-a/bin/activate
pip install -r k-classifier/requirements.txt
```

The loaders require local checkpoints under `k-classifier/models/` and the runtime
dependencies listed in `k-classifier/requirements.txt`. Recommended versions are pinned there, including `transformers==4.46.2`, to match the SIM-CoT/Coconut runtime more closely. If you do not activate the venv in your shell, run the same commands with `.venv-option-a/bin/python`.

Loader notes:

- CODI is loaded through a minimal Option-A inference wrapper that reconstructs the GPT-2 + PEFT LoRA + projection modules needed by the local checkpoint. This avoids importing the original CODI training module, which pulls in unrelated Transformers vision modules in some environments.
- Coconut loads the checkpoint against the original GPT-2-sized auxiliary model and only initializes the added latent-token embeddings after the checkpoint load, preserving a faithful `missing=0`, `unexpected=0` load.

## Smoke Tests

Before running any sweep, verify each backend can load and generate one `k=1`
response without writing result files:

```bash
python k-classifier/scripts/smoke_model_load.py --backend codi
python k-classifier/scripts/smoke_model_load.py --backend coconut
```

If a smoke test fails, inspect the labelled stage in the error: loader import,
prompt load, model load, or `k=1` inference. Do not run large sweeps until these
smoke tests pass.

## Running

Provide a model loader callable that returns `(model, tokenizer)` or an object
with `.model` and `.tokenizer`.

```bash
python k-classifier/scripts/run_k_sweep.py \
  --backend coconut \
  --k-max 6 \
  --n-examples 50 \
  --data k-classifier/data/examples.jsonl \
  --output k-classifier/results/k_sweep_results.jsonl \
  --model-loader your_package.loaders:load_coconut \
  --c-thought 2
```

With the local checkpoints downloaded under `k-classifier/models/`, use:

```bash
python k-classifier/scripts/run_k_sweep.py \
  --backend coconut \
  --k-max 6 \
  --n-examples 50 \
  --data k-classifier/data/examples.jsonl \
  --output k-classifier/results/k_sweep_results.jsonl \
  --model-loader src.model_loaders:load_coconut \
  --c-thought 2
```

For CODI:

```bash
python k-classifier/scripts/run_k_sweep.py \
  --backend codi \
  --k-max 6 \
  --n-examples 50 \
  --data k-classifier/data/examples.jsonl \
  --output k-classifier/results/k_sweep_results.jsonl \
  --model-loader your_package.loaders:load_codi
```

With the local CODI checkpoint:

```bash
python k-classifier/scripts/run_k_sweep.py \
  --backend codi \
  --k-max 6 \
  --n-examples 50 \
  --data k-classifier/data/examples.jsonl \
  --output k-classifier/results/k_sweep_results.jsonl \
  --model-loader src.model_loaders:load_codi
```

The loaders in `src/model_loaders.py` reconstruct the original project wrappers
around a local GPT-2 base model and then load the downloaded SIM-CoT weights.
They require PyTorch, Transformers, and for CODI also PEFT and Safetensors.

## Outputs

`results/k_sweep_results.jsonl` stores one row per example:

```json
{
  "example_id": "ex_001",
  "input": "question text",
  "gold_answer": "answer",
  "predictions": {"1": "...", "2": "...", "3": "..."},
  "scores": {"1": 0.0, "2": 0.0, "3": 1.0},
  "prediction_k1": "...",
  "prediction_k2": "...",
  "prediction_k3": "...",
  "score_k1": 0.0,
  "score_k2": 0.0,
  "score_k3": 1.0,
  "k_star": 3
}
```

`k_star` is the smallest `k` that reaches the maximum observed score for that
example. For scores `[0, 0, 1, 1, 1, 1]`, `k_star = 3`.

`results/k_sweep_summary.csv` includes accuracy by `k`, average score by `k`,
the distribution of `k_star`, and the percentage of examples where outputs
change across `k`.

## What To Inspect

- Do answers change when `k` changes?
- Does average score improve as `k` increases?
- Is there variation in `k_star` across examples?
- Do most examples collapse to the same `k_star`?

Local Coconut/SIM-CoT and CODI checkpoints are stored under `k-classifier/models/`
when downloaded, but they are ignored by git. Treat smoke-test output and debug
sweeps as diagnostics, not experimental results.

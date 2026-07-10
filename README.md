# adaptive-latent-reasoning

NLP final project (UdeSA): **adaptive latent chain-of-thought reasoning**.

Latent chain-of-thought systems such as [SIM-CoT](https://arxiv.org/pdf/2509.20317)
(ICLR 2026), CODI, and Coconut reason in a fixed number of implicit (latent) steps,
regardless of how hard the problem is. This project makes that latent budget
**variable at inference time**, so easy problems spend less compute than hard ones.

We explored the three approaches proposed in our survey, each attacking a different
axis of the latent budget:

| Option | Idea | Axis | Branch | Where in main |
|--------|------|------|--------|----------------|
| **A** | An auxiliary classifier predicts, upfront, how many latent steps `k` a prompt needs | number of steps, chosen before reasoning | `option-a-k-classifier` | [`k-classifier/`](k-classifier/README.md) |
| **B** | A distilled step-maturity signal decides how many latent vectors to spend inside each step (`c`) | vectors per step | `option-b-adaptive-vectors` | [`adaptive-vectors/`](adaptive-vectors/README.md) |
| **C** | A PonderNet halting head learns, from the task signal, when to stop reasoning | number of steps, decided on the fly | `option-c-pondernet` | [`pondernet/`](pondernet/README.md) |

Option C is the primary line of work (experiments 01-11); Options A and B are complete
studies with their own findings. The written report lives on the `paper` branch.

## Headline results

- **Option C (PonderNet halting).** On a held-out validation split, adaptive halting
  matches the fixed-K=6 baseline (40.80% greedy) while cutting average latent steps by
  up to 32% (40.6% at 2.93 steps), and halting tracks instance difficulty strongly
  (Spearman +0.675 between steps used and expression count). Full index with per-run
  docs: [`docs/experiments.md`](docs/experiments.md).
- **Option A (upfront k* classifier).** The k sweeps confirm the premise: the optimal
  latent budget varies per instance and saturates below the maximum (accuracy 40.8% at
  k=1 to ~64% at k=5-6 on the full train sweep). Oracle labels are heavily imbalanced
  (66% of examples peak at k=1), which shaped the classifier design. See
  [`k-classifier/README.md`](k-classifier/README.md).
- **Option B (adaptive vectors per step).** The mechanism works and trims 16-33% of
  latent compute at no accuracy cost, but on atomic GSM8K-Aug steps the required `c`
  is nearly constant (~2), so smart adaptivity has no headroom there; with coarser,
  variable-density steps the per-instance signal reappears (adaptive beats random by
  2.7 sigma). A clean, well-diagnosed negative-then-qualified result. See
  [`adaptive-vectors/RESULTS.md`](adaptive-vectors/RESULTS.md).
- **Methodology.** Mid-project we found that experiments had been selected on the test
  split (selection bias, not leakage). All headline numbers were re-validated on a
  never-touched 500-example validation split, which retired two earlier headlines
  (the 42.23% claim and the "recipe C beats A" finding, experiment 11).

## Repository layout

```
pondernet/           # Option C codebase: train.py, test.py, src/model.py, scripts/
k-classifier/            # Option A pipeline: k sweeps, labels, classifiers
adaptive-vectors/            # Option B codebase and study (gated by --option_b)
baselines/           # Reference implementations (Coconut, CODI, SIM-CoT), read-only
docs/experiments.md  # Experiment index (01-11) -> per-experiment -> per-run docs
docs/pipeline.md     # End-to-end workflow: artifacts -> train -> eval -> record
docs/parameters.md   # CLI flag reference and warm-start recipes
docs/papers/         # Paper summaries for context
models/, outputs/, results/, data/   # Run artifacts (gitignored, shared on disk)
```

## Branches

- `main`: everything integrated; the reference state of the project.
- `option-a-k-classifier`, `option-b-adaptive-vectors`, `option-c-pondernet`: the
  final state of each approach, each with its own README and results.
- `paper`: the written report (ACL format, in Spanish).

## Setup

```bash
uv sync
```

This covers Option C (the `pondernet/` code). Option A uses its own environment
(`.venv-option-a` + `k-classifier/requirements.txt`; see `k-classifier/AGENTS.md`). Pretrained
artifacts (GPT-2, the SIM-CoT CODI checkpoint, the extracted decoder) and datasets are
gitignored; see step 1 of [`docs/pipeline.md`](docs/pipeline.md) for where each comes
from.

## Running experiments

**Option C** runs are identified by an experiment folder and run id, from which the
scripts derive checkpoint/log/result paths:

```bash
cd pondernet
EXP=04-simcot-pondernet-gammasweep RUN=g0.05-gm3.0-ep5 \
  CUDA_VISIBLE_DEVICES=0 bash scripts/train_gpt2_gsm8k_pondernet.sh
EXP=04-simcot-pondernet-gammasweep RUN=g0.05-gm3.0-ep5 \
  bash scripts/eval_gpt2_gsm8k_pondernet.sh
```

Key flags: `--max_latent_steps` (K_max), `--pondernet_gamma` (halting pressure),
`--pondernet_geom_mean` (prior mean), `--pondernet_inf_threshold` (inference
threshold). Full reference: [`docs/parameters.md`](docs/parameters.md).

**Option A**:

```bash
python3 k-classifier/scripts/run_k_sweep.py --backend codi --k-max 8 \
  --data k-classifier/data/gsm8k_test_100.jsonl \
  --output k-classifier/results/sweep.jsonl \
  --model-loader src.model_loaders:load_codi
```

**Option B**:

```bash
bash adaptive-vectors/scripts/train_gpt2_gsm8k_optionb.sh
K=4 MMAX=3 EPS=0.05 bash adaptive-vectors/scripts/eval_gpt2_gsm8k_optionb.sh
```

## Context for contributors

See `AGENTS.md` for the agent briefing and `docs/pipeline.md` for the full Option C
workflow, including the experiment-scaffolding and run-recording conventions that keep
the experiment index consistent.

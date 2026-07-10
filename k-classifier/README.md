# Option A: a predictor of the latent reasoning budget

Option A trains a lightweight classifier that, given a task prompt, predicts how many
latent reasoning steps `k` the main model should use before answering:

```
task prompt -> classifier -> estimated k -> model reasons with that k -> answer
```

This folder contains the experimental pipeline for that option: the k sweeps that
establish whether a useful signal exists, the label construction, and two classifier
variants (a regression head over frozen sentence embeddings, and a multi-output
DistilBERT classifier).

## What k means

`k` is the amount of latent reasoning the model is allowed to do.

- In CODI, `k` is the number of internal latent iterations before generating the answer.
- In Coconut, `k` is the number of latent stages added to the prompt via special tokens.

## Stage 1: does k matter? (k sweeps)

The same model is run repeatedly over the same examples while varying `k`, and each
prediction is scored. If every `k` gave the same result there would be nothing to
predict; if different examples peak at different `k`, Option A makes sense.

For each example the sweep records the prompt, the gold answer, the prediction and
score at every `k`, and `k_star`: the smallest `k` that reaches the example's best
score, i.e. the minimum latent budget needed for its best answer.

### Pilot result (CODI, 100 GSM8K examples, k_max=8)

```
Accuracy by k:  k=1: 0.31  k=2: 0.30  k=3: 0.29  k=4: 0.35  k=5: 0.33  k=6: 0.36  k=7: 0.37  k=8: 0.37
k_star distribution:  k=1: 79  k=2: 10  k=3: 4  k=4: 5  k=5: 0  k=6: 1  k=7: 1  k=8: 0
Outputs changed across k: 96%
```

Changing `k` changes the answer in 96% of examples, accuracy rises from 31% (k=1) to
37% (k=7/8), and in 21 of 100 examples the best result is not at k=1. The latent
budget has a real effect, so learning to assign it is meaningful.

### Full-train sweep (CODI, n=7473, k_max=8)

```
Accuracy by k:  k=1: 0.408  k=2: 0.427  k=3: 0.479  k=4: 0.579  k=5: 0.637  k=6: 0.640  k=7: 0.637  k=8: 0.631
k_star distribution:  k=1: 4939 (66%)  k=2: 1243  k=3: 396  k=4: 446  k=5: 319  k=6: 75  k=7: 32  k=8: 23
Outputs changed across k: 97.4%
```

Accuracy climbs steeply up to k of about 5-6 (+23 points over k=1) and then saturates;
k=7 and k=8 do not improve. This confirms the project premise (an optimal latent budget
exists below the maximum) and suggests capping the predicted range at k in {1..5}.
The k_star distribution is heavily imbalanced toward k=1, which drives the design
decisions below. Raw numbers: `results/k_sweep_summary.csv`,
`results/k_sweep_train_full_codi.jsonl`. Design analysis:
[`docs/classifier-design.md`](docs/classifier-design.md).

## Stage 2a: regression classifier over frozen embeddings

Design (full rationale in [`docs/classifier-design.md`](docs/classifier-design.md)):

- Input representation r(x): frozen `all-MiniLM-L6-v2` sentence embeddings (384 dims),
  precomputed once, CPU-friendly, independent of the SIM-CoT backbone.
- Formulation: regression over k with a calibrated rounding/threshold step, not
  multiclass. The k_star distribution is so imbalanced (66% at k=1) that a multiclass
  model would predict "1" everywhere; and k is ordinal, which regression respects.
- Labels: k_star(x) from the sweeps, joined to embeddings by example id.
- Known asymmetry: underestimating k leaves answers wrong, overestimating only wastes
  compute, so the loss should penalize underestimation more.

## Stage 2b: multi-output k classifier (DistilBERT)

Trains a multi-output classifier predicting which latent budgets are likely to solve a
prompt. Each training example maps a prompt to a binary vector over k values (e.g.
`[0,0,1,1,1,0,0,0]`); the classifier outputs one logit per k and is trained with
`BCEWithLogitsLoss`.

```bash
# Build the dataset from a sweep
python3 k-classifier/scripts/build_k_classifier_dataset.py \
  --input k-classifier/results/k_sweep_train_full_codi.jsonl \
  --output k-classifier/data/k_classifier_train.jsonl \
  --k-min 1 --k-max 8

# Train
python3 k-classifier/scripts/train_k_classifier.py \
  --data k-classifier/data/k_classifier_train.jsonl \
  --output-dir k-classifier/results/k_classifier_distilbert \
  --model-name distilbert-base-uncased \
  --epochs 3 --batch-size 16 --lr 2e-5 --max-length 256 \
  --threshold 0.7 --fallback-k 6

# Predict k for a prompt
python3 k-classifier/scripts/predict_k.py \
  --checkpoint k-classifier/results/k_classifier_distilbert \
  --prompt "Natalia sold clips to 48 of her friends..."
```

## File structure

```
k-classifier/
├── data/              # jsonl datasets (sweep inputs, classifier training sets)
├── models/            # local checkpoints, gitignored
├── results/           # sweep outputs and summaries
├── scripts/           # experiment entry points
├── src/               # model loading, inference, sweep, and metrics logic
├── docs/              # classifier design notes
└── requirements.txt
```

Main modules and scripts:

```
src/model_runner.py    # runs the model with a given k
src/k_sweep.py         # sweeps k=1..k_max over a dataset
src/metrics.py         # exact match / accuracy
src/model_loaders.py   # loads CODI and Coconut from local checkpoints
src/k_classifier.py    # multi-output classifier model

scripts/run_k_sweep.py             # main sweep experiment
scripts/smoke_model_load.py        # checks that the models load and generate
scripts/prepare_gsm8k.py           # converts GSM8K to the expected format
scripts/download_models.py         # downloads the required local weights
scripts/build_k_classifier_dataset.py  # sweep results -> classifier dataset
scripts/train_k_classifier.py      # trains the multi-output classifier
scripts/predict_k.py               # single-prompt k prediction
```

## Dataset format

One JSON object per line:

```json
{"id": "ex_001", "input": "What is 2 + 2?", "gold": "4"}
```

## How to run

```bash
source .venv-option-a/bin/activate
python k-classifier/scripts/download_models.py

# Smoke tests
python k-classifier/scripts/smoke_model_load.py --backend codi
python k-classifier/scripts/smoke_model_load.py --backend coconut

# Sweep with CODI
python k-classifier/scripts/run_k_sweep.py \
  --backend codi --k-max 8 --n-examples 100 \
  --data k-classifier/data/gsm8k_test_100.jsonl \
  --output k-classifier/results/gsm8k_codi_k8_n100_results.jsonl \
  --model-loader src.model_loaders:load_codi

# Sweep with Coconut
python k-classifier/scripts/run_k_sweep.py \
  --backend coconut --k-max 6 --n-examples 100 \
  --data k-classifier/data/gsm8k_test_100.jsonl \
  --output k-classifier/results/gsm8k_coconut_k6_n100_results.jsonl \
  --model-loader src.model_loaders:load_coconut \
  --c-thought 2
```

## External resources

Model weights (the loaders expect them under `k-classifier/models/`):

- [SIM-CoT GPT-2 Coconut](https://huggingface.co/internlm/SIM_COT-GPT2-Coconut/tree/main) -> `k-classifier/models/SIM_COT-GPT2-Coconut`
- [SIM-CoT GPT-2 CODI](https://huggingface.co/internlm/SIM_COT-GPT2-CODI/tree/main) -> `k-classifier/models/SIM_COT-GPT2-CODI`
- [GPT-2 base](https://huggingface.co/openai-community/gpt2) -> `k-classifier/models/gpt2`

Dataset: [GSM8K](https://huggingface.co/datasets/openai/gsm8k)

## Remaining work

- Compare the trained classifiers against fixed-k baselines end to end.
- Report not only answer accuracy but also the average reasoning cost with predicted k.
- Address the k_star imbalance (reweighting, target transform, or an asymmetric loss).

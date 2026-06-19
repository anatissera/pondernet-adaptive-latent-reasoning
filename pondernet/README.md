# PonderNet adaptive halting for CODI

This adds a PonderNet-style (Banino et al. 2021) adaptive halting mechanism on
top of CODI's fixed-K latent chain-of-thought: instead of always running a
fixed number of latent reasoning steps `K`, the model learns a per-instance
halting probability at each step and can stop early.

> **Layout note**: `train.py` and `test.py` live at the root of `pondernet/` (not in `src/`) because they are CLI entry points invoked directly by the shell scripts in `scripts/`. `src/model.py` is the library they import. Run everything from the `pondernet/` directory.

Enabled via the `--pondernet True` flag (default `False` — the original
fixed-K SIM-CoT/CODI training path is unaffected when this is off).

## Hyperparameters

All flags below live in `src/model.py` (`TrainingArguments`) and are set in
`scripts/train_gpt2_gsm8k_pondernet.sh` / `scripts/eval_gpt2_gsm8k_pondernet.sh`.

| Flag | Current value | Meaning |
|---|---|---|
| `--max_latent_steps` | `6` | Hard upper bound on latent reasoning steps (`K_max`). The model can take at most this many steps even if it never decides to halt. |
| `--pondernet_halt_bias_init` | `-2.0` | Initial bias of the halting head (`halt_head`, a linear layer producing `λ_k`). A negative bias makes `sigmoid(logit)` start low, so the model begins by *not* wanting to halt early — gives the latent loop room to "warm up" before learning when to stop. |
| `--pondernet_beta` | `1.0` | Weight on the auxiliary decoder loss `L_step` (the per-step reconstruction loss CODI already uses) in the total loss. |
| `--pondernet_gamma` | `0.01` | Weight on `KL_geom`, the KL-divergence regularizer that pulls the learned halting distribution `p_k` toward a geometric prior. Keeps the model from either always running to `K_max` or halting too aggressively. |
| `--pondernet_geom_mean` | `3.0` | Mean of the geometric prior distribution used by `KL_geom`. This is *the* PonderNet hyperparameter that encodes "how many latent steps we expect the model to need on average" — it shapes (but does not hard-cap) the number of steps the model converges to using. |
| `--pondernet_inf_threshold` | `0.5` | Inference-time stopping threshold: the model halts as soon as its accumulated halting probability crosses this value (default `0.5`, i.e. "more likely halted than not"). Only used at inference (`test.py`), not during training. |

## Loss

```
L_total = L_pondernet + beta * L_step + gamma * KL_geom
```

where `L_pondernet = E_{k ~ p_k}[ CE(answer | first k latent steps) ]` is the
expected answer cross-entropy under the learned halting distribution
`p_k = (∏_{j<k} (1-λ_j)) · λ_k`, and `KL_geom` regularizes `p_k` against a
geometric prior with mean `pondernet_geom_mean`.

## Training

Launch with `EXP=<NN-exp> RUN=<run-id>`; the script derives `SAVE_DIR`/`LOG_DIR` from
them (artifacts land under `<models/checkpoints|outputs>/<NN-exp>/<run-id>/`). The
script refuses to run without a valid `EXP`/`RUN`.

```bash
EXP=04-simcot-pondernet-gammasweep RUN=g0.05-gm3.0-ep5 \
  CUDA_VISIBLE_DEVICES=0 bash scripts/train_gpt2_gsm8k_pondernet.sh
```

Training auto-resumes from the last checkpoint found in `output_dir` if the
run is interrupted (see `train.py`) — just re-run the same command. Explicit
`SAVE_DIR`/`LOG_DIR` still override the `EXP`/`RUN` derivation (back-compat).

### Warm-starting the auxiliary decoder from SIM-CoT

A vanilla-GPT-2 auxiliary decoder gives `L_step`/`L_pondernet` no real signal until
it has trained for a while (the "cold start" problem), so it is warm-started from a
SIM-CoT-trained decoder checkpoint.

That decoder lives at `models/pretrained/simcot-gpt2-decoder/` (gitignored, like all
model artifacts), so it is obtained on demand via the fetch script below — no copy is
committed to the repo. The training script defaults `DECODER_PATH` to
`../models/pretrained/simcot-gpt2-decoder` and passes `--decoder_path`, which `model.py`
loads as a drop-in replacement for the vanilla decoder (verified compatible: identical
GPT-2 124M architecture and vocab, so `pj_in`/`pj_out` resolve to `Identity` with no
extra untrained projector parameters).

To fetch (or regenerate) that checkpoint, run:

```bash
python scripts/fetch_simcot_decoder.py --out ../models/pretrained/simcot-gpt2-decoder
```

which downloads `internlm/SIM_COT-GPT2-CODI` and extracts the `decoder.*` weights into a
standalone GPT-2 checkpoint. See "Warm-start strategy" below for how the decoder-only and
full-model recipes are selected.

### Warm-start strategy (two recipes)

There are **two warm-start recipes**, both wired into `train.py` and the training
script, selected by which checkpoint variables you set. `GPT2_PATH`
(`model_name_or_path`) always stays a plain GPT-2 — it is only the backbone
scaffold, **never** the SIM-CoT CODI checkpoint (pointing it there loads the wrapper
as a bare GPT-2 and silently random-inits the backbone).

| Recipe | backbone (`codi.*`) | decoder (`decoder.*`) | enabled by |
|---|---|---|---|
| **decoder-only** | cold (vanilla GPT-2 + fresh LoRA) | warm | `--decoder_path` (loaded in `CODI.__init__`) |
| **full-model** | warm | warm (comes with the checkpoint) | `--simcot_ckpt` (`load_state_dict` after assembly) |

The SIM-CoT CODI checkpoint holds three namespaces — `codi.*` (245 tensors,
backbone + LoRA), `decoder.*` (149), `prj.*` (6) = 400 total — and
`fetch_simcot_decoder.py` extracts a standalone copy of exactly the `decoder.*`
subset for the decoder-only path. So **a full-model warm-start already includes the
decoder**: when `--simcot_ckpt` is set, the `--decoder_path` load in `__init__` is
overwritten by the same `decoder.*` weights and is therefore redundant (harmless —
identical tensors — just wasted compute).

**Selecting a recipe.** The script passes *both* flags. `SIMCOT_CKPT` defaults to the
SIM-CoT CODI checkpoint, so out of the box you get the **full-model** recipe; set
`SIMCOT_CKPT=""` to fall back to **decoder-only**:

```bash
# full-model warm-start (default): warm backbone + LoRA + decoder + prj
bash scripts/train_gpt2_gsm8k_pondernet.sh

# decoder-only warm-start: cold backbone, just bootstrap the halting signal
SIMCOT_CKPT="" bash scripts/train_gpt2_gsm8k_pondernet.sh
```

Like the decoder, the **full CODI checkpoint**
(`models/pretrained/simcot-gpt2-codi/`, ~730 MB) is **gitignored and not in the repo** — repo
owners share it on the same filesystem. There is no fetch script for it; a fresh
environment without that path present must either provide it (download
`internlm/SIM_COT-GPT2-CODI`) or run decoder-only via `SIMCOT_CKPT=""`. (The script
uses `${SIMCOT_CKPT-…}` rather than `:-`, so an explicit empty value is honored.)

In both cases `--pondernet` freezes the backbone and trains the LoRA adapters plus
the freshly-initialized `halt_head` (the only params not covered by a warm-start).
The `--simcot_ckpt` load is **sentinel-checked**: it raises if any checkpoint tensor
fails to map onto the model, or if a core weight (e.g. `decoder.lm_head.weight`) is
missing — guarding against the namespace-mismatch failure that once trained a model
to garbage. Note: never pass the CODI checkpoint as `model_name_or_path`; that is the
exact mistake `--simcot_ckpt` exists to avoid.

## Evaluation

Run eval at `--batch_size 1` (default) for faithful halting, on the **idle** GPU:

```bash
# Adaptive halting (PonderNet)
EXP=04-simcot-pondernet-gammasweep RUN=g0.05-gm3.0-ep5 CKPT=/path/to/checkpoint \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval_gpt2_gsm8k_pondernet.sh

# Fixed-K baseline, for comparison (set NUM_LATENT to whatever K you trained with)
EXP=01-simcot-baselines RUN=fixedk-k6 CKPT=/path/to/checkpoint NUM_LATENT=6 \
  bash scripts/eval_gpt2_gsm8k_fixedk.sh
```

`RESULTS_DIR` is derived as `../results/<NN-exp>/<run-id>/` (or pass it explicitly to
override). Both eval scripts self-write `command.sh`, `eval.log`, and `summary.json`
there. The PonderNet eval reports accuracy, the average number of latent steps used,
and an accuracy-vs-budget table (plus a per-instance JSON dump for offline plotting).

## Documentation

- [`../docs/pipeline.md`](../docs/pipeline.md) — end-to-end workflow: acquire artifacts, choose a warm-start recipe, train, evaluate, and record results in the run manifest (includes a Mermaid flow diagram).
- [`../docs/parameters.md`](../docs/parameters.md) — complete CLI flag reference, both warm-start recipes and the `model_name_or_path` trap, and the module/loss-term glossary.
- [`../docs/experiments.md`](../docs/experiments.md) — experiment index → per-experiment `experiment.md`/`runs.md` → per-run `<run-id>.md`: artifact layout, hparams, and accuracy results. Launch runs with `EXP`/`RUN`; record them by following the existing run entries' format.

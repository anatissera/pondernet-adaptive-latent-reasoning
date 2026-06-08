# PonderNet adaptive halting for CODI

This adds a PonderNet-style (Banino et al. 2021) adaptive halting mechanism on
top of CODI's fixed-K latent chain-of-thought: instead of always running a
fixed number of latent reasoning steps `K`, the model learns a per-instance
halting probability at each step and can stop early.

Enabled via the `--pondernet True` flag (default `False` — the original
fixed-K SIM-CoT/CODI training path is unaffected when this is off).

## Hyperparameters

All flags below live in `src/model.py` (`TrainingArguments`) and are set in
`scripts/train_gpt2_gsm8k_pondernet.sh` / `scripts/eval_gpt2_gsm8k_pondernet.sh`.

| Flag | Current value | Meaning |
|---|---|---|
| `--num_latent` | `6` | Hard upper bound on latent reasoning steps (`K_max`). The model can take at most this many steps even if it never decides to halt. |
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

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_gpt2_gsm8k_pondernet.sh
```

Training auto-resumes from the last checkpoint found in `output_dir` if the
run is interrupted (see `train.py`) — just re-run the same command.

## Evaluation

```bash
# Adaptive halting (PonderNet)
CKPT=/path/to/checkpoint bash scripts/eval_gpt2_gsm8k_pondernet.sh

# Fixed-K baseline, for comparison (set NUM_LATENT to whatever K you trained with)
CKPT=/path/to/checkpoint NUM_LATENT=6 bash scripts/eval_gpt2_gsm8k_fixedk.sh
```

The PonderNet eval reports accuracy, the average number of latent steps used,
and an accuracy-vs-budget table (plus a per-instance JSON dump in
`results_dir` for offline plotting).

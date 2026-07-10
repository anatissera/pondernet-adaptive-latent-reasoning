# Option B: adaptive vectors per step (the c axis)

This branch is the final state of **Option B** of the adaptive latent reasoning
project: making adaptive the number of latent vectors that compose each reasoning step
(`c`), instead of fixing it as a hyperparameter. It is the axis orthogonal to Option C
(PonderNet), which adapts the number of steps `K`.

For the project overview and the other two approaches (Option A: an upfront k*
classifier; Option C: PonderNet adaptive halting), see the `main` branch.

## What was done and found

All the work lives in [`adaptive-vectors/`](adaptive-vectors/README.md). The study ran to completion,
with a clean and well-diagnosed outcome:

1. **The mechanism works.** A small MLP distills the SIM-CoT decoder's per-step
   reconstruction loss and halts sub-vector generation at inference. It trims 16-33%
   of the latent compute with no accuracy cost against the ~39.5% SIM-CoT baseline.
2. **But on GSM8K-Aug with atomic steps there is no exploitable headroom.** The
   accuracy-vs-c curve saturates hard at c=2 for nearly every instance, learned halting
   does not beat random at matched budget, and the simple optimum is fixed c=2.
3. **The negative is granularity-specific, not a property of the method.** With
   coarser steps (2-3 operations each, so per-step density varies), the third vector
   contributes again and adaptive halting beats random by about 2.7 sigma; a cold-start
   control showed the c=1 warm-start anchor is load-bearing rather than just a bias.

Full results and diagnosis: [`adaptive-vectors/RESULTS.md`](adaptive-vectors/RESULTS.md). Traceable
implementation spec: [`adaptive-vectors/IMPLEMENTATION.md`](adaptive-vectors/IMPLEMENTATION.md).
Phase-by-phase decision log: [`adaptive-vectors/AGENTS.md`](adaptive-vectors/AGENTS.md).

## Structure

```
adaptive-vectors/            # The Option B codebase and docs (gated by --option_b)
SIM-CoT/             # Vendored reference implementations (this branch predates the
                     # baselines/ rename on main; content is the same)
main.py, pyproject.toml   # uv scaffolding
```

## How to run

Training and evaluation scripts, flags, checkpoints, and artifact paths are documented
in [`adaptive-vectors/IMPLEMENTATION.md`](adaptive-vectors/IMPLEMENTATION.md) (sections 4 and 5):

```bash
# Train (RTX 3060 defaults; env vars K,M,BS,ACCUM,EPOCHS,MAXSAMPLES,LR,LAMBDA_HALT,HALT_HEAD_LR)
bash adaptive-vectors/scripts/train_gpt2_gsm8k_optionb.sh

# Evaluate (bs=1; env vars K,MMAX,EPS; add --ob_random for the random baseline)
bash adaptive-vectors/scripts/eval_gpt2_gsm8k_optionb.sh

# Unbiased retrain variants
bash adaptive-vectors/scripts/train_gpt2_gsm8k_optionb_cold.sh        # cold start + coarse steps
bash adaptive-vectors/scripts/train_gpt2_gsm8k_optionb_warmcoarse.sh  # warm start + coarse steps
```

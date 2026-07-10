# Option B: adaptive vectors per step (the `c` axis)

Option B makes **the number of latent vectors composing each reasoning step** (`c`)
adaptive, instead of fixing it as a hyperparameter. It is the axis **orthogonal** to
Option C / `pondernet/`, which adapts the *number of steps* `K`.

```
                    K axis (Option C / pondernet)  ->  how many steps to reason?
problem  ->  SIM-CoT  ->                                                        ->  answer
                    c axis (Option B / this folder) ->  how many vectors per step?
```

- **Option C / pondernet** decides *when to stop reasoning* (number of steps `K`).
- **Option B** decides *how many sub-vectors to spend building each step* (`c`),
  without changing the number of steps.

The two are complementary and operate on different axes of the latent budget.

---

## The idea

SIM-CoT trains an **auxiliary decoder** that reconstructs the text of each reasoning
step from the latent vector `z_k`. That reconstruction produces a per-step loss,
`L_step,k`, which measures **how well the vector semantically represents its step**:
high means it does not capture it yet; low means it already does.

Normally `L_step` is discarded at inference along with the decoder. Option B
**distills that signal into a small MLP** that does survive inference:

```
L_hat_k = MLP(h_k)              # predicts L_step,k from the hidden state h_k
L_dist  = (L_hat_k - L_step,k)^2  # the MLP learns to imitate the decoder
```

At inference the decoder is discarded and the MLP is kept. Each step is built as a
sequence of sub-vectors `z_{k,1}, z_{k,2}, ...` generated one at a time; after each
sub-vector, `L_hat` is evaluated. When `L_hat` stops going down, the step is considered
**mature** and the model moves on to the next step.

Unlike ACT/PonderNet (which learn to stop from the task signal alone), the MLP does not
infer maturity from scratch: **it imitates what the decoder already knows**. Unlike
geometric convergence, it tracks **semantic maturity** instead of movement in latent
space.

---

## Training objective

```
L_total = lambda_ans * L_ans + lambda_step * L_step + lambda_halt * L_halt
L_halt  = L_dist + lambda * sum_k n_k * sigmoid(-L_hat_k)
```

`L_step` (SIM-CoT's decoder loss) **is kept**: it stabilizes the latent
representations and is central to SIM-CoT. The `lambda * sum_k n_k * sigmoid(-L_hat_k)`
term penalizes spending extra sub-vectors once the step already looks mature
(`sigmoid(-L_hat)` is high when `L_hat` is low). All weights are configurable from the
command line (`ob_*` flags in `src/model.py`).

---

## Status

Incremental implementation, gated by the `--option_b` flag (off by default, so the
inherited SIM-CoT path stays intact). The phase-by-phase log of decisions and changes
is in `AGENTS.md`; the results and conclusions are in `RESULTS.md`.

| Phase | What it does | Status |
|-------|--------------|--------|
| 0 | Scaffold + inert flags + docs | done |
| 1 | Feasibility probe (does `L_step` go down within a step?) | done (NO-GO on the pretrained checkpoint) |
| 2 | MLP head + `L_dist` | done |
| 3 | Ponder penalty + full objective | done |
| 4 | Adaptive sub-vector inference | done |
| 5 | Train + evaluate on GSM8K-Aug | done |
| 6 | Unbiased retrain (cold + coarse) + diagnosis | done |
| 7 | Warm + coarse (the missing 2x2 cell) | done |

---

## Structure

```
adaptive-vectors/
├── src/model.py        # CODI/SIM-CoT + distillation MLP + sub-vector loop
├── train.py            # training (inherited from the pondernet harness)
├── test.py             # evaluation; adaptive per-sub-vector halting
├── smoke_optionb.py    # overfit smoke test on a few examples
├── scripts/            # decoder fetch, setup, training/eval
├── README.md           # this file
├── RESULTS.md          # results and conclusions
├── IMPLEMENTATION.md   # traceable spec of the non-fixed-c implementation
└── AGENTS.md           # traceability log (decisions + changes per phase)
```

## Code base

The harness (backbone+LoRA+decoder assembly, data wiring, training/eval loop, and the
`gradient_checkpointing` gotcha handling) was copied from the `pondernet/` directory,
as the proven SIM-CoT/CODI copy. PonderNet's halting head and its objective (the `K`
axis) are **not** reused: they are replaced by the `c` axis logic. `k-classifier/`,
`pondernet/`, and the original SIM-CoT path are untouched.

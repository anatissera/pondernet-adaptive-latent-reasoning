# 11 - simcot-pondernet-k-recipe

**Status: complete (finding retired after validation re-eval).**

This experiment campaign ran on the `pondernet` / `feat/adaptive-k-from-scratch` line in
parallel with experiments 05-08 and was integrated here afterwards. Full run-by-run detail,
tables, and operational notes are in [runs.md](runs.md).

## What it tested

The warm-started SIM-CoT backbone was always trained with K=6 latent steps, so the halting
head may inherit a "6 steps is normal" prior instead of tracking true instance difficulty
(GSM8k-Aug's mode is 2-3 steps). This factorial asks whether breaking that prior at the
training-recipe level (not just via gamma pressure) concentrates halting closer to the
data's true step distribution, and at what accuracy cost.

Three training recipes, crossed with `K_max` in {4, 6, 8, 10, 12} (gamma=0.1,
geom_mean=3.0, 5 epochs, seed 42 for the factorial):

- **Recipe A** (baseline): warm-started backbone, frozen; train `lora_*` + `halt_head` only.
- **Recipe B** (cold): plain GPT-2 backbone, no full-model warm-start; train `lora_*` +
  `prj.*` + `halt_head`.
- **Recipe C** (unfreeze): warm-started backbone, train the whole backbone + `prj` +
  `halt_head`.

## Results

1. **Recipe B collapsed** to ~2% accuracy: a cold LoRA-scale training budget cannot
   recover what the SIM-CoT warm start provides. Negative result, definitive.
2. **On the GSM8K test split, recipe C beat A at all five K_max** (mean +0.97pt) and in
   3/3 seeds at K=6 (mean +0.83pt). This was the original headline.
3. **The headline did not survive the held-out validation re-eval.** The whole factorial
   had been compared on the same test split used for every selection decision, the same
   selection-bias issue found in experiments 01-07 (see `fix/validation`). Re-scored on
   the never-touched 500-example validation split, the sign flips: A wins 4/5 K_max
   (mean -0.36pt) and 2/3 seeds at K=6 (mean -0.20pt).

## Verdict

**Recipe C is retired as a finding.** The apparent C > A gap was selection bias from
repeatedly scoring ~20 runs on one fixed test set, not a real recipe difference; both
deltas sit inside each recipe's own seed-to-seed noise (~0.6pt). The durable outputs of
this campaign are methodological: the fixed-k eval mode (`test.py --fixed_k_eval`), the
per-step answer-readiness diagnostics, the multi-seed error-bar protocol, and the
validation-split discipline that experiment 10 inherits.

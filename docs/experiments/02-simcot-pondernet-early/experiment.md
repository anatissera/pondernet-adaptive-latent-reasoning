# 02: Early PonderNet runs (pre-gcfix)

> ⚠️ **Biased metric — test set, not reconcilable.** The numbers on this page were computed on the GSM8K **test** split (not the held-out validation set), and the checkpoint no longer exists, so they cannot be re-evaluated on validation. Treat them as historical/biased. See the [eval-split & leakage note](../../experiments.md#eval-split-and-leakage-note).

**Status:** superseded   **Dates:** 2026-06-08 → 2026-06-09

## What's being tested
The first PonderNet adaptive-halting runs on top of SIM-CoT. These explored learning rate
and warm-starting, but **all** of them were trained with the latent loop's KV cache
silently disabled (the `--gradient_checkpointing True` bug), so they train against a broken
objective. Kept for the record; superseded by [03](../03-simcot-pondernet-gcfix/experiment.md).

## Setup
- Method: PonderNet halt head on SIM-CoT, ep=40, seed=42, eval at halting thresholds 0.8/0.9.
- Varied: learning rate (`lr1e3`, `lr1e4`) and warm-start (`warmstart-lr1e4`).
- Held fixed (and broken): `--gradient_checkpointing True` → `use_cache=False` → every
  latent step runs context-free. See the Root Cause in
  [03/experiment.md](../03-simcot-pondernet-gcfix/experiment.md#root-cause).

## Findings
- Best of the batch: `lr1e4` @ thr=0.8 → **19.26%**, roughly **half** the 39.5% baseline.
- Warm-starting (`warmstart-lr1e4`, 15.24%) did not help — the cache bug dominates.
- Conclusion: these numbers are artifacts of the bug, not of the method. Do not use them
  as a PonderNet baseline; the real PonderNet baseline is `03/100k` (42.23%).

See [runs.md](runs.md) for the run table · artifacts under `<dir>/02-simcot-pondernet-early/`.
Dead/aborted siblings (`halthead-ep40`, `joint-ep40`, `joint-warmstart`) live under `<dir>/archive/`.

> **Weights deleted (2026-06-20).** All `models/checkpoints/02-simcot-pondernet-early/` and
> `models/checkpoints/archive/` weights were removed to reclaim disk (~5 GB) — they were
> known-bad (cache-bug objective) or dead/aborted, with zero scientific value. The findings
> above and any eval logs in `results/` are preserved; only the weights are gone (recoverable
> only by retraining, which there is no reason to do).

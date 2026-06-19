# 02: Early PonderNet runs (pre-gcfix)

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

# 03-simcot-pondernet-gcfix — Runs

See [experiment.md](experiment.md) for what's being tested (incl. the Root Cause writeup).

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| 100k | `--gradient_checkpointing False`, 100k data | **41.20%** @ ep5 thr0.5 (val, n=500) | 5.562 | ✅ done | [detail](100k.md) |

Re-validated 2026-06-23 (validation 500 ex, greedy): only ep4/ep5 survived (ep4 41.00%, ep5 41.20% @ thr0.5). The old headline ep2 = **42.23% was on test (biased); its checkpoint was deleted and cannot be re-validated**. Validation difficulty tracking Spearman(n_expr, steps) = +0.456 @ thr0.5.

Migrated from the old `simcot-pondernet-gcfix-100k` dir (renamed `gcfix-100k` → `100k`). The
earlier train-only `simcot-pondernet-100k` copy was superseded and archived
(`<dir>/archive/100k-pre-gcfix`).

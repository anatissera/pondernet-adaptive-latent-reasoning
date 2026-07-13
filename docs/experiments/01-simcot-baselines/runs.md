# 01-simcot-baselines - Runs

See [experiment.md](experiment.md) for what's being tested.

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| baseline-k6 | upstream default lr=1e-3 | **40.80%** (val, n=500, greedy) | 6.00 (fixed) | ✅ reference | [detail](baseline-k6.md) |
| fixedk-k6-lr1e4 | lr=1e-4 | 17.82% (test, biased) | 6.00 (fixed) | ⚠ known-bad (cache bug) | [detail](fixedk-k6-lr1e4.md) |

`baseline-k6` re-validated 2026-06-23 (validation 500 ex, greedy); prior 39.50% was test (1319 ex, avg-5-sample; biased). `fixedk-k6-lr1e4` not re-evaluated - its number is on test (biased) and it carries a separate bias banner.

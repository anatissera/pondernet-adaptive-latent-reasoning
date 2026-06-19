# 01-simcot-baselines — Runs

See [experiment.md](experiment.md) for what's being tested.

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| baseline-k6 | upstream default lr=1e-3 | **39.50%** (avg 5) | 6.00 (fixed) | ✅ reference | [detail](baseline-k6.md) |
| fixedk-k6-lr1e4 | lr=1e-4 | 17.82% (avg 5) | 6.00 (fixed) | ⚠ known-bad (cache bug) | [detail](fixedk-k6-lr1e4.md) |

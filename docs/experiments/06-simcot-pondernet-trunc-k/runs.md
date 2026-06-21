# 06-simcot-pondernet-trunc-k — Runs

See [experiment.md](experiment.md) for what's being tested.

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| `trunc-ki-fullscope-g0.05-b1.5-ep5` | trunc-K (K_i=n_i) + adaptive prior + full scope | 36.32% @ thr0.8 (ep5) | 3.660 @ thr0.5 | complete | −4.2pp vs exp-05 on acc; avg_steps −0.705 (thr0.5); Spearman 0.596; accuracy still rising at ep5 — not converged; OOM at bs=32 forced bs=24 (eff-batch 96 vs intended 128) |

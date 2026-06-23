# 06-simcot-pondernet-trunc-k — Runs

See [experiment.md](experiment.md) for what's being tested.

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| `trunc-ki-fullscope-g0.05-b1.5-ep5` | trunc-K (K_i=n_i) + adaptive prior + full scope | 36.32% @ thr0.8 (ep5) | 3.660 @ thr0.5 | ✅ complete | −4.2pp vs exp-05 on acc; avg_steps −0.705 (thr0.5); Spearman 0.596; accuracy still rising at ep5 — not converged; OOM at bs=32 forced bs=24 (eff-batch 96 vs intended 128) |
| `trunc-ki-fullscope-g0.05-b1.5-ep10` | same + 10 epochs, eff-batch 128 (OOM+resume mid-run) | 36.54% @ thr0.5 / 35.94% @ thr0.8 (ep8) | 3.766 @ thr0.5 | ✅ complete | [detail](trunc-ki-fullscope-g0.05-b1.5-ep10.md) — accuracy plateau in mid-36%/mid-35%; Spearman 0.592; avg_steps@thr0.8 5.194 (−0.068 vs exp-05); strong monotonic step-vs-difficulty signal; backbone still converging at ep8 |

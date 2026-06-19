# 02-simcot-pondernet-early — Runs

See [experiment.md](experiment.md) for what's being tested. All runs here predate the
gradient-checkpointing fix and are superseded.

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| lr1e4 | lr=1e-4, thr=0.8 | 19.26% (avg 5) | — | ⚠ known-bad (cache bug) | [detail](lr1e4.md) |
| warmstart-lr1e4 | warm-started, lr=1e-4 | 15.24% (avg 5) | — | ⚠ known-bad (cache bug) | [detail](warmstart-lr1e4.md) |
| lr1e3 | lr=1e-3 | not evaluated | — | ⚠ known-bad (cache bug) | [detail](lr1e3.md) |

Dead runs archived (not migrated as keepers): `halthead-ep40` (never evaluated),
`joint-ep40` (~0%, random backbone), `joint-warmstart` (crashed) → `<dir>/archive/`.

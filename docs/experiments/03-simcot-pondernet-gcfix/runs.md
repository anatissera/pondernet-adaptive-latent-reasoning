# 03-simcot-pondernet-gcfix — Runs

See [experiment.md](experiment.md) for what's being tested (incl. the Root Cause writeup).

| run | key variable | best accuracy | avg steps | status | detail |
|-----|-------------|--------------|-----------|--------|--------|
| 100k | `--gradient_checkpointing False`, 100k data | **42.23%** @ ep2 (thr=0.8) | 5.97 | ✅ done | [detail](100k.md) |

Migrated from the old `simcot-pondernet-gcfix-100k` dir (renamed `gcfix-100k` → `100k`). The
earlier train-only `simcot-pondernet-100k` copy was superseded and archived
(`<dir>/archive/100k-pre-gcfix`).

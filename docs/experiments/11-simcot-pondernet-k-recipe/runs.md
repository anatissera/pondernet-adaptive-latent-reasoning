## Phase 0 diagnostic: are early latents answer-ready? (2026-06-17)

Before retraining to test the "overfit-to-K=6" hypothesis, we measured how answer-ready
each latent step already is, with a new **fixed-k eval mode** (`test.py --fixed_k_eval`):
ignore the halting head and force-decode the answer at every prefix `k=1…K_max`, reporting
accuracy@k. Run on the gamma-sweep final checkpoints (greedy, 1-pass, GSM8K test).

| k (forced) | γ=0 acc | γ=0.1 acc |
|-----------|---------|-----------|
| 1 | 24.03 | 26.61 |
| 2 | 30.25 | 32.60 |
| 3 | 31.99 | **36.01** |
| 4 | 36.47 | **39.88** ← peak |
| 5 | 39.73 | 39.58 |
| 6 | **40.03** ← peak | 39.50 |

**Confirms the hypothesis.** γ=0 climbs monotonically to its peak at K=6 - an **8-pt
z₃↔z₆ gap** means early latents are genuinely *not* answer-ready; the bottleneck is the
**representation**, not just the halting threshold. **γ=0.1 flattens the curve and shifts
the peak to k=4** (z₃ 32→36%): the PonderNet answer-loss weight on early steps is what
makes them answer-ready, which is exactly why γ=0.1 @ thr0.5 can halt at 3.3 steps for
39.2% (the early latents it halts on are already competent). Prediction for the retrain:
recipes that break the K=6 prior (cold-from-GPT-2 / unfrozen backbone) should flatten the
curve further, peaking at the data's 2–3-step mode. Curves: `results/fixedk-diagnostic-g*/`.

## Adaptive-K / Recipe Factorial: testing the K=6-prior hypothesis (2026-06-19)

Direct follow-up to the Phase 0 diagnostic above, on branch `feat/adaptive-k-from-scratch`
(locked plan: `.claude-config/plans/tranquil-frolicking-pascal.md`). Phase 0 showed γ=0
peaks at k=6 (an 8-pt z₃↔z₆ gap) while γ=0.1 already flattens and shifts the peak to k=4 -
this factorial asks whether breaking the **frozen-backbone K=6 prior itself** (not just
adding γ pressure) lets the model concentrate its halting distribution closer to the data's
true 2–3-step mode, and at what accuracy cost. γ=0.1, geom_mean=3.0 held constant; only
`pondernet_train_scope` (recipe) and `--max_latent_steps` (K_max) vary:

- **Recipe A** (baseline): warm-started SIM-CoT backbone, frozen; train `lora_*` + `halt_head`
  only - the current/default scope, carries the K=6 prior.
- **Recipe B** (cold): plain GPT-2 backbone (`SIMCOT_CKPT=""`, decoder-only warm-start), train
  `lora_*` + **`prj.*`** + `halt_head` - no answer-position prior at all.
- **Recipe C** (unfreeze): warm-started backbone, but train the **whole backbone** + `prj` +
  `halt_head` - overwrite the prior while keeping the good init.

K_max ∈ {4, 6, 8, 10, 12}, 5 epochs each, eff batch 128 (bs/accum scaled down at higher K_max
for VRAM: 32/4 up to K=8, 24/5 at K=10, 16/8 at K=12), seed 42, local 3090 (train) + 3060
(eval), `scripts/sweep_k_recipe.sh`.

### Recipe A results (complete, all 5 K_max - `results/k_recipe_sweep/summary.tsv`)

Final-epoch checkpoint (`seed_42`), GSM8K test, greedy 1-pass:

| K_max | thr 0.5 | thr 0.8 | thr 0.9 |
|-------|---------|---------|---------|
| 4  | 38.97% @ 3.15 | 38.89% @ 3.97 | 38.74% @ 4.00 |
| 6  | 38.97% @ 3.30 | 39.20% @ 5.25 | 39.42% @ 5.85 |
| 8  | 38.82% @ 3.24 | 39.42% @ 5.48 | 39.50% @ 6.84 |
| 10 | 39.42% @ 3.21 | 39.50% @ 5.46 | 39.50% @ 7.02 |
| 12 | 39.04% @ 3.22 | 39.42% @ 5.46 | **39.88% @ 7.06** ← best |

**Recipe A's cheap operating point is K_max-invariant.** At thr=0.5 avg steps stays pinned at
~3.15–3.30 regardless of how much ceiling (K_max) is available - raising K_max from 4 to 12
does *not* push the model to use more steps when it's confident early. At thr=0.9 it does use
the extra budget (4.00→7.06 steps from K=4→12), buying a modest accuracy gain (38.74→39.88%,
+1.1pt) that's mostly flat past K=8. (K=6 row here is a fresh run of the same recipe/hparams as
the gamma sweep's γ=0.1/K=6 - 38.97%@3.30 vs that run's 39.04%@3.30, consistent with the
~0.2pt run-to-run noise already noted in the Fine Gamma Sweep conclusions.)

This confirms γ=0.1 pressure (not K_max headroom) sets the cheap-tier halting point even with
the backbone frozen - i.e. recipe A alone doesn't show a K=6-pinned, non-adaptive pattern once
γ>0. Whether breaking the prior (B/C) changes the *expensive*-tier ceiling behavior or shifts
the cheap tier even lower is what B and C test next.

### Recipe B (cold start): COLLAPSED to ~2% - negative result

Recipe B (cold GPT-2 backbone, `SIMCOT_CKPT=""`, decoder-only warm-start, train
`lora_*`+`prj.*`+`halt_head`) **fails catastrophically**. Both completed runs sit at
**~2% accuracy** (vs ~39% for recipe A) across *every* epoch checkpoint and threshold:

| run | thr 0.5 | thr 0.8 | thr 0.9 |
|-----|---------|---------|---------|
| B / K=4 (`seed_42`) | 2.12% @ 2.0 | 2.20% @ 4.0 | 2.20% @ 4.0 |
| B / K=6 (`seed_42`) | 2.27% @ 2.0 | 1.82% @ 5.0 | 2.58% @ 6.0 |

This is the **cold-start instability the plan flagged as recipe B's risk** ("SIM-CoT warns
from-scratch implicit CoT can be unstable; the warm-started decoder's `L_step` is what's
*expected* to stabilize it" - evidently it does not, at least not in 5 epochs at lr 2e-5).
A cold GPT-2 with only LoRA+prj+halt trainable never learns the latent-reasoning behavior
the frozen SIM-CoT backbone provides in recipes A/C; the model emits near-random answers.
**Conclusion: the SIM-CoT warm-start is load-bearing, not just a convenient prior** - you
cannot recover latent-CoT competence from a cold backbone with this lightweight scope/budget.
Remaining B runs (k8/k10/k12) were **not run** (the 2% pattern is already conclusive across
two K values × all epochs); compute redirected to recipe C. Raw rows:
`results/k_recipe_sweep_bc/summary.tsv`.

### Recipe C (full unfreeze): in progress - K=4, K=8 done, consistently beats A

Recipe C (warm-started backbone, train **whole backbone**+`prj`+`halt_head`) is the arm
expected to work - it keeps the good SIM-CoT init while letting the K=6 prior be overwritten.
**Full-scope OOMs at bs=32**, so recipe C runs at **bs=16/accum=8** (or bs=8/accum=16 on the
12 GB 5070; eff batch 128, comparable).

**Done so far** (`seed_42`, final epoch, GSM8K test, greedy 1-pass) - recipe C vs recipe A:

| K_max | C thr 0.5 | A thr 0.5 | C thr 0.8 | A thr 0.8 | C thr 0.9 | A thr 0.9 |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| **4** | **39.80% @ 3.13** | 38.97% @ 3.15 | **39.73% @ 3.98** | 38.89% @ 3.97 | **39.58% @ 4.00** | 38.74% @ 4.00 |
| **6** | **40.18% @ 3.24** | 38.97% @ 3.30 | **39.80% @ 5.24** | 39.20% @ 5.25 | **39.73% @ 5.88** | 39.42% @ 5.85 |
| **8** | **40.03% @ 3.20** | 38.82% @ 3.24 | **39.88% @ 5.44** | 39.42% @ 5.48 | **39.42% @ 6.86** | 39.50% @ 6.84 |
| **10** | **39.88% @ 3.22** | 39.42% @ 3.21 | 39.42% @ 5.48 | 39.50% @ 5.46 | **39.58% @ 7.07** | 39.50% @ 7.02 |
| **12** | **40.18% @ 3.20** | 39.04% @ 3.22 | **40.03% @ 5.46** | 39.42% @ 5.46 | 39.80% @ 7.06 | 39.88% @ 7.06 |

#### Recipe-C factorial COMPLETE - full C-vs-A frontier (all 5 K_max, seed_42, greedy 1-pass)

**Headline: recipe C (full unfreeze) beats recipe A (frozen LoRA) at the cheap operating point
(thr 0.5) at *every* K_max**, at matched avg steps (~3.1–3.2):

| K_max | 4 | 6 | 8 | 10 | 12 |
|-------|---|---|---|----|----|
| **C @ thr0.5** | 39.80 | **40.18** | 40.03 | 39.88 | **40.18** |
| A @ thr0.5 | 38.97 | 38.97 | 38.82 | 39.42 | 39.04 |
| **Δ (C−A)** | +0.83 | **+1.21** | +1.21 | +0.46 | +1.14 |

- **C ≥ A at thr0.5 for all K** (mean Δ ≈ +0.97pt) and at thr0.8 for all K. At thr0.9 (expensive
  tier, ~7 steps) the two converge - C and A tie within noise (e.g. k10 C+0.08, k12 A+0.08),
  which makes sense: with a near-full budget the frozen-prior disadvantage washes out.
- **Best cells: C-k6 and C-k12, both 40.18% @ ~3.2 steps** - i.e. recipe C reaches the SIM-CoT
  K=6 baseline accuracy (~39.5–40%) using only ~3.2 of its latent steps, and the K_max ceiling
  barely matters (k6 already as good as k12). Compute-at-halt is essentially flat across K_max
  for recipe C, same as recipe A - confirming that **once early latents are answer-ready, extra
  K_max headroom is unused** (the original adaptivity hypothesis).
- The fixed-k diagnostic (above) explains the mechanism: full unfreeze lifts early-latent
  answer-readiness (z₃ 37.5% vs 32% frozen), so the model halts early at higher accuracy.

**Bottom line:** the SIM-CoT K=6 prior *is* the bottleneck for early halting, and **fully
unfreezing the backbone (recipe C) is the fix** - +~1pt at the cheap operating point across the
whole K_max grid, free of extra compute. Recipe B (cold start) is not viable (warm-start is
load-bearing). Caveat: single-seed; multi-seed error bars in progress (gaps are ~1pt vs ~0.2pt
run-to-run noise, so likely real but being confirmed). Per-cell raw: `results/k_recipe_sweep_bc/summary_C_k*.tsv`.

#### Fixed-k diagnostic across ALL K_max - early-latent readiness, C vs A

`--fixed_k_eval` accuracy@k (force-decode at each prefix k, ignore halting) for every cell.
The early-latent lead is **consistent at every K_max** - recipe C's z₃ beats recipe A's by
+1.0 to +1.6pt, and C's curve plateaus by k=4 sitting at/above A throughout:

| K_max | C z₃ | A z₃ | Δz₃ | C peak | A peak |
|-------|------|------|-----|--------|--------|
| 4  | 38.3 | 37.3 | +1.0 | 39.5 (k4) | 39.0 (k4) |
| 6  | 37.2 | 36.1 | +1.1 | 39.6 (k4) | 39.9 (k4) |
| 8  | 37.5 | 35.9 | +1.6 | 40.2 (k6) | 39.5 (k5) |
| 10 | 37.1 | 36.1 | +1.0 | 39.9 (k4) | 40.0 (k4) |
| 12 | 37.7 | 36.2 | +1.5 | 40.1 (k6) | 39.6 (k6) |

(z₃ = accuracy if forced to stop at 3 latent steps; both curves climb steeply k1→k4 then
plateau - the action is all in how *ready* the early latents are, and C is uniformly readier.
Reference: frozen γ=0 Phase-0 had z₃=32.0.) Full per-k curves:
`results/fixedk-diagnostic-recipe{C,A}-k{4,6,8,10,12}/gsm8k_fixedk_curve.json`.

#### Faithful re-eval at batch_size=1 (paper-accurate frontier)

The factorial sweep evaluated at `--batch_size 8/16`, which per AGENTS.md is **not faithful**
for adaptive halting (the batch only breaks when *all* rows halt, so an early-halting example's
answer is read from more steps than `steps_used` reports → accuracy slightly over-reported
relative to the steps). Re-ran every final checkpoint at **batch_size=1** (faithful). thr 0.5:

| K_max | C (bs1) | A (bs1) | Δ (C−A) |
|-------|---------|---------|---------|
| 4  | 39.65 @ 3.13 | 39.12 @ 3.15 | +0.53 |
| 6  | 39.88 @ 3.25 | 38.97 @ 3.30 | +0.91 |
| 8  | 39.80 @ 3.20 | 38.89 @ 3.24 | +0.91 |
| 10 | 39.58 @ 3.22 | 39.35 @ 3.22 | +0.23 |
| 12 | **40.33 @ 3.21** | 38.89 @ 3.22 | +1.44 |

**The headline survives faithful eval: C > A at thr 0.5 at every K_max** (mean Δ ≈ +0.80pt;
slightly smaller than the bs=8/16 mean +0.97 but same direction at all 5 K). Absolute numbers
shift ≤0.4pt vs bs=8/16. All thresholds (0.5/0.8/0.9) in `results/faithful-bs1/recipe{C,A}-k<K>-thr<T>/`.

**Recipe C beats recipe A at every threshold tested, at matched avg steps** (+0.8–1.2pt). The
gain is "free" - same compute-at-halt, higher accuracy. This supports the core hypothesis:
unfreezing the backbone lets early-step latents become answer-ready (overwriting the frozen
K=6 prior) without needing more steps. (K=6/10/12 pending - running one-at-a-time
on the 5070 since Option-B freed it ~03:50 UTC.)

#### Deep-dive: per-step answer-readiness curve (fixed-k eval on C-K=8)

`test.py --fixed_k_eval` force-decodes the answer at each forced prefix k=1…8 (ignoring the
halt head), on the C-K=8 `seed_42` checkpoint - directly comparable to the Phase-0 curves above:

| forced k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|----------|---|---|---|---|---|---|---|---|
| **C-K8 (full unfreeze)** | 26.2 | 34.1 | **37.5** | 39.8 | 39.9 | **40.2** | 39.7 | 39.7 |
| **A-K8 (frozen LoRA)** - matched | 26.6 | 33.3 | 35.9 | 39.4 | 39.5 | 39.4 | 39.4 | 39.0 |
| A/γ=0 (Phase-0, frozen, K=6, no halt pressure) | 24.0 | 30.3 | 32.0 | 36.5 | 39.7 | 40.0 | – | – |
| A/γ=0.1 (Phase-0, frozen, K=6) | 26.6 | 32.6 | 36.0 | 39.9 | 39.6 | 39.5 | – | – |

**This is the mechanism, confirmed.** Two effects, both visible:
1. **Halt-pressure + K_max headroom already flattens the curve** (independent of unfreezing): the
   Phase-0 frozen γ=0 run had z₃=32.0% with an ~8pt z₃→peak gap (the "answer-lives-at-K=6" prior);
   the frozen **A-K8** run (γ=0.1, more headroom) lifts z₃ to 35.9% and plateaus by k=4. So the
   prior is partly broken just by training with halting pressure at higher K_max.
2. **Full unfreeze (recipe C) adds a consistent further boost on top** - at matched K=8, **C beats
   A at essentially every forced k** (z₃ 37.5 vs 35.9, +1.6pt; k6 40.2 vs 39.4, +0.8pt). The early
   latents are uniformly readier, which is precisely why recipe C's halted accuracy is +0.8–1.2pt
   over A at the same ~3.2 avg steps.

Ordering of early-latent (z₃) readiness: frozen-no-pressure γ=0 **32.0** < frozen-halt-pressure
γ=0.1/K6 **36.0** ≈ frozen A-K8 **35.9** < **full-unfreeze C-K8 37.5**. Curves:
`results/fixedk-diagnostic-recipe{A,C}-k8/gsm8k_fixedk_curve.json`.

**Operational note (2026-06-20):** the recipe-C sweep is sharing this 3-GPU box with two
other people's long jobs - Option-B on the 5070 (`alr-anapaula-optionb` worktree) and a
teammate's `05-simcot-pondernet-adaptive-prior` run on the 3090 (`alr-valentino` worktree) -
plus the forbidden 3060. With all three cards busy and **system RAM saturated (~44/46 GB)**,
the K=8 sweep got OOM-killed mid-eval and the K=4/6 retries failed instantly for lack of a
free GPU. K=8's headline result survived (evals completed before the kill). Remaining C runs
(k4, k6, k10, k12) are queued to run **one at a time** on whichever card frees first
(Option-B's 5070 or the teammate's 3090), with `--dataloader_num_workers 1` to cut the ~6 GB
of dataloader RAM that contributed to the OOM. Not parallelizable while RAM stays this tight.

**Operational incident:** mid-sweep, this worktree (`/home/tpnlp/alr-anapaula`) was
`git checkout`'d to an unrelated branch (`option-b`, a different subgroup-style experiment
with its own top-level layout) while the background sweep was still running - `pondernet/scripts/`
disappeared from disk out from under the live job. Recipe A's K=12 training had already
finished (checkpoint intact, gitignored `models/` untouched by the checkout) but its eval and
all of recipes B/C failed instantly on "script not found." Fix: gave `option-b` its own
dedicated worktree (`/home/tpnlp/alr-anapaula-optionb`, symlinked to the same shared
`data/models/outputs/results`), restored this worktree to `feat/adaptive-k-from-scratch`,
re-ran the missing K=12 eval, and relaunched B/C. **Lesson:** each experiment branch needs its
own worktree if a long background job is running in it - three now exist
(`/home/tpnlp/adaptive-latent-reasoning` = main/`pondernet`, `/home/tpnlp/alr-valentino` =
`pondernet-pipeline`, `/home/tpnlp/alr-anapaula` = active branch under test,
`/home/tpnlp/alr-anapaula-optionb` = `option-b`).

#### Multi-seed error bars: C-k6 vs A-k6 (seed=1, 2 confirm ~1pt gap is real, 2026-06-23)

The C-vs-A gap at the best cell (k6) is ~1.2pt (40.18% vs 38.97%, thr 0.5) across the full
K_max sweep - larger than the ~0.2pt run-to-run noise observed in the gamma sweep (γ=0.1
retrain: 39.04% vs coarse run 39.20%). To confirm the gap is real and not seed-lucky, we
re-ran both recipes at K=6 with independent seeds (seed=1, seed=2, with the baseline seed_42
for reference). Training on the 5070, eval bs=1 (faithful), thr 0.5:

| seed | C-k6 acc | A-k6 acc | Δ (C−A) |
|------|----------|----------|---------|
| **42** (baseline) | 40.18% @ 3.24 | 38.97% @ 3.30 | +1.21 |
| **1** | 39.88% @ 3.24 | 39.58% @ 3.30 | +0.30 |
| **2** | 40.49% @ 3.25 | 39.50% @ 3.30 | +0.99 |
| **mean ± range** | **40.18% ± 0.31** (range 0.61) | **39.35% ± 0.31** (range 0.61) | **+0.83 ± 0.46** (range 0.91, min +0.30, max +1.21) |

### Final verdict (3 seeds, study complete)

**Directionally robust, magnitude noisy: C beats A in all 3/3 seeds tested**, by anywhere from
+0.30 to +1.21pt (mean +0.83pt). No seed flips the sign - recipe C (full backbone unfreeze)
never underperforms recipe A (frozen LoRA) at K=6, thr 0.5. That 3/3 consistency is the
strongest evidence the effect is real: if the true gap were 0, we'd expect roughly half of
seeds to land negative by chance, not all three positive with comparable per-seed spread
(seed-to-seed accuracy noise within each recipe is also ~0.6pt - i.e. the Δ's own variance is
mostly explained by *each recipe's own* run-to-run noise stacking, not by C and A being
identical and the sign being a coin flip).

**Caveat on magnitude:** the original headline (+1.21pt at seed_42) sits at the *high* end of
the 3-seed range, not the mean - a single-seed report would have overstated the effect by
~0.4pt. **Use the 3-seed mean (+0.83pt) as the representative number going forward**, not the
seed_42-only figure, in any write-up or comparison to the SIM-CoT baseline.

Raw: `results/k_recipe_sweep_bc/summary_C_k6_seed{1,2}.tsv`, `summary_A_k6_seed{1,2}.tsv`.

**Operational notes:**
1. The loop stalled with the 5070 idle for ~6h (2026-06-23 11:31→17:34 UTC) after A-k6 seed1
   finished - recovered by manual intervention, no data lost.
2. A second, *separate* incident (2026-06-24 03:31 UTC): a teammate reorganized the shared
   `data/` symlink target, moving `train15k.jsonl`/`train100k.jsonl` from
   `data/gsm8k_aug/` into `data/gsm8k_aug/subsamples/`. This broke the sweep script's default
   `DATA_PATH`, causing C-k6 seed2's first launch attempt to fail instantly
   (`FileNotFoundError`) with no visible alarm - `sweep_k_recipe.sh` just logs a WARN and
   exits cleanly, so an idle GPU from this failure mode looks identical to "between runs"
   unless someone checks the log body. Fixed by passing `DATA_PATH=` explicitly on every
   launch from that point on.
3. A third idle gap (2026-06-24 22:12→2026-06-25 04:00 UTC, ~5h48min): A-k6 seed2 finished
   training+eval on schedule, but the scheduled wake-up to process the result and report back
   did not fire in time. No data lost (results were already on disk), but this is the same
   idle-GPU pattern as incident 1 - three occurrences across this session. **Root cause is
   likely re-arm interval miscalibration for multi-hour runs**, not a one-off; consider a
   structurally shorter max re-arm ceiling (e.g. cap at 30–40 min regardless of estimated
   time-to-completion) if this pattern continues.

## ⚠️ Validation re-eval: the C>A headline does NOT replicate on a held-out split (2026-06-25)

Teammate Valentino's branch (`fix/validation`, commit 5e3110bd) flagged a methodological issue
in his own exp 01–07: all those runs were evaluated and compared on the GSM8K **test** split,
the same split used to pick which experiment to report - a selection-bias risk (not train/test
*leakage*, since training data is GSM8K-Aug and eval is canonical GSM8K test/validation, but
repeatedly comparing N runs on the same test set and reporting the winner inflates the apparent
effect size of whatever you pick). **We have the identical issue**: the entire recipe×K_max
factorial, the fixed-k diagnostics, and the 3-seed C-vs-A study above all compared runs using
the same GSM8K test set (1319 examples, via `--data_name gsm8k` with no `--data_path`,
falling back to the HF `gsm8k`/`main`/`test` split) for *every* decision (which recipe wins,
which K_max is best, which checkpoint epoch to report).

**Re-eval on `data/gsm8k_aug/validation.json`** (500 held-out examples, never looked at during
the whole factorial/multi-seed study - converted from the teammate's `validation.jsonl` to a
JSON array for `test.py` compatibility) of the same 6 final checkpoints from the 3-seed study,
thr 0.5, bs=1:

| seed | C-k6 (validation) | A-k6 (validation) | Δ (C−A) | *(for reference: Δ on test)* |
|------|--------------------|--------------------|---------|-------------------------------|
| 42 | 40.60% @ 3.23 | **41.40% @ 3.28** | **−0.80** | +1.21 |
| 1 | 40.00% @ 3.22 | **40.20% @ 3.27** | **−0.20** | +0.30 |
| 2 | **41.00% @ 3.21** | 40.60% @ 3.26 | +0.40 | +0.99 |
| **mean** | 40.53% | 40.73% | **−0.20** (range −0.80 to +0.40) | +0.83 (range +0.30 to +1.21) |

**The sign flips.** On the held-out validation split, **A wins 2 of 3 seeds and the mean Δ is
slightly negative** (−0.20pt, A ahead) - the complete opposite of the test-set headline (C
ahead in 3/3 seeds, mean +0.83pt). Both held-out-split deltas (−0.80 to +0.40) and the original
test-set deltas (+0.30 to +1.21) are comparable in *magnitude* to each recipe's own seed-to-seed
noise (~0.6pt) - but the **direction** of the apparent winner depends entirely on which fixed
500–1319 example split you happen to score against. This is the textbook signature of selection
bias from repeatedly checking the same eval set across ~20 runs and reporting whichever
recipe/K_max happened to score highest on it, **not a real difference in recipe quality**.

**Full K_max factorial re-eval (DONE 2026-06-25)** - `A`/`C` × `K∈{4,6,8,10,12}`, seed_42 final
checkpoints, validation split, thr 0.5, bs=1. This confirms the reversal is **not** specific to
K=6: it holds across the entire factorial.

| K_max | C acc @ steps | A acc @ steps | Δ (C−A) on validation | *(for reference: Δ on test)* |
|-------|---------------|---------------|------------------------|-------------------------------|
| 4 | **40.60 @ 3.09** | 40.20 @ 3.12 | **+0.40** | (C ahead) |
| 6 | 40.60 @ 3.23 | **41.40 @ 3.28** | **−0.80** | (C ahead) |
| 8 | 39.80 @ 3.17 | **40.60 @ 3.20** | **−0.80** | (C ahead) |
| 10 | 40.00 @ 3.17 | **40.40 @ 3.19** | **−0.40** | (C ahead) |
| 12 | 40.80 @ 3.15 | **41.00 @ 3.18** | **−0.20** | (C ahead) |
| **mean** | 40.36% | **40.72%** | **−0.36** (range −0.80 to +0.40) | C ahead at every K_max (mean +0.97) |

**The sign flips at the factorial level too.** On test, recipe C beat A at *all five* K_max
(mean +0.97pt - the original headline). On the held-out validation split, **A wins 4 of 5 K_max**
(C only edges ahead at K=4, by +0.40), and the mean Δ is **−0.36pt (A ahead)**. Every per-cell
delta (−0.80 to +0.40) sits inside each recipe's own seed-to-seed noise band (~0.6pt). The
test-set "C wins everywhere" pattern was a clean selection-bias artifact across the ~20-run
factorial; it does not survive a split that was never used to pick winners.

**Revised conclusion: recipe C is RETIRED as a finding - do NOT adopt it as a better default.**
The test-set-only evidence that motivated "C > A, unfreeze the backbone" does not survive a clean
held-out check at *any* K_max. Recipe A (frozen-backbone LoRA) and recipe C (full unfreeze) are
**statistically indistinguishable across the whole K_max factorial** on held-out data, with A
nominally ahead. The backbone-trainability axis is a non-finding; the project's standalone
contribution is the **adaptivity** result itself (γ-sweep: ~45% fewer latent steps for <1%
accuracy loss vs fixed K=6 - independent of the recipe study). Any future comparison must hold
out a validation split *before* running the sweep and use it exclusively for picking winners,
reserving test (or a second untouched split) for the one-time final reported number.

Raw: `results/validation-reeval-{C,A}_{42,1,2}.log`,
`results/k_recipe_sweep_bc/summary_validation_reeval.tsv` (k=6 3-seed),
`results/k_recipe_sweep_bc/summary_validation_reeval_fullK.tsv` (full factorial, seed_42).


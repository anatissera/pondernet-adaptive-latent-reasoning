# exp-10 — GCP: the full 40-epoch from-scratch run on a Spot RTX PRO 6000

**Status (2026-07-11): complete.** 40/40 epochs trained and evaluated. Final accuracy
**18.88%** on the GSM8K test set (vs. 39.5% SIM-CoT warm-start baseline) — see § Results below
for the full curve and what it means. Both VMs are stopped.

The plan changed once the hardware was measured. Resuming from the epoch-12 hand-off checkpoint
was only ever attractive because 28 epochs on the L4 meant 13.5 days. At the measured
1.01 h/epoch, the *entire* 40-epoch run costs 40.4 h — so we pay 12 extra hours and get a run
with no inherited optimizer state, no rewound LR schedule, and nothing to caveat: exactly the
"40 epochs, one continuous cosine" that CODI/SIM-CoT report. The resume analysis below is kept
because it explains why that trade is worth taking.

Everything below was measured, not estimated. Where an earlier estimate was wrong, the wrong
number is kept alongside the measurement, because the size of the error is the lesson.

---

## The bottom line

| question | answer |
|---|---|
| Full from-scratch run (40 epochs) | **40.4 h** |
| Epochs reached in 24 h from cold start | **~24** (23.8 epochs fit in 24 h) |
| Resume ep12→40 (28 epochs), if it had been used | 28.2 h |
| Measured step time (RTX PRO 6000, BS=64 ACCUM=2) | **1.21 s/step** |
| Measured step time (L4, BS=16 ACCUM=8) | **13.84 s/step** |
| Speedup | **11.4×** |
| Same 28 epochs on the L4 | 13.5 days |

---

## Hardware and quota

The A100 quota requests are still sitting at `grantedValue: 0` (both
`NVIDIA-A100-80GB-GPUS` and the committed variant). Spot quota is a **separate queue** from
on-demand — `GPUS_ALL_REGIONS` only caps on-demand GPUs — and the Spot RTX PRO 6000 request
was granted **instantly**:

| quota (us-central1) | requested | granted |
|---|---|---|
| `PREEMPTIBLE-NVIDIA-RTX-PRO-6000-GPUS` | 1 | 1 ✅ |
| `PREEMPTIBLE-CPUS` | 48 | 48 ✅ |
| `SSD-TOTAL-GB` | 1000 | 300 ⏳ (manual review) |

> **Spot GPU quota needs Spot CPU quota too.** Both start at 0. Granting only the GPU leaves
> you unable to boot the VM that carries it.

**VM:** `alr-exp10-spot`, `g4-standard-48` (1× RTX PRO 6000 Blackwell, 96 GB, 48 vCPU,
180 GB RAM), zone `us-central1-b`. While stopped it costs nothing and the boot disk keeps the
venv, the data and every checkpoint.

Two platform constraints found the hard way:

- **G4 rejects `pd-balanced`.** Boot disk must be `hyperdisk-balanced`.
- **`SSD_TOTAL_GB` is 300 GB/region** and the old `exp10-l4` disk already spends 200 GB of it.
  Solved by dropping the separate checkpoint disk: a 100 GB boot disk has 73 GB free after the
  image, venv (6.8 GB) and GSM8k-Aug (97 MB), which holds all 40 checkpoints (~48 GB).

---

## The Blackwell trap (fixed, and verified on the card)

The RTX PRO 6000 is compute capability **sm_120**. `torch==2.7.1` from PyPI is built against
CUDA 12.6, whose wheels ship **no kernels for sm_120** — every CUDA op dies with
`no kernel image is available for execution on the device`. The L4 (sm_89) never showed this.

Fixed by pinning the cu128 index in `pyproject.toml` (`uv.lock` regenerated →
`torch 2.7.1+cu128`). Verified on the actual GPU:

```
gpu: NVIDIA RTX PRO 6000 Blackwell Server Edition sm_120 torch 2.7.1+cu128
cuda ok
```

`vm_bootstrap.sh` re-runs that assertion on every fresh setup, so this fails in seconds rather
than mid-run.

---

## Batch size: what memory is actually spent on

I claimed early on that VRAM was irrelevant here because the model is only ~250 M params
(≈4–5 GB of weights + Adam state). **That was wrong**, and measurement shows why: with
`K_max=12` the model decodes an answer after *every* latent prefix, so it retains twelve
separate answer-decode graphs per example. **Activations dominate, not weights.**

| BS × ACCUM | peak memory | result |
|---|---|---|
| 128 × 1 | ~116 GB needed | **OOM** (died at 92.7/95.6 GB requesting 4.48 GiB) |
| **64 × 2** | **62.8 / 95.6 GB** | ✅ **1.21 s/step**, GPU util 51% |

`BS=64 ACCUM=2` is the ceiling and is now the default. The effective batch stays 128, matching
exp-08 Run C. The 96 GB card is *not* oversized for this run — it is what makes `BS=64` possible
at all (the L4 was stuck at `BS=16 ACCUM=8`, i.e. 8 accumulation passes, each serializing the
12-step latent loop).

GPU utilisation sits at 51%: the serial latent loop, not the dataloader, is the limiter
(`--dataloader_num_workers` was raised 4→16 to rule the loader out).

> **`profile_batch_size.py` is misleading for this run.** It profiles `SEQ_LEN=256`, `K=6`,
> *with* gradient checkpointing. This run is 512 / K=12 / grad-ckpt **False** (mandatory — see
> the KV-cache gotcha in `AGENTS.md`). Use `scripts/gcp/measure_throughput.sh`, which drives
> the real wrapper.

---

## Timing, measured

Step time comes from the actual `train.log` of the L4 runs (recovered off the old VM's disk),
and from a live 200-step probe on the RTX PRO 6000.

- L4 probe (`probe-lr1e-3-wu0.05-ep4`): 11,996 steps in **46 h 07 m** → 13.84 s/step.
- L4 40-epoch run: 7,257 steps in **27 h 53 m** → 13.84 s/step. (Consistent.)
- RTX PRO 6000, BS=64 ACCUM=2: **1.21 s/step**, stable from step ~40 through 203.

At 2,999 steps/epoch:

| | h/epoch | ep12→40 (28 ep) | epochs in 24 h |
|---|---|---|---|
| RTX PRO 6000 | **1.01** | **28.2 h** | **23.8** → reaches ~ep36 |
| L4 | 11.53 | 323 h (13.5 d) | 2.1 |

An earlier estimate in this conversation put the RTX PRO 6000 at 1.5–3.5 s/step. The true
number is 1.21 s/step. The estimate was ~2× pessimistic — and the L4 baseline it was implicitly
calibrated against was off by 4–9×. **Do not plan from FLOPs; run `measure_throughput.sh`.**

Spot preemption costs at most one epoch (~1 h), since checkpoints are per-epoch.

---

## Resuming: why weights-only is not a resume

`transformers/trainer.py::_load_optimizer_and_scheduler` restores state **only if
`optimizer.pt` *and* `scheduler.pt` both exist** (line ~3603); otherwise it returns silently.
And the resume path fast-forwards only the *dataloader* (line ~1487), never the LR scheduler.

So loading the 556 MB weights-only tar into a resume does two things at once, without warning:

1. Adam's moments are zeroed.
2. The cosine scheduler rewinds to step 0.

Numerically, with 2,999 steps/epoch, 119,960 total steps and a 5,998-step (≈2 epoch) warmup:

| | correct | weights-only |
|---|---|---|
| LR at resume (ep12, step 35,988) | 8.386e-4, decaying | 0 → re-warms to peak 1e-3 over 2 epochs |
| LR at ep40 | ~0 | **2.265e-4** |
| cosine traversed | 100% | 70% |

The two faults partially cancel — the accidental re-warmup shields the cold Adam start — but
two harms remain. The model is driven back to peak LR on partly-converged weights, in exactly
the regime where this recipe already diverged once (`grad_norm` 20 → 2.5e11 at 3e-3). And it
**never anneals to zero**, forfeiting the final low-LR polish that the CODI/SIM-CoT baseline
gets. exp-10 exists to defeat the objection *"your accuracy came from someone else's
checkpoint"*; do not trade it for *"your LR schedule was not the one you reported."*

### What the local checkpoints are worth

`models/checkpoints/10-simcot-pondernet-fromscratch/` holds 6 checkpoints, all with full
optimizer + scheduler state. **None is usable as a starting point:**

- `probe-lr1e-3-wu0.05-ep4/checkpoint-{2999,5998,8997,11996}` — from a **4-epoch cosine**
  (`max_steps=11996`), already annealed to LR ≈ 1e-9. Their weights are not on the 40-epoch
  trajectory. Resuming one inside a 40-epoch run jumps LR from ~0 back to ~1e-3, undoing the
  anneal.
- `fromscratch-runC-.../checkpoint-{2999,5998}` — from the **LR=3e-3 run that diverged**.

The old `exp10-l4` VM disk was mounted and searched: it holds the same 6 checkpoints and no
`checkpoint-35988`. **The epoch-12 checkpoint exists only in the hand-off tar.** The probe's
value was validating the LR recipe, and it did that.

### Two modes in `vm_bootstrap.sh`

- **`RESUME_MODE=exact`** (default) — needs the 1.2 GB checkpoint. Drops it into `SAVE_DIR`;
  `get_last_checkpoint()` finds it and HF restores weights, moments, scheduler, RNG and
  `global_step`. 40 epochs total, original cosine. It **fails loudly** if `optimizer.pt` is
  absent, so you cannot silently get the broken resume.
- **`RESUME_MODE=weights`** — fallback. Does *not* pretend to resume: warm-starts via
  `--simcot_ckpt` (a plain `load_state_dict`) and runs a deliberate, self-consistent 28-epoch
  schedule — peak LR **8.386e-4** (what the original cosine held at ep12), short warmup to
  rebuild Adam's second moments, cosine to ~0 at ep40. **An approximation of the tail, not the
  tail.** If you use it, say so in this document.

> A latent bug, fixed: the run saves `pytorch_model.bin` (`--save_safetensors False`) but
> `--simcot_ckpt` loads via `safetensors.load_file()`. `weights` mode would have crashed on
> arrival. The bootstrap now converts the checkpoint first.

---

## Runbook

```bash
# 1. create (already done once; the VM exists and is STOPPED)
bash scripts/gcp/launch_spot.sh

# 2. push the repo (already done; it lives on the boot disk)
gcloud compute scp --zone=us-central1-b --recurse . alr-exp10-spot:/opt/alr --compress

# 3a. setup only — venv + CUDA assertion + data  (already done)
gcloud compute ssh --zone=us-central1-b alr-exp10-spot -- \
  'SETUP_ONLY=1 CKPT_DIR=/opt/alr/ckpt bash /opt/alr/scripts/gcp/vm_bootstrap.sh'

# 3b. when the 1.2GB checkpoint is in hand: start training
gcloud compute ssh --zone=us-central1-b alr-exp10-spot -- \
  'RESUME_MODE=exact CKPT_DIR=/opt/alr/ckpt bash /opt/alr/scripts/gcp/vm_bootstrap.sh'

# 4. survive preemptions (run locally, leave it open)
bash scripts/gcp/watchdog.sh
```

`vm_bootstrap.sh` installs `alr-train.service` (`Restart=always`, `enabled`), so training
restarts by itself whenever the VM boots. `watchdog.sh` does the booting: a preempted Spot VM
lands in `TERMINATED` and GCE will *not* restart it on its own (automatic restart covers host
errors, not preemption), so the loop polls and presses start, backing off through
`ZONE_RESOURCE_POOL_EXHAUSTED`.

**Costs run while the VM exists and is RUNNING.** Stop it whenever you are not training.

## Per-epoch evaluation, on a second GPU in a second project

The accuracy-vs-epoch curve needs all 40 epochs scored, but eval cannot share the training GPU:
the trainer saturates the RTX PRO 6000, and `AGENTS.md` mandates `--batch_size 1` (test.py only
breaks the latent loop once *every* example in the batch has halted, so batch>1 reports
`steps_used` from the batch-termination prefix, not from each example's own halt). Batch-1 eval
of 1319 examples cannot be batched away, so it gets its own card.

**Topology.** Training (project `adaptative-latent-reasoning`) publishes each epoch checkpoint
to `gs://alr-exp10-ckpts-244544686610`. An L4 (`alr-eval-l4`, project `tp-final-rl-kv-eviction`)
drains the bucket and publishes `summary.json` back. Only `pytorch_model.bin` crosses (578 MB):
`test.py` loads weights from `--ckpt_dir` but takes the tokenizer from `--model_name_or_path`
(`gpt2`, off the hub).

**Cost control.** The L4 is on-demand (~$0.71/h; the Spot CPU quota request is in manual review)
and bills for every RUNNING second, so it stays TERMINATED. `eval_orchestrator.sh`, a systemd
unit on the always-up training VM, starts it once `BATCH=6` checkpoints have piled up
unevaluated. The L4's startup-script drains everything pending and calls `poweroff`, returning
it to TERMINATED. **Measured: 234 s per epoch eval**, so a 6-epoch drain is ~24 min of L4 time;
all 40 epochs cost ~2.6 h ≈ **$2**, versus ~$30 for leaving it up for the whole run.

Cross-project IAM (all verified by actually reading and writing, not assumed):

| principal | grant | why |
|---|---|---|
| eval VM SA `928647578557-compute@…` | `roles/storage.objectUser` on the bucket | read checkpoints, write summaries (`objectViewer` alone silently blocks the write-back) |
| training VM SA `244544686610-compute@…` | `roles/compute.instanceAdmin.v1` on the eval project | `instances start` |
| training VM SA | `roles/iam.serviceAccountUser` on the eval SA | `instances start` needs `actAs` on the attached SA |

Two ordering hazards, both handled: a `.ready` marker is written *after* the weights (GCS makes
single-object writes atomic, never multi-object ones), and the uploader waits for the file size
to settle because HF creates the checkpoint directory before finishing the write.

### Results — run complete (40/40 epochs, 2026-07-11)

Training ran 38h 59m on the Spot RTX PRO 6000 (predicted 40.4h; no preemptions occurred, so
the difference is measurement noise around the 1.01h/epoch estimate, not lost time). Every
epoch was scored on the full 1319-example GSM8K test set at batch_size=1, K_max=12.

**Headline: 18.88% final accuracy, vs. the 39.5% SIM-CoT warm-start baseline — the model
converged to a plateau, it did not run out of epochs before converging.** Accuracy rose
6.22% → ~19% over epochs 1-11, then held inside **17-19.8%** for all 30 remaining epochs
(11-40), including the low-LR tail (36-40: 18.65/18.95/18.88/18.88%) where a late convergence
push would show up if one were coming. None appeared. `avg_steps_used` stayed flat at ~2.2 of
a K_max=12 budget for the entire run — the halting head is not spending more compute to buy
the accuracy the warm-start run gets, so the ~20.6pp gap is attributable to what the SIM-CoT
checkpoint's own ~40 epochs of pretraining contributed, which is exactly the confound exp-10
was designed to isolate.

Full per-epoch table (accuracy_pct, avg_steps_used from each epoch's `summary.json`):

| epoch | accuracy | avg steps | | epoch | accuracy | avg steps |
|---|---|---|---|---|---|---|
| 1 | 6.22% | 2.253 | | 21 | 19.11% | 2.230 |
| 2 | 9.33% | 2.053 | | 22 | 19.18% | 2.168 |
| 3 | 12.13% | 1.996 | | 23 | 19.26% | 2.216 |
| 4 | 12.36% | 2.067 | | 24 | 18.65% | 2.255 |
| 5 | 13.87% | 2.279 | | 25 | 19.48% | 2.232 |
| 6 | 14.86% | 2.059 | | 26 | 18.95% | 2.144 |
| 7 | 16.00% | 2.124 | | 27 | 18.42% | 2.163 |
| 8 | 15.16% | 2.275 | | 28 | 19.26% | 2.172 |
| 9 | 17.06% | 2.144 | | 29 | 19.11% | 2.167 |
| 10 | 18.35% | 2.281 | | 30 | 18.73% | 2.182 |
| 11 | 19.03% | 2.219 | | 31 | 18.95% | 2.176 |
| 12 | 18.27% | 2.179 | | 32 | 18.88% | 2.161 |
| 13 | 17.74% | 2.246 | | 33 | 18.80% | 2.187 |
| 14 | 19.41% | 2.274 | | 34 | 19.26% | 2.190 |
| 15 | 17.74% | 2.275 | | 35 | 18.95% | 2.196 |
| 16 | 16.91% | 2.177 | | 36 | 18.80% | 2.195 |
| 17 | 19.79% | 2.171 | | 37 | 18.65% | 2.183 |
| 18 | 18.95% | 2.195 | | 38 | 18.95% | 2.193 |
| 19 | 19.11% | 2.124 | | 39 | 18.88% | 2.183 |
| 20 | 18.50% | 2.203 | | 40 | 18.88% | 2.191 |

Full curve: `results-curve.png`. Raw `summary.json`/`eval.log` per epoch, the full `train.log`,
and the TensorBoard events are archived locally under `models/checkpoints/10-.../`,
`outputs/10-.../`, `results/10-.../` (gitignored — not in this repo). The final checkpoint
(`checkpoint-119960`, epoch 40, full with optimizer/scheduler/rng — 1.2 GB) was pulled to the
same `models/checkpoints/` path before the VMs were stopped.

Both cloud VMs (`alr-exp10-spot`, `alr-eval-l4`) are **STOPPED**, not deleted — their disks
persist. Total eval cost: 40 epochs × ~234s ≈ 2.6h of on-demand L4 ≈ $2. Training cost: ~39h
of Spot RTX PRO 6000.

## Scripts

| file | runs where | does |
|---|---|---|
| `scripts/gcp/launch_spot.sh` | local | creates the Spot G4 VM |
| `scripts/gcp/vm_bootstrap.sh` | train VM | venv, CUDA assertion, data, checkpoint, systemd unit |
| `scripts/gcp/measure_throughput.sh` | train VM | real s/step for a BS/ACCUM pair |
| `scripts/gcp/ckpt_upload.sh` | train VM | publishes each epoch checkpoint to GCS |
| `scripts/gcp/eval_orchestrator.sh` | train VM | boots the eval L4 once BATCH epochs are pending |
| `scripts/gcp/eval_worker.sh` | eval VM | drains pending epochs, publishes results, powers off |
| `scripts/gcp/watchdog.sh` | local | restarts the training VM after preemption |

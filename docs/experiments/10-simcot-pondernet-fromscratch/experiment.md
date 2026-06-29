# 10: from-scratch (vanilla GPT-2) — fair-comparison replication of exp-08 Run C

**Status:** in progress — recipe corrected & stability-validated on an L4; **full 40-epoch run pending on an A100** (this is the hand-off).
**Branch:** `experiment/10-fromscratch-gpt2`   **Dates:** 2026-06-26 → —

> **Running this on the A100? Read [§ Hand-off](#hand-off-run-the-full-40-epoch-job-on-an-a100-read-this) — it is self-contained.** The sections above it are context.

## Why

Every PonderNet experiment so far (03–08) warm-starts the **full** CODI model — backbone +
LoRA + prj + auxiliary decoder — from the downloaded SIM-CoT CODI checkpoint
(`models/pretrained/simcot-gpt2-codi`), then fine-tunes only **5 epochs at lr 2e-5**
(`train.py:271` loads `--simcot_ckpt` via `load_state_dict`). So the hard part — teaching
GPT-2 to reason in latent steps at all — is **inherited for free** from ~40 epochs of CODI
training we did not do. A fair-minded reviewer can object: *is the 41% from your adaptive
method, or from the checkpoint you started on?*

This experiment removes that objection by training the **same model and method as exp-08
Run C** (the min-steps operating point: γ=0.10, α=0.6, β=1.5 adaptive prior, K_max=12) but
**from vanilla GPT-2**, paying the full from-scratch cost ourselves. CODI's self-distillation
teacher (`ref_ce`/`distill_loss`) keeps cold latent training stable without a curriculum.

## What changed vs exp-08 Run C (and nothing else)

| knob | exp-08 Run C (warm-start) | this run (from scratch) | why |
|------|---------------------------|--------------------------|-----|
| backbone init | SIM-CoT CODI ckpt (`--simcot_ckpt`) | **vanilla `gpt2`** (`SIMCOT_CKPT=""`) | don't inherit latent reasoning |
| aux decoder init | SIM-CoT trained decoder | **vanilla `gpt2`** (`DECODER_PATH=gpt2`) | don't inherit the SIM-CoT decoder |
| train scope | `full` (decoder **frozen**) | **`full_dec`** (decoder **trained**) | SIM-CoT *learns* the decoder (paper Eq. 5–6); old scopes froze it only because they reused SIM-CoT's |
| learning rate | 2e-5 | **1e-3** (corrected — see Findings; paper says 3e-3 but it diverges in bf16) | 2e-5 barely moves a cold model |
| warmup_ratio | 0.03 | **0.05** | damp the cold-start gradient spike |
| epochs | 5 | **40** | CODI GPT-2 budget (Table E.3) |
| training data | train100k subsample | **full GSM8k-Aug `train.jsonl` (385,620 ex)** | faithful SIM-CoT data budget |
| `model_max_length` / `max_grad_norm` | 384 / 1.0 | **512 / 2.0** | align to the upstream CODI base recipe |
| eff. batch, prior, seed, K_max | 128, γ0.10/α0.6/β1.5, 42, 12 | **identical** | hold the method fixed |

### Code change (committed)

No existing `--pondernet_train_scope` trains the auxiliary decoder — all of `lora`,
`lora_prj`, `full` freeze `decoder.*` (every prior run reused SIM-CoT's trained decoder).
A from-scratch SIM-CoT run must learn its own decoder, so a new scope **`full_dec`** was
added (`train.py` scope block; `model.py` help text): `full` + unfreeze `decoder.*`.
Verified the decoder's 124.4M params become trainable only under `full_dec` (and, separately,
that the decoder weights actually move during training — 149/149 `decoder.*` tensors differ
from vanilla GPT-2 after 1 epoch, including layernorm/bias which weight-decay cannot touch).

## Findings — the LR=3e-3 divergence and the corrected recipe ⚠️

We launched the planned run (LR=3e-3, 40 epochs) on an NVIDIA **L4 (24 GB)** and watched it:

- **3e-3 diverges in bf16.** It warmed up to peak LR ~3e-3, ran clean for ~2200 steps, then
  **blew up at ~step 5820** (end of epoch 2): `grad_norm` jumped 20 → 2.5e11 in ~10 steps,
  NaN/Inf (`tensorboardX: NaN or Inf found`), `ce_loss` stuck bouncing ~2–4 and `ref_ce`
  saturated at ~11 (= ln|V|). Irreversible. This is exactly *Risk #1* below — pure LR
  instability, **not** a code/data bug (β=1.5 avoided the `masked_scatter` crash; the decoder
  was training fine).
- **1e-3 + warmup 0.05 is stable.** Restarted from scratch as a 4-epoch probe at **LR=1e-3,
  warmup_ratio=0.05** (everything else identical). It sailed through the same step-5820 region
  with `grad_norm` flat at ~0.5 and `ce_loss` dropping to ~0.35, and trained cleanly across
  epochs 1–4. **This is now the validated from-scratch recipe** and the default baked into
  `train_gpt2_gsm8k_pondernet_fromscratch.sh`.

The LR reduction (3e-3 → 1e-3) is the **only** deviation from the faithful SIM-CoT base, and
it is forced by numerical stability in bf16, not by the method. Everything else (LoRA,
optimizer, cosine schedule, decoder, distillation, full data, 40-epoch budget, K_max=12,
Run-C prior) is held identical, so the fair-comparison argument stands.

### Checkpoints we already have (from the L4 probe)

The L4 probe (`RUN=probe-lr1e-3-wu0.05-ep4`, LR=1e-3, 4-epoch cosine) produced full, resumable
HF checkpoints, one per epoch (~1.2 GB each: `pytorch_model.bin`, `optimizer.pt`,
`scheduler.pt`, `rng_state.pth`, `trainer_state.json`):

| checkpoint | epoch | global_step | LR at save |
|---|---|---|---|
| `checkpoint-2999` | 1 | 2999 | (cosine, mid) |
| `checkpoint-5998` | 2 | 5998 | (cosine, mid) |
| `checkpoint-8997` | 3 | 8997 | ~1.6e-4 |

⚠️ **These are from a *4-epoch* cosine schedule** (`max_steps=11996`, `num_train_epochs=4`),
not a 40-epoch one. See the Hand-off for how (and whether) to reuse them.

## Hyperparameter parity with upstream CODI

The base recipe is matched to the upstream CODI GPT-2 SIM-CoT script
(`baselines/CODI/scripts/train_gpt2_gsm8k-aug-decoder-2.sh`) so that what we train *is* a
faithful SIM-CoT, with only the adaptive-halting additions on top.

| hyperparameter | upstream CODI GPT-2 | this run | match? |
|---|---|---|---|
| backbone | `gpt2` | `gpt2` | ✓ |
| LoRA r / α / dropout | 128 / 32 / 0.1 | 128 / 32 / 0.1 | ✓ |
| LoRA target modules | `c_attn, c_proj, c_fc` | `c_attn, c_proj, c_fc` | ✓ |
| learning rate | 3e-3 | **1e-3** | ✗ **corrected** (3e-3 diverges in bf16; see Findings) |
| warmup_ratio | 0.03 | **0.05** | ✗ **corrected** (cold-start damping) |
| epochs | 40 | 40 | ✓ |
| eff. batch | 64 × 2 = 128 | 16 × 8 = 128 (L4) / 32 × 4 (A100) | ✓ (same effective) |
| weight_decay / sched | 0.1 / cosine | 0.1 / cosine | ✓ |
| `model_max_length` | 512 | 512 | ✓ |
| `max_grad_norm` | 2.0 | 2.0 | ✓ |
| prj_dim / prj_dropout | 768 / 0.0 | 768 / 0.0 | ✓ |
| `distill_loss_div_std` / `remove_eos` / `use_decoder` | True / True / True | True / True / True | ✓ |
| training data | full GSM8k-Aug | full `train.jsonl` (385,620 ex) | ✓ |
| **latent steps K** | **fixed `num_latent=6`** | **`max_latent_steps=12` (adaptive K_max)** | ✗ **intentional** — the method |
| **seed** | 11 | 42 | ✗ our convention |
| **halting head + KL-geom prior** | — | added (Run-C: γ0.10/α0.6/β1.5) | ✗ **intentional** — our contribution |

So the departures from a stock SIM-CoT base are: the three method pieces we are evaluating
(halting head, adaptive K_max=12, geometric prior), the LR/warmup correction (numerical
stability), and the seed. Everything defining the SIM-CoT base is identical.

---

## Hand-off: run the full 40-epoch job on an A100 (READ THIS)

**Goal:** one clean from-scratch run — cold GPT-2 → 40 epochs at the corrected stable recipe
→ per-epoch checkpoints shipped back. The decision (2026-06-29) is a **fresh** run, not a
resume: a single continuous 40-epoch cosine, the most defensible schedule. The 3 probe
checkpoints are an **optional compute-saver**, not the primary path (see step 6).

### 0. Prereqs
- A CUDA GPU. A100-40GB or -80GB ideal. ~Disk: 60 GB (env + data + 40 × 1.2 GB checkpoints).
- Internet (downloads `gpt2` + the GSM8k-Aug data from the HF hub).

### 1. Get the code (this branch)
```bash
git clone <repo-url> adaptive-latent-reasoning
cd adaptive-latent-reasoning
git checkout experiment/10-fromscratch-gpt2
```

### 2. Build the env
```bash
uv sync --python 3.12          # NOT 3.14 — torch has no 3.14 wheel
source .venv/bin/activate      # the training script calls bare `python`
```

### 3. Stage the training data (it is gitignored — not in the clone)
```bash
cd pondernet
python scripts/prep_gsm8k_aug.py     # downloads zen-E/GSM8k-Aug -> ../data/gsm8k_aug/train.jsonl (385,620 ex)
```
This is the only data the A100 needs (training). Eval data (`validation.jsonl`, `test.jsonl`)
is separate and only needed if you also evaluate here — see step 7.

### 4. Launch the run (one command)
```bash
cd pondernet
EXP=10-simcot-pondernet-fromscratch RUN=fromscratch-runC-lr1e-3-g0.10-a0.6-b1.5-k12-ep40 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
BS=32 ACCUM=4 \
bash scripts/train_gpt2_gsm8k_pondernet_fromscratch.sh
```
- `EXP`/`RUN` are **required** (a hook blocks the script otherwise) and set where artifacts land.
- `BS=32 ACCUM=4` keeps eff. batch **128** but runs faster than the L4's `16×8`. On A100-80GB
  try `BS=64 ACCUM=2`. Keep `BS×ACCUM=128`. If you OOM, drop `BS` and raise `ACCUM` to compensate.
- The wrapper bakes in the corrected recipe (cold GPT-2, `full_dec`, **LR 1e-3, warmup 0.05**,
  40 epochs, K_max 12, Run-C prior, `model_max_length 512`, `max_grad_norm 2.0`). The exact
  resolved command is auto-recorded to `outputs/<EXP>/<RUN>/command.sh`.
- **Disk is managed automatically** (so the per-epoch checkpoints don't fill the disk). A
  `CheckpointPruneCallback` keeps the **2 newest checkpoints fully resumable** (latest +
  previous) and, from every older epoch, strips only the resume-only optimizer/scheduler/rng
  state while **keeping the model weights + jsons** — so each epoch stays evaluable for the
  accuracy-vs-steps graphs (~0.58 GB/epoch instead of 1.2; ~24 GB for the full 40). Tune with
  `CKPT_KEEP_FULL` (default 2). If you need *minimal* disk and only care about the last few
  epochs, set `CKPT_KEEP_WEIGHTS=N` to also drop weights from epochs older than the newest N
  (keeps only the training-curve jsons).
- **Long-running:** put it in `tmux` (`tmux new -s train`, launch, detach with `Ctrl-b d`).
  It auto-resumes from the last checkpoint in `output_dir` if relaunched after a crash.
- **`--gradient_checkpointing` MUST stay False** (the default). It silently breaks the latent
  KV cache and trains against a broken objective. Do not turn it on.

### 5. Watch it (don't let a diverged run burn GPU)
Read scalars from TensorBoard logs under `outputs/10-simcot-pondernet-fromscratch/<RUN>/`:
```bash
python - <<'PY'
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ev = sorted(glob.glob("outputs/10-simcot-pondernet-fromscratch/**/events.*", recursive=True), key=os.path.getmtime)[-1]
ea = EventAccumulator(ev); ea.Reload()
for t in ["train/ce_loss","train/grad_norm","train/loss","train/learning_rate"]:
    s = ea.Scalars(t); print(f"{t:22s} last={s[-1].value:.4g} @ step {s[-1].step}")
PY
```
- **Healthy:** `ce_loss` trending down (settles well under 1), `grad_norm` bounded (~0.4–0.7
  at the LR peak). The probe held `grad_norm ≈ 0.5` flat through the step-5820 region.
- **Diverged (STOP the run):** `grad_norm` explodes (→1e7–1e18 / Inf) and/or `ce_loss` ramps
  toward ~11 and stays flat / NaN. If this happens, the LR is still too hot — drop to `LR=7e-4`
  (and/or `WARMUP=0.08`) and relaunch with a fresh `RUN`.

### 6. (OPTIONAL) Skip the first 3 epochs with our checkpoints — *only if you want to save compute*
Ana can send a zip of `checkpoint-2999/`, `checkpoint-5998/`, `checkpoint-8997/`. **The clean
result is the fresh run in step 4** (one 40-epoch schedule). If you instead want to fast-forward
past the ~3 epochs already trained, drop the latest checkpoint into this run's nested
`output_dir` so HF's auto-resume finds it:
```bash
# train.py nests output_dir; for a 40-epoch / lr-1e-3 / seed-42 run it is:
DEST=models/checkpoints/10-simcot-pondernet-fromscratch/fromscratch-runC-lr1e-3-g0.10-a0.6-b1.5-k12-ep40/default/gpt2/ep_40/lr_0.001/seed_42
mkdir -p "$DEST"
cp -r checkpoint-8997 "$DEST"/        # then launch step 4 with the SAME RUN; it resumes from 8997
```
⚠️ **Caveat (why this is not the default):** the checkpoint was trained under a *4-epoch*
cosine (LR already decayed to ~1.6e-4 by step 8997). Resuming it into a *40-epoch* schedule is
ambiguous — depending on HF version it either continues the old near-zero LR (so training
effectively ends ~1 epoch later) or rebuilds the cosine and **jumps the LR back up to ~9.9e-4**.
Either way it is not a single clean schedule. If you use it, **verify `train/learning_rate`
right after resume** looks sane before trusting the run. For the reported number, prefer the
fresh run. The checkpoints are most reliably used for (a) evidence the recipe is stable and
(b) evaluating an intermediate epoch (step 7).

### 7. (OPTIONAL) Evaluate here instead of shipping checkpoints back
Eval needs `validation.jsonl` (model selection) and `test.jsonl` (single final number) under
`data/gsm8k_aug/` — these are **not** in the HF dataset; get them from Ana / the team data.
```bash
EXP=10-simcot-pondernet-fromscratch RUN=<run> CKPT=<path-to-checkpoint-dir> \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval_gpt2_gsm8k_pondernet.sh --max_latent_steps 12 --batch_size 1
```
- **`--max_latent_steps 12` is required** — the eval script defaults to 6, but this model is
  K_max=12; without the override you cap the adaptive range.
- **`--batch_size 1` is mandatory** for faithful adaptive halting (see the repo AGENTS.md gotcha).
- Select the best epoch/threshold on **validation**, then report **one** number on **test**
  (do not select on test — the exp 01–07 leakage lesson). Run C's headline threshold is 0.5.

### 8. Ship results back
Send Ana the per-epoch checkpoints (`models/checkpoints/10-simcot-pondernet-fromscratch/<RUN>/.../checkpoint-*`)
and the logs (`outputs/.../<RUN>/`). She runs eval and records the final number in `docs/experiments/10-simcot-pondernet-fromscratch/runs.md`.

### Throughput / cost expectation
On the L4 it was ~13.8 s/step, ~3000 steps/epoch → ~11.5 h/epoch (~19 days for 40). The K=12
sequential latent loop is the bottleneck (GPU only ~17% utilized), so an A100 helps mostly via
a bigger micro-batch (fewer accumulation sub-steps) — expect a few× speedup, not 40×. If the
full 40 is too costly, the cheapest faithful cut is **fewer epochs** (each sees 3.85× more data
than the old 100k subsample) before touching data size; set `EPOCHS=<n>` on the launch.

---

## Risks / what to watch

1. **LR.** 3e-3 (paper) **diverges** in our bf16 setup (documented above); the validated recipe
   is **LR=1e-3 + warmup 0.05** (now the default). If even 1e-3 destabilizes on a different
   card/precision, fall back to `LR=7e-4` / `WARMUP=0.08`. **Failure signature:** `grad_norm`
   explosion and/or `ce_loss` ramping toward ~11 and staying flat → diverged.
2. **Never validated to convergence in-repo.** All previously-green runs were warm-start;
   from-scratch reaching ~SIM-CoT accuracy is the open question this experiment answers. Watch
   epoch-1–2 eval before trusting the full 40.
3. **`--gradient_checkpointing False`** is mandatory (breaks the latent KV cache; see exp-03).
4. **Compute.** ~120k optimizer steps for 40 epochs — hence the A100.

## Reporting (for the presentation)

Compare three points on GSM8K (validation, greedy, bs=1):
- **SIM-CoT / CODI baseline** (fixed K=6) — exp-01 `baseline-k6` = 40.80%.
- **exp-08 Run C** (warm-start + adaptive) — 40.6% @ 2.93 steps.
- **this run** (from-scratch + adaptive) — the honest "we trained the whole method from GPT-2"
  number; its gap to exp-08 Run C quantifies how much the warm-start was contributing.

See [runs.md](runs.md) once results land · artifacts under `<dir>/10-simcot-pondernet-fromscratch/`.

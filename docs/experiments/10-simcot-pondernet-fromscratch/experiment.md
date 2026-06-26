# 10: from-scratch (vanilla GPT-2) — fair-comparison replication of exp-08 Run C

**Status:** planned (to run on an external machine)   **Dates:** 2026-06-26 → —

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
teacher (`ref_ce`/`distill_loss`) keeps cold latent training stable without a curriculum —
this is the published CODI-on-GPT-2 recipe (SIM-CoT paper Table E.3: lr 3×10⁻³, 40 epochs),
with PonderNet adaptive halting layered on top.

## What changed vs exp-08 Run C (and nothing else)

| knob | exp-08 Run C (warm-start) | this run (from scratch) | why |
|------|---------------------------|--------------------------|-----|
| backbone init | SIM-CoT CODI ckpt (`--simcot_ckpt`) | **vanilla `gpt2`** (`SIMCOT_CKPT=""`) | don't inherit latent reasoning |
| aux decoder init | SIM-CoT trained decoder | **vanilla `gpt2`** (`DECODER_PATH=gpt2`) | don't inherit the SIM-CoT decoder |
| train scope | `full` (decoder **frozen**) | **`full_dec`** (decoder **trained**) | SIM-CoT *learns* the decoder (paper Eq. 5–6); the old scopes froze it only because they reused SIM-CoT's |
| learning rate | 2e-5 | **3e-3** | CODI GPT-2 from-scratch LR (Table E.3); 2e-5 barely moves a cold model |
| epochs | 5 | **40** | CODI GPT-2 budget (Table E.3) |
| training data | train100k subsample | **full GSM8k-Aug `train.jsonl` (385,620 ex)** | faithful SIM-CoT data budget (Table E.3 is 40 ep on the full set) |
| `model_max_length` / `max_grad_norm` | 384 / 1.0 | **512 / 2.0** | align to the upstream CODI base recipe (see parity table) |
| eff. batch, prior, seed, K_max | 128, γ0.10/α0.6/β1.5, 42, 12 | **identical** | hold the method fixed |

### Code change required (committed)

No existing `--pondernet_train_scope` trains the auxiliary decoder — all of `lora`,
`lora_prj`, `full` freeze `decoder.*` (every prior run reused SIM-CoT's trained decoder).
A from-scratch SIM-CoT run must learn its own decoder, so a new scope **`full_dec`** was
added (`train.py` scope block; `model.py` help text): `full` + unfreeze `decoder.*`.
Verified the decoder's 124.4M params become trainable only under `full_dec`.

## Hyperparameter parity with upstream CODI

The base recipe is matched to the upstream CODI GPT-2 SIM-CoT script
(`baselines/CODI/scripts/train_gpt2_gsm8k-aug-decoder-2.sh`) so that what we train *is* a
faithful SIM-CoT, with only the adaptive-halting additions on top. The wrapper sets
`model_max_length 512` and `max_grad_norm 2.0` explicitly because our main training script
defaults to 384 / 1.0; everything else already coincides.

| hyperparameter | upstream CODI GPT-2 | this run | match? |
|---|---|---|---|
| backbone | `gpt2` | `gpt2` | ✓ |
| LoRA r / α / dropout | 128 / 32 / 0.1 | 128 / 32 / 0.1 | ✓ (same `LoraConfig` code) |
| LoRA target modules | `c_attn, c_proj, c_fc` | `c_attn, c_proj, c_fc` | ✓ |
| LoRA init | `lora_init` (zero/gaussian) | `lora_init` | ✓ |
| learning rate | 3e-3 | 3e-3 | ✓ |
| epochs | 40 | 40 | ✓ |
| eff. batch | 64 × 2 = 128 | 16 × 8 = 128 | ✓ (same effective) |
| weight_decay / warmup / sched | 0.1 / 0.03 / cosine | 0.1 / 0.03 / cosine | ✓ |
| `model_max_length` | 512 | 512 | ✓ (wrapper override) |
| `max_grad_norm` | 2.0 | 2.0 | ✓ (wrapper override) |
| prj_dim / prj_dropout | 768 / 0.0 | 768 / 0.0 | ✓ |
| `distill_loss_div_std` / `remove_eos` / `use_decoder` | True / True / True | True / True / True | ✓ |
| training data | full GSM8k-Aug | full `train.jsonl` (385,620 ex) | ✓ |
| **latent steps K** | **fixed `num_latent=6`** | **`max_latent_steps=12` (adaptive K_max)** | ✗ **intentional** — this is the method |
| **seed** | 11 | 42 | ✗ our experiment convention |
| **halting head + KL-geom prior** | — | added (Run-C: γ0.10/α0.6/β1.5) | ✗ **intentional** — our contribution |

So the **only** departures from a stock SIM-CoT base are the three method pieces we are
actually evaluating: the halting head, the adaptive K_max=12 (vs fixed 6), and the geometric
halting prior. Everything that defines the SIM-CoT base — LoRA, optimizer, schedule, decoder,
distillation, data — is held identical.

## How to recreate it, step by step

1. **Clone + env.** On the target machine: `git checkout experiment/10-fromscratch-gpt2`,
   then `uv sync`. Activate the venv (`source .venv/bin/activate`) so bare `python` resolves.
2. **Stage the data** (gitignored — see the box below): place `train.jsonl`,
   `validation.jsonl`, `test.jsonl` under `data/gsm8k_aug/`.
3. **Train** from vanilla GPT-2 (one command — the wrapper assembles the SIM-CoT base recipe
   + `full_dec` + Run-C halting; see *Launch* below). This builds the SIM-CoT CODI model
   (LoRA-adapted GPT-2 backbone + prj + trained aux decoder, self-distilled against the
   explicit-CoT teacher) **and** trains the halting head jointly, all from scratch.
4. **Select** the best epoch/threshold on `validation` (bs=1, greedy) — see *Eval*.
5. **Report** one final number on `test`, alongside the SIM-CoT/CODI baseline (exp-01) and
   exp-08 Run C, so the warm-start contribution is quantified.

## Launch (on the external machine)

```bash
cd pondernet
EXP=10-simcot-pondernet-fromscratch RUN=fromscratch-runC-g0.10-a0.6-b1.5-k12-ep40 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idle-gpu> \
bash scripts/train_gpt2_gsm8k_pondernet_fromscratch.sh
```

The wrapper `scripts/train_gpt2_gsm8k_pondernet_fromscratch.sh` bakes in the from-scratch
recipe (no warm-start, `DECODER_PATH=gpt2`, `full_dec`, lr 3e-3, 40 epochs, eff. batch 128,
**full `train.jsonl`** via `MAX_SAMPLES=400000` ≥ the 385,620-example set) and forwards Run-C
prior flags; everything stays env-overridable (`LR`, `EPOCHS`, `BS`, `ACCUM`, `MAX_SAMPLES`,
`DATA_PATH`, `GAMMA`, `PRIOR_OFFSET`, `PRIOR_SCALE`). It calls the main training script, so the
exact resolved command is auto-recorded to `outputs/<EXP>/<RUN>/command.sh`.

### Before launching on the remote box
- ⚠️ **The datasets are NOT in git — they must be copied manually.** `data/`, `models/`, and
  `outputs/` are all **gitignored**, so a fresh `git clone` on the remote VM has the code but
  **none of the data**. Before training, copy the three GSM8k-Aug splits over (`scp`/`rsync`),
  preserving the paths the scripts expect:
  - `data/gsm8k_aug/train.jsonl` — training (~100 MB, 385,620 ex)
  - `data/gsm8k_aug/validation.jsonl` — model selection after training
  - `data/gsm8k_aug/test.jsonl` — the single final report number

  The backbone/decoder (`gpt2`) download automatically from the HF hub, so `models/` does **not**
  need shipping for this run; `outputs/`/`results/` are created fresh by the scripts.
- **`uv sync`** to build the venv; the training script calls bare `python`, so run inside the
  activated venv (`source .venv/bin/activate`) or put `.venv/bin` on `PATH`.
- Keep **`--gradient_checkpointing False`** (already the default — it breaks the latent KV
  cache; see exp-03). Needs the VRAM; eff. batch 128 via bs16×accum8 is the proven-clean
  sequence (bs32 OOMed on the 3090, but a bigger remote card may fit bs32×accum4).

## Eval (after training)

```bash
EXP=10-simcot-pondernet-fromscratch RUN=<run> \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idle-gpu> \
bash scripts/eval_gpt2_gsm8k_pondernet.sh --batch_size 1   # bs=1 is mandatory for faithful halting
```

`save_total_limit 3` keeps the last 3 epoch checkpoints. Evaluate each on **validation**
(n=500, greedy, **bs=1**), pick the best epoch/threshold, then report **one** final number on
the **test** split — do not select on test (the exp 01–07 leakage lesson). Run C's headline
operating point is **thr0.5**.

## Risks / what to watch

1. **LR 3e-3 is aggressive.** Smoke (bs=8, no warmup) showed jumpy loss (7.5→3.9→15.6→17.6);
   the real recipe (eff. batch 128 + 3% warmup + cosine) damps this. **Failure signature:**
   `ce_loss`/`l_pondernet` ramps toward ~11 (=ln|V|) and stays flat → diverged; fall back to
   **LR=1e-3** (and/or raise `--warmup_ratio` to 0.05). β=1.5 already avoids the
   `masked_scatter` crash that β≤1 triggers (~epoch 2).
2. **Never validated to convergence in-repo.** All green runs were warm-start; from-scratch
   reaching ~SIM-CoT accuracy is the open question this experiment answers. Watch epoch-1–2
   eval before trusting the full 40.
3. **Compute.** 40 epochs × 385k ≈ **120k optimizer steps** (eff. batch 128, ~3 k steps/epoch)
   — vs exp-08's 5 × 100k, i.e. **~30× the wall-clock** — hence the external machine. This is
   the faithful SIM-CoT budget; if it's too long, the cheapest cut is fewer epochs (each now
   sees 3.85× more data than the old 100k subsample) before touching the data size.

## Smoke validation (done on the 3090, 2026-06-26, before hand-off)

Ran the from-scratch config end-to-end for a few steps via the wrapper:
no warm-start, decoder from vanilla GPT-2, `full_dec`. All loss terms finite and engaged —
`l_pondernet` (answer), `distill_loss`+`ref_ce` (CODI self-distillation teacher),
`explain_loss` (decoder step-supervision, now **trainable**), `kl_geom`, and `p_mean_step`
responds (6.6→4.4→7.3). No NaN/crash. Confirms the recipe assembles and trains; it does **not**
confirm final accuracy (that's the 40-epoch run's job).

## Reporting (for the presentation)

Compare three points on GSM8K (validation, greedy, bs=1):
- **SIM-CoT / CODI baseline** (fixed K=6) — exp-01 `baseline-k6` = 40.80%.
- **exp-08 Run C** (warm-start + adaptive) — 40.6% @ 2.93 steps.
- **this run** (from-scratch + adaptive) — the honest "we trained the whole method from GPT-2"
  number; its gap to exp-08 Run C quantifies exactly how much the warm-start was contributing.

See [runs.md](runs.md) once results land · artifacts under `<dir>/10-simcot-pondernet-fromscratch/`.

#!/usr/bin/env bash
# Runs ON the VM, once. Installs the venv, materializes the data, seeds the epoch-12
# checkpoint, and installs a systemd unit that restarts training after every preemption.
#
# RESUME_MODE
# -----------
#   scratch  cold GPT-2, one continuous 40-epoch cosine (~40.4 h at the measured 1.01 h/epoch).
#            No inherited state, nothing to caveat. This is what CODI/SIM-CoT actually do.
#
#   exact   (default, REQUIRES the 1.2GB checkpoint with optimizer.pt + scheduler.pt)
#           The checkpoint is dropped straight into SAVE_DIR. train.py's get_last_checkpoint
#           picks it up and HF Trainer restores weights, optimizer moments, LR scheduler
#           position, RNG and global_step. The run continues as if it never stopped: 40
#           epochs total, LR follows the original cosine from 8.39e-4 down to ~0.
#
#   weights (fallback, weights-only 556MB tar)
#           transformers' _load_optimizer_and_scheduler() only restores state when BOTH
#           optimizer.pt AND scheduler.pt exist; otherwise it returns silently. So a
#           weights-only "resume" would zero Adam's moments AND rewind the LR scheduler to
#           step 0 — re-running a 2-epoch warmup back up to peak 1e-3, then decaying over a
#           40-epoch horizon while only 28 epochs of steps remain, ending at 2.27e-4 instead
#           of ~0. Instead of pretending to resume, this mode warm-starts the weights via
#           --simcot_ckpt (a plain load_state_dict) and runs a deliberate, self-consistent
#           28-epoch schedule: peak LR 8.386e-4 (what the original cosine held at epoch 12),
#           a short warmup to rebuild Adam's second moments, cosine to ~0 at epoch 40.
#           This is an approximation of the tail, not the tail. Document it in experiment.md.
set -euxo pipefail

RESUME_MODE="${RESUME_MODE:-exact}"
SETUP_ONLY="${SETUP_ONLY:-0}"   # 1 = venv + CUDA check + data, then stop. No checkpoint, no systemd.
CKPT_DIR="${CKPT_DIR:-/mnt/ckpt}"
REPO=/opt/alr
CKPT_URL="${CKPT_URL:-https://storage.googleapis.com/alr-ckpt-ep12-handoff/ckpt-ep12-weights.tar}"

export EXP="${EXP:-10-simcot-pondernet-fromscratch}"
export RUN="${RUN:-fromscratch-runC-lr1e-3-g0.10-a0.6-b1.5-k12-ep40}"
SAVE_DIR=$CKPT_DIR/models/checkpoints/$EXP/$RUN

# --- toolchain ---------------------------------------------------------------------------
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"
uv sync --python 3.12

# Blackwell is sm_120; the default cu126 wheel has no kernels for it. Fail loudly here
# rather than 40 minutes into a training run.
uv run python - <<'PY'
import torch
cap = torch.cuda.get_device_capability(0)
print("gpu:", torch.cuda.get_device_name(0), "sm_%d%d" % cap, "torch", torch.__version__)
torch.zeros(8, device="cuda").sum().item()          # raises if no kernel image for sm_120
assert torch.cuda.is_bf16_supported(), "bf16 unsupported?!"
print("cuda ok")
PY

# --- data + epoch-12 checkpoint -----------------------------------------------------------
mkdir -p "$SAVE_DIR" "$CKPT_DIR"/{init,outputs,results}
[ -f data/gsm8k_aug/train.jsonl ] || (cd pondernet && uv run --project "$REPO" python scripts/prep_gsm8k_aug.py)

if [ "$SETUP_ONLY" = 1 ]; then
  echo "==> SETUP_ONLY: venv, CUDA and data ready. Skipping checkpoint + systemd."
  exit 0
fi

if [ "$RESUME_MODE" != scratch ] && [ ! -d $CKPT_DIR/init/checkpoint-35988 ]; then
  curl -fsSL "$CKPT_URL" -o /tmp/ckpt.tar
  tar -xf /tmp/ckpt.tar -C $CKPT_DIR/init && rm /tmp/ckpt.tar
fi

# The run saves with --save_safetensors False (pytorch_model.bin), but train.py's
# --simcot_ckpt warm-start path reads with safetensors.load_file(). Convert once so
# RESUME_MODE=weights has something it can actually load.
if [ "$RESUME_MODE" != scratch ] && [ -f $CKPT_DIR/init/checkpoint-35988/pytorch_model.bin ] && \
   [ ! -f $CKPT_DIR/init/checkpoint-35988/model.safetensors ]; then
  uv run python - "$CKPT_DIR/init/checkpoint-35988" <<'PY'
import sys, torch
from safetensors.torch import save_file
d = sys.argv[1]
sd = torch.load(f"{d}/pytorch_model.bin", map_location="cpu", weights_only=True)
save_file({k: v.contiguous() for k, v in sd.items()}, f"{d}/model.safetensors")
print(f"converted {len(sd)} tensors to safetensors")
PY
fi

# --- recipe: identical to the wrapper's defaults except where the resume mode forces a change
# BS*ACCUM must stay 128 (the effective batch of exp-08 Run C). BS=64 ACCUM=2 is the ceiling,
# measured on the actual RTX PRO 6000: it peaks at 62.8/95.6 GB and runs 1.21 s/step, while
# BS=128 ACCUM=1 OOMs (needs ~116 GB of activations — the K_max=12 latent loop keeps a
# separate answer-decode graph per step, so activations, not weights, dominate memory).
COMMON=(EXP="$EXP" RUN="$RUN" SAVE_DIR="$SAVE_DIR"
        LOG_DIR=$CKPT_DIR/outputs/$EXP/$RUN
        BS="${BS:-64}" ACCUM="${ACCUM:-2}")   # measured ceiling; BS=128 OOMs

case "$RESUME_MODE" in
  scratch)
    # Cold GPT-2, one continuous 40-epoch cosine — exactly what CODI/SIM-CoT do. No checkpoint,
    # no inherited optimizer state, nothing to caveat in the writeup. At 1.01 h/epoch this costs
    # ~40.4 h, versus 13.5 days on the L4 — which is the only reason resuming was ever worth it.
    RECIPE=("${COMMON[@]}" EPOCHS=40)   # LR/WARMUP: wrapper defaults (1e-3 / 0.05); SIMCOT_CKPT="" is set by the wrapper
    ;;
  exact)
    [ -f $CKPT_DIR/init/checkpoint-35988/optimizer.pt ] || {
      echo "RESUME_MODE=exact needs optimizer.pt; you have the weights-only tar." >&2
      echo "Either fetch the 1.2GB checkpoint or re-run with RESUME_MODE=weights." >&2; exit 1; }
    cp -rn $CKPT_DIR/init/checkpoint-35988 "$SAVE_DIR"/
    RECIPE=("${COMMON[@]}" EPOCHS=40)
    ;;
  weights)
    RECIPE=("${COMMON[@]}" EPOCHS=28 LR=8.386e-4 WARMUP=0.005
            SIMCOT_CKPT=$CKPT_DIR/init/checkpoint-35988/model.safetensors)
    ;;
  *) echo "RESUME_MODE must be scratch|exact|weights, got '$RESUME_MODE'" >&2; exit 1 ;;
esac

# --- systemd: restarts training on boot (i.e. after every preemption) ---------------------
# get_last_checkpoint(SAVE_DIR) makes this idempotent: each restart picks up the newest
# fully-resumable epoch checkpoint that OUR run wrote (CKPT_KEEP_FULL=2 keeps them complete).
sudo tee /etc/systemd/system/alr-train.service >/dev/null <<UNIT
[Unit]
Description=exp-10 from-scratch GPT-2 (PonderNet) — epochs 12..40
After=network-online.target
RequiresMountsFor=$CKPT_DIR

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO/pondernet
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=${RECIPE[*]}
# Trailing flags land after the wrapper's fixed ones and argparse keeps the LAST
# occurrence, so this overrides --dataloader_num_workers 4 without editing team scripts.
# 16 workers: the 48-vCPU G4 can afford it and tokenize-on-the-fly stops gating the GPU.
ExecStart=/bin/bash -lc 'uv run --project $REPO bash scripts/train_gpt2_gsm8k_pondernet_fromscratch.sh --dataloader_num_workers 16'
Restart=on-failure
RestartSec=30
StandardOutput=append:$CKPT_DIR/outputs/train.log
StandardError=append:$CKPT_DIR/outputs/train.log

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now alr-train.service
echo "==> training started (RESUME_MODE=$RESUME_MODE). tail -f $CKPT_DIR/outputs/train.log"

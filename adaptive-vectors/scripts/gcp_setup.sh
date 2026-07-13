#!/usr/bin/env bash
# Provision a GCP L4 VM and stage the repo to run the adaptive-K / warm-start factorial
# (scripts/sweep_k_recipe.sh) off-box, so the local 3090 stays free.
#
# IMPORTANT - GPU/arch note: the local RTX 5070 is Blackwell (sm_120) and the pinned
# torch in this repo has NO kernels for it ("no kernel image is available"). The L4 is
# Ada (sm_89) and IS supported by that torch, which is exactly why L4 is the cloud pick.
# Do NOT switch to an even newer card (e.g. an sm_120 instance) without rebuilding torch.
#
# This script is intended to be run in stages, not blindly end-to-end. Each stage is a
# function; call them in order once you've set the env vars below and authenticated
# (`gcloud auth login` / `gcloud config set project <id>`). Interactive logins must be
# run by you in your own shell.
#
# Usage:
#   export GCP_PROJECT=my-project ZONE=us-central1-a INSTANCE=alr-l4
#   bash scripts/gcp_setup.sh create     # provision the spot L4 VM
#   bash scripts/gcp_setup.sh stage      # rsync repo + models + data to the VM
#   bash scripts/gcp_setup.sh remote     # uv sync + smoke test on the VM
#   bash scripts/gcp_setup.sh sweep      # launch sweep_k_recipe.sh in tmux on the VM
#   bash scripts/gcp_setup.sh fetch      # pull results/ back to this box
#   bash scripts/gcp_setup.sh delete     # tear the VM down (stop billing)

set -euo pipefail

GCP_PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
ZONE="${ZONE:-us-central1-a}"               # L4 also in us-east4-c, europe-west4-b
INSTANCE="${INSTANCE:-alr-l4}"
MACHINE="${MACHINE:-g2-standard-8}"          # 8 vCPU / 32 GB, 1x L4 24 GB
DISK_GB="${DISK_GB:-100}"
# Deep Learning VM image w/ CUDA 12.x (matches the repo's torch). Alternative: a clean
# Ubuntu 22.04 + manual uv install.
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu123-ubuntu-2204-py310}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
REMOTE_DIR="${REMOTE_DIR:-~/alr-anapaula}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

create() {
    # Spot/preemptible: ~$0.28/hr vs ~$0.85 on-demand. Per-epoch checkpointing makes
    # preemption recoverable (resume from the latest epoch checkpoint).
    gcloud compute instances create "$INSTANCE" \
        --project "$GCP_PROJECT" --zone "$ZONE" \
        --machine-type "$MACHINE" \
        --accelerator type=nvidia-l4,count=1 \
        --provisioning-model SPOT --instance-termination-action STOP \
        --maintenance-policy TERMINATE \
        --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" \
        --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-ssd \
        --metadata install-nvidia-driver=True
    echo "[gcp] created $INSTANCE in $ZONE. Wait ~60s for the driver install, then: bash $0 stage"
}

stage() {
    # Transfer repo code + the gitignored artifacts the sweep needs:
    #   models/pretrained/{gpt2,simcot-gpt2-codi,simcot-gpt2-decoder}  (warm-start sources)
    #   data/gsm8k_aug/{train100k.jsonl,test.jsonl}                    (train/eval data)
    # Excludes the large local checkpoints/outputs/results (regenerated on the VM).
    gcloud compute ssh "$INSTANCE" --project "$GCP_PROJECT" --zone "$ZONE" --command "mkdir -p $REMOTE_DIR"
    local ssh_cmd="gcloud compute ssh $INSTANCE --project $GCP_PROJECT --zone $ZONE --command"
    # rsync over the gcloud ssh tunnel
    gcloud compute scp --recurse --project "$GCP_PROJECT" --zone "$ZONE" \
        "$LOCAL_ROOT/pondernet" "$LOCAL_ROOT/pyproject.toml" "$LOCAL_ROOT/uv.lock" \
        "$LOCAL_ROOT/AGENTS.md" "$LOCAL_ROOT/docs" \
        "$INSTANCE":"$REMOTE_DIR/"
    gcloud compute ssh "$INSTANCE" --project "$GCP_PROJECT" --zone "$ZONE" \
        --command "mkdir -p $REMOTE_DIR/models/pretrained $REMOTE_DIR/data/gsm8k_aug"
    gcloud compute scp --recurse --project "$GCP_PROJECT" --zone "$ZONE" \
        "$LOCAL_ROOT/models/pretrained/gpt2" \
        "$LOCAL_ROOT/models/pretrained/simcot-gpt2-codi" \
        "$LOCAL_ROOT/models/pretrained/simcot-gpt2-decoder" \
        "$INSTANCE":"$REMOTE_DIR/models/pretrained/"
    gcloud compute scp --project "$GCP_PROJECT" --zone "$ZONE" \
        "$LOCAL_ROOT/data/gsm8k_aug/train100k.jsonl" \
        "$LOCAL_ROOT/data/gsm8k_aug/test.jsonl" \
        "$INSTANCE":"$REMOTE_DIR/data/gsm8k_aug/"
    echo "[gcp] staged. Next: bash $0 remote"
}

remote() {
    # Install uv, sync deps, verify the GPU + a tiny smoke run (recipe B at K=6, 30 steps).
    gcloud compute ssh "$INSTANCE" --project "$GCP_PROJECT" --zone "$ZONE" --command "
        set -e
        command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH=\$HOME/.local/bin:\$PATH
        cd $REMOTE_DIR && uv sync
        cd $REMOTE_DIR/pondernet
        uv run python -c 'import torch; print(\"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
        echo '[gcp] smoke: recipe B, K=6, 30 steps'
        CUDA_VISIBLE_DEVICES=0 SIMCOT_CKPT='' TRAIN_SCOPE=lora_prj \
          SAVE_DIR=../models/checkpoints/_smoke LOG_DIR=../outputs/_smoke \
          bash scripts/train_gpt2_gsm8k_pondernet.sh \
            --pondernet_train_scope lora_prj --max_latent_steps 6 \
            --max_train_samples 256 --num_train_epochs 1 --save_strategy no 2>&1 | tail -20
    "
    echo "[gcp] smoke done. Inspect loss decreasing, then: bash $0 sweep"
}

sweep() {
    # Single L4 => TRAIN_GPU=EVAL_GPU=0 (eval runs after each train). tmux so it survives
    # ssh drops; spot preemption resumes from the latest epoch checkpoint on relaunch.
    gcloud compute ssh "$INSTANCE" --project "$GCP_PROJECT" --zone "$ZONE" --command "
        export PATH=\$HOME/.local/bin:\$PATH
        cd $REMOTE_DIR/pondernet
        tmux new-session -d -s ksweep \
          'CUDA_DEVICE_ORDER=PCI_BUS_ID TRAIN_GPU=0 EVAL_GPU=0 EVAL_BATCH_SIZE=16 \
           bash scripts/sweep_k_recipe.sh 2>&1 | tee ../results/k_recipe_sweep.log'
        echo '[gcp] sweep launched in tmux session ksweep. Attach: tmux attach -t ksweep'
    "
}

fetch() {
    gcloud compute scp --recurse --project "$GCP_PROJECT" --zone "$ZONE" \
        "$INSTANCE":"$REMOTE_DIR/results/k_recipe_sweep" "$LOCAL_ROOT/results/"
    echo "[gcp] results pulled to $LOCAL_ROOT/results/k_recipe_sweep"
}

delete() {
    gcloud compute instances delete "$INSTANCE" --project "$GCP_PROJECT" --zone "$ZONE" --quiet
    echo "[gcp] $INSTANCE deleted."
}

cmd="${1:-}"
case "$cmd" in
    create|stage|remote|sweep|fetch|delete) "$cmd" ;;
    *) echo "usage: $0 {create|stage|remote|sweep|fetch|delete}" >&2; exit 1 ;;
esac

#!/usr/bin/env bash
# Create the Spot G4 VM (1x RTX PRO 6000, 96GB) that finishes exp-10 from epoch 12 to 40.
#
# WHY SPOT, AND WHAT SURVIVES A PREEMPTION
# ----------------------------------------
# Spot capacity is 60-91% cheaper but Google can reclaim it with ~30s notice. Three things
# make that survivable here:
#   1. --instance-termination-action=STOP: a preemption STOPS the VM, it does not delete it.
#      The boot disk - which holds the venv, the data and every checkpoint - persists.
#   2. train.py already calls trainer.train(resume_from_checkpoint=get_last_checkpoint(...)),
#      and the wrapper saves --save_strategy epoch, so a restart resumes from the last
#      completed epoch. Worst case a preemption costs < 1 epoch of compute.
#   3. The systemd unit installed by vm_bootstrap.sh is `enabled`, so training restarts by
#      itself the moment the VM boots again. watchdog.sh (run locally) does the booting.
#
# DISKS: G4 rejects pd-balanced outright - it only takes hyperdisk-balanced.
#
# IMAGE: common-cu129-ubuntu-2204-nvidia-580 ships driver 580, which is the first branch that
# supports Blackwell (sm_120). Pair it with the cu128 torch wheel pinned in pyproject.toml.
#
# Usage:  bash pondernet/scripts/gcp/launch_spot.sh
# Env:    PROJECT ZONE VM BOOT_SIZE MACHINE
set -euo pipefail

PROJECT="${PROJECT:-adaptative-latent-reasoning}"
ZONE="${ZONE:-us-central1-b}"          # us-central1-b and -f are the only us-central1 zones with G4
VM="${VM:-alr-exp10-spot}"
# No separate data disk: a 100GB boot disk leaves 73GB free after the DLVM image, venv and
# GSM8k-Aug, which holds all 40 epoch checkpoints (~48GB). This also dodges SSD_TOTAL_GB,
# capped at 300GB/region here and already 200GB spent on the old exp10-l4 disk.
BOOT_SIZE="${BOOT_SIZE:-100GB}"
MACHINE="${MACHINE:-g4-standard-48}"   # 1x RTX PRO 6000, 48 vCPU, 180GB RAM

g() { gcloud --project="$PROJECT" "$@"; }

if g compute instances describe "$VM" --zone="$ZONE" >/dev/null 2>&1; then
  echo "==> VM $VM already exists; starting it if stopped"
  g compute instances start "$VM" --zone="$ZONE" || true
  exit 0
fi

echo "==> creating Spot VM $VM ($MACHINE) in $ZONE"
g compute instances create "$VM" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size="$BOOT_SIZE" \
  --boot-disk-type=hyperdisk-balanced \
  --scopes=cloud-platform \
  --metadata=install-nvidia-driver=True

cat <<MSG

==> VM created. Next:

  # 1. copy this repo onto the VM (it lives on the boot disk, survives preemption)
  gcloud compute scp --project=$PROJECT --zone=$ZONE --recurse \\
      . $VM:/opt/alr --compress

  # 2. bootstrap: venv, data, checkpoint, systemd unit (see vm_bootstrap.sh for RESUME_MODE)
  gcloud compute ssh --project=$PROJECT --zone=$ZONE $VM -- \\
      'RESUME_MODE=exact CKPT_DIR=/opt/alr/ckpt bash /opt/alr/pondernet/scripts/gcp/vm_bootstrap.sh'

  # 3. keep it alive across preemptions (run locally, leave it open)
  PROJECT=$PROJECT ZONE=$ZONE VM=$VM bash pondernet/scripts/gcp/watchdog.sh

MSG

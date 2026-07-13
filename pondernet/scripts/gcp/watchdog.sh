#!/usr/bin/env bash
# Restart the Spot VM after Google preempts it. Run this locally and leave it open.
#
# A preemption with --instance-termination-action=STOP leaves the instance in TERMINATED
# state with its disks intact. Nothing in GCE brings it back on its own (Spot VMs cannot use
# automatic restart - that flag only covers host errors), so something outside has to press
# start. This loop is that something. Once the VM boots, the enabled alr-train.service picks
# training back up from the last epoch checkpoint.
#
# `instances start` fails with ZONE_RESOURCE_POOL_EXHAUSTED when there is no spare Spot
# capacity. That is expected and temporary: we just keep asking, backing off so we do not
# hammer the API while a region is drained.
set -uo pipefail

PROJECT="${PROJECT:-adaptative-latent-reasoning}"
ZONE="${ZONE:-us-central1-b}"
VM="${VM:-alr-exp10-spot}"
POLL="${POLL:-60}"          # seconds between status checks while healthy
BACKOFF_MAX="${BACKOFF_MAX:-900}"

backoff=$POLL
while true; do
  status=$(gcloud compute instances describe "$VM" --project="$PROJECT" --zone="$ZONE" \
             --format='value(status)' 2>/dev/null) || status=MISSING

  case "$status" in
    RUNNING)
      backoff=$POLL
      sleep "$POLL"
      ;;
    TERMINATED|SUSPENDED)
      echo "[$(date -Is)] $VM is $status (preempted) - restarting"
      if gcloud compute instances start "$VM" --project="$PROJECT" --zone="$ZONE" 2>&1; then
        echo "[$(date -Is)] restarted; training resumes from the last epoch checkpoint"
        backoff=$POLL
      else
        echo "[$(date -Is)] no Spot capacity; retrying in ${backoff}s"
        sleep "$backoff"
        backoff=$(( backoff * 2 > BACKOFF_MAX ? BACKOFF_MAX : backoff * 2 ))
        continue
      fi
      sleep "$POLL"
      ;;
    MISSING)
      echo "[$(date -Is)] $VM does not exist in $ZONE - nothing to watch"; exit 1
      ;;
    *)  # PROVISIONING, STAGING, STOPPING...
      sleep "$POLL"
      ;;
  esac
done

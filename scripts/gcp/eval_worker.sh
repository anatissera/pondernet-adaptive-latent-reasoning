#!/usr/bin/env bash
# Runs ON the eval VM (an L4 in project tp-final-rl-kv-eviction), from its startup-script.
# Scores every not-yet-evaluated epoch on the GSM8K test split, then powers the VM off.
#
# WHY A SECOND GPU: the trainer saturates the RTX PRO 6000, and AGENTS.md requires eval at
# --batch_size 1 (test.py only breaks the latent loop once EVERY example in the batch has
# halted, so batch>1 decodes early-halting examples from more steps than steps_used reports).
# Batch-1 eval of 1319 examples is slow and cannot be batched away, so it gets its own card.
#
# WHY IT POWERS ITSELF OFF: this VM is on-demand and bills per second it is RUNNING. The
# orchestrator on the training VM boots it once a batch of checkpoints has piled up; draining
# and halting is how it stops billing without anyone watching. A guest `poweroff` puts a GCE
# instance into TERMINATED, which is exactly the state the orchestrator waits for.
#
# Idempotent per checkpoint (a summary.json in the bucket means "done"), so a preemption or a
# crashed boot just re-drains. Set NO_POWEROFF=1 to keep the VM up (debugging).
set -uo pipefail

REPO=/opt/alr
BUCKET="${BUCKET:-gs://alr-exp10-ckpts-244544686610}"
NO_POWEROFF="${NO_POWEROFF:-0}"
WORK=$REPO/evalwork
mkdir -p "$WORK"

export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"
cd "$REPO/pondernet"

epdir() { printf 'ep%02d' "$1"; }

drained=0
while true; do
  pending=""
  # Extract the step, THEN sort numerically. `sort -t- -k2 -n` on the full gs:// URL is wrong:
  # the bucket name (alr-exp10-ckpts-…) contains hyphens, so field 2 is "exp10" on every line,
  # every key ties, and sort falls back to lexicographic - putting checkpoint-11996 before
  # checkpoint-2999. Everything still got evaluated, but epochs came out of order.
  for step in $(gcloud storage ls "$BUCKET/ckpt/*/.ready" 2>/dev/null \
                  | sed -nE 's#.*/checkpoint-([0-9]+)/\.ready$#\1#p' | sort -n); do
    epoch=$(( step / 2999 ))
    # a summary.json in the bucket is the only source of truth for "already evaluated"
    gcloud storage ls "$BUCKET/results/$(epdir "$epoch")/summary.json" >/dev/null 2>&1 && continue
    pending="$pending $step"
  done

  [ -z "$pending" ] && break

  for step in $pending; do
    epoch=$(( step / 2999 ))
    local_ck="$WORK/checkpoint-$step"; mkdir -p "$local_ck"
    echo "[$(date -Is)] epoch $epoch (step $step): downloading"
    gcloud storage cp "$BUCKET/ckpt/checkpoint-$step/pytorch_model.bin" \
        "$local_ck/pytorch_model.bin" --quiet || { echo "download failed"; continue; }

    rdir="$WORK/results/$(epdir "$epoch")"
    mkdir -p "$rdir"                       # the redirect below opens before eval's own mkdir
    wlog="$rdir/worker.log"
    t0=$(date +%s)
    echo "[$(date -Is)] epoch $epoch: evaluating on test.jsonl (1319 ex, bs=1, K=12)"
    EXP=10-simcot-pondernet-fromscratch RUN=eval-ep$epoch RESULTS_DIR="$rdir" \
    CKPT="$local_ck" GPT2_PATH=gpt2 DATA_PATH=../data/gsm8k_aug/test.jsonl \
    BATCH_SIZE=1 THRESHOLD=0.5 \
      bash scripts/eval_gpt2_gsm8k_pondernet.sh --max_latent_steps 12 > "$wlog" 2>&1
    dt=$(( $(date +%s) - t0 ))

    if [ -f "$rdir/summary.json" ]; then
      python3 -c "
import json
d = json.load(open('$rdir/summary.json'))
print(f\"[eval] epoch $epoch  acc={d['accuracy_pct']}%  avg_steps={d['avg_steps_used']}  ({$dt}s)\")"
      gcloud storage cp "$rdir/summary.json" "$BUCKET/results/$(epdir "$epoch")/summary.json" --quiet
      gcloud storage cp "$wlog" "$BUCKET/results/$(epdir "$epoch")/eval.log" --quiet
      drained=$(( drained + 1 ))
    else
      # No summary.json means the eval died. Without a failure cap the outer loop would keep
      # seeing this epoch as pending and retry it forever, and the VM would never power off.
      fails=$(( $(cat "$WORK/.fail-$step" 2>/dev/null || echo 0) + 1 ))
      echo "$fails" > "$WORK/.fail-$step"
      echo "[$(date -Is)] epoch $epoch: EVAL FAILED (${dt}s, attempt $fails) - see $wlog"
      tail -5 "$wlog"
      gcloud storage cp "$wlog" "$BUCKET/results/$(epdir "$epoch")/FAILED.log" --quiet
      if [ "$fails" -ge 2 ]; then
        echo "[$(date -Is)] epoch $epoch: giving up, marking done so the drain can finish"
        echo '{"accuracy_pct": null, "avg_steps_used": null, "total_examples": 0, "error": "eval failed twice"}' \
          | gcloud storage cp - "$BUCKET/results/$(epdir "$epoch")/summary.json" --quiet
      fi
    fi
    rm -f "$local_ck/pytorch_model.bin"    # 578MB each; do not fill the boot disk
  done
done

echo "[$(date -Is)] drained $drained checkpoint(s); nothing pending"
if [ "$NO_POWEROFF" != 1 ]; then
  echo "[$(date -Is)] powering off - billing stops once GCE reports TERMINATED"
  sudo poweroff
fi

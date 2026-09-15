#!/bin/bash
# Auto-retrying wrapper for an idempotent Phase 4 training script (one that
# skips already-completed named runs by checking for their saved log on
# disk - resume_round2_seeds.py, run_model_selection.py, etc). Written
# 2026-09-15 after two unattended-run crashes on external-drive I/O
# hiccups that needed a human to notice and manually relaunch. This keeps
# retrying the same command on any non-zero exit, with backoff, until it
# succeeds - safe specifically because re-invoking a script that already
# skips finished work never redoes expensive already-completed training.
#
# Usage: ./run_resilient.sh <python-script.py> [args...]
#
# 2026-09-15 addendum: a bare unmount/remount of the external drive can
# leave an already-running shell's cwd pointing at a dead inode from the
# old mount instance - confirmed via `lsof -p <pid> | grep cwd` showing
# "No such file or directory" - even though the *path* is perfectly valid
# again for a fresh process. A relative `python3 script.py` in that shell
# then fails "No such file or directory" forever, no matter how many times
# it retries, because the stale cwd handle never heals itself. Fix: `cd`
# to an absolute path fresh at the top of every attempt, not once outside
# the loop - each `cd` gets a brand new directory handle against whatever
# mount is live right now.
set -uo pipefail

TRAINING_DIR="/Volumes/Hard Drive/FujiApp Project/FujiApp-master/training"
MAX_ATTEMPTS=20
BACKOFF=30

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  echo "=== run_resilient: attempt $attempt/$MAX_ATTEMPTS: python3 $* ==="
  cd "$TRAINING_DIR" || { echo "=== run_resilient: cd failed, drive likely still down, retrying in ${BACKOFF}s ==="; sleep "$BACKOFF"; attempt=$((attempt + 1)); continue; }
  python3 "$@"
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "=== run_resilient: succeeded on attempt $attempt ==="
    exit 0
  fi
  echo "=== run_resilient: attempt $attempt failed (exit $status), retrying in ${BACKOFF}s ==="
  sleep "$BACKOFF"
  attempt=$((attempt + 1))
  # gentle backoff growth, capped, so a persistent (non-transient) failure
  # doesn't hammer the drive/process every 30s forever
  if [ "$BACKOFF" -lt 300 ]; then
    BACKOFF=$((BACKOFF * 2))
  fi
done

echo "=== run_resilient: gave up after $MAX_ATTEMPTS attempts ==="
exit 1

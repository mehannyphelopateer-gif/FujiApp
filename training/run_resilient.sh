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
set -uo pipefail

MAX_ATTEMPTS=20
BACKOFF=30

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  echo "=== run_resilient: attempt $attempt/$MAX_ATTEMPTS: python3 $* ==="
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

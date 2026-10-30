#!/bin/bash
# Strategy chain F4 -- the driver (systemd --user service strategy-chain-f4.service). Stages in order; a done stage is
# skipped; a stage that failed twice (one automatic retry) writes FAILED_<stage> and the driver stops cleanly (exit 0, so
# Restart=on-failure does not loop). $C/STOP is honoured between stages (and propagated by the stages between tasks).
# Every transition rewrites STATUS.md / STATUS.json (a timer also rewrites them every 5 minutes).
set -u
export CHAIN_ROOT=${CHAIN_ROOT:-$(cd "$(dirname "$0")" && pwd)}
STAGE=driver
source "$CHAIN_ROOT/lib.sh"
status(){ (cd "$CODE" && $PY tools/strategy_chain_f4.py status --chain "$C" --quiet) >> "$C/logs/status.log" 2>&1; }
log "driver start (code $CODE_SHA)"
for S in S1 S1R G2 S2 S3 Q R; do
  STAGE=$S
  [ -f "$C/state/$S.done" ] && continue
  [ -f "$C/state/FAILED_$S" ] && { log "FAILED_$S present: the driver stops (see the runbook to retry)"; status; exit 0; }
  while true; do
    stopped && { log "STOP flag: driver exits before $S"; status; exit 0; }
    n=$(cat "$C/state/$S.attempts" 2>/dev/null || echo 0)
    echo $((n + 1)) > "$C/state/$S.attempts"
    state started
    status
    log "stage $S attempt $((n + 1))"
    bash "$C/stages/$S.sh" >> "$C/logs/$S.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      [ -f "$C/state/$S.done" ] || state done
      log "stage $S done"
      status
      break
    elif [ $rc -eq 75 ]; then
      log "stage $S stopped by the STOP flag (resumable)"
      rm -f "$C/state/$S.attempts"                                  # a stop is not a failed attempt
      status
      exit 0
    else
      reason=$(tail -1 "$C/state/$S.reason" 2>/dev/null || echo "rc=$rc (see logs/$S.log)")
      state failed "" "" "$reason"
      if [ $rc -eq 11 ] || [ $((n + 1)) -ge 2 ]; then
        state FAILED "" "" "$reason"
        log "FAILED_$S: $reason"
        status
        exit 0
      fi
      log "stage $S failed (rc=$rc): one automatic retry"
    fi
  done
done
STAGE=driver
log "chain complete"
status
exit 0

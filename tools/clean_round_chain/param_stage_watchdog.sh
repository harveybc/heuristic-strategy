#!/bin/bash
# Watchdog of the strategy parameter stage (coordinator orders 2026-10-27 / 28, after a silent memcg OOM of the DEAP
# pool and a false stall alarm during a long pool build). Every POLL seconds:
#   * a new "[param-stage] DEAP FAILED" line in LOG                                   -> failure
#   * the DEAP process is running, the pool is NOT ready yet and its build has lasted more than BUILD_LIMIT
#     (2 x the measured build time of the last run, default 1800 s when none is known) -> failure
#   * the pool is ready ("pool_ready" logged) and the ARCHIVE has not grown for STALL seconds (default 1200;
#     the clock starts when the pool is ready, never during the admission wait or the build) -> failure
# A failure writes an ALERT line. The watchdog NEVER sends a signal (coordinator order 2026-10-28: process control only
# through flag files and new launches; a stop needing a signal goes to the owner): when the DEAP process is still
# alive it writes DEAP_NEEDS_STOP with the pids and the exact command, and exits; when it is gone (a failed or
# finished-with-error run) it resumes the stage ONCE from the last checkpoint with workers - 2; a second failure
# writes DEAP_GAVE_UP (the slot plan then frees the slots) and exits. STOP_WATCHDOG in the root: exit at the next poll.
# Usage: param_stage_watchdog.sh <matrix root>
#   env: JOBS_DEAP MEM_DEAP CODE VERIFY_PREFIX WARM_ARCHIVE JOBS_VERIFY (passed to the resume), POLL STALL BUILD_LIMIT_SEC,
#        DRY_RUN=1 (alert only: no stop, no resume -- for the watchdog's own test)
set -u
R=$1; OUT=$R/deap; J=${JOBS_DEAP:-6}; resumed=0
POLL=${POLL:-60}; STALL=${STALL:-1200}
log(){ echo "$(date -u +%FT%TZ) [param-watchdog] $*" >> $R/LOG; }
last_build=$(grep -h "pool_ready_sec" $R/deap_run*_superseded/PROGRESS.log 2>/dev/null | python3 -c "
import json, sys
v = [json.loads(l).get('pool_ready_sec') for l in sys.stdin if l.strip().startswith('{')]
v = [x for x in v if x]
print(int(v[-1]) if v else 0)" 2>/dev/null)
BUILD_LIMIT=${BUILD_LIMIT_SEC:-$(( ${last_build:-0} > 0 ? 2 * last_build : 1800 ))}
deap_pids(){ ps -eo pid,args | awk -v s="strategy_param_deap.py run --subset $OUT/SUBSET.json" '$2 ~ /python/ && index($0, s) {print $1}'; }
n0=$(grep -c "\[param-stage\] DEAP FAILED" $R/LOG)
log "armed: stage failure lines so far $n0; build limit ${BUILD_LIMIT} s (2 x the last measured build ${last_build:-?} s), stall limit ${STALL} s after pool_ready; one resume with workers - 2${DRY_RUN:+ (DRY RUN)}"
build_t0=""; ready_t0=""
while [ ! -f $OUT/RESULT.json ]; do
  sleep $POLL
  [ -f $R/STOP_WATCHDOG ] && { log "STOP_WATCHDOG: exit"; exit 0; }
  [ -f $OUT/FAILED_DEGENERATE_GEN0 ] && { log "FAILED_DEGENERATE_GEN0 present: the search stopped itself; no resume; watchdog exits"; exit 0; }
  now=$(date +%s)
  n=$(grep -c "\[param-stage\] DEAP FAILED" $R/LOG)
  pids=$(deap_pids)
  why=""
  [ "$n" -gt "$n0" ] && why="DEAP FAILED in LOG"
  if [ -z "$why" ] && [ -n "$pids" ]; then
    [ -z "$build_t0" ] && build_t0=$now                         # the DEAP process exists (admitted): the build clock starts
    if grep -q "pool_ready" $OUT/PROGRESS.log 2>/dev/null && [ "$(stat -c %Y $OUT/PROGRESS.log)" -ge "$(( build_t0 - POLL ))" ]; then
      [ -z "$ready_t0" ] && ready_t0=$now
      m=$(stat -c %Y $OUT/ARCHIVE.jsonl 2>/dev/null || echo 0); [ "$m" -lt "$ready_t0" ] && m=$ready_t0
      [ $(( now - m )) -gt $STALL ] && why="no ARCHIVE growth for $(( (now - m) / 60 )) min after the pool was ready"
    else
      [ $(( now - build_t0 )) -gt $BUILD_LIMIT ] && why="pool build exceeded ${BUILD_LIMIT} s"
    fi
  elif [ -z "$pids" ]; then
    build_t0=""; ready_t0=""                                    # admission wait or between runs: no clock runs
  fi
  [ -z "$why" ] && continue
  log "ALERT: $why (archive $( [ -f $OUT/ARCHIVE.jsonl ] && wc -l < $OUT/ARCHIVE.jsonl || echo 0) evaluations, checkpoint generation $(python3 -c "import pickle; print(pickle.load(open('$OUT/CHECKPOINT.pkl','rb'))['gen'])" 2>/dev/null))"
  if [ -n "${DRY_RUN:-}" ]; then
    log "DRY RUN: no stop, no resume; watchdog exits"; exit 0
  fi
  if [ $resumed -ge 1 ]; then
    touch $R/DEAP_GAVE_UP; log "ALERT: second failure after the resume: DEAP_GAVE_UP written, no further resume"; exit 1
  fi
  alive=$(deap_pids | tr '\n' ' ')
  if [ -n "${alive// /}" ]; then
    echo "$(date -u +%FT%TZ) $why; DEAP pids still alive: $alive; owner command: kill -TERM $alive" > $R/DEAP_NEEDS_STOP
    log "ALERT: the DEAP is still alive (pids $alive): DEAP_NEEDS_STOP written for the owner; no signal sent; watchdog exits"; exit 1
  fi
  J=$(( J > 4 ? J - 2 : 2 )); resumed=1; n0=$(grep -c "\[param-stage\] DEAP FAILED" $R/LOG); build_t0=""; ready_t0=""
  log "resuming the stage from the checkpoint with $J workers"
  FORCE_START=1 TIME_DEAP=${TIME_DEAP:-43200} JOBS_DEAP=$J MEM_DEAP=${MEM_DEAP:-15G} CODE=${CODE:-$R/code} JOBS_VERIFY=${JOBS_VERIFY:-10} VERIFY_PREFIX=${VERIFY_PREFIX:-verify} WARM_ARCHIVE=${WARM_ARCHIVE:-} setsid nohup bash $R/code/tools/clean_round_chain/run_strategy_param_stage.sh $R >> $R/logs/param_stage_resume.log 2>&1 < /dev/null &
done
log "RESULT.json present: watchdog exits"

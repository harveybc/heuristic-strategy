#!/bin/bash
# Watchdog of the strategy parameter stage (coordinator order 2026-10-27, after a silent memcg OOM of the DEAP pool):
# every 60 s; on a new "DEAP FAILED" line in LOG, or no ARCHIVE growth for 20 min while the DEAP should be running, it
# writes an ALERT line to LOG and resumes the stage ONCE from the last checkpoint with workers - 2 (its own DEAP job is
# stopped first when it hangs). A second failure writes DEAP_GAVE_UP (the slot plan then frees the slots) and exits.
# Usage: param_stage_watchdog.sh <matrix root>   env: JOBS_DEAP MEM_DEAP CODE (the DEAP's code root) passed on
set -u
R=$1; OUT=$R/deap; J=${JOBS_DEAP:-6}; resumed=0; t_ref=$(date +%s)   # stall clock: from the arming / the resume, not from an old ARCHIVE
log(){ echo "$(date -u +%FT%TZ) [param-watchdog] $*" >> $R/LOG; }
n0=$(grep -c "\[param-stage\] DEAP FAILED" $R/LOG)
log "armed: stage failure lines so far $n0; stall limit 20 min; one resume with workers - 2"
while [ ! -f $OUT/RESULT.json ]; do
  sleep 60
  n=$(grep -c "\[param-stage\] DEAP FAILED" $R/LOG)
  m=$(stat -c %Y $OUT/ARCHIVE.jsonl 2>/dev/null || echo 0); [ "$m" -lt "$t_ref" ] && m=$t_ref
  age=$(( $(date +%s) - m ))
  running=$(pgrep -fc "strategy_param_deap.py run --subset $OUT/SUBSET.json")
  why=""
  [ "$n" -gt "$n0" ] && why="DEAP FAILED in LOG"
  [ -z "$why" ] && [ "$running" -gt 0 ] && [ $age -gt 1200 ] && why="no ARCHIVE growth for $((age / 60)) min"
  [ -z "$why" ] && continue
  log "ALERT: $why (archive $(wc -l < $OUT/ARCHIVE.jsonl) evaluations, checkpoint generation $(python3 -c "import pickle; print(pickle.load(open('$OUT/CHECKPOINT.pkl','rb'))['gen'])" 2>/dev/null))"
  if [ $resumed -ge 1 ]; then
    touch $R/DEAP_GAVE_UP; log "ALERT: second failure after the resume: DEAP_GAVE_UP written, no further resume"; exit 1
  fi
  pkill -TERM -f "strategy_param_deap.py run --subset $OUT/SUBSET.json"; sleep 30
  J=$(( J > 4 ? J - 2 : 2 )); resumed=1; n0=$(grep -c "\[param-stage\] DEAP FAILED" $R/LOG); t_ref=$(date +%s)
  log "resuming the stage from the checkpoint with $J workers"
  FORCE_START=1 JOBS_DEAP=$J MEM_DEAP=${MEM_DEAP:-15G} CODE=${CODE:-$R/code} JOBS_VERIFY=${JOBS_VERIFY:-10} VERIFY_PREFIX=${VERIFY_PREFIX:-verify} WARM_ARCHIVE=${WARM_ARCHIVE:-} setsid nohup bash $R/code/tools/clean_round_chain/run_strategy_param_stage.sh $R >> $R/logs/param_stage_resume.log 2>&1 < /dev/null &
done
log "RESULT.json present: watchdog exits"

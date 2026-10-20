#!/bin/bash
# Autonomous complexification sweep of the temporal ladder, run ON the GPU worker, detached (setsid nohup):
#   for u in UNITS: E1-T (BC+DAgger) -> E2-T (SAC +/- demos), sequential, one job at a time, 2024 never read.
# Resumable: an arm whose result JSON exists is skipped, so relaunching the same command continues the sweep.
# A failing step is retried once, then the sweep moves on (the failure is recorded in SWEEP_LOG).
# After every step the summary (tables + capacity curve) is regenerated; SWEEP_DONE marks the end.
# Usage: bash tools/expert_guided_sweep_t.sh <ROOT> <DATA> <BASELINE_U2_DIR> [UNITS]
set -u
ROOT=$1; DATA=$2; BASE=$3; UNITS=${4:-"4 8 16 32 64"}
CODE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
CR=${CR:-crispdm-run}
LOG=$ROOT/SWEEP_LOG
mkdir -p "$ROOT"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
summary() {
  $CR -q -m 1G -t 300s -n sweep-sum -- $PY "$CODE/tools/expert_guided_sweep_summary_t.py" --root "$ROOT" \
    --baseline-u2 "$BASE" --units "2,$(echo $UNITS | tr ' ' ',')" >> "$LOG" 2>&1
}
step() {  # name, result json, command...
  local name=$1 out=$2; shift 2
  if [ -f "$out" ]; then say "skip $name (done)"; return 0; fi
  for attempt in 1 2; do
    say "start $name attempt $attempt"
    "$@" > "$ROOT/$name.log" 2>&1
    local rc=$?
    say "end $name attempt $attempt rc=$rc"
    [ -f "$out" ] && return 0
  done
  say "FAILED $name (moving on)"
  return 1
}
rm -f "$ROOT/SWEEP_DONE"
say "sweep start units=[$UNITS] code=$CODE"
cd "$CODE"
for U in $UNITS; do
  D=$ROOT/u$U
  step "e1t_u$U" "$D/e1t/E1T.json" $CR -q -m 2G -t 21600s -n e1t-u$U -- $PY tools/expert_guided_e1t.py run \
    --admitted "$DATA/admitted" --csv-dir "$DATA/sel15" --v4-dir "$DATA/v4" --out "$D/e1t" --features core15 \
    --units $U --max-epochs 1000 --skip-2024
  summary
  if [ -f "$D/e1t/E1T.json" ]; then
    step "e2t_u$U" "$D/e2t/E2T.json" $CR -q -m 3G -t 43200s -n e2t-u$U -- $PY tools/expert_guided_e2t.py run \
      --admitted "$DATA/admitted" --csv-dir "$DATA/sel15" --v4-dir "$DATA/v4" --e1-run "$D/e1t" --out "$D/e2t" \
      --features core15 --units $U --skip-2024
    summary
  fi
done
say "sweep done"
date -u +%FT%TZ > "$ROOT/SWEEP_DONE"

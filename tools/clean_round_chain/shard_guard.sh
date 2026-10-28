#!/bin/bash
# Memory guard of a matrix shard on a shared worker (coordinator order 2026-10-27): every 60 s it reads the host's
# admission state; when the memory free for new loads (MemAvailable minus the live reservations and the desktop
# reserve, the admission controller's own figure) stays below zero for 3 checks, or memory pressure exceeds the
# admission limit, it writes STOP (the shard driver starts no new cell) and RETURNED, so the main host takes the shard's
# remaining cells back. It NEVER sends a signal (coordinator order 2026-10-28): the cell jobs already running finish
# under their own crispdm caps; a stop that needs a signal goes to the owner. Exits when the shard driver is gone.
# Usage: shard_guard.sh <shard root>
set -u
R=$1; RES=${RES:-$R/results_v10}
A=~/.local/libexec/crispdm/crispdm_admission.py
log(){ echo "$(date -u +%FT%TZ) [shard-guard] $*" >> $R/LOG; }
bad=0
log "guard start"
while pgrep -f "^bash $R/matrix_driver_v2.sh" >/dev/null; do
  st=$(python3 $A state 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['host_free_for_new_bytes'], d['pressure_some_avg10'], d['pressure_admit_max'])" 2>/dev/null)
  read -r free psi pmax <<< "${st:-0 0 25}"
  if [ "${free%.*}" -lt 0 ] || [ "$(python3 -c "print(int(float('$psi') > float('$pmax')))")" = "1" ]; then bad=$((bad + 1)); else bad=0; fi
  if [ $bad -ge 3 ]; then
    touch $R/STOP
    echo "$(date -u +%FT%TZ) free_for_new=$free psi=$psi" > $R/RETURNED
    log "memory below the shard's need (free_for_new=$free bytes, psi=$psi): STOP and RETURNED written (no signal sent)"
    exit 0
  fi
  sleep 60
done
log "driver gone: guard exits"

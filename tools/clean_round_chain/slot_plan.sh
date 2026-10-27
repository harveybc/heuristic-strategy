#!/bin/bash
# Slot plan of the main host (coordinator orders 2026-10-27). When the DEAP writes RESULT.json, ALL batch slots go to
# the DEAP-point re-verification (verify_k0: 10 jobs x 1.5 GB) and the canonical v10 stops launching here (it goes on
# on the shard only). When the DEAP is given up (DEAP_GAVE_UP, written by the stage watchdog after its one resume
# failed), the slots go back to the canonical v10 here (10 jobs) and an ALERT line is written: no idle slots either way.
# Usage: slot_plan.sh <matrix root>
set -u
R=$1; VK=${VERIFY_ROOT:-$R/verify_k0}; OLDV=${OLD_VERIFY_ROOT:-}   # the re-verification root of this DEAP run; an earlier run's root goes to 0
log(){ echo "$(date -u +%FT%TZ) [slot-plan] $*" >> $R/LOG; }
log "armed: RESULT.json -> $(basename $VK) 10 jobs${OLDV:+, $(basename $OLDV) 0}, v10 0 here; DEAP_GAVE_UP -> v10 10 jobs here"
while true; do
  if [ -f $R/deap/RESULT.json ]; then
    echo 0 > $R/JOBS_OVERRIDE; mkdir -p $VK; echo 10 > $VK/JOBS_OVERRIDE; [ -n "$OLDV" ] && echo 0 > $OLDV/JOBS_OVERRIDE
    log "DEAP done: main-host v10 to 0 jobs (records so far kept: defect discovery + canonical-vs-optimised), $(basename $VK) to 10 jobs${OLDV:+, $(basename $OLDV) to 0}; ranks 2+ held by STOP until a decision"
    exit 0
  fi
  if [ -f $R/DEAP_GAVE_UP ]; then
    echo 10 > $R/JOBS_OVERRIDE
    log "ALERT: DEAP given up: the slots return to the canonical v10 here (10 jobs) until a decision"
    exit 1
  fi
  sleep 60
done

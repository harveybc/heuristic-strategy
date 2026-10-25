#!/bin/bash
# Early-close audit chain (lane L9, 2026-10-25; owner: early close is indispensable). CPU only, sequential, resumable
# (one JSON per unit), memory-capped; never touches the v6 chain, its STATE, the package code/ or PINS. Step 1: the
# baseline walk-forward units (validation year INNER_B) and A_zz_hold / A_both_hold (INNER_A) -> report. Step 2: waits
# for "[chain-v6] chain end (v6)" and audits the four WF_V1 units (INNER_A calibration block + INNER_B) -> report.
Q=/var/tmp/expert-guided-clean-a1-20261024; W=/var/tmp/early-close-audit-20261025
PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run
mkdir -p $W/out
log(){ echo "$(date -u +%FT%TZ) [early-close] $*" >> $Q/LOG; }
cd $W/code || exit 1
au(){ # unit recipe split periods [extra dirs...]
  local u=$1 r=$2 sp=$3 per=$4; shift 4; local ex=""; for d in "$@"; do ex="$ex --extra-arm-dir $Q/out/$d"; done
  [ -f $W/out/EARLY_CLOSE_$u.json ] && return 0
  [ -f $Q/out/$u/E3M.json ] || { log "audit $u skipped: no E3M.json"; return 0; }
  for try in $(seq 1 36); do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $CR -q -m 4G -t 14400s -n ecaudit-$u -- $PY tools/expert_guided_early_close_audit.py run --arm-dir $Q/out/$u $ex \
      --admitted $Q/data/admitted --store $Q/data/store/full_pool_eurusd_h1.parquet --recipe $Q/recipe/$r --split-contract $sp --periods $per \
      --out $W/out/EARLY_CLOSE_$u.json > $W/out/$u.log 2>&1 && { log "audit $u DONE"; return 0; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $W/out/$u.log && { log "audit $u not admitted (try $try)"; sleep 600; } || { log "audit $u FAILED (see $W/out/$u.log)"; return 1; }
  done
}
log "chain start: baseline audit -> report -> wait for the v6 end -> WF_V1 audit -> report"
au A_zz_hold CONSUMER_RECIPE.json consolidated INNER_A
au A_both_hold CONSUMER_RECIPE.json consolidated INNER_A A_both_hold_seeds2
for f in wf2022 wf2023; do for a in zz_hold both_hold both_hold_shorter; do au WF_${f}_${a} CONSUMER_RECIPE_${f}.json $f INNER_B; done; done
B=$(ls $W/out/EARLY_CLOSE_A_*.json $W/out/EARLY_CLOSE_WF_wf*.json 2>/dev/null | grep -v WF_V1)
$PY tools/expert_guided_early_close_audit.py report --units $B --out-json $Q/report/EARLY_CLOSE_AUDIT_BASELINE.json --out-md $Q/report/EARLY_CLOSE_AUDIT_BASELINE.md \
  --title "Early-close audit, baseline (walk-forward validation years; A arms on INNER_A, biased by D1, reference only)" > $W/out/report_baseline.log 2>&1 \
  && log "baseline report written" || log "baseline report FAILED (see $W/out/report_baseline.log)"
for i in $(seq 1 288); do grep -q "\[chain-v6\] chain end (v6)" $Q/LOG && break; sleep 300; done
grep -q "\[chain-v6\] chain end (v6)" $Q/LOG || { log "v6 chain did not end within 24 h: WF_V1 audit not run"; exit 5; }
for f in wf2022 wf2023; do for a in both_hold_v1 zz_hold_v1; do au WF_V1_${f}_${a} CONSUMER_RECIPE_${f}.json $f INNER_A,INNER_B; done; done
V=$(ls $W/out/EARLY_CLOSE_WF_V1_*.json 2>/dev/null)
[ -n "$V" ] && $PY tools/expert_guided_early_close_audit.py report --units $V --out-json $Q/report/EARLY_CLOSE_AUDIT_V1.json --out-md $Q/report/EARLY_CLOSE_AUDIT_V1.md \
  --title "Early-close audit, WF_V1 (action contract v1; INNER_A = calibration block, INNER_B = validation year; every fitness variant of the ensemble)" > $W/out/report_v1.log 2>&1 \
  && log "WF_V1 report written" || log "WF_V1 report FAILED or no unit (see $W/out/report_v1.log)"
log "chain end (early-close)"

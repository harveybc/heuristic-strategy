#!/bin/bash
# Trading-detail report chain (lane L13, 2026-10-25; owner: full trading details before any comparison). CPU only,
# sequential, resumable (one JSON per unit), memory-capped; runs from its own code copy and never touches the package
# code/, PINS, STATE or the running chains. Step 1: the four WF_V13 units -> report. Step 2: waits for
# "[chain-v13] chain end", then the WF_C2_13 and WF_C2G_13 units -> report.
Q=/var/tmp/expert-guided-clean-a1-20261024; W=/var/tmp/trading-detail-20261026
PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run
mkdir -p $W/out $Q/report/trades
log(){ echo "$(date -u +%FT%TZ) [trading-detail] $*" >> $Q/LOG; }
cd $W/code || exit 1
td(){ # unit recipe split
  local u=$1 r=$2 sp=$3
  [ -f $W/out/TRADING_$u.json ] && return 0
  [ -f $Q/out/$u/E3M.json ] || { log "detail $u skipped: no E3M.json"; return 0; }
  for try in $(seq 1 36); do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $CR -q -m 4G -t 14400s -n trdetail-$u -- $PY tools/expert_guided_trading_report.py run --arm-dir $Q/out/$u \
      --admitted $Q/data/admitted --store $Q/data/store/full_pool_eurusd_h1.parquet --recipe $Q/recipe/$r --split-contract $sp \
      --periods INNER_A,INNER_B,OBS_2024 --csv-dir $Q/report/trades --out $W/out/TRADING_$u.json > $W/out/$u.log 2>&1 && { log "detail $u DONE"; return 0; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $W/out/$u.log && { log "detail $u not admitted (try $try)"; sleep 600; } || { log "detail $u FAILED (see $W/out/$u.log)"; return 1; }
  done
}
log "chain start: WF_V13 -> report -> wait for the v13 chain end -> WF_C2_13 / WF_C2G_13 -> report"
for f in wf2022 wf2023; do for a in zz_hold_v13 both_hold_v13; do td WF_V13_${f}_$a CONSUMER_RECIPE_${f}.json $f; done; done
V=$(ls $W/out/TRADING_WF_V13_*.json 2>/dev/null)
[ -n "$V" ] && $PY tools/expert_guided_trading_report.py report --units $V --out-json $Q/report/TRADING_DETAIL_V13.json --out-md $Q/report/TRADING_DETAIL_V13.md \
  --title "Trading detail, WF_V13 (contract v1.3; seeds and seed ensemble, primary fitness)" > $W/out/report_v13.log 2>&1 \
  && log "V13 report written" || log "V13 report FAILED (see $W/out/report_v13.log)"
for i in $(seq 1 288); do grep -q "\[chain-v13\] chain end" $Q/LOG && break; sleep 300; done
grep -q "\[chain-v13\] chain end" $Q/LOG || { log "v13 chain did not end within 24 h: C2/C2G detail not run"; exit 5; }
for f in wf2022 wf2023; do td WF_C2_13_${f}_zz_feat_c2_13 CONSUMER_RECIPE_${f}.json $f; td WF_C2G_13_${f}_zz_feat_c2g_13 CONSUMER_RECIPE_${f}.json $f; done
C=$(ls $W/out/TRADING_WF_C2_13_*.json $W/out/TRADING_WF_C2G_13_*.json 2>/dev/null)
[ -n "$C" ] && $PY tools/expert_guided_trading_report.py report --units $C --out-json $Q/report/TRADING_DETAIL_C2_C2G_13.json --out-md $Q/report/TRADING_DETAIL_C2_C2G_13.md \
  --title "Trading detail, WF_C2_13 / WF_C2G_13 (consequence learning; seeds and seed ensemble, primary fitness)" > $W/out/report_c2.log 2>&1 \
  && log "C2/C2G report written" || log "C2/C2G report FAILED (see $W/out/report_c2.log)"
log "chain end (trading-detail)"

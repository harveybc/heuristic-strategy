#!/bin/bash
# S2 -- stage-2 DEAP over the non-level genes (GENOME_F4_STAGE2: open_rate, z_veto, burst_vol_ratio [1.2, 2.5], the
# closer genes ...) at the frozen level point: Sharpe only with Deb constraints, POP x GEN, from scratch (the canonical
# genome in generation 0), gen-0 degeneracy gate, resumable from CHECKPOINT.pkl, STOP between generations.
STAGE=S2; source "$CHAIN_ROOT/lib.sh"
[ -f "${QUAL_DIR:-$R/qual}/PROPERTY_PASSED_$CODE_SHA" ] || { reason "gate missing: PROPERTY_PASSED_$CODE_SHA"; exit 11; }
deap(){ # out params [warm]
  local O=$1 P=$2 WA=${3:-}
  if [ ! -f "$O/RESULT.json" ]; then
    [ -f "$O/FAILED_DEGENERATE_GEN0" ] && { reason "FAILED_DEGENERATE_GEN0 ($O)"; return 11; }
    mkdir -p "$O"; rm -f "$O/STOP"
    bg_stop "$O/STOP" -- crun "f4chain-$(basename $O)" "$S2_MEM" 432000s -- bash -c "cd '$CODE' && BEHAVIOUR_MATRIX_ALLOWED=1 $PY tools/strategy_param_deap.py run --subset '$SUBSET' --params '$P' --freeze-level-genes --population $POP --generations $GEN --jobs $S2_JOBS --admitted '$DATA/admitted' --csv-dir '$DATA/sel15' ${WA:+--warm-archive '$WA'} --out '$O' >> '$O/run.log' 2>&1"
    local rc=$?
    [ -f "$O/FAILED_DEGENERATE_GEN0" ] && { reason "FAILED_DEGENERATE_GEN0 ($O)"; return 11; }
    stopped && return 75
    [ -f "$O/RESULT.json" ] || { reason "DEAP ended rc=$rc without RESULT.json ($O/run.log)"; return 1; }
  fi
  (cd "$CODE" && $PY tools/strategy_param_deap.py best --archive "$O/ARCHIVE.jsonl" --params "$P" --k 0 --allow-infeasible --out "$O/CHAMPION.json") > "$O/best.log" 2>&1 || { reason "best point extraction failed"; return 1; }
  return 0
}
deap "$W/S2" "$W/S1R/FROZEN_PARAMS.json"; rc=$?
[ $rc -eq 0 ] || exit $rc
state done "$W/S1R/FROZEN_PARAMS.json,$SUBSET" "$W/S2/ARCHIVE.jsonl,$W/S2/RESULT.json,$W/S2/CHAMPION.json"

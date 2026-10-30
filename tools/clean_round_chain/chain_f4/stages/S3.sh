#!/bin/bash
# S3 -- the top S3_K stage-1 points re-evaluated with the stage-2 champion's genes; another point (no worse on the
# constraint) winning by more than 1 SE of the paired per-world Sharpe difference -> stage 2 re-run ONCE from it (S2b,
# warm start = genomes of the S2 archive); then stop iterating. FINAL_CHAMPION.json is the result.
STAGE=S3; source "$CHAIN_ROOT/lib.sh"
O=$W/S3; mkdir -p "$O/eval"
cd "$CODE" || exit 1
$PY tools/strategy_chain_f4.py top5 --freeze "$W/S1R/FREEZE.json" --k $S3_K --out "$O/TOP5.json" || exit 1
if ! grep -q '"done": true' "$O/eval/PROGRESS.log" 2>/dev/null; then
  stopped && exit 75
  rm -f "$O/eval/STOP"
  bg_stop "$O/eval/STOP" -- crun f4chain-S3 "$S1_MEM" 86400s -- bash -c "cd '$CODE' && $PY tools/strategy_level_grid.py run --subset '$SUBSET' --params '$W/S2/CHAMPION.json' --points-json '$O/TOP5.json' --admitted '$DATA/admitted' --csv-dir '$DATA/sel15' --jobs $S1_JOBS --out '$O/eval' >> '$O/eval.log' 2>&1"
  stopped && exit 75
  grep -q '"done": true' "$O/eval/PROGRESS.log" 2>/dev/null || { reason "stage-3 evaluation incomplete ($O/eval.log)"; exit 1; }
fi
$PY tools/strategy_level_grid.py compare --out "$O/eval" --frozen 0 > "$O/compare.log" 2>&1 || { reason "stage-3 compare failed"; exit 1; }
$PY tools/strategy_chain_f4.py s3 --stage3 "$O/eval/STAGE3_RESULT.json" --champion "$W/S2/CHAMPION.json" --out "$O/S3_DECISION.json" --out-params "$O/RERUN_PARAMS.json" || exit 1
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$O/S3_DECISION.json'))['rerun_stage2'] else 1)"; then
  STAGE=S3
  O2=$W/S2b
  if [ ! -f "$O2/RESULT.json" ]; then
    [ -f "$O2/FAILED_DEGENERATE_GEN0" ] && { reason "FAILED_DEGENERATE_GEN0 (S2b)"; exit 11; }
    mkdir -p "$O2"; rm -f "$O2/STOP"
    bg_stop "$O2/STOP" -- crun f4chain-S2b "$S2_MEM" 432000s -- bash -c "cd '$CODE' && BEHAVIOUR_MATRIX_ALLOWED=1 $PY tools/strategy_param_deap.py run --subset '$SUBSET' --params '$O/RERUN_PARAMS.json' --freeze-level-genes --population $POP --generations $GEN --jobs $S2_JOBS --admitted '$DATA/admitted' --csv-dir '$DATA/sel15' --warm-archive '$W/S2/ARCHIVE.jsonl' --out '$O2' >> '$O2/run.log' 2>&1"
    [ -f "$O2/FAILED_DEGENERATE_GEN0" ] && { reason "FAILED_DEGENERATE_GEN0 (S2b)"; exit 11; }
    stopped && exit 75
    [ -f "$O2/RESULT.json" ] || { reason "S2b ended without RESULT.json"; exit 1; }
  fi
  $PY tools/strategy_param_deap.py best --archive "$O2/ARCHIVE.jsonl" --params "$O/RERUN_PARAMS.json" --k 0 --allow-infeasible --out "$O2/CHAMPION.json" > "$O2/best.log" 2>&1 || exit 1
  cp "$O2/CHAMPION.json" "$W/FINAL_CHAMPION.json"
  echo '{"final_from": "S2b (stage-3 winner)"}' > "$O/FINAL_FROM.json"
else
  cp "$W/S2/CHAMPION.json" "$W/FINAL_CHAMPION.json"
  echo '{"final_from": "S2 (frozen point kept)"}' > "$O/FINAL_FROM.json"
fi
state done "$W/S1R/FREEZE.json,$W/S2/CHAMPION.json" "$O/S3_DECISION.json,$W/FINAL_CHAMPION.json"

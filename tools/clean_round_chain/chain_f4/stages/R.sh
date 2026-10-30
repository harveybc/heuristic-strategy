#!/bin/bash
# R -- REPORT_F4.md + REPORT_F4.json + the q-ladder plots (Sharpe and profit against the head MAE and MAE / naive),
# assembled from the chain's files only; the residual list under "most likely wrong"; the FINAL marker last.
STAGE=R; source "$CHAIN_ROOT/lib.sh"
cd "$CODE" || exit 1
$PY tools/strategy_chain_f4.py report --chain "$C" --out "$W/R" > "$W/R.log" 2>&1 || { reason "report failed ($W/R.log)"; exit 1; }
cp "$W/R/REPORT_F4.md" "$C/REPORT_F4.md"
state done "$W/FINAL_CHAMPION.json,$W/Q/D9.json" "$W/R/REPORT_F4.md,$W/R/REPORT_F4.json"
python3 -c "import json,time; print(json.dumps({'final': True, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'report': '$C/REPORT_F4.md', 'code_sha': '$CODE_SHA'}))" > "$C/FINAL"

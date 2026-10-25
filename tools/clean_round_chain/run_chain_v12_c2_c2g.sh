#!/bin/bash
# clean round v9 chain (lane L11, 2026-10-25): ONE sequential, resumable chain that replaces the stopped v6 / early-close /
# c2 drivers. WF_V1 had zero trades (strategy space unreachable) and is SUPERSEDED_INFEASIBLE_STRATEGY_SPACE; this chain runs
#   1. install the STAGED files (verified against stage_v12/STAGE.sha256), README_LAUNCH.txt / LAUNCH_STATUS.md out of PINS
#   2. gate stages += WF_V12, WF_C2, WF_C2G (owner's words)
#   3. WF_V12 (contract v1.2: scale-adaptive genes) units -> sanity gate -> ensembles -> sanity gate -> early-close audit -> gate
#   4. C2 outcome tables -> WF_C2 units -> gate -> ensembles -> gate -> audit -> gate
#   5. WF_C2G units (C2 + closer-only order geometry) on the same outcome tables (digests recorded, re-checked) -> same steps
#   6. reports (WF_V12 + closure D1-D6/E, WF_C2, WF_C2G, early close, sanity of every unit), each with its sanity table
#   7. "[chain-v12] chain end"
# FAILED_SANITY units are excluded from every comparison (the report tools refuse a verdict that involves one). New STATE
# marks only (v12_*, and the new unit names); the old WF_V1 marks are never reused. Nothing here asks a human.
Q=/var/tmp/expert-guided-clean-a1-20261024; PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run
S=$Q/stage_v12; SPEC=$Q/CLEAN_ROUND_SPEC_v9_host.json; REP=$Q/report; ECD=$REP/early_close_v12; ECB=/var/tmp/early-close-audit-20261025/out
SG="$PY tools/expert_guided_sanity_gate.py"
mkdir -p $REP $Q/STATE $Q/contracts $Q/outcomes $ECD
log(){ echo "$(date -u +%FT%TZ) [chain-v12] $*" >> $Q/LOG; }
test -f $Q/LAUNCH_EXPERT_GUIDED.json || { log "gate missing: refusing"; exit 3; }
log "chain start: install -> WF_V12 -> WF_C2 -> WF_C2G -> reports (sanity gate after every unit and ensemble)"

# 1. install (verified), docs out of PINS, gate extended
( cd $S && sha256sum -c --quiet STAGE.sha256 ) || { log "staged files differ from STAGE.sha256: refusing"; exit 6; }
if [ ! -f $Q/STATE/v12_installed ]; then
  while read -r h f; do install -D -m 0644 "$S/$f" "$Q/$f" || { log "install $f FAILED"; exit 6; }; done < $S/STAGE.sha256
  cd $Q || exit 1
  cp PINS.sha256 PINS.sha256.before_v12
  grep -v "  \./README_LAUNCH.txt$\|  \./LAUNCH_STATUS.md$" PINS.sha256 > PINS.tmp && mv PINS.tmp PINS.sha256
  while read -r h f; do grep -v "  \./$f\$" PINS.sha256 > PINS.tmp && mv PINS.tmp PINS.sha256; echo "$h  ./$f" >> PINS.sha256; done < $S/STAGE.sha256
  sha256sum -c --quiet PINS.sha256 || { log "PINS check FAILED after the v12 install"; exit 6; }
  $PY - <<'PYEOF' || { log "gate update FAILED"; exit 6; }
import json
p = "/var/tmp/expert-guided-clean-a1-20261024/LAUNCH_EXPERT_GUIDED.json"
g = json.load(open(p))
st = g.get("stage") if isinstance(g.get("stage"), list) else [g.get("stage")]
words = {"WF_V12": "ejecuta y orquesta todo ... solucionar todos los problemas (2026-10-25)",
         "WF_C2": "ejecuta y orquesta ... todas las tareas requeridas para solucionar todos los problemas que encontraste (2026-10-25)",
         "WF_C2G": "Sí, como brazo extra (2026-10-25)"}
for k, w in words.items():
    if k not in st:
        st.append(k)
    g.setdefault("owner_words_by_stage", {})[k] = w
g["stage"] = st
json.dump(g, open(p, "w"), indent=1, ensure_ascii=False)
PYEOF
  touch $Q/STATE/v12_installed
  log "v12 installed: $(wc -l < $S/STAGE.sha256) files, PINS re-verified (docs excluded), gate stage += WF_V12 WF_C2 WF_C2G"
fi
cd $Q/code || exit 1
$PY tools/expert_guided_clean_round.py localize --spec $Q/CLEAN_ROUND_SPEC_v9.json --root $Q --out $SPEC > $Q/localize_v9.log 2>&1 || { log "localize v9 FAILED"; exit 7; }

gate(){ # unit [audit json]: writes out/<unit>/SANITY.json, logs the verdict (never stops the chain)
  local u=$1 au=${2:-}
  local extra=""; [ -n "$au" ] && [ -f "$au" ] && extra="--audit $au"
  local r; r=$(CUDA_VISIBLE_DEVICES="" $CR -q -m 2G -t 900s -n sanity-$u -- $SG unit --unit-dir $Q/out/$u --log $Q/out/$u.log $extra 2>&1 | tail -1)
  log "sanity $u: $r"
}
ens(){ # unit recipe split periods
  local u=$1 r=$2 sp=$3 per=$4
  [ -f $Q/out/$u/E3M.json ] || { log "ensemble $u skipped: no E3M.json"; return; }
  [ -f $Q/out/$u/ENSEMBLE.json ] && return
  for try in $(seq 1 36); do CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $CR -q -m 6G -t 14400s -n ens-$u -- $PY tools/expert_guided_seed_ensemble.py --arm-dir $Q/out/$u \
      --admitted $Q/data/admitted --store $Q/data/store/full_pool_eurusd_h1.parquet --recipe $Q/recipe/$r --split-contract $sp --periods $per > $Q/out/$u.ensemble.log 2>&1 \
      && { log "ensemble $u DONE"; return; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $Q/out/$u.ensemble.log && { log "ensemble $u not admitted (try $try)"; sleep 600; } || { log "ensemble $u FAILED"; return; }
  done
}
au(){ # unit recipe split
  local u=$1 r=$2 sp=$3
  [ -f $ECD/EARLY_CLOSE_$u.json ] && return 0
  [ -f $Q/out/$u/E3M.json ] || { log "audit $u skipped: no E3M.json"; return 0; }
  for try in $(seq 1 36); do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $CR -q -m 4G -t 14400s -n ecaudit-$u -- $PY tools/expert_guided_early_close_audit.py run --arm-dir $Q/out/$u \
      --admitted $Q/data/admitted --store $Q/data/store/full_pool_eurusd_h1.parquet --recipe $Q/recipe/$r --split-contract $sp --periods INNER_A,INNER_B \
      --out $ECD/EARLY_CLOSE_$u.json > $ECD/$u.log 2>&1 && { log "audit $u DONE"; return 0; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ECD/$u.log && { log "audit $u not admitted (try $try)"; sleep 600; } || { log "audit $u FAILED (see $ECD/$u.log)"; return 1; }
  done
}
stage(){ # STAGE unit-arms...: emit + run the queue, then gate -> ensemble -> gate -> audit -> gate for every unit
  local st=$1; shift
  [ -f $Q/STATE/v12_stage_$st ] && { log "stage $st already done"; return; }
  $PY tools/expert_guided_clean_round.py emit --spec $SPEC --stage $st --out $Q/queue_$st.sh > $Q/emit_$st.log 2>&1 || { log "emit $st FAILED (see emit_$st.log)"; return; }
  bash $Q/queue_$st.sh
  for f in wf2022 wf2023; do for a in "$@"; do gate ${st}_${f}_$a; done; done
  for f in wf2022 wf2023; do for a in "$@"; do ens ${st}_${f}_$a CONSUMER_RECIPE_${f}.json $f TRAIN,INNER_A,INNER_B,OBS_2024; gate ${st}_${f}_$a; done; done
  for f in wf2022 wf2023; do for a in "$@"; do au ${st}_${f}_$a CONSUMER_RECIPE_${f}.json $f; gate ${st}_${f}_$a $ECD/EARLY_CLOSE_${st}_${f}_$a.json; done; done
  touch $Q/STATE/v12_stage_$st
  log "stage $st done (units, ensembles, audits, sanity)"
}

# 3. WF_V12
stage WF_V12 zz_hold_v12 both_hold_v12
# 4. C2 outcome tables per fold (engine replay of every entry; CPU, capped; stamped), digests recorded for C2G
for f in wf2022 wf2023; do
  [ -f $Q/STATE/v12_outcomes_$f ] && continue
  for try in $(seq 1 36); do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $CR -q -m 4G -t 7200s -n c2-outcomes-$f -- $PY tools/expert_guided_c2.py outcomes --admitted $Q/data/admitted \
      --store $Q/data/store/full_pool_eurusd_h1.parquet --recipe $Q/recipe/CONSUMER_RECIPE_$f.json --split-contract $f --out $Q/outcomes/OUTCOMES_ENTRY_$f.npz \
      > $Q/outcomes/outcomes_$f.log 2>&1 && { touch $Q/STATE/v12_outcomes_$f; log "outcomes $f DONE"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $Q/outcomes/outcomes_$f.log && { log "outcomes $f not admitted (try $try)"; sleep 600; } || { log "outcomes $f FAILED"; exit 9; }
  done
done
[ -f $Q/outcomes/OUTCOMES.sha256 ] || ( cd $Q/outcomes && sha256sum OUTCOMES_ENTRY_wf2022.npz OUTCOMES_ENTRY_wf2023.npz > OUTCOMES.sha256 )
stage WF_C2 zz_feat_c2
# 5. WF_C2G on the same outcome tables (digests re-checked; E3-M also checks each table's stamp)
( cd $Q/outcomes && sha256sum -c --quiet OUTCOMES.sha256 ) || { log "outcome tables changed since WF_C2: refusing WF_C2G"; exit 10; }
stage WF_C2G zz_feat_c2g
# 6. reports, each with its sanity table; the baseline audits (lane L9) beside the new ones for the C2 / C2G rules
for f in wf2022 wf2023; do cp -n $ECB/EARLY_CLOSE_WF_${f}_zz_hold.json $ECD/ 2>/dev/null; done
CUDA_VISIBLE_DEVICES="" $CR -q -m 2G -t 900s -n sanity-all -- $SG all --results $Q/out --audit-glob "$ECD/EARLY_CLOSE_*.json" --audit-glob "$ECB/EARLY_CLOSE_*.json" \
  --label WF_V1_=SUPERSEDED_INFEASIBLE_STRATEGY_SPACE --out-json $REP/SANITY_GATE_ALL.json --out-md $REP/SANITY_GATE_ALL.md --title "after the v12 chain" > $REP/sanity_all.log 2>&1 \
  && log "sanity of every unit written" || log "sanity of every unit FAILED (see report/sanity_all.log)"
$PY tools/expert_guided_wf_report.py wf-v1 --stage-key walk_forward_v12 --spec $SPEC --results $Q/out --out-json $REP/RESULTS_WF_V12.json --out-md $REP/RESULTS_WF_V12_2026_10_25.md \
  --closure-template $Q/OUTPUT_CONTRACT_V1_CLOSURE.md --closure-out $REP/OUTPUT_CONTRACT_V12_CLOSURE.md > $REP/report_wf_v12.log 2>&1 && log "report WF_V12 written" || log "report WF_V12 FAILED (see report/report_wf_v12.log)"
$PY tools/expert_guided_c2.py report --stage WF_C2 --spec $SPEC --results $Q/out --audit-dir $ECD --out-json $REP/RESULTS_WF_C2.json --out-md $REP/RESULTS_WF_C2_2026_10_25.md \
  > $REP/report_wf_c2.log 2>&1 && log "report WF_C2 written" || log "report WF_C2 FAILED (see report/report_wf_c2.log)"
$PY tools/expert_guided_c2.py report --stage WF_C2G --spec $SPEC --results $Q/out --audit-dir $ECD --out-json $REP/RESULTS_WF_C2G.json --out-md $REP/RESULTS_WF_C2G_2026_10_25.md \
  > $REP/report_wf_c2g.log 2>&1 && log "report WF_C2G written" || log "report WF_C2G FAILED (see report/report_wf_c2g.log)"
V=$(ls $ECD/EARLY_CLOSE_WF_V12_*.json $ECD/EARLY_CLOSE_WF_C2_*.json $ECD/EARLY_CLOSE_WF_C2G_*.json 2>/dev/null)
[ -n "$V" ] && $PY tools/expert_guided_early_close_audit.py report --units $V --out-json $REP/EARLY_CLOSE_AUDIT_V12.json --out-md $REP/EARLY_CLOSE_AUDIT_V12.md \
  --title "Early-close audit, WF_V12 / WF_C2 / WF_C2G (INNER_A = calibration block, INNER_B = validation year; every fitness variant of the ensemble)" > $ECD/report.log 2>&1 \
  && log "early-close v12 report written" || log "early-close v12 report FAILED or no unit (see $ECD/report.log)"
log "chain end"

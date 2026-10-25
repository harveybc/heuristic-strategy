#!/bin/bash
# clean round v10 chain (lane L12, 2026-10-25): ONE sequential, resumable chain after the v12 chain, whose 8 units were all
# FAILED_SANITY (seeds opening one side only; C2/C2G ensembles below 2 trades/week out of sample). v12 units are archived as
# SUPERSEDED_ONE_SIDED_AND_BOUNDARY_ACTIVITY. This chain runs
#   1. install the STAGED files (verified against stage_v13/STAGE.sha256); docs stay out of PINS
#   2. gate stages += WF_V13, WF_C2_13, WF_C2G_13 (owner's words)
#   3. WF_V13 (contract v1.3: per-side genes, side cap, activity CI rule) -> units -> gate -> ensembles -> gate -> audit -> gate
#   4. WF_C2_13 on the v12 chain's stamped outcome tables (digests re-checked) -> same steps
#   5. WF_C2G_13 on the same tables -> same steps
#   6. reports (WF_V13 + closure, WF_C2_13, WF_C2G_13, early close, sanity of every unit with the superseded labels)
#   7. "[chain-v13] chain end"
# FIRST-SEED / FIRST-ENSEMBLE STOP RULE: after the first seed of the FIRST unit of each stage, and after the first ensemble of
# each stage, the sanity check runs on INNER_A and INNER_B; zero trades, < 2 trades/week, a side above 90 %, no agent close
# or a single exit reason stops the chain with the cause logged. New STATE marks only (v13_*, new unit names).
Q=/var/tmp/expert-guided-clean-a1-20261024; PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run
S=$Q/stage_v13; SPEC=$Q/CLEAN_ROUND_SPEC_v10_host.json; REP=$Q/report; ECD=$REP/early_close_v13; ECB=/var/tmp/early-close-audit-20261025/out
SG="$PY tools/expert_guided_sanity_gate.py"; FS="$PY tools/clean_round_chain/first_seed_check.py"
mkdir -p $REP $Q/STATE $Q/contracts $Q/outcomes $ECD
log(){ echo "$(date -u +%FT%TZ) [chain-v13] $*" >> $Q/LOG; }
test -f $Q/LAUNCH_EXPERT_GUIDED.json || { log "gate missing: refusing"; exit 3; }
log "chain start: install -> WF_V13 -> WF_C2_13 -> WF_C2G_13 -> reports (sanity gate after every unit and ensemble; first-seed / first-ensemble stop rule)"

# 1. install (verified), docs out of PINS, gate extended
( cd $S && sha256sum -c --quiet STAGE.sha256 ) || { log "staged files differ from STAGE.sha256: refusing"; exit 6; }
if [ ! -f $Q/STATE/v13_installed ]; then
  while read -r h f; do install -D -m 0644 "$S/$f" "$Q/$f" || { log "install $f FAILED"; exit 6; }; done < $S/STAGE.sha256
  cd $Q || exit 1
  cp PINS.sha256 PINS.sha256.before_v13
  grep -v "  \./README_LAUNCH.txt$\|  \./LAUNCH_STATUS.md$" PINS.sha256 > PINS.tmp && mv PINS.tmp PINS.sha256
  while read -r h f; do grep -v "  \./$f\$" PINS.sha256 > PINS.tmp && mv PINS.tmp PINS.sha256; echo "$h  ./$f" >> PINS.sha256; done < $S/STAGE.sha256
  sha256sum -c --quiet PINS.sha256 || { log "PINS check FAILED after the v13 install"; exit 6; }
  $PY - <<'PYEOF' || { log "gate update FAILED"; exit 6; }
import json
p = "/var/tmp/expert-guided-clean-a1-20261024/LAUNCH_EXPERT_GUIDED.json"
g = json.load(open(p))
st = g.get("stage") if isinstance(g.get("stage"), list) else [g.get("stage")]
words = {"WF_V13": "arreglalo y continua; actividad: Con confianza estadística (2026-10-25)",
         "WF_C2_13": "arreglalo y continua; actividad: Con confianza estadística (2026-10-25)",
         "WF_C2G_13": "Sí, como brazo extra; arreglalo y continua; actividad: Con confianza estadística (2026-10-25)"}
for k, w in words.items():
    if k not in st:
        st.append(k)
    g.setdefault("owner_words_by_stage", {})[k] = w
g["stage"] = st
json.dump(g, open(p, "w"), indent=1, ensure_ascii=False)
PYEOF
  touch $Q/STATE/v13_installed
  log "v13 installed: $(wc -l < $S/STAGE.sha256) files, PINS re-verified (docs excluded), gate stage += WF_V13 WF_C2_13 WF_C2G_13"
fi
cd $Q/code || exit 1
$PY tools/expert_guided_clean_round.py localize --spec $Q/CLEAN_ROUND_SPEC_v10.json --root $Q --out $SPEC > $Q/localize_v10.log 2>&1 || { log "localize v10 FAILED"; exit 7; }

gate(){ # unit [audit json]: writes out/<unit>/SANITY.json, logs the verdict
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
stage(){ # STAGE first-arm arms...: emit; run the queue with the first-seed stop rule; gate -> ensembles (first-ensemble stop rule) -> gate -> audit -> gate
  local st=$1; shift
  [ -f $Q/STATE/v13_stage_$st ] && { log "stage $st already done"; return; }
  $PY tools/expert_guided_clean_round.py emit --spec $SPEC --stage $st --out $Q/queue_$st.sh > $Q/emit_$st.log 2>&1 || { log "emit $st FAILED (see emit_$st.log)"; exit 11; }
  local first=${st}_wf2022_$1
  setsid bash $Q/queue_$st.sh &
  local qp=$!
  if [ ! -f $Q/STATE/v13_firstseed_$st ]; then
    local r rc
    r=$(CUDA_VISIBLE_DEVICES="" $FS seed --log $Q/out/$first.log --unit $first --pid $qp 2>&1); rc=$?
    log "first seed $first: $(echo "$r" | tail -1)"
    if [ $rc -ne 0 ]; then
      kill -TERM -- -$qp 2>/dev/null; sleep 5; kill -KILL -- -$qp 2>/dev/null
      log "STOPPED by the first-seed rule at $first (rc $rc); nothing after it runs"; exit 20
    fi
    touch $Q/STATE/v13_firstseed_$st
  fi
  wait $qp
  for f in wf2022 wf2023; do for a in "$@"; do gate ${st}_${f}_$a; done; done
  local ef=0
  for f in wf2022 wf2023; do for a in "$@"; do
    ens ${st}_${f}_$a CONSUMER_RECIPE_${f}.json $f TRAIN,INNER_A,INNER_B,OBS_2024; gate ${st}_${f}_$a
    if [ $ef -eq 0 ] && [ ! -f $Q/STATE/v13_firstens_$st ]; then
      local r rc
      r=$($FS ensemble --sanity $Q/out/${st}_${f}_$a/SANITY.json --unit ${st}_${f}_$a 2>&1); rc=$?
      log "first ensemble ${st}_${f}_$a: $(echo "$r" | tail -1 | cut -c1-600)"
      [ $rc -ne 0 ] && { log "STOPPED by the first-ensemble rule at ${st}_${f}_$a (rc $rc)"; exit 21; }
      touch $Q/STATE/v13_firstens_$st
    fi
    ef=1
  done; done
  for f in wf2022 wf2023; do for a in "$@"; do au ${st}_${f}_$a CONSUMER_RECIPE_${f}.json $f; gate ${st}_${f}_$a $ECD/EARLY_CLOSE_${st}_${f}_$a.json; done; done
  touch $Q/STATE/v13_stage_$st
  log "stage $st done (units, ensembles, audits, sanity)"
}

# 3. WF_V13
stage WF_V13 zz_hold_v13 both_hold_v13
# 4. WF_C2_13 on the v12 chain's stamped outcome tables (digests re-checked; E3-M also checks each stamp)
( cd $Q/outcomes && sha256sum -c --quiet OUTCOMES.sha256 ) || { log "outcome tables changed since the v12 chain: refusing WF_C2_13"; exit 10; }
stage WF_C2_13 zz_feat_c2_13
# 5. WF_C2G_13 on the same tables
( cd $Q/outcomes && sha256sum -c --quiet OUTCOMES.sha256 ) || { log "outcome tables changed: refusing WF_C2G_13"; exit 10; }
stage WF_C2G_13 zz_feat_c2g_13
# 6. reports, each with its sanity table; the baseline audits (lane L9) beside the new ones
for f in wf2022 wf2023; do cp -n $ECB/EARLY_CLOSE_WF_${f}_zz_hold.json $ECD/ 2>/dev/null; done
CUDA_VISIBLE_DEVICES="" $CR -q -m 2G -t 900s -n sanity-all -- $SG all --results $Q/out --audit-glob "$ECD/EARLY_CLOSE_*.json" --audit-glob "$REP/early_close_v12/EARLY_CLOSE_*.json" \
  --audit-glob "$ECB/EARLY_CLOSE_*.json" --label WF_V1_=SUPERSEDED_INFEASIBLE_STRATEGY_SPACE --label WF_V12_=SUPERSEDED_ONE_SIDED_AND_BOUNDARY_ACTIVITY \
  --label WF_C2_wf=SUPERSEDED_ONE_SIDED_AND_BOUNDARY_ACTIVITY --label WF_C2G_wf=SUPERSEDED_ONE_SIDED_AND_BOUNDARY_ACTIVITY \
  --out-json $REP/SANITY_GATE_ALL.json --out-md $REP/SANITY_GATE_ALL.md --title "after the v13 chain" > $REP/sanity_all_v13.log 2>&1 \
  && log "sanity of every unit written" || log "sanity of every unit FAILED (see report/sanity_all_v13.log)"
$PY tools/expert_guided_wf_report.py wf-v1 --stage-key walk_forward_v13 --spec $SPEC --results $Q/out --out-json $REP/RESULTS_WF_V13.json --out-md $REP/RESULTS_WF_V13_2026_10_25.md \
  --closure-template $Q/OUTPUT_CONTRACT_V1_CLOSURE.md --closure-out $REP/OUTPUT_CONTRACT_V13_CLOSURE.md > $REP/report_wf_v13.log 2>&1 && log "report WF_V13 written" || log "report WF_V13 FAILED (see report/report_wf_v13.log)"
$PY tools/expert_guided_c2.py report --stage WF_C2_13 --spec $SPEC --results $Q/out --audit-dir $ECD --out-json $REP/RESULTS_WF_C2_13.json --out-md $REP/RESULTS_WF_C2_13_2026_10_25.md \
  > $REP/report_wf_c2_13.log 2>&1 && log "report WF_C2_13 written" || log "report WF_C2_13 FAILED (see report/report_wf_c2_13.log)"
$PY tools/expert_guided_c2.py report --stage WF_C2G_13 --spec $SPEC --results $Q/out --audit-dir $ECD --out-json $REP/RESULTS_WF_C2G_13.json --out-md $REP/RESULTS_WF_C2G_13_2026_10_25.md \
  > $REP/report_wf_c2g_13.log 2>&1 && log "report WF_C2G_13 written" || log "report WF_C2G_13 FAILED (see report/report_wf_c2g_13.log)"
V=$(ls $ECD/EARLY_CLOSE_WF_V13_*.json $ECD/EARLY_CLOSE_WF_C2_13_*.json $ECD/EARLY_CLOSE_WF_C2G_13_*.json 2>/dev/null)
[ -n "$V" ] && $PY tools/expert_guided_early_close_audit.py report --units $V --out-json $REP/EARLY_CLOSE_AUDIT_V13.json --out-md $REP/EARLY_CLOSE_AUDIT_V13.md \
  --title "Early-close audit, WF_V13 / WF_C2_13 / WF_C2G_13 (INNER_A = calibration block, INNER_B = validation year; every fitness variant of the ensemble)" > $ECD/report.log 2>&1 \
  && log "early-close v13 report written" || log "early-close v13 report FAILED or no unit (see $ECD/report.log)"
log "chain end"

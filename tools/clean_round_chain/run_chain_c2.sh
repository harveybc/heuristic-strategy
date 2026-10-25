#!/bin/bash
# clean round v8 chain, stage WF_C2 (lane L10, 2026-10-25; causal-ladder C2: learn the consequences of our actions).
# Waits for the early-close v1 audit to end ("[early-close] chain end (early-close)"; if that marker never appears, for
# "[chain-v6] chain end (v6)" plus 2 h), then: install the STAGED files (verified against stage_wf_c2/STAGE.sha256) ->
# re-pin (README_LAUNCH.txt / LAUNCH_STATUS.md are documentation and leave PINS from now on; their drift stopped the v6
# chain once) -> gate += WF_C2 -> localize spec v8 -> build the outcome tables per fold (engine replay, CPU, capped) ->
# emit + run WF_C2 (2 folds x zz_feat_c2 x 5 seeds, 10G, one unit at a time) -> C2 seed ensembles -> C2-aware early-close
# audit -> RESULTS_WF_C2 report. Sequential; resumable (STATE marks, output files); nothing here asks a human. Owner's
# words (2026-10-25): "ejecuta y orquesta ... todas las tareas requeridas para solucionar todos los problemas que encontraste".
Q=/var/tmp/expert-guided-clean-a1-20261024; PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run
S=$Q/stage_wf_c2; SPEC=$Q/CLEAN_ROUND_SPEC_v8_host.json; REP=$Q/report; ECB=/var/tmp/early-close-audit-20261025/out; ECD=$REP/early_close_c2
mkdir -p $REP $Q/STATE $Q/contracts $Q/outcomes $ECD
log(){ echo "$(date -u +%FT%TZ) [chain-c2] $*" >> $Q/LOG; }
test -f $Q/LAUNCH_EXPERT_GUIDED.json || { log "gate missing: refusing"; exit 3; }
# 0. wait (poll 5 min, max 36 h)
log "waiting for the early-close end marker (fallback: v6 end + 2 h)"
v6_seen=""
for i in $(seq 1 432); do
  grep -q "\[early-close\] chain end (early-close)" $Q/LOG && break
  if grep -q "\[chain-v6\] chain end (v6)" $Q/LOG; then
    [ -z "$v6_seen" ] && v6_seen=$(date +%s)
    [ $(( $(date +%s) - v6_seen )) -ge 7200 ] && { log "early-close marker absent 2 h after the v6 end: proceeding"; break; }
  fi
  sleep 300
done
grep -q "\[early-close\] chain end (early-close)\|\[chain-v6\] chain end (v6)" $Q/LOG || { log "neither marker within 36 h: stopping"; exit 5; }
log "chain start: install C2 -> outcomes -> WF_C2 units -> C2 ensembles -> early-close audit -> report"
# 1. staged files -> package (verified first), pins replaced line by line, docs out of PINS, gate extended
( cd $S && sha256sum -c --quiet STAGE.sha256 ) || { log "staged files differ from STAGE.sha256: refusing"; exit 6; }
if [ ! -f $Q/STATE/c2_installed ]; then
  while read -r h f; do install -D -m 0644 "$S/$f" "$Q/$f" || { log "install $f FAILED"; exit 6; }; done < $S/STAGE.sha256
  cd $Q || exit 1
  cp PINS.sha256 PINS.sha256.before_c2
  grep -v "  \./README_LAUNCH.txt$\|  \./LAUNCH_STATUS.md$" PINS.sha256 > PINS.tmp && mv PINS.tmp PINS.sha256
  while read -r h f; do grep -v "  \./$f\$" PINS.sha256 > PINS.tmp && mv PINS.tmp PINS.sha256; echo "$h  ./$f" >> PINS.sha256; done < $S/STAGE.sha256
  sha256sum -c --quiet PINS.sha256 || { log "PINS check FAILED after the C2 install"; exit 6; }
  $PY - <<'PYEOF' || { log "gate update FAILED"; exit 6; }
import json
p = "/var/tmp/expert-guided-clean-a1-20261024/LAUNCH_EXPERT_GUIDED.json"
g = json.load(open(p))
st = g.get("stage") if isinstance(g.get("stage"), list) else [g.get("stage")]
if "WF_C2" not in st:
    st.append("WF_C2")
g["stage"] = st
g.setdefault("owner_words_by_stage", {})["WF_C2"] = "ejecuta y orquesta ... todas las tareas requeridas para solucionar todos los problemas que encontraste (2026-10-25)"
json.dump(g, open(p, "w"), indent=1)
PYEOF
  touch $Q/STATE/c2_installed
  log "C2 installed: $(wc -l < $S/STAGE.sha256) files, PINS re-verified (docs excluded), gate stage += WF_C2"
fi
cd $Q/code || exit 1
$PY tools/expert_guided_clean_round.py localize --spec $Q/CLEAN_ROUND_SPEC_v8.json --root $Q --out $SPEC > $Q/localize_v8.log 2>&1 || { log "localize v8 FAILED"; exit 7; }
# 2. outcome tables per fold (engine replay of every entry; CPU, capped; stamped)
for f in wf2022 wf2023; do
  [ -f $Q/STATE/c2_outcomes_$f ] && continue
  for try in $(seq 1 36); do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $CR -q -m 4G -t 7200s -n c2-outcomes-$f -- $PY tools/expert_guided_c2.py outcomes --admitted $Q/data/admitted \
      --store $Q/data/store/full_pool_eurusd_h1.parquet --recipe $Q/recipe/CONSUMER_RECIPE_$f.json --split-contract $f --out $Q/outcomes/OUTCOMES_ENTRY_$f.npz \
      > $Q/outcomes/outcomes_$f.log 2>&1 && { touch $Q/STATE/c2_outcomes_$f; log "outcomes $f DONE"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $Q/outcomes/outcomes_$f.log && { log "outcomes $f not admitted (try $try)"; sleep 600; } || { log "outcomes $f FAILED"; exit 9; }
  done
done
# 3. stage WF_C2
$PY tools/expert_guided_clean_round.py emit --spec $SPEC --stage WF_C2 --out $Q/queue_WF_C2.sh > $Q/emit_WF_C2.log 2>&1 || { log "emit WF_C2 FAILED (see emit_WF_C2.log)"; exit 8; }
bash $Q/queue_WF_C2.sh
# 4. C2 seed ensembles (every fitness variant; INNER_B scored after freeze; OBS_2024 one read for the arm's variant)
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
for f in wf2022 wf2023; do ens WF_C2_${f}_zz_feat_c2 CONSUMER_RECIPE_${f}.json $f TRAIN,INNER_A,INNER_B,OBS_2024; done
# 5. early-close audit of the C2 units (C2-aware tool); the baseline audits (lane L9) are copied beside them for the report
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
for f in wf2022 wf2023; do au WF_C2_${f}_zz_feat_c2 CONSUMER_RECIPE_${f}.json $f; cp -n $ECB/EARLY_CLOSE_WF_${f}_zz_hold.json $ECD/ 2>/dev/null; done
V=$(ls $ECD/EARLY_CLOSE_WF_C2_*.json 2>/dev/null)
[ -n "$V" ] && $PY tools/expert_guided_early_close_audit.py report --units $V --out-json $REP/EARLY_CLOSE_AUDIT_C2.json --out-md $REP/EARLY_CLOSE_AUDIT_C2.md \
  --title "Early-close audit, WF_C2 (consequence learning; INNER_A = calibration block, INNER_B = validation year; every fitness variant of the ensemble)" > $ECD/report.log 2>&1 \
  && log "early-close C2 report written" || log "early-close C2 report FAILED or no unit (see $ECD/report.log)"
# 6. WF_C2 report (pre-registered rules; both baselines on the same fold)
$PY tools/expert_guided_c2.py report --spec $SPEC --results $Q/out --audit-dir $ECD --out-json $REP/RESULTS_WF_C2.json --out-md $REP/RESULTS_WF_C2_2026_10_25.md \
  > $REP/report_wf_c2.log 2>&1 && log "report WF_C2 written" || log "report WF_C2 FAILED (see report/report_wf_c2.log)"
log "chain end (c2)"

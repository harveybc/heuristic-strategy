#!/bin/bash
# Bring the WF_V1 report, the filled closure table and the v1 evidence into the worktree (run on the coordinator when LOG
# shows "[chain-v6] chain end (v6)"); no commit.
H=${EG_HOST:?set EG_HOST to the ssh alias of the host that holds the package}; Q=/var/tmp/expert-guided-clean-a1-20261024; W=/var/tmp/hs-expert-guided-rl-20261019/docs/short_horizon/clean_round
mkdir -p $W/evidence/wf_v1 $W/evidence/walk_forward
scp -q $H:$Q/report/RESULTS_WF_V1.json $W/evidence/wf_v1/ 2>/dev/null
scp -q $H:$Q/report/RESULTS_WF_V1_2026_10_25.md $W/RESULTS_WF_V1_2026_10_25.md 2>/dev/null
scp -q $H:$Q/report/OUTPUT_CONTRACT_V1_CLOSURE.md $W/OUTPUT_CONTRACT_V1_CLOSURE.md 2>/dev/null
scp -q $H:$Q/report/RESULTS_WALK_FORWARD.json $W/evidence/walk_forward/ 2>/dev/null
scp -q $H:$Q/report/RESULTS_WALK_FORWARD_2026_10_25.md $W/RESULTS_WALK_FORWARD_2026_10_25.md 2>/dev/null
for f in wf2022 wf2023; do for a in both_hold_v1 zz_hold_v1; do u=WF_V1_${f}_${a}
  scp -q $H:$Q/out/$u/ENSEMBLE.json $W/evidence/wf_v1/ENSEMBLE_$u.json 2>/dev/null
  scp -q $H:$Q/out/$u/E3M.json $W/evidence/wf_v1/E3M_$u.json 2>/dev/null; done; done
ssh $H "tail -40 $Q/LOG" > $W/evidence/wf_v1/LOG_tail.txt
echo "== WF_V1 verdicts"; grep -A 8 "## Veredictos" $W/RESULTS_WF_V1_2026_10_25.md 2>/dev/null
echo "== fitness variants"; grep -A 40 "## Variantes de fitness" $W/RESULTS_WF_V1_2026_10_25.md 2>/dev/null | grep "^|\|^- " | head -40
echo "== closure"; grep "^| D\|^| E" $W/OUTPUT_CONTRACT_V1_CLOSURE.md 2>/dev/null

#!/bin/bash
# Bring the host's report + evidence into the worktree (run on the coordinator when LOG shows "[chain-v5] chain end (v5)"); no commit.
H=dragon-lan; Q=/var/tmp/expert-guided-clean-a1-20261024; W=/var/tmp/hs-expert-guided-rl-20261019/docs/short_horizon/clean_round
mkdir -p $W/evidence/b_ensemble $W/evidence/walk_forward
scp -q $H:$Q/report/RESULTS_B_AND_ENSEMBLE.json $H:$Q/DECISION_A_5SEED.json $W/evidence/b_ensemble/ 2>/dev/null
scp -q $H:$Q/report/RESULTS_B_AND_ENSEMBLE_2026_10_25.md $W/RESULTS_B_AND_ENSEMBLE_2026_10_25.md 2>/dev/null
scp -q $H:$Q/report/RESULTS_WALK_FORWARD.json $W/evidence/walk_forward/ 2>/dev/null
scp -q $H:$Q/report/RESULTS_WALK_FORWARD_2026_10_25.md $W/RESULTS_WALK_FORWARD_2026_10_25.md 2>/dev/null
for u in $(ssh $H "ls $Q/out | grep -v '\.log$'"); do scp -q $H:$Q/out/$u/ENSEMBLE.json $W/evidence/b_ensemble/ENSEMBLE_$u.json 2>/dev/null; done
for f in wf2022 wf2023; do for a in zz_hold both_hold both_hold_shorter; do u=WF_${f}_${a}; scp -q $H:$Q/out/$u/E3M.json $W/evidence/walk_forward/E3M_$u.json 2>/dev/null; mv -f $W/evidence/b_ensemble/ENSEMBLE_$u.json $W/evidence/walk_forward/ 2>/dev/null; done; done
ssh $H "tail -30 $Q/LOG" > $W/evidence/walk_forward/LOG_tail.txt
echo "== B + ensemble"; sed -n 1,60p $W/RESULTS_B_AND_ENSEMBLE_2026_10_25.md 2>/dev/null | grep "^|" | head -40
echo "== walk-forward"; grep -A 12 "## Veredictos" $W/RESULTS_WALK_FORWARD_2026_10_25.md 2>/dev/null; grep "^|" $W/RESULTS_WALK_FORWARD_2026_10_25.md 2>/dev/null | head -20

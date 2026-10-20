# Modular controlled study: requirement to evidence

| ID | Behavior | Evidence |
| --- | --- | --- |
| MC01 | Direction/task routing, abstention, no reversal | test_modular_controlled routing/gate tests |
| MC02 | Temporal prefix/current observation preserved | existing test_expert_guided_temporal and data/environment tests |
| MC03 | One entry/day, protections, volume, 12h rules, next open | native env suite plus controlled rollout witness |
| MC04 | Labels purged; inference independent of labels | next-pivot/exit support tests; forbidden-label inference witness |
| MC05 | Aggregate capacity matched before scores | parameter-count test, CONFIG.json |
| MC06 | Calibration only INNER_A | calibrate fixed period; frozen config before other period evaluation |
| MC07 | Epoch-end mean monitor and exact weight reload | CPU native fit witness; GPU_SMOKE/PREFLIGHT; per-module receipts |
| MC08 | Finite sequence, wait, retry, adoption, integrity | test_modular_campaign; STATE/STATUS/FAILURE receipts |
| MC09 | Trade costs, direction totals, zero activity | native env suite; scalar metrics witness; all trade/equity records |
| MC10 | Publication and independent copy | existing predictor Outbox/readback; new terminal hash validation; live receipts pending |

Pre-implementation test collection failed because the new module did not exist.
First integrated focused suite: 43 passed in 11.66 seconds on CPU. Synthetic
tests do not certify financial advantage. Real data admission, GPU smoke,
warehouse readback and independent-copy receipts are emitted outside git.

Deployment acceptance: runtime f6ddd53c18c34a12f809498d124874ba82ee908f,
spec fb808aa85830deac27cc838fad9cc326ac3d4bb57790b51e782f583250b946a6.
43 focused CPU tests also pass on the worker (6.09 s). Actual synthetic GPU
probe completed 2 epochs, 67,644,928 allocated peak bytes and bitwise reload.
The first scientific cell closed, the second started automatically, 8/8 reports
were delivered with exact readback and the first terminal directory passed
independent-copy hashes. No campaign integrity block. Remaining fits are owned
by the timers; this record does not claim they have all completed.

Derived analysis was added on the collector side only after deployment. Seven
orchestration/analysis tests pass. The GPU runtime is kept at its original pin.
The analysis shows original-three/new-two/all-five seed summaries, paired seed
differences and daily-growth gaps. It cannot alter weights or select a candidate.

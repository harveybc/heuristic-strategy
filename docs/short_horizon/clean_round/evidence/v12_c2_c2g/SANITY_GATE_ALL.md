# Sanity gate on every unit (after the v12 chain)

Owner 2026-10-25: a run with zero trades, one-sided trades or no early close measures nothing. A unit is FAILED_SANITY when any seed or its ensemble shows, on the calibration block (INNER_A) or the decision block (INNER_B): zero trades, < 2.0 trades/week, one side > 90 % of entries, zero agent closes, all exits by one reason, undefined Sharpe, a strategy search with no valid point, or a replay that differs from the run. FAILED_SANITY units are excluded from every comparison. UNMEASURED = the unit's artifacts do not record that quantity (never assumed).

Units: 30; PASS 14; FAILED_SANITY 16.

## Per unit

| unit | status | causes (count) | unmeasured |
| --- | --- | --- | --- |
| A_both_hold | **PASS** | — | — |
| A_both_hold_shorter | **PASS** | — | side_balance |
| A_opp_hold | **PASS** | — | side_balance |
| A_opp_hold_shorter | **PASS** | — | side_balance |
| A_opp_optimum | **PASS** | — | side_balance |
| A_prime_multires_deap | **PASS** | — | side_balance |
| A_prime_multires_deap_weighted | **PASS** | — | side_balance |
| A_prime_multires_default | **PASS** | — | side_balance |
| A_zz_hold | **PASS** | — | — |
| A_zz_hold_shorter | **PASS** | — | side_balance |
| B_winner_aux | **PASS** | — | side_balance |
| B_winner_minimal_state | **FAILED_SANITY** | unit_failed (1) — no E3M.json: the unit was stopped or ended before writing its results (last log line: E3M_RECIPE_CONFLICT: the shared recipe has no agent feedback (agent_state none)) | — |
| B_winner_multires_aux | **PASS** | — | side_balance |
| WF_C2G_wf2022_zz_feat_c2g | **FAILED_SANITY** | below_2_trades_per_week (3); one_side_over_90pct (2) | — |
| WF_C2G_wf2023_zz_feat_c2g | **FAILED_SANITY** | below_2_trades_per_week (2) | — |
| WF_C2_wf2022_zz_feat_c2 | **FAILED_SANITY** | below_2_trades_per_week (2); one_side_over_90pct (2) | — |
| WF_C2_wf2023_zz_feat_c2 | **FAILED_SANITY** | below_2_trades_per_week (4); single_exit_reason (2) | — |
| WF_V12_wf2022_both_hold_v12 | **FAILED_SANITY** | below_2_trades_per_week (1); one_side_over_90pct (2) | — |
| WF_V12_wf2022_zz_hold_v12 | **FAILED_SANITY** | below_2_trades_per_week (1); one_side_over_90pct (2) | — |
| WF_V12_wf2023_both_hold_v12 | **FAILED_SANITY** | below_2_trades_per_week (1); one_side_over_90pct (2) | — |
| WF_V12_wf2023_zz_hold_v12 | **FAILED_SANITY** | one_side_over_90pct (4) | — |
| WF_V1_wf2022_both_hold_v1 | **FAILED_SANITY** | zero_agent_closes (1) | ensemble (no ENSEMBLE.json yet), side_balance |
| WF_V1_wf2022_zz_hold_v1 | **FAILED_SANITY** | zero_trades (10); below_2_trades_per_week (10); strategy_share_valid_zero (5) | ensemble (no ENSEMBLE.json yet), side_balance |
| WF_V1_wf2023_both_hold_v1 | **FAILED_SANITY** | unit_failed (1) — no E3M.json: the unit was stopped or ended before writing its results (last log line: {"variant": "FULL", "seed": 20261021, "arbiter": "action_contract_v1", "strategy": {"open_threshold": 0.269508, "gain_margin": -0.657809, "close_threshold": 0.5) | — |
| WF_wf2022_both_hold | **FAILED_SANITY** | zero_agent_closes (2) | side_balance |
| WF_wf2022_both_hold_shorter | **PASS** | — | side_balance |
| WF_wf2022_zz_hold | **PASS** | — | side_balance |
| WF_wf2023_both_hold | **FAILED_SANITY** | below_2_trades_per_week (1) | side_balance |
| WF_wf2023_both_hold_shorter | **FAILED_SANITY** | below_2_trades_per_week (1) | side_balance |
| WF_wf2023_zz_hold | **FAILED_SANITY** | below_2_trades_per_week (3); one_side_over_90pct (1) | side_balance |

## Labels

- WF_V1_wf2022_both_hold_v1: SUPERSEDED_INFEASIBLE_STRATEGY_SPACE
- WF_V1_wf2022_zz_hold_v1: SUPERSEDED_INFEASIBLE_STRATEGY_SPACE
- WF_V1_wf2023_both_hold_v1: SUPERSEDED_INFEASIBLE_STRATEGY_SPACE


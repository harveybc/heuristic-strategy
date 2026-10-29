# Profile sanity tier report

Records 135 (errors 0); PASS effective 134, FAIL 1; exec profiles {'full_f4': 135} (expected full_f4: OK); levels_clipped total 0 in 0 records (OK).

## Engine refusals by reason

| reason | count |
|---|---:|

Gate counters: {'days': 51611, 'placed': 83174, 'deferred_observing': 0, 'deferred_below_best': 97282, 'below_threshold': 0, 'entry_gate_closed': 60702}

## Horizon mix (executed, mean over records with entries)

{'h3': 0.01348906936060389, 'h6': 0.07414649584274911, 'h12': 0.3063238848775651, 'h24': 0.6060405499190819} over 135 records; dominance flags: none

## Activity

{'trades_per_week_median': 8.36789513269582, 'trades_per_week_p10_p90': [4.350273987321371, 10.31739550875685], 'early_close_share_pooled': 0.03112751581022916, 'records_with_zero_trades': 0}

## Net per family (median of the per-record median net sigma per trade)

| family | records | median net sigma / trade |
|---|---:|---:|
| B_cal | 3 | 0.588 |
| B_fine | 3 | 0.761 |
| B_gap | 3 | 0.556 |
| B_spread | 3 | 0.592 |
| B_trans | 3 | 0.563 |
| B_trend | 3 | 0.659 |
| B_vol | 3 | 0.608 |
| B_voldyn | 3 | 0.786 |
| F1 | 3 | 0.573 |
| F2 | 3 | 0.427 |
| F3 | 6 | 0.572 |
| F4 | 3 | 0.377 |
| F5 | 3 | 0.579 |
| H1 | 6 | 0.719 |
| H2 | 6 | 0.474 |
| H3 | 3 | 0.780 |
| H4 | 6 | 0.457 |
| H5 | 3 | 0.553 |
| H6 | 3 | 0.429 |
| I1 | 6 | 0.488 |
| I2 | 3 | 0.520 |
| I3 | 3 | 0.645 |
| I4 | 3 | 0.565 |
| LHS | 51 | 0.257 |

W14 control status: {'DEFECT': 9}; swapped-reference code symmetry max 3.439470930288735e-13 sigma; trades without a matched menu slot: 0

Sizing by side (q, funding reserve in pips, loss per unit in pips): {'BUY': {'trades': 40033, 'q_p10_p50_p90': [0.3953326940536499, 0.5670532584190369, 0.8792092204093933], 'share_q_below_1': 0.931656383483626, 'funding_reserve_pips_p50_p90': [2.8000001907348633, 6.579999923706055], 'loss_per_unit_pips_p50': 19.124433517456055}, 'SELL': {'trades': 43141, 'q_p10_p50_p90': [0.4717654883861542, 0.6267924904823303, 0.9525164365768433], 'share_q_below_1': 0.9172249136552235, 'funding_reserve_pips_p50_p90': [0.0, 0.0], 'loss_per_unit_pips_p50': 18.610666275024414}}

E-series episode early-close share, strategy vs O*: {'strategy_early_p10_p50_p90': [0.401350250261902, 0.4672442941673711, 0.5738109425785481], 'Ostar_early_p10_p50_p90': [0.401350250261902, 0.4672442941673711, 0.5738109425785481], 'max_abs_gap_to_Ostar': 0.0}

Burst gate outside the planted windows (calm-bar trigger share): {'records': 135, 'ratios': [1.5], 'calm_trigger_share_raw_p50_p90': [0.05973998065971849, 0.08430213817556678], 'calm_trigger_share_held_p50_p90': [0.14505211131406467, 0.1821639626087891]}

Control conflicts (CONTROL_CONFLICT, never PASS): []

W13 columns (menu form governing / sealed form): {'governing_PASS': 58, 'sealed_PASS': 54, 'sealed_FAIL': 4}; Z1 control: {'BUY:DEFECT': 135, 'SELL:DEFECT': 134, 'SELL:VALIDATED': 1}; W8 clause (b): {'records': 77, 'clause_b_PASS': 77, 'not_evaluable_elements': 125, 'Ostar_pass_with_sealed_clause_b_False': 57, 'Ostar_pass_with_sealed_clause_b_True': 20}; closer support: {'status': {'OK': 135}, 'records_extended': 24, 'max_extra_passes': 3}

## Witnesses (governing forms)

| witness | PASS | FAIL | NOT_EVALUABLE | CONTROL_CONFLICT | sealed-unvalidated (strict) | defect-corrected | excused by O* |
|---|---:|---:|---:|---:|---:|---:|---:|
| W0_activity | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| W10_spread_window | 26 | 0 | 0 | 0 | 0 | 0 | 0 |
| W11_cutoff | 54 | 0 | 0 | 0 | 54 | 0 | 0 |
| W12_missing | 18 | 0 | 0 | 0 | 0 | 0 | 0 |
| W13_tpsl_track_sigma | 58 | 0 | 0 | 0 | 0 | 0 | 0 |
| W14_mirror | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| W15_no_open_high_risk | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| W16_sl_tp_every_order | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| W17_no_open_in_burst | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| W1_side | 135 | 0 | 0 | 0 | 0 | 77 | 0 |
| W2_truth_positive_open | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| W2_wrong_side | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| W3_nop | 135 | 0 | 0 | 0 | 0 | 10 | 0 |
| W4_early_close_justified | 135 | 0 | 0 | 0 | 0 | 135 | 0 |
| W4_sl_exits | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| W5_reverse | 135 | 0 | 0 | 0 | 0 | 77 | 0 |
| W6_volume | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| W7_best_cell | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| W8_early_close_at_switch | 77 | 0 | 0 | 0 | 35 | 23 | 0 |
| W8_old_direction_after_lag | 76 | 0 | 0 | 1 | 25 | 49 | 0 |
| W9_gap_loss_bounded | 47 | 0 | 0 | 0 | 46 | 0 | 0 |
| WA_side_comparison | 135 | 0 | 0 | 0 | 135 | 0 | 0 |
| exit:E1_sl_destined_closed_early_BUY | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E1_sl_destined_closed_early_SELL | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E1p_closed_before_SL_BUY | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E1p_closed_before_SL_SELL | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E2_exit_regret_BUY | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E2_exit_regret_SELL | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E2p_regret_SL_destined_BUY | 133 | 2 | 0 | 0 | 0 | 0 | 2 |
| exit:E2p_regret_SL_destined_SELL | 133 | 2 | 0 | 0 | 0 | 0 | 2 |
| exit:E3_tick_agreement_BUY | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E3_tick_agreement_SELL | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E4_no_destructive_close_BUY | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E4_no_destructive_close_SELL | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E5_early_close_vs_Ostar_BUY | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E5_early_close_vs_Ostar_SELL | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E6_exit_mirror | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E7_lateness_vs_Pstar_BUY | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E7_lateness_vs_Pstar_SELL | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E8_closer_quality_ladder_BUY | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E8_closer_quality_ladder_SELL | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:EP_counts_BUY | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:EP_counts_SELL | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:Z1_zigzag_exit_label_BUY | 135 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:Z1_zigzag_exit_label_SELL | 135 | 0 | 0 | 0 | 0 | 0 | 0 |

## Trade summary (record schema)

records 135; net sigma / trade mean 0.491 (median 0.512); win rate 1.000; trades / record 616; SL pips median 17.0; TP pips median 42.0; hold bars median 2.0; Sharpe mean 18.25

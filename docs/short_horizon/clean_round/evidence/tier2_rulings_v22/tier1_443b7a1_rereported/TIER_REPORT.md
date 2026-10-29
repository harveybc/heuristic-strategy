# Profile sanity tier report

Records 135 (errors 0); PASS effective 134, FAIL 1; exec profiles {'full_f4': 135} (expected full_f4: OK); levels_clipped total 0 in 0 records (OK).

## Engine refusals by reason

| reason | count |
|---|---:|

Gate counters: {'days': 51603, 'placed': 90209, 'deferred_observing': 0, 'deferred_below_best': 121205, 'below_threshold': 0, 'entry_gate_closed': 74908}

## Horizon mix (executed, mean over records with entries)

{'h3': 0.013022459524432053, 'h6': 0.07363979049597041, 'h12': 0.29157808786239114, 'h24': 0.6217596621172063} over 135 records; dominance flags: none

## Activity

{'trades_per_week_median': 9.23176104007736, 'trades_per_week_p10_p90': [4.690662941871709, 11.312775330396477], 'early_close_share_pooled': 0.030562360740059195, 'records_with_zero_trades': 0}

## Net per family (median of the per-record median net sigma per trade)

| family | records | median net sigma / trade |
|---|---:|---:|
| B_cal | 3 | 0.596 |
| B_fine | 3 | 0.765 |
| B_gap | 3 | 0.558 |
| B_spread | 3 | 0.599 |
| B_trans | 3 | 0.566 |
| B_trend | 3 | 0.661 |
| B_vol | 3 | 0.614 |
| B_voldyn | 3 | 0.790 |
| F1 | 3 | 0.582 |
| F2 | 3 | 0.433 |
| F3 | 6 | 0.575 |
| F4 | 3 | 0.383 |
| F5 | 3 | 0.581 |
| H1 | 6 | 0.720 |
| H2 | 6 | 0.477 |
| H3 | 3 | 0.784 |
| H4 | 6 | 0.459 |
| H5 | 3 | 0.555 |
| H6 | 3 | 0.437 |
| I1 | 6 | 0.494 |
| I2 | 3 | 0.530 |
| I3 | 3 | 0.652 |
| I4 | 3 | 0.569 |
| LHS | 51 | 0.255 |

W14 control status: {'DEFECT': 9}; swapped-reference code symmetry max 3.135269821541442e-13 sigma; trades without a matched menu slot: 0

Sizing by side (q, funding reserve in pips, loss per unit in pips): {'BUY': {'trades': 43110, 'q_p10_p50_p90': [0.3959341049194336, 0.5653115510940552, 0.8708560466766357], 'share_q_below_1': 0.9334029227557411, 'funding_reserve_pips_p50_p90': [2.8000001907348633, 5.964000225067139], 'loss_per_unit_pips_p50': 19.135643005371094}, 'SELL': {'trades': 47099, 'q_p10_p50_p90': [0.47171586751937866, 0.6248942017555237, 0.9376363754272461], 'share_q_below_1': 0.9215482282001741, 'funding_reserve_pips_p50_p90': [0.0, 0.0], 'loss_per_unit_pips_p50': 18.614761352539062}}

E-series episode early-close share, strategy vs O*: {'strategy_early_p10_p50_p90': [0.401350250261902, 0.4672442941673711, 0.5738109425785481], 'Ostar_early_p10_p50_p90': [0.401350250261902, 0.4672442941673711, 0.5738109425785481], 'max_abs_gap_to_Ostar': 0.0}

Burst gate outside the planted windows (calm-bar trigger share): {'records': 0}

Control conflicts (CONTROL_CONFLICT, never PASS): [{'record': 'LHS_50_s2', 'witness': 'W8_old_direction_after_lag', 'form': 'W8_old_direction_after_lag_v2', 'strategy': [{'leg': 'trend_down', 'old_direction_share_after_lag': 0.6375, 'truth_old_direction_opportunity': 0.5226039783001808, 'n': 80}, {'leg': 'trend_up', 'old_direction_share_after_lag': 0.49829351535836175, 'truth_old_direction_opportunity': 0.4838709677419355, 'n': 293}], 'control': [{'leg': 'trend_down', 'old_direction_share_after_lag': 0.6419753086419753, 'truth_old_direction_opportunity': 0.5226039783001808, 'n': 81}, {'leg': 'trend_up', 'old_direction_share_after_lag': 0.4641638225255973, 'truth_old_direction_opportunity': 0.4838709677419355, 'n': 293}]}]

W13 columns (menu form governing / sealed form): {'governing_PASS': 58, 'sealed_PASS': 54, 'sealed_FAIL': 4}; Z1 control: {'BUY:DEFECT': 135, 'SELL:DEFECT': 134, 'SELL:VALIDATED': 1}; W8 clause (b): {'records': 77, 'clause_b_PASS': 77, 'not_evaluable_elements': 150, 'Ostar_pass_with_sealed_clause_b_False': 57, 'Ostar_pass_with_sealed_clause_b_True': 20}; closer support: {'status': {'OK': 135}, 'records_extended': 24, 'max_extra_passes': 3}

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
| W8_early_close_at_switch | 77 | 0 | 0 | 0 | 35 | 22 | 0 |
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

records 135; net sigma / trade mean 0.493 (median 0.513); win rate 1.000; trades / record 668; SL pips median 17.0; TP pips median 42.0; hold bars median 2.0; Sharpe mean 18.87

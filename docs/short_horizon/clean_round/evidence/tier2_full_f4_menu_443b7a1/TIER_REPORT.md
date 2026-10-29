# Profile sanity tier report

Records 4345 (errors 0); PASS effective 4232, FAIL 113; exec profiles {'full_f4': 4345} (expected full_f4: OK); levels_clipped total 0 in 0 records (OK).

## Engine refusals by reason

| reason | count |
|---|---:|

Gate counters: {'days': 1660297, 'placed': 3185595, 'deferred_observing': 0, 'deferred_below_best': 5155308, 'below_threshold': 0, 'entry_gate_closed': 415023}

## Horizon mix (executed, mean over records with entries)

{'h3': 0.00654550555551369, 'h6': 0.04792491551581819, 'h12': 0.30780561135752565, 'h24': 0.6377239675711424} over 4345 records; dominance flags: none

## Activity

{'trades_per_week_median': 9.915117653379177, 'trades_per_week_p10_p90': [5.3430751047598575, 12.28752551842699], 'early_close_share_pooled': 0.027834674527050678, 'records_with_zero_trades': 0}

## Net per family (median of the per-record median net sigma per trade)

| family | records | median net sigma / trade |
|---|---:|---:|
| B_cal | 10 | 0.592 |
| B_fine | 45 | 0.628 |
| B_gap | 20 | 0.584 |
| B_spread | 10 | 0.610 |
| B_trans | 120 | 0.541 |
| B_trend | 35 | 0.511 |
| B_vol | 15 | 0.617 |
| B_voldyn | 30 | 0.582 |
| F1 | 315 | 0.521 |
| F2 | 735 | 0.514 |
| F3 | 360 | 0.529 |
| F4 | 175 | 0.536 |
| F5 | 315 | 0.536 |
| H1 | 140 | 0.550 |
| H2 | 80 | 0.459 |
| H3 | 270 | 0.590 |
| H4 | 40 | 0.486 |
| H5 | 40 | 0.595 |
| H6 | 160 | 0.495 |
| I1 | 630 | 0.538 |
| I2 | 150 | 0.393 |
| I3 | 300 | 0.446 |
| I4 | 30 | 0.527 |
| LHS | 320 | 0.396 |

W14 control status: {'DEFECT': 869}; swapped-reference code symmetry max 6.354916592954396e-13 sigma; trades without a matched menu slot: 0

Sizing by side (q, funding reserve in pips, loss per unit in pips): {'BUY': {'trades': 1473721, 'q_p10_p50_p90': [0.4227052927017212, 0.6087871193885803, 1.0], 'share_q_below_1': 0.894338887754195, 'funding_reserve_pips_p50_p90': [0.7000000476837158, 4.199999809265137], 'loss_per_unit_pips_p50': 17.1464786529541}, 'SELL': {'trades': 1711874, 'q_p10_p50_p90': [0.4633181393146515, 0.6470965147018433, 1.0], 'share_q_below_1': 0.8896203809392513, 'funding_reserve_pips_p50_p90': [0.0, 0.0], 'loss_per_unit_pips_p50': 17.25860595703125}}

E-series episode early-close share, strategy vs O*: {'strategy_early_p10_p50_p90': [0.3986883942766296, 0.46726190476190477, 0.5498583177828461], 'Ostar_early_p10_p50_p90': [0.3986883942766296, 0.46726190476190477, 0.5498583177828461], 'max_abs_gap_to_Ostar': 0.0}

Control conflicts (CONTROL_CONFLICT, never PASS): [{'record': 'LHS_02_s0', 'witness': 'W8_old_direction_after_lag', 'form': 'W8_old_direction_after_lag_v2', 'strategy': [{'leg': 'trend_down', 'old_direction_share_after_lag': 0.6041666666666666, 'truth_old_direction_opportunity': 0.4954128440366973, 'n': 48}, {'leg': 'trend_up', 'old_direction_share_after_lag': 0.4126984126984127, 'truth_old_direction_opportunity': 0.46382189239332094, 'n': 126}], 'control': [{'leg': 'trend_down', 'old_direction_share_after_lag': 0.6041666666666666, 'truth_old_direction_opportunity': 0.4954128440366973, 'n': 48}, {'leg': 'trend_up', 'old_direction_share_after_lag': 0.4015748031496063, 'truth_old_direction_opportunity': 0.46382189239332094, 'n': 127}]}, {'record': 'LHS_04_s0', 'witness': 'W1_side', 'form': 'W1_side_v2', 'strategy': [{'leg': 'trend_down', 'n': 42, 'share_in_direction': 0.5, 'truth_share_in_direction': 0.4641255605381166, 'deviation': 0.0358744394618834}, {'leg': 'range', 'n': 108, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.18118838262723158}, {'leg': 'trend_up', 'n': 155, 'share_in_direction': 0.632258064516129, 'truth_share_in_direction': 0.5691460055096419, 'deviation': 0.06311205900648714}], 'control': [{'leg': 'trend_down', 'n': 42, 'share_in_direction': 0.3333333333333333, 'truth_share_in_direction': 0.4641255605381166, 'deviation': 0.13079222720478328}, {'leg': 'range', 'n': 108, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.16266986410871304}, {'leg': 'trend_up', 'n': 155, 'share_in_direction': 0.6645161290322581, 'truth_share_in_direction': 0.5691460055096419, 'deviation': 0.09537012352261615}]}, {'record': 'LHS_50_s2', 'witness': 'W8_old_direction_after_lag', 'form': 'W8_old_direction_after_lag_v2', 'strategy': [{'leg': 'trend_down', 'old_direction_share_after_lag': 0.6375, 'truth_old_direction_opportunity': 0.5226039783001808, 'n': 80}, {'leg': 'trend_up', 'old_direction_share_after_lag': 0.49829351535836175, 'truth_old_direction_opportunity': 0.4838709677419355, 'n': 293}], 'control': [{'leg': 'trend_down', 'old_direction_share_after_lag': 0.6419753086419753, 'truth_old_direction_opportunity': 0.5226039783001808, 'n': 81}, {'leg': 'trend_up', 'old_direction_share_after_lag': 0.4641638225255973, 'truth_old_direction_opportunity': 0.4838709677419355, 'n': 293}]}, {'record': 'LHS_59_s0', 'witness': 'W1_side', 'form': 'W1_side_v2', 'strategy': [{'leg': 'trend_down', 'n': 66, 'share_in_direction': 0.3181818181818182, 'truth_share_in_direction': 0.47357512953367875, 'deviation': 0.15539331135186057}, {'leg': 'trend_up', 'n': 92, 'share_in_direction': 0.6521739130434783, 'truth_share_in_direction': 0.5227272727272727, 'deviation': 0.12944664031620556}, {'leg': 'range', 'n': 168, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.11295018672067852}], 'control': [{'leg': 'trend_down', 'n': 70, 'share_in_direction': 0.2857142857142857, 'truth_share_in_direction': 0.47357512953367875, 'deviation': 0.18786084381939305}, {'leg': 'trend_up', 'n': 95, 'share_in_direction': 0.7052631578947368, 'truth_share_in_direction': 0.5227272727272727, 'deviation': 0.18253588516746416}, {'leg': 'range', 'n': 172, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.19102327642167521}]}, {'record': 'LHS_59_s1', 'witness': 'W1_side', 'form': 'W1_side_v2', 'strategy': [{'leg': 'trend_down', 'n': 66, 'share_in_direction': 0.36363636363636365, 'truth_share_in_direction': 0.4900314795383001, 'deviation': 0.12639511590193647}, {'leg': 'trend_up', 'n': 95, 'share_in_direction': 0.6736842105263158, 'truth_share_in_direction': 0.5546935608999224, 'deviation': 0.11899064962639333}, {'leg': 'range', 'n': 163, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.15295420955501127}], 'control': [{'leg': 'trend_down', 'n': 70, 'share_in_direction': 0.32857142857142857, 'truth_share_in_direction': 0.4900314795383001, 'deviation': 0.16146005096687155}, {'leg': 'trend_up', 'n': 100, 'share_in_direction': 0.76, 'truth_share_in_direction': 0.5546935608999224, 'deviation': 0.20530643910007756}, {'leg': 'range', 'n': 166, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.17679189157142045}]}, {'record': 'LHS_59_s2', 'witness': 'W1_side', 'form': 'W1_side_v2', 'strategy': [{'leg': 'trend_down', 'n': 67, 'share_in_direction': 0.3880597014925373, 'truth_share_in_direction': 0.46381243628950053, 'deviation': 0.0757527347969632}, {'leg': 'trend_up', 'n': 92, 'share_in_direction': 0.6956521739130435, 'truth_share_in_direction': 0.5395962732919255, 'deviation': 0.15605590062111802}, {'leg': 'range', 'n': 169, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.016133584669818457}], 'control': [{'leg': 'trend_down', 'n': 69, 'share_in_direction': 0.2608695652173913, 'truth_share_in_direction': 0.46381243628950053, 'deviation': 0.20294287107210923}, {'leg': 'trend_up', 'n': 96, 'share_in_direction': 0.71875, 'truth_share_in_direction': 0.5395962732919255, 'deviation': 0.17915372670807456}, {'leg': 'range', 'n': 176, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.06195795260957104}]}, {'record': 'M_F3_up_down_late_high_s0', 'witness': 'W8_old_direction_after_lag', 'form': 'W8_old_direction_after_lag_v2', 'strategy': [{'leg': 'trend_up', 'old_direction_share_after_lag': 0.7443609022556391, 'truth_old_direction_opportunity': 0.5089722675367048, 'n': 133}], 'control': [{'leg': 'trend_up', 'old_direction_share_after_lag': 0.6422764227642277, 'truth_old_direction_opportunity': 0.5089722675367048, 'n': 246}]}, {'record': 'M_I1_upmedium_downmedium_middle_s0', 'witness': 'W8_old_direction_after_lag', 'form': 'W8_old_direction_after_lag_v2', 'strategy': [{'leg': 'trend_up', 'old_direction_share_after_lag': 0.6216867469879518, 'truth_old_direction_opportunity': 0.5047826086956522, 'n': 415}], 'control': [{'leg': 'trend_up', 'old_direction_share_after_lag': 0.6084070796460177, 'truth_old_direction_opportunity': 0.5047826086956522, 'n': 452}]}, {'record': 'M_I3_vramp_down_gentle_up_down_trend_first_s0', 'witness': 'W8_old_direction_after_lag', 'form': 'W8_old_direction_after_lag_v2', 'strategy': [{'leg': 'trend_up', 'old_direction_share_after_lag': 0.7307692307692307, 'truth_old_direction_opportunity': 0.5110332749562172, 'n': 78}], 'control': [{'leg': 'trend_up', 'old_direction_share_after_lag': 0.6150627615062761, 'truth_old_direction_opportunity': 0.5110332749562172, 'n': 717}]}, {'record': 'M_LHS_05_s0', 'witness': 'W1_side', 'form': 'W1_side_v2', 'strategy': [{'leg': 'range', 'n': 51, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.17555555555555552}, {'leg': 'trend_up', 'n': 170, 'share_in_direction': 0.5, 'truth_share_in_direction': 0.5251396648044693, 'deviation': 0.025139664804469275}, {'leg': 'range', 'n': 159, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.07472704402515723}, {'leg': 'trend_up', 'n': 224, 'share_in_direction': 0.47767857142857145, 'truth_share_in_direction': 0.4973943254198031, 'deviation': 0.01971575399123171}], 'control': [{'leg': 'range', 'n': 51, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.17555555555555552}, {'leg': 'trend_up', 'n': 170, 'share_in_direction': 0.5, 'truth_share_in_direction': 0.5251396648044693, 'deviation': 0.025139664804469275}, {'leg': 'range', 'n': 160, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.07114999999999999}, {'leg': 'trend_up', 'n': 227, 'share_in_direction': 0.4713656387665198, 'truth_share_in_direction': 0.4973943254198031, 'deviation': 0.026028686653283284}]}, {'record': 'S_LHS_05_s0', 'witness': 'W1_side', 'form': 'W1_side_v2', 'strategy': [{'leg': 'range', 'n': 56, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.20422367165128852}, {'leg': 'trend_down', 'n': 163, 'share_in_direction': 0.5521472392638037, 'truth_share_in_direction': 0.52824184566428, 'deviation': 0.023905393599523705}, {'leg': 'range', 'n': 158, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.02940400843881852}, {'leg': 'trend_down', 'n': 222, 'share_in_direction': 0.545045045045045, 'truth_share_in_direction': 0.5063437139561707, 'deviation': 0.0387013310888743}], 'control': [{'leg': 'range', 'n': 56, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.20422367165128852}, {'leg': 'trend_down', 'n': 163, 'share_in_direction': 0.5521472392638037, 'truth_share_in_direction': 0.52824184566428, 'deviation': 0.023905393599523705}, {'leg': 'range', 'n': 159, 'share_in_direction': nan, 'truth_share_in_direction': nan, 'deviation': 0.0261399371069182}, {'leg': 'trend_down', 'n': 225, 'share_in_direction': 0.5422222222222223, 'truth_share_in_direction': 0.5063437139561707, 'deviation': 0.035878508266051545}]}]

W13 columns (menu form governing / sealed form): {'governing_PASS': 1075, 'sealed_PASS': 1058, 'sealed_FAIL': 17}; Z1 control: {'BUY:DEFECT': 4325, 'SELL:DEFECT': 4307, 'SELL:VALIDATED': 38, 'BUY:VALIDATED': 20}; W8 clause (b): {'records': 1855, 'clause_b_PASS': 1847, 'not_evaluable_elements': 1852, 'Ostar_pass_with_sealed_clause_b_False': 1156, 'Ostar_pass_with_sealed_clause_b_True': 699, 'clause_b_NOT_EVALUABLE': 8}; closer support: {'status': {'OK': 4345}, 'records_extended': 312, 'max_extra_passes': 6}

## Witnesses (governing forms)

| witness | PASS | FAIL | NOT_EVALUABLE | CONTROL_CONFLICT | sealed-unvalidated (strict) | defect-corrected | excused by O* |
|---|---:|---:|---:|---:|---:|---:|---:|
| W0_activity | 4342 | 3 | 0 | 0 | 0 | 0 | 0 |
| W10_spread_window | 220 | 0 | 0 | 0 | 1 | 0 | 0 |
| W11_cutoff | 430 | 0 | 0 | 0 | 430 | 0 | 0 |
| W12_missing | 215 | 0 | 0 | 0 | 0 | 0 | 0 |
| W13_tpsl_track_sigma | 1075 | 0 | 0 | 0 | 0 | 0 | 0 |
| W14_mirror | 869 | 0 | 0 | 0 | 0 | 0 | 0 |
| W15_no_open_high_risk | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| W16_sl_tp_every_order | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| W17_no_open_in_burst | 102 | 23 | 0 | 0 | 0 | 0 | 0 |
| W1_side | 4290 | 49 | 0 | 6 | 0 | 1839 | 0 |
| W2_truth_positive_open | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| W2_wrong_side | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| W3_nop | 4345 | 0 | 0 | 0 | 0 | 83 | 0 |
| W4_early_close_justified | 4342 | 3 | 0 | 0 | 0 | 4342 | 0 |
| W4_sl_exits | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| W5_reverse | 4345 | 0 | 0 | 0 | 13 | 1842 | 0 |
| W6_volume | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| W7_best_cell | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| W8_early_close_at_switch | 1851 | 0 | 4 | 0 | 854 | 509 | 0 |
| W8_old_direction_after_lag | 1826 | 24 | 0 | 5 | 850 | 979 | 0 |
| W9_gap_loss_bounded | 515 | 0 | 0 | 0 | 471 | 0 | 0 |
| WA_side_comparison | 4345 | 0 | 0 | 0 | 4345 | 0 | 0 |
| exit:E1_sl_destined_closed_early_BUY | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E1_sl_destined_closed_early_SELL | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E1p_closed_before_SL_BUY | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E1p_closed_before_SL_SELL | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E2_exit_regret_BUY | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E2_exit_regret_SELL | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E2p_regret_SL_destined_BUY | 4282 | 63 | 0 | 0 | 0 | 0 | 63 |
| exit:E2p_regret_SL_destined_SELL | 4290 | 55 | 0 | 0 | 0 | 0 | 55 |
| exit:E3_tick_agreement_BUY | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E3_tick_agreement_SELL | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E4_no_destructive_close_BUY | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E4_no_destructive_close_SELL | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E5_early_close_vs_Ostar_BUY | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E5_early_close_vs_Ostar_SELL | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E6_exit_mirror | 869 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E7_lateness_vs_Pstar_BUY | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E7_lateness_vs_Pstar_SELL | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E8_closer_quality_ladder_BUY | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:E8_closer_quality_ladder_SELL | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:EP_counts_BUY | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:EP_counts_SELL | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:Z1_zigzag_exit_label_BUY | 4345 | 0 | 0 | 0 | 0 | 0 | 0 |
| exit:Z1_zigzag_exit_label_SELL | 4343 | 2 | 0 | 0 | 0 | 0 | 0 |

## Trade summary (record schema)

records 4345; net sigma / trade mean 0.522 (median 0.516); win rate 1.000; trades / record 733; SL pips median 17.0; TP pips median 42.0; hold bars median 2.0; Sharpe mean 19.40

## Notes

- Early-close share under the menu is order geometry (coordinator ruling 2026-10-29): O* matches it, E1prime = 1.0 everywhere and the SL-destined share drives it; the 56-87 % range was a descriptive sigma-profile number, never a quota.

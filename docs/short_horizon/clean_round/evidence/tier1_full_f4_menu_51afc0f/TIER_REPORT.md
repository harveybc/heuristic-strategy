# Profile sanity tier report

Records 135 (errors 0); PASS effective 121, FAIL 14; exec profiles {'full_f4': 135} (expected full_f4: OK); levels_clipped total 0 in 0 records (OK).

## Engine refusals by reason

| reason | count |
|---|---:|

Gate counters: {'days': 51664, 'placed': 26324, 'deferred_observing': 0, 'deferred_below_best': 3820, 'below_threshold': 0, 'entry_gate_closed': 1468}

## Horizon mix (executed, mean over records with entries)

{'h3': 0.004746187639674472, 'h6': 0.05006417380752288, 'h12': 0.30117474850295617, 'h24': 0.6440148900498465} over 135 records; dominance flags: none

## Activity

{'trades_per_week_median': 2.6947458901901795, 'trades_per_week_p10_p90': [1.5781669710970236, 3.26979692704416], 'early_close_share_pooled': 0.030960340373803373, 'records_with_zero_trades': 0}

## Net per family (median of the per-record median net sigma per trade)

| family | records | median net sigma / trade |
|---|---:|---:|
| B_cal | 3 | 0.587 |
| B_fine | 3 | 0.780 |
| B_gap | 3 | 0.565 |
| B_spread | 3 | 0.597 |
| B_trans | 3 | 0.572 |
| B_trend | 3 | 0.683 |
| B_vol | 3 | 0.607 |
| B_voldyn | 3 | 0.797 |
| F1 | 3 | 0.603 |
| F2 | 3 | 0.435 |
| F3 | 6 | 0.575 |
| F4 | 3 | 0.388 |
| F5 | 3 | 0.595 |
| H1 | 6 | 0.735 |
| H2 | 6 | 0.485 |
| H3 | 3 | 0.786 |
| H4 | 6 | 0.451 |
| H5 | 3 | 0.563 |
| H6 | 3 | 0.439 |
| I1 | 6 | 0.489 |
| I2 | 3 | 0.545 |
| I3 | 3 | 0.654 |
| I4 | 3 | 0.551 |
| LHS | 51 | 0.275 |

W14 control status: {'DEFECT': 9}; swapped-reference code symmetry max 3.115285807098189e-13 sigma; trades without a matched menu slot: 0

W13 columns (menu form governing / sealed form): {'governing_PASS': 58, 'sealed_PASS': 48, 'sealed_FAIL': 10}; Z1 control: {'BUY:DEFECT': 135, 'SELL:DEFECT': 134, 'SELL:VALIDATED': 1}; W8 clause (b): {'records': 77, 'clause_b_PASS': 75, 'not_evaluable_elements': 12, 'Ostar_pass_with_sealed_clause_b_False': 57, 'Ostar_pass_with_sealed_clause_b_True': 20, 'clause_b_NOT_EVALUABLE': 2}; closer support: {'status': {'OK': 135}, 'records_extended': 24, 'max_extra_passes': 3}

## Witnesses (governing forms)

| witness | PASS | FAIL | NOT_EVALUABLE | sealed-unvalidated (strict) | defect-corrected | excused by O* |
|---|---:|---:|---:|---:|---:|---:|
| W0_activity | 135 | 0 | 0 | 0 | 0 | 0 |
| W10_spread_window | 26 | 0 | 0 | 0 | 0 | 0 |
| W11_cutoff | 54 | 0 | 0 | 54 | 0 | 0 |
| W12_missing | 18 | 0 | 0 | 0 | 0 | 0 |
| W13_tpsl_track_sigma | 58 | 0 | 0 | 0 | 0 | 0 |
| W14_mirror | 9 | 0 | 0 | 0 | 0 | 0 |
| W15_no_open_high_risk | 135 | 0 | 0 | 0 | 0 | 0 |
| W16_sl_tp_every_order | 135 | 0 | 0 | 0 | 0 | 0 |
| W17_no_open_in_burst | 9 | 0 | 0 | 0 | 0 | 0 |
| W1_side | 122 | 13 | 0 | 0 | 77 | 0 |
| W2_truth_positive_open | 135 | 0 | 0 | 0 | 0 | 0 |
| W2_wrong_side | 135 | 0 | 0 | 0 | 0 | 0 |
| W3_nop | 135 | 0 | 0 | 0 | 10 | 0 |
| W4_early_close_justified | 135 | 0 | 0 | 0 | 135 | 0 |
| W4_sl_exits | 135 | 0 | 0 | 0 | 0 | 0 |
| W5_reverse | 135 | 0 | 0 | 0 | 77 | 0 |
| W6_volume | 135 | 0 | 0 | 0 | 0 | 0 |
| W7_best_cell | 135 | 0 | 0 | 0 | 0 | 0 |
| W8_early_close_at_switch | 76 | 1 | 0 | 34 | 21 | 0 |
| W8_old_direction_after_lag | 73 | 4 | 0 | 25 | 50 | 0 |
| W9_gap_loss_bounded | 47 | 0 | 0 | 46 | 0 | 0 |
| WA_side_comparison | 135 | 0 | 0 | 135 | 0 | 0 |
| exit:E1_sl_destined_closed_early_BUY | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E1_sl_destined_closed_early_SELL | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E1p_closed_before_SL_BUY | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E1p_closed_before_SL_SELL | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E2_exit_regret_BUY | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E2_exit_regret_SELL | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E2p_regret_SL_destined_BUY | 133 | 2 | 0 | 0 | 0 | 2 |
| exit:E2p_regret_SL_destined_SELL | 133 | 2 | 0 | 0 | 0 | 2 |
| exit:E3_tick_agreement_BUY | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E3_tick_agreement_SELL | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E4_no_destructive_close_BUY | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E4_no_destructive_close_SELL | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E5_early_close_vs_Ostar_BUY | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E5_early_close_vs_Ostar_SELL | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E6_exit_mirror | 9 | 0 | 0 | 0 | 0 | 0 |
| exit:E7_lateness_vs_Pstar_BUY | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E7_lateness_vs_Pstar_SELL | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E8_closer_quality_ladder_BUY | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:E8_closer_quality_ladder_SELL | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:EP_counts_BUY | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:EP_counts_SELL | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:Z1_zigzag_exit_label_BUY | 135 | 0 | 0 | 0 | 0 | 0 |
| exit:Z1_zigzag_exit_label_SELL | 135 | 0 | 0 | 0 | 0 | 0 |

## Trade summary (record schema)

records 135; net sigma / trade mean 0.508 (median 0.507); win rate 1.000; trades / record 195; SL pips median 17.0; TP pips median 42.0; hold bars median 3.0; Sharpe mean 12.11

## Notes

- Early-close share under the menu is order geometry (coordinator ruling 2026-10-29): O* matches it, E1prime = 1.0 everywhere and the SL-destined share drives it; the 56-87 % range was a descriptive sigma-profile number, never a quota.
- fine_counter_pivot and nop_hard are side-neutral (unchanged by ruling); their activity cost is reported as a finding (truth-positive winner ticks removed per side and horizon, veto diagnosis).

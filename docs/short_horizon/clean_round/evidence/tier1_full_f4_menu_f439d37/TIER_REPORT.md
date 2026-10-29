# Profile sanity tier report

Records 135 (errors 0); PASS effective 0, FAIL 135; exec profiles {'full_f4': 135} (expected full_f4: OK); levels_clipped total 0 in 0 records (OK).

## Engine refusals by reason

| reason | count |
|---|---:|

Gate counters: {'days': 51638, 'placed': 39078, 'deferred_observing': 0, 'deferred_below_best': 10060, 'below_threshold': 0, 'entry_gate_closed': 3113}

## Horizon mix (executed, mean over records with entries)

{'h3': 0.011866377006962204, 'h6': 0.0936140126744913, 'h12': 0.29563779650075384, 'h24': 0.5988818138177926} over 135 records; dominance flags: none

## Activity

{'trades_per_week_median': 3.842269259697002, 'trades_per_week_p10_p90': [2.666380143977651, 4.618459224239819], 'early_close_share_pooled': 0.021418701059419624, 'records_with_zero_trades': 0}

## Net per family (median of the per-record median net sigma per trade)

| family | records | median net sigma / trade |
|---|---:|---:|
| B_cal | 3 | 0.586 |
| B_fine | 3 | 0.769 |
| B_gap | 3 | 0.567 |
| B_spread | 3 | 0.602 |
| B_trans | 3 | 0.572 |
| B_trend | 3 | 0.692 |
| B_vol | 3 | 0.613 |
| B_voldyn | 3 | 0.807 |
| F1 | 3 | 0.594 |
| F2 | 3 | 0.434 |
| F3 | 6 | 0.575 |
| F4 | 3 | 0.385 |
| F5 | 3 | 0.591 |
| H1 | 6 | 0.733 |
| H2 | 6 | 0.473 |
| H3 | 3 | 0.780 |
| H4 | 6 | 0.456 |
| H5 | 3 | 0.565 |
| H6 | 3 | 0.446 |
| I1 | 6 | 0.495 |
| I2 | 3 | 0.546 |
| I3 | 3 | 0.660 |
| I4 | 3 | 0.550 |
| LHS | 51 | 0.271 |

W14 control status: {'DEFECT': 9}; swapped-reference code symmetry max 2.779998453661392e-13 sigma; trades without a matched menu slot: 0

## Witnesses (governing forms)

| witness | PASS | FAIL | NOT_EVALUABLE | sealed-unvalidated (strict) | defect-corrected | excused by O* |
|---|---:|---:|---:|---:|---:|---:|
| W0_activity | 135 | 0 | 0 | 0 | 0 | 0 |
| W10_spread_window | 26 | 0 | 0 | 0 | 0 | 0 |
| W11_cutoff | 54 | 0 | 0 | 54 | 0 | 0 |
| W12_missing | 18 | 0 | 0 | 0 | 0 | 0 |
| W13_tpsl_track_sigma | 53 | 5 | 0 | 0 | 5 | 0 |
| W14_mirror | 9 | 0 | 0 | 0 | 0 | 0 |
| W15_no_open_high_risk | 135 | 0 | 0 | 0 | 0 | 0 |
| W16_sl_tp_every_order | 135 | 0 | 0 | 0 | 0 | 0 |
| W17_no_open_in_burst | 8 | 1 | 0 | 0 | 0 | 0 |
| W1_side | 108 | 27 | 0 | 0 | 77 | 0 |
| W2_truth_positive_open | 135 | 0 | 0 | 0 | 0 | 0 |
| W2_wrong_side | 135 | 0 | 0 | 0 | 0 | 0 |
| W3_nop | 135 | 0 | 0 | 0 | 10 | 0 |
| W4_early_close_justified | 135 | 0 | 0 | 0 | 135 | 0 |
| W4_sl_exits | 135 | 0 | 0 | 0 | 0 | 0 |
| W5_reverse | 135 | 0 | 0 | 0 | 77 | 0 |
| W6_volume | 135 | 0 | 0 | 0 | 0 | 0 |
| W7_best_cell | 135 | 0 | 0 | 0 | 0 | 0 |
| W8_early_close_at_switch | 70 | 7 | 0 | 35 | 20 | 0 |
| W8_old_direction_after_lag | 76 | 1 | 0 | 25 | 50 | 0 |
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
| exit:Z1_zigzag_exit_label_BUY | 0 | 135 | 0 | 0 | 0 | 0 |
| exit:Z1_zigzag_exit_label_SELL | 1 | 134 | 0 | 0 | 0 | 0 |

## Trade summary (record schema)

records 135; net sigma / trade mean 0.500 (median 0.510); win rate 1.000; trades / record 289; SL pips median 17.0; TP pips median 42.0; hold bars median 3.0; Sharpe mean 14.61

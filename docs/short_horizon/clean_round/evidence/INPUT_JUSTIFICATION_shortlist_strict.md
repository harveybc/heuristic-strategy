# Input justification: `shortlist_strict` (116 columns, 25 extractor groups)

Every column below reaches the consumer exactly as listed: store value -> (decided transform, already applied in the store) -> fixed TRAIN z-score with the constants of the fit block -> window. Nothing else: no rolling statistic, no clipping, no winsorizing, no NaN fill (a window with a non-finite value is excluded and counted), no agent feedback (the consumer runs with agent state `none`; `minimal` is a declared control).

- Preprocessing decisions: {'NONE': 90, 'LOG': 13, 'DIFF': 8, 'ASINH': 5} (rule: ADF p > 0.05 and KPSS p < 0.05 on TRAIN -> DIFF; positive and strongly skewed -> ASINH only if it reduces the measured skew; otherwise NONE).
- Selection: {'RETAINED': 115, 'RESCUED_INTERACTION': 1}
- Encoding: v3 encoding contrasts gave <= 0.02 nats, inconsistent across models/tasks; consumer evidence: consumer arm strict+encodings INNER_A -1.38 vs strict -1.13 (no gain). Adopted: no -> every column keeps its base encoding (sin/cos clock, regime probabilities, linear event hours, raw ranges, differenced level).
- Heavy tails KEPT (not clipped): 28 columns reach |z| > 10 in the fit block or in 2024; listed in the table (max |z|). This is measured information, not an error; any transform of them requires its own matched contrast.

| column | family | preprocessing (metric) | selection | encoding | TRAIN mean / std | max abs z fit / 2023H2 / 2024 |
|---|---|---|---|---|---|---|
| OPEN@diff | inv.raw | DIFF (ADF p 0.763, KPSS p 0.01, skew -0.508, kurt -0.744) | RETAINED nan [5] | - | -1.4e-06 / 0.00116 | 15 / 8.87 / 9.35 |
| HIGH@diff | inv.raw | DIFF (ADF p 0.755, KPSS p 0.01, skew -0.507, kurt -0.745) | RETAINED nan [5] | - | -1.4e-06 / 0.00105 | 19.8 / 9.68 / 10 |
| LOW@diff | inv.raw | DIFF (ADF p 0.765, KPSS p 0.01, skew -0.509, kurt -0.744) | RETAINED nan [5] | - | -1.41e-06 / 0.00105 | 16 / 8.01 / 10.4 |
| CLOSE@diff | inv.raw | DIFF (ADF p 0.764, KPSS p 0.01, skew -0.508, kurt -0.744) | RETAINED nan [5] | base encoding | -1.42e-06 / 0.00115 | 15 / 8.86 / 9.44 |
| RAW_BAR_LOG_RETURN | inv.raw | NONE (ADF p 0, KPSS p 0.1, skew 0.136, kurt 14.9) | RETAINED nan [5] | - | -8.79e-07 / 0.00106 | 16.4 / 8.96 / 9.83 |
| RAW_LOG_RANGE@log | inv.raw | LOG (ADF p 1.47e-12, KPSS p 0.01, skew 3.09, kurt 20.1) | RETAINED nan [5] | - | -6.82 / 0.677 | 4.41 / 3.27 / 3.85 |
| RAW_CLOSE_LOG_RETURN | inv.raw | NONE (ADF p 0, KPSS p 0.1, skew 0.0648, kurt 14.9) | RETAINED nan [5] | - | -1.28e-06 / 0.00106 | 16.3 / 8.89 / 9.85 |
| RAW_ELAPSED_HOURS | inv.raw | NONE (ADF p 0, KPSS p 0.1, skew 10.9, kurt 118) | RETAINED nan [5] | - | 1.4 / 4.41 | 16.9 / 18.3 / 18.3 |
| RSI | inv.momentum | NONE (ADF p 2.17e-30, KPSS p 0.0258, skew -0.0254, kurt -0.297) | RETAINED nan [4] | - | 49.7 / 12.6 | 3.28 / 3.33 / 3.02 |
| Stochastic_K | inv.momentum | NONE (ADF p 0, KPSS p 0.0334, skew 0.0254, kurt -1.29) | RETAINED nan [4] | - | 49.7 / 27.6 | 1.81 / 1.81 / 1.81 |
| Stochastic_D | inv.momentum | NONE (ADF p 0, KPSS p 0.0322, skew 0.0269, kurt -1.27) | RETAINED nan [4] | - | 49.7 / 26.7 | 1.84 / 1.82 / 1.84 |
| CCI | inv.momentum | NONE (ADF p 0, KPSS p 0.1, skew -0.00258, kurt -0.0652) | RETAINED nan [4] | - | -0.646 / 118 | 5.03 / 4.7 / 5.09 |
| WilliamsR | inv.momentum | NONE (ADF p 0, KPSS p 0.0348, skew 0.0248, kurt -1.3) | RETAINED nan [4] | - | -50.3 / 29.8 | 1.69 / 1.69 / 1.69 |
| ROC | inv.momentum | NONE (ADF p 0, KPSS p 0.1, skew 0.0943, kurt 4.06) | RETAINED nan [4] | - | -0.000741 / 0.324 | 8.18 / 4.95 / 5.64 |
| BB_position | inv.momentum | NONE (ADF p 0, KPSS p 0.1, skew 0.0112, kurt -0.617) | RETAINED nan [4] | - | 0.498 / 0.342 | 3.1 / 3.11 / 3.12 |
| price_minus_ema | inv.momentum | NONE (ADF p 2.33e-30, KPSS p 0.0951, skew 0.022, kurt 2.33) | RETAINED nan [4] | - | -1.72e-05 / 0.00255 | 6.41 / 4.73 / 5.1 |
| hod_sin | calendar | NONE (ADF p 0, KPSS p 0.1, skew -0.000616, kurt -1.5) | RETAINED nan [2] | base encoding | -0.000107 / 0.707 | 1.41 / 1.41 / 1.41 |
| hod_cos | calendar | NONE (ADF p 0, KPSS p 0.1, skew 0.0018, kurt -1.5) | RETAINED nan [2] | base encoding | -0.000906 / 0.707 | 1.42 / 1.42 / 1.42 |
| dow_sin | calendar | NONE (ADF p 0, KPSS p 0.1, skew -0.357, kurt -1.22) | RETAINED nan [2] | - | 0.348 / 0.522 | 2.16 / 2.16 / 2.16 |
| dow_cos | calendar | NONE (ADF p 0, KPSS p 0.1, skew 0.159, kurt -1.65) | RETAINED nan [2] | - | -0.0596 / 0.776 | 1.36 / 1.36 / 1.36 |
| cal.utc_hod_sin | calendar | NONE (ADF p 0, KPSS p 0.1, skew -0.000856, kurt -1.5) | RETAINED nan [2] | - | 0.000132 / 0.707 | 1.41 / 1.41 / 1.41 |
| cal.utc_hod_cos | calendar | NONE (ADF p 0, KPSS p 0.1, skew 0.00185, kurt -1.5) | RETAINED nan [2] | - | -0.000903 / 0.707 | 1.42 / 1.42 / 1.42 |
| cal.utc_dow_sin | calendar | NONE (ADF p 0, KPSS p 0.1, skew -0.384, kurt -1.16) | RETAINED nan [2] | - | 0.345 / 0.527 | 2.14 / 2.14 / 2.14 |
| cal.utc_dow_cos | calendar | NONE (ADF p 0, KPSS p 0.1, skew 0.127, kurt -1.65) | RETAINED nan [2] | - | -0.0468 / 0.775 | 1.35 / 1.35 / 1.35 |
| cal.utc_session_asia | calendar | NONE (ADF p 0, KPSS p 0.1, skew 0.917, kurt -1.16) | RETAINED nan [2] | - | 0.291 / 0.454 | 1.56 / 1.56 / 1.56 |
| cal.utc_session_europe | calendar | NONE (ADF p 0, KPSS p 0.1, skew 1.15, kurt -0.671) | RETAINED nan [2] | - | 0.25 / 0.433 | 1.73 / 1.73 / 1.73 |
| cal.utc_session_us | calendar | NONE (ADF p 0, KPSS p 0.1, skew 0.707, kurt -1.5) | RETAINED nan [2] | - | 0.333 / 0.471 | 1.41 / 1.41 / 1.41 |
| cal.london_hod_sin | calendar | NONE (ADF p 0, KPSS p 0.1, skew -0.000856, kurt -1.5) | RETAINED nan [2] | - | 0.000132 / 0.707 | 1.41 / 1.41 / 1.41 |
| cal.london_hod_cos | calendar | NONE (ADF p 0, KPSS p 0.1, skew 0.00185, kurt -1.5) | RETAINED nan [2] | - | -0.000903 / 0.707 | 1.42 / 1.42 / 1.42 |
| px.logret_1 | px.logret | NONE (ADF p 0, KPSS p 0.1, skew 0.0648, kurt 14.9) | RETAINED nan [4] | - | -1.28e-06 / 0.00106 | 16.3 / 8.89 / 9.85 |
| px.logret_2 | px.logret | NONE (ADF p 0, KPSS p 0.1, skew 0.146, kurt 10.7) | RETAINED nan [4] | - | -2.56e-06 / 0.00148 | 14.4 / 7.95 / 7.86 |
| px.logret_4 | px.logret | NONE (ADF p 0, KPSS p 0.1, skew 0.218, kurt 7.5) | RETAINED nan [4] | - | -5.1e-06 / 0.00208 | 11.3 / 6.3 / 8.3 |
| px.logret_6 | px.logret | NONE (ADF p 0, KPSS p 0.1, skew 0.186, kurt 5.49) | RETAINED nan [4] | - | -7.45e-06 / 0.00253 | 8.63 / 5.62 / 6.66 |
| px.logret_12 | px.logret | NONE (ADF p 0, KPSS p 0.1, skew 0.049, kurt 3.4) | RETAINED nan [4] | - | -1.55e-05 / 0.00353 | 6.95 / 4.84 / 5.9 |
| px.logret_24 | px.logret | NONE (ADF p 1.71e-28, KPSS p 0.1, skew 0.0774, kurt 1.85) | RETAINED nan [4] | - | -3.12e-05 / 0.00491 | 7.24 / 4.13 / 3.9 |
| px.logret_48 | px.logret | NONE (ADF p 3.24e-28, KPSS p 0.052, skew 0.00613, kurt 1.48) | RETAINED nan [4] | - | -5.76e-05 / 0.00707 | 5.37 / 2.85 / 2.85 |
| px.range_1@log | px.range | LOG (ADF p 1.47e-12, KPSS p 0.01, skew 3.09, kurt 20.1) | RETAINED nan [2] | base encoding | -6.82 / 0.677 | 4.41 / 3.27 / 3.85 |
| px.range_2@log | px.range | LOG (ADF p 7.08e-15, KPSS p 0.01, skew 2.62, kurt 12.6) | RETAINED nan [2] | base encoding | -6.45 / 0.661 | 4.01 / 3.22 / 3.49 |
| px.range_4@log | px.range | LOG (ADF p 1.88e-18, KPSS p 0.01, skew 2.29, kurt 8.8) | RETAINED nan [2] | base encoding | -6.07 / 0.634 | 3.7 / 2.94 / 3.39 |
| px.range_6@log | px.range | LOG (ADF p 2.39e-19, KPSS p 0.01, skew 2.1, kurt 7.09) | RETAINED nan [2] | base encoding | -5.83 / 0.609 | 3.51 / 2.98 / 3.4 |
| px.range_12 | px.range | NONE (ADF p 2.16e-22, KPSS p 0.01, skew 1.83, kurt 5.11) | RETAINED nan [2] | base encoding | 0.00516 / 0.00308 | 7.56 / 4.09 / 5.36 |
| px.range_24 | px.range | NONE (ADF p 8.99e-19, KPSS p 0.01, skew 1.59, kurt 3.82) | RETAINED nan [2] | base encoding | 0.0075 / 0.00382 | 7.65 / 3.55 / 4.21 |
| px.range_48 | px.range | NONE (ADF p 1.37e-20, KPSS p 0.01, skew 1.53, kurt 3.43) | RETAINED nan [2] | base encoding | 0.0106 / 0.00515 | 6.14 / 2.15 / 2.51 |
| px.close_loc_1 | px.close_loc | NONE (ADF p 0, KPSS p 0.01, skew 0.00551, kurt -1.27) | RETAINED nan [4] | - | 0.495 / 0.302 | 1.67 / 1.67 / 1.67 |
| px.close_loc_2 | px.close_loc | NONE (ADF p 0, KPSS p 0.01, skew 0.000645, kurt -1.26) | RETAINED nan [4] | - | 0.497 / 0.297 | 1.69 / 1.69 / 1.69 |
| px.close_loc_4 | px.close_loc | NONE (ADF p 0, KPSS p 0.01, skew 0.00565, kurt -1.27) | RETAINED nan [4] | - | 0.499 / 0.296 | 1.69 / 1.69 / 1.69 |
| px.close_loc_6 | px.close_loc | NONE (ADF p 0, KPSS p 0.01, skew 0.0145, kurt -1.29) | RETAINED nan [4] | - | 0.499 / 0.297 | 1.69 / 1.69 / 1.69 |
| px.close_loc_12 | px.close_loc | NONE (ADF p 0, KPSS p 0.0237, skew 0.0165, kurt -1.3) | RETAINED nan [4] | - | 0.498 / 0.298 | 1.69 / 1.69 / 1.69 |
| px.close_loc_24 | px.close_loc | NONE (ADF p 0, KPSS p 0.0721, skew 0.0468, kurt -1.3) | RETAINED nan [4] | - | 0.497 / 0.298 | 1.69 / 1.69 / 1.69 |
| px.close_loc_48 | px.close_loc | NONE (ADF p 6.84e-30, KPSS p 0.0246, skew 0.0485, kurt -1.33) | RETAINED nan [4] | - | 0.494 / 0.301 | 1.68 / 1.68 / 1.68 |
| px.er_1 | px.er | NONE (ADF p , KPSS p , skew , kurt ) | RETAINED nan [2] | - | 1 / 1 | 0 / 0 / 0 |
| px.er_2 | px.er | NONE (ADF p 0, KPSS p 0.1, skew -0.706, kurt -1.07) | RETAINED nan [2] | - | 0.714 / 0.342 | 2.09 / 2.09 / 2.09 |
| px.er_4 | px.er | NONE (ADF p 0, KPSS p 0.1, skew 0.158, kurt -1.21) | RETAINED nan [2] | - | 0.506 / 0.312 | 1.62 / 1.62 / 1.62 |
| px.er_6 | px.er | NONE (ADF p 0, KPSS p 0.1, skew 0.396, kurt -0.855) | RETAINED nan [2] | - | 0.418 / 0.275 | 2.12 / 2.12 / 2.12 |
| px.er_12 | px.er | NONE (ADF p 0, KPSS p 0.1, skew 0.633, kurt -0.297) | RETAINED nan [2] | - | 0.306 / 0.213 | 3.25 / 2.99 / 3.15 |
| px.er_24 | px.er | NONE (ADF p 0, KPSS p 0.1, skew 0.747, kurt -0.0106) | RETAINED nan [2] | - | 0.225 / 0.16 | 3.77 / 3.69 / 3.4 |
| px.er_48 | px.er | NONE (ADF p 2.06e-30, KPSS p 0.1, skew 0.746, kurt 0.0707) | RETAINED nan [2] | - | 0.161 / 0.116 | 4.16 / 4.06 / 3.91 |
| px.range_ratio_1_24 | px.range_ratio | NONE (ADF p 0, KPSS p 0.1, skew 1.69, kurt 4.18) | RETAINED nan [2] | base encoding | 0.199 / 0.132 | 6.05 / 6.05 / 6.05 |
| px.hours_since_session_open | px.hours_since_session_open | NONE (ADF p 0, KPSS p 0.1, skew 0.00813, kurt -1.2) | RETAINED nan [3] | - | 59.2 / 34.6 | 1.73 / 1.73 / 1.73 |
| cdl.body_pips | cdl | NONE (ADF p 0, KPSS p 0.1, skew 0.0814, kurt 12.4) | RETAINED nan [3] | - | -0.01 / 11.5 | 15.1 / 8.93 / 9.43 |
| cdl.upper_wick_pips@asinh | cdl | ASINH (ADF p 1.96e-23, KPSS p 0.01, skew 3.48, kurt 26.4) | RETAINED nan [3] | - | 1.68 / 0.926 | 3.81 / 2.74 / 2.92 |
| cdl.lower_wick_pips@asinh | cdl | ASINH (ADF p 6.87e-25, KPSS p 0.01, skew 2.8, kurt 14) | RETAINED nan [3] | - | 1.64 / 0.934 | 3.37 / 2.77 / 3.53 |
| cdl.range_pips@asinh | cdl | ASINH (ADF p 2.29e-14, KPSS p 0.01, skew 2.92, kurt 17.7) | RETAINED nan [3] | - | 3.2 / 0.657 | 4.38 / 3.31 / 3.64 |
| cdl.close_minus_low_pips@asinh | cdl | ASINH (ADF p 7.63e-24, KPSS p 0.01, skew 3.54, kurt 26.9) | RETAINED nan [3] | - | 2.25 / 1.02 | 3.55 / 3.02 / 2.62 |
| cdl.high_minus_close_pips@asinh | cdl | ASINH (ADF p 1.37e-22, KPSS p 0.01, skew 3.26, kurt 20.5) | RETAINED nan [3] | - | 2.28 / 1.01 | 3.35 / 2.77 / 3.07 |
| fx.usdchf.ret_1 | fx.usdchf | NONE (ADF p 0, KPSS p 0.1, skew -0.275, kurt 16.3) | RETAINED nan [2] | - | -3.58e-06 / 0.00105 | 17.5 / 8.28 / 10.2 |
| fx.usdchf.ret_4 | fx.usdchf | NONE (ADF p 0, KPSS p 0.1, skew -0.407, kurt 7.81) | RETAINED nan [2] | - | -1.46e-05 / 0.00205 | 11.2 / 6.89 / 6.61 |
| fx.usdchf.ret_24 | fx.usdchf | NONE (ADF p 9.26e-29, KPSS p 0.1, skew -0.53, kurt 3.56) | RETAINED nan [2] | - | -8.48e-05 / 0.00494 | 7.67 / 3.97 / 5.53 |
| fx.usdchf.rv_24@log | fx.usdchf | LOG (ADF p 3.29e-14, KPSS p 0.01, skew 2.48, kurt 10.2) | RETAINED nan [2] | - | -5.43 / 0.374 | 4.22 / 2.72 / 3.15 |
| dly.gold.ret_1d | dly.gold | NONE (ADF p 2.74e-29, KPSS p 0.1, skew -0.208, kurt 4.03) | RETAINED nan [2] | - | 0.000352 / 0.0107 | 5.34 / 2.82 / 3.29 |
| dly.gold.ret_5d | dly.gold | NONE (ADF p 1.48e-18, KPSS p 0.0105, skew -0.157, kurt 2.58) | RETAINED nan [2] | - | 0.00137 / 0.0234 | 5.17 / 2.15 / 2.34 |
| cot.eur.net_long | cot.eur | NONE (ADF p 7.46e-08, KPSS p 0.01, skew 1.22, kurt 0.665) | RETAINED nan [2] | - | 3.97e+04 / 6.78e+04 | 2.54 / 1.95 / 1.7 |
| cot.eur.net_change | cot.eur | NONE (ADF p 2.16e-21, KPSS p 0.1, skew 0.0401, kurt 2.84) | RETAINED nan [2] | - | 1.28e+03 / 6.7e+04 | 2.92 / 2.59 / 1.77 |
| cot.eur.net_share | cot.eur | NONE (ADF p 4.2e-08, KPSS p 0.01, skew 0.131, kurt -0.14) | RETAINED nan [2] | - | 0.191 / 0.338 | 2.79 / 1.67 / 2.41 |
| cot.eur.net_z156 | cot.eur | NONE (ADF p 7.17e-08, KPSS p 0.01, skew 1.19, kurt 0.827) | RETAINED nan [2] | - | 0.263 / 1.03 | 2.85 / 1.68 / 1.96 |
| fac.fx_f1.ret_1 | fac.fx_f1 | NONE (ADF p 0, KPSS p 0.1, skew -0.635, kurt 21.4) | RETAINED nan [2] | - | 0.00991 / 1.54 | 21.2 / 6.72 / 7.02 |
| fac.fx_f1.ret_4 | fac.fx_f1 | NONE (ADF p 0, KPSS p 0.1, skew -0.336, kurt 7.74) | RETAINED nan [2] | - | 0.0404 / 2.98 | 12.3 / 6.4 / 5.51 |
| fac.fx_f1.ret_24 | fac.fx_f1 | NONE (ADF p 1.19e-28, KPSS p 0.1, skew -0.624, kurt 5.25) | RETAINED nan [2] | - | 0.234 / 7.31 | 9.76 / 3.86 / 2.87 |
| fac.fx_f1.kalman_slope | fac.fx_f1 | NONE (ADF p 0, KPSS p 0.1, skew -0.463, kurt 12.6) | RETAINED nan [2] | - | 0.00711 / 0.603 | 17.6 / 7.03 / 6.38 |
| ev.tokyo_fix.hours_since | ev | NONE (ADF p 0, KPSS p 0.1, skew 0.000622, kurt -1.2) | RETAINED nan [4] | base encoding | 11.6 / 6.92 | 1.66 / 1.66 / 1.66 |
| ev.tokyo_fix.hours_to | ev | NONE (ADF p 0, KPSS p 0.1, skew -0.000622, kurt -1.2) | RETAINED nan [4] | base encoding | 12.4 / 6.92 | 1.66 / 1.66 / 1.66 |
| ev.london_fix.hours_since | ev | NONE (ADF p 0, KPSS p 0.1, skew -0.00145, kurt -1.2) | RETAINED nan [4] | base encoding | 11.5 / 6.93 | 1.66 / 1.66 / 1.66 |
| ev.london_fix.hours_to | ev | NONE (ADF p 0, KPSS p 0.1, skew 0.00145, kurt -1.2) | RETAINED nan [4] | base encoding | 12.5 / 6.93 | 1.66 / 1.66 / 1.66 |
| ev.ny_cut.hours_since | ev | NONE (ADF p 0, KPSS p 0.1, skew -0.00114, kurt -1.21) | RETAINED nan [4] | base encoding | 11.5 / 6.93 | 1.66 / 1.66 / 1.66 |
| ev.ny_cut.hours_to | ev | NONE (ADF p 0, KPSS p 0.1, skew 0.00114, kurt -1.21) | RETAINED nan [4] | base encoding | 12.5 / 6.93 | 1.66 / 1.66 / 1.66 |
| act.p_hit | act | NONE (ADF p 0, KPSS p 0.1, skew 0.429, kurt -1.03) | RETAINED nan [4] | - | 0.289 / 0.251 | 2.54 / 2.26 / 2.06 |
| wv.lc.A4@diff | wavelet/log_close | DIFF (ADF p 0.764, KPSS p 0.01, skew -0.604, kurt -0.632) | RETAINED P_decided+Cgroup [15] | - | -3.82e-06 / 0.00297 | 13.3 / 7.17 / 7.78 |
| wv.lc.D4 | wavelet/log_close | NONE (ADF p 0, KPSS p 0.1, skew -0.0845, kurt 3.31) | RETAINED P_decided+Cgroup [15] | - | -7.2e-07 / 0.00392 | 6.28 / 4.43 / 4.7 |
| wv.lc.D3 | wavelet/log_close | NONE (ADF p 0, KPSS p 0.1, skew -0.0761, kurt 5.2) | RETAINED P_decided+Cgroup [15] | - | -2.6e-07 / 0.00197 | 9.95 / 5.68 / 5.45 |
| wv.lc.D2 | wavelet/log_close | NONE (ADF p 0, KPSS p 0.1, skew -0.0852, kurt 7.25) | RETAINED P_decided+Cgroup [15] | - | -5.03e-08 / 0.00103 | 12.9 / 6.3 / 6.15 |
| wv.lc.D1 | wavelet/log_close | NONE (ADF p 0, KPSS p 0.1, skew -0.0326, kurt 10.1) | RETAINED P_decided+Cgroup [15] | - | -5.7e-08 / 0.000627 | 11.8 / 7.59 / 7.81 |
| wv.lc.energy_D4@log | wavelet/log_close | LOG (ADF p 0.000276, KPSS p 0.01, skew 2.81, kurt 10.2) | RETAINED P_decided+Cgroup [15] | - | -11.6 / 0.792 | 3.23 / 1.72 / 2.66 |
| wv.lc.energy_D3@log | wavelet/log_close | LOG (ADF p 0.00692, KPSS p 0.01, skew 2.55, kurt 9.31) | RETAINED P_decided+Cgroup [15] | - | -12.8 / 0.761 | 3.5 / 2.21 / 2.66 |
| wv.lc.energy_D2@log | wavelet/log_close | LOG (ADF p 0.0322, KPSS p 0.01, skew 2.35, kurt 7.78) | RETAINED P_decided+Cgroup [15] | - | -14 / 0.734 | 3.58 / 1.84 / 2.4 |
| wv.lc.energy_D1@diff | wavelet/log_close | DIFF (ADF p 0.118, KPSS p 0.01, skew 1.82, kurt 3.48) | RETAINED P_decided+Cgroup [15] | - | 5.35e-12 / 1.16e-07 | 10.3 / 3.02 / 2.65 |
| wv.r1.A4 | wavelet/logret_1 | NONE (ADF p 0, KPSS p 0.1, skew 0.173, kurt 9.66) | RETAINED P_decided+Cgroup [9] | - | -3.82e-06 / 0.00297 | 13.3 / 7.17 / 7.78 |
| wv.r1.D4 | wavelet/logret_1 | NONE (ADF p 0, KPSS p 0.1, skew -0.0437, kurt 3.89) | RETAINED P_decided+Cgroup [9] | - | 4.63e-08 / 0.00107 | 7.24 / 4.81 / 5.44 |
| wv.r1.D3 | wavelet/logret_1 | NONE (ADF p 0, KPSS p 0.1, skew -0.12, kurt 5.67) | RETAINED P_decided+Cgroup [9] | - | 1.69e-07 / 0.00105 | 12 / 6.32 / 5.17 |
| wv.r1.D2 | wavelet/logret_1 | NONE (ADF p 0, KPSS p 0.1, skew -0.0182, kurt 6.97) | RETAINED P_decided+Cgroup [9] | - | -1.33e-07 / 0.00106 | 10.2 / 6.52 / 6.51 |
| wv.r1.D1 | wavelet/logret_1 | NONE (ADF p 0, KPSS p 0.1, skew -0.0705, kurt 9.43) | RETAINED P_decided+Cgroup [9] | - | 3.94e-08 / 0.00108 | 13.3 / 6.64 / 6.51 |
| wv.r1.energy_D4@log | wavelet/logret_1 | LOG (ADF p 1.76e-05, KPSS p 0.01, skew 2.81, kurt 10.4) | RETAINED P_decided+Cgroup [9] | - | -14.2 / 0.799 | 3.56 / 1.99 / 2.88 |
| wv.r1.energy_D3@log | wavelet/logret_1 | LOG (ADF p 0.0066, KPSS p 0.01, skew 2.79, kurt 12.5) | RETAINED P_decided+Cgroup [9] | - | -14.1 / 0.752 | 3.7 / 1.86 / 2.49 |
| wv.r1.energy_D2@log | wavelet/logret_1 | LOG (ADF p 0.0476, KPSS p 0.01, skew 2.29, kurt 6.94) | RETAINED P_decided+Cgroup [9] | - | -14 / 0.721 | 3.4 / 1.89 / 2.44 |
| wv.r1.energy_D1@diff | wavelet/logret_1 | DIFF (ADF p 0.114, KPSS p 0.01, skew 1.73, kurt 2.96) | RETAINED P_decided+Cgroup [9] | - | 1.74e-11 / 1.89e-07 | 12.3 / 2.67 / 4.06 |
| hb.r1.phase_cos | hilbert/logret_1 | NONE (ADF p 0, KPSS p 0.1, skew -0.00827, kurt -1.42) | RETAINED S:phase [6] | - | 0.00271 / 0.688 | 1.46 / 1.46 / 1.46 |
| hb.r1.phase_sin | hilbert/logret_1 | NONE (ADF p 0, KPSS p 0.1, skew 0.00807, kurt -1.53) | RETAINED S:phase [6] | - | -0.00284 / 0.725 | 1.38 / 1.38 / 1.38 |
| hb.r1.inst_freq | hilbert/logret_1 | NONE (ADF p 0, KPSS p 0.1, skew -1.16, kurt 0.612) | RETAINED S:phase [6] | - | 0.154 / 0.24 | 2.73 / 2.73 / 2.73 |
| wv.usd.energy_D4@log | strength/fx.usd_strength.ret_1 | LOG (ADF p 1.62e-06, KPSS p 0.01, skew 4.05, kurt 21.5) | RETAINED C:energy_D4 [3] | - | -14.4 / 0.821 | 3.75 / 1.8 / 2.89 |
| wv.eur.A4 | strength/fx.eur_strength.ret_1 | NONE (ADF p 0, KPSS p 0.1, skew -0.286, kurt 10.3) | RETAINED C:A4 [6] | - | 2.38e-05 / 0.00235 | 13.8 / 5.02 / 10.6 |
| kf.lc.level_minus_close | kalman/log_close | NONE (ADF p 0, KPSS p 0.1, skew -0.0718, kurt 14.9) | RETAINED S:approximation [5] | - | 1.23e-09 / 2.08e-05 | 16.3 / 8.9 / 9.83 |
| px.rv_24 | px.rv | NONE (ADF p 3e-12, KPSS p 0.01, skew 1.76, kurt 4.71) | RESCUED_INTERACTION nan [1] | - | 0.00468 / 0.00226 | 6.67 / 2.97 / 4.25 |
| rg.gmm.p0 | gmm/logret_1+rv_24+log_range | NONE (ADF p 6.59e-11, KPSS p 0.01, skew -0.0491, kurt -1.87) | RETAINED P_decided+Cgroup [12] | base encoding | 0.494 / 0.444 | 1.12 / 1.12 / 1.12 |
| rg.gmm.p1 | gmm/logret_1+rv_24+log_range | NONE (ADF p 9.33e-21, KPSS p 0.01, skew 0.375, kurt -1.67) | RETAINED P_decided+Cgroup [12] | base encoding | 0.417 / 0.415 | 1.39 / 1.39 / 1.39 |
| rg.gmm.p2 | gmm/logret_1+rv_24+log_range | NONE (ADF p 1.86e-20, KPSS p 0.01, skew 2.95, kurt 7.17) | RETAINED P_decided+Cgroup [12] | base encoding | 0.0886 / 0.249 | 3.65 / 3.65 / 3.65 |
| rg.gmm.entropy | gmm/logret_1+rv_24+log_range | NONE (ADF p 0, KPSS p 0.01, skew 0.929, kurt -0.491) | RETAINED P_decided+Cgroup [12] | base encoding | 0.246 / 0.213 | 2.43 / 2.41 / 2.46 |
| fd.lc.d0.3@diff | fracdiff/log_close | DIFF (ADF p 0.0788, KPSS p 0.01, skew -0.64, kurt -0.424) | RETAINED P_raw+Cgroup [4] | - | -1.1e-07 / 0.00113 | 15.5 / 8.38 / 9.49 |

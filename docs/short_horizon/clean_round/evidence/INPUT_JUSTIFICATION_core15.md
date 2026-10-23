# Input justification: `core15` (15 columns, 5 extractor groups)

Every column below reaches the consumer exactly as listed: store value -> (decided transform, already applied in the store) -> fixed TRAIN z-score with the constants of the fit block -> window. Nothing else: no rolling statistic, no clipping, no winsorizing, no NaN fill (a window with a non-finite value is excluded and counted), no agent feedback (the consumer runs with agent state `none`; `minimal` is a declared control).

- Preprocessing decisions: {'NONE': 13, 'DIFF': 1, 'LOG': 1} (rule: ADF p > 0.05 and KPSS p < 0.05 on TRAIN -> DIFF; positive and strongly skewed -> ASINH only if it reduces the measured skew; otherwise NONE).
- Selection: {'RETAINED': 15}
- Encoding: v3 encoding contrasts gave <= 0.02 nats, inconsistent across models/tasks; consumer evidence: consumer arm strict+encodings INNER_A -1.38 vs strict -1.13 (no gain). Adopted: no -> every column keeps its base encoding (sin/cos clock, regime probabilities, linear event hours, raw ranges, differenced level).
- Heavy tails KEPT (not clipped): 3 columns reach |z| > 10 in the fit block or in 2024; listed in the table (max |z|). This is measured information, not an error; any transform of them requires its own matched contrast.

| column | family | preprocessing (metric) | selection | encoding | TRAIN mean / std | max abs z fit / 2023H2 / 2024 |
|---|---|---|---|---|---|---|
| HIGH@diff | inv.raw | DIFF (ADF p 0.755, KPSS p 0.01, skew -0.507, kurt -0.745) | RETAINED  [5] | - | -1.4e-06 / 0.00105 | 19.8 / 9.68 / 10 |
| RAW_LOG_RANGE@log | inv.raw | LOG (ADF p 1.47e-12, KPSS p 0.01, skew 3.09, kurt 20.1) | RETAINED  [5] | - | -6.82 / 0.677 | 4.41 / 3.27 / 3.85 |
| RAW_CLOSE_LOG_RETURN | inv.raw | NONE (ADF p 0, KPSS p 0.1, skew 0.0648, kurt 14.9) | RETAINED  [5] | - | -1.28e-06 / 0.00106 | 16.3 / 8.89 / 9.85 |
| RAW_ELAPSED_HOURS | inv.raw | NONE (ADF p 0, KPSS p 0.1, skew 10.9, kurt 118) | RETAINED  [5] | - | 1.4 / 4.41 | 16.9 / 18.3 / 18.3 |
| ATR | inv.volatility | NONE (ADF p 3.98e-13, KPSS p 0.01, skew 1.58, kurt 4.27) | RETAINED  [1] | - | 0.00153 / 0.000577 | 6.62 / 1.52 / 2.9 |
| RSI | inv.momentum | NONE (ADF p 2.17e-30, KPSS p 0.0258, skew -0.0254, kurt -0.297) | RETAINED  [4] | - | 49.7 / 12.6 | 3.28 / 3.33 / 3.02 |
| MACD_Histogram | inv.trend | NONE (ADF p 0, KPSS p 0.1, skew -0.0597, kurt 2.52) | RETAINED  [1] | - | 2.96e-07 / 0.000371 | 6.22 / 4.8 / 5.46 |
| ADX | inv.trend | NONE (ADF p 4.5e-30, KPSS p 0.1, skew 0.879, kurt 0.418) | RETAINED  [1] | - | 26.5 / 10.9 | 4.28 / 3.52 / 3.23 |
| DI_minus | inv.trend | NONE (ADF p 2.32e-30, KPSS p 0.0529, skew 0.518, kurt 0.253) | RETAINED  [1] | - | 20.7 / 7.36 | 5.15 / 3.48 / 5.39 |
| Stochastic_D | inv.momentum | NONE (ADF p 0, KPSS p 0.0322, skew 0.0269, kurt -1.27) | RETAINED  [4] | - | 49.7 / 26.7 | 1.84 / 1.82 / 1.84 |
| WilliamsR | inv.momentum | NONE (ADF p 0, KPSS p 0.0348, skew 0.0248, kurt -1.3) | RETAINED  [4] | - | -50.3 / 29.8 | 1.69 / 1.69 / 1.69 |
| rolling_std_24 | inv.volatility | NONE (ADF p 3.22e-30, KPSS p 0.01, skew 1.79, kurt 5.04) | RETAINED  [1] | - | 0.00188 / 0.00117 | 6.83 / 5.34 / 6.16 |
| hod_sin | calendar | NONE (ADF p 0, KPSS p 0.1, skew -0.000616, kurt -1.5) | RETAINED  [2] | base encoding | -0.000107 / 0.707 | 1.41 / 1.41 / 1.41 |
| hod_cos | calendar | NONE (ADF p 0, KPSS p 0.1, skew 0.0018, kurt -1.5) | RETAINED  [2] | base encoding | -0.000906 / 0.707 | 1.42 / 1.42 / 1.42 |
| dow_cos | calendar | NONE (ADF p 0, KPSS p 0.1, skew 0.159, kurt -1.65) | RETAINED  [2] | - | -0.0596 / 0.776 | 1.36 / 1.36 / 1.36 |

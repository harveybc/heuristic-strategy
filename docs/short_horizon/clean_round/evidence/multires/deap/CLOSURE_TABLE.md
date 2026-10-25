# Multi-resolution thresholds: closure table (hindsight label demonstrations; NOT a strategy result)

| period | metric | model (chosen thresholds) | naive, same rows | skill | literature value + source | comparability |
| --- | --- | --- | --- | --- | --- | --- |
| DECISION_2020_2022 | Sharpe daily sqrt365 / max DD % (hindsight label demonstration, replay contract) | Sharpe 16.70 / DD 1.17 % / 4.6 orders/wk / valid True | default_thresholds: Sharpe 11.54 / DD 3.87 % / 4.9 orders/wk / valid True | delta Sharpe +5.17; delta DD -2.70 pp | NOT_COMPARABLE: hindsight demonstration under a private replay contract; no published value | same bars, same contract (1 entry/day, 1:50, >= 2 orders/wk validity), same costs; both hindsight |
| DECISION_2020_2022 | Sharpe daily sqrt365 / max DD % (hindsight label demonstration, replay contract) | Sharpe 16.70 / DD 1.17 % / 4.6 orders/wk / valid True | registered_optimum_demonstration: Sharpe 13.23 / DD 1.29 % / 4.9 orders/wk / valid True | delta Sharpe +3.47; delta DD -0.13 pp | NOT_COMPARABLE: hindsight demonstration under a private replay contract; no published value | same bars, same contract (1 entry/day, 1:50, >= 2 orders/wk validity), same costs; both hindsight |
| VALIDATION_2023 | Sharpe daily sqrt365 / max DD % (hindsight label demonstration, replay contract) | Sharpe 17.82 / DD 1.35 % / 4.4 orders/wk / valid True | default_thresholds: Sharpe 12.25 / DD 1.73 % / 4.9 orders/wk / valid True | delta Sharpe +5.57; delta DD -0.38 pp | NOT_COMPARABLE: hindsight demonstration under a private replay contract; no published value | same bars, same contract (1 entry/day, 1:50, >= 2 orders/wk validity), same costs; both hindsight |
| VALIDATION_2023 | Sharpe daily sqrt365 / max DD % (hindsight label demonstration, replay contract) | Sharpe 17.82 / DD 1.35 % / 4.4 orders/wk / valid True | registered_optimum_demonstration: Sharpe 13.40 / DD 1.07 % / 4.8 orders/wk / valid True | delta Sharpe +4.43; delta DD +0.29 pp | NOT_COMPARABLE: hindsight demonstration under a private replay contract; no published value | same bars, same contract (1 entry/day, 1:50, >= 2 orders/wk validity), same costs; both hindsight |
| OBSERVED_2024 | Sharpe daily sqrt365 / max DD % (hindsight label demonstration, replay contract) | Sharpe 16.52 / DD 1.34 % / 4.1 orders/wk / valid True | default_thresholds: NOT_READ | NOT_COMPARABLE: comparator not read (single-read rule for 2024) | NOT_COMPARABLE: hindsight demonstration, no published comparator | same rows available only for the chosen point |
| OBSERVED_2024 | Sharpe daily sqrt365 / max DD % (hindsight label demonstration, replay contract) | Sharpe 16.52 / DD 1.34 % / 4.1 orders/wk / valid True | registered_optimum_demonstration: NOT_READ | NOT_COMPARABLE: comparator not read (single-read rule for 2024) | NOT_COMPARABLE: hindsight demonstration, no published comparator | same rows available only for the chosen point |

Chosen thresholds: `{"f_close_take": 0.6031, "min_votes": 1, "theta_close": 0.0847, "theta_f": 0.5905, "theta_mae": 0.2339, "theta_vol": 0.556}`
Rule: best VALIDATION_2023 Sharpe among union-front points valid on 2020-2022 and 2023 (63 points; 63 valid on both). Bound hits: {"theta_f": {"lower": 0, "upper": 0, "n": 63}, "theta_mae": {"lower": 7, "upper": 0, "n": 63}, "theta_vol": {"lower": 0, "upper": 0, "n": 63}, "theta_close": {"lower": 0, "upper": 0, "n": 63}, "f_close_take": {"lower": 5, "upper": 0, "n": 63}}

Convergence:

| seed | generations | stop | HV gen 0 | HV final | delta last 20 | evaluations | simulations | seconds |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 80 | CONVERGED | 2617.77 | 2665.24 | 0.045 % | 3888 | 3709 | 865 |
| 2 | 80 | CONVERGED | 2590.01 | 2667.89 | 0.064 % | 3888 | 3720 | 868 |
| 3 | 80 | CONVERGED | 2580.65 | 2662.41 | 0.132 % | 3888 | 3715 | 865 |

HINDSIGHT LIMITATION: every figure is a replay of hindsight labels (pivots known after the fact); it measures the quality of the TRAINING TARGET under the replay contract, never a tradable strategy or a profitability claim; the only real test is the student's generalization on the inner block in the clean round.

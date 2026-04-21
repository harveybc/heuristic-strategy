# TASK II-6.4: Path B Held-Out Evaluation

**Date:** 2026-04-20  
**Status:** COMPLETE — FAIL

## HO_B3 — TFT Regression Yearly (6 windows)

| Window | Test SR | Trades | MaxDD | CR |
|--------|---------|--------|-------|----|
| W1 (2020-H2) | +0.098 | 1 | 16.9% | 1156.58 |
| W2 (2021-H2) | +0.005 | 58 | 29.2% | 1.61 |
| W3 (2022-H2) | -0.017 | 1 | 38.2% | -204.73 |
| W4 (2023-H2) | -0.055 | 5 | 57.7% | -48.46 |
| W5 (2024-H2) | -0.007 | 54 | 30.1% | 0.22 |
| W6 (2025-H2) | -0.060 | 83 | 67.1% | -2.24 |

- **Aggregate HO SR:** -0.006 (IS: +0.024)
- **Consistency:** 33.3% (2/6) — needs ≥60%
- **Total trades:** 202
- **Kill criteria:** FAIL K-1 (agg SR≤0), FAIL K-3 (median CR), FAIL K-5 (consistency)
- **DSR p-value:** 0.000
- **Verdict:** HELD_OUT_FAILED

## Analysis

B3 TFT was already the weakest IS winner (SR=+0.024). The HO results confirm it has no predictive power:
- Extremely variable trade counts (1 to 83) indicate unstable signal
- Max drawdowns of 38–67% in later windows show severe degradation
- Only W1 produced meaningfully positive SR; all others are near zero or negative
- The IS edge was likely an artifact of look-ahead in feature engineering or overfit to the training distribution

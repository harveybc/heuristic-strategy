# TASK II-6.3: Path A Held-Out Evaluation

**Date:** 2026-04-20  
**Status:** COMPLETE — ALL FAIL

## Experiments

### HO_A1 — BTC Momentum Yearly (6 windows, pop=30, gen=20)
| Window | Test SR | Trades | MaxDD | CR |
|--------|---------|--------|-------|----|
| W1 (2020-H2) | +0.078 | 75 | 8.8% | 1.75 |
| W2 (2021-H2) | -0.036 | 76 | 15.1% | -0.73 |
| W3 (2022-H2) | -0.050 | 53 | 15.3% | -1.00 |
| W4 (2023-H2) | -0.122 | 47 | 12.3% | -2.28 |
| W5 (2024-H2) | +0.003 | 69 | 11.5% | 0.05 |
| W6 (2025-H2) | +0.046 | 66 | 6.0% | 1.06 |

- **Aggregate HO SR:** -0.013 (IS: +0.155)
- **Consistency:** 50% (3/6) — needs ≥60%
- **Kill criteria:** FAIL K-1 (agg SR≤0), FAIL K-3 (median CR<2.0), FAIL K-5 (consistency<60%)
- **DSR p-value:** 0.000
- **Verdict:** HELD_OUT_FAILED

### HO_A2 — BTC Momentum Monthly (72 windows, pop=30, gen=20)
- **Aggregate HO SR:** +0.083 (IS: +0.156)
- **Consistency:** 54.2% (39/72) — needs ≥60%
- **Total trades:** 699
- **Kill criteria:** PASS K-1, PASS K-2, FAIL K-3, FAIL K-5
- **DSR p-value:** 0.000
- **Verdict:** HELD_OUT_FAILED
- **Note:** Positive aggregate SR but insufficient consistency and cost ratio. Sharpe skewness=+1.92 (fat-tailed wins).

### HO_A6 — BTC Momentum HPO Yearly (6 windows, pop=80, gen=50)
| Window | Test SR | Trades | MaxDD | CR |
|--------|---------|--------|-------|----|
| W1 (2020-H2) | +0.036 | 35 | 11.4% | 0.79 |
| W2 (2021-H2) | +0.026 | 77 | 8.3% | 0.58 |
| W3 (2022-H2) | -0.015 | 77 | 10.6% | -0.32 |
| W4 (2023-H2) | -0.048 | 61 | 12.6% | -0.98 |
| W5 (2024-H2) | +0.197 | 95 | 6.5% | 5.17 |
| W6 (2025-H2) | +0.055 | 67 | 7.3% | 1.27 |

- **Aggregate HO SR:** +0.042 (IS: +0.142)
- **Consistency:** 66.7% (4/6) — PASS K-5
- **Kill criteria:** PASS K-1, PASS K-2, FAIL K-3 (median CR=0.69), PASS K-5
- **DSR p-value:** 0.000
- **Verdict:** HELD_OUT_FAILED
- **Note:** A6 is the closest to passing — positive SR, good consistency — but median cost ratio well below 2.0 threshold and DSR is zero.

## Summary

All three Path A strategies fail HO validation. The IS Sharpes (+0.14 to +0.16) degrade dramatically out-of-sample. A6 shows the most robustness (4/6 positive windows, aggregate SR>0) but cannot overcome the multiple-testing penalty (N=17, E[max(SR)]=1.92) in the DSR framework.

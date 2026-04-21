# TASK II-6.5: Deflated Sharpe Ratio — Final Verdict

**Date:** 2026-04-20  
**Status:** COMPLETE — Part II-Redux closes as NULL

## DSR Framework

- **N strategies tested:** 17 (full multiple-testing universe)
- **E[max(SR)] for N=17:** 1.9192 (Bailey & López de Prado, 2014)
- **Kill criteria:** K-1 (agg SR>0), K-2 (worst 2yr rolling>-0.9), K-3 (median CR≥2.0), K-5 (consistency≥60%), K-6 (MAE ratio for Path B)

## Final Verdict Matrix

| Strategy | IS SR | HO SR | Consistency | Kill Pass? | DSR p-value | Verdict |
|----------|-------|-------|-------------|------------|-------------|---------|
| A1 Yearly | +0.155 | -0.013 | 50% (3/6) | No (K1,K3,K5) | 0.000 | HELD_OUT_FAILED |
| A2 Monthly | +0.156 | +0.083 | 54% (39/72) | No (K3,K5) | 0.000 | HELD_OUT_FAILED |
| A6 HPO Yearly | +0.142 | +0.042 | 67% (4/6) | No (K3) | 0.000 | HELD_OUT_FAILED |
| B3 TFT Yearly | +0.024 | -0.006 | 33% (2/6) | No (K1,K3,K5) | 0.000 | HELD_OUT_FAILED |

## Key Findings

1. **All DSR p-values = 0.000** — No strategy survives the multiple-testing correction. With 17 strategies tried, E[max(SR)]=1.92 sets a very high bar. The best HO SR of +0.083 (A2) is 54× smaller than required.

2. **IS→HO degradation is universal:**
   - A1: +0.155 → -0.013 (109% degradation)
   - A2: +0.156 → +0.083 (47% degradation)
   - A6: +0.142 → +0.042 (70% degradation)
   - B3: +0.024 → -0.006 (125% degradation)

3. **A6 is the most robust** — only strategy passing K-1, K-2, and K-5. Failed solely on K-3 (median cost ratio 0.68 vs threshold 2.0). Under a relaxed framework without K-3, A6 would be borderline but still fails DSR.

4. **A2 monthly granularity** — 72-window evaluation confirms the signal is real but weak (mean SR=+0.083) with high variance (skew=+1.92, kurtosis=13.49). Consistency at 54% falls short of the 60% threshold.

## Part III Recommendation

**Neither pathway is validated. Part II-Redux closes as null.**

No strategy should advance to Part III (live paper trading). The BTC 4H momentum signal exists in-sample but does not survive held-out validation under rigorous statistical testing. Further work would require:
- New feature engineering approaches (not explored in this study)
- Alternative asset classes with stronger causal structure
- Fundamentally different model architectures

## Artifacts

- DSR JSON: `deliverables/dsr_verdict_results.json`
- A1 results: `logs/HO_A1_btc_momentum_yearly/`
- A2 results: `logs/HO_A2_btc_momentum_monthly/`
- A6 results: `logs/HO_A6_btc_momentum_hpo/`
- B3 results: `logs/HO_B3_tft_yearly/`

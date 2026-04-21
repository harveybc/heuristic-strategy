# TASK II-7-fix.3: Cross-Asset Independence Test

**Generated:** 2026-04-21 08:57 UTC
**IS Period:** 2019-01-01 to 2019-12-31

## 1. BTC-ETH Return Correlation

| Metric | Value |
|--------|-------|
| Pearson correlation (1h returns) | **0.7764** |
| N aligned 1h bars | 8708 |

> **High correlation (0.776 > 0.70):** BTC and ETH share substantial common factor.
> ETH may not be independent of BTC market beta.

## 2. OLS Regression: ETH ~ BTC

| Parameter | Value |
|-----------|-------|
| R² | 0.6028 |
| BTC coefficient (β) | 0.9229 |
| BTC p-value | 0.00e+00 |
| Intercept | -0.000059 |

**Interpretation:** 60.3% of ETH hourly return variance is explained by BTC returns.

## 3. PCMCI+ on Original ETH IS 2019

- Samples used: 8571
- Lagged links found (p<0.05): 1
- macd_hist τ=1: **PRESENT**  MCI=0.1779  p=0.0000

| Feature | τ | MCI | p-value |
|---------|---|-----|---------|
| macd_hist | 1 | 0.1779 | 0.0000 |

## 4. PCMCI+ on Residual ETH (After BTC Factor Removal)

- Samples used: 8570
- Lagged links found (p<0.05): 1
- macd_hist τ=1: **NOT PRESENT** (p≥0.05)

| Feature | τ | MCI | p-value |
|---------|---|-----|---------|
| roc_12 | 1 | 0.1185 | 0.0000 |

## 5. Comparison: Original vs Residual ETH

| | Original ETH | Residual ETH |
|-|-------------|-------------|
| macd_hist τ=1 present | True | False |
| macd_hist τ=1 MCI | 0.1779 | — |
| Total lagged links | 1 | 1 |

## 6. Conclusion

**COMMON_FACTOR_ARTIFACT**

macd_hist τ=1 disappears in residual ETH. ETH α finding was BTC factor artifact, not independent ETH signal.

## 7. Implications for Part III

ETH 1h technical α configuration should be reviewed: the causal finding partially or fully reflects BTC market beta rather than independent ETH structure. User must decide whether ETH 1h remains a Part III candidate.

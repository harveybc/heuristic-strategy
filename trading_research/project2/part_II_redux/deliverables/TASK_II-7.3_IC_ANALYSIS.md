# TASK II-7.3 — IC Analysis

**Stage**: II-7 (RL Configuration Reconnaissance)
**Date**: 2026-04-21
**Status**: COMPLETE

---

## Objective

Measure the predictive strength of each feature via Spearman rank Information Coefficient (IC) and IC Information Ratio (ICIR) across multiple forecast horizons. Pass threshold: |ICIR| ≥ 0.3 on a rolling window of 720 bars.

## Alpha Configs Tested

Configs carried forward from II-7.2:
- Run 3: BTC 1h, technical features
- Run 11: ETH 1h, technical features

---

## Results

### 3: btc_1h_technical

Asset: BTC | Timeframe: 1h | Total bars: 61,285

**Overall verdict**: PROCEED_TO_RL

| Horizon (h) | Best Feature | Best ICIR | Pass (|ICIR|≥0.3) |
|-------------|-------------|-----------|-------------------|
| 1 | momentum_5 | -6.1529 | YES |
| 6 | returns | -2.7694 | YES |
| 12 | returns | -2.1957 | YES |
| 24 | momentum_20 | -2.7399 | YES |

#### Feature IC Details (all horizons)

| Feature | h=1 ICIR | h=6 ICIR | h=12 ICIR | h=24 ICIR |
|---------|------|------|------|------|
| returns | -2.7934 ✓ | -2.7694 ✓ | -2.1957 ✓ | -1.5492 ✓ |
| log_returns | -2.7934 ✓ | -2.7694 ✓ | -2.1957 ✓ | -1.5492 ✓ |
| rsi | -3.3698 ✓ | -0.6734 ✓ | -0.0711 | -1.8498 ✓ |
| macd_hist | -3.7782 ✓ | -0.1389 | 0.5779 ✓ | -1.1658 ✓ |
| bb_pos | -5.2700 ✓ | -2.2675 ✓ | -0.7311 ✓ | -2.4051 ✓ |
| volume_ratio | 0.5220 ✓ | 0.4106 ✓ | 0.1461 | 0.3467 ✓ |
| ema_cross | -2.0869 ✓ | -1.6611 ✓ | -1.1034 ✓ | -2.0884 ✓ |
| atr_norm | 1.8345 ✓ | 1.3177 ✓ | 1.0711 ✓ | 1.2983 ✓ |
| obv_delta | -2.9431 ✓ | -2.1797 ✓ | -0.7703 ✓ | -0.9282 ✓ |
| momentum_5 | -6.1529 ✓ | -1.9836 ✓ | -0.7414 ✓ | -2.1724 ✓ |
| momentum_20 | -2.3172 ✓ | -1.0543 ✓ | -1.1831 ✓ | -2.7399 ✓ |
| volatility_20 | 1.6285 ✓ | 1.3774 ✓ | 1.1350 ✓ | 1.1632 ✓ |

### 11: eth_1h_technical

Asset: ETH | Timeframe: 1h | Total bars: 61,285

**Overall verdict**: PROCEED_TO_RL

| Horizon (h) | Best Feature | Best ICIR | Pass (|ICIR|≥0.3) |
|-------------|-------------|-----------|-------------------|
| 1 | momentum_5 | -3.9083 | YES |
| 6 | returns | -2.2542 | YES |
| 12 | volatility_20 | 1.3515 | YES |
| 24 | bb_pos | -1.7846 | YES |

#### Feature IC Details (all horizons)

| Feature | h=1 ICIR | h=6 ICIR | h=12 ICIR | h=24 ICIR |
|---------|------|------|------|------|
| returns | -2.3381 ✓ | -2.2542 ✓ | -1.2205 ✓ | -1.3000 ✓ |
| log_returns | -2.3381 ✓ | -2.2542 ✓ | -1.2205 ✓ | -1.3000 ✓ |
| rsi | -2.8566 ✓ | -0.3432 ✓ | -0.0498 | -1.4486 ✓ |
| macd_hist | -3.6583 ✓ | 0.2389 | 0.3841 ✓ | -1.2199 ✓ |
| bb_pos | -3.4796 ✓ | -1.0988 ✓ | -0.3327 ✓ | -1.7846 ✓ |
| volume_ratio | 0.4849 ✓ | 0.4919 ✓ | 0.8576 ✓ | 1.4644 ✓ |
| ema_cross | -1.9645 ✓ | -1.1025 ✓ | -0.8107 ✓ | -1.1356 ✓ |
| atr_norm | 1.5156 ✓ | 1.7338 ✓ | 1.2323 ✓ | 1.0634 ✓ |
| obv_delta | -2.5068 ✓ | -1.8370 ✓ | -0.7418 ✓ | -1.3463 ✓ |
| momentum_5 | -3.9083 ✓ | -1.2707 ✓ | -0.2073 | -1.5453 ✓ |
| momentum_20 | -2.2580 ✓ | -0.7667 ✓ | -0.9305 ✓ | -1.5805 ✓ |
| volatility_20 | 1.6969 ✓ | 1.8993 ✓ | 1.3515 ✓ | 1.0874 ✓ |

---

## Conclusion

Configs proceeding to RL pilots (II-7.4/5): **btc_1h_technical, eth_1h_technical**

Both BTC 1h and ETH 1h show measurable IC structure on `macd_hist` (confirmed by II-7.2 causal analysis). The IC analysis provides the predictive horizon and feature ranking needed to configure RL reward shaping in II-7.4.

## Deliverables

- `deliverables/ic_results_II7.json` — Full IC statistics
- `deliverables/TASK_II-7.3_IC_ANALYSIS.md` — This document

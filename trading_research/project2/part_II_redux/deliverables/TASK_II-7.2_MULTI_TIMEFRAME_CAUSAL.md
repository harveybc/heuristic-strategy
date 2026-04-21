# TASK II-7.2 — Multi-Timeframe Causal Analysis

**Stage**: II-7 (RL Configuration Reconnaissance)
**Date**: April 2026
**Status**: COMPLETE

---

## Objective

Determine which (asset, timeframe, feature-set) configurations exhibit statistically significant Granger-causal structure using Tigramite PCMCI+. Only configurations that survive as α-class (strong causal signal) are carried forward to IC analysis (II-7.3) and RL pilot training (II-7.4/5).

---

## Method

**Framework**: Tigramite 5.x, PCMCI+ algorithm
**Independence test**: ParCorr (partial correlation)
**Parameters**:
- τ_max = 10 lags
- pc_alpha = 0.01 (skeleton discovery)
- alpha_level = 0.05 (MCI significance)
- IS period: 2019-01-01 → 2019-12-31 (1-year in-sample)
- max_samples = 5,000 (random subsample when IS data > 5000 rows)

**Classification criteria**:
| Class | Condition |
|-------|-----------|
| α (alpha) | ≥1 lagged link with \|MCI\| > 0.10 AND p < 0.01 |
| β (beta) | ≥1 lagged link with \|MCI\| > 0.05 AND p < 0.05 (not α) |
| γ (gamma) | No significant lagged links |

**Features computed inline** (12 features per bar):
`returns`, `log_returns`, `rsi`, `macd_hist`, `bb_pos`, `volume_ratio`, `ema_cross`, `atr_norm`, `obv_delta`, `momentum_5`, `momentum_20`, `volatility_20`

**Auxiliary feature sets** tested:
- `funding`: funding_rate merged via forward-fill
- `onchain`: AdrActCnt, TxCnt, HashRate merged daily via forward-fill
- `blockchain_com`: mempool_size, confirmed_tx_per_block, hash_rate (daily)

**Target variable**: `forward_return_6` (6-bar ahead log return)

Script: `scripts/stage_ii7_multitimeframe_causal.py`

---

## Run Matrix

| Run ID | Label | Asset | Timeframe | Feature Set |
|--------|-------|-------|-----------|-------------|
| 1 | btc_5m_technical | BTC | 5m | technical |
| 2 | btc_15m_technical | BTC | 15m | technical |
| 3 | btc_1h_technical | BTC | 1h | technical |
| 4 | btc_4h_technical | BTC | 4h | technical |
| 5 | btc_4h_funding | BTC | 4h | technical + funding |
| 6 | btc_4h_onchain | BTC | 4h | technical + onchain |
| 7 | btc_4h_all | BTC | 4h | technical + funding + onchain + blockchain_com |
| 8 | btc_1h_funding | BTC | 1h | technical + funding |
| 9 | eth_5m_technical | ETH | 5m | technical |
| 10 | eth_15m_technical | ETH | 15m | technical |
| 11 | eth_1h_technical | ETH | 1h | technical |
| 12 | eth_1h_funding | ETH | 1h | technical + funding |
| 13 | eth_4h_technical | ETH | 4h | technical |
| 14 | eth_4h_funding | ETH | 4h | technical + funding |

---

## Results

### Alpha Runs (carry forward)

| Run | Label | Samples | Link | τ | MCI | p-value |
|-----|-------|---------|------|---|-----|---------|
| 3 | btc_1h_technical | 5,000 | macd_hist | 1 | 0.1911 | 4.41e-42 |
| 11 | eth_1h_technical | 5,000 | macd_hist | 1 | 0.1779 | 1.34e-36 |

### Beta Runs

None.

### Gamma Runs (eliminated)

Runs: 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14

---

## Interpretation

1. **MACD histogram at τ=1** is the sole surviving causal feature in both alpha runs. This implies that 1-bar-lagged MACD momentum is the statistically strongest causal signal for 6-bar-ahead returns in both BTC and ETH at the 1h timeframe.

2. **5m and 15m configurations (runs 1, 2, 9, 10)** returned gamma. At finer timeframes, the IS period (2019, ~13K 5m bars) produces higher noise-to-signal ratio and PCMCI+ alpha controlled at 0.05 finds no significant structure. This does not rule out exploitability but causal signal is not detectable.

3. **4h configurations with auxiliary features (runs 4–8, 13–14)** returned gamma. Run 4 IS effective sample was 665 bars (BTC 4h in 2019 with forward-look dropna), borderline for PCMCI+. The auxiliary features (funding, onchain, blockchain) add dimensionality without sufficient IS data to achieve significance. Re-running with IS extended to 2019–2021 is recommended if 4h configs are needed in Phase 2.

4. **BTC 1h > ETH 1h** by MCI magnitude (0.191 vs 0.178), both highly significant (p < 1e-35). Both are carried forward to II-7.3.

---

## Deliverables

- `deliverables/causal_results_II7.json` — Full PCMCI+ output for all 14 runs
- `deliverables/TASK_II-7.2_MULTI_TIMEFRAME_CAUSAL.md` — This document

---

## Conclusion

Two configurations survive causal screening:
- **BTC 1h, technical features, macd_hist τ=1** (α, MCI=0.191)
- **ETH 1h, technical features, macd_hist τ=1** (α, MCI=0.178)

These two are carried forward to II-7.3 (IC analysis) and II-7.4/5 (RL infrastructure and pilots).

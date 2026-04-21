# TASK II-7.1 — Data Acquisition for RL Reconnaissance

**Stage**: II-7 (RL Configuration Reconnaissance)
**Date**: April 2026
**Status**: COMPLETE — Validation Gate: PASS

---

## Objective

Assemble a multi-asset, multi-timeframe dataset covering BTC and ETH from 2019-01-01 to 2025-12-31 suitable for causal analysis (II-7.2), IC analysis (II-7.3), and RL pilot training (II-7.4/5). Four data streams were acquired and validated against a 6-test statistical battery.

---

## Data Sources

### A. Binance OHLCV — `data/raw/binance/`
Script: `scripts/fetch_binance_extended.py`

| File | Asset | Timeframe | Rows | Range |
|------|-------|-----------|------|-------|
| `btcusdt_5m_2019_2025.parquet` | BTC | 5m | 735,311 | 2019-01-01 → 2025-12-31 |
| `btcusdt_15m_2019_2025.parquet` | BTC | 15m | 245,107 | 2019-01-01 → 2025-12-31 |
| `btcusdt_1h_2019_2025.parquet` | BTC | 1h | 61,285 | 2019-01-01 → 2025-12-31 |
| `ethusdt_5m_2019_2025.parquet` | ETH | 5m | 735,311 | 2019-01-01 → 2025-12-31 |
| `ethusdt_15m_2019_2025.parquet` | ETH | 15m | 245,107 | 2019-01-01 → 2025-12-31 |
| `ethusdt_1h_2019_2025.parquet` | ETH | 1h | 61,285 | 2019-01-01 → 2025-12-31 |
| `ethusdt_4h_2019_2025.parquet` | ETH | 4h | 15,332 | 2019-01-01 → 2025-12-31 |

Note: BTC 4h sourced from legacy file `btcusd_4h_2017_2025.csv` (Part II historical data).

All 7 parquet files pass the 6-test validation battery: bar_count_realistic, monotonic_no_dupes, fat_tails, volatility_clustering, tiny_nonzero_autocorr, no_gbm_fingerprint.

### B. Binance Perpetual Futures Funding Rates — `data/raw/binance/`
Script: `scripts/fetch_binance_funding.py`

| File | Rows | Range |
|------|------|-------|
| `funding_btcusdt_2019_2025.csv` | 6,911 | 2019-09-10 → 2025-12-29 |
| `funding_ethusdt_2019_2025.csv` | 6,677 | 2019-11-27 → 2025-12-29 |

Note: Funding rates begin at perpetual futures launch dates (BTC Sep 2019, ETH Nov 2019). Pre-launch periods use forward-fill = 0 when merged.

### C. CoinMetrics Community On-Chain — `data/raw/coinmetrics/`
Script: `scripts/fetch_coinmetrics_community.py`

Available metrics on community tier (probed via timeseries/asset-metrics endpoint):
- `AdrActCnt` — Active address count (daily)
- `TxCnt` — Transaction count (daily)
- `HashRate` — Network hash rate (daily)

| File | Asset | Rows | Range |
|------|-------|------|-------|
| `btc_daily_metrics_2019_2025.csv` | BTC | 2,557 | 2019-01-01 → 2025-12-31 |
| `eth_daily_metrics_2019_2025.csv` | ETH | 2,557 | 2019-01-01 → 2025-12-31 |

Note: CoinMetrics v4 community `catalog/asset-metrics` endpoint does not accept an `assets` filter parameter. The final implementation probes each metric individually via the timeseries endpoint.

### D. Blockchain.com Supplementary BTC Metrics — `data/raw/blockchain_com/`
Script: `scripts/fetch_blockchain_com.py`

| Metric | Key | Rows |
|--------|-----|------|
| Mempool size | `mempool-size` | 2,557 |
| Confirmed tx per block | `n-transactions-per-block` | 2,554 |
| Hash rate | `hash-rate` | 2,554 |

Output file: `btc_metrics_2019_2025.csv` (2,557 rows, missing_ratio=0.001)

Note: Blockchain.com API returns mixed-frequency timestamps. Fixed by applying `.dt.floor("D")` + `.groupby().mean()` before merge.

---

## Validation Results

Validation script: `scripts/validate_phase1_data.py`
Report: `data/validation/II-7_data_validation.md`

**Overall gate: PASS**

All OHLCV parquet files pass all 6 tests. Funding, CoinMetrics, and Blockchain.com pass non-empty and missing-ratio checks.

---

## Blockers Encountered and Resolutions

| Blocker | Resolution |
|---------|-----------|
| `pyarrow` not installed in tensorflow conda env | `pip install pyarrow fastparquet` |
| CoinMetrics `catalog/asset-metrics?assets=btc` → HTTP 400 | Community API does not support `assets` param on catalog endpoint; switched to per-metric timeseries probing |
| Blockchain.com merge missing_ratio=0.66 | API returns intraday timestamps; applied `.dt.floor("D")` + `.groupby().mean()` normalization |

---

## Conclusion

All four data streams successfully acquired and validated. Dataset is ready for:
- II-7.2: Multi-timeframe causal analysis (PCMCI+)
- II-7.3: IC analysis on surviving alpha configurations
- II-7.4/7.5: RL environment construction and pilot training

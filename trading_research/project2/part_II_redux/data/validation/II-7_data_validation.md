# Stage II-7.1 Data Validation

Overall gate: PASS

## OHLCV (6-test battery)

### btcusdt_15m_2019_2025.parquet
- Rows: 245107
- Range: 2019-01-01 00:00:00+00:00 -> 2025-12-31 00:00:00+00:00
- bar_count_realistic: PASS
- monotonic_no_dupes: PASS
- fat_tails: PASS
- volatility_clustering: PASS
- tiny_nonzero_autocorr: PASS
- no_gbm_fingerprint: PASS
- Metrics: expected_rows=245376.000000, kurtosis=119.180558, acf1_r2=0.271450, acf1_r=-0.009226, ljung_box_p=0.000000, jarque_bera_p=0.000000, runs_p=0.000000, gbm_rejects_3=3.000000

### btcusdt_1h_2019_2025.parquet
- Rows: 61285
- Range: 2019-01-01 00:00:00+00:00 -> 2025-12-31 00:00:00+00:00
- bar_count_realistic: PASS
- monotonic_no_dupes: PASS
- fat_tails: PASS
- volatility_clustering: PASS
- tiny_nonzero_autocorr: PASS
- no_gbm_fingerprint: PASS
- Metrics: expected_rows=61344.000000, kurtosis=52.356376, acf1_r2=0.159748, acf1_r=-0.017960, ljung_box_p=0.000000, jarque_bera_p=0.000000, runs_p=0.000000, gbm_rejects_3=3.000000

### btcusdt_5m_2019_2025.parquet
- Rows: 735311
- Range: 2019-01-01 00:00:00+00:00 -> 2025-12-31 00:00:00+00:00
- bar_count_realistic: PASS
- monotonic_no_dupes: PASS
- fat_tails: PASS
- volatility_clustering: PASS
- tiny_nonzero_autocorr: PASS
- no_gbm_fingerprint: PASS
- Metrics: expected_rows=736128.000000, kurtosis=175.636539, acf1_r2=0.254206, acf1_r=-0.031038, ljung_box_p=0.000000, jarque_bera_p=0.000000, runs_p=0.000000, gbm_rejects_3=3.000000

### ethusdt_15m_2019_2025.parquet
- Rows: 245107
- Range: 2019-01-01 00:00:00+00:00 -> 2025-12-31 00:00:00+00:00
- bar_count_realistic: PASS
- monotonic_no_dupes: PASS
- fat_tails: PASS
- volatility_clustering: PASS
- tiny_nonzero_autocorr: PASS
- no_gbm_fingerprint: PASS
- Metrics: expected_rows=245376.000000, kurtosis=71.911378, acf1_r2=0.264844, acf1_r=0.002463, ljung_box_p=0.000000, jarque_bera_p=0.000000, runs_p=0.000000, gbm_rejects_3=3.000000

### ethusdt_1h_2019_2025.parquet
- Rows: 61285
- Range: 2019-01-01 00:00:00+00:00 -> 2025-12-31 00:00:00+00:00
- bar_count_realistic: PASS
- monotonic_no_dupes: PASS
- fat_tails: PASS
- volatility_clustering: PASS
- tiny_nonzero_autocorr: PASS
- no_gbm_fingerprint: PASS
- Metrics: expected_rows=61344.000000, kurtosis=32.000794, acf1_r2=0.160301, acf1_r=-0.006573, ljung_box_p=0.000000, jarque_bera_p=0.000000, runs_p=0.000000, gbm_rejects_3=3.000000

### ethusdt_4h_2019_2025.parquet
- Rows: 15332
- Range: 2019-01-01 00:00:00+00:00 -> 2025-12-31 00:00:00+00:00
- bar_count_realistic: PASS
- monotonic_no_dupes: PASS
- fat_tails: PASS
- volatility_clustering: PASS
- tiny_nonzero_autocorr: PASS
- no_gbm_fingerprint: PASS
- Metrics: expected_rows=15336.000000, kurtosis=15.406105, acf1_r2=0.164690, acf1_r=-0.023004, ljung_box_p=0.000000, jarque_bera_p=0.000000, runs_p=0.000000, gbm_rejects_3=3.000000

### ethusdt_5m_2019_2025.parquet
- Rows: 735311
- Range: 2019-01-01 00:00:00+00:00 -> 2025-12-31 00:00:00+00:00
- bar_count_realistic: PASS
- monotonic_no_dupes: PASS
- fat_tails: PASS
- volatility_clustering: PASS
- tiny_nonzero_autocorr: PASS
- no_gbm_fingerprint: PASS
- Metrics: expected_rows=736128.000000, kurtosis=137.266479, acf1_r2=0.426544, acf1_r=-0.026455, ljung_box_p=0.000000, jarque_bera_p=0.000000, runs_p=0.000000, gbm_rejects_3=3.000000

## Funding / On-chain / Supplementary Checks

### Funding
- funding_btcusdt_2019_2025.csv: rows=6911 range=2019-09-10 08:00:00+00:00 -> 2025-12-29 16:00:00+00:00 missing=0.000 nonempty=PASS missing_check=PASS
- funding_ethusdt_2019_2025.csv: rows=6677 range=2019-11-27 08:00:00+00:00 -> 2025-12-29 16:00:00+00:00 missing=0.000 nonempty=PASS missing_check=PASS

### CoinMetrics
- btc_daily_metrics_2019_2025.csv: rows=2557 range=2019-01-01 00:00:00+00:00 -> 2025-12-31 00:00:00+00:00 missing=0.000 nonempty=PASS missing_check=PASS
- eth_daily_metrics_2019_2025.csv: rows=2557 range=2019-01-01 00:00:00+00:00 -> 2025-12-31 00:00:00+00:00 missing=0.157 nonempty=PASS missing_check=PASS

### Blockchain.com
- btc_metrics_2019_2025.csv: rows=2557 range=2019-01-01 00:00:00+00:00 -> 2025-12-31 00:00:00+00:00 missing=0.001 nonempty=PASS missing_check=PASS


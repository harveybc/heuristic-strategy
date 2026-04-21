# Stage II-5: Path B — Supervised ML Predictor Synthesis

**Date:** 2026-04-19
**Asset:** BTC/USD 4-hour
**Data:** 18,307 feature rows (2017-08-20 to 2025-04-17)
**Features:** 12 F-6 technical indicators + 6-bar forward log-return target
**Windows:** 2 yearly anchored expanding windows (IS 2017–2019, HO 2018/2019)
**Framework:** TensorFlow 2.19 (Omega), TensorFlow 2.21 (Dragon)
**Machines:** Omega (RTX 4070), Dragon (RTX 4090 GPU)

---

## 1. Experimental Design

### 1.1 Feature Set (F-6)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `adx` | Average Directional Index (trend strength) |
| 2 | `di_spread` | DI+ minus DI- (trend direction) |
| 3 | `atr_pct` | ATR as % of close (volatility) |
| 4 | `atr_ratio` | ATR(14)/ATR(50) (volatility regime) |
| 5 | `bb_width_pct` | Bollinger Band width % (volatility) |
| 6 | `bb_position` | Price position within Bollinger Bands |
| 7 | `rsi` | Relative Strength Index (momentum) |
| 8 | `roc_12` | Rate of Change, 12 bars (momentum) |
| 9 | `price_vs_ema50` | Price relative to EMA-50 |
| 10 | `ema_alignment` | EMA-12/EMA-26 alignment (trend) |
| 11 | `stoch_k` | Stochastic %K (momentum) |
| 12 | `macd_hist` | MACD histogram (momentum) |

**Target:** 6-bar forward log-return of Close price.

### 1.2 Protocol

- **Lookback:** 24 bars (96 hours = 4 days)
- **Epochs:** 200 (with early stopping, patience=20)
- **Batch size:** 64
- **Normalisation:** Z-score (fit on train only)
- **Embargo:** 6 bars between train and validation
- **Signal → Trade:** Position = sign(prediction) × 1-bar return, 10 bps cost per trade
- **Sequence construction:** Sliding window of size 24 over normalised features → (samples, 24, 12) 3D input

### 1.3 Models

| ID | Model | Architecture | Parameters |
|----|-------|-------------|------------|
| B1 | CNN | 1D Convolutional + Dense head | ~12K |
| B2 | LSTM | Multi-branch Bayesian LSTM (DenseFlipout) | ~65K |
| B3 | TFT | Temporal Fusion Transformer | ~302K |
| B4 | TCN | Temporal Convolutional Network (dilated causal) | ~302K |
| B5 | ANN | Feedforward Dense network | ~65K |
| B6 | Transformer | Multi-head attention encoder | ~130K |

---

## 2. Results

### 2.1 Per-Window Results

| Exp | Model | W1 Sharpe | W1 Trades | W1 MaxDD | W2 Sharpe | W2 Trades | W2 MaxDD |
|-----|-------|-----------|-----------|----------|-----------|-----------|----------|
| B1 | CNN | +0.004 | 40 | 31.2% | -0.029 | 1 | 58.6% |
| B2 | LSTM | -0.072 | 549 | 104.6% | -0.091 | 369 | 130.7% |
| B3 | **TFT** | **+0.017** | **37** | **30.8%** | **+0.032** | **58** | **20.1%** |
| B4 | TCN | +0.005 | 131 | 42.8% | -0.051 | 139 | 79.3% |
| B5 | ANN | -0.039 | 1 | 78.7% | -0.028 | 15 | 58.6% |
| B6 | Transformer | -0.079 | 506 | 118.3% | -0.123 | 526 | 172.8% |

### 2.2 Aggregate Results

| Exp | Model | Agg Sharpe | Mean±Std | Max DD | Trades | Consistency | Final Equity | K-3 | K-5 |
|-----|-------|-----------|----------|--------|--------|-------------|-------------|-----|-----|
| B1 | CNN | -0.013 | -0.013±0.016 | 58.2% | 41 | 50% | $6,188 | FAIL | FAIL |
| B2 | LSTM | -0.082 | -0.082±0.009 | 229.6% | 918 | 0% | -$14,250 | FAIL | FAIL |
| **B3** | **TFT** | **+0.024** | **+0.024±0.008** | **30.8%** | **95** | **100%** | **$17,217** | **PASS** | **PASS** |
| B4 | TCN | -0.024 | -0.023±0.028 | 77.2% | 270 | 50% | $3,015 | FAIL | FAIL |
| B5 | ANN | -0.034 | -0.034±0.005 | 107.2% | 16 | 0% | $13 | FAIL | FAIL |
| B6 | Transformer | -0.101 | -0.101±0.022 | 295.3% | 1,032 | 0% | -$19,964 | FAIL | FAIL |

### 2.3 Kill Criteria Assessment

| Criterion | Threshold | B3 (TFT) | All Others |
|-----------|-----------|-----------|------------|
| K-1 (Held-out SR > 0) | > 0 | PASS (+0.024) | FAIL |
| K-2 (Worst 2yr SR > -0.9) | > -0.9 | PASS | Mixed |
| K-3 (Cost ratio ≥ 2.0) | ≥ 2.0 | PASS (4.3, 5.2) | FAIL |
| K-5 (Window consistency) | ≥ 60% | PASS (100%) | FAIL (0–50%) |

---

## 3. Analysis

### 3.1 Best Model: TFT (Temporal Fusion Transformer)

The TFT is the **only model to pass all kill criteria**:
- Consistent positive Sharpe across both windows (+0.017, +0.032)
- Conservative trading frequency (37–58 trades per window)
- Controlled drawdowns (20–31%)
- Strong cost ratio (4.3×–5.2×), confirming gross profits far exceed costs
- 72% return over IS period ($10K → $17.2K)

The TFT's attention mechanism and variable selection networks appear well-suited for the sparse, regime-dependent signal structure of BTC/USD 4h data.

### 3.2 Failure Modes

| Model | Failure Mode | Detail |
|-------|-------------|--------|
| CNN | Inconsistency | W1 slight profit, W2 near-zero trades — model collapsed to flat |
| LSTM | Overtrading + noise fitting | 549–369 trades/window, negative returns, DD > 100% |
| TCN | Directional collapse | W1 marginally positive, W2 strongly negative |
| ANN | Under-trading | 1–15 trades, near-zero predictions → near-zero action |
| Transformer | Catastrophic overtrading | 500+ trades/window, DD up to 173%, worst performer |

### 3.3 Key Observations

1. **Attention > Recurrence for BTC:** TFT outperformed LSTM, TCN, and vanilla Transformer. The variable selection and temporal attention gates focus on the most informative bars, filtering noise.

2. **Simpler ≠ better:** ANN and CNN underperformed despite lower parameter counts. The temporal structure in 4h bars requires sequence-aware architectures.

3. **Overtrading is the dominant failure mode:** LSTM and Transformer generated 500–1000 trades per 6-month window, amplifying noise and costs.

4. **Small data regime:** With only ~5,200 4h BTC bars available (2017-08 to 2019-12), model capacity must be constrained. TFT's attention naturally provides this regularisation.

5. **Cost sensitivity:** At 10 bps per trade, only TFT and CNN (W1) achieved positive cost ratios. More frequent traders (LSTM, TCN, Transformer) lost more to friction.

---

## 4. Binary Classification Variants (B1b–B4b)

### 4.1 Design

To test whether framing the problem as binary classification (up/down) rather than regression improves signal quality, we ran binary variants of the top four architectures. Changes:
- **Target:** Binarised as `(y > 0).astype(float32)` — i.e., 1 if 6-bar forward return is positive, 0 otherwise
- **Output:** Sigmoid activation + binary cross-entropy loss
- **Signal → Trade:** Position = +1 if P(up) > 0.5, else −1 (10 bps cost per trade)
- **Machine:** Dragon (RTX 4090), TensorFlow 2.21

### 4.2 Binary Results — Per Window

| Exp | Model | W1 SR | W1 Trades | W1 MaxDD | W2 SR | W2 Trades | W2 MaxDD |
|-----|-------|-------|-----------|----------|-------|-----------|----------|
| B1b | Binary CNN | -0.039 | 1 | 78.7% | -0.039 | 142 | 75.0% |
| B2b | Binary LSTM | +0.028 | 32 | 24.9% | -0.071 | 34 | 109.5% |
| B3b | Binary TFT | -0.035 | 38 | 62.8% | -0.078 | 78 | 116.7% |
| B4b | Binary TCN | +0.014 | 123 | 29.1% | -0.024 | 88 | 40.0% |

### 4.3 Binary Results — Aggregate

| Exp | Model | Agg SR | Consistency | Trades | Equity | K-Pass? |
|-----|-------|--------|-------------|--------|--------|---------|
| B1b | Binary CNN | -0.039 | 0% (0/2) | 143 | -$1,644 | FAIL |
| B2b | Binary LSTM | -0.022 | 50% (1/2) | 66 | $3,366 | FAIL |
| B3b | Binary TFT | -0.057 | 0% (0/2) | 116 | -$6,821 | FAIL |
| B4b | Binary TCN | -0.006 | 50% (1/2) | 211 | $8,353 | FAIL |

### 4.4 Binary vs Regression Comparison

| Architecture | Regression SR | Binary SR | Delta | Winner |
|-------------|--------------|-----------|-------|--------|
| CNN | -0.013 | -0.039 | -0.026 | Regression |
| LSTM | -0.082 | -0.022 | +0.060 | Binary |
| **TFT** | **+0.024** | **-0.057** | **-0.081** | **Regression** |
| TCN | -0.024 | -0.006 | +0.018 | Binary |

**Key finding:** Binary classification does **not** improve results. The regression TFT (+0.024) remains the only passing model. Binary TFT actually becomes the worst binary variant (-0.057), losing the nuanced magnitude information that made regression TFT successful. LSTM and TCN improve slightly in binary mode but remain negative.

The binarisation discards return magnitude, which appears crucial for the TFT's attention mechanism. The sparse, high-magnitude moves in BTC/USD 4h data carry the signal — binary classification treats a +5% and +0.01% move identically, destroying this information.

---

## 5. Comparison: Path B vs Path A

| Metric | A1 (btc_momentum) | A6 (momentum HPO) | B3 (TFT Reg.) | B4b (TCN Bin.) |
|--------|-------------------|--------------------|----------------|----------------|
| Agg Sharpe | 0.155 | 0.142 | 0.024 | -0.006 |
| Consistency | 100% | 100% | 100% | 50% |
| Max DD | 18.5% | 22.8% | 30.8% | 41.3% |
| Total Trades | 112 | 119 | 95 | 211 |
| Final Equity | $14,916 | $15,081 | $17,217 | $8,353 |

Path A's heuristic momentum strategy achieves higher Sharpe with far lower drawdown and fewer trades. Path B's regression TFT produces higher absolute returns but with substantially more risk. Binary classification variants uniformly fail to improve on regression.

---

## 6. Verdict

**B3 (TFT regression) is the only Path B candidate that survives kill criteria.**

All 4 binary classification variants (B1b–B4b) fail — confirming that return-magnitude information is critical for the ML signal on BTC/USD 4h data.

Regression TFT does not clearly outperform Path A (A1/A6 btc_momentum) on a risk-adjusted basis, but it provides the highest absolute equity ($17,217 vs $14,916–$15,081).

**Recommendation:** Proceed with TFT (regression) as the Path B representative for the final synthesis. Consider ensemble (Path A + Path B) exploration in future work.

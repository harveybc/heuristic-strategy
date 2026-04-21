# F-5: Data Pipeline Specification for Project 2

**Date**: 2025-06-17  
**Scope**: End-to-end data pipeline design from source acquisition through feature engineering to model training/evaluation for all Project 2 experimental paths  
**Depends on**: F-3 (data catalog), F-4 (asset/timeframe), F-6 (causal opportunities), F-7 (unsupervised opportunities), F-8 (infrastructure audit)

---

## 1. Pipeline Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Stage 1    │    │   Stage 2    │    │   Stage 3    │    │   Stage 4   │    │   Stage 5    │
│ Acquisition  │───▶│  Preprocessing│───▶│  Feature Eng │───▶│  Windowing  │───▶│   Training   │
│              │    │              │    │              │    │  + Splitting │    │  + Evaluation│
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
     │                    │                   │                   │                    │
  OANDA API          preprocessor         feature-eng         Orchestrator         predictor /
  HistData             repo               repo              (NEW — G-1)        heuristic-strategy
  FRED API                                                                      
  yfinance                                                                     
```

---

## 2. Stage 1: Data Acquisition

### 2.1 Sources and Scripts

| Data Type | Source | Format | Script / Method | Frequency |
|-----------|--------|--------|-----------------|-----------|
| EUR/USD 1h OHLC (2005-2024) | HistData (bulk) + OANDA (recent) | CSV | Download manually (HistData) + `oandapyV20` API call | One-time bulk + incremental |
| EUR/USD 1h OHLC (live top-up) | OANDA v20 API | JSON → CSV | `prediction_provider` feeder plugin or custom script | Daily cron |
| USD/JPY, GBP/USD 1h OHLC | OANDA v20 API | JSON → CSV | Same as EUR/USD | Part III start |
| SPY, BTC/USD daily OHLC | yfinance | CSV | `yfinance.download()` | Part III start |
| US macro (10Y yield, DXY, VIX, CPI, NFP) | FRED API | CSV | `fredapi` Python package | Monthly (macro release cadence) |
| CFTC EUR net positioning | CFTC CoT reports | CSV | `cot_reports` Python package or manual download | Weekly |

### 2.2 Storage Convention

```
heuristic-strategy/trading_research/project2/data/
├── raw/
│   ├── eurusd_1h_2005_2024.csv          # EUR/USD hourly, full history
│   ├── usdjpy_1h_2005_2024.csv          # USD/JPY hourly (Part III)
│   ├── spy_daily_1993_2024.csv          # SPY daily (Part III)
│   ├── btcusd_daily_2015_2024.csv       # BTC/USD daily (Part IV+)
│   ├── macro_fred_monthly.csv           # FRED macro indicators
│   └── cftc_eur_weekly.csv              # CFTC positioning
├── processed/
│   ├── eurusd_4h_2005_2024.csv          # Resampled to 4h
│   ├── eurusd_daily_2005_2024.csv       # Resampled to daily
│   └── eurusd_4h_features_full.csv      # After feature engineering
├── windows/
│   ├── window_001/                       # Per rolling window
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   ├── window_002/
│   │   └── ...
│   └── window_manifest.json             # Window boundaries, dates, bar counts
└── models/
    ├── window_001/
    │   ├── model.keras
    │   ├── norm_params.json
    │   └── metrics.json
    └── ...
```

### 2.3 Data Quality Checks

| Check | Method | Threshold |
|-------|--------|-----------|
| Missing bars | Count gaps > 1h in 1h data | < 0.5% missing after business-hours filter |
| Duplicate timestamps | `df.duplicated(subset=['DATE_TIME'])` | Zero duplicates |
| Price sanity | Min/max bounds, % change > 5% flagged | Manual review of flagged bars |
| Weekend/holiday filter | Remove Sat-Sun bars (FX closes Fri 17:00 EST → Sun 17:00 EST) | Standard FX calendar |
| Timezone consistency | All data in UTC | Verified at acquisition |

---

## 3. Stage 2: Preprocessing

### 3.1 Resampling (1h → 4h / Daily)

| Timeframe | Method | OHLC Aggregation |
|-----------|--------|-----------------|
| 4h | `df.resample('4h')` | O=first, H=max, L=min, C=last, V=sum |
| Daily | `df.resample('1D')` | Same |

**Tool**: Custom script (not preprocessor repo — resampling is pre-preprocessing).

### 3.2 Preprocessor Pipeline

Uses existing `preprocessor` repo plugins in sequence:

```bash
preprocessor --plugin cleaner --input raw/eurusd_4h.csv --output cleaned.csv
preprocessor --plugin normalizer --input cleaned.csv --output normalized.csv --save_params norm_params.json
preprocessor --plugin feature_selector --input normalized.csv --output selected.csv
```

**Key for rolling retraining**: The `--save_params` flag on normalizer saves mean/std per column. Each rolling window must:
1. Fit normalization on training portion only
2. Apply saved params to validation/test portions (no look-ahead)
3. Save params alongside model for inference

### 3.3 Preprocessor Config per Window

```json
{
  "input_file": "windows/window_001/train_raw.csv",
  "output_file": "windows/window_001/train.csv",
  "plugin": "normalizer",
  "save_config": "windows/window_001/norm_params.json",
  "method": "z_score"
}
```

---

## 4. Stage 3: Feature Engineering

### 4.1 Feature-eng Pipeline

Uses existing `feature-eng` repo plugins:

```bash
feature-eng --plugin tech_indicator --input eurusd_4h.csv --output features_tech.csv
feature-eng --plugin direction_labels --input features_tech.csv --output features_labeled.csv
```

### 4.2 Feature Sets by Experimental Phase (from F-4 §6.3)

| Phase | Feature Set | Count | Plugins Used |
|-------|-------------|-------|-------------|
| Phase 0 (baseline) | P1 technical only | 12 | tech_indicator |
| Phase 1 (causal filter) | PCMCI+ root causes + leading TE | 5 | tech_indicator → manual filter |
| Phase 2 (+ macro) | Phase 1 + FRED/CFTC | 11-13 | tech_indicator + custom macro merge |
| Phase 3 (+ cross-asset) | Phase 2 + SPY/BTC/Gold returns | 14-17 | Phase 2 + merge script |
| Phase 4 (+ autoencoder) | Phase 3 compressed | 4-6 | Phase 3 → autoencoder (UL-3) |

### 4.3 Feature Engineering for Macro Data

Macro data arrives at different frequencies than price data and requires alignment:

| Macro Feature | Raw Frequency | Alignment Method |
|---------------|--------------|-----------------|
| US 10Y yield | Daily | Forward-fill to 4h bars |
| DXY index | Daily | Forward-fill to 4h bars |
| VIX | Daily | Forward-fill to 4h bars |
| US-EU rate differential | Monthly | Forward-fill, step function |
| CFTC EUR net positioning | Weekly | Forward-fill (Tuesday release → effective Wednesday 4h bars) |
| CPI YoY | Monthly | Forward-fill |

**Script needed**: `merge_macro_features.py` — aligns multi-frequency data to target timeframe with forward-fill (no look-ahead).

---

## 5. Stage 4: Windowing and Splitting

### 5.1 Rolling Window Orchestrator (NEW — addresses G-1)

This is the **critical missing piece** identified in F-8. The orchestrator manages:

1. **Window boundary definition**: Given (start_date, end_date, train_years, val_years, step_years), generate all windows
2. **Per-window data extraction**: Slice full feature CSV into train/val/test portions
3. **Per-window normalization**: Fit on train, apply to val/test
4. **Per-window training dispatch**: Call predictor or heuristic-strategy with window-specific config
5. **Per-window evaluation**: Collect metrics, log to experiment tracker
6. **Cross-window aggregation**: Summarize OOS performance across all windows

### 5.2 Window Definitions

#### Design A: Anchored Expanding (Conservative)

```
Window 1: Train [2005-2008] → Val [2009] → Test [2010]
Window 2: Train [2005-2009] → Val [2010] → Test [2011]
Window 3: Train [2005-2010] → Val [2011] → Test [2012]
...
Window 10: Train [2005-2017] → Val [2018] → Test [2019]
Held-out:  Train [2005-2019] → Test [2020-2024]
```

#### Design B: Sliding Window (Aggressive)

```
Window 1: Train [2005-2007] → Val [2008-H1] → Test [2008-H2]
Window 2: Train [2005.5-2008] → Val [2008.5-2009] → Test [2009-H2]
...
(6-month step, 3-year train, 6-month val, 6-month test)
```

#### Design C: Regime-Triggered (Experimental — depends on UL-2)

```
Windows defined by change-point detection:
  When Wasserstein(F[t-w:t], F[t-2w:t-w]) > τ → start new window
  Train: all data up to trigger point
  Val: last 6 months before trigger
  Test: 6 months after trigger
```

**Default**: Design A (anchored expanding) with annual steps. Matches heuristic-strategy's existing WFO.

### 5.3 Window Manifest Format

```json
{
  "design": "anchored_expanding",
  "asset": "EURUSD",
  "timeframe": "4h",
  "windows": [
    {
      "id": 1,
      "train_start": "2005-01-01",
      "train_end": "2008-12-31",
      "val_start": "2009-01-01",
      "val_end": "2009-12-31",
      "test_start": "2010-01-01",
      "test_end": "2010-12-31",
      "train_bars": 6400,
      "val_bars": 1600,
      "test_bars": 1600
    }
  ],
  "held_out": {
    "start": "2020-01-01",
    "end": "2024-12-31"
  }
}
```

---

## 6. Stage 5: Training and Evaluation

### 6.1 Path A Pipeline (Adaptive Heuristic)

```
For each window w:
  1. feature-eng → features_w.csv
  2. [Optional] Re-fit GMM on train_w (UL-1)
  3. heuristic-strategy --plugin regime_adaptive \
       --input features_w.csv \
       --optimize  (DEAP GA on train_w)
  4. Evaluate on val_w → select best params
  5. Backtest on test_w → record OOS metrics
  6. Log: {window_id, params, train_sharpe, val_sharpe, test_sharpe, max_dd, trades, cost_ratio}
```

**Existing tool**: `heuristic-strategy/walk_forward_optimizer.py` — may need minor adaptation for configurable window boundaries.

### 6.2 Path B Pipeline (Supervised ML Rolling)

```
For each window w:
  1. feature-eng → features_w.csv
  2. preprocessor --plugin normalizer --fit_on train_w --apply_to {train_w, val_w, test_w}
  3. predictor --plugin {cnn|lstm|transformer|tft} \
       --train train_w.csv \
       --validate val_w.csv \
       --epochs 100 --early_stopping patience=10 \
       --save_model models/window_w/model.keras
  4. predictor --plugin {same} --evaluate test_w.csv --load_model models/window_w/model.keras
  5. heuristic-strategy --plugin ls_pred_strategy \
       --predictions predictor_output_w.csv \
       --backtest test_w
  6. Log: {window_id, model_type, train_mae, val_mae, test_mae, train_f1, val_f1, test_f1, sharpe, max_dd}
```

**Gap**: Steps 1-6 must be orchestrated by the new rolling retraining orchestrator (G-1). Each step uses existing CLI tools.

### 6.3 Path C Pipeline (RL — requires G-3 resolution)

```
For each window w:
  1. feature-eng → features_w.csv (state representation)
  2. Initialize RL environment (gym-fx — DOES NOT EXIST YET)
  3. Train RL agent on train_w episodes
  4. Evaluate on val_w → hyperparameter selection
  5. Test on test_w → record OOS metrics
  6. Log: {window_id, agent_type, train_reward, val_reward, test_reward, sharpe, max_dd}
```

---

## 7. Experiment Tracking

### 7.1 Tracking System

Adopt doin-node's experiment tracker pattern (CSV + optional OLAP):

**CSV format** (one row per window per experiment):
```
experiment_id,path,window_id,model_type,train_start,train_end,test_start,test_end,
train_sharpe,val_sharpe,test_sharpe,train_mae,val_mae,test_mae,
max_dd,num_trades,cost_ratio,params_json,timestamp
```

**Location**: `trading_research/project2/data/experiments/experiment_log.csv`

### 7.2 Experiment Naming Convention

```
{path}_{model}_{feature_phase}_{window_design}_{timestamp}
Example: pathB_tft_phase1_anchored_20250620_143022
```

---

## 8. Compute Allocation

### 8.1 Machine Assignment

| Machine | GPU | Primary Role | Backup Role |
|---------|-----|-------------|-------------|
| **Omega** | RTX 4070 (12GB) | Path A (heuristic optimization — CPU-heavy, light GPU) | Path B small models |
| **Dragon** | RTX 4090 (24GB) | Path B (ML training — GPU-heavy) | Path C (RL training) |
| **Gamma** | RTX 5070 Ti (16GB) | Path B overflow / CI-5 rolling PCMCI | Path A parallel runs |

### 8.2 Estimated Compute per Path

| Path | Per Window | × Windows | Total | Machine |
|------|-----------|-----------|-------|---------|
| A (Heuristic WFO) | ~30 min (GA 50 gen × 20 pop) | 10 | ~5 hours | Omega |
| B (ML per model type) | ~20 min (100 epochs TFT) | 10 | ~3.3 hours/model × 4 models = ~13 hours | Dragon |
| C (RL) | ~1 hour (1M steps) | 10 | ~10 hours | Dragon/Gamma |

Total Part II compute: ~30 hours across 3 machines. Parallelizable.

---

## 9. Pipeline Implementation Plan

### 9.1 Scripts to Build (Part II Prep)

| Script | Purpose | Effort | Priority |
|--------|---------|--------|----------|
| `download_data.py` | Acquire all raw data from sources | Small | Pre-Part II |
| `resample_data.py` | 1h → 4h, daily resampling with proper OHLC aggregation | Small | Pre-Part II |
| `merge_macro_features.py` | Align multi-frequency macro data to target timeframe | Medium | Phase 2 features |
| `rolling_orchestrator.py` | **G-1 resolution**: window management, per-window pipeline dispatch, experiment logging | **Large** | Part II start — **critical path** |
| `window_manifest_generator.py` | Generate window boundaries from config | Small | Part II start |
| `aggregate_results.py` | Cross-window OOS summary, statistical tests | Medium | Part II evaluation |

### 9.2 Existing Scripts to Adapt

| Script | Repo | Adaptation Needed |
|--------|------|------------------|
| `walk_forward_optimizer.py` | heuristic-strategy | Make window boundaries configurable (currently hardcoded to yearly) |
| `experiment_tracker.py` | doin-node | Extract from DOIN context for standalone use |
| `cluster_regime_analysis.py` | causal-inference | Parameterize for rolling re-fit (UL-1) |

---

## 10. Data Pipeline Validation Checklist

Before Part II experiments begin, verify:

- [ ] EUR/USD 1h data 2005-2024 downloaded and gap-checked
- [ ] Resampling to 4h produces expected bar count (~33,000)
- [ ] Feature engineering produces all 12 P1 features without NaN (except warmup period)
- [ ] Normalization save/load round-trips correctly (fit on train, apply to test, reload params)
- [ ] Window manifest generates expected boundaries (10 windows for anchored expanding)
- [ ] Per-window data slicing produces correct date ranges and bar counts
- [ ] One end-to-end pilot run (single window, single model) completes and logs metrics
- [ ] Experiment tracker CSV appends correctly across multiple runs

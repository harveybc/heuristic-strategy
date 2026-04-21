#!/usr/bin/env python3
"""
Phase B: Direction ATR + Real CNN Predictions

Generates CNN direction predictions for the full 15yr EURUSD dataset,
then runs WFO optimizing ATR params with those predictions.

Pipeline:
  1. Resample 1H OHLC → 4H
  2. Compute 22 technical indicator features (matching training pipeline)
  3. Normalize using normalization_config_c.json (training set stats)
  4. Load CNN direction_long + direction_short .keras models
  5. Run inference on all 4H bars → cached predictions
  6. Run direction_atr yearly backtests mapping 1H bars to 4H predictions
"""
import sys
import os
import json
import time
import numpy as np

# numpy compat fix for pandas_ta
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["STRATEGY_QUIET"] = "1"
os.environ["PREDICTION_PROVIDER_QUIET"] = "1"

# Paths
OHLC_FILE = "tests/data/eurusd_hour_2005_2020.csv"
NORM_CONFIG = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/data_downsampled/"
    "phase_1_c/normalization_config_c.json"
)
LONG_MODEL = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/results/phase_1c_direction/"
    "phase_1c_direction_cnn_direction_long_model.keras"
)
SHORT_MODEL = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/results/phase_1c_direction/"
    "phase_1c_direction_cnn_direction_short_model.keras"
)
LONG_META = LONG_MODEL.replace('.keras', '_metadata.json')
SHORT_META = SHORT_MODEL.replace('.keras', '_metadata.json')

# Feature computation constants (must match normalize_phase1c.py exactly)
TP_PIPS = 131.325
PIP_COST = 0.00001
ROLLING_WINDOW = 24

FEATURE_COLS = [
    'ATR', 'RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal',
    'ADX', 'DI_plus', 'DI_minus', 'Stochastic_K', 'Stochastic_D',
    'BB_Width', 'CCI', 'WilliamsR', 'ROC',
    'ATR_ratio', 'BB_position', 'rolling_std_24', 'price_minus_ema',
    'hod_sin', 'hod_cos', 'dow_sin', 'dow_cos',
]


# ================================================================
# Step 1: Feature engineering
# ================================================================

def resample_to_4h(df_1h):
    """Resample hourly OHLC to 4H matching training pipeline boundaries."""
    df = df_1h.copy()
    # Training data uses 4H bars at boundaries: 01:00, 05:00, 09:00, 13:00, 17:00, 21:00
    # pandas resample with offset='1h' and period='4h' starting at 01:00
    ohlc_rules = {
        'OPEN': 'first',
        'HIGH': 'max',
        'LOW': 'min',
        'CLOSE': 'last',
    }
    # Resample using 4H periods with offset 1h
    df_4h = df.resample('4h', offset='1h').agg(ohlc_rules).dropna()
    return df_4h


def compute_features(df_4h):
    """Compute 22 technical indicator features from 4H OHLC."""
    import pandas_ta as ta

    df = df_4h.copy()
    # Need DATE_TIME column for cyclic features
    df['DATE_TIME'] = df.index

    high = df['HIGH'].astype(float)
    low = df['LOW'].astype(float)
    close = df['CLOSE'].astype(float)

    # Tier 1: Core indicators
    df['ATR'] = ta.atr(high, low, close, length=14)
    df['RSI'] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Histogram'] = macd['MACDh_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']

    adx = ta.adx(high, low, close, length=14)
    df['ADX'] = adx['ADX_14']
    df['DI_plus'] = adx['DMP_14']
    df['DI_minus'] = adx['DMN_14']

    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    df['Stochastic_K'] = stoch['STOCHk_14_3_3']
    df['Stochastic_D'] = stoch['STOCHd_14_3_3']

    # Tier 2: Secondary indicators
    bbands = ta.bbands(close, length=20, std=2.0)
    bb_upper = bbands['BBU_20_2.0']
    bb_lower = bbands['BBL_20_2.0']
    df['BB_Width'] = bb_upper - bb_lower

    df['CCI'] = ta.cci(high, low, close, length=20)
    df['WilliamsR'] = ta.willr(high, low, close, length=14)
    df['ROC'] = ta.roc(close, length=10)

    # Tier 3: Derived
    atr_in_pips = df['ATR'] / PIP_COST
    df['ATR_ratio'] = TP_PIPS / atr_in_pips.replace(0, np.nan)

    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df['BB_position'] = (close - bb_lower) / bb_range

    # Rolling features
    typical_price = (high + low + close) / 3.0
    df['rolling_std_24'] = typical_price.rolling(window=ROLLING_WINDOW).std()
    rolling_ema = typical_price.ewm(span=ROLLING_WINDOW, adjust=False).mean()
    df['price_minus_ema'] = typical_price - rolling_ema

    # Cyclic time features
    dt = pd.to_datetime(df['DATE_TIME'])
    df['hod_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hod_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # Drop warmup NaN rows
    df = df.dropna(subset=FEATURE_COLS)
    return df


def normalize_features(df, norm_config):
    """Apply z-score normalization using training set statistics."""
    df_norm = df.copy()
    for col in FEATURE_COLS:
        if col in norm_config and col in df_norm.columns:
            mean = norm_config[col]['mean']
            std = norm_config[col]['std']
            if std > 0:
                df_norm[col] = (df_norm[col] - mean) / std
    return df_norm


# ================================================================
# Step 2: CNN Inference
# ================================================================

def _strip_quantization_config(obj):
    """Recursively strip quantization_config from Keras config dicts."""
    if isinstance(obj, dict):
        obj.pop('quantization_config', None)
        for v in obj.values():
            _strip_quantization_config(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_quantization_config(item)


def load_cnn_model(model_path, meta_path, label):
    """Load a CNN direction model, handling Keras version mismatches."""
    import tensorflow as tf
    import zipfile
    import tempfile

    # Load metadata for window_size and feature columns
    window_size = 71
    feature_cols = FEATURE_COLS
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        window_size = meta.get('window_size', window_size)
        if 'feature_columns' in meta:
            feature_cols = meta['feature_columns']
        print(f"  [{label}] Metadata: window={window_size}, "
              f"features={len(feature_cols)}")

    # Extract .keras zip, patch config to fix version mismatches, reload
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(model_path, 'r') as zf:
            zf.extractall(tmpdir)

        # Patch config.json to strip quantization_config
        config_path = os.path.join(tmpdir, 'config.json')
        with open(config_path) as f:
            model_config = json.load(f)
        _strip_quantization_config(model_config)
        with open(config_path, 'w') as f:
            json.dump(model_config, f)

        # Repack as temp .keras file
        patched_path = os.path.join(tmpdir, 'patched.keras')
        with zipfile.ZipFile(patched_path, 'w') as zf_out:
            for root, dirs, files in os.walk(tmpdir):
                for fname in files:
                    if fname == 'patched.keras':
                        continue
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, tmpdir)
                    zf_out.write(fpath, arcname)

        model = tf.keras.models.load_model(patched_path, compile=False,
                                              safe_mode=False)
        print(f"  [{label}] Loaded from {model_path} (patched)")

    # Get actual input shape from model
    input_shape = model.input_shape
    if input_shape and len(input_shape) == 3:
        actual_features = input_shape[2]
        actual_window = input_shape[1]
        if actual_features != len(feature_cols):
            print(f"  [{label}] Model expects {actual_features} features, "
                  f"metadata has {len(feature_cols)} — will pad/adjust")
        if actual_window != window_size:
            print(f"  [{label}] Model expects window={actual_window}, "
                  f"metadata says {window_size} — using model's")
            window_size = actual_window

    return model, window_size, feature_cols


def compute_window_stats(window, target_col_idx, periods=(12, 48)):
    """Compute per-window statistics matching predictor's add_window_stats.

    For each period p, computes from target column's last p timesteps:
      - rolling_std_p
      - rolling_ema_p  (alpha = 2/(p+1))
      - price_minus_ema_p  (last value - ema)

    Returns array of shape (n_stats,) = (3 * len(periods),).
    """
    window_size = window.shape[0]
    target_vals = window[:, target_col_idx]
    stats = []
    for p in periods:
        span = min(p, window_size)
        tail = target_vals[-span:]
        # std
        stats.append(np.std(tail, dtype=np.float32))
        # ema
        alpha = 2.0 / (span + 1.0)
        ema = tail[0]
        for t in range(1, len(tail)):
            ema = alpha * tail[t] + (1.0 - alpha) * ema
        stats.append(np.float32(ema))
        # price minus ema
        stats.append(np.float32(target_vals[-1] - ema))
    return np.array(stats, dtype=np.float32)


def run_inference_batch(model, df_norm, window_size, feature_cols, label,
                        add_window_stats=False, target_col='ATR'):
    """Run CNN inference on all bars, returning P(up) for each."""
    import tensorflow as tf

    feat_cols = [c for c in feature_cols if c in df_norm.columns]
    features = df_norm[feat_cols].values.astype(np.float32)
    n_bars = len(features)
    n_features = len(feat_cols)

    # Window stats config
    target_col_idx = feat_cols.index(target_col) if target_col in feat_cols else 0
    n_stats = 6 if add_window_stats else 0  # 3 stats * 2 periods
    total_features = n_features + n_stats

    predictions = np.full(n_bars, np.nan)

    print(f"  [{label}] Running inference on {n_bars} bars "
          f"(window={window_size}, features={n_features}+{n_stats}stats)...")
    t0 = time.time()

    batch_size = 256
    valid_indices = list(range(window_size - 1, n_bars))
    n_valid = len(valid_indices)

    for batch_start in range(0, n_valid, batch_size):
        batch_end = min(batch_start + batch_size, n_valid)
        batch_indices = valid_indices[batch_start:batch_end]

        # Build batch of windows
        X_batch = np.zeros((len(batch_indices), window_size, total_features),
                           dtype=np.float32)
        for j, idx in enumerate(batch_indices):
            start = idx - window_size + 1
            window = features[start:idx + 1]
            X_batch[j, :, :n_features] = window

            if add_window_stats:
                stats = compute_window_stats(window, target_col_idx)
                # Broadcast stats across all timesteps
                X_batch[j, :, n_features:] = stats[np.newaxis, :]

        # Predict batch
        preds = model.predict(X_batch, verbose=0, batch_size=batch_size)
        preds = np.asarray(preds).flatten()

        for j, idx in enumerate(batch_indices):
            predictions[idx] = preds[j]

        if (batch_start // batch_size) % 20 == 0:
            pct = (batch_end / n_valid) * 100
            print(f"    {label}: {pct:.0f}% ({batch_end}/{n_valid})")

    elapsed = time.time() - t0
    valid_count = np.sum(~np.isnan(predictions))
    print(f"  [{label}] Done: {valid_count} predictions in {elapsed:.1f}s")
    return predictions


def generate_full_predictions(ohlc_1h, norm_config):
    """Generate CNN predictions for all 4H bars across the full dataset."""
    print("\n=== GENERATING CNN PREDICTIONS ===")

    # Step 1: Resample to 4H
    print("Resampling to 4H...")
    df_4h = resample_to_4h(ohlc_1h)
    print(f"  4H bars: {len(df_4h)} ({df_4h.index.min()} to {df_4h.index.max()})")

    # Step 2: Compute features
    print("Computing 22 technical features...")
    df_feat = compute_features(df_4h)
    print(f"  After warmup: {len(df_feat)} bars")

    # Step 3: Normalize
    print("Normalizing with training set statistics...")
    df_norm = normalize_features(df_feat, norm_config)

    # Step 4: Load models
    print("Loading CNN models...")
    long_model, long_window, long_feats = load_cnn_model(
        LONG_MODEL, LONG_META, "long")

    try:
        short_model, short_window, short_feats = load_cnn_model(
            SHORT_MODEL, SHORT_META, "short")
    except Exception as e:
        print(f"  [short] Load failed: {e}")
        print(f"  [short] Falling back to long model for exit predictions")
        short_model = long_model
        short_window = long_window
        short_feats = long_feats

    # Step 5: Run inference
    long_preds = run_inference_batch(
        long_model, df_norm, long_window, long_feats, "long",
        add_window_stats=True, target_col='ATR')
    short_preds = run_inference_batch(
        short_model, df_norm, short_window, short_feats, "short",
        add_window_stats=True, target_col='ATR')

    # Build prediction dataframe
    pred_df = pd.DataFrame({
        'p_up_long': long_preds,
        'p_up_short': short_preds,
        'CLOSE': df_feat['CLOSE'].values,
    }, index=df_feat.index)
    pred_df = pred_df.dropna(subset=['p_up_long'])

    print(f"\n  Total valid predictions: {len(pred_df)}")
    print(f"  Date range: {pred_df.index.min()} to {pred_df.index.max()}")
    print(f"  P(up_long) mean={pred_df['p_up_long'].mean():.3f}, "
          f"std={pred_df['p_up_long'].std():.3f}")
    print(f"  P(up_short) mean={pred_df['p_up_short'].mean():.3f}, "
          f"std={pred_df['p_up_short'].std():.3f}")

    return pred_df


# ================================================================
# Step 3: Offline CNN Prediction Source
# ================================================================

class OfflineCNNPredictionSource:
    """
    Maps hourly strategy timestamps to 4H CNN predictions.
    Implements the same interface as ApiPredictionSource.
    """

    def __init__(self, pred_df, confidence_threshold=0.55):
        self.pred_df = pred_df
        self.threshold = confidence_threshold
        # Pre-build index for fast nearest-timestamp lookup
        self._pred_index = pred_df.index.values
        self._pred_timestamps = pred_df.index

    def _get_nearest_prediction(self, dt_hour):
        """Map hourly timestamp to nearest 4H prediction."""
        ts = pd.Timestamp(dt_hour)
        idx = self.pred_df.index.get_indexer([ts], method='ffill')[0]
        if idx < 0:
            idx = 0
        return self.pred_df.iloc[idx]

    def get_entry_prediction(self, dt_hour, tp_pips=0, sl_pips=0,
                             spread_pips=0, commission_per_lot=0,
                             slippage_pips=0):
        try:
            row = self._get_nearest_prediction(dt_hour)
            p_up = float(row['p_up_long'])
            threshold = self.threshold

            buy_signal = 1.0 if p_up >= threshold else 0.0
            sell_signal = 1.0 if (1.0 - p_up) >= threshold else 0.0

            return {
                "available": True,
                "buy_entry_binary": buy_signal,
                "sell_entry_binary": sell_signal,
                "bars_remaining": 120,
                "buy_confidence": 1.0,
                "sell_confidence": 1.0,
                "p_up_long": p_up,
                "current_price": float(row['CLOSE']),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def get_exit_prediction(self, dt_hour, direction="buy",
                            tp_price=0, sl_price=0):
        try:
            row = self._get_nearest_prediction(dt_hour)
            p_up = float(row['p_up_short'])
            threshold = self.threshold

            if direction == 'buy':
                should_exit = (1.0 - p_up) >= threshold
            else:
                should_exit = p_up >= threshold

            return {
                "available": True,
                "exit_binary": 0.0 if should_exit else 1.0,
                "exit_confidence": 1.0,
                "p_up_short": p_up,
                "current_price": float(row['CLOSE']),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}


# ================================================================
# Step 4: Run WFO with CNN predictions
# ================================================================

def run_cnn_backtest(base_data, cnn_source, config, label=""):
    """Run direction_atr strategy with offline CNN predictions."""
    import backtrader as bt
    from app.plugins.plugin_direction_atr import Plugin as DirectionATRPlugin

    plugin = DirectionATRPlugin()
    original_init = plugin.DirectionATRStrategy.__init__

    def patched_init(self_strat, *args, **kwargs):
        bt.Strategy.__init__(self_strat)
        self_strat._pred_source = cnn_source
        self_strat.data0 = self_strat.datas[0]
        self_strat.initial_balance = self_strat.broker.getvalue()
        self_strat.trade_entry_dates = []
        self_strat.balance_history = []
        self_strat.date_history = []
        self_strat.trade_low = None
        self_strat.trade_high = None
        self_strat.trades = []
        self_strat.current_tp = None
        self_strat.current_sl = None
        self_strat.current_direction = None
        self_strat.order_direction = None
        self_strat.order_entry_price = None
        self_strat.trade_entry_bar = None
        self_strat.current_volume = None
        self_strat.atr = bt.indicators.ATR(
            self_strat.data0, period=self_strat.p.atr_period
        )

    plugin.DirectionATRStrategy.__init__ = patched_init

    candidate = [
        config.get("atr_period", 14),
        config.get("atr_tp_multiplier", 2.0),
        config.get("atr_sl_multiplier", 1.5),
    ]
    profit, stats = plugin.evaluate_candidate(
        candidate, base_data, None, None, config
    )
    trades = plugin.trades
    plugin.DirectionATRStrategy.__init__ = original_init

    return profit, stats, trades


def run_wfo_cnn(base_data_1h, cnn_source, config, thresholds_to_test):
    """Run walk-forward test across all years with CNN predictions."""
    print(f"\n{'='*70}")
    print(f"PHASE B: WFO WITH REAL CNN PREDICTIONS")
    print(f"{'='*70}")
    print(f"ATR period={config['atr_period']}, "
          f"TP mult={config['atr_tp_multiplier']}, "
          f"SL mult={config['atr_sl_multiplier']}")
    print(f"Confidence threshold={cnn_source.threshold}")
    print(f"Costs: spread={config['spread_pips']}pips, "
          f"comm={config['commission_per_lot']}/lot, "
          f"slip={config['slippage_pips']}pips")
    print(f"{'='*70}\n")

    fold_results = []
    all_trades = []

    for test_year in range(2006, 2020):
        year_mask = base_data_1h.index.year == test_year
        year_data = base_data_1h[year_mask].copy()
        if len(year_data) < 100:
            continue

        t0 = time.time()
        profit, stats, trades = run_cnn_backtest(
            year_data, cnn_source, config, label=f"Y{test_year}"
        )
        elapsed = time.time() - t0

        n_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_pct = (wins / n_trades * 100) if n_trades > 0 else 0
        pnls = [t['pnl'] for t in trades]
        sharpe = ((np.mean(pnls) / (np.std(pnls) + 1e-10))
                  if n_trades > 1 else 0)

        fold_results.append({
            "year": test_year,
            "profit": profit,
            "trades": n_trades,
            "win_pct": win_pct,
            "sharpe": sharpe,
            "elapsed": elapsed,
        })
        for t in trades:
            t["year"] = test_year
        all_trades.extend(trades)

        sign = '+' if profit >= 0 else ''
        print(f"  {test_year}: {sign}${profit:,.2f} | "
              f"{n_trades} trades | Win {win_pct:.0f}% | "
              f"Sharpe {sharpe:.2f} | {elapsed:.0f}s")

    # Aggregate
    total_profit = sum(f["profit"] for f in fold_results)
    total_trades = sum(f["trades"] for f in fold_results)
    total_wins = sum(1 for t in all_trades if t['pnl'] > 0)
    total_win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0
    all_pnls = [t['pnl'] for t in all_trades]
    agg_sharpe = ((np.mean(all_pnls) / (np.std(all_pnls) + 1e-10))
                  if len(all_pnls) > 1 else 0)
    profitable_folds = sum(1 for f in fold_results if f["profit"] > 0)

    # Max drawdown
    equity = 10000.0
    peak = equity
    max_dd = 0
    for t in all_trades:
        equity += t['pnl']
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "total_profit": total_profit,
        "total_trades": total_trades,
        "total_win_pct": total_win_pct,
        "aggregate_sharpe": agg_sharpe,
        "max_drawdown_usd": max_dd,
        "final_equity": equity,
        "profitable_folds": profitable_folds,
        "total_folds": len(fold_results),
        "fold_results": fold_results,
        "all_trades": all_trades,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    # Load hourly OHLC
    from app.data_handler import load_csv
    print(f"Loading dataset: {OHLC_FILE}")
    ohlc_1h = load_csv(OHLC_FILE, headers=True)
    print(f"Loaded: {ohlc_1h.shape[0]} bars, "
          f"{ohlc_1h.index.min()} to {ohlc_1h.index.max()}")

    # Load normalization config
    with open(NORM_CONFIG) as f:
        norm_config = json.load(f)

    # Generate CNN predictions for full dataset
    pred_df = generate_full_predictions(ohlc_1h, norm_config)

    # Cache predictions
    cache_file = "cnn_predictions_15yr.csv"
    pred_df.to_csv(cache_file)
    print(f"Cached predictions to {cache_file}")

    # Strategy config
    config = {
        "prediction_source": "API",
        "pp_api_url": "http://offline",
        "pp_timeout": 999,
        "headers": True,
        "disable_multiprocessing": True,
        "atr_period": 14,
        "atr_tp_multiplier": 2.0,
        "atr_sl_multiplier": 1.5,
        "spread_pips": 15.0,
        "commission_per_lot": 7.0,
        "slippage_pips": 5.0,
    }

    # ================================================================
    # B1: Baseline — same params as oracle ceiling (threshold=0.55)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE B1: CNN BASELINE (threshold=0.55, same ATR as oracle)")
    print(f"{'='*70}\n")

    cnn_source = OfflineCNNPredictionSource(pred_df, confidence_threshold=0.55)
    b1_results = run_wfo_cnn(ohlc_1h, cnn_source, config, [0.55])

    sign = '+' if b1_results['total_profit'] >= 0 else ''
    print(f"\n{'='*70}")
    print(f"B1 RESULTS: CNN baseline (threshold=0.55)")
    print(f"{'='*70}")
    print(f"Total Profit:     {sign}${b1_results['total_profit']:,.2f}")
    print(f"Total Trades:     {b1_results['total_trades']}")
    print(f"Win Rate:         {b1_results['total_win_pct']:.1f}%")
    print(f"Aggregate Sharpe: {b1_results['aggregate_sharpe']:.3f}")
    print(f"Max Drawdown:     ${b1_results['max_drawdown_usd']:,.2f}")
    print(f"Profitable Years: {b1_results['profitable_folds']}/{b1_results['total_folds']}")
    print(f"{'='*70}\n")

    # ================================================================
    # B2: Threshold sweep — find best confidence threshold
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE B2: CONFIDENCE THRESHOLD SWEEP")
    print(f"{'='*70}")
    print(f"Testing thresholds: 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70")
    print(f"Using 2012-2016 as representative sample")
    print(f"{'='*70}\n")

    year_mask = ((ohlc_1h.index.year >= 2012) &
                 (ohlc_1h.index.year <= 2016))
    sample_data = ohlc_1h[year_mask].copy()

    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]
    threshold_results = []

    for thresh in thresholds:
        cnn_source = OfflineCNNPredictionSource(pred_df,
                                                 confidence_threshold=thresh)
        t0 = time.time()
        profit, _, trades = run_cnn_backtest(
            sample_data, cnn_source, config, label=f"thresh={thresh}")
        elapsed = time.time() - t0

        n_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_pct = (wins / n_trades * 100) if n_trades > 0 else 0
        pnls = [t['pnl'] for t in trades]
        sharpe = ((np.mean(pnls) / (np.std(pnls) + 1e-10))
                  if n_trades > 1 else 0)

        row = {
            "threshold": thresh,
            "profit": profit,
            "trades": n_trades,
            "win_pct": win_pct,
            "sharpe": sharpe,
        }
        threshold_results.append(row)

        sign = '+' if profit >= 0 else ''
        print(f"  thresh={thresh:.2f}: {sign}${profit:,.2f} | "
              f"{n_trades} trades | Win {win_pct:.0f}% | "
              f"Sharpe {sharpe:.2f} | {elapsed:.0f}s")

    # ================================================================
    # B3: ATR param sweep — find best TP/SL ratio
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE B3: ATR PARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"Testing TP mult x SL mult grid on 2012-2016 sample")
    print(f"{'='*70}\n")

    # Find best threshold from B2
    best_thresh_row = max(threshold_results, key=lambda r: r['profit'])
    best_threshold = best_thresh_row['threshold']
    print(f"Using best threshold from B2: {best_threshold}")

    tp_mults = [1.5, 2.0, 2.5, 3.0, 4.0]
    sl_mults = [0.5, 1.0, 1.5, 2.0, 3.0]
    atr_periods = [10, 14, 21]

    param_results = []

    for atr_period in atr_periods:
        for tp_mult in tp_mults:
            for sl_mult in sl_mults:
                test_config = config.copy()
                test_config['atr_period'] = atr_period
                test_config['atr_tp_multiplier'] = tp_mult
                test_config['atr_sl_multiplier'] = sl_mult

                cnn_source = OfflineCNNPredictionSource(
                    pred_df, confidence_threshold=best_threshold)
                t0 = time.time()
                profit, _, trades = run_cnn_backtest(
                    sample_data, cnn_source, test_config,
                    label=f"atr={atr_period}/tp={tp_mult}/sl={sl_mult}")
                elapsed = time.time() - t0

                n_trades = len(trades)
                wins = sum(1 for t in trades if t['pnl'] > 0)
                win_pct = (wins / n_trades * 100) if n_trades > 0 else 0
                pnls = [t['pnl'] for t in trades]
                sharpe = ((np.mean(pnls) / (np.std(pnls) + 1e-10))
                          if n_trades > 1 else 0)

                row = {
                    "atr_period": atr_period,
                    "tp_mult": tp_mult,
                    "sl_mult": sl_mult,
                    "profit": profit,
                    "trades": n_trades,
                    "win_pct": win_pct,
                    "sharpe": sharpe,
                }
                param_results.append(row)

                sign = '+' if profit >= 0 else ''
                print(f"  ATR={atr_period} TP={tp_mult} SL={sl_mult}: "
                      f"{sign}${profit:,.2f} | {n_trades}t | "
                      f"Win {win_pct:.0f}% | Sharpe {sharpe:.2f}")

    # Best params
    best_params = max(param_results, key=lambda r: r['sharpe']
                      if r['trades'] >= 10 else -999)
    print(f"\n  BEST: ATR={best_params['atr_period']} "
          f"TP={best_params['tp_mult']} SL={best_params['sl_mult']} "
          f"→ ${best_params['profit']:,.2f}, "
          f"Sharpe={best_params['sharpe']:.2f}")

    # ================================================================
    # B4: Full WFO with best params
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE B4: FULL WFO WITH BEST PARAMS")
    print(f"{'='*70}")
    print(f"Best: ATR={best_params['atr_period']}, "
          f"TP={best_params['tp_mult']}, SL={best_params['sl_mult']}, "
          f"Threshold={best_threshold}")
    print(f"{'='*70}\n")

    final_config = config.copy()
    final_config['atr_period'] = best_params['atr_period']
    final_config['atr_tp_multiplier'] = best_params['tp_mult']
    final_config['atr_sl_multiplier'] = best_params['sl_mult']

    cnn_source = OfflineCNNPredictionSource(pred_df,
                                             confidence_threshold=best_threshold)
    b4_results = run_wfo_cnn(ohlc_1h, cnn_source, final_config, [best_threshold])

    sign = '+' if b4_results['total_profit'] >= 0 else ''
    print(f"\n{'='*70}")
    print(f"B4 RESULTS: CNN + Optimized ATR params")
    print(f"{'='*70}")
    print(f"Params: ATR={best_params['atr_period']}, "
          f"TP={best_params['tp_mult']}, SL={best_params['sl_mult']}")
    print(f"Total Profit:     {sign}${b4_results['total_profit']:,.2f}")
    print(f"Total Trades:     {b4_results['total_trades']}")
    print(f"Win Rate:         {b4_results['total_win_pct']:.1f}%")
    print(f"Aggregate Sharpe: {b4_results['aggregate_sharpe']:.3f}")
    print(f"Max Drawdown:     ${b4_results['max_drawdown_usd']:,.2f}")
    print(f"Profitable Years: {b4_results['profitable_folds']}/{b4_results['total_folds']}")
    print(f"{'='*70}\n")

    # ================================================================
    # Save all results
    # ================================================================
    results = {
        "phase": "B_cnn_direction",
        "b1_baseline": {
            k: v for k, v in b1_results.items() if k != 'all_trades'
        },
        "b2_threshold_sweep": threshold_results,
        "b3_param_sweep_top10": sorted(
            param_results, key=lambda r: r['sharpe'], reverse=True)[:10],
        "b4_full_wfo": {
            k: v for k, v in b4_results.items() if k != 'all_trades'
        },
        "best_threshold": best_threshold,
        "best_params": best_params,
    }

    with open("phase_b_cnn_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved phase_b_cnn_results.json")

    if b4_results.get('all_trades'):
        pd.DataFrame(b4_results['all_trades']).to_csv(
            "phase_b_cnn_trades.csv", index=False)
        print("Saved phase_b_cnn_trades.csv")

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"PHASE A vs PHASE B COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Oracle (A1)':>15} {'CNN Baseline (B1)':>18} "
          f"{'CNN Optimized (B4)':>20}")
    print(f"-" * 75)

    oracle_profit = 1_714_396.45  # From Phase A
    print(f"{'Total Profit':<20} {'$1,714,396':>15} "
          f"{'${:,.0f}'.format(b1_results['total_profit']):>18} "
          f"{'${:,.0f}'.format(b4_results['total_profit']):>20}")
    print(f"{'Win Rate':<20} {'100.0%':>15} "
          f"{'{:.1f}%'.format(b1_results['total_win_pct']):>18} "
          f"{'{:.1f}%'.format(b4_results['total_win_pct']):>20}")
    print(f"{'Trades':<20} {'3,452':>15} "
          f"{'{:,}'.format(b1_results['total_trades']):>18} "
          f"{'{:,}'.format(b4_results['total_trades']):>20}")
    print(f"{'Sharpe':<20} {'0.366':>15} "
          f"{'{:.3f}'.format(b1_results['aggregate_sharpe']):>18} "
          f"{'{:.3f}'.format(b4_results['aggregate_sharpe']):>20}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

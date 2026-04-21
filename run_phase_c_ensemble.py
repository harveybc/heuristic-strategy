#!/usr/bin/env python3
"""
Phase C: Ensemble Voting — Multiple Direction Models

Loads CNN, ANN, and Logistic direction models (long + short),
runs inference with each, combines predictions via:
  C1: Individual model baselines
  C2: Ensemble methods (avg prob, majority vote, weighted avg)
  C3: Best ensemble threshold + ATR param sweep
  C4: Full 14yr WFO with best ensemble config
  C5: Comparison table (A vs B vs C)
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

# ================================================================
# Paths
# ================================================================
OHLC_FILE = "tests/data/eurusd_hour_2005_2020.csv"
NORM_CONFIG = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/data_downsampled/"
    "phase_1_c/normalization_config_c.json"
)

RESULTS_DIR = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/results/phase_1c_direction/"
)

# Model definitions: (label, keras_file, meta_file, add_window_stats)
LONG_MODELS = [
    ("cnn_long", "phase_1c_direction_cnn_direction_long_model.keras",
     "phase_1c_direction_cnn_direction_long_model_metadata.json", True),
    ("ann_long", "phase_1c_direction_ann_direction_long_model.keras",
     "phase_1c_direction_ann_direction_long_model_metadata.json", False),
    ("logistic_long", "phase_1c_direction_logistic_direction_long_model.keras",
     "phase_1c_direction_logistic_direction_long_model_metadata.json", False),
]

SHORT_MODELS = [
    ("cnn_short", "phase_1c_direction_cnn_direction_short_model.keras",
     "phase_1c_direction_cnn_direction_short_model_metadata.json", True),
    ("ann_short", "phase_1c_direction_ann_direction_short_model.keras",
     "phase_1c_direction_ann_direction_short_model_metadata.json", False),
    ("logistic_short", "phase_1c_direction_logistic_direction_short_model.keras",
     "phase_1c_direction_logistic_direction_short_model_metadata.json", False),
]

# Test-set AUC_ROC scores for weighting (from inference results)
MODEL_AUC = {
    "cnn_long": 0.6114,
    "ann_long": 0.6188,
    "logistic_long": 0.6115,
    "cnn_short": 0.6699,
    "ann_short": 0.6577,
    "logistic_short": 0.6790,
}

# Feature computation constants
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
# Step 1: Feature engineering (reused from Phase B)
# ================================================================

def resample_to_4h(df_1h):
    """Resample hourly OHLC to 4H matching training pipeline boundaries."""
    ohlc_rules = {
        'OPEN': 'first', 'HIGH': 'max', 'LOW': 'min', 'CLOSE': 'last',
    }
    df_4h = df_1h.resample('4h', offset='1h').agg(ohlc_rules).dropna()
    return df_4h


def compute_features(df_4h):
    """Compute 22 technical indicator features from 4H OHLC."""
    import pandas_ta as ta

    df = df_4h.copy()
    df['DATE_TIME'] = df.index

    high = df['HIGH'].astype(float)
    low = df['LOW'].astype(float)
    close = df['CLOSE'].astype(float)

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

    bbands = ta.bbands(close, length=20, std=2.0)
    bb_upper = bbands['BBU_20_2.0']
    bb_lower = bbands['BBL_20_2.0']
    df['BB_Width'] = bb_upper - bb_lower

    df['CCI'] = ta.cci(high, low, close, length=20)
    df['WilliamsR'] = ta.willr(high, low, close, length=14)
    df['ROC'] = ta.roc(close, length=10)

    atr_in_pips = df['ATR'] / PIP_COST
    df['ATR_ratio'] = TP_PIPS / atr_in_pips.replace(0, np.nan)

    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df['BB_position'] = (close - bb_lower) / bb_range

    typical_price = (high + low + close) / 3.0
    df['rolling_std_24'] = typical_price.rolling(window=ROLLING_WINDOW).std()
    rolling_ema = typical_price.ewm(span=ROLLING_WINDOW, adjust=False).mean()
    df['price_minus_ema'] = typical_price - rolling_ema

    dt = pd.to_datetime(df['DATE_TIME'])
    df['hod_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hod_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

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
# Step 2: Model loading & inference
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


def positional_encoding(window, channels):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    import tensorflow as tf
    position = np.arange(window)[:, np.newaxis]
    dims = np.arange(channels)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(channels))
    angle_rads = position * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, :], tf.float32)


# Model architecture configurations (from inference configs + optimization params)
MODEL_CONFIGS = {
    "cnn_long": {
        "arch": "cnn", "window": 71, "channels": 28,
        "positional_encoding": False,
        "intermediate_layers": 2, "initial_layer_size": 64,
        "layer_size_divisor": 16, "head_layers": 2,
        "l2_reg": 1e-7, "activation": "elu",
    },
    "cnn_short": {
        "arch": "cnn", "window": 110, "channels": 28,
        "positional_encoding": True,
        "intermediate_layers": 5, "initial_layer_size": 54,
        "layer_size_divisor": 8, "head_layers": 1,
        "l2_reg": 0.0006623696070059423, "activation": "elu",
    },
    "ann_long": {
        "arch": "ann", "window": 72, "channels": 22,
        "positional_encoding": True,
        "hidden_units": 128, "num_hidden_layers": 2,
        "dropout_rate": 0.1, "activation": "elu",
    },
    "ann_short": {
        "arch": "ann", "window": 72, "channels": 22,
        "positional_encoding": True,
        "hidden_units": 128, "num_hidden_layers": 2,
        "dropout_rate": 0.1, "activation": "elu",
    },
    "logistic_long": {
        "arch": "logistic", "window": 72, "channels": 22,
        "positional_encoding": True,
        "l2_reg": 1e-6,
    },
    "logistic_short": {
        "arch": "logistic", "window": 72, "channels": 22,
        "positional_encoding": True,
        "l2_reg": 1e-6,
    },
}


def _build_ann_model(cfg):
    """Rebuild ANN direction model architecture."""
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Lambda, Flatten, Dense, Dropout
    from tensorflow.keras.models import Model

    w, c = cfg["window"], cfg["channels"]
    inputs = Input(shape=(w, c), name="input_layer")
    if cfg.get("positional_encoding", False):
        pe = positional_encoding(w, c)
        x = Lambda(lambda t, pe=pe: t + pe,
                    name="add_positional_encoding")(inputs)
    else:
        x = inputs
    x = Flatten(name="flatten_inputs")(x)
    for i in range(cfg.get("num_hidden_layers", 2)):
        x = Dense(cfg.get("hidden_units", 128),
                  activation=cfg.get("activation", "elu"),
                  name=f"shared_dense_{i}")(x)
        dr = cfg.get("dropout_rate", 0.0)
        if dr > 0:
            x = Dropout(dr, name=f"shared_dropout_{i}")(x)
    output = Dense(1, activation="sigmoid", name="output_horizon_1")(x)
    return Model(inputs=inputs, outputs=[output], name="DirectionANN")


def _build_logistic_model(cfg):
    """Rebuild Logistic direction model architecture."""
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Lambda, Flatten, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2

    w, c = cfg["window"], cfg["channels"]
    inputs = Input(shape=(w, c), name="input_layer")
    if cfg.get("positional_encoding", False):
        pe = positional_encoding(w, c)
        x = Lambda(lambda t, pe=pe: t + pe,
                    name="add_positional_encoding")(inputs)
    else:
        x = inputs
    x = Flatten(name="flatten_inputs")(x)
    l2_val = cfg.get("l2_reg", 0.0)
    reg = l2(l2_val) if l2_val > 0 else None
    output = Dense(1, activation="sigmoid", name="output_horizon_1",
                   kernel_regularizer=reg)(x)
    return Model(inputs=inputs, outputs=[output], name="DirectionLogistic")


def _build_cnn_model(cfg):
    """Rebuild CNN direction model architecture."""
    import tensorflow as tf
    from tensorflow.keras.layers import (Input, Lambda, Conv1D, Dense,
                                          Bidirectional, LSTM)
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2

    w, c = cfg["window"], cfg["channels"]
    act = cfg.get("activation", "elu")
    l2_val = cfg.get("l2_reg", 1e-4)
    init_size = cfg.get("initial_layer_size", 128)
    divisor = cfg.get("layer_size_divisor", 2)
    n_inter = max(1, int(cfg.get("intermediate_layers", 1)))
    n_head = max(1, int(cfg.get("head_layers", 1)))

    inputs = Input(shape=(w, c), name="input_layer")
    if cfg.get("positional_encoding", False):
        pe = positional_encoding(w, c)
        x = Lambda(lambda t, pe=pe: t + pe,
                    name="add_positional_encoding")(inputs)
    else:
        x = inputs

    sizes = [init_size] + [
        max(8, init_size // (divisor ** i)) for i in range(1, n_inter)]
    for i, filt in enumerate(sizes):
        x = Conv1D(filters=filt, kernel_size=3, strides=2,
                   padding="causal", activation=act,
                   kernel_regularizer=l2(l2_val),
                   name=f"conv_{i+1}")(x)

    last_root = sizes[-1]
    base_head = max(8, last_root // 2)
    head_sizes = [base_head] + [
        max(8, base_head // (divisor ** i)) for i in range(1, n_head)]
    for j, fj in enumerate(head_sizes):
        x = Conv1D(filters=fj, kernel_size=3, strides=2,
                   padding="same", activation=act,
                   kernel_regularizer=l2(l2_val),
                   name=f"head_conv{j+1}")(x)

    last_head = head_sizes[-1]
    lstm_units = max(8, last_head // 2)
    x = Bidirectional(
        LSTM(max(1, lstm_units // 2), return_sequences=False),
        name="bilstm")(x)

    output = Dense(1, activation="sigmoid", name="output_horizon_1")(x)
    return Model(inputs=inputs, outputs=[output], name="DirectionCNN")


def load_model_rebuild(model_path, label):
    """Load model by rebuilding architecture + loading weights from .keras zip."""
    import tensorflow as tf
    import zipfile
    import tempfile

    cfg = MODEL_CONFIGS.get(label)
    if not cfg:
        raise ValueError(f"No config for model {label}")

    needs_rebuild = cfg.get("positional_encoding", False)

    if not needs_rebuild:
        # No Lambda layer — can use load_model directly (with patches)
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(model_path, 'r') as zf:
                zf.extractall(tmpdir)
            config_path = os.path.join(tmpdir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    mc = json.load(f)
                _strip_quantization_config(mc)
                with open(config_path, 'w') as f:
                    json.dump(mc, f)
            patched_path = os.path.join(tmpdir, 'patched.keras')
            with zipfile.ZipFile(patched_path, 'w') as zf_out:
                for root, dirs, files in os.walk(tmpdir):
                    for fname in files:
                        if fname == 'patched.keras':
                            continue
                        fpath = os.path.join(root, fname)
                        arcname = os.path.relpath(fpath, tmpdir)
                        zf_out.write(fpath, arcname)
            model = tf.keras.models.load_model(
                patched_path, compile=False, safe_mode=False)
        print(f"  [{label}] Loaded via load_model (no Lambda)")
    else:
        # Has Lambda — rebuild architecture, load weights only
        arch = cfg["arch"]
        if arch == "ann":
            model = _build_ann_model(cfg)
        elif arch == "logistic":
            model = _build_logistic_model(cfg)
        elif arch == "cnn":
            model = _build_cnn_model(cfg)
        else:
            raise ValueError(f"Unknown arch: {arch}")

        # Extract weights from .keras zip
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(model_path, 'r') as zf:
                zf.extractall(tmpdir)
            weights_path = os.path.join(tmpdir, "model.weights.h5")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"No model.weights.h5 in {model_path}")
            model.load_weights(weights_path)
        print(f"  [{label}] Rebuilt {arch} + loaded weights "
              f"(window={cfg['window']}, feat={cfg['channels']})")

    return model, cfg["window"]


def compute_window_stats(window, target_col_idx, periods=(12, 48)):
    """Compute per-window statistics matching predictor's add_window_stats."""
    window_size = window.shape[0]
    target_vals = window[:, target_col_idx]
    stats = []
    for p in periods:
        span = min(p, window_size)
        tail = target_vals[-span:]
        stats.append(np.std(tail, dtype=np.float32))
        alpha = 2.0 / (span + 1.0)
        ema = tail[0]
        for t in range(1, len(tail)):
            ema = alpha * tail[t] + (1.0 - alpha) * ema
        stats.append(np.float32(ema))
        stats.append(np.float32(target_vals[-1] - ema))
    return np.array(stats, dtype=np.float32)


def run_inference(model, df_norm, window_size, label,
                  add_window_stats=False, target_col='ATR'):
    """Run model inference on all bars, returning P(up) for each."""
    import tensorflow as tf

    feat_cols = [c for c in FEATURE_COLS if c in df_norm.columns]
    features = df_norm[feat_cols].values.astype(np.float32)
    n_bars = len(features)
    n_features = len(feat_cols)

    target_col_idx = (feat_cols.index(target_col)
                      if target_col in feat_cols else 0)
    n_stats = 6 if add_window_stats else 0
    total_features = n_features + n_stats

    predictions = np.full(n_bars, np.nan)

    print(f"  [{label}] Inference: {n_bars} bars "
          f"(window={window_size}, feat={n_features}+{n_stats}stats)")
    t0 = time.time()

    batch_size = 256
    valid_indices = list(range(window_size - 1, n_bars))
    n_valid = len(valid_indices)

    for batch_start in range(0, n_valid, batch_size):
        batch_end = min(batch_start + batch_size, n_valid)
        batch_indices = valid_indices[batch_start:batch_end]

        X_batch = np.zeros((len(batch_indices), window_size, total_features),
                           dtype=np.float32)
        for j, idx in enumerate(batch_indices):
            start = idx - window_size + 1
            window = features[start:idx + 1]
            X_batch[j, :, :n_features] = window

            if add_window_stats:
                stats = compute_window_stats(window, target_col_idx)
                X_batch[j, :, n_features:] = stats[np.newaxis, :]

        preds = model.predict(X_batch, verbose=0, batch_size=batch_size)
        preds = np.asarray(preds).flatten()

        for j, idx in enumerate(batch_indices):
            predictions[idx] = preds[j]

    elapsed = time.time() - t0
    valid_count = np.sum(~np.isnan(predictions))
    mean_p = np.nanmean(predictions)
    print(f"  [{label}] Done: {valid_count} preds in {elapsed:.1f}s, "
          f"mean P(up)={mean_p:.3f}")
    return predictions


def load_all_models_and_predict(df_norm):
    """Load all available models and generate predictions."""
    long_preds = {}
    short_preds = {}
    loaded_long = []
    loaded_short = []

    # Load long models
    print("\n--- Loading LONG direction models ---")
    for label, keras_file, meta_file, add_ws in LONG_MODELS:
        model_path = os.path.join(RESULTS_DIR, keras_file)
        try:
            model, window = load_model_rebuild(model_path, label)
            preds = run_inference(model, df_norm, window, label,
                                 add_window_stats=add_ws)
            long_preds[label] = preds
            loaded_long.append(label)
            del model
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Load short models
    print("\n--- Loading SHORT direction models ---")
    for label, keras_file, meta_file, add_ws in SHORT_MODELS:
        model_path = os.path.join(RESULTS_DIR, keras_file)
        try:
            model, window = load_model_rebuild(model_path, label)
            preds = run_inference(model, df_norm, window, label,
                                 add_window_stats=add_ws)
            short_preds[label] = preds
            loaded_short.append(label)
            del model
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Loaded {len(loaded_long)} long models: {loaded_long}")
    print(f"  Loaded {len(loaded_short)} short models: {loaded_short}")

    return long_preds, short_preds, loaded_long, loaded_short


# ================================================================
# Step 3: Ensemble combination methods
# ================================================================

def ensemble_average(preds_dict, labels):
    """Simple average of P(up) across models."""
    arrs = [preds_dict[l] for l in labels]
    stacked = np.stack(arrs, axis=0)
    return np.nanmean(stacked, axis=0)


def ensemble_weighted_avg(preds_dict, labels, weights_dict):
    """Weighted average of P(up) using AUC-based weights."""
    arrs = [preds_dict[l] for l in labels]
    weights = np.array([weights_dict.get(l, 0.5) for l in labels])
    # Normalize weights above 0.5 baseline (chance = 0.5 AUC)
    adj_weights = weights - 0.5
    adj_weights = np.maximum(adj_weights, 0.01)
    adj_weights = adj_weights / adj_weights.sum()

    stacked = np.stack(arrs, axis=0)
    # Weighted mean: sum(w_i * pred_i) for each bar
    result = np.zeros(stacked.shape[1], dtype=np.float32)
    for i, w in enumerate(adj_weights):
        result += w * stacked[i]
    return result


def ensemble_majority_vote(preds_dict, labels, threshold=0.5):
    """Majority vote: fraction of models agreeing P(up) >= threshold."""
    arrs = [preds_dict[l] for l in labels]
    stacked = np.stack(arrs, axis=0)
    votes = (stacked >= threshold).astype(np.float32)
    return np.nanmean(votes, axis=0)


# ================================================================
# Step 4: Prediction source for backtest
# ================================================================

class EnsemblePredictionSource:
    """Maps hourly timestamps to 4H ensemble predictions."""

    def __init__(self, pred_df, confidence_threshold=0.55):
        self.pred_df = pred_df
        self.threshold = confidence_threshold

    def _get_nearest(self, dt_hour):
        ts = pd.Timestamp(dt_hour)
        idx = self.pred_df.index.get_indexer([ts], method='ffill')[0]
        if idx < 0:
            idx = 0
        return self.pred_df.iloc[idx]

    def get_entry_prediction(self, dt_hour, tp_pips=0, sl_pips=0,
                             spread_pips=0, commission_per_lot=0,
                             slippage_pips=0):
        try:
            row = self._get_nearest(dt_hour)
            p_up = float(row['p_up_long'])
            buy_signal = 1.0 if p_up >= self.threshold else 0.0
            sell_signal = 1.0 if (1.0 - p_up) >= self.threshold else 0.0
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
            row = self._get_nearest(dt_hour)
            p_up = float(row['p_up_short'])
            if direction == 'buy':
                should_exit = (1.0 - p_up) >= self.threshold
            else:
                should_exit = p_up >= self.threshold
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
# Step 5: Backtest infrastructure (from Phase B)
# ================================================================

def run_backtest(base_data, source, config, label=""):
    """Run direction_atr strategy with prediction source."""
    import backtrader as bt
    from app.plugins.plugin_direction_atr import Plugin as DirectionATRPlugin

    plugin = DirectionATRPlugin()
    original_init = plugin.DirectionATRStrategy.__init__

    def patched_init(self_strat, *args, **kwargs):
        bt.Strategy.__init__(self_strat)
        self_strat._pred_source = source
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


def run_wfo(base_data_1h, source, config):
    """Run 14-year walk-forward test."""
    fold_results = []
    all_trades = []

    for test_year in range(2006, 2020):
        year_mask = base_data_1h.index.year == test_year
        year_data = base_data_1h[year_mask].copy()
        if len(year_data) < 100:
            continue

        t0 = time.time()
        profit, stats, trades = run_backtest(
            year_data, source, config, label=f"Y{test_year}")
        elapsed = time.time() - t0

        n_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_pct = (wins / n_trades * 100) if n_trades > 0 else 0
        pnls = [t['pnl'] for t in trades]
        sharpe = ((np.mean(pnls) / (np.std(pnls) + 1e-10))
                  if n_trades > 1 else 0)

        fold_results.append({
            "year": test_year, "profit": profit, "trades": n_trades,
            "win_pct": win_pct, "sharpe": sharpe, "elapsed": elapsed,
        })
        for t in trades:
            t["year"] = test_year
        all_trades.extend(trades)

        sign = '+' if profit >= 0 else ''
        print(f"  {test_year}: {sign}${profit:,.2f} | "
              f"{n_trades} trades | Win {win_pct:.0f}% | "
              f"Sharpe {sharpe:.2f} | {elapsed:.0f}s")

    total_profit = sum(f["profit"] for f in fold_results)
    total_trades = sum(f["trades"] for f in fold_results)
    total_wins = sum(1 for t in all_trades if t['pnl'] > 0)
    total_win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0
    all_pnls = [t['pnl'] for t in all_trades]
    agg_sharpe = ((np.mean(all_pnls) / (np.std(all_pnls) + 1e-10))
                  if len(all_pnls) > 1 else 0)
    profitable_folds = sum(1 for f in fold_results if f["profit"] > 0)

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


def print_results(label, results):
    sign = '+' if results['total_profit'] >= 0 else ''
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Total Profit:     {sign}${results['total_profit']:,.2f}")
    print(f"Total Trades:     {results['total_trades']}")
    print(f"Win Rate:         {results['total_win_pct']:.1f}%")
    print(f"Aggregate Sharpe: {results['aggregate_sharpe']:.3f}")
    print(f"Max Drawdown:     ${results['max_drawdown_usd']:,.2f}")
    print(f"Profitable Years: {results['profitable_folds']}/{results['total_folds']}")
    print(f"{'='*70}")


def quick_sample_test(sample_data, pred_df, config, label, threshold=0.55):
    """Run a quick 5yr sample test, return (profit, trades, win%, sharpe)."""
    source = EnsemblePredictionSource(pred_df, confidence_threshold=threshold)
    profit, _, trades = run_backtest(sample_data, source, config, label=label)
    n_trades = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    win_pct = (wins / n_trades * 100) if n_trades > 0 else 0
    pnls = [t['pnl'] for t in trades]
    sharpe = ((np.mean(pnls) / (np.std(pnls) + 1e-10))
              if n_trades > 1 else 0)
    return profit, n_trades, win_pct, sharpe


# ================================================================
# MAIN
# ================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    from app.data_handler import load_csv

    # Load data
    print(f"Loading dataset: {OHLC_FILE}")
    ohlc_1h = load_csv(OHLC_FILE, headers=True)
    print(f"Loaded: {ohlc_1h.shape[0]} bars, "
          f"{ohlc_1h.index.min()} to {ohlc_1h.index.max()}")

    with open(NORM_CONFIG) as f:
        norm_config = json.load(f)

    # Feature engineering
    print("\nResampling to 4H...")
    df_4h = resample_to_4h(ohlc_1h)
    print(f"  4H bars: {len(df_4h)}")

    print("Computing 22 technical features...")
    df_feat = compute_features(df_4h)
    print(f"  After warmup: {len(df_feat)} bars")

    print("Normalizing...")
    df_norm = normalize_features(df_feat, norm_config)

    # ================================================================
    # Load all models and generate predictions
    # ================================================================
    print(f"\n{'='*70}")
    print(f"LOADING ALL MODELS AND GENERATING PREDICTIONS")
    print(f"{'='*70}")

    long_preds, short_preds, loaded_long, loaded_short = \
        load_all_models_and_predict(df_norm)

    if not loaded_long:
        print("ERROR: No long models loaded. Cannot continue.")
        return

    # ================================================================
    # Build ensemble prediction DataFrames
    # ================================================================
    print(f"\n{'='*70}")
    print(f"BUILDING ENSEMBLE PREDICTIONS")
    print(f"{'='*70}")

    n_bars = len(df_feat)
    close_vals = df_feat['CLOSE'].values
    feat_index = df_feat.index

    # Individual model predictions
    individual_dfs = {}
    for label in loaded_long:
        p_long = long_preds[label]
        # For short, use matching architecture if available, else fallback
        short_label = label.replace('_long', '_short')
        if short_label in short_preds:
            p_short = short_preds[short_label]
        elif loaded_short:
            p_short = short_preds[loaded_short[0]]
            print(f"  {label}: no matching short, using {loaded_short[0]}")
        else:
            p_short = p_long
            print(f"  {label}: no short models, using long for exit")

        df = pd.DataFrame({
            'p_up_long': p_long,
            'p_up_short': p_short,
            'CLOSE': close_vals,
        }, index=feat_index)
        individual_dfs[label] = df.dropna(subset=['p_up_long'])
        print(f"  {label}: {len(individual_dfs[label])} valid predictions, "
              f"P(up) mean={np.nanmean(p_long):.3f}")

    # Ensemble predictions
    ensemble_dfs = {}

    # E1: Simple average
    avg_long = ensemble_average(long_preds, loaded_long)
    if loaded_short:
        avg_short = ensemble_average(short_preds, loaded_short)
    else:
        avg_short = avg_long
    ensemble_dfs['avg'] = pd.DataFrame({
        'p_up_long': avg_long, 'p_up_short': avg_short, 'CLOSE': close_vals,
    }, index=feat_index).dropna(subset=['p_up_long'])
    print(f"\n  Ensemble AVG: {len(ensemble_dfs['avg'])} preds, "
          f"P(up) mean={np.nanmean(avg_long):.3f}")

    # E2: Weighted average (by AUC)
    wavg_long = ensemble_weighted_avg(long_preds, loaded_long, MODEL_AUC)
    if loaded_short:
        wavg_short = ensemble_weighted_avg(short_preds, loaded_short, MODEL_AUC)
    else:
        wavg_short = wavg_long
    ensemble_dfs['wavg'] = pd.DataFrame({
        'p_up_long': wavg_long, 'p_up_short': wavg_short, 'CLOSE': close_vals,
    }, index=feat_index).dropna(subset=['p_up_long'])
    print(f"  Ensemble WAVG: {len(ensemble_dfs['wavg'])} preds, "
          f"P(up) mean={np.nanmean(wavg_long):.3f}")

    # E3: Majority vote (fraction of models saying up at 0.50)
    mvote_long = ensemble_majority_vote(long_preds, loaded_long, threshold=0.50)
    if loaded_short:
        mvote_short = ensemble_majority_vote(
            short_preds, loaded_short, threshold=0.50)
    else:
        mvote_short = mvote_long
    ensemble_dfs['vote'] = pd.DataFrame({
        'p_up_long': mvote_long, 'p_up_short': mvote_short,
        'CLOSE': close_vals,
    }, index=feat_index).dropna(subset=['p_up_long'])
    print(f"  Ensemble VOTE: {len(ensemble_dfs['vote'])} preds, "
          f"P(up) mean={np.nanmean(mvote_long):.3f}")

    # E4: Unanimity vote (all must agree)
    # For N models: unanimous buy when all P(up) >= 0.5, i.e., vote_frac = 1.0
    # Use high threshold on vote fraction
    # Already captured by using threshold=1.0 on vote predictions

    # Cache all predictions
    cache = {}
    for label, df in individual_dfs.items():
        cache[label] = df
    for label, df in ensemble_dfs.items():
        cache[f"ens_{label}"] = df
    cache_file = "ensemble_predictions_15yr.csv"
    # Save the main ensemble (avg) to CSV for reference
    ensemble_dfs['avg'].to_csv(cache_file)
    print(f"\n  Cached ensemble predictions to {cache_file}")

    # ================================================================
    # C1: Individual model baselines (5yr sample, best B params)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE C1: INDIVIDUAL MODEL BASELINES (2012-2016)")
    print(f"{'='*70}")
    print(f"Using Phase B best params: ATR=10, TP=3.0, SL=3.0")
    print(f"{'='*70}\n")

    config = {
        "prediction_source": "API",
        "pp_api_url": "http://offline",
        "pp_timeout": 999,
        "headers": True,
        "disable_multiprocessing": True,
        "atr_period": 10,
        "atr_tp_multiplier": 3.0,
        "atr_sl_multiplier": 3.0,
        "spread_pips": 15.0,
        "commission_per_lot": 7.0,
        "slippage_pips": 5.0,
    }

    year_mask = ((ohlc_1h.index.year >= 2012) &
                 (ohlc_1h.index.year <= 2016))
    sample_data = ohlc_1h[year_mask].copy()

    c1_results = {}
    # Test each individual model
    for label in list(individual_dfs.keys()):
        for thresh in [0.55, 0.70]:
            key = f"{label}_t{thresh}"
            profit, trades, win_pct, sharpe = quick_sample_test(
                sample_data, individual_dfs[label], config,
                label=key, threshold=thresh)
            c1_results[key] = {
                "profit": profit, "trades": trades,
                "win_pct": win_pct, "sharpe": sharpe,
            }
            sign = '+' if profit >= 0 else ''
            print(f"  {key}: {sign}${profit:,.2f} | "
                  f"{trades}t | Win {win_pct:.0f}% | Sharpe {sharpe:.3f}")

    # Test each ensemble method
    for ens_label in ensemble_dfs:
        for thresh in [0.55, 0.60, 0.65, 0.70]:
            key = f"ens_{ens_label}_t{thresh}"
            profit, trades, win_pct, sharpe = quick_sample_test(
                sample_data, ensemble_dfs[ens_label], config,
                label=key, threshold=thresh)
            c1_results[key] = {
                "profit": profit, "trades": trades,
                "win_pct": win_pct, "sharpe": sharpe,
            }
            sign = '+' if profit >= 0 else ''
            print(f"  {key}: {sign}${profit:,.2f} | "
                  f"{trades}t | Win {win_pct:.0f}% | Sharpe {sharpe:.3f}")

    # Find best ensemble method + threshold
    best_c1 = max(c1_results.items(),
                  key=lambda x: x[1]['sharpe'] if x[1]['trades'] >= 10 else -999)
    print(f"\n  BEST C1: {best_c1[0]} → ${best_c1[1]['profit']:,.2f}, "
          f"Sharpe={best_c1[1]['sharpe']:.3f}")

    # ================================================================
    # C2: Threshold sweep on best ensemble method
    # ================================================================
    # Parse best ensemble method
    best_c1_key = best_c1[0]
    # Determine which pred_df to use
    if best_c1_key.startswith("ens_"):
        parts = best_c1_key.split('_t')
        ens_method = parts[0].replace('ens_', '')
        best_pred_df = ensemble_dfs[ens_method]
        best_method_label = f"ensemble_{ens_method}"
    else:
        parts = best_c1_key.split('_t')
        model_key = parts[0]
        best_pred_df = individual_dfs[model_key]
        best_method_label = model_key

    print(f"\n{'='*70}")
    print(f"PHASE C2: FINE THRESHOLD SWEEP ({best_method_label})")
    print(f"{'='*70}\n")

    fine_thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65,
                       0.68, 0.70, 0.75, 0.80]
    c2_results = []
    for thresh in fine_thresholds:
        profit, trades, win_pct, sharpe = quick_sample_test(
            sample_data, best_pred_df, config,
            label=f"t={thresh}", threshold=thresh)
        c2_results.append({
            "threshold": thresh, "profit": profit, "trades": trades,
            "win_pct": win_pct, "sharpe": sharpe,
        })
        sign = '+' if profit >= 0 else ''
        print(f"  thresh={thresh:.2f}: {sign}${profit:,.2f} | "
              f"{trades}t | Win {win_pct:.0f}% | Sharpe {sharpe:.3f}")

    best_thresh_row = max(c2_results,
                          key=lambda r: r['sharpe'] if r['trades'] >= 10 else -999)
    best_threshold = best_thresh_row['threshold']
    print(f"\n  BEST threshold: {best_threshold} → "
          f"${best_thresh_row['profit']:,.2f}, "
          f"Sharpe={best_thresh_row['sharpe']:.3f}")

    # ================================================================
    # C3: ATR param sweep with best ensemble + threshold
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE C3: ATR PARAMETER SWEEP ({best_method_label}, "
          f"thresh={best_threshold})")
    print(f"{'='*70}\n")

    atr_periods = [10, 14, 21]
    tp_mults = [1.5, 2.0, 2.5, 3.0, 4.0]
    sl_mults = [0.5, 1.0, 1.5, 2.0, 3.0]

    c3_results = []
    for atr_period in atr_periods:
        for tp_mult in tp_mults:
            for sl_mult in sl_mults:
                test_config = config.copy()
                test_config['atr_period'] = atr_period
                test_config['atr_tp_multiplier'] = tp_mult
                test_config['atr_sl_multiplier'] = sl_mult

                profit, trades, win_pct, sharpe = quick_sample_test(
                    sample_data, best_pred_df, test_config,
                    label=f"atr={atr_period}/tp={tp_mult}/sl={sl_mult}",
                    threshold=best_threshold)

                c3_results.append({
                    "atr_period": atr_period, "tp_mult": tp_mult,
                    "sl_mult": sl_mult, "profit": profit, "trades": trades,
                    "win_pct": win_pct, "sharpe": sharpe,
                })
                sign = '+' if profit >= 0 else ''
                print(f"  ATR={atr_period} TP={tp_mult} SL={sl_mult}: "
                      f"{sign}${profit:,.2f} | {trades}t | "
                      f"Win {win_pct:.0f}% | Sharpe {sharpe:.3f}")

    best_params = max(c3_results,
                      key=lambda r: r['sharpe'] if r['trades'] >= 10 else -999)
    print(f"\n  BEST: ATR={best_params['atr_period']} "
          f"TP={best_params['tp_mult']} SL={best_params['sl_mult']} → "
          f"${best_params['profit']:,.2f}, Sharpe={best_params['sharpe']:.3f}")

    # ================================================================
    # C4: Full 14yr WFO with best ensemble config
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE C4: FULL WFO WITH BEST ENSEMBLE CONFIG")
    print(f"{'='*70}")
    print(f"Method: {best_method_label}")
    print(f"Threshold: {best_threshold}")
    print(f"ATR={best_params['atr_period']}, "
          f"TP={best_params['tp_mult']}, SL={best_params['sl_mult']}")
    print(f"{'='*70}\n")

    final_config = config.copy()
    final_config['atr_period'] = best_params['atr_period']
    final_config['atr_tp_multiplier'] = best_params['tp_mult']
    final_config['atr_sl_multiplier'] = best_params['sl_mult']

    source = EnsemblePredictionSource(
        best_pred_df, confidence_threshold=best_threshold)
    c4_results = run_wfo(ohlc_1h, source, final_config)
    print_results("C4: ENSEMBLE FULL WFO", c4_results)

    # Also run with Phase B best params for direct comparison
    print(f"\n--- Phase B reference (CNN only, ATR=10 TP=3.0 SL=3.0 t=0.70) ---")
    b_ref_config = config.copy()
    b_ref_config['atr_period'] = 10
    b_ref_config['atr_tp_multiplier'] = 3.0
    b_ref_config['atr_sl_multiplier'] = 3.0

    # Use CNN-only predictions for reference
    if 'cnn_long' in individual_dfs:
        b_source = EnsemblePredictionSource(
            individual_dfs['cnn_long'], confidence_threshold=0.70)
        b_ref_results = run_wfo(ohlc_1h, b_source, b_ref_config)
        print_results("B4 Reference: CNN-only", b_ref_results)
    else:
        b_ref_results = None

    # ================================================================
    # C5: Summary comparison
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE A vs B vs C COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Oracle (A)':>12} {'CNN (B4)':>12} "
          f"{'Ensemble (C4)':>15}")
    print(f"-" * 62)
    print(f"{'Total Profit':<20} {'$1,714,396':>12} "
          f"{'${:,.0f}'.format(-4585):>12} "
          f"{'${:,.0f}'.format(c4_results['total_profit']):>15}")
    print(f"{'Win Rate':<20} {'100.0%':>12} {'46.3%':>12} "
          f"{'{:.1f}%'.format(c4_results['total_win_pct']):>15}")
    print(f"{'Trades':<20} {'3,452':>12} {'2,745':>12} "
          f"{'{:,}'.format(c4_results['total_trades']):>15}")
    print(f"{'Sharpe':<20} {'0.366':>12} {'-0.043':>12} "
          f"{'{:.3f}'.format(c4_results['aggregate_sharpe']):>15}")
    print(f"{'Max Drawdown':<20} {'$8,206':>12} {'$8,389':>12} "
          f"{'${:,.0f}'.format(c4_results['max_drawdown_usd']):>15}")
    print(f"{'Prof. Years':<20} {'14/14':>12} {'6/14':>12} "
          f"{'{}/{}'.format(c4_results['profitable_folds'], c4_results['total_folds']):>15}")
    print(f"{'='*70}")

    # ================================================================
    # Save results
    # ================================================================
    save_data = {
        "phase": "C_ensemble",
        "method": best_method_label,
        "best_threshold": best_threshold,
        "best_params": best_params,
        "c1_individual_baselines": c1_results,
        "c2_threshold_sweep": c2_results,
        "c3_param_sweep_top10": sorted(
            c3_results, key=lambda r: r['sharpe'], reverse=True)[:10],
        "c4_full_wfo": {
            k: v for k, v in c4_results.items() if k != 'all_trades'
        },
        "models_loaded_long": loaded_long,
        "models_loaded_short": loaded_short,
    }

    with open("phase_c_ensemble_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\nSaved phase_c_ensemble_results.json")

    if c4_results.get('all_trades'):
        pd.DataFrame(c4_results['all_trades']).to_csv(
            "phase_c_ensemble_trades.csv", index=False)
        print("Saved phase_c_ensemble_trades.csv")


if __name__ == "__main__":
    main()

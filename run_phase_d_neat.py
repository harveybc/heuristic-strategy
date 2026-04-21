#!/usr/bin/env python3
"""
Phase D: NEAT-Enhanced Direction Prediction

Runs extended NEAT hyperparameter optimization for the best direction
prediction architecture (ANN Long from Phase C), then tests the evolved
champion against the heuristic strategy using the same backtest pipeline.

Stages:
  D0: Create reduced-gen NEAT configs (practical budget)
  D1: Run NEAT optimization via predictor subprocess (ANN long + short)
  D2: Load NEAT champion models + generate 15yr predictions
  D3: Threshold sweep (in-sample 2012-2016)
  D4: ATR parameter sweep with best threshold
  D5: Full 14yr WFO with NEAT champion config
  D6: A vs B vs C vs D comparison table

Usage:
  python run_phase_d_neat.py                   # Full run (NEAT + eval)
  python run_phase_d_neat.py --skip-neat       # Skip NEAT, use existing models
  python run_phase_d_neat.py --neat-gens 5     # 5 gens per stage (default)
  python run_phase_d_neat.py --neat-pop 15     # Population size (default 15)
"""
import sys
import os
import json
import time
import shutil
import subprocess
import tempfile
import argparse
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
PREDICTOR_DIR = os.path.expanduser("~/Documents/GitHub/predictor")
PREDICTOR_APP_DIR = os.path.join(PREDICTOR_DIR, "app")
RESULTS_DIR = os.path.join(
    PREDICTOR_DIR, "examples/results/phase_1c_direction")
OHLC_FILE = "tests/data/eurusd_hour_2005_2020.csv"
NORM_CONFIG = os.path.join(
    PREDICTOR_DIR,
    "examples/data_downsampled/phase_1_c/normalization_config_c.json",
)

# Original NEAT configs (will be modified for Phase D)
NEAT_CONFIG_TEMPLATE_LONG = os.path.join(
    PREDICTOR_DIR,
    "examples/config/phase_1c_direction/optimization/"
    "phase_1c_direction_ann_direction_long_1d_optimization_config.json",
)
NEAT_CONFIG_TEMPLATE_SHORT = os.path.join(
    PREDICTOR_DIR,
    "examples/config/phase_1c_direction/optimization/"
    "phase_1c_direction_ann_direction_short_1d_optimization_config.json",
)

# Phase D output files
PHASE_D_RESULTS_JSON = "phase_d_neat_results.json"
PHASE_D_TRADES_CSV = "phase_d_neat_trades.csv"
PHASE_D_CONFIG_DIR = "phase_d_neat_configs"

# Phase C reference results
PHASE_C_RESULTS_JSON = "phase_c_ensemble_results.json"

# Feature constants (from Phase C)
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
# D0: NEAT configuration helpers
# ================================================================

def create_neat_config(template_path, output_path, neat_gens, neat_pop,
                       direction="long"):
    """Create a modified NEAT config with reduced generations for Phase D."""
    with open(template_path) as f:
        config = json.load(f)

    # Resolve all relative paths to absolute (relative to predictor repo root)
    path_keys = [
        "x_train_file", "y_train_file",
        "x_validation_file", "y_validation_file",
        "x_test_file", "y_test_file",
        "use_normalization_json",
    ]
    for key in path_keys:
        val = config.get(key)
        if val and not os.path.isabs(val):
            config[key] = os.path.join(PREDICTOR_DIR, val)

    # Redirect output files to Phase D namespace
    base_dir = os.path.join(RESULTS_DIR, "phase_d")
    os.makedirs(base_dir, exist_ok=True)

    config["population_size"] = neat_pop
    config["optimization_patience"] = max(neat_gens, 5)

    # Calculate total gens: gens_per_stage × 4 stages
    total_gens = neat_gens * 4
    config["n_generations"] = total_gens

    # Update stage generations
    for stage in config.get("optimization_stages", []):
        stage["generations"] = neat_gens

    # Don't resume (fresh NEAT search for Phase D)
    config["optimization_resume"] = False

    # Output files in Phase D namespace (all absolute paths)
    config["output_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_prediction.csv")
    config["results_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_results.csv")
    config["loss_plot_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_loss_plot.png")
    config["model_plot_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_model_plot.png")
    config["uncertainties_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_uncertainties.csv")
    config["predictions_plot_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_predictions_plot.png")
    config["memory_log_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_rss.csv")

    # NEAT-specific outputs (all absolute paths)
    config["optimization_statistics"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_optimization_stats.json")
    config["optimization_parameters_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_optimization_parameters.json")
    config["optimization_resume_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_optimization_resume.json")
    config["optimization_candidate_history"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_optimization_candidate_history.csv")

    # Save the model to Phase D directory (absolute path)
    config["model_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_model.keras")
    config["save_model"] = config["model_file"]
    config["model_metadata_file"] = os.path.join(
        base_dir, f"phase_d_ann_direction_{direction}_model_metadata.json")

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"  Created NEAT config: {output_path}")
    print(f"    Population: {neat_pop}, Gens/stage: {neat_gens}, "
          f"Total gens: {total_gens}")
    return config


def run_neat_optimization(config_path, direction="long"):
    """Run NEAT optimization via predictor subprocess."""
    print(f"\n  Launching NEAT optimization for ANN direction {direction}...")
    print(f"  Config: {config_path}")

    # Run from predictor/app directory so imports resolve correctly
    cmd = [
        sys.executable, "main.py",
        "--load_config", config_path,
    ]
    env = os.environ.copy()
    # Ensure predictor's app directory is on Python path
    env["PYTHONPATH"] = PREDICTOR_APP_DIR + ":" + PREDICTOR_DIR + ":" + \
        env.get("PYTHONPATH", "")
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    env["PREDICTOR_QUIET"] = "1"

    start = time.time()

    proc = subprocess.Popen(
        cmd,
        cwd=PREDICTOR_APP_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Monitor output - print key lines
    champion_fitness = None
    gen_count = 0
    for line in proc.stdout:
        line = line.rstrip()
        # Print progress-relevant lines
        if any(kw in line.lower() for kw in [
            "generation", "champion", "best_fitness", "stage",
            "species", "converged", "patience", "error", "fail",
            "optimization complete", "total candidates",
        ]):
            print(f"    [{direction}] {line}")
        if "generation" in line.lower() and "stage" in line.lower():
            gen_count += 1
        if "champion_fitness" in line.lower() or "best_fitness" in line.lower():
            try:
                # Try to extract fitness value
                parts = line.split("=")
                if len(parts) >= 2:
                    champion_fitness = float(parts[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    proc.wait()
    elapsed = time.time() - start
    print(f"\n  NEAT {direction} completed in {elapsed:.0f}s "
          f"(exit code: {proc.returncode})")

    if proc.returncode != 0:
        print(f"  WARNING: NEAT optimization exited with code {proc.returncode}")

    return proc.returncode, elapsed, champion_fitness


# ================================================================
# Feature engineering (from Phase C)
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
# Model loading & inference (adapted from Phase C)
# ================================================================

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


def build_ann_model(window, channels, hidden_units=128,
                    num_hidden_layers=2, dropout_rate=0.1,
                    use_pe=True, activation="elu"):
    """Build ANN direction model with given hyperparameters."""
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Lambda, Flatten, Dense, Dropout
    from tensorflow.keras.models import Model

    inputs = Input(shape=(window, channels), name="input_layer")
    if use_pe:
        pe = positional_encoding(window, channels)
        x = Lambda(lambda t, pe=pe: t + pe,
                    name="add_positional_encoding")(inputs)
    else:
        x = inputs
    x = Flatten(name="flatten_inputs")(x)
    for i in range(num_hidden_layers):
        x = Dense(hidden_units, activation=activation,
                  name=f"shared_dense_{i}")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f"shared_dropout_{i}")(x)
    output = Dense(1, activation="sigmoid", name="output_horizon_1")(x)
    return Model(inputs=inputs, outputs=[output], name="DirectionANN")


def load_neat_champion(direction, phase_d_results_dir=None):
    """Load the NEAT-evolved champion model.

    Tries Phase D results first, falls back to original Phase 1c results.
    Returns (model, window_size, config_dict, source_label).
    """
    import tensorflow as tf
    import zipfile

    phase_d_dir = phase_d_results_dir or os.path.join(RESULTS_DIR, "phase_d")
    # Try Phase D champion first
    d_model = os.path.join(
        phase_d_dir, f"phase_d_ann_direction_{direction}_model.keras")
    d_params = os.path.join(
        phase_d_dir,
        f"phase_d_ann_direction_{direction}_optimization_parameters.json")
    d_meta = os.path.join(
        phase_d_dir,
        f"phase_d_ann_direction_{direction}_model_metadata.json")

    # Also check original models
    orig_model = os.path.join(
        RESULTS_DIR,
        f"phase_1c_direction_ann_direction_{direction}_model.keras")
    orig_params = os.path.join(
        RESULTS_DIR,
        f"phase_1c_direction_ann_direction_{direction}"
        f"_optimization_parameters.json")

    # Determine which model to use
    if os.path.exists(d_model) and os.path.exists(d_params):
        model_path = d_model
        params_path = d_params
        meta_path = d_meta
        source = "phase_d_neat"
        print(f"  [{direction}] Using Phase D NEAT champion")
    elif os.path.exists(orig_model):
        model_path = orig_model
        # For original models, prefer metadata (reflects actual training)
        # over optimization_parameters (which may differ from final model)
        orig_meta = os.path.join(
            RESULTS_DIR,
            f"phase_1c_direction_ann_direction_{direction}"
            f"_model_metadata.json")
        params_path = orig_params if os.path.exists(orig_params) else None
        meta_path = orig_meta
        source = "phase_1c_original"
        print(f"  [{direction}] Using original Phase 1c model "
              f"(no Phase D champion found)")
    else:
        raise FileNotFoundError(
            f"No model found for ANN direction {direction}")

    # Load metadata first (reflects actual trained model dimensions)
    meta = {}
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # Load optimization parameters (may differ from actual model)
    opt_params = {}
    if params_path and os.path.exists(params_path):
        with open(params_path) as f:
            raw = json.load(f)
        opt_params = raw.get("parameters", raw)

    # For Phase D NEAT champions, optimization params are authoritative
    # For original Phase 1c models, metadata is authoritative
    if source == "phase_d_neat":
        params = opt_params
    else:
        # Use metadata for dimensions, opt_params for architecture details
        params = opt_params.copy()
        if "window_size" in meta:
            params["window_size"] = meta["window_size"]

    # Extract model hyperparameters
    window = int(params.get("window_size", 72))
    hidden = int(params.get("hidden_units", 128))
    layers = int(params.get("num_hidden_layers", 2))
    dropout = float(params.get("dropout_rate", 0.1))
    use_pe = bool(params.get("positional_encoding", True))
    activation = params.get("activation", "elu")

    # Determine channel count from metadata or fallback
    channels = 22  # default FEATURE_COLS count
    if meta and "feature_columns" in meta:
        channels = len(meta["feature_columns"])
    elif params.get("add_window_stats", False):
        channels = 28  # 22 features + 6 window stats

    # Build architecture and load weights
    model = build_ann_model(
        window, channels, hidden, layers, dropout, use_pe, activation)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(model_path, 'r') as zf:
            zf.extractall(tmpdir)
        weights_path = os.path.join(tmpdir, "model.weights.h5")
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            # Try loading with keras load_model as fallback
            model = tf.keras.models.load_model(
                model_path, compile=False, safe_mode=False)

    config = {
        "window": window, "channels": channels,
        "hidden_units": hidden, "num_hidden_layers": layers,
        "dropout_rate": dropout, "positional_encoding": use_pe,
        "activation": activation,
        "add_window_stats": params.get("add_window_stats", False),
    }

    print(f"  [{direction}] Loaded: window={window}, ch={channels}, "
          f"hidden={hidden}×{layers}, dropout={dropout}, PE={use_pe}")
    print(f"  [{direction}] Source: {source}")

    return model, window, config, source


def compute_window_stats(window_data, target_col_idx, periods=(12, 48)):
    """Compute per-window statistics matching predictor's add_window_stats."""
    window_size = window_data.shape[0]
    target_vals = window_data[:, target_col_idx]
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

        X_batch = np.zeros(
            (len(batch_indices), window_size, total_features),
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


# ================================================================
# Prediction source for backtest
# ================================================================

class NEATPredictionSource:
    """Maps hourly timestamps to 4H NEAT champion predictions."""

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
# Backtest infrastructure (from Phase C)
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


def quick_sample_test(sample_data, pred_df, config, label, threshold=0.55):
    """Run a quick 5yr sample test, return (profit, trades, win%, sharpe)."""
    source = NEATPredictionSource(pred_df, confidence_threshold=threshold)
    profit, _, trades = run_backtest(sample_data, source, config, label=label)
    n_trades = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    win_pct = (wins / n_trades * 100) if n_trades > 0 else 0
    pnls = [t['pnl'] for t in trades]
    sharpe = ((np.mean(pnls) / (np.std(pnls) + 1e-10))
              if n_trades > 1 else 0)
    return profit, n_trades, win_pct, sharpe


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


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase D: NEAT-Enhanced Direction Prediction")
    parser.add_argument("--skip-neat", action="store_true",
                        help="Skip NEAT optimization, use existing models")
    parser.add_argument("--neat-gens", type=int, default=5,
                        help="NEAT generations per stage (default: 5)")
    parser.add_argument("--neat-pop", type=int, default=15,
                        help="NEAT population size (default: 15)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    from app.data_handler import load_csv

    total_start = time.time()

    # ================================================================
    # D0: Setup
    # ================================================================
    print(f"{'='*70}")
    print(f"PHASE D: NEAT-ENHANCED DIRECTION PREDICTION")
    print(f"{'='*70}")
    print(f"NEAT generations/stage: {args.neat_gens}")
    print(f"NEAT population: {args.neat_pop}")
    print(f"Total NEAT evaluations budget: "
          f"~{args.neat_gens * 4 * args.neat_pop}")
    print(f"Skip NEAT: {args.skip_neat}")
    print(f"{'='*70}\n")

    neat_state = {"long": {}, "short": {}}

    # ================================================================
    # D1: NEAT Optimization
    # ================================================================
    if not args.skip_neat:
        print(f"\n{'='*70}")
        print(f"PHASE D1: NEAT HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*70}\n")

        os.makedirs(PHASE_D_CONFIG_DIR, exist_ok=True)

        for direction, template in [
            ("long", NEAT_CONFIG_TEMPLATE_LONG),
            ("short", NEAT_CONFIG_TEMPLATE_SHORT),
        ]:
            if not os.path.exists(template):
                print(f"  WARNING: No template config for {direction}: "
                      f"{template}")
                print(f"  Skipping {direction} NEAT optimization")
                continue

            config_path = os.path.join(
                PHASE_D_CONFIG_DIR,
                f"phase_d_ann_direction_{direction}_neat_config.json")
            neat_cfg = create_neat_config(
                template, config_path, args.neat_gens, args.neat_pop,
                direction)

            ret_code, elapsed, fitness = run_neat_optimization(
                os.path.abspath(config_path), direction)

            neat_state[direction] = {
                "return_code": ret_code,
                "elapsed_seconds": elapsed,
                "champion_fitness": fitness,
            }

            # Check what was produced
            params_file = neat_cfg["optimization_parameters_file"]
            if os.path.exists(params_file):
                with open(params_file) as f:
                    champ_params = json.load(f)
                print(f"\n  Champion params ({direction}):")
                p = champ_params.get("parameters", champ_params)
                for k, v in p.items():
                    print(f"    {k}: {v}")
                neat_state[direction]["champion_params"] = p
            else:
                print(f"  WARNING: No champion params file at {params_file}")

            history_file = neat_cfg["optimization_candidate_history"]
            if os.path.exists(history_file):
                n_lines = sum(1 for _ in open(history_file)) - 1
                print(f"  Candidates evaluated: {n_lines}")
                neat_state[direction]["candidates_evaluated"] = n_lines

        # ----- D1.5: Retrain with champion params to save model files -----
        print(f"\n{'='*70}")
        print(f"PHASE D1.5: RETRAIN WITH CHAMPION PARAMS (save .keras models)")
        print(f"{'='*70}\n")
        for direction in ["long", "short"]:
            state = neat_state.get(direction, {})
            champ_p = state.get("champion_params")
            if not champ_p or state.get("return_code", 1) != 0:
                print(f"  [{direction}] No champion / NEAT failed – skipping retrain")
                continue

            neat_cfg_path = os.path.join(
                PHASE_D_CONFIG_DIR,
                f"phase_d_ann_direction_{direction}_neat_config.json")
            if not os.path.exists(neat_cfg_path):
                print(f"  [{direction}] NEAT config not found – skipping retrain")
                continue

            with open(neat_cfg_path) as f:
                retrain_cfg = json.load(f)

            # Disable NEAT optimizer – just train once
            retrain_cfg["use_optimizer"] = False
            retrain_cfg.pop("optimizer_plugin", None)
            retrain_cfg.pop("optimization_stages", None)
            retrain_cfg.pop("optimization_resume", None)
            retrain_cfg.pop("optimization_resume_file", None)

            # Apply champion hyperparameters
            for k, v in champ_p.items():
                if v is not None:
                    retrain_cfg[k] = v

            # Predictor uses 'save_model' key (not 'model_file')
            model_out = retrain_cfg["model_file"]
            retrain_cfg["save_model"] = model_out

            retrain_cfg_path = os.path.join(
                PHASE_D_CONFIG_DIR,
                f"phase_d_ann_direction_{direction}_retrain_config.json")
            with open(retrain_cfg_path, 'w') as f:
                json.dump(retrain_cfg, f, indent=4)

            print(f"  [{direction}] Retraining with champion params …")
            cmd = [sys.executable, "main.py",
                   "--load_config", os.path.abspath(retrain_cfg_path)]
            env = os.environ.copy()
            env["PYTHONPATH"] = PREDICTOR_APP_DIR + ":" + PREDICTOR_DIR + \
                ":" + env.get("PYTHONPATH", "")
            env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
            env["PREDICTOR_QUIET"] = "1"

            t0 = time.time()
            proc = subprocess.Popen(
                cmd, cwd=PREDICTOR_APP_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True)
            for line in proc.stdout:
                print(f"    {line.rstrip()}")
            proc.wait()
            elapsed_rt = time.time() - t0
            if os.path.exists(model_out):
                print(f"  [{direction}] Model saved: {model_out} "
                      f"({elapsed_rt:.0f}s)")
            else:
                print(f"  [{direction}] WARNING: model file not produced!")

    else:
        print(f"\n{'='*70}")
        print(f"PHASE D1: SKIPPING NEAT (using existing models)")
        print(f"{'='*70}\n")

        # Even in skip-neat mode, retrain if champion params exist but
        # model files are missing (e.g. previous run with the bug).
        phase_d_dir = os.path.join(RESULTS_DIR, "phase_d")
        for direction in ["long", "short"]:
            model_path = os.path.join(
                phase_d_dir,
                f"phase_d_ann_direction_{direction}_model.keras")
            params_path = os.path.join(
                phase_d_dir,
                f"phase_d_ann_direction_{direction}"
                f"_optimization_parameters.json")
            neat_cfg_path = os.path.join(
                PHASE_D_CONFIG_DIR,
                f"phase_d_ann_direction_{direction}_neat_config.json")

            if os.path.exists(model_path):
                print(f"  [{direction}] Model exists: {model_path}")
                continue
            if not os.path.exists(params_path) or \
               not os.path.exists(neat_cfg_path):
                print(f"  [{direction}] No params/config – using Phase 1C")
                continue

            print(f"  [{direction}] Model missing – retraining from "
                  f"champion params …")
            with open(params_path) as f:
                champ_p = json.load(f)
            with open(neat_cfg_path) as f:
                retrain_cfg = json.load(f)

            retrain_cfg["use_optimizer"] = False
            retrain_cfg.pop("optimizer_plugin", None)
            retrain_cfg.pop("optimization_stages", None)
            retrain_cfg.pop("optimization_resume", None)
            retrain_cfg.pop("optimization_resume_file", None)
            for k, v in champ_p.items():
                if v is not None:
                    retrain_cfg[k] = v

            # Predictor uses 'save_model' key (not 'model_file')
            retrain_cfg["save_model"] = model_path

            retrain_cfg_path = os.path.join(
                PHASE_D_CONFIG_DIR,
                f"phase_d_ann_direction_{direction}_retrain_config.json")
            with open(retrain_cfg_path, 'w') as f:
                json.dump(retrain_cfg, f, indent=4)

            cmd = [sys.executable, "main.py",
                   "--load_config", os.path.abspath(retrain_cfg_path)]
            env = os.environ.copy()
            env["PYTHONPATH"] = PREDICTOR_APP_DIR + ":" + PREDICTOR_DIR + \
                ":" + env.get("PYTHONPATH", "")
            env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
            env["PREDICTOR_QUIET"] = "1"

            t0 = time.time()
            proc = subprocess.Popen(
                cmd, cwd=PREDICTOR_APP_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True)
            for line in proc.stdout:
                print(f"    {line.rstrip()}")
            proc.wait()
            elapsed_rt = time.time() - t0
            if os.path.exists(model_path):
                print(f"  [{direction}] Model saved: {model_path} "
                      f"({elapsed_rt:.0f}s)")
            else:
                print(f"  [{direction}] WARNING: model file not produced!")

    # ================================================================
    # D2: Load champion models and generate predictions
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE D2: LOAD NEAT CHAMPIONS + GENERATE PREDICTIONS")
    print(f"{'='*70}\n")

    # Load data
    print(f"Loading dataset: {OHLC_FILE}")
    ohlc_1h = load_csv(OHLC_FILE, headers=True)
    print(f"Loaded: {ohlc_1h.shape[0]} bars, "
          f"{ohlc_1h.index.min()} to {ohlc_1h.index.max()}")

    with open(NORM_CONFIG) as f:
        norm_config = json.load(f)

    print("\nResampling to 4H...")
    df_4h = resample_to_4h(ohlc_1h)
    print(f"  4H bars: {len(df_4h)}")

    print("Computing 22 technical features...")
    df_feat = compute_features(df_4h)
    print(f"  After warmup: {len(df_feat)} bars")

    print("Normalizing...")
    df_norm = normalize_features(df_feat, norm_config)

    # Load champions
    model_info = {}
    preds = {}
    phase_d_dir = os.path.join(RESULTS_DIR, "phase_d")

    for direction in ["long", "short"]:
        try:
            model, window, cfg, source = load_neat_champion(
                direction, phase_d_dir)
            p = run_inference(
                model, df_norm, window, f"neat_{direction}",
                add_window_stats=cfg.get("add_window_stats", False))
            preds[direction] = p
            model_info[direction] = {
                "window": window, "source": source, "config": cfg}
            del model
        except Exception as e:
            print(f"  FAILED to load {direction} champion: {e}")
            import traceback
            traceback.print_exc()

    if "long" not in preds:
        print("ERROR: No long direction model loaded. Cannot continue.")
        return

    # Build prediction DataFrame
    n_bars = len(df_feat)
    close_vals = df_feat['CLOSE'].values
    feat_index = df_feat.index

    p_long = preds["long"]
    p_short = preds.get("short", p_long)

    pred_df = pd.DataFrame({
        'p_up_long': p_long,
        'p_up_short': p_short,
        'CLOSE': close_vals,
    }, index=feat_index).dropna(subset=['p_up_long'])

    print(f"\n  NEAT predictions: {len(pred_df)} valid bars, "
          f"P(up) mean={np.nanmean(p_long):.3f}")

    # ================================================================
    # D3: Threshold sweep (in-sample 2012-2016)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE D3: THRESHOLD SWEEP (NEAT ANN, 2012-2016)")
    print(f"{'='*70}\n")

    config = {
        "prediction_source": "API",
        "pp_api_url": "http://offline",
        "pp_timeout": 999,
        "headers": True,
        "disable_multiprocessing": True,
        "atr_period": 14,
        "atr_tp_multiplier": 3.0,
        "atr_sl_multiplier": 3.0,
        "spread_pips": 15.0,
        "commission_per_lot": 7.0,
        "slippage_pips": 5.0,
    }

    year_mask = ((ohlc_1h.index.year >= 2012) &
                 (ohlc_1h.index.year <= 2016))
    sample_data = ohlc_1h[year_mask].copy()

    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65,
                  0.68, 0.70, 0.75, 0.80, 0.85, 0.90]
    d3_results = []
    for thresh in thresholds:
        profit, trades, win_pct, sharpe = quick_sample_test(
            sample_data, pred_df, config,
            label=f"t={thresh}", threshold=thresh)
        d3_results.append({
            "threshold": thresh, "profit": profit, "trades": trades,
            "win_pct": win_pct, "sharpe": sharpe,
        })
        sign = '+' if profit >= 0 else ''
        print(f"  thresh={thresh:.2f}: {sign}${profit:,.2f} | "
              f"{trades}t | Win {win_pct:.0f}% | Sharpe {sharpe:.3f}")

    best_thresh_row = max(
        d3_results,
        key=lambda r: r['sharpe'] if r['trades'] >= 10 else -999)
    best_threshold = best_thresh_row['threshold']
    print(f"\n  BEST threshold: {best_threshold} → "
          f"${best_thresh_row['profit']:,.2f}, "
          f"Sharpe={best_thresh_row['sharpe']:.3f}")

    # ================================================================
    # D4: ATR parameter sweep with best threshold
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE D4: ATR PARAMETER SWEEP (thresh={best_threshold})")
    print(f"{'='*70}\n")

    atr_periods = [10, 14, 21]
    tp_mults = [1.5, 2.0, 2.5, 3.0, 4.0]
    sl_mults = [0.5, 1.0, 1.5, 2.0, 3.0]

    d4_results = []
    for atr_period in atr_periods:
        for tp_mult in tp_mults:
            for sl_mult in sl_mults:
                test_config = config.copy()
                test_config['atr_period'] = atr_period
                test_config['atr_tp_multiplier'] = tp_mult
                test_config['atr_sl_multiplier'] = sl_mult

                profit, trades, win_pct, sharpe = quick_sample_test(
                    sample_data, pred_df, test_config,
                    label=f"atr={atr_period}/tp={tp_mult}/sl={sl_mult}",
                    threshold=best_threshold)

                d4_results.append({
                    "atr_period": atr_period, "tp_mult": tp_mult,
                    "sl_mult": sl_mult, "profit": profit, "trades": trades,
                    "win_pct": win_pct, "sharpe": sharpe,
                })
                sign = '+' if profit >= 0 else ''
                print(f"  ATR={atr_period} TP={tp_mult} SL={sl_mult}: "
                      f"{sign}${profit:,.2f} | {trades}t | "
                      f"Win {win_pct:.0f}% | Sharpe {sharpe:.3f}")

    best_atr = max(
        d4_results,
        key=lambda r: r['sharpe'] if r['trades'] >= 10 else -999)
    print(f"\n  BEST: ATR={best_atr['atr_period']} "
          f"TP={best_atr['tp_mult']} SL={best_atr['sl_mult']} → "
          f"${best_atr['profit']:,.2f}, Sharpe={best_atr['sharpe']:.3f}")

    # ================================================================
    # D5: Full 14yr WFO with best NEAT config
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE D5: FULL WFO WITH BEST NEAT CONFIG")
    print(f"{'='*70}")
    print(f"Model: NEAT ANN Direction ({model_info.get('long', {}).get('source', '?')})")
    print(f"Threshold: {best_threshold}")
    print(f"ATR={best_atr['atr_period']}, "
          f"TP={best_atr['tp_mult']}, SL={best_atr['sl_mult']}")
    print(f"{'='*70}\n")

    final_config = config.copy()
    final_config['atr_period'] = best_atr['atr_period']
    final_config['atr_tp_multiplier'] = best_atr['tp_mult']
    final_config['atr_sl_multiplier'] = best_atr['sl_mult']

    source = NEATPredictionSource(
        pred_df, confidence_threshold=best_threshold)
    d5_results = run_wfo(ohlc_1h, source, final_config)
    print_results("D5: NEAT ANN FULL WFO", d5_results)

    # ================================================================
    # D6: Comparison table (A vs B vs C vs D)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE A vs B vs C vs D COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Oracle (A)':>12} {'CNN (B4)':>12} "
          f"{'Ensemble (C4)':>15} {'NEAT (D5)':>12}")
    print(f"-" * 75)
    print(f"{'Total Profit':<20} {'$1,714,396':>12} "
          f"{'$-4,585':>12} "
          f"{'$-1,207':>15} "
          f"{'${:,.0f}'.format(d5_results['total_profit']):>12}")
    print(f"{'Win Rate':<20} {'100.0%':>12} {'46.3%':>12} "
          f"{'43.9%':>15} "
          f"{'{:.1f}%'.format(d5_results['total_win_pct']):>12}")
    print(f"{'Trades':<20} {'3,452':>12} {'2,745':>12} "
          f"{'1,566':>15} "
          f"{d5_results['total_trades']:>12,}")
    print(f"{'Sharpe':<20} {'0.366':>12} {'-0.043':>12} "
          f"{'-0.031':>15} "
          f"{d5_results['aggregate_sharpe']:>12.3f}")
    print(f"{'Max Drawdown':<20} {'$8,206':>12} {'$8,389':>12} "
          f"{'$4,486':>15} "
          f"{'${:,.0f}'.format(d5_results['max_drawdown_usd']):>12}")
    pf = f"{d5_results['profitable_folds']}/{d5_results['total_folds']}"
    print(f"{'Prof. Years':<20} {'14/14':>12} {'6/14':>12} "
          f"{'6/14':>15} "
          f"{pf:>12}")
    print(f"{'='*75}")

    # ================================================================
    # Save results
    # ================================================================
    total_elapsed = time.time() - total_start

    results_out = {
        "phase": "D",
        "description": "NEAT-Enhanced Direction Prediction",
        "neat_config": {
            "gens_per_stage": args.neat_gens,
            "population_size": args.neat_pop,
            "skip_neat": args.skip_neat,
        },
        "neat_state": neat_state,
        "model_info": {
            d: {k: v for k, v in info.items() if k != "config"}
            for d, info in model_info.items()
        },
        "d3_threshold_sweep": d3_results,
        "d3_best_threshold": best_threshold,
        "d4_atr_sweep_best": best_atr,
        "d5_wfo": {
            "total_profit": d5_results["total_profit"],
            "total_trades": d5_results["total_trades"],
            "total_win_pct": d5_results["total_win_pct"],
            "aggregate_sharpe": d5_results["aggregate_sharpe"],
            "max_drawdown_usd": d5_results["max_drawdown_usd"],
            "final_equity": d5_results["final_equity"],
            "profitable_folds": d5_results["profitable_folds"],
            "total_folds": d5_results["total_folds"],
            "fold_results": d5_results["fold_results"],
        },
        "total_elapsed_seconds": total_elapsed,
    }

    with open(PHASE_D_RESULTS_JSON, 'w') as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\nResults saved to {PHASE_D_RESULTS_JSON}")

    # Save trades
    if d5_results["all_trades"]:
        trades_df = pd.DataFrame(d5_results["all_trades"])
        trades_df.to_csv(PHASE_D_TRADES_CSV, index=False)
        print(f"Trades saved to {PHASE_D_TRADES_CSV}")

    print(f"\nPhase D completed in {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()

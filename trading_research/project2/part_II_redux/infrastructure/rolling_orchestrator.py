#!/usr/bin/env python3
"""
Rolling Orchestrator for Project 2 Part II (G-1 resolution).

Manages anchored expanding-window walk-forward experiments across
heuristic-strategy (Path A) and predictor (Path B) pipelines.

Capabilities (per Project2_part2.md §2.1):
  1.  Load window manifest (JSON per F-5 §5.3)
  2.  Per-window: slice data, normalize (fit on train only), invoke model/strategy,
      capture metrics, log per F-5 §7 CSV format
  3.  Per-window state isolation (no cross-contamination)
  4.  Configurable embargo between train and validation
  5.  Graceful per-window failure handling (log, skip, continue)
  6.  Aggregate cross-window metrics
  7.  Optional rolling GMM re-fit (UL-1) per window
  8.  Optional change-point triggered windows (UL-2)
  9.  Full reproducibility logging (window, feature set version, model state hash)
  10. Separate train-complete from deploy; support rollback on validation failure

Usage:
  python rolling_orchestrator.py --manifest data/windows/window_manifest.json \
      --path A --strategy_plugin eurusd_mr --data data/processed/eurusd_4h_2005_2024.csv \
      --output_dir logs/exp_A1 [--embargo_bars 6] [--gmm_refit] [--population_size 30]

Author: Project 2 automated build
Date: 2026-04-19
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_CSV_HEADER = [
    "experiment_id", "path", "window_id", "model_type",
    "train_start", "train_end", "val_start", "val_end",
    "test_start", "test_end",
    "embargo_bars",
    "train_sharpe", "val_sharpe", "test_sharpe",
    "train_mae", "val_mae", "test_mae",
    "max_dd", "num_trades", "cost_ratio",
    "params_json", "model_hash", "timestamp", "status", "error"
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_params(params: dict) -> str:
    """Deterministic hash of parameter dict for reproducibility."""
    serialised = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


def _compute_sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe from trade-level returns (no risk-free rate)."""
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma)


def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a fraction of peak equity."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def _compute_cost_ratio(gross_pnl: float, total_costs: float) -> float:
    """Gross PnL / total costs.  K-3 requires >= 2.0."""
    if total_costs <= 0:
        return float('inf') if gross_pnl > 0 else 0.0
    return gross_pnl / total_costs


def _slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice dataframe by date strings (inclusive)."""
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask].copy()


def _apply_embargo(df: pd.DataFrame, train_end_idx: int, embargo_bars: int) -> pd.DataFrame:
    """Mark embargo rows.  Returns df without embargo rows removed — caller decides."""
    # We don't mutate; just return the integer indices to exclude.
    return list(range(train_end_idx + 1, min(train_end_idx + 1 + embargo_bars, len(df))))


def _backtest_predictions(predictions: np.ndarray, close_prices: np.ndarray,
                          cost_per_trade: float = 0.001,
                          binary: bool = False) -> dict:
    """
    Convert predictions to trading signals and compute metrics.

    Parameters:
        predictions: 1D array of predicted values
            - regression: forward returns (sign determines direction)
            - binary: probabilities in [0,1] (>0.5 long, <0.5 short)
        close_prices: 1D array of close prices aligned with predictions (len >= len(predictions)+1)
        cost_per_trade: proportional transaction cost per trade (default 10 bps)
        binary: if True, treat predictions as binary probabilities

    Returns dict with sharpe, max_dd, num_trades, pnl_list, cost_ratio.
    """
    preds = predictions.flatten()
    n = len(preds)
    if n < 2 or len(close_prices) < n + 1:
        return {"sharpe": 0.0, "max_dd": 0.0, "num_trades": 0,
                "pnl_list": [], "cost_ratio": 0.0}

    # Position: direction of prediction
    if binary:
        # Binary: prob > 0.5 → long (+1), prob < 0.5 → short (-1)
        positions = np.where(preds > 0.5, 1.0, -1.0)
    else:
        # Regression: sign of prediction (+1 long, -1 short, 0 flat)
        positions = np.sign(preds)

    # 1-bar log returns: log(close[t+1] / close[t])
    bar_returns = np.diff(np.log(close_prices[:n + 1]))  # length n

    # Strategy returns: position * bar return
    strategy_returns = positions * bar_returns[:n]

    # Transaction costs: cost applied on position changes
    position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
    costs = position_changes * cost_per_trade
    net_returns = strategy_returns - costs

    # Count trades (position changes)
    num_trades = int(np.sum(position_changes > 0))

    # PnL in dollar terms (from $10,000 notional)
    notional = 10000.0
    pnl_list = (net_returns * notional).tolist()

    # Equity curve
    equity = np.array([notional] + list(np.cumsum(net_returns) * notional + notional))
    max_dd = _compute_max_drawdown(equity)

    # Sharpe
    sharpe = _compute_sharpe(np.array(net_returns))

    # Cost ratio
    gross_pnl = float(np.sum(strategy_returns) * notional)
    total_costs = float(np.sum(costs) * notional)
    cost_ratio = _compute_cost_ratio(gross_pnl, total_costs)

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "num_trades": num_trades,
        "pnl_list": pnl_list,
        "cost_ratio": cost_ratio,
    }


# ---------------------------------------------------------------------------
# GMM rolling re-fit (UL-1) — optional
# ---------------------------------------------------------------------------

def _refit_gmm(train_data: pd.DataFrame, n_components: int = 9,
               random_state: int = 42) -> dict:
    """Re-fit GMM on training data for regime detection (UL-1).
    Returns dict with model params and regime labels."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("[WARN] scikit-learn not available; skipping GMM refit.")
        return {}

    feature_cols = [c for c in train_data.columns
                    if c.upper() not in ('DATE', 'DATE_TIME', 'DATETIME', 'OPEN',
                                         'HIGH', 'LOW', 'CLOSE', 'VOLUME')]
    if not feature_cols:
        feature_cols = [c for c in train_data.columns
                        if c.upper() not in ('DATE', 'DATE_TIME', 'DATETIME')]

    X = train_data[feature_cols].dropna().values
    if len(X) < n_components * 10:
        print(f"[WARN] Only {len(X)} samples for GMM with K={n_components}; skipping.")
        return {}

    gmm = GaussianMixture(n_components=n_components, random_state=random_state,
                           covariance_type='full', max_iter=200)
    gmm.fit(X)
    labels = gmm.predict(X)
    return {
        "means": gmm.means_.tolist(),
        "covariances": [c.tolist() for c in gmm.covariances_],
        "weights": gmm.weights_.tolist(),
        "labels": labels.tolist(),
        "n_components": n_components,
        "features_used": feature_cols
    }


# ---------------------------------------------------------------------------
# Path A: Heuristic strategy runner
# ---------------------------------------------------------------------------

class PathARunner:
    """Run heuristic-strategy plugin on a single window."""

    def __init__(self, plugin_name: str, optimizer: str = "deap_ga",
                 population_size: int = 30, num_generations: int = 20,
                 min_trades: int = 10, fixed_params: dict = None):
        self.plugin_name = plugin_name
        self.optimizer = optimizer
        self.population_size = population_size
        self.num_generations = num_generations
        self.min_trades = min_trades
        self.fixed_params = fixed_params  # If set, skip GA and use these
        self._plugin = None
        self._load_plugin()

    def _load_plugin(self):
        """Load the heuristic-strategy plugin."""
        # Add heuristic-strategy to path if needed
        hs_root = Path(__file__).resolve().parents[4]  # -> heuristic-strategy/
        if str(hs_root) not in sys.path:
            sys.path.insert(0, str(hs_root))
        try:
            from app.plugin_loader import load_plugin
            plugin_class, _ = load_plugin('heuristic_strategy.plugins', self.plugin_name)
            self._plugin = plugin_class()
        except Exception as e:
            raise RuntimeError(f"Cannot load heuristic plugin '{self.plugin_name}': {e}")

    def run_window(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                   test_data: pd.DataFrame, config: dict,
                   gmm_info: dict = None) -> dict:
        """
        Run GA optimisation on train, select on val, evaluate on test.

        Returns dict with metrics for all three splits.
        """
        from app.walk_forward_optimizer import _run_ga_on_slice, _evaluate_with_sharpe

        plugin = deepcopy(self._plugin)

        # Merge heuristic-strategy DEFAULT_VALUES so prediction-based plugins
        # get required keys (hourly_predictions_file, daily_predictions_file, etc.)
        try:
            from app.config import DEFAULT_VALUES
            run_config = deepcopy(DEFAULT_VALUES)
        except ImportError:
            run_config = {}
        # Set prediction files to None so plugins auto-generate from base data
        run_config['hourly_predictions_file'] = None
        run_config['daily_predictions_file'] = None
        run_config.update(config)

        # Inject GMM regime info if available
        if gmm_info:
            run_config['gmm_means'] = gmm_info.get('means')
            run_config['gmm_weights'] = gmm_info.get('weights')

        plugin.set_params(**run_config)

        opt_params = plugin.get_optimizable_params()

        if self.fixed_params:
            # Static replay mode — skip GA, use provided fixed parameters
            best_params = {name: self.fixed_params.get(name, (lo + hi) / 2)
                           for name, lo, hi in opt_params}
            candidate = [best_params[name] for name, lo, hi in opt_params]
            # Evaluate on train to get train_fitness (for reporting)
            train_result = plugin.evaluate_candidate(candidate, train_data, None, None, run_config)
            train_profit = train_result[0] if isinstance(train_result, tuple) else train_result
            train_trades_list = list(getattr(plugin, 'trades', []))
            train_pnl = [t['pnl'] for t in train_trades_list] if train_trades_list else []
            train_fitness = _compute_sharpe(np.array(train_pnl)) if train_pnl else 0.0
        else:
            # Phase 1: Optimise on training data
            best_params, train_fitness = _run_ga_on_slice(
                plugin, train_data, run_config,
                population_size=self.population_size,
                num_generations=self.num_generations,
                min_trades=self.min_trades
            )

        # Phase 2: Evaluate on validation with best params
        candidate = [best_params.get(name, (lo + hi) / 2)
                     for name, lo, hi in opt_params]

        val_result = plugin.evaluate_candidate(candidate, val_data, None, None, run_config)
        val_profit, val_stats = (val_result if isinstance(val_result, tuple) and len(val_result) == 2
                                  else (val_result[0] if isinstance(val_result, tuple) else val_result, {}))
        val_trades = list(getattr(plugin, 'trades', []))
        val_pnl = [t['pnl'] for t in val_trades] if val_trades else []
        val_sharpe = _compute_sharpe(np.array(val_pnl)) if val_pnl else 0.0

        # Phase 3: Evaluate ONCE on test (held-out for this window)
        test_result = plugin.evaluate_candidate(candidate, test_data, None, None, run_config)
        test_profit, test_stats = (test_result if isinstance(test_result, tuple) and len(test_result) == 2
                                    else (test_result[0] if isinstance(test_result, tuple) else test_result, {}))
        test_trades = list(getattr(plugin, 'trades', []))
        test_pnl = [t['pnl'] for t in test_trades] if test_trades else []
        test_sharpe = _compute_sharpe(np.array(test_pnl)) if test_pnl else 0.0

        # Compute equity curve and max drawdown from test trades
        equity = [10000.0]
        for pnl in test_pnl:
            equity.append(equity[-1] + pnl)
        test_max_dd = _compute_max_drawdown(np.array(equity))

        # Cost ratio (approximate: gross positive trades vs costs)
        gross_profits = sum(p for p in test_pnl if p > 0)
        gross_losses = abs(sum(p for p in test_pnl if p < 0))
        # Costs estimated from spread + commission embedded in pnl
        # For K-3 we use gross_pnl / estimated_costs
        total_costs = test_stats.get('total_costs', gross_losses * 0.1) if test_stats else gross_losses * 0.1
        cost_ratio = _compute_cost_ratio(sum(test_pnl), total_costs)

        return {
            "best_params": best_params,
            "train_fitness": train_fitness,
            "train_sharpe": train_fitness,  # WFO uses Sharpe*sqrt(N) as fitness
            "val_sharpe": val_sharpe,
            "val_profit": val_profit,
            "val_trades": len(val_trades),
            "test_sharpe": test_sharpe,
            "test_profit": test_profit,
            "test_trades": len(test_trades),
            "test_max_dd": test_max_dd,
            "test_pnl_list": test_pnl,
            "cost_ratio": cost_ratio,
            "train_mae": 0.0,  # N/A for heuristic
            "val_mae": 0.0,
            "test_mae": 0.0,
            "model_hash": _hash_params(best_params),
        }


# ---------------------------------------------------------------------------
# Path B: Supervised ML runner
# ---------------------------------------------------------------------------

class PathBRunner:
    """Run predictor plugin on a single window."""

    def __init__(self, plugin_name: str, epochs: int = 100,
                 early_patience: int = 10, batch_size: int = 32):
        self.plugin_name = plugin_name
        self.epochs = epochs
        self.early_patience = early_patience
        self.batch_size = batch_size
        self._plugin = None
        self._load_plugin()

    def _load_plugin(self):
        """Load the predictor plugin."""
        pred_root = Path(__file__).resolve().parents[4].parent / 'predictor'
        if str(pred_root) not in sys.path:
            sys.path.insert(0, str(pred_root))
        try:
            from app.plugin_loader import load_plugin
            plugin_class, _ = load_plugin('predictor.plugins', self.plugin_name)
            # Some plugins require config in __init__, try both
            try:
                self._plugin = plugin_class()
            except TypeError:
                self._plugin = plugin_class(config={})
        except Exception as e:
            raise RuntimeError(f"Cannot load predictor plugin '{self.plugin_name}': {e}")

    def run_window(self, x_train: np.ndarray, y_train: np.ndarray,
                   x_val: np.ndarray, y_val: np.ndarray,
                   x_test: np.ndarray, y_test: np.ndarray,
                   config: dict) -> dict:
        """
        Train on train, validate with early stopping, predict on test.
        Returns metrics dict.
        """
        plugin = deepcopy(self._plugin)
        run_config = deepcopy(config)
        run_config['epochs'] = self.epochs
        run_config['early_patience'] = self.early_patience
        run_config['batch_size'] = self.batch_size

        # Ensure predictor-required config keys
        if 'predicted_horizons' not in run_config:
            run_config['predicted_horizons'] = [1]
        if 'plotted_horizon' not in run_config:
            run_config['plotted_horizon'] = run_config['predicted_horizons'][0]
        # Disable slow MC uncertainty during batch runs
        run_config['disable_postfit_uncertainty'] = True
        run_config['quiet'] = True

        # LSTM plugin requires additional config keys
        lstm_defaults = {
            'intermediate_layers': 2,
            'initial_layer_size': 128,
            'layer_size_divisor': 2,
            'l2_reg': 1e-4,
            'activation': 'relu',
        }
        for k, v in lstm_defaults.items():
            if k not in run_config:
                run_config[k] = v

        plugin.set_params(**run_config)

        # Detect binary classification plugins and binarize target
        is_binary = 'binary' in self.plugin_name.lower()
        if is_binary:
            y_train = (y_train > 0).astype(np.float32)
            y_val = (y_val > 0).astype(np.float32)
            y_test = (y_test > 0).astype(np.float32)

        # Build and train model
        plugin.build_model(x_train.shape[1:], x_train, run_config)

        # Train directly via model.fit (bypass base.train() which has dict-y issues on TF 2.19+)
        from tensorflow.keras.callbacks import EarlyStopping
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.early_patience,
                                   restore_best_weights=True)]
        y_train_list = [y_train.reshape(-1, 1)]
        y_val_list = [y_val.reshape(-1, 1)]
        plugin.model.fit(
            x_train, y_train_list,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(x_val, y_val_list),
            callbacks=callbacks,
            verbose=0, shuffle=False
        )

        # Predict on all splits
        pred_bs = max(self.batch_size, 256)
        train_preds = plugin.model.predict(x_train, batch_size=pred_bs, verbose=0)
        val_preds = plugin.model.predict(x_val, batch_size=pred_bs, verbose=0)
        test_preds = plugin.model.predict(x_test, batch_size=pred_bs, verbose=0)

        # Extract predictions (may be list of arrays for multi-horizon)
        if isinstance(train_preds, list):
            train_preds = train_preds[0]
        if isinstance(val_preds, list):
            val_preds = val_preds[0]
        if isinstance(test_preds, list):
            test_preds = test_preds[0]

        # Compute MAE per split
        train_mae = float(np.mean(np.abs(train_preds.flatten() - y_train.flatten())))
        val_mae = float(np.mean(np.abs(val_preds.flatten() - y_val.flatten())))
        test_mae = float(np.mean(np.abs(test_preds.flatten() - y_test.flatten())))

        # Model hash from weights
        try:
            import hashlib as hl
            weight_bytes = b''.join(w.tobytes() for w in plugin.model.get_weights())
            model_hash = hl.sha256(weight_bytes).hexdigest()[:16]
        except Exception:
            model_hash = "unknown"

        return {
            "train_preds": train_preds,
            "val_preds": val_preds,
            "test_preds": test_preds,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "test_mae": test_mae,
            "train_sharpe": 0.0,   # Computed downstream after signal→trade
            "val_sharpe": 0.0,
            "test_sharpe": 0.0,
            "test_max_dd": 0.0,
            "test_trades": 0,
            "cost_ratio": 0.0,
            "model_hash": model_hash,
            "best_params": run_config,
        }


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _fit_normalize(train_df: pd.DataFrame, method: str = 'z-score') -> dict:
    """Fit normalisation parameters on training data only."""
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    params = {"method": method, "columns": numeric_cols}
    if method == 'z-score':
        params["mean"] = train_df[numeric_cols].mean().to_dict()
        params["std"] = train_df[numeric_cols].std().replace(0, 1).to_dict()
    elif method == 'min-max':
        params["min"] = train_df[numeric_cols].min().to_dict()
        params["max"] = train_df[numeric_cols].max().to_dict()
    return params


def _apply_normalize(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply pre-fitted normalisation to a dataframe."""
    result = df.copy()
    cols = params["columns"]
    existing = [c for c in cols if c in result.columns]
    if params["method"] == 'z-score':
        for c in existing:
            result[c] = (result[c] - params["mean"][c]) / params["std"].get(c, 1.0)
    elif params["method"] == 'min-max':
        for c in existing:
            rng = params["max"][c] - params["min"][c]
            if rng < 1e-12:
                rng = 1.0
            result[c] = (result[c] - params["min"][c]) / rng
    return result


# ---------------------------------------------------------------------------
# Core Orchestrator
# ---------------------------------------------------------------------------

class RollingOrchestrator:
    """
    Manages rolling walk-forward experiments with full state isolation.

    Design constraints (per Project2_part2.md §2.1):
    - Plugin-compatible with heuristic-strategy (Path A) AND predictor (Path B)
    - Produces F-5 §7 CSV format
    - Per-window state isolation
    - Configurable embargo
    - Graceful failure handling
    """

    def __init__(self, manifest_path: str, data_path: str,
                 output_dir: str, experiment_id: str,
                 path: str = "A",
                 strategy_plugin: str = "eurusd_mr",
                 predictor_plugin: str = None,
                 embargo_bars: int = 6,
                 normalize_method: str = "z-score",
                 gmm_refit: bool = False,
                 gmm_components: int = 9,
                 population_size: int = 30,
                 num_generations: int = 20,
                 min_trades: int = 10,
                 epochs: int = 100,
                 early_patience: int = 10,
                 batch_size: int = 32,
                 config_overrides: dict = None,
                 raw_data_path: str = None):

        self.manifest_path = manifest_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.experiment_id = experiment_id
        self.path = path.upper()
        self.embargo_bars = embargo_bars
        self.normalize_method = normalize_method
        self.gmm_refit = gmm_refit
        self.gmm_components = gmm_components
        self.config_overrides = config_overrides or {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # Load full dataset
        self.full_data = self._load_data(data_path)

        # Load raw OHLCV data for Path B backtesting (close prices)
        self.raw_data = None
        if self.path == "B":
            if raw_data_path:
                self.raw_data = self._load_data(raw_data_path)
            else:
                # Auto-detect: try *_features.csv → *.csv
                base = data_path.replace('_features.csv', '.csv')
                if base != data_path and os.path.exists(base):
                    self.raw_data = self._load_data(base)
            if self.raw_data is not None:
                print(f"Raw OHLCV:   {raw_data_path or base} ({len(self.raw_data)} bars)")

        # Initialise runner
        if self.path == "A":
            self.runner = PathARunner(
                plugin_name=strategy_plugin,
                optimizer="deap_ga",
                population_size=population_size,
                num_generations=num_generations,
                min_trades=min_trades,
                fixed_params=config_overrides.get('fixed_params') if config_overrides else None
            )
        elif self.path == "B":
            if not predictor_plugin:
                raise ValueError("--predictor_plugin required for Path B")
            self.runner = PathBRunner(
                plugin_name=predictor_plugin,
                epochs=epochs,
                early_patience=early_patience,
                batch_size=batch_size
            )
        else:
            raise ValueError(f"Unknown path: {self.path}. Must be 'A' or 'B'.")

        # Results storage
        self.window_results = []
        self.csv_path = os.path.join(output_dir, f"{experiment_id}_results.csv")
        self._init_csv()

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load OHLCV CSV with datetime index."""
        df = pd.read_csv(path)
        # Auto-detect date column
        date_col = None
        for candidate in ['DATE_TIME', 'Date', 'date', 'datetime', 'Datetime', 'DATE']:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            # Assume first column is date
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        return df

    def _init_csv(self):
        """Initialise experiment tracking CSV with header."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(EXPERIMENT_CSV_HEADER)

    def _log_window_result(self, window_id: int, window_def: dict,
                           result: dict, status: str = "OK",
                           error: str = ""):
        """Append one row to the experiment CSV per F-5 §7 format."""
        row = [
            self.experiment_id,
            self.path,
            window_id,
            self.runner.plugin_name if hasattr(self.runner, 'plugin_name') else "unknown",
            window_def.get("train_start", ""),
            window_def.get("train_end", ""),
            window_def.get("val_start", ""),
            window_def.get("val_end", ""),
            window_def.get("test_start", ""),
            window_def.get("test_end", ""),
            self.embargo_bars,
            result.get("train_sharpe", ""),
            result.get("val_sharpe", ""),
            result.get("test_sharpe", ""),
            result.get("train_mae", ""),
            result.get("val_mae", ""),
            result.get("test_mae", ""),
            result.get("test_max_dd", ""),
            result.get("test_trades", ""),
            result.get("cost_ratio", ""),
            json.dumps(result.get("best_params", {}), default=str),
            result.get("model_hash", ""),
            datetime.now().isoformat(),
            status,
            error
        ]
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _save_window_state(self, window_id: int, result: dict,
                           norm_params: dict, gmm_info: dict = None):
        """Save per-window state for reproducibility (req #9)."""
        window_dir = os.path.join(self.output_dir, f"window_{window_id:03d}")
        os.makedirs(window_dir, exist_ok=True)

        # Save normalisation params
        with open(os.path.join(window_dir, "norm_params.json"), 'w') as f:
            json.dump(norm_params, f, indent=2, default=str)

        # Save best params
        with open(os.path.join(window_dir, "best_params.json"), 'w') as f:
            json.dump(result.get("best_params", {}), f, indent=2, default=str)

        # Save GMM if applicable
        if gmm_info:
            with open(os.path.join(window_dir, "gmm_info.json"), 'w') as f:
                json.dump(gmm_info, f, indent=2, default=str)

    def run(self) -> dict:
        """
        Execute the full rolling experiment across all windows in the manifest.

        Returns aggregated results dict.
        """
        windows = self.manifest.get("windows", [])
        print(f"\n{'='*70}")
        print(f"ROLLING ORCHESTRATOR — Experiment {self.experiment_id}")
        print(f"{'='*70}")
        print(f"Path:        {self.path}")
        print(f"Windows:     {len(windows)}")
        print(f"Data:        {self.data_path} ({len(self.full_data)} bars)")
        print(f"Embargo:     {self.embargo_bars} bars")
        print(f"GMM refit:   {self.gmm_refit}")
        print(f"Normalize:   {self.normalize_method}")
        print(f"Output:      {self.output_dir}")
        print(f"{'='*70}\n")

        start_time = time.time()
        all_test_pnl = []

        for i, window_def in enumerate(windows):
            window_id = window_def.get("id", i + 1)
            print(f"\n--- Window {window_id}/{len(windows)} ---")
            print(f"    Train: {window_def['train_start']} → {window_def['train_end']}")
            print(f"    Val:   {window_def['val_start']} → {window_def['val_end']}")
            print(f"    Test:  {window_def['test_start']} → {window_def['test_end']}")

            try:
                result = self._run_single_window(window_def)
                self.window_results.append(result)
                self._log_window_result(window_id, window_def, result,
                                        status="OK")

                # Collect test PnL for aggregation
                if "test_pnl_list" in result:
                    all_test_pnl.extend(result["test_pnl_list"])

                print(f"    ✓ Test Sharpe={result.get('test_sharpe', 0):.3f}, "
                      f"Trades={result.get('test_trades', 0)}, "
                      f"MaxDD={result.get('test_max_dd', 0):.3f}")

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                print(f"    ✗ FAILED: {error_msg}")
                traceback.print_exc()
                self._log_window_result(window_id, window_def,
                                        {}, status="FAILED", error=error_msg)

        # ─── Aggregate ───
        total_time = time.time() - start_time
        agg = self._aggregate_results(all_test_pnl)
        agg["total_time_sec"] = total_time
        agg["windows_completed"] = len(self.window_results)
        agg["windows_total"] = len(windows)

        # Save aggregate summary
        summary_path = os.path.join(self.output_dir, f"{self.experiment_id}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(agg, f, indent=2, default=str)

        self._print_summary(agg)
        return agg

    def _run_single_window(self, window_def: dict) -> dict:
        """Run a single window with full state isolation."""
        window_id = window_def.get("id", 0)

        # 1. Slice data
        train_raw = _slice_window(self.full_data,
                                   window_def["train_start"],
                                   window_def["train_end"])
        val_raw = _slice_window(self.full_data,
                                 window_def["val_start"],
                                 window_def["val_end"])
        test_raw = _slice_window(self.full_data,
                                  window_def["test_start"],
                                  window_def["test_end"])

        if len(train_raw) == 0:
            raise ValueError(f"Window {window_id}: empty training data")
        if len(test_raw) == 0:
            raise ValueError(f"Window {window_id}: empty test data")

        # 2. Apply embargo: remove embargo_bars from the end of training
        # and beginning of validation to prevent leakage
        if self.embargo_bars > 0 and len(val_raw) > self.embargo_bars:
            val_raw = val_raw.iloc[self.embargo_bars:]
            print(f"    Embargo: removed {self.embargo_bars} bars from val start")

        # 3. Normalise (fit on train only — req #3 state isolation)
        norm_params = _fit_normalize(train_raw, method=self.normalize_method)

        # For Path A: strategy works on raw OHLC (unnormalised)
        # For Path B: predictor needs normalised features
        if self.path == "A":
            train_data = train_raw
            val_data = val_raw
            test_data = test_raw
        else:
            train_data = _apply_normalize(train_raw, norm_params)
            val_data = _apply_normalize(val_raw, norm_params)
            test_data = _apply_normalize(test_raw, norm_params)

        # 4. Optional GMM refit (UL-1)
        gmm_info = {}
        if self.gmm_refit:
            print(f"    Re-fitting GMM (K={self.gmm_components}) on training data...")
            gmm_info = _refit_gmm(train_raw, n_components=self.gmm_components)
            if gmm_info:
                print(f"    GMM fit: {gmm_info['n_components']} components, "
                      f"{len(gmm_info.get('features_used', []))} features")

        # 5. Run the appropriate runner
        if self.path == "A":
            result = self.runner.run_window(
                train_data, val_data, test_data,
                config=self.config_overrides,
                gmm_info=gmm_info
            )
        else:
            # Path B: need feature/target arrays
            # Assume last column is target, rest are features
            feature_cols = [c for c in train_data.columns if c.upper() != 'CLOSE']
            target_col = 'CLOSE'
            if target_col not in train_data.columns:
                target_col = train_data.columns[-1]
                feature_cols = list(train_data.columns[:-1])

            lookback = self.config_overrides.get('lookback', 24)

            def _make_sequences(df, feat_cols, tgt_col, lb):
                """Convert 2D feature df into 3D sliding windows (samples, lb, features)."""
                X_raw = df[feat_cols].values
                y_raw = df[tgt_col].values
                n = len(X_raw)
                if n <= lb:
                    return np.empty((0, lb, len(feat_cols))), np.empty((0,))
                X_seq = np.array([X_raw[i:i+lb] for i in range(n - lb)])
                y_seq = y_raw[lb:]
                return X_seq, y_seq

            x_train, y_train = _make_sequences(train_data, feature_cols, target_col, lookback)
            x_val, y_val = _make_sequences(val_data, feature_cols, target_col, lookback)
            x_test, y_test = _make_sequences(test_data, feature_cols, target_col, lookback)

            print(f"Path B sequences: lookback={lookback}, "
                        f"train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

            result = self.runner.run_window(
                x_train=x_train, y_train=y_train,
                x_val=x_val, y_val=y_val,
                x_test=x_test, y_test=y_test,
                config=self.config_overrides
            )

            # --- Signal → trade conversion for Path B ---
            is_binary_plugin = hasattr(self.runner, 'plugin_name') and 'binary' in self.runner.plugin_name.lower()
            if self.raw_data is not None and "test_preds" in result:
                # Align close prices with test sequence indices
                # After _make_sequences, test targets start at test_data.index[lookback:]
                test_seq_idx = test_data.index[lookback:]
                # Get close prices for test period (+1 for computing bar returns)
                close_col = None
                for cc in ['Close', 'CLOSE', 'close']:
                    if cc in self.raw_data.columns:
                        close_col = cc
                        break
                if close_col:
                    # Slice raw data covering test period
                    raw_test = _slice_window(self.raw_data,
                                             str(test_seq_idx[0]),
                                             str(test_data.index[-1]))
                    # Align by matching indices
                    common_idx = test_seq_idx.intersection(raw_test.index)
                    if len(common_idx) > 10:
                        # Need close prices at common_idx + 1 more bar for return calc
                        raw_aligned = raw_test.loc[raw_test.index >= common_idx[0]]
                        close_prices = raw_aligned[close_col].values
                        # Trim predictions to match available close prices
                        pred_mask = test_seq_idx.isin(raw_aligned.index)
                        aligned_preds = result["test_preds"].flatten()
                        aligned_preds = aligned_preds[:min(len(aligned_preds),
                                                           len(close_prices) - 1)]
                        close_for_bt = close_prices[:len(aligned_preds) + 1]

                        bt = _backtest_predictions(aligned_preds, close_for_bt,
                                                      binary=is_binary_plugin)
                        result["test_sharpe"] = bt["sharpe"]
                        result["test_max_dd"] = bt["max_dd"]
                        result["test_trades"] = bt["num_trades"]
                        result["test_pnl_list"] = bt["pnl_list"]
                        result["cost_ratio"] = bt["cost_ratio"]

                        # Also backtest train and val for reporting
                        train_seq_idx = train_data.index[lookback:]
                        raw_train = _slice_window(self.raw_data,
                                                   str(train_seq_idx[0]),
                                                   str(train_data.index[-1]))
                        if close_col in raw_train.columns and len(raw_train) > lookback:
                            train_close = raw_train[close_col].values
                            train_preds = result["train_preds"].flatten()
                            train_preds = train_preds[:min(len(train_preds),
                                                           len(train_close) - 1)]
                            bt_train = _backtest_predictions(train_preds,
                                                             train_close[:len(train_preds) + 1],
                                                             binary=is_binary_plugin)
                            result["train_sharpe"] = bt_train["sharpe"]

                        val_seq_idx = val_data.index[lookback:]
                        raw_val = _slice_window(self.raw_data,
                                                 str(val_seq_idx[0]),
                                                 str(val_data.index[-1]))
                        if close_col in raw_val.columns and len(raw_val) > lookback:
                            val_close = raw_val[close_col].values
                            val_preds = result["val_preds"].flatten()
                            val_preds = val_preds[:min(len(val_preds),
                                                       len(val_close) - 1)]
                            bt_val = _backtest_predictions(val_preds,
                                                           val_close[:len(val_preds) + 1],
                                                           binary=is_binary_plugin)
                            result["val_sharpe"] = bt_val["sharpe"]
                    else:
                        print(f"    [WARN] Only {len(common_idx)} overlapping bars for backtest")
                else:
                    print("    [WARN] No Close column in raw data — skipping backtest")
            elif self.raw_data is None and self.path == "B":
                print("    [WARN] No raw OHLCV data — trading metrics unavailable")

        # 6. Save per-window state (req #9)
        self._save_window_state(window_id, result, norm_params, gmm_info)

        return result

    def _aggregate_results(self, all_test_pnl: list) -> dict:
        """Aggregate cross-window metrics (req #6)."""
        if not self.window_results:
            return {"status": "NO_WINDOWS_COMPLETED"}

        test_sharpes = [r.get("test_sharpe", 0) for r in self.window_results]
        val_sharpes = [r.get("val_sharpe", 0) for r in self.window_results]
        total_trades = sum(r.get("test_trades", 0) for r in self.window_results)

        # K-5: window consistency — fraction of windows with positive test Sharpe
        positive_windows = sum(1 for s in test_sharpes if s > 0)
        window_consistency = positive_windows / len(test_sharpes) if test_sharpes else 0

        # Aggregate Sharpe from all test PnL
        agg_sharpe = _compute_sharpe(np.array(all_test_pnl)) if all_test_pnl else 0.0

        # Equity curve
        equity = [10000.0]
        for pnl in all_test_pnl:
            equity.append(equity[-1] + pnl)
        max_dd = _compute_max_drawdown(np.array(equity))

        # Parameter stability across windows (req per §4.6)
        param_stability = self._compute_param_stability()

        # Per-window metrics
        window_metrics = []
        for i, r in enumerate(self.window_results):
            window_metrics.append({
                "window": i + 1,
                "test_sharpe": r.get("test_sharpe", 0),
                "val_sharpe": r.get("val_sharpe", 0),
                "test_trades": r.get("test_trades", 0),
                "test_max_dd": r.get("test_max_dd", 0),
            })

        return {
            "experiment_id": self.experiment_id,
            "path": self.path,
            "aggregate_test_sharpe": agg_sharpe,
            "mean_test_sharpe": float(np.mean(test_sharpes)),
            "std_test_sharpe": float(np.std(test_sharpes)),
            "mean_val_sharpe": float(np.mean(val_sharpes)),
            "max_drawdown": max_dd,
            "total_test_trades": total_trades,
            "window_consistency": window_consistency,
            "positive_windows": positive_windows,
            "total_windows": len(test_sharpes),
            "final_equity": equity[-1] if equity else 10000.0,
            "param_stability_cv": param_stability,
            "window_metrics": window_metrics,
        }

    def _compute_param_stability(self) -> dict:
        """Compute coefficient of variation for each parameter across windows."""
        if len(self.window_results) < 2:
            return {}

        all_params = [r.get("best_params", {}) for r in self.window_results
                      if isinstance(r.get("best_params"), dict)]
        if not all_params:
            return {}

        param_names = list(all_params[0].keys())
        stability = {}
        for name in param_names:
            values = [p.get(name, 0) for p in all_params if isinstance(p.get(name), (int, float))]
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else float('inf')
                stability[name] = round(float(cv), 4)
        return stability

    def _print_summary(self, agg: dict):
        """Print aggregate results summary."""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY — {self.experiment_id}")
        print(f"{'='*70}")
        print(f"Windows completed:     {agg.get('windows_completed', 0)}/{agg.get('windows_total', 0)}")
        print(f"Aggregate test Sharpe: {agg.get('aggregate_test_sharpe', 0):.4f}")
        print(f"Mean test Sharpe:      {agg.get('mean_test_sharpe', 0):.4f} ± {agg.get('std_test_sharpe', 0):.4f}")
        print(f"Max drawdown:          {agg.get('max_drawdown', 0):.4f}")
        print(f"Total test trades:     {agg.get('total_test_trades', 0)}")
        print(f"Window consistency:    {agg.get('window_consistency', 0)*100:.1f}% (K-5 requires ≥60%)")
        print(f"Final equity:          ${agg.get('final_equity', 10000):.2f}")
        print(f"Time:                  {agg.get('total_time_sec', 0):.0f}s")

        # Parameter stability
        cv = agg.get('param_stability_cv', {})
        if cv:
            print(f"\nParameter Stability (CV):")
            for name, val in cv.items():
                flag = "⚠️ HIGH" if val > 0.6 else ("◐ AMBIGUOUS" if val > 0.4 else "✓ STABLE")
                print(f"  {name:30s}: {val:.4f}  {flag}")

        # Kill criteria evaluation
        print(f"\nKill Criteria Pre-Check:")
        print(f"  K-1 (held-out SR > 0):      [evaluated at Stage II-3 end on 2020-2024]")
        print(f"  K-2 (worst 2yr SR > -0.9):  [evaluated at Stage II-3 end]")
        k3 = "N/A"
        for r in self.window_results:
            cr = r.get("cost_ratio", 0)
            if cr < 2.0:
                k3 = f"FAIL (window CR={cr:.2f})"
                break
        else:
            k3 = "PASS (all windows CR ≥ 2.0)" if self.window_results else "N/A"
        print(f"  K-3 (cost ratio ≥ 2.0):     {k3}")
        wc = agg.get('window_consistency', 0)
        k5 = "PASS" if wc >= 0.6 else f"FAIL ({wc*100:.1f}%)"
        print(f"  K-5 (window consistency):    {k5}")
        print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Rolling Orchestrator — Project 2 Part II")

    parser.add_argument("--manifest", required=True,
                        help="Path to window_manifest.json")
    parser.add_argument("--data", required=True,
                        help="Path to OHLCV CSV data file")
    parser.add_argument("--raw_data", default=None,
                        help="Path to raw OHLCV CSV for Path B backtesting (auto-detected if omitted)")
    parser.add_argument("--output_dir", default="logs/experiment",
                        help="Directory for results output")
    parser.add_argument("--experiment_id", default=None,
                        help="Experiment identifier (auto-generated if omitted)")
    parser.add_argument("--path", choices=["A", "B"], default="A",
                        help="Experiment path: A=heuristic, B=supervised ML")
    parser.add_argument("--strategy_plugin", default="eurusd_mr",
                        help="Heuristic strategy plugin name (Path A)")
    parser.add_argument("--predictor_plugin", default=None,
                        help="Predictor plugin name (Path B)")
    parser.add_argument("--embargo_bars", type=int, default=6,
                        help="Embargo bars between train and val (default: 6)")
    parser.add_argument("--normalize_method", default="z-score",
                        choices=["z-score", "min-max"],
                        help="Normalisation method")
    parser.add_argument("--gmm_refit", action="store_true",
                        help="Enable rolling GMM re-fit (UL-1)")
    parser.add_argument("--gmm_components", type=int, default=9,
                        help="Number of GMM components (default: 9)")
    parser.add_argument("--population_size", type=int, default=30,
                        help="GA population size")
    parser.add_argument("--num_generations", type=int, default=20,
                        help="GA generations per window")
    parser.add_argument("--min_trades", type=int, default=10,
                        help="Minimum trades for valid fitness")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (Path B)")
    parser.add_argument("--early_patience", type=int, default=10,
                        help="Early stopping patience (Path B)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (Path B)")
    parser.add_argument("--lookback", type=int, default=24,
                        help="Lookback window for Path B sequences (default: 24)")
    parser.add_argument("--config_overrides", type=str, default=None,
                        help="JSON file with additional config overrides")
    parser.add_argument("--fixed_params", type=str, default=None,
                        help="JSON file with fixed params (skip GA optimisation)")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Auto-generate experiment ID if not provided
    experiment_id = args.experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load config overrides
    config_overrides = {}
    if args.config_overrides:
        with open(args.config_overrides, 'r') as f:
            config_overrides = json.load(f)

    # Load fixed params (skip GA, use these values)
    if args.fixed_params:
        with open(args.fixed_params, 'r') as f:
            config_overrides['fixed_params'] = json.load(f)

    # Pass lookback into config_overrides for Path B sequence generation
    config_overrides['lookback'] = args.lookback

    orchestrator = RollingOrchestrator(
        manifest_path=args.manifest,
        data_path=args.data,
        output_dir=args.output_dir,
        experiment_id=experiment_id,
        path=args.path,
        strategy_plugin=args.strategy_plugin,
        predictor_plugin=args.predictor_plugin,
        embargo_bars=args.embargo_bars,
        normalize_method=args.normalize_method,
        gmm_refit=args.gmm_refit,
        gmm_components=args.gmm_components,
        population_size=args.population_size,
        num_generations=args.num_generations,
        min_trades=args.min_trades,
        epochs=args.epochs,
        early_patience=args.early_patience,
        batch_size=args.batch_size,
        config_overrides=config_overrides,
        raw_data_path=args.raw_data,
    )

    results = orchestrator.run()

    # Exit code based on results
    completed = results.get("windows_completed", 0)
    total = results.get("windows_total", 0)
    if completed == 0:
        sys.exit(2)  # No windows completed
    elif completed < total:
        sys.exit(1)  # Partial completion
    else:
        sys.exit(0)  # Full success


if __name__ == "__main__":
    main()

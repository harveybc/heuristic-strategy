#!/usr/bin/env python3
"""
Phase 4 — Track B: Test the Regime-Filter Hypothesis

Task B.1: Define perfect regime oracle (lookahead-based)
Task B.2: Apply regime oracle to top 5 cells with 4 variants (V0-V3)
Task B.3: Determine rescue viability
Task B.4: Build real regime classifier from feature store data
Task B.5: Decision

Distributed: designed for dragon (main compute), omega (consolidation).
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.oracle_sensitivity import download_asset_data, generate_oracle_signal
from trading_research.transaction_cost_model import apply_cost_to_returns
from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation,
    periods_per_year_for_timeframe
)
from trading_research.audit_noise_budget import get_strategy_positions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FEATURE_STORE = os.path.join(os.path.dirname(__file__), "feature_store")
EXTENDED_DATA = os.path.join(os.path.dirname(__file__), "extended_data")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Extended start dates
EXTENDED_START = {
    "BTC/USD": "2014-09-17", "ETH/USD": "2017-08-01",
    "XAU/USD": "2000-01-01", "XAG/USD": "2000-01-01", "CL": "2000-01-01",
    "EUR/USD": "2003-01-01", "USD/JPY": "2003-01-01", "GBP/USD": "2003-01-01",
    "AUD/USD": "2003-01-01",
}

YFINANCE_TICKERS = {
    "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD",
    "XAU/USD": "GC=F", "XAG/USD": "SI=F", "CL": "CL=F",
    "EUR/USD": "EURUSD=X", "USD/JPY": "USDJPY=X", "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
}

# Top 5 cells for regime filter test
TOP_CELLS = [
    {"asset": "XAU/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "BTC/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "XAU/USD", "timeframe": "daily", "strategy": "momentum"},
    {"asset": "XAG/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "EUR/USD", "timeframe": "daily", "strategy": "mean_reversion"},
]


def load_extended_data(asset, timeframe):
    """Load extended price data — try cached CSV first, then download."""
    import yfinance as yf
    safe = asset.replace("/", "_")
    csv_path = os.path.join(EXTENDED_DATA, f"{safe}_{timeframe}.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if len(df) > 100:
            return df

    ticker = YFINANCE_TICKERS.get(asset)
    if not ticker:
        return None
    start = EXTENDED_START.get(asset, "2003-01-01")
    interval = {"weekly": "1wk", "daily": "1d"}.get(timeframe, "1d")

    try:
        df = yf.download(ticker, start=start, end="2025-12-31",
                         interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 0:
            os.makedirs(EXTENDED_DATA, exist_ok=True)
            df.to_csv(csv_path)
        return df if len(df) > 100 else None
    except Exception as e:
        print(f"    Download failed for {asset}: {e}")
        return None


# ============================================================
# TASK B.1 — Perfect regime oracle (uses future information)
# ============================================================
def compute_regime_labels(net_returns, ppy, window_years=2.0, step_bars=1):
    """
    For each bar, compute the Sharpe of the next `window_years` window.
    Labels:
      +1 = favorable (Sharpe ≥ +0.3)
       0 = neutral
      -1 = hostile (Sharpe < -0.3)
    Uses FUTURE information — this is the ceiling, not deployable.
    """
    n = len(net_returns)
    window_bars = int(window_years * ppy)
    labels = np.zeros(n)

    for i in range(n):
        end = i + window_bars
        if end > n:
            # Not enough future data — mark neutral
            labels[i] = 0
            continue
        window_ret = net_returns[i:end]
        sr = annualized_sharpe(window_ret, ppy)
        if sr >= 0.3:
            labels[i] = 1
        elif sr < -0.3:
            labels[i] = -1
        else:
            labels[i] = 0

    return labels


# ============================================================
# TASK B.2 — Apply regime oracle with V0/V1/V2/V3 variants
# ============================================================
def apply_variant(positions, regime_labels, variant):
    """Apply regime filter variant to positions."""
    filtered = positions.copy()
    if variant == "V0":
        return filtered  # no filter
    elif variant == "V1":
        # Stand-aside: flat when hostile
        filtered[regime_labels == -1] = 0
        return filtered
    elif variant == "V2":
        # Reverse in hostile, flat in neutral
        hostile = regime_labels == -1
        filtered[hostile] *= -1  # reverse
        return filtered
    elif variant == "V3":
        # Full in favorable, half in neutral, flat in hostile
        hostile = regime_labels == -1
        neutral = regime_labels == 0
        filtered[hostile] = 0
        filtered[neutral] *= 0.5
        return filtered
    return filtered


def evaluate_variant(log_ret, positions, abs_ret, asset, ppy):
    """Compute metrics for a filtered variant."""
    gross = positions[:-1] * log_ret[1:]
    net = apply_cost_to_returns(gross, positions[:-1], asset, abs_ret[:-1])

    sr = annualized_sharpe(net, ppy)
    bh_sr = annualized_sharpe(log_ret[1:], ppy)
    rolling = rolling_window_evaluation(net, ppy)

    # Max drawdown
    equity = np.cumsum(net)
    eq_curve = np.exp(equity)
    peak = np.maximum.accumulate(eq_curve)
    dd = (peak - eq_curve) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    # Trades per year
    pos_changes = np.abs(np.diff(positions))
    n_trades = np.sum(pos_changes > 0)
    n_years = len(log_ret) / ppy
    trades_per_year = n_trades / n_years if n_years > 0 else 0

    return {
        "sharpe": round(sr, 4),
        "bh_sharpe": round(bh_sr, 4),
        "edge_sharpe": round(sr - bh_sr, 4),
        "regime_robustness": rolling["regime_robustness"],
        "worst_window_sharpe": rolling["worst_window_sharpe"],
        "max_drawdown": round(max_dd, 4),
        "trades_per_year": round(trades_per_year, 1),
        "n_bars": len(log_ret),
    }


def run_cell_regime_test(cell, seed=42):
    """Full regime filter test for one cell."""
    asset = cell["asset"]
    timeframe = cell["timeframe"]
    strategy = cell["strategy"]

    print(f"\n  {asset} / {timeframe} / {strategy}:")

    df = load_extended_data(asset, timeframe)
    if df is None:
        return {"error": "no data", **cell}

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    ppy = periods_per_year_for_timeframe(timeframe)
    abs_ret = np.abs(log_ret)

    # Generate oracle signal (noise=0 — perfect foresight)
    sig_0 = generate_oracle_signal(log_ret, noise_sigma=0.0, horizon=1, seed=seed)
    base_positions = get_strategy_positions(strategy, log_ret, sig_0, close)

    # Compute base net returns for regime label generation
    base_gross = base_positions[:-1] * log_ret[1:]
    base_net = apply_cost_to_returns(base_gross, base_positions[:-1], asset, abs_ret[:-1])

    # Task B.1: Compute regime labels (base_net has n-1 entries, pad to match positions)
    regime_labels_short = compute_regime_labels(base_net, ppy)
    # Pad with 0 (neutral) at front to align with positions array
    regime_labels = np.concatenate([[0], regime_labels_short])

    n_favorable = int(np.sum(regime_labels == 1))
    n_hostile = int(np.sum(regime_labels == -1))
    n_neutral = int(np.sum(regime_labels == 0))
    print(f"    Regime labels: {n_favorable} favorable, {n_neutral} neutral, {n_hostile} hostile "
          f"(of {len(regime_labels)} bars)")

    # Task B.2: Apply V0-V3
    variants = {}
    for v_name in ["V0", "V1", "V2", "V3"]:
        v_positions = apply_variant(base_positions, regime_labels, v_name)
        metrics = evaluate_variant(log_ret, v_positions, abs_ret, asset, ppy)
        variants[v_name] = metrics
        print(f"    {v_name}: SR={metrics['sharpe']:+.3f}, Edge={metrics['edge_sharpe']:+.3f}, "
              f"Worst2Y={metrics['worst_window_sharpe']:+.3f}, DD={metrics['max_drawdown']:.1%}, "
              f"Trades/yr={metrics['trades_per_year']:.0f}")

    # Task B.3: Rescue analysis
    v0_worst = variants["V0"]["worst_window_sharpe"]
    rescued_by = []
    for v_name in ["V1", "V2", "V3"]:
        v = variants[v_name]
        if (v["worst_window_sharpe"] > -0.5 and
            v["edge_sharpe"] > 0.5 and
            v["trades_per_year"] > variants["V0"]["trades_per_year"] * 0.3):
            rescued_by.append(v_name)

    is_rescued = len(rescued_by) > 0
    print(f"    Rescue: {'YES by ' + ', '.join(rescued_by) if is_rescued else 'NO — none pass all 3 criteria'}")

    # Also report at relaxed thresholds (-0.75, -1.0)
    rescued_075 = [v for v in ["V1", "V2", "V3"]
                   if variants[v]["worst_window_sharpe"] > -0.75 and variants[v]["edge_sharpe"] > 0.5]
    rescued_100 = [v for v in ["V1", "V2", "V3"]
                   if variants[v]["worst_window_sharpe"] > -1.0 and variants[v]["edge_sharpe"] > 0.5]

    return {
        **cell,
        "n_bars": len(log_ret),
        "regime_distribution": {
            "favorable": n_favorable,
            "neutral": n_neutral,
            "hostile": n_hostile,
        },
        "variants": variants,
        "rescued_at_minus_05": rescued_by,
        "rescued_at_minus_075": rescued_075,
        "rescued_at_minus_100": rescued_100,
        "is_rescued": is_rescued,
    }


# ============================================================
# TASK B.4 — Real regime classifier
# ============================================================
def build_regime_classifier(cell, regime_labels_cache=None, seed=42):
    """
    For cells where oracle rescues, try to build a real regime classifier
    using feature store data (macro + COT + technical).
    Uses logistic regression + gradient boosting.
    """
    asset = cell["asset"]
    timeframe = cell["timeframe"]
    safe = asset.replace("/", "_")

    # Load feature store daily data
    feat_path = os.path.join(FEATURE_STORE, f"{safe}_daily.csv")
    if not os.path.exists(feat_path):
        return {"error": f"No feature store for {asset}"}

    feat_df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # Load extended price data for regime labels
    df = load_extended_data(asset, timeframe)
    if df is None:
        return {"error": "no price data"}

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    ppy = periods_per_year_for_timeframe(timeframe)
    abs_ret = np.abs(log_ret)

    sig_0 = generate_oracle_signal(log_ret, noise_sigma=0.0, horizon=1, seed=seed)
    positions = get_strategy_positions(cell["strategy"], log_ret, sig_0, close)
    gross = positions[:-1] * log_ret[1:]
    net = apply_cost_to_returns(gross, positions[:-1], asset, abs_ret[:-1])
    regime_labels_short = compute_regime_labels(net, ppy)
    # Pad with 0 (neutral) at front to align with price_dates
    regime_labels = np.concatenate([[0], regime_labels_short])

    # Align features with price data
    price_dates = df.index
    labels_series = pd.Series(regime_labels, index=price_dates, name="regime")

    # For weekly timeframe, need weekly features — resample daily
    if timeframe == "weekly":
        feat_weekly = feat_df.resample("W").last()
        merged = labels_series.to_frame().join(feat_weekly, how="inner")
    else:
        merged = labels_series.to_frame().join(feat_df, how="inner")

    if len(merged) < 100:
        return {"error": f"insufficient aligned data: {len(merged)} rows"}

    # Prepare features
    feature_cols = [c for c in merged.columns if c != "regime"
                    and not c.startswith("Open") and not c.startswith("High")
                    and not c.startswith("Low") and not c.startswith("Volume")]
    X = merged[feature_cols].copy()
    y = merged["regime"].copy()

    # Drop rows with NaN
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    if len(X) < 100:
        return {"error": f"insufficient valid data after NaN drop: {len(X)} rows"}

    # Binary: hostile vs not-hostile
    y_binary = (y == -1).astype(int)

    # Time-series split: train on first 70%, test on last 30%
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_binary.iloc[:split_idx], y_binary.iloc[split_idx:]

    print(f"    Classifier: {len(X_train)} train, {len(X_test)} test, "
          f"{int(y_train.sum())} hostile in train, {int(y_test.sum())} hostile in test")

    results = {}

    # Simple fill for NaNs
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # 1. Logistic Regression
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr.fit(X_train_sc, y_train)
        y_pred_lr = lr.predict(X_test_sc)

        results["logistic_regression"] = {
            "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred_lr), 4),
            "precision": round(precision_score(y_test, y_pred_lr, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_lr, zero_division=0), 4),
            "n_test": len(y_test),
            "n_hostile_test": int(y_test.sum()),
            "n_predicted_hostile": int(y_pred_lr.sum()),
        }
        print(f"      LR: bal_acc={results['logistic_regression']['balanced_accuracy']:.3f}, "
              f"prec={results['logistic_regression']['precision']:.3f}, "
              f"recall={results['logistic_regression']['recall']:.3f}")
    except ImportError:
        results["logistic_regression"] = {"error": "sklearn not available"}
        print("      LR: sklearn not available")

    # 2. Gradient Boosting
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed, subsample=0.8
        )
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)

        results["gradient_boosting"] = {
            "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred_gb), 4),
            "precision": round(precision_score(y_test, y_pred_gb, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_gb, zero_division=0), 4),
            "n_test": len(y_test),
            "n_hostile_test": int(y_test.sum()),
            "n_predicted_hostile": int(y_pred_gb.sum()),
        }
        print(f"      GB: bal_acc={results['gradient_boosting']['balanced_accuracy']:.3f}, "
              f"prec={results['gradient_boosting']['precision']:.3f}, "
              f"recall={results['gradient_boosting']['recall']:.3f}")

        # Feature importance
        importances = gb.feature_importances_
        top_idx = np.argsort(importances)[-5:][::-1]
        results["top_features"] = [
            {"name": feature_cols[i], "importance": round(float(importances[i]), 4)}
            for i in top_idx
        ]
        print(f"      Top features: {[f['name'] for f in results['top_features']]}")
    except ImportError:
        results["gradient_boosting"] = {"error": "sklearn not available"}

    # 3. Strategy performance with real classifier
    try:
        # Use GB predictions on test set
        if "error" not in results.get("gradient_boosting", {}):
            # Map predictions back to time series
            test_dates = X_test.index
            pred_labels = np.zeros(len(log_ret))
            for date, pred in zip(test_dates, y_pred_gb):
                # Find matching bar in price data
                matches = np.where(price_dates >= date)[0]
                if len(matches) > 0:
                    idx = matches[0]
                    if idx < len(pred_labels):
                        pred_labels[idx] = -1 if pred == 1 else 0

            # Apply V1 (stand-aside) with real classifier
            real_positions = positions.copy()
            real_positions[pred_labels == -1] = 0

            real_gross = real_positions[:-1] * log_ret[1:]
            real_net = apply_cost_to_returns(real_gross, real_positions[:-1], asset, abs_ret[:-1])
            real_sr = annualized_sharpe(real_net, ppy)
            real_rolling = rolling_window_evaluation(real_net, ppy)

            results["real_classifier_v1"] = {
                "sharpe": round(real_sr, 4),
                "worst_window_sharpe": real_rolling["worst_window_sharpe"],
                "regime_robustness": real_rolling["regime_robustness"],
            }
            print(f"      Real V1: SR={real_sr:+.3f}, worst2Y={real_rolling['worst_window_sharpe']:+.3f}")
    except Exception as e:
        results["real_classifier_v1"] = {"error": str(e)}

    return results


def main():
    print("=" * 70)
    print("PHASE 4 — TRACK B: REGIME FILTER HYPOTHESIS TEST")
    print("=" * 70)

    # Tasks B.1-B.3: Run regime oracle test for all 5 cells
    print("\n--- Tasks B.1-B.3: Perfect Regime Oracle Test ---")
    cell_results = []
    for cell in TOP_CELLS:
        result = run_cell_regime_test(cell)
        cell_results.append(result)

    # Task B.4: Real classifier for rescued cells
    print("\n--- Task B.4: Real Regime Classifier ---")
    classifier_results = {}
    for result in cell_results:
        if result.get("is_rescued") or True:  # Run for all to see signal quality
            cell_key = f"{result['asset']}_{result['timeframe']}_{result['strategy']}"
            print(f"\n  Classifier for {result['asset']}/{result['timeframe']}/{result['strategy']}:")
            clf = build_regime_classifier(result)
            classifier_results[cell_key] = clf

    # Task B.5: Decision
    print("\n" + "=" * 70)
    print("TRACK B — TASK B.5: DECISION")
    print("=" * 70)

    n_rescued = sum(1 for r in cell_results if r.get("is_rescued", False))
    n_rescued_075 = sum(1 for r in cell_results if len(r.get("rescued_at_minus_075", [])) > 0)
    n_rescued_100 = sum(1 for r in cell_results if len(r.get("rescued_at_minus_100", [])) > 0)

    print(f"\n  Cells rescued at -0.5 threshold: {n_rescued} / {len(cell_results)}")
    print(f"  Cells rescued at -0.75 threshold: {n_rescued_075} / {len(cell_results)}")
    print(f"  Cells rescued at -1.0 threshold: {n_rescued_100} / {len(cell_results)}")

    # Check classifier quality
    best_bal_acc = 0
    for ck, cv in classifier_results.items():
        for model in ["logistic_regression", "gradient_boosting"]:
            if model in cv and "balanced_accuracy" in cv[model]:
                best_bal_acc = max(best_bal_acc, cv[model]["balanced_accuracy"])

    print(f"  Best classifier balanced accuracy: {best_bal_acc:.3f}")

    # Decision logic
    if n_rescued >= 1 and best_bal_acc >= 0.55:
        decision = "REGIME_FILTER_VIABLE"
        notes = f"{n_rescued} cells rescued by perfect oracle, classifier bal_acc={best_bal_acc:.3f} ≥ 0.55"
    elif n_rescued >= 1 and best_bal_acc < 0.55:
        decision = "REGIME_DETECTABLE_BUT_NOT_CLASSIFIABLE"
        notes = f"{n_rescued} cells rescued but classifier bal_acc={best_bal_acc:.3f} < 0.55"
    elif n_rescued == 0 and n_rescued_100 >= 1:
        decision = "REGIME_FILTER_MARGINAL"
        notes = f"No cells rescued at -0.5 but {n_rescued_100} at -1.0 — threshold recalibration may help"
    else:
        decision = "REGIME_FILTER_NOT_VIABLE"
        notes = f"Perfect oracle cannot rescue strategies — problem is structural, not regime-based"

    print(f"\n  DECISION: {decision}")
    print(f"  {notes}")

    # Save results
    output = {
        "track": "B",
        "decision": decision,
        "decision_notes": notes,
        "n_cells_tested": len(cell_results),
        "n_rescued_at_minus_05": n_rescued,
        "n_rescued_at_minus_075": n_rescued_075,
        "n_rescued_at_minus_100": n_rescued_100,
        "best_classifier_balanced_accuracy": best_bal_acc,
        "cell_results": cell_results,
        "classifier_results": {k: v for k, v in classifier_results.items()},
    }

    output_path = os.path.join(RESULTS_DIR, "phase4_track_b_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()

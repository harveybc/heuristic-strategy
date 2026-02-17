#!/usr/bin/env python3
"""
Noise sensitivity analysis with BOTH trade frequency modes:
- High frequency: max_trades_per_5days=20 (unlimited)
- Low frequency: max_trades_per_5days=3 (pattern day trading limit)

Runs on 1h bars with prediction_interval_minutes=240 (4h predictions).
Uses bars_per_day=6 for daily prediction spacing (4h data).

Produces: CSV results + SQLite OLAP for Metabase.
"""
import csv, datetime, json, os, sys, sqlite3, time
import numpy as np
import pandas as pd
import backtrader as bt
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from app.heuristic_strategy import HeuristicStrategy

OUTPUT_DIR = REPO / "examples" / "results" / "noise_sensitivity_4h"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data files (1h bars for realistic simulation)
BASE_FILE = REPO / "examples/data/phase_1/phase_1_base_d3.csv"

# Noise levels to sweep
NOISE_LEVELS = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01]

# Two frequency modes
FREQ_MODES = {
    "high_freq": {"max_trades_per_5days": 20, "label": "High Frequency (unlimited)"},
    "low_freq": {"max_trades_per_5days": 3, "label": "Low Frequency (PDT limit)"},
}

# Base strategy params
BASE_PARAMS = {
    "pip_cost": 0.00001,
    "rel_volume": 0.02,
    "min_order_volume": 10000,
    "max_order_volume": 1000000,
    "leverage": 1000,
    "profit_threshold": 5,
    "min_drawdown_pips": 10,
    "tp_multiplier": 0.9,
    "sl_multiplier": 2.0,
    "lower_rr_threshold": 0.5,
    "upper_rr_threshold": 2.0,
    "bars_per_day": 6,  # 4h prediction spacing
    "bar_compression_minutes": 60,  # 1h market bars
    "prediction_interval_minutes": 240,  # 4h prediction refresh
}

TIME_HORIZON = 6  # 6 days
BARS_PER_DAY = 6  # For 4h daily predictions


def load_base_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["DATE_TIME"])
    df.set_index("DATE_TIME", inplace=True)
    return df


def create_ideal_hourly_predictions(base_df, horizon):
    """Short-term: 1-bar-ahead predictions (1h each)."""
    close = base_df["CLOSE"] if "CLOSE" in base_df.columns else base_df.iloc[:, -1]
    blocks = []
    for i in range(len(close) - horizon):
        block = close.iloc[i + 1: i + 1 + horizon].values.flatten()
        blocks.append(block)
    cols = [f"Prediction_h_{j+1}" for j in range(horizon)]
    return pd.DataFrame(blocks, index=close.index[:-horizon], columns=cols)


def create_ideal_daily_predictions(base_df, horizon, bars_per_day=6):
    """Long-term: bars_per_day-spaced predictions."""
    close = base_df["CLOSE"] if "CLOSE" in base_df.columns else base_df.iloc[:, -1]
    nrows = len(close)
    required = horizon * bars_per_day
    if nrows < required:
        return pd.DataFrame()
    blocks = []
    for i in range(nrows - required):
        block = []
        for d in range(1, horizon + 1):
            idx = i + d * bars_per_day
            if idx < nrows:
                block.append(close.iloc[idx])
        if block:
            blocks.append(block)
    cols = [f"Prediction_d_{j+1}" for j in range(len(blocks[0]))]
    return pd.DataFrame(blocks, index=close.index[:len(blocks)], columns=cols)


def add_noise(df, std):
    if df is None or df.empty or std == 0.0:
        return df.copy() if df is not None else df
    noise = np.random.normal(0.0, std, size=df.shape)
    noisy = df.copy()
    noisy[:] = noisy.values + noise
    return noisy


def run_backtest(base_data, hourly_preds, daily_preds, params):
    """Run a single backtest, return metrics."""
    merged = pd.DataFrame()
    if hourly_preds is not None and not hourly_preds.empty:
        for col in hourly_preds.columns:
            merged[col] = hourly_preds[col]
    if daily_preds is not None and not daily_preds.empty:
        for col in daily_preds.columns:
            merged[col] = daily_preds[col]

    if merged.empty:
        return {"profit": 0, "num_trades": 0, "win_pct": 0, "max_dd": 0}

    merged.index.name = "DATE_TIME"
    temp_file = f"/tmp/noise_sens_{os.getpid()}.csv"
    merged.reset_index().to_csv(temp_file, index=False)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        HeuristicStrategy,
        pred_file=temp_file,
        pip_cost=params["pip_cost"],
        rel_volume=params["rel_volume"],
        min_order_volume=params["min_order_volume"],
        max_order_volume=params["max_order_volume"],
        leverage=params["leverage"],
        profit_threshold=params["profit_threshold"],
        min_drawdown_pips=params["min_drawdown_pips"],
        tp_multiplier=params["tp_multiplier"],
        sl_multiplier=params["sl_multiplier"],
        lower_rr_threshold=params["lower_rr_threshold"],
        upper_rr_threshold=params["upper_rr_threshold"],
        max_trades_per_5days=params["max_trades_per_5days"],
        bars_per_day=params["bars_per_day"],
        bar_compression_minutes=params["bar_compression_minutes"],
        prediction_interval_minutes=params["prediction_interval_minutes"],
        date_start=params["date_start"],
        date_end=params["date_end"],
    )

    data_feed = bt.feeds.PandasData(dataname=base_data)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(10000.0)

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        results = cerebro.run()

    strat = results[0]
    trades = strat.trades
    final_balance = cerebro.broker.getvalue()
    profit = final_balance - 10000.0
    num_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_pct = (wins / num_trades * 100) if num_trades > 0 else 0
    max_dd = max((t["max_dd"] for t in trades), default=0)

    if os.path.exists(temp_file):
        os.remove(temp_file)

    return {"profit": profit, "num_trades": num_trades, "win_pct": win_pct, "max_dd": max_dd}


def main():
    print("Loading base data...")
    base_data = load_base_data(str(BASE_FILE))

    # Set date range from actual data
    BASE_PARAMS["date_start"] = base_data.index.min().to_pydatetime()
    BASE_PARAMS["date_end"] = base_data.index.max().to_pydatetime()

    print("Generating ideal predictions...")
    hourly_ideal = create_ideal_hourly_predictions(base_data, TIME_HORIZON)
    daily_ideal = create_ideal_daily_predictions(base_data, TIME_HORIZON, bars_per_day=BARS_PER_DAY)

    # Align
    common = base_data.index.intersection(hourly_ideal.index).intersection(daily_ideal.index)
    base_aligned = base_data.loc[common]
    hourly_ideal = hourly_ideal.loc[common]
    daily_ideal = daily_ideal.loc[common]
    print(f"Aligned: {len(common)} rows, hourly={hourly_ideal.shape[1]}cols, daily={daily_ideal.shape[1]}cols")

    all_results = []

    for freq_name, freq_config in FREQ_MODES.items():
        print(f"\n{'='*60}")
        print(f"Mode: {freq_config['label']} (max_trades_per_5days={freq_config['max_trades_per_5days']})")
        print(f"{'='*60}")

        params = {**BASE_PARAMS, **freq_config}

        # Cross-sensitivity: hourly × daily noise
        noise_grid = list(product(NOISE_LEVELS, NOISE_LEVELS))
        print(f"Grid: {len(NOISE_LEVELS)}×{len(NOISE_LEVELS)} = {len(noise_grid)} combos")

        for i, (h_noise, d_noise) in enumerate(noise_grid):
            np.random.seed(42)
            hourly_noisy = add_noise(hourly_ideal, h_noise)
            daily_noisy = add_noise(daily_ideal, d_noise)

            result = run_backtest(base_aligned, hourly_noisy, daily_noisy, params)
            result["freq_mode"] = freq_name
            result["max_trades_per_5days"] = freq_config["max_trades_per_5days"]
            result["hourly_noise"] = h_noise
            result["daily_noise"] = d_noise
            all_results.append(result)

            tag = f"[{i+1}/{len(noise_grid)}]"
            print(f"  {tag} h={h_noise:.4f} d={d_noise:.4f} → ${result['profit']:.0f}, "
                  f"{result['num_trades']}t, {result['win_pct']:.0f}%w")

    # Save CSV
    csv_file = OUTPUT_DIR / "noise_sensitivity_dual_freq.csv"
    fieldnames = ["freq_mode", "max_trades_per_5days", "hourly_noise", "daily_noise",
                  "profit", "num_trades", "win_pct", "max_dd"]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved: {csv_file}")

    # SQLite OLAP
    db_file = OUTPUT_DIR / "noise_sensitivity_4h_olap.db"
    conn = sqlite3.connect(db_file)
    conn.execute("DROP TABLE IF EXISTS experiments")
    conn.execute("""CREATE TABLE experiments (
        freq_mode TEXT, max_trades_per_5days INT,
        hourly_noise REAL, daily_noise REAL,
        profit REAL, num_trades INT, win_pct REAL, max_dd REAL
    )""")
    for r in all_results:
        conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?)",
                     (r["freq_mode"], r["max_trades_per_5days"],
                      r["hourly_noise"], r["daily_noise"],
                      r["profit"], r["num_trades"], r["win_pct"], r["max_dd"]))

    # Create marginal views
    for freq in FREQ_MODES:
        conn.execute(f"""CREATE VIEW IF NOT EXISTS v_{freq}_daily_marginal AS
            SELECT daily_noise, AVG(profit) as avg_profit, AVG(num_trades) as avg_trades,
                   AVG(win_pct) as avg_win
            FROM experiments WHERE freq_mode='{freq}'
            GROUP BY daily_noise ORDER BY daily_noise""")
        conn.execute(f"""CREATE VIEW IF NOT EXISTS v_{freq}_hourly_marginal AS
            SELECT hourly_noise, AVG(profit) as avg_profit, AVG(num_trades) as avg_trades,
                   AVG(win_pct) as avg_win
            FROM experiments WHERE freq_mode='{freq}'
            GROUP BY hourly_noise ORDER BY hourly_noise""")

    conn.commit()
    conn.close()
    print(f"OLAP saved: {db_file}")
    print(f"\nTotal experiments: {len(all_results)}")


if __name__ == "__main__":
    main()

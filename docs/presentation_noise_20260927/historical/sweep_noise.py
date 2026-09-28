#!/usr/bin/env python3
"""
Noise sweep for direction_ideal_oracle.

For each noise_std level:
  1. POST /api/v1/predict/set_noise  → set noise & reset F1 counters
  2. Run backtest via plugin.evaluate_candidate()
  3. GET  /api/v1/predict/metrics    → read F1/precision/recall
  4. Record [noise_std, F1, profit, win%, trades, sharpe, max_dd, ...]

Saves results to sweep_noise_results.csv and prints the "naive point"
(minimum F1 at which profit > 0).
"""
import sys, os, csv, time, requests

sys.path.insert(0, "/home/harveybc/Documents/GitHub/heuristic-strategy")
os.chdir("/home/harveybc/Documents/GitHub/heuristic-strategy")
os.environ["STRATEGY_QUIET"] = "1"

from app.data_processor import process_data
from app.plugins.plugin_direction_atr import Plugin

PP_URL = "http://127.0.0.1:8000"

# Fixed strategy params (ATR=14, TP=2.0, SL=1.0 — good mid-range)
ATR_PERIOD = 14
TP_MULT = 2.0
SL_MULT = 1.0

# Noise levels: from perfect (0.0) to very noisy (2.0)
NOISE_LEVELS = [
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00,
    1.20, 1.50, 2.00,
]

config = {
    "base_dataset_file": "tests/data/phase_2_3_base_d3.csv",
    "prediction_source": "API",
    "pp_api_url": PP_URL,
    "pp_timeout": 10.0,
    "headers": True,
}

print("Loading data...")
datasets = process_data(config)
base_data = datasets["base"]
print(f"Data loaded: {base_data.shape}")

results = []
total = len(NOISE_LEVELS)

for i, noise_std in enumerate(NOISE_LEVELS, 1):
    # 1. Set noise on PP and reset metrics
    resp = requests.post(f"{PP_URL}/api/v1/predict/set_noise",
                         json={"noise_std": noise_std}, timeout=5)
    if resp.status_code != 200:
        print(f"[{i}/{total}] ERROR setting noise={noise_std}: {resp.text}")
        continue

    # 2. Run backtest
    plugin = Plugin()
    plugin.set_params(
        spread_pips=15.0,
        commission_per_lot=7.0,
        slippage_pips=5.0,
        swap_per_lot_per_day=10.0,
        leverage=100,
        rel_volume=0.10,
        max_trades_per_5days=5,
        exit_enabled=True,
    )
    candidate = [ATR_PERIOD, TP_MULT, SL_MULT]
    profit, stats = plugin.evaluate_candidate(
        individual=candidate,
        base_data=base_data,
        hourly_predictions=None,
        daily_predictions=None,
        config=config,
    )

    # 3. Read F1 metrics from PP
    resp = requests.get(f"{PP_URL}/api/v1/predict/metrics", timeout=5)
    metrics = resp.json() if resp.status_code == 200 else {}

    f1 = metrics.get("f1", -1)
    precision = metrics.get("precision", -1)
    recall = metrics.get("recall", -1)
    accuracy = metrics.get("accuracy", -1)
    tp_count = metrics.get("tp", 0)
    fp_count = metrics.get("fp", 0)
    tn_count = metrics.get("tn", 0)
    fn_count = metrics.get("fn", 0)

    row = {
        "noise_std": noise_std,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "profit": profit,
        "trades": stats["num_trades"],
        "win_pct": stats["win_pct"],
        "max_dd": stats["max_dd"],
        "sharpe": stats["sharpe"],
        "tp": tp_count,
        "fp": fp_count,
        "tn": tn_count,
        "fn": fn_count,
    }
    results.append(row)

    print(f"[{i}/{total}] noise_std={noise_std:.2f} => "
          f"F1={f1:.4f} Profit={profit:.0f} "
          f"Trades={stats['num_trades']} Win%={stats['win_pct']:.1f} "
          f"Sharpe={stats['sharpe']:.1f} "
          f"Prec={precision:.4f} Rec={recall:.4f}")

# Save CSV
outfile = "/home/harveybc/Documents/GitHub/heuristic-strategy/sweep_noise_results.csv"
fieldnames = ["noise_std", "f1", "precision", "recall", "accuracy",
              "profit", "trades", "win_pct", "max_dd", "sharpe",
              "tp", "fp", "tn", "fn"]
with open(outfile, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)
print(f"\nSaved to {outfile}")

# Find naive point (first noise level where profit <= 0)
print("\n" + "=" * 70)
print("NOISE SWEEP SUMMARY")
print("=" * 70)
print(f"{'noise_std':>10} {'F1':>8} {'Profit':>12} {'Trades':>7} {'Win%':>7} {'Sharpe':>8}")
print("-" * 70)

naive_f1 = None
for r in results:
    marker = ""
    if naive_f1 is None and r["profit"] <= 0:
        naive_f1 = r["f1"]
        marker = " <-- NAIVE POINT"
    print(f"{r['noise_std']:>10.2f} {r['f1']:>8.4f} {r['profit']:>12.0f} "
          f"{r['trades']:>7} {r['win_pct']:>7.1f} {r['sharpe']:>8.1f}{marker}")

print("-" * 70)
if naive_f1 is not None:
    # Find the last profitable row
    last_profitable = None
    for r in results:
        if r["profit"] > 0:
            last_profitable = r
    if last_profitable:
        print(f"\nLast profitable: noise_std={last_profitable['noise_std']:.2f}, "
              f"F1={last_profitable['f1']:.4f}, Profit={last_profitable['profit']:.0f}")
    print(f"Naive point:     F1 = {naive_f1:.4f} (first noise level with profit <= 0)")
    print(f"\nTo be profitable, your model needs F1 > ~{naive_f1:.4f}")
else:
    print("\nAll noise levels were profitable! Try higher noise_std values.")

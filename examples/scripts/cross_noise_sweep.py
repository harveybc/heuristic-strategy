#!/usr/bin/env python3
"""Cross-sensitivity sweep: independent hourly and daily noise.

Uses gaussian_noise_hourly_stddev and gaussian_noise_daily_stddev config keys.

Usage:
    STRATEGY_QUIET=1 PYTHONPATH=./ python3 -u examples/scripts/cross_noise_sweep.py
"""

import csv, json, os, subprocess, sys, sqlite3, time
from itertools import product
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO / "examples" / "results" / "cross_sensitivity"
BASE_FILE = "examples/data/phase_1/phase_1_base_d3.csv"
LOAD_PARAMS = "examples/results/phase_2_1_daily/phase_2_1_ann_25200_1d_results_high_frequency_parameters.json"
NOISE_LEVELS = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01]


def run_one(h_noise, d_noise, output_dir):
    tag = f"h{h_noise:.4f}_d{d_noise:.4f}"
    summary_file = output_dir / f"summary_{tag}.csv"
    
    generic = max(h_noise, d_noise, 0.0001)
    args = [
        sys.executable, "app/main.py",
        "--base_dataset_file", BASE_FILE,
        "--max_trades_per_5days", "20",
        "--prefix", f"_{tag}",
        "--load_parameters", LOAD_PARAMS,
        "--gaussian_noise_hourly_mean", str(h_noise),
        "--gaussian_noise_hourly_stddev", str(h_noise),
        "--gaussian_noise_daily_mean", str(d_noise),
        "--gaussian_noise_daily_stddev", str(d_noise),
        "--gaussian_noise_mean", str(generic),
        "--gaussian_noise_stddev", str(generic),
        "--summary_csv_file", str(summary_file),
        "--trades_csv_file", str(output_dir / f"trades_{tag}.csv"),
        "--balance_plot_file", str(output_dir / f"balance_{tag}.png"),
        "--save_config", str(output_dir / f"config_{tag}.json"),
    ]

    env = os.environ.copy()
    env["STRATEGY_QUIET"] = "1"
    env["PYTHONPATH"] = str(REPO)

    result = subprocess.run(args, capture_output=True, text=True, cwd=str(REPO), env=env, timeout=600)

    if result.returncode != 0:
        return {"error": (result.stderr or result.stdout)[-100:]}

    # The strategy may ignore --summary_csv_file and use auto-derived path
    # Check both the explicit path and any recent summary with the tag
    if not summary_file.exists():
        # Strategy appends noise_suffix to filenames
        import glob
        pattern = str(output_dir / f"summary_{tag}*.csv")
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if candidates:
            summary_file = Path(candidates[0])

    if not summary_file.exists():
        return {"error": f"no summary at {summary_file.name}"}

    with open(summary_file) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"error": "empty summary"}
    
    r = rows[0]
    return {
        "profit": round(float(r.get("profit", 0)), 2),
        "num_trades": int(r.get("num_trades", 0)),
        "win_pct": round(float(r.get("win_pct", 0)), 1),
        "max_dd": round(float(r.get("max_dd", 0)), 2),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean old noise files from repo root
    for pattern in ["summary_gn_*", "config_out_gn_*", "trades_gn_*", "debug_log_gn_*", "parameters_gn_*", "balance_plot_gn_*"]:
        import glob
        for f in glob.glob(str(REPO / pattern)):
            os.remove(f)

    grid = list(product(NOISE_LEVELS, NOISE_LEVELS))
    print(f"Cross-sensitivity: {len(NOISE_LEVELS)}x{len(NOISE_LEVELS)} = {len(grid)} combos", flush=True)

    results = []
    t0 = time.time()
    for i, (h_noise, d_noise) in enumerate(grid):
        r = run_one(h_noise, d_noise, OUTPUT_DIR)
        r["hourly_noise"] = h_noise
        r["daily_noise"] = d_noise
        results.append(r)

        profit = r.get("profit", 0)
        trades = r.get("num_trades", 0)
        err = f" ERR:{r.get('error','')[:40]}" if "error" in r else ""
        tag = "+" if profit > 0 else "-"
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (len(grid) - i - 1) if i > 0 else 0
        print(f"  [{i+1}/{len(grid)}] h={h_noise:.3f} d={d_noise:.3f} ${profit:>8,.0f} t={trades:>3} {tag} ({eta:.0f}s){err}", flush=True)

    # Save CSV
    csv_path = OUTPUT_DIR / "cross_sensitivity_results.csv"
    keys = ["hourly_noise", "daily_noise", "profit", "num_trades", "win_pct", "max_dd"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # SQLite OLAP
    db_path = OUTPUT_DIR / "cross_sensitivity_olap.db"
    conn = sqlite3.connect(str(db_path))
    df = pd.DataFrame(results)
    valid = [k for k in keys if k in df.columns]
    df[valid].to_sql("cross_sensitivity", conn, if_exists="replace", index=False)
    for dim in ["daily", "hourly"]:
        conn.execute(f"DROP VIEW IF EXISTS v_{dim}_marginal")
        conn.execute(f"CREATE VIEW v_{dim}_marginal AS SELECT {dim}_noise, AVG(profit) avg_profit, AVG(num_trades) avg_trades, AVG(win_pct) avg_win FROM cross_sensitivity GROUP BY {dim}_noise")
    conn.commit()
    conn.close()

    ok = [r for r in results if "error" not in r]
    df = pd.DataFrame(ok) if ok else pd.DataFrame()
    print(f"\n{'='*60}", flush=True)
    print(f"Done: {len(ok)}/{len(grid)}  CSV: {csv_path}", flush=True)
    if not df.empty:
        print("\nDAILY marginal:", flush=True)
        for dn, g in df.groupby("daily_noise"):
            print(f"  d={dn:.3f}: ${g['profit'].mean():>8,.0f} t={g['num_trades'].mean():.0f} w={g['win_pct'].mean():.0f}%", flush=True)
        print("\nHOURLY marginal:", flush=True)
        for hn, g in df.groupby("hourly_noise"):
            print(f"  h={hn:.3f}: ${g['profit'].mean():>8,.0f} t={g['num_trades'].mean():.0f} w={g['win_pct'].mean():.0f}%", flush=True)


if __name__ == "__main__":
    main()

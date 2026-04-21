#!/usr/bin/env python3
"""
Merge WFO results from distributed machines.
Combines fold results from omega, dragon, and gamma into one report.
"""
import json
import sys
import os
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

# Result files to merge (machine → (json, csv))
RESULT_FILES = {
    "omega_2009_2010": {
        "log": "wfo_run_folds_2009_2010.log",  # parse from log
    },
    "omega_2011_2013": {
        "json": "wfo_results_2011_2013.json",
        "csv": "wfo_oos_trades_2011_2013.csv",
    },
    "dragon_2014_2016": {
        "json": "wfo_results_2014_2016.json",
        "csv": "wfo_oos_trades_2014_2016.csv",
    },
    "gamma_2017_2019": {
        "json": "wfo_results_2017_2019.json",
        "csv": "wfo_oos_trades_2017_2019.csv",
    },
}

# Also check for remote results that were scp'd back
REMOTE_FILES = {
    "dragon": {
        "host": "harveybc@192.168.1.235",
        "port": 62024,
        "json": "wfo_results_2014_2016.json",
        "csv": "wfo_oos_trades_2014_2016.csv",
    },
    "gamma": {
        "host": "harveybc@192.168.0.106",
        "port": 62024,
        "json": "wfo_results_2017_2019.json",
        "csv": "wfo_oos_trades_2017_2019.csv",
    },
}


def fetch_remote_results():
    """SCP result files from remote machines."""
    base = "/home/harveybc/Documents/GitHub/heuristic-strategy"
    for name, info in REMOTE_FILES.items():
        for ftype in ["json", "csv"]:
            local = info[ftype]
            if not os.path.exists(local):
                remote = f"{info['host']}:{base}/{info[ftype]}"
                cmd = f"scp -P {info['port']} {remote} {local} 2>/dev/null"
                print(f"Fetching {name} {ftype}...")
                os.system(cmd)


def parse_fold_from_log(logfile):
    """Parse fold results from wfo_run.log for folds completed before kill."""
    folds = []
    if not os.path.exists(logfile):
        return folds

    with open(logfile) as f:
        lines = f.readlines()

    current_fold = None
    for line in lines:
        line = line.strip()
        if line.startswith("[FOLD ") and "Train:" in line:
            year = int(line.split("]")[0].replace("[FOLD ", ""))
            current_fold = {"test_year": year}
        elif line.startswith("Best params:") and current_fold:
            params_str = line.replace("Best params: ", "")
            try:
                current_fold["best_params"] = eval(params_str)
            except:
                current_fold["best_params"] = {}
        elif line.startswith("Train fitness") and current_fold:
            try:
                current_fold["train_fitness"] = float(line.split(": ")[1])
            except:
                pass
        elif line.startswith("OOS Result:") and current_fold:
            # Parse: OOS Result: Profit=$2203.35, Trades=9, Win%=66.7%, Sharpe=0.38 (1685s)
            parts = line.replace("OOS Result: ", "")
            try:
                profit = float(parts.split("Profit=$")[1].split(",")[0])
                trades = int(parts.split("Trades=")[1].split(",")[0])
                win_pct = float(parts.split("Win%=")[1].split("%")[0])
                sharpe = float(parts.split("Sharpe=")[1].split(" ")[0])
                current_fold["oos_profit"] = profit
                current_fold["oos_trades"] = trades
                current_fold["oos_win_pct"] = win_pct
                current_fold["oos_sharpe"] = sharpe
                folds.append(current_fold)
            except:
                pass
            current_fold = None

    return folds


def main():
    os.chdir("/home/harveybc/Documents/GitHub/heuristic-strategy")

    # Try to fetch remote results
    fetch_remote_results()

    all_folds = []
    all_trades = []

    # 1. Parse omega folds 2009-2010 from log
    log_folds = parse_fold_from_log("wfo_run_folds_2009_2010.log")
    if log_folds:
        print(f"Loaded {len(log_folds)} folds from omega log (2009-2010)")
        all_folds.extend(log_folds)

    # 2. Load JSON results from each machine
    for name, info in RESULT_FILES.items():
        if "json" not in info:
            continue
        jfile = info["json"]
        if os.path.exists(jfile):
            with open(jfile) as f:
                data = json.load(f)
            print(f"Loaded {name}: {len(data.get('fold_results',[]))} folds from {jfile}")
            all_folds.extend(data.get("fold_results", []))

        cfile = info.get("csv")
        if cfile and os.path.exists(cfile):
            df = pd.read_csv(cfile)
            all_trades.append(df)
            print(f"  + {len(df)} OOS trades from {cfile}")

    if not all_folds:
        print("ERROR: No results found. Check that machines have completed.")
        sys.exit(1)

    # Sort by year
    all_folds.sort(key=lambda f: f["test_year"])

    # Aggregate
    total_profit = sum(f["oos_profit"] for f in all_folds)
    total_trades = sum(f["oos_trades"] for f in all_folds)
    total_wins = sum(
        int(f["oos_trades"] * f["oos_win_pct"] / 100 + 0.5) for f in all_folds
    )
    total_win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # Compute aggregate Sharpe from trades if available
    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        if "pnl" in trades_df.columns:
            pnls = trades_df["pnl"].values
            agg_sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) if len(pnls) > 1 else 0
        else:
            agg_sharpe = np.mean([f["oos_sharpe"] for f in all_folds])
    else:
        agg_sharpe = np.mean([f["oos_sharpe"] for f in all_folds])

    # Equity curve
    equity = 10000.0
    peak = equity
    max_dd = 0
    for f in all_folds:
        equity += f["oos_profit"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Print report
    print(f"\n{'='*70}")
    print(f"MERGED WALK-FORWARD RESULTS (OUT-OF-SAMPLE)")
    print(f"{'='*70}")
    print(f"Folds:            {len(all_folds)}")
    print(f"Total OOS Profit: ${total_profit:,.2f}")
    print(f"Total OOS Trades: {total_trades}")
    print(f"Win Rate:         {total_win_pct:.1f}%")
    print(f"Aggregate Sharpe: {agg_sharpe:.3f}")
    print(f"Max DD (equity):  ${max_dd:,.2f}")
    print(f"Final Equity:     ${equity:,.2f}")
    print()

    print(f"{'Year':<6} {'Train':<12} {'OOS Profit':>12} {'Trades':>8} "
          f"{'Win%':>7} {'Sharpe':>8}")
    print("-" * 60)
    for f in all_folds:
        train_range = f.get("train_range", f"?-{f['test_year']-1}")
        print(f"{f['test_year']:<6} {train_range:<12} "
              f"${f['oos_profit']:>10,.2f} {f['oos_trades']:>8} "
              f"{f['oos_win_pct']:>6.1f}% {f['oos_sharpe']:>7.2f}")
    print(f"{'='*60}")

    profitable = sum(1 for f in all_folds if f["oos_profit"] > 0)
    print(f"Profitable folds: {profitable}/{len(all_folds)} "
          f"({profitable/len(all_folds)*100:.0f}%)")

    # Save merged results
    merged = {
        "total_oos_profit": total_profit,
        "total_oos_trades": total_trades,
        "total_win_pct": total_win_pct,
        "aggregate_sharpe": agg_sharpe,
        "max_drawdown_usd": max_dd,
        "final_equity": equity,
        "fold_results": all_folds,
    }
    with open("wfo_results_merged.json", "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\nMerged results saved to wfo_results_merged.json")

    if all_trades:
        trades_df.to_csv("wfo_oos_trades_merged.csv", index=False)
        print(f"Merged trades saved to wfo_oos_trades_merged.csv")


if __name__ == "__main__":
    main()

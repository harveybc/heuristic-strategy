#!/usr/bin/env python3
"""Replicate all strategy experiments from results_strategy/ config_out.json files.

Reads each config_out.json, re-runs with same params, compares to original.

Usage:
    STRATEGY_QUIET=1 PYTHONPATH=./ python3 examples/scripts/replicate_strategy.py [--phase phase_1_daily]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS_STRATEGY = REPO / "examples" / "results_strategy"
RESULTS = REPO / "examples" / "results"


def find_experiments(phase_filter=None):
    experiments = []
    for phase_dir in sorted(RESULTS_STRATEGY.iterdir()):
        if not phase_dir.is_dir():
            continue
        if phase_filter and phase_dir.name != phase_filter:
            continue
        for cfg_file in sorted(phase_dir.glob("*config_out*.json")):
            with open(cfg_file) as f:
                cfg = json.load(f)
            experiments.append({"phase": phase_dir.name, "config_file": cfg_file, "config": cfg})
    return experiments


def run_experiment(cfg):
    """Run strategy using config_out params. Returns (summary_dict, error_msg)."""
    args = ["python3", "app/main.py"]
    
    # Infer base_dataset_file if missing (phase_1 configs don't save it)
    base_ds = cfg.get("base_dataset_file")
    if not base_ds:
        daily_cfg = cfg.get("predictor_daily_config_file", "")
        if "phase_1" in daily_cfg:
            base_ds = "examples/data/phase_1/phase_1_base_d3.csv"
        elif "phase_2" in daily_cfg:
            phase = daily_cfg.split("/")[-1].split("_")[0] + "_" + daily_cfg.split("/")[-1].split("_")[1]
            base_ds = f"examples/data/{phase}/base_d3.csv"

    for key in ["predictor_daily_config_file", "predictor_hourly_config_file",
                "load_parameters", "prefix", "max_trades_per_5days"]:
        val = cfg.get(key)
        if val is not None:
            args.extend([f"--{key}", str(val)])
    if base_ds:
        args.extend(["--base_dataset_file", base_ds])

    args.extend(["--gaussian_noise_mean", "0", "--gaussian_noise_stddev", "0"])

    env = os.environ.copy()
    env["STRATEGY_QUIET"] = "1"
    env["PYTHONPATH"] = str(REPO)

    result = subprocess.run(args, capture_output=True, text=True, cwd=str(REPO), env=env, timeout=300)
    
    if result.returncode != 0:
        return None, result.stderr[-300:] if result.stderr else result.stdout[-300:]

    # The strategy auto-derives output path from predictor config + prefix
    # Read from the summary_csv_file path in config_out (which is where results/ puts it)
    summary_path = cfg.get("summary_csv_file")
    if summary_path:
        full_path = REPO / summary_path
        if full_path.exists():
            with open(full_path) as f:
                rows = list(csv.DictReader(f))
            return rows[0] if rows else None, None
    
    return None, "summary not found"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", help="Filter by phase")
    args = parser.parse_args()

    experiments = find_experiments(args.phase)
    print(f"Found {len(experiments)} experiments")

    results = []
    for i, exp in enumerate(experiments):
        phase = exp["phase"]
        cfg = exp["config"]
        name = exp["config_file"].stem.replace("_config_out", "")

        # Load original from results_strategy
        orig_summary_name = Path(cfg.get("summary_csv_file", "")).name
        orig_path = RESULTS_STRATEGY / phase / orig_summary_name
        orig_profit = None
        if orig_path.exists() and orig_path.is_file():
            with open(orig_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                orig_profit = float(rows[0]["profit"])

        rep, err = run_experiment(cfg)

        if rep is None:
            print(f"[{i+1}/{len(experiments)}] {name}: FAILED — {(err or 'unknown')[:80]}")
            results.append({"name": name, "status": "failed"})
            continue

        rep_profit = float(rep["profit"])
        rep_trades = int(rep["num_trades"])
        diff = rep_profit - orig_profit if orig_profit is not None else 0
        match = "✅" if abs(diff) < 1.0 else ("⚠️" if abs(diff) < 50 else "❌")
        
        print(f"[{i+1}/{len(experiments)}] {name}: orig=${orig_profit:>8,.1f} rep=${rep_profit:>8,.1f} Δ={diff:+.1f} t={rep_trades} {match}")
        results.append({"name": name, "status": "ok", "orig": orig_profit, "rep": rep_profit, "diff": diff})

    ok = [r for r in results if r["status"] == "ok"]
    matched = [r for r in ok if abs(r.get("diff", 999)) < 1.0]
    print(f"\nTotal={len(results)} OK={len(ok)} Matched={len(matched)}")


if __name__ == "__main__":
    main()

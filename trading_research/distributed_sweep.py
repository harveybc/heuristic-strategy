#!/usr/bin/env python3
"""
Phase 1.2 Distributed Oracle Sweep Launcher

Distributes the 360-cell grid (12 assets × 5 timeframes × 6 strategies) across
3 machines: omega (local), dragon (192.168.1.235), gamma (192.168.0.106).

Machines are weighted by GPU/CPU power:
  - dragon (RTX 4090): 40% of cells
  - gamma  (RTX 5070 Ti): 35% of cells
  - omega  (RTX 4070, local): 25% of cells

Strategy: split by asset groups to maximize data cache reuse.
"""
import subprocess
import sys
import os
import time
import json
import signal

# ─── Configuration ──────────────────────────────────────────────────────────
MACHINES = {
    "dragon": {
        "host": "192.168.1.235",
        "port": 62024,
        "user": "harveybc",
        "assets": ["BTC/USD", "ETH/USD", "CL", "XAU/USD", "XAG/USD"],  # 5 assets
    },
    "gamma": {
        "host": "192.168.0.106",
        "port": 62024,
        "user": "harveybc",
        "assets": ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"],  # 4 assets
    },
    "omega": {
        "host": "localhost",
        "port": None,
        "user": None,
        "assets": ["AUD/JPY", "EUR/JPY", "GBP/JPY"],  # 3 assets
    },
}

TIMEFRAMES = ["daily", "weekly", "4h", "1h"]
# Skip 15min for now — yfinance only gives 60 days of 15min data
# which is too little for meaningful evaluation. Add later with better data source.

STRATEGIES = ["momentum", "mean_reversion", "breakout", "carry_momentum", "vol_regime_switch"]

REMOTE_DIR = "/tmp/trading_research"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(LOCAL_DIR, "results")


def copy_code_to_remote(machine_name: str):
    """SCP the trading_research package to remote machine."""
    m = MACHINES[machine_name]
    if m["host"] == "localhost":
        return

    host = m["host"]
    port = m["port"]
    user = m["user"]

    print(f"  Copying code to {machine_name}...")

    # Create remote dir
    subprocess.run(
        ["ssh", "-p", str(port), f"{user}@{host}",
         f"mkdir -p {REMOTE_DIR}"],
        check=True, capture_output=True
    )

    # SCP the package files
    files_to_copy = [
        "__init__.py",
        "transaction_cost_model.py",
        "evaluation_harness.py",
        "oracle_sensitivity.py",
    ]
    for f in files_to_copy:
        local_path = os.path.join(LOCAL_DIR, f)
        subprocess.run(
            ["scp", "-P", str(port), local_path,
             f"{user}@{host}:{REMOTE_DIR}/{f}"],
            check=True, capture_output=True
        )
    print(f"  ✓ Code deployed to {machine_name}:{REMOTE_DIR}")


def launch_remote(machine_name: str) -> subprocess.Popen:
    """Launch oracle sweep on a remote machine via SSH."""
    m = MACHINES[machine_name]
    assets = m["assets"]
    assets_str = " ".join(f'"{a}"' for a in assets)
    tf_str = " ".join(f'"{t}"' for t in TIMEFRAMES)
    strat_str = " ".join(f'"{s}"' for s in STRATEGIES)
    output = f"{REMOTE_DIR}/results/oracle_sweep_{machine_name}.json"

    cmd = (
        f"cd {REMOTE_DIR} && "
        f"python3 -u oracle_sensitivity.py "
        f"--assets {assets_str} "
        f"--timeframes {tf_str} "
        f"--strategies {strat_str} "
        f"--output {output}"
    )

    if m["host"] == "localhost":
        # Local execution
        local_output = os.path.join(RESULTS_DIR, f"oracle_sweep_{machine_name}.json")
        local_cmd = (
            f"cd {LOCAL_DIR} && "
            f"python3 -u oracle_sensitivity.py "
            f"--assets {assets_str} "
            f"--timeframes {tf_str} "
            f"--strategies {strat_str} "
            f"--output {local_output}"
        )
        print(f"  Launching on {machine_name} (local)...")
        proc = subprocess.Popen(
            local_cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
    else:
        host = m["host"]
        port = m["port"]
        user = m["user"]
        ssh_cmd = (
            f"ssh -p {port} {user}@{host} "
            f"'PYTHONPATH={REMOTE_DIR}:$PYTHONPATH {cmd}'"
        )
        print(f"  Launching on {machine_name} ({host})...")
        proc = subprocess.Popen(
            ssh_cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )

    return proc


def collect_remote_results(machine_name: str):
    """SCP results from remote machine back to local."""
    m = MACHINES[machine_name]
    if m["host"] == "localhost":
        return  # Already local

    host = m["host"]
    port = m["port"]
    user = m["user"]
    remote_file = f"{REMOTE_DIR}/results/oracle_sweep_{machine_name}.json"
    local_file = os.path.join(RESULTS_DIR, f"oracle_sweep_{machine_name}.json")

    subprocess.run(
        ["scp", "-P", str(port),
         f"{user}@{host}:{remote_file}", local_file],
        check=True, capture_output=True
    )
    print(f"  ✓ Results collected from {machine_name}")


def merge_results():
    """Merge results from all workers into a single ranked table."""
    all_results = []
    all_failed = []

    for machine in MACHINES:
        fpath = os.path.join(RESULTS_DIR, f"oracle_sweep_{machine}.json")
        if not os.path.exists(fpath):
            print(f"  ⚠ Missing results from {machine}")
            continue
        with open(fpath) as f:
            data = json.load(f)
        all_results.extend(data.get("results", []))
        all_failed.extend(data.get("failed", []))

    # Sort by noise budget (descending) then oracle Sharpe (descending)
    all_results.sort(key=lambda x: (-x["noise_budget"], -x["oracle_sharpe"]))

    # Summary
    print(f"\n{'='*100}")
    print(f"ORACLE SWEEP RESULTS — MERGED")
    print(f"{'='*100}")
    print(f"  Completed: {len(all_results)}, Failed: {len(all_failed)}")

    # Table header
    print(f"\n  {'Rank':>4} {'Asset':<10} {'TF':<8} {'Strategy':<18} │ "
          f"{'Budget':>7} {'Oracle SR':>10} {'B&H SR':>8} │ "
          f"{'Robustness':>11} {'Interest':>9}")
    print(f"  {'─'*4} {'─'*10} {'─'*8} {'─'*18} │ "
          f"{'─'*7} {'─'*10} {'─'*8} │ "
          f"{'─'*11} {'─'*9}")

    for i, r in enumerate(all_results[:50]):
        rob = r.get("noise_results", [{}])[0].get("regime_robustness", "N/A")
        rob_str = f"{rob:.2f}" if isinstance(rob, float) else rob
        interesting = r.get("noise_results", [{}])[0].get("is_interesting", False)
        int_str = "YES ★" if interesting else "no"

        print(f"  {i+1:>4} {r['asset']:<10} {r['timeframe']:<8} {r['strategy']:<18} │ "
              f"{r['noise_budget']:>6.2f}σ {r['oracle_sharpe']:>+9.3f} {r['bh_sharpe']:>+7.3f} │ "
              f"{rob_str:>11} {int_str:>9}")

    # Top 30 for Phase 1.3
    top30 = all_results[:30]
    surviving = [r for r in all_results
                 if r.get("noise_results", [{}])[0].get("is_interesting", False)]

    print(f"\n  Cells with regime_robustness ≥ 0.5 AND oracle Sharpe ≥ 0.4: {len(surviving)}")

    if len(surviving) < 5:
        print(f"\n  ⚠⚠⚠ KILL CRITERION: fewer than 5 surviving cells!")
        print(f"  The search space may be empty. See Phase 5 decision tree.")

    # Save merged
    merged_path = os.path.join(RESULTS_DIR, "oracle_sweep_merged.json")
    with open(merged_path, "w") as f:
        json.dump({
            "total_completed": len(all_results),
            "total_failed": len(all_failed),
            "surviving_cells": len(surviving),
            "top30": top30,
            "all_results": all_results,
            "failed": all_failed,
        }, f, indent=2, default=str)
    print(f"\n  Merged results saved to {merged_path}")

    return all_results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_cells = sum(
        len(m["assets"]) * len(TIMEFRAMES) * len(STRATEGIES)
        for m in MACHINES.values()
    )
    print("=" * 80)
    print("PHASE 1.2: DISTRIBUTED ORACLE SWEEP")
    print("=" * 80)
    print(f"Total cells: {total_cells}")
    print(f"Timeframes: {TIMEFRAMES} (15min excluded — insufficient yfinance history)")
    print(f"Strategies: {STRATEGIES}")
    print()
    for name, m in MACHINES.items():
        n = len(m["assets"]) * len(TIMEFRAMES) * len(STRATEGIES)
        print(f"  {name:>8}: {m['assets']} → {n} cells")
    print()

    # Step 1: Deploy code to remote machines
    print("Step 1: Deploying code to remote machines...")
    for name in MACHINES:
        if MACHINES[name]["host"] != "localhost":
            copy_code_to_remote(name)

    # Step 2: Launch all workers
    print("\nStep 2: Launching workers...")
    processes = {}
    for name in MACHINES:
        processes[name] = launch_remote(name)
        time.sleep(1)

    # Step 3: Monitor progress
    print("\nStep 3: Waiting for completion (streaming output)...")
    print("-" * 80)

    finished = set()
    try:
        while len(finished) < len(processes):
            for name, proc in processes.items():
                if name in finished:
                    continue
                line = proc.stdout.readline()
                if line:
                    print(f"  [{name}] {line.rstrip()}")
                elif proc.poll() is not None:
                    finished.add(name)
                    rc = proc.returncode
                    status = "✓ DONE" if rc == 0 else f"✗ FAILED (rc={rc})"
                    print(f"  [{name}] {status}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Killing workers...")
        for proc in processes.values():
            proc.terminate()
        sys.exit(1)

    print("-" * 80)

    # Step 4: Collect remote results
    print("\nStep 4: Collecting results...")
    for name in MACHINES:
        if MACHINES[name]["host"] != "localhost":
            try:
                collect_remote_results(name)
            except Exception as e:
                print(f"  ⚠ Failed to collect from {name}: {e}")

    # Step 5: Merge and rank
    print("\nStep 5: Merging results...")
    merge_results()


if __name__ == "__main__":
    main()

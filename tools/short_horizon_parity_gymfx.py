#!/usr/bin/env python3
"""PARITY_CHECK: replay the sealed ZigZag expert series and the smoke-probe series as candidate policies
in the gym-fx simulation (agent-multi's replay worker, commit e4d01fa0) and compare metric by metric
with this repository's hourly simulator under the closest contract both allow.

Policy replayed: at a pivot (expert) or a detector signal (probe) the decision is long / short on that
bar; the environment fills at the NEXT bar open and a later opposite decision flips the position
(gym-fx has no per-trade SL/TP; the same hold-to-reversal policy is run here with symbolic levels,
labelled PARITY_ONLY_NOT_A_STRATEGY). Fixed units, no compounding, matched per-fill costs, no swap.
Nothing here is a strategy result; it measures the two simulators against each other.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import calendar as CAL  # noqa: E402
from app.short_horizon import execution as X  # noqa: E402
from app.short_horizon import zigzag as Z  # noqa: E402
from app.short_horizon.blocks import OK  # noqa: E402
from tools.short_horizon_zigzag_study import Ctx, params_of  # noqa: E402

UNITS = 90_000.0
CASH = 10_000.0
PERIOD = ("2024-01-01", "2025-01-01")
# matched per-fill costs: gym-fx charges commission (fraction of notional) and slippage_perc per fill;
# ours charges half spread + slippage (price units) + commission per lot. At price ~1.1 they agree.
OURS = X.Costs(spread_price=0.6e-4, slippage_price=0.2e-4, commission_per_lot_round_trip=7.0,
               financing_per_lot_per_rollover=0.0)
GYM_COMMISSION = 3.5 / 100_000 / 1.1          # 3.5 USD per lot per side as a fraction of notional at 1.1
GYM_SLIPPAGE = (0.3e-4 + 0.2e-4) / 1.1          # half spread + slippage as a fraction of price per fill


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def env_config(bars_csv):
    return {"input_data_file": str(bars_csv), "date_column": "DATE_TIME", "price_column": "CLOSE", "headers": True,
            "window_size": 32, "feature_columns": [], "include_price_window": True, "include_agent_state": True,
            "env_mode": "training", "mode": "training", "action_space_mode": "continuous",
            "continuous_action_threshold": 0.1, "continuous_action_contract": "legacy_directional_v1",
            "initial_cash": CASH, "min_equity": 0.0, "position_size": UNITS, "commission": GYM_COMMISSION,
            "slippage": GYM_SLIPPAGE, "leverage": 100.0, "timeframe": "1h", "simulation_engine": "backtrader",
            "solvency_mode": "normal_realistic", "data_feed_plugin": "default_data_feed", "broker_plugin": "default_broker",
            "strategy_plugin": "default_strategy", "metrics_plugin": "default_metrics",
            "preprocessor_plugin": "default_preprocessor", "reward_plugin": "pnl_reward", "reward_scale": 1.0}


def probe_signals(run, ctx):
    seal = json.loads((run / "DETECTOR_SEAL.json").read_text())
    P = pd.read_csv(run / "detector_predictions.csv")
    k, tau = int(seal["k_bars"]), float(seal["threshold"])
    pv, pp = P.p_valley.to_numpy(float), P.p_peak.to_numpy(float)
    sig, last = [], -10 ** 9
    for t in range(len(P)):
        if not np.isfinite(pv[t]) or t - last < k:
            continue
        if pv[t] >= tau and pv[t] >= pp[t]:
            sig.append((t, 1)),
            last = t
        elif pp[t] >= tau and pp[t] > pv[t]:
            sig.append((t, -1))
            last = t
    return sig, seal


def decisions_for(signals, bars_index, lake_index):
    """Map lake bar positions to the CSV's bar keys; every CSV bar gets a decision (flat = HOLD)."""
    pos = {ts: i for i, ts in enumerate(lake_index)}
    by_bar = {}
    for b, d in signals:
        ts = lake_index[b]
        by_bar[ts] = "long" if d > 0 else "short"
    out = {}
    for ts in bars_index:
        key = pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).isoformat()   # the worker keys bars as naive ISO instants
        out[key] = {"long": 1.0, "short": -1.0, "flat": 0.0}[by_bar.get(ts, "flat")]
    return out, sum(1 for v in out.values() if v != 0.0)


def ours(ctx, signals, bars_csv_index):
    """Hold-to-reversal replay in this repository's simulator: fixed units, symbolic SL/TP (PARITY_ONLY)."""
    segs = CAL.derive_segments(ctx.bars.index, max_gap_hours=24)
    cal = CAL.load_calendar({"calendar_id": "PARITY_NO_FORCED_FLAT", "version": "parity", "segments": segs,
                             "entry_delay_after_open_hours": 0, "force_flat_before_close_hours": 0,
                             "entry_cutoff_before_flat_hours": 0})
    blocks = Z.signal_blocks(ctx.bars, cal, signals)
    orders = []
    for k, (b, d) in enumerate(signals):
        if blocks.status[k] != OK:
            orders.append(None)
            continue
        r = float(blocks.ref[k])
        orders.append(X.BlockOrder(d, "MARKET", None, r + d * 1.0, r - d * 1.0, UNITS, 1.0))
    a, b = (pd.Timestamp(x, tz="UTC") for x in PERIOD)
    res = X.run_backtest(ctx.bars, blocks, orders, OURS, cal, CASH, (a, b))
    m = res.metrics
    clipped = sum(1 for k in range(len(blocks)) if blocks.status[k] == OK and blocks.in_split((a, b))[k]
                  and blocks.end[k] >= cal.segments[blocks.segment[k]].force_flat and
                  (k + 1 >= len(signals) or ctx.bars.index[min(signals[k + 1][0] + 1, len(ctx.bars) - 1)] > blocks.end[k]))
    return {"net": m["net_profit_usd"], "sharpe": m["sharpe_daily_sqrt365"], "max_dd_usd": m["max_drawdown_usd"],
            "max_dd_pct": m["max_drawdown_pct"], "trades": m["trades"], "exit_mix": {"reversal_or_time": m["exit_time"], "tp": m["exit_tp"], "sl": m["exit_sl"]},
            "blocks_clipped_at_session_close": clipped, "costs_usd": m["cost_spread_usd"] + m["cost_slippage_usd"] + m["cost_commission_usd"],
            "gross_usd": m["gross_usd"]}


def gym_metrics(result):
    steps = pd.DataFrame(result["steps"])
    eq = CASH * (1.0 + steps.environment_return_training_reward.cumsum())
    peak = np.maximum.accumulate(np.r_[CASH, eq.to_numpy()])
    curve = np.r_[CASH, eq.to_numpy()]
    dd = peak - curve
    day = pd.Series(eq.to_numpy(), index=pd.to_datetime(steps.t)).groupby(lambda t: t.normalize()).last()
    eqd = np.r_[CASH, day.to_numpy()]
    ret = np.diff(eqd) / eqd[:-1]
    sd = ret.std(ddof=1) if len(ret) > 1 else 0.0
    flips = int((np.sign(steps.position).diff().fillna(0) != 0).sum())
    return {"net": float(eq.iloc[-1] - CASH), "sharpe": float(ret.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0,
            "max_dd_usd": float(dd.max()), "max_dd_pct": float((dd / peak).max() * 100), "trades": result.get("trades"),
            "position_changes": flips, "steps": int(result["steps_taken"]), "stopped": result["stopped_because"],
            "termination_cause": result.get("termination_cause")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--bars-csv", required=True)
    ap.add_argument("--agent-multi", required=True, help="read-only worktree at e4d01fa0")
    ap.add_argument("--gym-fx", required=True, help="read-only worktree")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(a.data)
    choice = json.loads((run / "CHOICE.json").read_text())
    p = params_of(choice["params"])
    piv = ctx.pivots(p)
    expert = Z.signals_from_pivots(piv)
    probe, seal = probe_signals(run, ctx)
    bars = pd.read_csv(a.bars_csv, parse_dates=["DATE_TIME"])
    bars_index = pd.DatetimeIndex(bars.DATE_TIME).tz_localize("UTC")
    lake_index = ctx.bars.index
    a24, b24 = (pd.Timestamp(x, tz="UTC") for x in PERIOD)
    runs, counts = [], {}
    for name, sig in (("expert", expert), ("probe", probe)):
        sig24 = [(b, d) for b, d in sig if a24 <= lake_index[b] < b24]
        actions, n = decisions_for(sig24, bars_index, lake_index)
        counts[name] = n
        (out / f"decisions_{name}.jsonl").write_text("\n".join(json.dumps({"t": k, "action": {1.0: "long", -1.0: "short", 0.0: "flat"}[v]})
                                                               for k, v in actions.items()) + "\n")
        runs.append({"name": name, "kind": "series", "actions": actions})
    runs.append({"name": "flat", "kind": "constant", "value": 0.0})
    cfg = env_config(Path(a.bars_csv).resolve())
    worker = Path(a.agent_multi) / "m5phet_policy/src/agent_multi_m5phet/sweep_worker.py"
    t0 = time.time()
    done = subprocess.run([a.python, str(worker)], input=json.dumps({"gym_fx": str(Path(a.gym_fx).resolve()), "env_config": cfg,
                                                                      "max_steps": len(bars), "runs": runs}),
                          text=True, capture_output=True, cwd=str(Path(a.gym_fx).resolve()),
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=3000)
    lines = [l for l in done.stdout.splitlines() if l.startswith("{")]
    if not lines:
        raise SystemExit(f"worker failed ({done.returncode}): {done.stderr[-2000:]}")
    answer = json.loads(lines[-1])
    if "error" in answer:
        raise SystemExit(answer["error"])
    (out / "gymfx_raw.json").write_text(json.dumps(answer)[:2_000_000])
    table = {}
    for name, sig in (("expert", expert), ("probe", probe)):
        sig24 = [(b, d) for b, d in sig if a24 - pd.Timedelta(days=7) <= lake_index[b] < b24]
        table[name] = {"ours_hourly_simulator_PARITY_ONLY": ours(ctx, sig24, bars_index),
                       "gym_fx_replay": gym_metrics(answer["runs"][name]), "decisions_in_2024": counts[name]}
    table["flat"] = {"gym_fx_replay": gym_metrics(answer["runs"]["flat"])}
    doc = {"label": "PARITY_CHECK", "period": PERIOD, "units_fixed": UNITS, "initial_cash": CASH,
           "expert_series_sha256": json.loads((run / "demonstrations" / "MANIFEST.json").read_text())["sha256"],
           "probe_predictions_sha256": seal["sha256"], "bars_csv_sha256": sha(a.bars_csv), "bars_rows": int(len(bars)),
           "agent_multi_worker": str(worker.relative_to(Path(a.agent_multi))) + " @ e4d01fa0", "gym_fx_head": "e287ea2",
           "env_config": {k: v for k, v in cfg.items() if k != "input_data_file"}, "ours_costs": OURS.__dict__,
           "worker_seconds": answer.get("seconds"), "wall_seconds": time.time() - t0, "table": table,
           "contract_differences": [
               "latency: both fill a decision at the NEXT bar open (gym-fx: backtrader market order; ours: MARKET at bar lo open)",
               "exits: gym-fx has no per-trade SL/TP (only fixed-pip brackets via direct_fixed_sltp), so the parity policy is hold-to-reversal; "
               "our strategy result uses per-trade TP = next pivot level and SL = 2 sigma and is NOT what is compared here",
               "session calendar: ours force-flats at the session close (weekly gap) and refuses entries outside sessions; gym-fx holds through the gap",
               "intrabar rule: ours = open first, stop before TP, limit trade-through; gym-fx (backtrader) evaluates bracket legs by its own order, unused here",
               "spread: ours charges half the hour-dependent spread per fill (0,3 pip mean used here as a flat half spread for parity); gym-fx has no spread, emulated as slippage_perc",
               "commission: ours 3,5 USD per lot per side in cash; gym-fx a fraction of notional (equal at price 1,1, drifts with price)",
               "swap: ours debits financing per rollover with a triple day (disabled here, financing 0); gym-fx has no financing in the default broker",
               "sizing: ours compounds rel_volume x balance x leverage (disabled here: fixed units); gym-fx position_size is fixed units per order",
               "ruin / caps: ours stops at ruin, caps 4 orders/day and enforces the frequency rule (disabled here); gym-fx terminates at min_equity (0 here)"]}
    (out / "PARITY.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else {m: vv[m] for m in ("net", "sharpe", "max_dd_usd", "trades")})
                          for kk, vv in v.items() if isinstance(vv, dict)} for k, v in table.items()}, default=str))


if __name__ == "__main__":
    main()

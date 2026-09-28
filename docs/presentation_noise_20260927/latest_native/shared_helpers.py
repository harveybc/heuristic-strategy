"""Reproducible price-noise experiment around the paired persistence MAE."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("STRATEGY_QUIET", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtrader as bt
import numpy as np
import pandas as pd
from app.plugins import plugin_long_short_predictions as legacy

OFFSETS = np.array([1, 2, 3, 4, 5, 6, 24, 48, 72, 96, 120, 144])
COLUMNS = [f"Prediction_h_{i}" for i in range(1, 7)] + [f"Prediction_d_{i}" for i in range(1, 7)]
RATIOS = [0, .05, .1, .15, .2, .3, .4, .5, .6, .75, .9, .95, 1, 1.05, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3]
SEEDS = [42, 43, 44]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def build_targets(frame):
    if len(frame) <= 146:
        raise ValueError("Insufficient rows for predictions and liquidation")
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Unique sorted datetime index required")
    if ((frame.index.minute != 0) | (frame.index.second != 0)).any():
        raise ValueError("Only exact hourly bar labels are supported")
    values = frame[["OPEN", "HIGH", "LOW", "CLOSE"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite OHLC")
    closes = frame.CLOSE.to_numpy(dtype=float)
    n = len(frame) - 144
    target = closes[np.arange(n)[:, None] + OFFSETS]
    return target, OFFSETS.copy()


def measured_mae(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2 or not prediction.size:
        raise ValueError("Prediction/target populations differ")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite prediction/target")
    return np.mean(np.abs(prediction-target), axis=0, dtype=np.float64)


def naive_errors(frame, target):
    current = frame.CLOSE.to_numpy(dtype=float)[:len(target)]
    return measured_mae(np.repeat(current[:, None], target.shape[1], axis=1), target)


def make_noise(seed, shape):
    return np.random.default_rng(seed).standard_normal(shape)


def equivalent_sigma(naive, draws):
    scale = np.mean([np.mean(np.abs(z[:, -1])) for z in draws])
    if not np.isfinite(naive) or naive <= 0 or scale <= 0:
        raise ValueError("Positive naive and noise scale required")
    return float(naive/scale)


class ForexCosts(bt.CommInfoBase):
    params = (("commission", 7.0/100000), ("commtype", bt.CommInfoBase.COMM_FIXED),
              ("stocklike", True), ("leverage", 100.0), ("interest_long", True))

    def __init__(self):
        super().__init__()
        self.swap_debits = 0.0

    def _get_credit_interest(self, data, size, price, days, dt0, dt1):
        cost = abs(size)/100000 * 10.0 * days
        self.swap_debits += cost
        return cost


class RecordedLegacy(legacy.Plugin.HeuristicStrategy):
    def __init__(self, origin_count, *args, **kwargs):
        self.origin_count = origin_count
        self.fills = []
        self.closed_trades = []
        self.equity = []
        self.pending = None
        self.rejections = 0
        self.order_entry_price = None
        super().__init__(*args, **kwargs)

    def buy(self, *args, **kwargs):
        order = super().buy(*args, **kwargs)
        self.pending = order
        return order

    def sell(self, *args, **kwargs):
        order = super().sell(*args, **kwargs)
        self.pending = order
        return order

    def next(self):
        self.equity.append((self.data.datetime.datetime(0), float(self.broker.getvalue())))
        if self.pending is not None and self.pending.alive():
            return
        # All declared origins have forecasts; only the two extra bars liquidate.
        if len(self) > self.origin_count:
            if self.position and len(self) == self.origin_count + 1:
                self.close()
            return
        super().next()

    def notify_order(self, order):
        if order.status == order.Completed:
            self.fills.append({"time": bt.num2date(order.executed.dt).isoformat(),
                               "size": float(order.executed.size), "price": float(order.executed.price),
                               "commission_usd": float(self.broker.getcommissioninfo(self.data).getcommission(
                                   order.executed.size, order.executed.price)),
                               "commission_and_swap_usd": float(order.executed.comm)})
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.pending = None
        if order.status in [order.Margin, order.Rejected, order.Expired]:
            self.rejections += 1
            if not self.position:
                self.current_direction = self.current_tp = self.current_sl = None
        super().notify_order(order)

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trades.append({"open": bt.num2date(trade.dtopen).isoformat(),
                                       "close": bt.num2date(trade.dtclose).isoformat(),
                                       "gross_usd": float(trade.pnl),
                                       "net_usd": float(trade.pnlcomm),
                                       "commission_and_swap_usd": float(trade.commission)})
        # Preserve legacy state resets, but do not use its double-debited trade metrics.
        super().notify_trade(trade)

    def stop(self):
        if self.position or (self.pending is not None and self.pending.alive()):
            raise RuntimeError("Unsettled final position/order")


def equity_statistics(equity, initial=10000.0):
    daily = equity.resample("D").last().ffill()
    previous = np.r_[initial, daily.to_numpy()[:-1]]
    returns = daily.to_numpy()/previous - 1
    sd = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns)/sd*np.sqrt(365)) if sd > 0 else None
    values = np.r_[initial, equity.to_numpy()]
    dd = 1-values/np.maximum.accumulate(values)
    return {"profit_usd": float(values[-1]-initial), "final_equity_usd": float(values[-1]),
            "sharpe_daily_365": sharpe, "max_drawdown_fraction": float(max(dd))}, daily


def run_cell(frame, target, predictions, out, cell_id):
    out.mkdir(parents=True, exist_ok=False)
    n = len(target)
    pred_frame = pd.DataFrame(predictions, index=frame.index[:n], columns=COLUMNS)
    pred_path = out / "predictions.csv"
    pred_frame.to_csv(pred_path, index_label="DATE_TIME", float_format="%.17g")
    reread = pd.read_csv(pred_path, index_col="DATE_TIME", parse_dates=True, float_precision="round_trip")
    if not reread.index.equals(pred_frame.index):
        raise ValueError("Prediction origin identity changed")
    np.testing.assert_array_equal(reread.to_numpy(), predictions)
    errors = measured_mae(reread.to_numpy(), target)
    independent = [math.fsum(abs(float(p)-float(y)) for p, y in zip(reread.iloc[:, j], target[:, j]))/n
                   for j in range(len(OFFSETS))]
    np.testing.assert_allclose(errors, independent, atol=1e-14, rtol=0)
    prediction_digest = digest(pred_path)
    bars = frame.iloc[:n+2].rename(columns=str.lower).copy()
    bars["volume"] = 0.0
    bars["openinterest"] = 0.0
    params = {k: v for k, v in legacy.Plugin.plugin_params.items()
              if k not in {"spread_pips", "commission_per_lot", "slippage_pips"}}
    engine = bt.Cerebro(stdstats=False)
    engine.addstrategy(RecordedLegacy, origin_count=n, pred_file=str(pred_path), **params)
    engine.adddata(bt.feeds.PandasData(dataname=bars))
    costs = ForexCosts()
    engine.broker.addcommissioninfo(costs)
    engine.broker.setcash(10000.0)
    engine.broker.set_slippage_fixed((.00015+.00005)/2, slip_open=True, slip_limit=True)
    cpu, wall = time.process_time(), time.monotonic()
    result = engine.run()[0]
    consumed = result.pred_df.to_numpy(dtype=float)
    np.testing.assert_allclose(consumed, predictions, atol=5e-16, rtol=0)
    errors = measured_mae(consumed, target)
    cpu, wall = time.process_time()-cpu, time.monotonic()-wall
    equity = pd.Series([e[1] for e in result.equity], index=pd.DatetimeIndex([e[0] for e in result.equity]))
    stats, daily = equity_statistics(equity)
    net = math.fsum(t["net_usd"] for t in result.closed_trades)
    commissions = math.fsum(f["commission_usd"] for f in result.fills)
    if abs(stats["profit_usd"]-net) > 1e-5:
        raise RuntimeError(f"Broker/trade PnL mismatch: {stats['profit_usd']} vs {net}")
    gross = math.fsum(t["gross_usd"] for t in result.closed_trades)
    if abs(gross-commissions-costs.swap_debits-net) > 1e-5:
        raise RuntimeError(f"Cost ledger does not reconcile: gross={gross}, commission={commissions}, swap={costs.swap_debits}, net={net}")
    daily.to_csv(out/"daily_equity.csv", header=["equity_usd"], index_label="date")
    pd.DataFrame(result.closed_trades, columns=["open","close","gross_usd","net_usd","commission_and_swap_usd"]).to_csv(out/"trades.csv", index=False)
    pd.DataFrame(result.fills, columns=["time","size","price","commission_usd","commission_and_swap_usd"]).to_csv(out/"fills.csv", index=False)
    record = {"cell": cell_id, **stats, "mae_by_horizon": dict(zip(map(str, OFFSETS), map(float, errors))),
              "n_origins": n, "n_closed_trades": len(result.closed_trades),
              "n_rejected_orders": result.rejections, "commission_usd": commissions,
              "swap_usd": costs.swap_debits, "final_position": float(result.position.size),
              "cpu_seconds": cpu, "wall_seconds": wall, "prediction_csv_sha256": prediction_digest,
              "prediction_retention": "reconstruct from retained target, standard-normal draws and sigma",
              "artifacts": {f: digest(out/f) for f in ["daily_equity.csv", "trades.csv", "fills.csv"]}}
    write_json(out/"result.json", record)
    pred_path.unlink()
    return record


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.data, parse_dates=["DATE_TIME"], index_col="DATE_TIME")
    target, _ = build_targets(frame)
    naive = naive_errors(frame, target)
    draws = [make_noise(s, target.shape) for s in SEEDS]
    sigma = equivalent_sigma(float(naive[-1]), draws)
    shutil.copy2(args.data, out/"input.csv")
    shutil.copy2(__file__, out/"runner.py")
    shutil.copy2(ROOT/"app/plugins/plugin_long_short_predictions.py", out/"legacy_strategy.py")
    shutil.copy2(ROOT/"app/policies/prediction_entry_exit.py", out/"legacy_policy.py")
    shutil.copy2(ROOT/"docs/noise_price_sweep/PLAN.md", out/"PLAN.md")
    np.save(out/"targets.npy", target, allow_pickle=False)
    for seed, z in zip(SEEDS, draws):
        np.save(out/f"standard_normal_seed{seed}.npy", z, allow_pickle=False)
    manifest = {"kind": "NEW_PRICE_NOISE_SWEEP", "data": str(args.data.resolve()),
                "source_sha256": digest(args.data), "n_bars": len(frame), "n_origins": len(target),
                "origin_first": frame.index[0], "origin_last": frame.index[len(target)-1],
                "liquidation_last": frame.index[len(target)+1], "horizons_bars": OFFSETS.tolist(),
                "horizon_semantics": "observation offsets, not elapsed hours across gaps",
                "naive_mae": dict(zip(map(str, OFFSETS), map(float, naive))),
                "noise": "independent additive zero-mean Gaussian in raw price units before strategy",
                "sigma_ref_measured_mean_H144": sigma,
                "sigma_ref_expectation_H144": float(naive[-1]*np.sqrt(np.pi/2)),
                "seeds": SEEDS, "ratios": [0,1] if args.pilot else RATIOS,
                "pilot": args.pilot, "strategy_params": legacy.Plugin.plugin_params,
                "execution": {"initial_usd":10000, "leverage":100, "commission_per_lot_per_side_usd":7,
                              "spread_price":.00015, "slippage_price":.00005, "swap_per_lot_day_usd":10,
                              "fill":"next bar open with configured slippage capped to bar range",
                              "exit_variant":"E, fixed before scores", "optimization":False},
                "versions":{"numpy": np.__version__, "pandas":pd.__version__, "backtrader":bt.__version__},
                "files":{p.name:digest(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"manifest.json", manifest)
    print(json.dumps({"n_origins":len(target), "naive_H144":float(naive[-1]), "sigma_ref":sigma}), flush=True)
    records = []
    seeds = SEEDS[:1] if args.pilot else SEEDS
    for ratio in manifest["ratios"]:
        for seed, z in zip(seeds, draws):
            cell = f"noise_{ratio:g}_seed{seed}"
            predictions = target + sigma*ratio*z
            record = run_cell(frame, target, predictions, out/cell, cell)
            record.update({"kind":"noise", "noise_ratio":ratio, "sigma_price":sigma*ratio, "seed":seed})
            write_json(out/cell/"result.json", record)
            records.append(record)
            write_json(out/"results.json", records)
            print(json.dumps({k: record[k] for k in ["cell","profit_usd","sharpe_daily_365","n_closed_trades","n_rejected_orders","cpu_seconds"]}), flush=True)
    predictions = np.repeat(frame.CLOSE.to_numpy()[:len(target), None], len(OFFSETS), axis=1)
    record = run_cell(frame, target, predictions, out/"persistence", "persistence")
    record.update({"kind":"persistence", "noise_ratio":None, "sigma_price":None, "seed":None})
    records.append(record)
    write_json(out/"persistence/result.json", record)
    write_json(out/"results.json", records)
    rows = [{**{k:v for k,v in r.items() if not isinstance(v,dict)},
             **{f"mae_H{h}":v for h,v in r["mae_by_horizon"].items()}} for r in records]
    pd.DataFrame(rows).to_csv(out/"sweep.csv", index=False)
    if not args.pilot:
        crossing = [r["mae_by_horizon"]["144"] for r in records if r["noise_ratio"] == 1]
        assert len(crossing) == 3
        assert abs(np.mean(crossing)-naive[-1]) < 1e-12
    write_json(out/"completion.json", {"status":"COMPLETE", "cells":len(records),
               "cpu_seconds_backtests":sum(r["cpu_seconds"] for r in records),
               "manifest_sha256":digest(out/"manifest.json"), "results_sha256":digest(out/"results.json"),
               "sweep_sha256":digest(out/"sweep.csv")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT/"tests/data/phase_2_3_base_d3.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args())

"""V2: elapsed-time targets, chronological forecasts and native protective orders."""
from __future__ import annotations
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("STRATEGY_QUIET", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import backtrader as bt
import numpy as np
import pandas as pd
from tools.price_noise_sweep import (
    OFFSETS, SEEDS, RATIOS, COLUMNS, ForexCosts, digest, write_json,
    equivalent_sigma, equity_statistics,
)
from app.policies.prediction_entry_exit import (
    PredictionEntryExitParameters, calculate_entry_geometry, compute_legacy_order_size,
)


def elapsed_targets(frame):
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Unique ordered timestamps required")
    values = frame[["OPEN", "HIGH", "LOW", "CLOSE"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite OHLC")
    if ((frame.HIGH < frame[["OPEN", "CLOSE"]].max(axis=1)) |
        (frame.LOW > frame[["OPEN", "CLOSE"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC range")
    positions = np.column_stack([frame.index.get_indexer(frame.index+pd.Timedelta(hours=int(h)))
                                 for h in OFFSETS])
    valid = (positions[:, 5] >= 0) & (positions[:, -1] >= 0) & (np.arange(len(frame)) < len(frame)-2)
    origins = np.flatnonzero(valid)
    if not len(origins):
        raise ValueError("No complete elapsed-time forecast origins")
    target = frame.CLOSE.to_numpy()[np.maximum(positions[valid], 0)].astype(float)
    target[positions[valid] < 0] = np.nan
    return frame.index[valid], target, origins


def masked_mae(predictions, target):
    valid = np.isfinite(target)
    if predictions.shape != target.shape or not np.array_equal(np.isfinite(predictions), valid):
        raise ValueError("Forecast population differs from available targets")
    return np.array([np.mean(abs(predictions[valid[:,j],j]-target[valid[:,j],j])) for j in range(target.shape[1])])


def first_barrier(direction, predictions, tp, sl):
    values = np.asarray(predictions, dtype=float)
    if direction not in {"long", "short"} or not np.isfinite(values).all():
        raise ValueError("Finite ordered forecast and valid direction required")
    if not np.isfinite([tp, sl]).all() or (tp <= sl if direction == "long" else tp >= sl):
        raise ValueError("Invalid barrier geometry")
    for price in values:
        if (price <= sl if direction == "long" else price >= sl):
            return "SL"
        if (price >= tp if direction == "long" else price <= tp):
            return "TP"
    return None


def margin_order_size(balance, fraction, leverage, price, allocation_cap):
    if not np.isfinite([balance,fraction,leverage,price,allocation_cap]).all():
        raise ValueError("Nonfinite margin sizing")
    if balance < 0 or not 0 < fraction <= 1 or leverage <= 0 or price <= 0 or allocation_cap < 0:
        raise ValueError("Invalid margin sizing")
    return min(allocation_cap, balance*fraction*leverage/price)


class ExplicitCosts(ForexCosts):
    # Exact cash debit: commission plus fixed spread/slippage proxy, every side.
    params = (("commission", 7.0/100000 + .00010),)


class ProtectiveBroker(bt.brokers.BackBroker):
    def __init__(self):
        super().__init__()
        self.ambiguous_bars = []

    def _try_exec_market(self, order, popen, phigh, plow):
        fraction = order.info.get("margin_fraction")
        if order.info.get("reason") == "ENTRY" and fraction is not None:
            if order.data.datetime[0] <= order.created.dt:
                return
            if self.positions[order.data]:
                raise RuntimeError("Margin-capped entry requires the declared flat starting position")
            balance = self.cash
            leverage = self.getcommissioninfo(order.data).get_leverage()
            original = abs(order.executed.remsize)
            units = margin_order_size(min(balance,order.info.decision_balance),fraction,leverage,popen,original)
            order.addinfo(original_requested_units=original,fill_balance=balance)
            # Resize the pending parent and its unfilled protective children together.
            for member in self._pchildren[order.ref]:
                if member.executed.size:
                    raise RuntimeError("Cannot resize an executed bracket")
                signed = math.copysign(units,member.created.size)
                member.size = member.created.size = member.executed.remsize = signed
        return super()._try_exec_market(order,popen,phigh,plow)

    def _bracketize(self, order, cancel=False):
        super()._bracketize(order, cancel=cancel)
        # Backtrader otherwise delays child activation to the following bar.
        if order.parent is None and not cancel and order.status == order.Completed:
            while self._toactivate:
                self._toactivate.popleft().activate()

    def _try_exec_stop(self, order, popen, phigh, plow, pcreated, pclose):
        siblings = self._pchildren.get(getattr(order.parent, "ref", None), ())
        limit = next((o for o in siblings if o.exectype == bt.Order.Limit and o.alive()), None)
        if limit is not None:
            target = limit.created.price
            # The observed open precedes all unknown intrabar movements.
            if (popen <= target if order.isbuy() else popen >= target):
                return
            stop_hit = phigh >= pcreated if order.isbuy() else plow <= pcreated
            limit_hit = plow <= target if order.isbuy() else phigh >= target
            open_stop = popen >= pcreated if order.isbuy() else popen <= pcreated
            if stop_hit and limit_hit and not open_stop:
                self.ambiguous_bars.append({"time": str(order.data.datetime.datetime()),
                                            "rule": "SL_FIRST", "stop": pcreated, "target": target})
        return super()._try_exec_stop(order, popen, phigh, plow, pcreated, pclose)


class ProtectedForecastStrategy(bt.Strategy):
    params = (("predictions", None), ("last_origin_position", None), ("margin_fraction", None))

    def __init__(self):
        self.forecasts = self.p.predictions
        self.forecast_rows = {dt.to_pydatetime(): row for dt, row in
                              zip(self.forecasts.index, self.forecasts.to_numpy())}
        self.config = PredictionEntryExitParameters()
        self.market = None
        self.children = []
        self.geometry = None
        self.entries = deque()
        self.equity, self.fills, self.closed_trades, self.decisions = [], [], [], []
        self.rejections = 0

    def exit_market(self, reason):
        for order in self.children:
            if order.alive():
                self.cancel(order)
        self.children = []
        self.market = self.close()
        self.market.addinfo(reason=reason, tp=self.geometry.take_profit_price, sl=self.geometry.stop_loss_price)

    def next(self):
        dt = self.data.datetime.datetime()
        self.equity.append((dt, float(self.broker.getvalue())))
        if self.market is not None and self.market.alive():
            return
        if len(self)-1 > self.p.last_origin_position:
            if self.position:
                self.exit_market("FINAL_LIQUIDATION")
            return
        pred = self.forecast_rows.get(dt)
        if pred is None:
            return
        if self.position:
            g = self.geometry
            passage = first_barrier(g.direction, pred[np.isfinite(pred)], g.take_profit_price, g.stop_loss_price)
            self.decisions.append({"time": str(dt), "stage": "exit", "first": passage,
                                   "predictions": [float(x) if np.isfinite(x) else None for x in pred]})
            if passage == "SL":
                self.exit_market("FORECAST_SL_FIRST")
            return
        while self.entries and (dt-self.entries[0]).days >= 5:
            self.entries.popleft()
        if len(self.entries) >= self.config.max_trades_per_5days:
            return
        g = calculate_entry_geometry(current_price=self.data.close[0],
                                     long_horizon_predictions=pred[6:], params=self.config)
        if g is None:
            return
        long_path = pred[6:][np.isfinite(pred[6:])]
        passage = first_barrier(g.direction, long_path, g.take_profit_price, g.stop_loss_price)
        self.decisions.append({"time": str(dt), "stage": "entry", "first": passage,
                               "predictions": [float(x) if np.isfinite(x) else None for x in pred]})
        if passage != "TP":
            return
        size = compute_legacy_order_size(reward_risk_ratio=g.reward_risk_ratio,
                                        available_cash=self.broker.getcash(), params=self.config)
        balance = self.broker.getvalue()
        if self.p.margin_fraction is not None:
            rr_cap = compute_legacy_order_size(reward_risk_ratio=g.reward_risk_ratio,
                                              available_cash=math.inf,params=self.config)
            size = margin_order_size(balance,self.p.margin_fraction,self.config.leverage,self.data.close[0],rr_cap)
        if size <= 0:
            return
        method = self.buy_bracket if g.direction == "long" else self.sell_bracket
        orders = method(size=size, exectype=bt.Order.Market,
                        stopprice=g.stop_loss_price, limitprice=g.take_profit_price)
        for order, reason in zip(orders, ["ENTRY", "SL", "TP"]):
            order.addinfo(reason=reason, tp=g.take_profit_price, sl=g.stop_loss_price,
                          margin_fraction=self.p.margin_fraction,decision_balance=balance)
        self.market, self.children = orders[0], orders[1:]
        self.geometry = g
        self.entries.append(dt)

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.executed.dt <= order.created.dt:
                raise RuntimeError("Execution not after forecast decision")
            self.fills.append({"time": str(bt.num2date(order.executed.dt)),
                "created": str(bt.num2date(order.created.dt)), "reason": order.info.reason,
                "order_type": order.getordername(), "requested_price": float(order.created.price),
                "tp": float(order.info.tp), "sl": float(order.info.sl),
                "decision_balance_usd": order.info.get("decision_balance"),
                "fill_balance_usd": order.info.get("fill_balance"),
                "requested_units": order.info.get("original_requested_units"),
                "entry_margin_usd": abs(order.executed.size)*order.executed.price/self.config.leverage if order.info.reason == "ENTRY" else None,
                "size": float(order.executed.size), "price": float(order.executed.price),
                "commission_usd": abs(order.executed.size)*7/100000,
                "friction_usd": abs(order.executed.size)*.00010,
                "all_costs_with_swap_usd": float(order.executed.comm)})
        if order.status in [order.Margin, order.Rejected, order.Expired]:
            self.rejections += 1

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trades.append({"open": str(bt.num2date(trade.dtopen)),
                "close": str(bt.num2date(trade.dtclose)), "gross_usd": float(trade.pnl),
                "net_usd": float(trade.pnlcomm), "costs_usd": float(trade.commission)})

    def stop(self):
        if self.position or self.broker.get_orders_open():
            raise RuntimeError("Final position or orders are not flat")


def run_cell(frame, origins, target, positions, predictions, out, cell, margin_fraction=None):
    out.mkdir(parents=True, exist_ok=False)
    predicted = pd.DataFrame(predictions, index=origins, columns=COLUMNS)
    bars = frame.iloc[:positions[-1]+3].rename(columns=str.lower)
    engine = bt.Cerebro(stdstats=False)
    engine.setbroker(ProtectiveBroker(shortcash=margin_fraction is None))
    costs = ExplicitCosts()
    engine.broker.addcommissioninfo(costs)
    engine.broker.setcash(10000)
    engine.adddata(bt.feeds.PandasData(dataname=bars))
    engine.addstrategy(ProtectedForecastStrategy, predictions=predicted, last_origin_position=int(positions[-1]),margin_fraction=margin_fraction)
    cpu, wall = time.process_time(), time.monotonic()
    result = engine.run()[0]
    cpu, wall = time.process_time()-cpu, time.monotonic()-wall
    for decision in result.decisions:
        np.testing.assert_array_equal(np.asarray(decision["predictions"], dtype=float), predicted.loc[pd.Timestamp(decision["time"])])
    equity = pd.Series([x[1] for x in result.equity], index=pd.DatetimeIndex([x[0] for x in result.equity]))
    stats, daily = equity_statistics(equity)
    gross = math.fsum(t["gross_usd"] for t in result.closed_trades)
    commission = math.fsum(f["commission_usd"] for f in result.fills)
    friction = math.fsum(f["friction_usd"] for f in result.fills)
    net = math.fsum(t["net_usd"] for t in result.closed_trades)
    if abs(gross-commission-friction-costs.swap_debits-net) > 1e-5 or abs(net-stats["profit_usd"]) > 1e-5:
        raise RuntimeError("Cost/trade/equity reconciliation failed")
    if result.rejections:
        raise RuntimeError(f"Rejected orders: {result.rejections}")
    daily.to_csv(out/"daily_equity.csv", index_label="date", header=["equity_usd"])
    pd.DataFrame(result.fills, columns=["time","created","reason","order_type","requested_price","tp","sl","decision_balance_usd","fill_balance_usd","requested_units","entry_margin_usd","size","price","commission_usd",
                 "friction_usd","all_costs_with_swap_usd"]).to_csv(out/"fills.csv", index=False)
    pd.DataFrame(result.closed_trades, columns=["open","close","gross_usd","net_usd","costs_usd"]).to_csv(out/"trades.csv", index=False)
    write_json(out/"decisions.json", result.decisions)
    write_json(out/"ambiguous_bars.json", engine.broker.ambiguous_bars)
    record = {"cell": cell, **stats, "n_origins": len(origins),
        "mae_by_horizon": dict(zip(map(str, OFFSETS), map(float, masked_mae(predictions, target)))),
        "n_by_horizon": dict(zip(map(str, OFFSETS), map(int, np.isfinite(target).sum(axis=0)))),
        "n_closed_trades": len(result.closed_trades), "n_rejected_orders": result.rejections,
        "commission_usd": commission, "friction_usd": friction, "swap_usd": costs.swap_debits,
        "final_position": 0, "n_ambiguous_bars": len(engine.broker.ambiguous_bars),
        "max_margin_fraction": margin_fraction,
        "n_early_exits": sum(f["reason"] == "FORECAST_SL_FIRST" for f in result.fills),
        "cpu_seconds": cpu, "wall_seconds": wall,
        "artifacts": {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"result.json", record)
    return record


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    margin_fraction = getattr(args,"margin_fraction",None)
    frame = pd.read_csv(args.data, parse_dates=["DATE_TIME"], index_col="DATE_TIME", float_precision="round_trip")
    origins, target, positions = elapsed_targets(frame)
    current = np.repeat(frame.CLOSE.iloc[positions].to_numpy()[:, None], 12, axis=1)
    current[~np.isfinite(target)] = np.nan
    naive = masked_mae(current, target)
    draws = [np.random.default_rng(s).standard_normal(target.shape) for s in SEEDS]
    sigma = equivalent_sigma(float(naive[-1]), draws)
    ratios = [0, 1] if args.pilot else RATIOS
    seeds = [42] if args.pilot else SEEDS
    shutil.copy2(args.data, out/"input.csv")
    shutil.copy2(__file__, out/"runner.py")
    shutil.copy2(ROOT/"tools/price_noise_sweep.py", out/"shared_helpers.py")
    shutil.copy2(ROOT/"app/policies/prediction_entry_exit.py", out/"legacy_policy.py")
    shutil.copy2(ROOT/"docs/noise_price_sweep/CORRECTED_PLAN.md", out/"PLAN.md")
    if margin_fraction is not None:
        shutil.copy2(ROOT/"docs/noise_price_sweep/MARGIN5_PLAN.md",out/"MARGIN_PLAN.md")
    np.save(out/"targets.npy", target, allow_pickle=False)
    pd.DataFrame({"DATE_TIME": origins, "source_row": positions}).to_csv(out/"origins.csv", index=False)
    for seed, z in zip(SEEDS, draws):
        np.save(out/f"standard_normal_seed{seed}.npy", z, allow_pickle=False)
    manifest = {"kind": "PRICE_NOISE_SWEEP_MARGIN5_V3" if margin_fraction is not None else "CORRECTED_ELAPSED_PROTECTED_PRICE_SWEEP_V2", "pilot": args.pilot,
        "n_bars": len(frame), "n_origins": len(origins), "excluded_origins": len(frame)-len(origins),
        "origin_first": str(origins[0]), "origin_last": str(origins[-1]),
        "liquidation_last": str(frame.index[positions[-1]+2]), "horizons_hours": OFFSETS.tolist(),
        "horizon_semantics": "exact elapsed hours; H6/H144 mandatory; unavailable intermediates explicitly absent",
        "n_by_horizon": dict(zip(map(str, OFFSETS), map(int, np.isfinite(target).sum(axis=0)))),
        "naive_mae": dict(zip(map(str, OFFSETS), map(float, naive))),
        "sigma_ref_measured_mean_H144": sigma, "seeds": seeds, "ratios": ratios,
        "execution": {"initial_usd": 10000, "commission_per_lot_side_usd": 7,
                      "fixed_friction_price_per_side": .00010, "swap_per_lot_day_usd": 10,
                      "ambiguous_bar": "SL_FIRST_UNLESS_OPEN_ALREADY_CROSSED_TP",
                      "native_bracket_entry_bar_protected": True, "forecast_rule": "ORDERED_FIRST_PASSAGE",
                      "noise_all_twelve_columns": True},
        "versions": {"backtrader": bt.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "files": {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    manifest["execution"].update(max_margin_fraction=margin_fraction,leverage=100,
                                sizing="margin budget, currency conversion and execution gap cap" if margin_fraction is not None else "legacy units cap")
    write_json(out/"manifest.json", manifest)
    records = []
    try:
        for ratio in ratios:
            for seed in seeds:
                cell = f"noise_{ratio:g}_seed{seed}"
                pred = target + sigma*ratio*draws[SEEDS.index(seed)]
                r = run_cell(frame, origins, target, positions, pred, out/cell, cell,margin_fraction=margin_fraction)
                r.update(kind="noise", noise_ratio=ratio, sigma_price=sigma*ratio, seed=seed)
                write_json(out/cell/"result.json", r)
                records.append(r)
                write_json(out/"results.json", records)
                print(json.dumps({"cell": cell, "profit": r["profit_usd"], "trades": r["n_closed_trades"]}), flush=True)
        r = run_cell(frame, origins, target, positions, current, out/"persistence", "persistence",margin_fraction=margin_fraction)
        r.update(kind="persistence", noise_ratio=None, sigma_price=None, seed=None)
        write_json(out/"persistence/result.json", r)
        records.append(r)
        write_json(out/"results.json", records)
        pd.DataFrame([{k:v for k,v in r.items() if k not in {"artifacts", "mae_by_horizon"}} |
                      {f"mae_h{h}":v for h,v in r["mae_by_horizon"].items()} for r in records]).to_csv(out/"sweep.csv", index=False)
        write_json(out/"completion.json", {"status":"COMPLETE", "cells":len(records),
            "manifest_sha256":digest(out/"manifest.json"), "results_sha256":digest(out/"results.json"),
            "sweep_sha256":digest(out/"sweep.csv")})
    except Exception as exc:
        write_json(out/"FAILED.json", {"type":type(exc).__name__, "message":str(exc), "completed":len(records)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT/"tests/data/phase_2_3_base_d3.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--margin-fraction",type=float,default=.05)
    run(parser.parse_args())

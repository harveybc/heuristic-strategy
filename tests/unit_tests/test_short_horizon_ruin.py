"""Ruin handling: an account at or below zero equity has no Sharpe and stops trading (realistic runs).

Written before the ``ruined`` metric, the ``ruin_floor`` stop-out and the ruin-aware validity existed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.short_horizon import analysis as A
from app.short_horizon import blocks as B
from app.short_horizon import execution as X
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars


def losing_orders(bl, bars, units):
    """Market orders that always lose (stop placed 1 pip under the open of a falling block)."""
    out = []
    for k in range(len(bl)):
        if bl.status[k] != B.OK:
            out.append(None)
            continue
        out.append(X.BlockOrder(1, "MARKET", None, 50.0, 0.0001, units))
    return out


def falling_market():
    bars = synthetic_bars(seed=60, segments=4)
    n = len(bars)
    close = 1.30 - 0.0015 * np.arange(n)          # steady fall: every long loses
    bars = bars.assign(CLOSE=close, OPEN=np.r_[1.30, close[:-1]])
    bars["HIGH"] = bars[["OPEN", "CLOSE"]].max(axis=1) + 0.00005
    bars["LOW"] = bars[["OPEN", "CLOSE"]].min(axis=1) - 0.00005
    return bars


def test_ruined_account_has_no_sharpe_and_stop_out_halts_orders():
    bars = falling_market()
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    orders = losing_orders(bl, bars, 1_000_000)
    free = X.run_backtest(bars, bl, orders, X.Costs(), cal, 10_000.0, split)
    assert free.metrics["ruined"] and free.metrics["end_equity_usd"] < 0
    assert math.isnan(free.metrics["sharpe_daily_sqrt365"]) and math.isnan(free.metrics["sortino_daily_sqrt365"])
    stop = X.run_backtest(bars, bl, orders, X.Costs(), cal, 10_000.0, split, ruin_floor=0.0)
    assert stop.metrics["ruined"] and stop.metrics["ruin_skipped_orders"] > 0
    assert stop.metrics["trades"] < free.metrics["trades"]
    # once realized equity is at or below zero no further entry happens
    realized, stopped_at = 10_000.0, None
    for t in sorted(stop.trades, key=lambda r: r["entry_time"]):
        assert stopped_at is None or t["entry_time"] < stopped_at
        realized += t["net_usd"]
        if realized <= 0 and stopped_at is None:
            stopped_at = t["exit_time"]


def test_healthy_account_is_not_ruined():
    bars = synthetic_bars(seed=61, segments=3)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    res = X.run_backtest(bars, bl, [None] * len(bl), X.Costs(), cal, 10_000.0, split, ruin_floor=0.0)
    assert res.metrics["ruined"] is False and res.metrics["ruin_skipped_orders"] == 0


def test_validity_excludes_ruined_seeds_and_derives_ruin_from_drawdown_pct():
    df = pd.DataFrame({"level": [0] * 4, "seed": [1, 2, 3, 4], "trades_per_week": [5.0] * 4,
                       "max_orders_per_day": [1] * 4, "max_drawdown_pct": [20.0, 100.0, 140.0, 30.0]})
    seeds, levels = A.frequency_validity(df, 3, 4, min_valid_seeds=3)
    assert seeds.tolist() == [True, False, False, True] and levels.to_dict() == {0: False}
    assert A.ruined(df).tolist() == [False, True, True, False]

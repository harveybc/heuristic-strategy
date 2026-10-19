"""Owner rule 2026-10-19: >= 3 (variant 4) trades per business week, <= 4 (variant 2) orders per day.

Written before the order cap, the frequency metrics and analysis.frequency_validity existed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import analysis as A
from app.short_horizon import blocks as B
from app.short_horizon import execution as X
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars


def market_orders(bl):
    return [X.BlockOrder(1, "MARKET", None, 50.0, 0.0001, 10_000) if bl.status[k] == B.OK else None
            for k in range(len(bl))]


@pytest.mark.parametrize("cap", [4, 2, 1])
def test_order_cap_per_business_day_is_enforced_in_the_strategy(cap):
    bars = synthetic_bars(seed=40, segments=3)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    res = X.run_backtest(bars, bl, market_orders(bl), X.Costs(), cal, 10_000.0, split, max_orders_per_day=cap)
    days = {}
    for t in res.trades:
        seg = cal.segments[t["segment"]]
        key = (t["segment"], (t["entry_time"] - seg.market_open) // pd.Timedelta(hours=24))
        days[key] = days.get(key, 0) + 1
    assert max(days.values()) <= cap
    assert res.metrics["max_orders_per_day"] <= cap
    uncapped = X.run_backtest(bars, bl, market_orders(bl), X.Costs(), cal, 10_000.0, split)
    assert uncapped.metrics["max_orders_per_day"] == 4  # B = 6 tiles a business day into at most 4 blocks
    # the cap only removes later orders of the same day: the first order of each day is unchanged
    first = {}
    for t in uncapped.trades:
        seg = cal.segments[t["segment"]]
        key = (t["segment"], (t["entry_time"] - seg.market_open) // pd.Timedelta(hours=24))
        first.setdefault(key, t["entry_time"])
    capped_first = {}
    for t in res.trades:
        seg = cal.segments[t["segment"]]
        key = (t["segment"], (t["entry_time"] - seg.market_open) // pd.Timedelta(hours=24))
        capped_first.setdefault(key, t["entry_time"])
    assert first == capped_first


def test_trades_per_week_counts_weeks_with_eligible_blocks():
    bars = synthetic_bars(seed=41, segments=3)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 24)
    split = (bars.index[0], bars.index[-1] + HOUR)
    res = X.run_backtest(bars, bl, market_orders(bl), X.Costs(), cal, 10_000.0, split)
    weeks = {tuple(bl.start[k].isocalendar()[:2]) for k in res.evaluated_blocks}
    assert res.metrics["weeks_with_eligible_blocks"] == len(weeks)
    assert res.metrics["trades_per_week"] == pytest.approx(len(res.trades) / len(weeks))
    none = X.run_backtest(bars, bl, [None] * len(bl), X.Costs(), cal, 10_000.0, split)
    assert none.metrics["trades_per_week"] == 0 and none.metrics["max_orders_per_day"] == 0


def test_frequency_validity_drops_seeds_then_levels():
    df = pd.DataFrame({"level": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                       "seed": [1, 2, 3, 4, 5] * 2,
                       "trades_per_week": [5, 5, 2, 5, 5, 5, 2, 2, 2, 5],
                       "max_orders_per_day": [1] * 10,
                       "sharpe": [1.0, 2.0, 99.0, 3.0, 4.0, 1.0, 9.0, 9.0, 9.0, 1.0]})
    seeds, levels = A.frequency_validity(df, min_per_week=3, max_per_day=4, min_valid_seeds=3)
    assert seeds.tolist() == [True, True, False, True, True, True, False, False, False, True]
    assert levels.to_dict() == {0: True, 1: False}
    assert df[seeds & (df.level == 0)].sharpe.mean() == pytest.approx(2.5)  # the invalid seed is excluded
    strict, lv4 = A.frequency_validity(df, min_per_week=6, max_per_day=4)
    assert not strict.any() and not lv4.any()
    capped, _ = A.frequency_validity(df.assign(max_orders_per_day=3), min_per_week=3, max_per_day=2)
    assert not capped.any()

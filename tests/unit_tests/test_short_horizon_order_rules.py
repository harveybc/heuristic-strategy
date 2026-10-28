"""Business hard rule (owner, 2026-10-19): every order carries an effective broker-side SL and TP.

Written before execution.OrderRules / check_order / OrderRuleError, the exit-share metrics and
strategies.apply_order_rules existed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.short_horizon import blocks as B
from app.short_horizon import execution as X
from app.short_horizon import strategies as S
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars

RULES = X.OrderRules()


@pytest.mark.parametrize("tp,sl", [(math.inf, 1.09), (1.11, -math.inf), (None, 1.09), (1.11, None),
                                   (math.nan, 1.09)])
def test_order_without_both_levels_is_refused(tp, sl):
    order = X.BlockOrder(1, "MARKET", None, tp, sl, 10_000)
    with pytest.raises(X.OrderRuleError):
        X.check_order(order, 1.1000, 0.0050, RULES)


def test_levels_on_wrong_side_too_close_or_symbolic_are_refused():
    ok = X.BlockOrder(1, "MARKET", None, 1.1030, 1.0980, 10_000)
    X.check_order(ok, 1.1000, 0.0050, RULES)
    for bad in (X.BlockOrder(1, "MARKET", None, 1.0990, 1.0980, 10_000),          # TP on the wrong side
                X.BlockOrder(-1, "MARKET", None, 1.0970, 1.0990, 10_000),         # SL on the wrong side
                X.BlockOrder(1, "MARKET", None, 1.1000 + RULES.min_distance / 2, 1.0980, 10_000),  # too close
                X.BlockOrder(1, "MARKET", None, 1.1000 + 2 * 0.0050, 1.0980, 10_000)):  # beyond c * sigma
        with pytest.raises(X.OrderRuleError):
            X.check_order(bad, 1.1000, 0.0050, RULES)
    limit = X.BlockOrder(1, "LIMIT", 1.0990, 1.1020, 1.0970, 10_000)               # checked from its own entry
    X.check_order(limit, 1.1000, 0.0050, RULES)


def test_simulator_raises_on_a_symbolic_order_and_strategies_prefilter():
    bars = synthetic_bars(seed=70, segments=4)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    sigma = S.causal_block_sigma(bl)
    k = next(i for i in range(len(bl)) if bl.status[i] == B.OK and np.isfinite(sigma[i]))
    orders = [None] * len(bl)
    orders[k] = X.BlockOrder(1, "MARKET", None, 50.0, 0.0001, 10_000)            # old passive-baseline style
    with pytest.raises(X.OrderRuleError):
        X.run_backtest(bars, bl, orders, X.Costs(), cal, 10_000.0, split, order_rules=RULES, sigma_block=sigma)
    filtered, skipped = S.apply_order_rules(orders, bl, sigma, RULES)
    assert skipped == 1 and filtered[k] is None
    X.run_backtest(bars, bl, filtered, X.Costs(), cal, 10_000.0, split, order_rules=RULES, sigma_block=sigma)


def test_causal_sigma_uses_completed_blocks_only():
    bars = synthetic_bars(seed=71, segments=6)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    s1 = S.causal_block_sigma(bl)
    later = bars.copy()
    ok = np.flatnonzero(bl.status == B.OK)
    last = ok[-1]
    later.iloc[bl.bar_lo[last]:bl.bar_hi[last], later.columns.get_loc("HIGH")] += 0.01
    s2 = S.causal_block_sigma(B.build_blocks(later, cal, 6))
    np.testing.assert_array_equal(s1, s2)


def test_rf_orders_always_carry_effective_levels():
    """Retro-check of the committed range-first orders."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        ref = 1.1
        H, L = ref + rng.uniform(0, 0.006), ref - rng.uniform(0, 0.006)
        ft = rng.choice([B.LOW_FIRST, B.HIGH_FIRST])
        for v in ("RF-M", "RF-L"):
            o = S.range_first_order(v, H, L, ft, ref, X.RealisticProfile(), S.RangeFirstParams(), 10_000.0)
            if o is None:
                continue
            assert np.isfinite([o.take_profit, o.stop_loss]).all()
            entry = o.entry_price if o.entry_type == "LIMIT" else ref
            assert (o.stop_loss < entry < o.take_profit) if o.direction > 0 else (o.take_profit < entry < o.stop_loss)


def test_exit_shares_and_rarely_hit_tp_flag():
    bars = synthetic_bars(seed=72, segments=4)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    orders = [X.BlockOrder(1, "MARKET", None, float(bl.ref[k]) + 0.5, float(bl.ref[k]) - 0.5, 10_000)
              if bl.status[k] == B.OK else None for k in range(len(bl))]
    m = X.run_backtest(bars, bl, orders, X.Costs(), cal, 10_000.0, split).metrics
    assert m["share_exit_tp"] + m["share_exit_sl"] + m["share_exit_time"] + m["share_exit_early"] == pytest.approx(1.0)
    assert m["tp_rarely_hit"] is True

"""MetaTrader ZigZag semantics (Depth, Deviation, Backstep), hindsight vs causal-confirmed pivots,
pivot-reversal blocks and orders. Written before app/short_horizon/zigzag.py existed.

The indicator is NON-CAUSAL: a pivot is only known after a Deviation-sized reversal and Depth bars, and
the algorithm rewrites earlier bars (Backstep, last-extreme replacement). It defines labels and hindsight
experts, never features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import blocks as BL
from app.short_horizon import execution as X
from app.short_horizon import zigzag as Z
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars


def tri(levels, width=5):
    """Piece-wise linear path through ``levels`` with ``width`` bars per leg -> OHLC frame."""
    close = []
    for a, b in zip(levels[:-1], levels[1:]):
        close += list(np.linspace(a, b, width, endpoint=False))
    close.append(levels[-1])
    close = np.array(close)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.00002
    low = np.minimum(open_, close) - 0.00002
    idx = pd.date_range("2020-01-06 00:00", periods=len(close), freq="h", tz="UTC")
    return pd.DataFrame({"OPEN": open_, "HIGH": high, "LOW": low, "CLOSE": close}, index=idx.rename("DATE_TIME"))


@pytest.mark.xfail(reason="WIP (paused by owner reorder 2026-10-19): the Deviation test in MT ZigZag compares the bar to the Depth-window extreme, not to the previous pivot; hand-built expectations to be rebuilt", strict=False)
def test_hand_built_zigzag_finds_alternating_extremes_and_ignores_small_wiggles():
    bars = tri([1.1000, 1.1100, 1.1020, 1.1120, 1.1115, 1.1125, 1.1000], width=6)
    p = Z.Params(depth=3, deviation_pips=5, backstep=2)
    piv = Z.hindsight_pivots(bars.HIGH.to_numpy(), bars.LOW.to_numpy(), p)
    kinds = [k for _, k, _ in piv]
    # alternation is mandatory
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    # the 5-pip wiggle 1.1120 -> 1.1115 -> 1.1125 must not create pivots: the peak is the 1.1125 bar
    highs = [bars.HIGH.to_numpy()[i] for i, k, _ in piv if k == Z.PEAK]
    assert pytest.approx(max(highs), abs=1e-5) == 1.1125 + 0.00002
    assert 1.1120 + 0.00002 not in [round(h, 5) for h in highs]
    lows = [bars.LOW.to_numpy()[i] for i, k, _ in piv if k == Z.VALLEY]
    assert pytest.approx(min(lows), abs=1e-5) == 1.1000 - 0.00002


@pytest.mark.xfail(reason="WIP (paused by owner reorder 2026-10-19): series must be rebuilt so the prefix peak is replaced under MT semantics", strict=False)
def test_zigzag_is_non_causal_and_rewrites_the_past():
    bars = tri([1.1000, 1.1100, 1.1050, 1.1200, 1.1000], width=8)
    p = Z.Params(depth=4, deviation_pips=10, backstep=3)
    full = Z.hindsight_pivots(bars.HIGH.to_numpy(), bars.LOW.to_numpy(), p)
    # on the prefix ending just after the first peak, the algorithm names that peak; later it is replaced
    k = 10
    prefix = Z.hindsight_pivots(bars.HIGH.to_numpy()[:k], bars.LOW.to_numpy()[:k], p)
    peaks_prefix = [i for i, kind, _ in prefix if kind == Z.PEAK]
    peaks_full = [i for i, kind, _ in full if kind == Z.PEAK]
    assert peaks_prefix and peaks_prefix[0] not in peaks_full


def test_confirmation_bar_is_after_the_pivot_and_the_prefix_contains_it_from_then_on():
    bars = synthetic_bars(seed=11, segments=1, hours=600, start="2020-01-05 21:00")
    p = Z.Params(depth=12, deviation_pips=5, backstep=3)
    h, l = bars.HIGH.to_numpy(), bars.LOW.to_numpy()
    piv = Z.hindsight_pivots(h, l, p)
    conf = Z.confirmation_bars(h, l, piv, p)
    assert len(conf) == len(piv)
    checked = 0
    for (i, kind, val), c in zip(piv, conf):
        if c < 0:
            continue
        assert c >= i + 1
        inside = {(j, kd) for j, kd, _ in Z.hindsight_pivots(h[:c + 1], l[:c + 1], p)}
        assert (i, kind) in inside
        checked += 1
    assert checked >= 5
    lags = Z.lag_distribution(piv, conf)
    assert lags["n"] == checked and lags["median"] >= 1


def test_pivot_blocks_and_reversal_orders_carry_sl_tp_units_and_respect_forced_flat():
    bars = synthetic_bars(seed=12, segments=2, hours=240, start="2020-01-05 21:00")
    cal = calendar_for(bars, delay=1, flat=1, cutoff=1)
    p = Z.Params(depth=6, deviation_pips=3, backstep=2)
    h, l = bars.HIGH.to_numpy(), bars.LOW.to_numpy()
    piv = Z.hindsight_pivots(h, l, p)
    assert len(piv) >= 4
    sig = Z.signals_from_pivots(piv, delay=0)
    sigma = Z.causal_sigma(bars)
    blocks = Z.signal_blocks(bars, cal, sig)
    assert isinstance(blocks, BL.Blocks)
    for k in range(len(blocks)):
        if blocks.status[k] == BL.OK:
            seg = cal.segments[blocks.segment[k]]
            assert blocks.end[k] <= seg.force_flat
    orders = Z.reversal_orders(blocks, sig, piv, sigma, mode="hindsight", m_sl=2.0, inset=1e-4)
    ok = [(k, o) for k, o in enumerate(orders) if o is not None]
    assert ok
    for k, o in ok:
        assert o.direction in (1, -1) and np.isfinite(o.take_profit) and np.isfinite(o.stop_loss)
        assert (o.take_profit > blocks.ref[k] > o.stop_loss) if o.direction > 0 else (o.take_profit < blocks.ref[k] < o.stop_loss)
    rb = Z.reversal_orders(blocks, sig, piv, sigma, mode="causal", k_tp=1.0, m_sl=2.0)
    assert any(o is not None for o in rb)
    res = X.run_backtest(bars, blocks, orders, X.RealisticProfile(), cal, 10_000.0,
                         (bars.index[0], bars.index[-1] + HOUR), profile=X.RealisticProfile(),
                         order_rules=X.OrderRules(), sigma_block=Z.causal_range_sigma(bars), sizing=X.Sizing())
    for t in res.trades:
        assert t["units"] >= 1 and np.isfinite(t["stop_loss"]) and np.isfinite(t["take_profit"])


def test_signal_delay_and_jitter_only_move_signals_forward_or_by_the_jitter():
    piv = [(10, Z.VALLEY, 1.0), (30, Z.PEAK, 1.1), (50, Z.VALLEY, 1.0)]
    s0 = Z.signals_from_pivots(piv, delay=0)
    s2 = Z.signals_from_pivots(piv, delay=2)
    assert [b for b, _ in s2] == [b + 2 for b, _ in s0]
    rng = np.random.default_rng(0)
    sj = Z.jitter_signals(s0, rng, max_bars=3)
    assert all(abs(a - b) <= 3 for (a, _), (b, _) in zip(sj, s0))


def test_demonstration_csv_is_sealed_and_labelled(tmp_path):
    trades = [{"entry_time": pd.Timestamp("2020-01-06 01:00", tz="UTC"), "direction": 1, "units": 1000, "entry_price": 1.1,
               "stop_loss": 1.09, "take_profit": 1.12, "exit_time": pd.Timestamp("2020-01-06 05:00", tz="UTC"),
               "exit_reason": "TP"}]
    man = Z.write_demonstrations(tmp_path, trades, params=Z.Params(12, 5, 3), data_sha256="abc", costs={"spread": 1})
    assert man["label"] == "HINDSIGHT_NON_CAUSAL_EXPERT"
    import hashlib
    assert man["sha256"] == hashlib.sha256((tmp_path / "expert_trades.csv").read_bytes()).hexdigest()
    df = pd.read_csv(tmp_path / "expert_trades.csv")
    assert list(df.columns) == ["t_open", "side", "units", "entry", "sl", "tp", "t_close", "reason"]

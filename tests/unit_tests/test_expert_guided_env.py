"""Expert-guided ladder: environment, action and business rules (written before any fit).

Owner hard rules: every order has an effective SL and TP and integer units from the formula; no entry in
the first 12 h after a reopen, flat 12 h before a closure; at most 4 entries per business day; the weekly
rule is a validity flag; entries and closes are counted separately; decide after the completed bar, fill at
the next open. Plus the hindsight labels' support/censoring and the observation builder's prefix invariance.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import execution as X
from app.short_horizon import zigzag as Z
from app.short_horizon.expert_guided import data as DG
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import oracle as OR
from app.short_horizon import calendar as CAL

HOUR = pd.Timedelta(hours=1)


def synthetic_bars(seed=0, segments=2, hours=120, start="2024-01-07 17:00", gap_hours=49):
    """Same generator as tests/unit_tests/test_short_horizon.py (copied: that module imports app.policies,
    which needs trading_contracts, absent in the bounded test environment)."""
    rng = np.random.default_rng(seed)
    stamps = []
    t = pd.Timestamp(start, tz="UTC")
    for _ in range(segments):
        stamps += [t + k * HOUR for k in range(hours)]
        t = stamps[-1] + gap_hours * HOUR
    n = len(stamps)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    open_ = np.r_[1.10, close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.0006, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.0006, n)
    return pd.DataFrame({"OPEN": open_, "HIGH": high, "LOW": low, "CLOSE": close},
                        index=pd.DatetimeIndex(stamps, name="DATE_TIME"))


def calendar_for(bars, delay=12, flat=12, cutoff=0):
    segs = CAL.derive_segments(bars.index, max_gap_hours=24)
    return CAL.load_calendar({"calendar_id": "TEST", "version": "t1", "segments": segs,
                              "entry_delay_after_open_hours": delay,
                              "force_flat_before_close_hours": flat,
                              "entry_cutoff_before_flat_hours": cutoff})


def _env(seed=3, segments=3, hours=120, **cfg):
    bars = synthetic_bars(seed=seed, segments=segments, hours=hours, start="2020-01-05 21:00")
    cal = calendar_for(bars, delay=12, flat=12, cutoff=0)
    sigma = np.full(len(bars), 0.0010)
    sigma_rule = np.full(len(bars), 0.0030)
    env = E.BarEnv(bars, cal, sigma, sigma_rule, bars.index[0], bars.index[-1] + HOUR, E.EnvConfig(**cfg))
    return bars, cal, env


def _first_open_bar(env, seg=0):
    s = env.cal.segments[seg]
    return int(np.searchsorted(env.v, s.operational_start.value)) - 1    # decision bar whose fill bar is the op start


def test_entry_fills_at_the_next_open_with_effective_levels_and_formula_units():
    bars, cal, env = _env()
    b = _first_open_bar(env) + 2
    info = env.step(b, E.LONG, tp_mult=1.5, sl_mult=2.0)
    assert info["entry"] or info["close"] in ("SL", "TP")
    t = env.trades[0] if env.trades else None
    p = env.pos
    entry_price = p.entry_price if p else t["entry_price"]
    assert entry_price == bars.OPEN.iloc[b + 1]
    sl = p.sl if p else t["stop_loss"]
    tp = p.tp if p else t["take_profit"]
    ref = bars.CLOSE.iloc[b]
    assert sl < ref < tp
    assert math.isclose(tp - ref, 1.5 * 0.0010) and math.isclose(ref - sl, 2.0 * 0.0010)
    units = p.units if p else t["units"]
    assert units == math.floor(1.0 * 0.10 * 10_000.0 * 100.0 / ref) and isinstance(units, int) and units >= 1


def test_no_entry_in_the_first_12_hours_after_a_reopen_and_none_into_the_forced_flat():
    bars, cal, env = _env()
    seg = cal.segments[1]
    b_reopen = int(np.searchsorted(env.v, seg.market_open.value))
    for b in range(b_reopen, b_reopen + 11):
        ok, why = env.entry_gate(b)
        assert not ok and why == "BEFORE_OPERATIONAL_START"
        assert env.step(b, E.SHORT)["refused"] == "BEFORE_OPERATIONAL_START"
    b_ff = int(np.searchsorted(env.v, seg.force_flat.value)) - 2      # fill bar ends at the forced flat
    ok, why = env.entry_gate(b_ff)
    assert not ok and why == "ENTRY_CUTOFF"
    assert env.counts["entries"] == 0


def test_positions_are_closed_before_the_forced_flat_and_never_cross_a_closure():
    bars, cal, env = _env(sl_bounds=(2.9, 3.0), tp_bounds=(2.9, 3.0))
    b = _first_open_bar(env)
    env.step(b, E.LONG, tp_mult=3.0, sl_mult=3.0)
    for k in range(b + 1, env.last):
        env.step(k, E.HOLD)
    m = env.result()
    for t in env.trades:
        seg = cal.segments[t["segment"]]
        assert t["exit_time"] <= seg.force_flat
        assert cal.segment_of(t["entry_time"]) == cal.segment_of(t["exit_time"] - HOUR)
    assert m["n_entries"] == m["n_closes"]


def test_daily_cap_refuses_the_fifth_entry_of_a_business_day():
    bars, cal, env = _env()
    b = _first_open_bar(env)
    entries = 0
    for k in range(b, b + 12):
        info = env.step(k, E.LONG if k % 2 == 0 else E.SHORT, tp_mult=3.0, sl_mult=3.0)
        entries += int(info["entry"])
    assert entries <= 4
    assert env.counts["refused_DAILY_CAP"] >= 1
    assert max(env.entries_day.values()) == 4


def test_reverse_is_one_close_plus_one_entry_both_costed_and_same_direction_is_ignored():
    bars, cal, env = _env(tp_bounds=(2.9, 3.0), sl_bounds=(2.9, 3.0))
    b = _first_open_bar(env)
    env.step(b, E.LONG, tp_mult=3.0, sl_mult=3.0)
    if env.pos is None:
        pytest.skip("protected out on the entry bar")
    env.step(b + 1, E.LONG, tp_mult=3.0, sl_mult=3.0)
    assert env.counts["ignored_same_direction"] == 1 and env.counts["entries"] == 1
    info = env.step(b + 2, E.SHORT, tp_mult=3.0, sl_mult=3.0)
    assert info["close"] == "REVERSAL" and info["entry"]
    assert env.counts["entries"] == 2 and env.counts["closes"] >= 1
    t = env.trades[0]
    assert t["close_reason"] == "REVERSAL" and t["exit_price"] == bars.OPEN.iloc[b + 3]
    assert t["entry_cost_usd"] > 0 and t["exit_cost_usd"] > 0
    assert env.pos.entry_price == bars.OPEN.iloc[b + 3]


def test_stop_before_take_profit_inside_one_bar_and_gap_fills_at_open():
    assert X._protect(1, 1.105, 1.095, 1.100, 1.106, 1.094)[0] == "SL"
    r = X._protect(1, 1.105, 1.095, 1.090, 1.10, 1.08)
    assert r[0] == "SL" and r[1] == 1.090 and r[2]


def test_ledger_reconciles_costs_include_swap_and_weekly_validity_flag():
    bars, cal, env = _env(tp_bounds=(2.9, 3.0), sl_bounds=(2.9, 3.0))
    b = _first_open_bar(env)
    env.step(b, E.LONG, tp_mult=3.0, sl_mult=3.0)
    for k in range(b + 1, env.last):
        env.step(k, E.HOLD)
    m = env.result()                         # _account raises when the cash ledger does not reconcile
    recon = m["gross_usd"] - sum(m[f"cost_{c}_usd"] for c in ("spread", "slippage", "commission", "financing"))
    assert math.isclose(recon, m["net_profit_usd"], rel_tol=1e-9, abs_tol=1e-6)
    assert m["orders_per_week"] < 3 and not m["valid_weekly_rule"] and not m["valid"]
    held = [t for t in env.trades if t["rollovers"] > 0]
    assert all(t["financing_usd"] != 0 for t in held)


def test_levels_are_clipped_into_the_effective_band_for_students_and_refused_for_the_expert():
    bars, cal, env = _env()
    b = _first_open_bar(env)
    env.sigma_rule[:] = 0.0010                 # effective band: <= 1.5 x 0.0010
    env.step(b, E.LONG, tp_mult=3.0, sl_mult=3.0)
    assert env.counts["levels_clipped"] == 1
    p = env.pos or None
    if p is not None:
        assert abs(p.tp - bars.CLOSE.iloc[b]) <= 1.5 * 0.0010 + 1e-12
    bars2, cal2, env2 = _env(level_policy="refuse")
    env2.sigma_rule[:] = 0.0010
    assert env2.step(b, E.LONG, tp_mult=3.0, sl_mult=3.0)["refused"] == "LEVELS"


def test_agent_state_is_known_after_the_bar_and_has_declared_names():
    bars, cal, env = _env()
    b = _first_open_bar(env)
    s = env.state(b)
    assert s.shape == (len(E.STATE_NAMES),) and s[0] == 0 and s[-1] == 1.0
    env.step(b, E.SHORT, tp_mult=3.0, sl_mult=3.0)
    if env.pos is not None:
        s2 = env.state(b + 1)
        assert s2[0] == -1 and s2[3] > 0 and s2[4] > 0


# ------------------------------------------------------------------ labels, oracle, observations
def _pivots(n=400, seed=5):
    bars = synthetic_bars(seed=seed, segments=1, hours=n, start="2020-01-05 21:00")
    h, l = bars.HIGH.to_numpy(), bars.LOW.to_numpy()
    p = Z.Params(3, 5.0, 0)
    piv = Z.hindsight_pivots(h, l, p)
    conf = Z.confirmation_bars_fast(h, l, piv, p)
    return bars, piv, conf


def test_hindsight_labels_support_and_censoring():
    bars, piv, conf = _pivots()
    n = len(bars)
    y = OR.pivot_labels(n, piv)
    sup = OR.label_support_end(n, piv, conf)
    for (i, kind, _), c in zip(piv, conf):
        assert y[i] == (OR.LONG_CLASS if kind == Z.VALLEY else OR.SHORT_CLASS)
        assert sup[i] == c
    # a bar after the last pivot (or whose next pivot never locks) is censored, never NONE-with-support
    last = piv[-1][0]
    assert (sup[last + 1:] == -1).all()
    assert (sup[sup >= 0] >= np.flatnonzero(sup >= 0)).all()       # support never precedes the bar
    # repaint/tail perturbation after the support end leaves the label unchanged (A02)
    b = piv[len(piv) // 2][0]
    cut = sup[b] + 1
    h, l = bars.HIGH.to_numpy().copy(), bars.LOW.to_numpy().copy()
    rng = np.random.default_rng(0)
    h2, l2 = h.copy(), l.copy()
    h2[cut:] += rng.uniform(0, 0.01, n - cut)
    l2[cut:] -= rng.uniform(0, 0.01, n - cut)
    piv2 = Z.hindsight_pivots(h2, l2, Z.Params(3, 5.0, 0))
    y2 = OR.pivot_labels(n, piv2)
    assert (y2[:b + 1] == y[:b + 1]).all()


def test_confirmed_labels_are_at_confirmation_bars():
    bars, piv, conf = _pivots()
    yc, _ = OR.confirmed_labels(len(bars), piv, conf)
    for (i, kind, _), c in zip(piv, conf):
        if c >= 0:
            assert yc[c] != 0 and c >= i


def test_leg_oracle_reproduces_the_expert_at_pivots_and_relabels_any_state():
    bars, piv, conf = _pivots()
    c = bars.CLOSE.to_numpy()
    orc = OR.LegOracle(c, piv, min_remaining_sigma=0.0)
    checked = 0
    for i, kind, lvl in piv[:-1]:
        if orc.remaining[i] < 0:       # the next pivot level is on the wrong side of close[i]: the expert's
            assert orc.action(i, 0, True, 0.001)[0] == E.HOLD      # hindsight order would be invalid too
            continue
        checked += 1
        a, tp, sl = orc.action(i, 0, True, 0.001)
        assert a == (E.LONG if kind == Z.VALLEY else E.SHORT)
        assert sl == 2.0
        d = 1 if kind == Z.VALLEY else -1
        a2, _, _ = orc.action(i, d, True, 0.001)
        assert a2 == E.HOLD
        a3, _, _ = orc.action(i, -d, False, 0.001)
        assert a3 == E.CLOSE
        a4, _, _ = orc.action(i, -d, True, 0.001)
        assert a4 == (E.LONG if d > 0 else E.SHORT)
    assert checked > len(piv) // 2


def test_observation_windows_are_prefix_invariant_and_exclude_the_current_row_from_scaling():
    rng = np.random.default_rng(1)
    X1 = rng.normal(size=(600, 4))
    w1, ok1 = DG.causal_rolling_windows(X1)
    X2 = X1.copy()
    X2[400:] += rng.normal(size=(200, 4)) * 5
    w2, ok2 = DG.causal_rolling_windows(X2)
    assert np.allclose(w1[:400][ok1[:400]], w2[:400][ok2[:400]])
    assert not ok1[:256].any() and ok1[300]
    i = 300
    mu, sd = X1[i - 256:i].mean(0), X1[i - 256:i].std(0)
    assert np.allclose(w1[i, -1], (X1[i] - mu) / sd, atol=1e-5)
    assert np.allclose(w1[i, 0], (X1[i - 24] - X1[i - 24 - 256:i - 24].mean(0)) / X1[i - 24 - 256:i - 24].std(0), atol=1e-5)


def test_calendar_from_closures_is_the_exact_complement():
    closures = [["2020-01-03T22:00:00+00:00", "2020-01-05T22:00:00+00:00"],
                ["2020-01-10T22:00:00+00:00", "2020-01-12T22:00:00+00:00"]]
    cal = DG.calendar_from_closures(closures)
    s = cal.segments[0]
    assert s.market_open == pd.Timestamp("2020-01-05T22:00:00+00:00")
    assert s.market_close == pd.Timestamp("2020-01-10T22:00:00+00:00")
    assert s.operational_start == s.market_open + 12 * HOUR and s.force_flat == s.market_close - 12 * HOUR

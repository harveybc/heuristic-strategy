"""Declared execution-layer variants (fix lane F3, audit A4-1/2/4/7/9/10/12): the default engine path stays
byte-identical; each variant does exactly what it declares on synthetic bars with a synthetic weekly calendar."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from app.short_horizon.expert_guided import data as DG
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import exec_variants as V

SIG, SIG_RULE = 0.003, 0.004          # constant causal sigma / range sigma -> effective cap 1.5 * 0.004 / 0.003 = 2 sigma


def synth(weeks=2, seed=0, drift=0.0):
    """Hourly bars Sun 22:00 -> Fri 21:00 for ``weeks`` weeks from 2021-01-03, weekend closures, flat-ish prices."""
    rng = np.random.default_rng(seed)
    idx, closures = [], [("2021-01-01T22:00:00+00:00", "2021-01-03T22:00:00+00:00")]
    t = pd.Timestamp("2021-01-03 22:00", tz="UTC")
    for w in range(weeks):
        idx += list(pd.date_range(t, periods=5 * 24 - 1, freq="h"))
        close_at = t + pd.Timedelta(hours=5 * 24 - 1)
        closures.append((close_at.isoformat(), (close_at + pd.Timedelta(hours=48)).isoformat()))
        t = close_at + pd.Timedelta(hours=48)
    idx = pd.DatetimeIndex(idx)
    n = len(idx)
    r = rng.normal(drift, 0.0004, n)
    c = 1.1 + np.cumsum(r)
    o = np.r_[1.1, c[:-1]]
    h = np.maximum(o, c) + 0.0003
    l = np.minimum(o, c) - 0.0003
    bars = pd.DataFrame({"OPEN": o, "HIGH": h, "LOW": l, "CLOSE": c}, index=idx)
    cal = DG.calendar_from_closures(closures)
    return bars, cal, np.full(n, SIG), np.full(n, SIG_RULE)


def run(bars, cal, sig, sr, cfg, act):
    env = E.BarEnv(bars, cal, sig, sr, bars.index[0], bars.index[-1] + pd.Timedelta(hours=1), cfg)
    for b in range(env.first, env.last):
        st = env.state(b)
        a, tp, sl = act(b, b, st, env)
        env.step(b, a, tp, sl)
    return env, env.result()


CAP1 = E.EnvConfig(level_policy="clip", max_entries_per_day=1, sizing=E.X.Sizing(rel_volume=0.10, leverage=50.0))


def random_policy(seed=1, p=0.3):
    rng = np.random.default_rng(seed)

    def f(b, r, st, env):
        if int(st[0]) != 0:
            return (E.CLOSE if rng.random() < 0.2 else E.HOLD), 1.0, 1.0
        if rng.random() < p:
            return (E.LONG if rng.random() < 0.5 else E.SHORT), 1.0, 1.0
        return E.HOLD, 1.0, 1.0
    return f


def test_identity_and_legacy_default():
    assert V.LEGACY.is_legacy() and V.ExecVariant().identity == V.LEGACY.identity
    assert not V.ExecVariant(sizing_mode="risk").is_legacy()
    assert V.ExecVariant(sizing_mode="risk").identity != V.LEGACY.identity
    with pytest.raises(ValueError):
        V.ExecVariant(sizing_mode="kelly")
    with pytest.raises(ValueError):
        V.ExecVariant(risk_pct=0.5)


def test_default_path_is_byte_identical_with_and_without_the_legacy_variant():
    bars, cal, sig, sr = synth()
    _, m0 = run(bars, cal, sig, sr, CAP1, random_policy())
    _, m1 = run(bars, cal, sig, sr, dataclasses.replace(CAP1, variant=V.LEGACY), random_policy())
    for k in ("net_profit_usd", "trades", "n_entries", "n_close_SL", "n_close_TP", "n_refused_DAILY_CAP"):
        assert m0[k] == m1[k]


def test_risk_sizing_units_follow_the_stop_distance_and_the_rel_volume_cap():
    bars, cal, sig, sr = synth()
    cfg = dataclasses.replace(CAP1, variant=V.ExecVariant(sizing_mode="risk", risk_pct=0.01))
    opened = {}

    def f(b, r, st, env):
        if int(st[0]) == 0 and not opened and env.entry_gate(b)[0]:
            opened["b"] = b
            return E.LONG, 1.0, 2.0
        return E.HOLD, 1.0, 2.0
    env, m = run(bars, cal, sig, sr, cfg, f)
    t = env.trades[0]
    dist = abs(t["entry_price"] - t["stop_loss"])
    expected = int(np.floor(0.01 * 10_000 / abs(bars.CLOSE.iloc[opened["b"]] - t["stop_loss"])))
    assert t["units"] == expected and t["units"] <= cfg.sizing.units_max(10_000, bars.CLOSE.iloc[opened["b"]])
    assert t["units"] * dist <= 0.01 * 10_000 * 1.05                  # the stop risks ~1 % (fill at the next open)
    # the legacy sizing on the same order is the fixed 5x notional exposure (much larger here)
    legacy = CAP1.sizing.order_units(10_000, float(bars.CLOSE.iloc[opened["b"]]), 1.0)
    assert legacy > expected


def test_risk_sizing_refuses_below_minimum_units():
    assert V.risk_units(10.0, 1.1, 0.5, 0.01, E.X.Sizing(rel_volume=0.10, leverage=50.0)) is None
    assert V.risk_units(10_000.0, 1.1, float("nan"), 0.01, E.X.Sizing(rel_volume=0.10, leverage=50.0)) is None


def reverse_policy(after=3):
    state = {"opened": None, "done": False}

    def f(b, r, st, env):
        if state["done"]:
            return E.HOLD, 1.0, 1.0
        if int(st[0]) == 0 and state["opened"] is None and env.entry_gate(b)[0]:
            state["opened"] = b
            return E.LONG, 1.0, 1.0
        if int(st[0]) > 0 and state["opened"] is not None and b == state["opened"] + after:
            state["done"] = True
            return E.SHORT, 1.0, 1.0
        return E.HOLD, 1.0, 1.0
    return f


def test_reversal_is_one_entry_decision_under_the_cap_only_in_the_variant():
    bars, cal, sig, sr = synth()
    env0, m0 = run(bars, cal, sig, sr, CAP1, reverse_policy())
    assert m0["n_close_REVERSAL"] == 1 and m0["n_refused_DAILY_CAP"] >= 1 and m0["n_entries"] == 1
    cfg = dataclasses.replace(CAP1, variant=V.ExecVariant(reverse_counts_as_one_entry=True))
    env1, m1 = run(bars, cal, sig, sr, cfg, reverse_policy())
    assert m1["n_close_REVERSAL"] == 1 and m1["n_entries"] == 2 and m1["n_reversal_cap_waived"] == 1
    assert env1.trades[1]["direction"] == -1


def test_cap_still_enforced_for_a_new_entry_after_a_waived_reversal():
    bars, cal, sig, sr = synth()
    cfg = dataclasses.replace(CAP1, variant=V.ExecVariant(reverse_counts_as_one_entry=True))
    seq = {"n": 0}

    def f(b, r, st, env):
        ok, _ = env.entry_gate(b)
        if int(st[0]) == 0 and ok and seq["n"] == 0:
            seq["n"] = 1
            return E.LONG, 1.0, 1.0
        if int(st[0]) > 0 and seq["n"] == 1:
            seq["n"] = 2
            return E.SHORT, 1.0, 1.0           # reversal: waived
        if int(st[0]) < 0 and seq["n"] == 2:
            seq["n"] = 3
            return E.CLOSE, 1.0, 1.0
        if int(st[0]) == 0 and seq["n"] == 3:
            return E.LONG, 1.0, 1.0            # a NEW entry: refused (DAILY_CAP) until the next business day
        return E.HOLD, 1.0, 1.0
    env, m = run(bars, cal, sig, sr, cfg, f)
    assert m["n_reversal_cap_waived"] == 1 and m["n_refused_DAILY_CAP"] >= 1
    days = [env.business_day(t["entry_bar_abs"]) for t in env.trades]
    assert days[0] == days[1] and all(d != days[0] for d in days[2:])   # the reversal shared the slot; new entries waited


def test_variant_entry_cutoff_before_the_forced_flat():
    bars, cal, sig, sr = synth()
    seg = cal.segments[0]
    ff = seg.force_flat
    want = {"b": None}

    def at(hours_before):
        fill_ts = ff - pd.Timedelta(hours=hours_before)
        k = int(np.flatnonzero(bars.index.as_unit("ns").asi8 == fill_ts.value)[0])
        return k - 1

    def f(b, r, st, env):
        return (E.LONG, 1.0, 1.0) if (int(st[0]) == 0 and b == want["b"]) else (E.HOLD, 1.0, 1.0)
    cfg = dataclasses.replace(CAP1, variant=V.ExecVariant(entry_cutoff_hours_before_flat=24.0))
    want["b"] = at(23)
    _, m = run(bars, cal, sig, sr, cfg, f)
    assert m["n_entries"] == 0 and m["n_refused_VARIANT_CUTOFF"] == 1
    want["b"] = at(25)
    _, m = run(bars, cal, sig, sr, cfg, f)
    assert m["n_entries"] == 1 and m["n_refused_VARIANT_CUTOFF"] == 0
    want["b"] = at(23)
    _, m = run(bars, cal, sig, sr, CAP1, f)                         # legacy: the calendar cutoff alone allows it
    assert m["n_entries"] == 1


def test_calendar_clock_mask_adds_only_the_session_clock_and_leaves_the_engine_modes_unchanged():
    mask = V.calendar_clock_mask("none")
    kept = [n for n, m in zip(E.STATE_NAMES, mask) if m > 0]
    assert kept == ["hours_since_operational_start_days", "hours_to_forced_flat_days"]
    assert tuple(E.AGENT_STATE_MODES) == ("full", "minimal", "none")
    with pytest.raises(ValueError):
        V.calendar_clock_mask("calendar")


def test_short_stop_triggers_on_ask_only_in_the_variant():
    bars, cal, sig, sr = synth()
    bars = bars.copy()
    c = 1.1
    bars[["OPEN", "HIGH", "LOW", "CLOSE"]] = np.column_stack([np.full(len(bars), c)] * 4)
    bars["HIGH"] += 0.0001
    bars["LOW"] -= 0.0001
    seg = cal.segments[0]
    k0 = int(np.flatnonzero(bars.index.as_unit("ns").asi8 == seg.operational_start.value)[0])   # first fillable bar
    sl_dist = 1.0 * SIG                                                              # SL 1 sigma = 30 pips above
    # bar k0 + 3: bid high reaches SL - 1 pip (below the stop) -> legacy no hit; ask (= bid + spread) crosses
    bars.iloc[k0 + 3, bars.columns.get_loc("HIGH")] = c + sl_dist - 0.0001

    def f(b, r, st, env):
        return (E.SHORT, 3.0, 1.0) if (int(st[0]) == 0 and b == k0 - 1) else (E.HOLD, 3.0, 1.0)
    env0, m0 = run(bars, cal, sig, sr, CAP1, f)
    assert m0["n_entries"] == 1 and m0["n_close_SL"] == 0
    prof = CAP1.profile
    sp = prof.spread_at(int(bars.index.as_unit("ns").asi8[k0 + 3]), int(seg.market_open.value))
    cfg = dataclasses.replace(CAP1, variant=V.ExecVariant(short_trigger_on_ask=True))
    env1, m1 = run(bars, cal, sig, sr, cfg, f)
    if sp >= 0.0001:
        assert m1["n_close_SL"] == 1 and env1.trades[0]["exit_bar_abs"] == k0 + 3
    else:
        assert m1["n_close_SL"] == 0


def scored_policy(scores_by_offset):
    """Open LONG at every eligible flat bar with a score drawn from the bar's hour-of-day offset table."""
    def f(b, r, st, env):
        if int(st[0]) != 0:
            return E.HOLD, 1.0, 1.0, -np.inf
        hour = pd.Timestamp(int(env.v[b + 1]), tz="UTC").hour
        return E.LONG, 1.0, 1.0, float(scores_by_offset.get(hour, 0.0))
    return f


def test_best_of_day_gate_defers_to_the_running_best_without_look_ahead():
    bars, cal, sig, sr = synth(weeks=1)
    scores = {10: 0.2, 11: 0.9, 12: 0.5, 13: 0.95, 14: 0.1}        # fill hours (UTC) after the operational start 10:00
    env0, m0 = run(bars, cal, sig, sr, CAP1, V.first_of_day(scored_policy(scores)))
    first_hours = [t["entry_time"].hour for t in env0.trades]
    assert first_hours and first_hours[0] == 10                     # the audited rule: the first qualifying bar of the day
    gate = V.BestOfDayGate(scored_policy(scores), observe_bars=1)
    env1, m1 = run(bars, cal, sig, sr, CAP1, gate)
    best_hours = [t["entry_time"].hour for t in env1.trades]
    assert best_hours and best_hours[0] == 11                       # after observing 1 bar (0.2), 0.9 beats the best seen
    assert gate.stats["deferred_observing"] >= 1 and gate.stats["placed"] == len(env1.trades)
    gate2 = V.BestOfDayGate(scored_policy(scores), observe_bars=1, threshold=0.92)
    env2, _ = run(bars, cal, sig, sr, CAP1, gate2)
    assert env2.trades and env2.trades[0]["entry_time"].hour == 13 and gate2.stats["below_threshold"] > 0


def test_early_close_requirement():
    assert V.early_close_ok(0.10, 5.0) == (True, "OK")
    ok, why = V.early_close_ok(0.01, 5.0)
    assert not ok and why.startswith("EARLY_CLOSE_SHARE_BELOW_MIN")
    ok, why = V.early_close_ok(0.30, -1.0)
    assert not ok and why.startswith("EARLY_CLOSE_VALUE_LCB_NEGATIVE")


def test_effective_cap_and_clipping_match_the_engine():
    bars, cal, sig, sr = synth(weeks=1)
    env = E.BarEnv(bars, cal, sig, sr, bars.index[0], bars.index[-1] + pd.Timedelta(hours=1), CAP1)
    b = env.first + 30
    assert abs(V.effective_cap(env, b) - 2.0) < 1e-9
    tp, sl, clipped = V.clip_multiples(env, b, 3.0, 2.5)
    assert clipped and tp == 2.0 and sl == 2.0
    lv = env._levels(b, 1, 3.0, 2.5)
    assert lv is not None and abs((lv[0] - bars.CLOSE.iloc[b]) / SIG - tp) < 1e-6
    prof = V.cap_profile(env, np.arange(env.first, env.last))
    assert prof["n"] > 0 and abs(prof["p50"] - 2.0) < 1e-9

"""Trading-detail report (owner 2026-10-25: full trading details so a wrong strategy is caught before any comparison).

A real engine pass over synthetic bars with scripted decisions (longs and shorts, agent closes, TP / SL / forced-flat
exits): every correctness check must PASS on the engine's own records; tampering one record must make exactly the
matching check FAIL. Side statistics, exit shares, activity, streaks and excursions have known answers computed
independently from the raw records."""
import copy
import math

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import calendar as CAL
from app.short_horizon.expert_guided import env as E
from tools import expert_guided_trading_report as T

HOUR = pd.Timedelta(hours=1)
SIG = 0.0010


def world(seed=5):
    rng = np.random.default_rng(seed)
    stamps, t = [], pd.Timestamp("2020-01-05 21:00", tz="UTC")
    for _ in range(4):
        stamps += [t + k * HOUR for k in range(120)]
        t = stamps[-1] + 49 * HOUR
    n = len(stamps)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    open_ = np.r_[1.10, close[:-1]]
    high = np.maximum(open_, close) + 0.0002
    low = np.minimum(open_, close) - 0.0002
    bars = pd.DataFrame({"OPEN": open_, "HIGH": high, "LOW": low, "CLOSE": close}, index=pd.DatetimeIndex(stamps, name="DATE_TIME"))
    segs = CAL.derive_segments(bars.index, max_gap_hours=24)
    cal = CAL.load_calendar({"calendar_id": "TEST", "version": "t1", "segments": segs, "entry_delay_after_open_hours": 12,
                             "force_flat_before_close_hours": 12, "entry_cutoff_before_flat_hours": 0})
    cfg = E.EnvConfig(level_policy="clip", max_entries_per_day=1, sizing=E.X.Sizing(rel_volume=0.10, leverage=50.0))
    sigma, sr = np.full(n, SIG), np.full(n, 3 * SIG)
    mk = lambda: E.BarEnv(bars, cal, sigma, sr, bars.index[0], bars.index[-1] + HOUR, cfg)  # noqa: E731
    return mk, cfg


def run_scripted(mk):
    """Alternate long / short entries every 7 bars; close every third position by the agent after 3 bars."""
    env = mk()
    opened = {}
    for b in range(env.first, env.last):
        p = env.pos
        if p is None and (b - env.first) % 7 == 0:
            env.step(b, E.LONG if ((b - env.first) // 7) % 2 == 0 else E.SHORT, 1.5, 1.0)
            if env.pos is not None:
                opened[env.pos.entry_bar] = ((b - env.first) // 7) % 3 == 0
            continue
        if p is not None and opened.get(p.entry_bar) and b - p.entry_bar >= 2:
            env.step(b, E.CLOSE, 1.5, 1.0)
            continue
        env.step(b, E.HOLD, 1.5, 1.0)
    m = env.result()
    return env, m


@pytest.fixture(scope="module")
def replay():
    mk, cfg = world()
    env, m = run_scripted(mk)
    df = T.trade_frame(env.trades, mk())
    return mk, cfg, env, m, df


def test_engine_records_pass_every_correctness_check(replay):
    mk, cfg, env, m, df = replay
    reasons = set(df["close_reason"])
    assert {"AGENT_CLOSE"} <= reasons and ({"TP", "SL"} & reasons) and set(df["side"]) == {"BUY", "SELL"}
    cor = T.correctness(df, mk(), cfg, cfg.capital)
    bad = {k: v for k, v in cor.items() if v["status"] != "PASS"}
    assert not bad, bad
    assert all(v["checked"] == len(df) for v in cor.values())


@pytest.mark.parametrize("mutate, check", [
    (lambda d: d.__setitem__("stop_loss", float("nan")), "sl_and_tp_on_every_order"),
    (lambda d: d.__setitem__("entry_price", d["entry_price"] + 0.0001), "entry_fill_at_next_open"),
    (lambda d: d.__setitem__("units", d["units"] + 7), "units_follow_the_sizing_rule"),
    (lambda d: d.__setitem__("net_usd", d["net_usd"] + 1.0), "net_equals_gross_minus_costs"),
    (lambda d: d.__setitem__("rel_volume", 0.3), "rel_volume_in_0.10_0.20"),
    (lambda d: d.__setitem__("commission_usd", 0.0), "costs_positive_on_every_trade"),
])
def test_a_tampered_record_fails_exactly_its_check(replay, mutate, check):
    mk, cfg, env, m, df = replay
    d = df.copy()
    row = d.iloc[3].to_dict()
    mutate(row)
    for k, v in row.items():
        d.at[d.index[3], k] = v
    cor = T.correctness(d, mk(), cfg, cfg.capital)
    assert cor[check]["status"] == "FAIL" and cor[check]["failed"] >= 1


def test_overlap_daily_cap_and_protection_are_caught(replay):
    mk, cfg, env, m, df = replay
    d = df.copy()
    d.at[d.index[2], "entry_time"] = d.loc[d.index[1], "exit_time"] - HOUR             # starts before the previous exit
    cor = T.correctness(d, mk(), cfg, cfg.capital)
    assert cor["no_overlapping_positions"]["status"] == "FAIL"
    tp_rows = df.index[df["close_reason"] == "TP"]
    if len(tp_rows):
        d2 = df.copy()
        d2.at[tp_rows[0], "close_reason"] = "SL"                                         # recorded reason contradicts the bars
        assert T.correctness(d2, mk(), cfg, cfg.capital)["protection_rederived_matches_engine"]["status"] == "FAIL"
    cfg2 = E.EnvConfig(level_policy="clip", max_entries_per_day=0, sizing=cfg.sizing)
    assert T.correctness(df, mk(), cfg2, cfg.capital)["at_most_max_entries_per_day"]["status"] == "FAIL"
    cfg3 = E.EnvConfig(level_policy="clip", max_entries_per_day=1, sizing=E.X.Sizing(rel_volume=0.10, leverage=100.0))
    assert T.correctness(df, mk(), cfg3, cfg.capital)["leverage_at_most_1_50"]["status"] == "FAIL"


def test_side_exit_and_equity_statistics_have_their_known_answers(replay):
    mk, cfg, env, m, df = replay
    d = T.detail(df, mk(), m, cfg, cfg.capital, {})
    for side, sgn in (("BUY", 1), ("SELL", -1)):
        rec = [t for t in env.trades if t["direction"] == sgn]
        net = np.array([t["net_usd"] for t in rec])
        s = d["direction"][side]
        assert s["trades"] == len(rec)
        assert s["win_rate"] == pytest.approx((net > 0).mean())
        assert s["net_usd"] == pytest.approx(net.sum())
        if (net > 0).any() and (net <= 0).any():
            assert s["profit_factor"] == pytest.approx(net[net > 0].sum() / -net[net <= 0].sum())
    assert d["direction"]["buy_share"] + d["direction"]["sell_share"] == pytest.approx(1.0)
    shares = [d["exits"][r]["share"] for r in ("TP", "SL", "TIME", "AGENT_CLOSE", "REVERSAL")]
    assert sum(shares) == pytest.approx(1.0)
    assert d["exits"]["AGENT_CLOSE"]["count"] == sum(1 for t in env.trades if t["close_reason"] == "AGENT_CLOSE")
    assert d["equity"]["net_profit_usd"] == pytest.approx(m["net_profit_usd"])
    assert sum(d["equity"]["monthly_pnl_usd"].values()) == pytest.approx(m["net_profit_usd"], abs=1e-6)
    assert d["activity"]["orders"] == len(env.trades) == m["n_entries"]
    assert d["activity"]["orders_per_week_mean"] == pytest.approx(m["orders_per_week"])


def test_excursions_are_measured_over_the_bars_the_position_was_open(replay):
    mk, cfg, env, m, df = replay
    ev = mk()
    for _, t in df.iterrows():
        k, x = int(t.entry_bar), int(t.exit_bar)
        last = x if t.exit_time.value > ev.v[x] else x - 1
        last = max(last, k)
        hi, lo = ev.h[k:last + 1].max(), ev.l[k:last + 1].min()
        mae = (t.entry_price - lo) / SIG if t.direction > 0 else (hi - t.entry_price) / SIG
        assert t.mae_sigma == pytest.approx(mae) and t.mfe_sigma >= 0 - 1e-12


def test_small_helpers_have_known_answers():
    assert T.longest_losing_streak([1, -1, -2, 0, 3, -1]) == 3
    assert T.boot_mean_ci([2, 2, 2, 2]) == (2.0, 2.0)
    lo, hi = T.boot_mean_ci([0, 4] * 50)
    assert lo < 2.0 < hi
    assert T.correctness(pd.DataFrame(), None, None, 1.0)["net_equals_gross_minus_costs"]["status"] == "NOT_APPLICABLE_NO_TRADES"

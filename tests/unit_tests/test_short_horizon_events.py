"""Causal event-window strategy (lead design, 2026-10-19): fixture readers, clock table, surprise
scaling, family estimates, event blocks and the event-widened spread profile.

Written before app/short_horizon/events.py existed. The Investing.com fixture is
FIXTURE_RESEARCH_ONLY: tests on the real file are skipped when it is absent and copy no row.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import blocks as BL
from app.short_horizon import events as EV
from app.short_horizon import execution as X
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "feature-eng" / "tests" / "data"
CAL_FIXTURE = FIXTURE_DIR / "economic_calendar_2011_2021.csv"
PX_FIXTURE = FIXTURE_DIR / "eurusd_hour_2005_2020_ohlc.csv"

ROWS = (
    '2011/01/07,8:30:00,"United States   ","High Volatility Expected  ","Nonfarm Payrolls   ",'
    '"Worse Than Expected ","K  ",103,150.00,"vised From 39K>71 "\n'
    '2011/07/08,7:30:00,"United States   ","High Volatility Expected  ","Nonfarm Payrolls   ",'
    '"Worse Than Expected ","K  ",18,105.00,"vised From 54K>25 "\n'
    '2018/03/09,8:30:00,"United States   ","High Volatility Expected  ","Nonfarm Payrolls   ",'
    '"Better Than Expected","K  ",313,205.00,"239K"\n'
    '2011/01/13,7:45:00,"Euro Zone       ","High Volatility Expected  ","ECB Interest Rate Decision",'
    ',"%  ",1.00,1.00,1.00\n'
    '2011/01/13,9:00:00,"Japan           ","Low Volatility Expected   ","Something",,"%  ",1,1,1\n'
    '2025/02/07,8:30:00,"United States   ","High Volatility Expected  ","Nonfarm Payrolls   ",'
    ',"K  ",100,120.00,110\n'
)


def test_fixture_rows_convert_with_the_per_period_offset_table_not_a_dst_rule():
    df = EV.parse_calendar_fixture(io.StringIO(ROWS[: ROWS.rfind("2025")]))
    nfp = df[df.family == "Nonfarm Payrolls"].sort_values("release_utc")
    # 2011-01: fixed UTC-5 -> 13:30 UTC; 2011-07 shows 7:30 under the same fixed offset -> 12:30 UTC
    assert nfp.release_utc.iloc[0] == pd.Timestamp("2011-01-07 13:30", tz="UTC")
    assert nfp.release_utc.iloc[1] == pd.Timestamp("2011-07-08 12:30", tz="UTC")
    # 2018-03-09 lies in the exporter's UTC-4 window -> 12:30 UTC
    assert nfp.release_utc.iloc[2] == pd.Timestamp("2018-03-09 12:30", tz="UTC")
    assert float(nfp.actual.iloc[0]) == 103.0 and float(nfp.forecast.iloc[0]) == 150.0
    assert (df.currency[df.country == "United States"] == "USD").all()
    assert (df.currency[df.country == "Euro Zone"] == "EUR").all()
    assert df.impact.isin(["High", "Low"]).all()


def test_fixture_reader_refuses_test_rows():
    with pytest.raises(EV.TestRowsError):
        EV.parse_calendar_fixture(io.StringIO(ROWS))


def test_offset_table_matches_the_data_lane_switch_dates():
    assert EV.exporter_offset_hours(pd.Timestamp("2017-12-31")) == 5
    assert EV.exporter_offset_hours(pd.Timestamp("2018-02-28")) == 4
    assert EV.exporter_offset_hours(pd.Timestamp("2018-09-12")) == 5
    assert EV.exporter_offset_hours(pd.Timestamp("2019-02-01")) == 4
    assert EV.exporter_offset_hours(pd.Timestamp("2019-11-01")) == 5
    assert EV.exporter_offset_hours(pd.Timestamp("2020-02-03")) == 4
    assert EV.exporter_offset_hours(pd.Timestamp("2020-09-01")) == 5
    assert EV.exporter_offset_hours(pd.Timestamp("2021-02-03")) == 4


@pytest.mark.skipif(not CAL_FIXTURE.exists(), reason="research-only fixture absent")
def test_real_fixture_nfp_rows_land_on_8_30_new_york_after_conversion():
    df = EV.parse_calendar_fixture(CAL_FIXTURE)
    nfp = df[(df.family == "Nonfarm Payrolls") & (df.currency == "USD")]
    assert len(nfp) > 100
    local = nfp.release_utc.dt.tz_convert("America/New_York")
    share = float(((local.dt.hour == 8) & (local.dt.minute == 30)).mean())
    assert share >= 0.95, share
    first = nfp.sort_values("release_utc").iloc[0]
    assert first.release_utc == pd.Timestamp("2011-01-07 13:30", tz="UTC")
    assert float(first.actual) == 103.0 and float(first.forecast) == 150.0


def test_surprise_scale_uses_prior_releases_of_the_family_only():
    t = pd.date_range("2011-01-01", periods=12, freq="30D", tz="UTC")
    df = pd.DataFrame({"release_utc": t, "family": ["A"] * 12, "currency": "USD",
                       "actual": np.arange(12, dtype=float), "forecast": 0.0})
    df.loc[11, "actual"] = 1000.0  # a huge last surprise must not change earlier scales
    out = EV.surprise(df, min_prior=4)
    assert out.surprise.iloc[:4].isna().all()
    assert np.isfinite(out.surprise.iloc[4:]).all()
    ref = EV.surprise(df.iloc[:11].copy(), min_prior=4)
    np.testing.assert_allclose(out.scale.iloc[:11].to_numpy(), ref.scale.to_numpy())


def test_clock_offset_detected_from_the_nfp_jump_bar():
    bars = synthetic_bars(seed=3, segments=3, hours=120, start="2011-01-02 21:00")
    # label the bars two hours ahead of UTC: the jump sits at label 15 when the release is 13:30 UTC
    rel = pd.DatetimeIndex([pd.Timestamp("2011-01-07 13:30", tz="UTC"), pd.Timestamp("2011-01-14 13:30", tz="UTC")])
    lab = bars.copy()
    lab.index = bars.index.tz_localize(None) + 2 * HOUR
    for r in rel:
        j = lab.index.get_loc((r.tz_localize(None) + 2 * HOUR).floor("h"))
        lab.iloc[j, lab.columns.get_loc("CLOSE")] += 0.02
        lab.iloc[j, lab.columns.get_loc("HIGH")] += 0.02
        lab.iloc[j + 1, lab.columns.get_loc("OPEN")] += 0.02
    info = EV.detect_clock_offset(lab, rel)
    assert info["offset_hours"] == 2 and info["status"] == "CLOCK_DETECTED_NOT_VERIFIED"
    utc = EV.bars_to_utc(lab, info["offset_hours"])
    assert utc.index[0] == bars.index[0]


def _events_on(bars, n=6, family="A"):
    rel = [bars.index[20 + 24 * i] + pd.Timedelta(minutes=30) for i in range(n)]
    return pd.DataFrame({"release_utc": rel, "family": family, "currency": "USD",
                         "actual": 1.0, "forecast": 0.0, "surprise": [1.0, -1.0, 2.0, -2.0, 0.5, -0.5][:n]})


def test_event_windows_are_causal_and_measure_post_release_bars():
    bars = synthetic_bars(seed=5, segments=2, hours=240, start="2011-01-02 21:00")
    ev = _events_on(bars)
    w = EV.event_windows(bars, ev, max_window=6)
    assert len(w) == len(ev)
    k = int(w.release_bar.iloc[0])
    assert bars.index[k] <= ev.release_utc.iloc[0] < bars.index[k] + HOUR
    assert w.entry_bar.iloc[0] == k + 1
    o = bars.OPEN.to_numpy()
    c = bars.CLOSE.to_numpy()
    np.testing.assert_allclose(w["drift_6"].iloc[0], c[k + 6] - o[k + 1])
    assert w.realized_vol_6.iloc[0] > 0 and 1 <= w.half_life.iloc[0] <= 6
    # changing bars before the release changes nothing after the release bar labels
    later = bars.copy()
    later.iloc[: k - 1, :] += 0.01
    w2 = EV.event_windows(later, ev, max_window=6)
    np.testing.assert_allclose(w.drift_6.to_numpy(), w2.drift_6.to_numpy())


def test_family_estimates_fit_on_the_decision_period_only_and_gate_by_significance():
    rng = np.random.default_rng(0)
    n = 80
    t = pd.date_range("2011-01-10", periods=n, freq="7D", tz="UTC")
    s = rng.standard_normal(n)
    w = pd.DataFrame({"release_utc": t, "family": ["A"] * 40 + ["B"] * 40, "surprise": s,
                      "drift_4": np.r_[0.0010 * s[:40] + rng.normal(0, 0.0002, 40), rng.normal(0, 0.0005, 40)],
                      "half_life": 2.0})
    est = EV.fit_families(w, window=4, decision_end=pd.Timestamp("2016-01-01", tz="UTC"), min_n=20, alpha=0.05)
    assert est.loc["A", "sign"] == 1 and est.loc["A", "tradable"]
    assert not est.loc["B", "tradable"]
    # rows at or after decision_end are ignored
    w_late = w.copy()
    w_late.loc[w_late.family == "B", "release_utc"] = pd.Timestamp("2016-06-01", tz="UTC")
    est2 = EV.fit_families(w_late, window=4, decision_end=pd.Timestamp("2016-01-01", tz="UTC"), min_n=20)
    assert "B" not in est2.index


def test_event_blocks_and_orders_carry_units_sl_tp_and_never_pass_the_forced_flat():
    bars = synthetic_bars(seed=7, segments=2, hours=120, start="2011-01-02 21:00")
    cal = calendar_for(bars, delay=1, flat=1, cutoff=1)
    ev = _events_on(bars, n=3)
    ev.loc[2, "release_utc"] = bars.index[118] + pd.Timedelta(minutes=30)   # near the segment close
    w = EV.event_windows(bars, ev, max_window=6)
    blocks = EV.event_blocks(bars, cal, w, time_stop=np.array([4, 4, 4]))
    assert isinstance(blocks, BL.Blocks) and len(blocks) == 3
    assert blocks.bar_lo[0] == w.entry_bar.iloc[0] and blocks.bar_hi[0] == w.entry_bar.iloc[0] + 4
    assert blocks.status[2] != BL.OK or blocks.end[2] <= cal.segments[0].force_flat
    sigma = np.full(3, 0.004)
    orders = EV.directional_orders(blocks, direction=np.array([1, -1, 1]), tp_distance=np.array([0.002, 0.002, 0.002]),
                                   sl_distance=np.array([0.003, 0.003, 0.003]), size=np.array([1.0, 0.5, 1.0]))
    o = orders[0]
    assert o.direction == 1 and o.take_profit > blocks.ref[0] > o.stop_loss and o.percentual_order_size == 1.0
    X.check_order(o, float(blocks.ref[0]), float(sigma[0]), X.OrderRules())
    with pytest.raises(X.OrderRuleError):
        X.check_order(EV.directional_orders(blocks, np.array([1]), np.array([0.00001]), np.array([0.003]),
                                            np.array([1.0]))[0], float(blocks.ref[0]), float(sigma[0]), X.OrderRules())
    res = X.run_backtest(bars, blocks, orders, X.RealisticProfile(), cal, 10_000.0,
                         (bars.index[0], bars.index[-1] + HOUR), profile=EV.EventProfile(),
                         order_rules=X.OrderRules(), sigma_block=sigma, sizing=X.Sizing())
    assert res.metrics["trades"] >= 1
    for t in res.trades:
        assert t["units"] >= 1 and np.isfinite(t["stop_loss"]) and np.isfinite(t["take_profit"])
        assert t["entry_bar_abs"] in set(blocks.bar_lo)


def test_event_profile_widens_the_spread_in_the_release_hours_only():
    base = X.RealisticProfile()
    rel = pd.Timestamp("2011-01-07 13:30", tz="UTC")
    prof = EV.EventProfile(event_spread=8e-4, release_bars=EV.release_bar_labels([rel]))
    open_ns = pd.Timestamp("2011-01-02 21:00", tz="UTC").value
    assert prof.entry_delay_bars == 0
    assert prof.spread_at(pd.Timestamp("2011-01-07 13:00", tz="UTC").value, open_ns) == 8e-4
    assert prof.spread_at(pd.Timestamp("2011-01-07 14:00", tz="UTC").value, open_ns) == 8e-4
    t = pd.Timestamp("2011-01-07 15:00", tz="UTC").value
    assert prof.spread_at(t, open_ns) == base.spread_at(t, open_ns)
    assert prof.scaled(2.0).event_spread == 16e-4


def test_straddle_orders_are_oco_with_both_legs_checked():
    bars = synthetic_bars(seed=9, segments=1, hours=120, start="2011-01-02 21:00")
    cal = calendar_for(bars, delay=1, flat=1, cutoff=1)
    ev = _events_on(bars, n=2)
    w = EV.event_windows(bars, ev, max_window=6)
    blocks = EV.straddle_blocks(bars, cal, w, time_stop=np.array([4, 4]))
    assert blocks.bar_lo[0] == w.release_bar.iloc[0]
    orders = EV.straddle_orders(blocks, entry_distance=np.array([0.001, 0.001]), tp_distance=np.array([0.002, 0.002]),
                                sl_distance=np.array([0.003, 0.003]), size=np.array([1.0, 1.0]))
    assert hasattr(orders[0], "legs") and len(orders[0].legs) == 2
    from app.short_horizon.vstrategies import check_oco
    check_oco(orders[0], 0.004, X.OrderRules())

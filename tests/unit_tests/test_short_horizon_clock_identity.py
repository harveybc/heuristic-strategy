"""Clock identity for the EURUSD H1 lake labels (plan items 14c/16).

Written before ``app/short_horizon/clock_identity.py`` and ``bar_source.py`` existed. Every frame
here is synthetic; no lake or admitted row is read by this suite.

Rules under test (declared tz-database rules, never a hand-applied hour shift):
  AMERICA_NEW_YORK_WALL  label = America/New_York wall time of the bar start
  EU_DST_UTC_MINUS_5     label = Europe/London wall time of the bar start minus 5 h
  FIXED_UTC_MINUS_5      label = UTC minus 5 h all year (Etc/GMT+5)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import bar_source as BS
from app.short_horizon import clock_identity as CI
from app.short_horizon import data as D

NY = "AMERICA_NEW_YORK_WALL"
EU = "EU_DST_UTC_MINUS_5"
EST = "FIXED_UTC_MINUS_5"
HOUR = pd.Timedelta(hours=1)


def L(*stamps):
    return pd.DatetimeIndex([pd.Timestamp(s) for s in stamps])


def U(*stamps):
    return pd.DatetimeIndex([pd.Timestamp(s, tz="UTC") for s in stamps])


# ----------------------------------------------------------------------------- the rules themselves
def test_rules_are_tz_database_rules_not_offsets():
    for rid in (NY, EU, EST):
        r = CI.rule(rid)
        assert r.tz in ("America/New_York", "Europe/London", "Etc/GMT+5")
        assert isinstance(r.label_minus_wall_hours, int)
    with pytest.raises(CI.ClockError):
        CI.rule("PLUS_5_HOURS")


def test_winter_and_summer_offsets():
    # deep winter: every rule is UTC-5
    for rid in (NY, EU, EST):
        utc, _ = CI.label_to_utc(L("2024-01-10 12:00"), rid)
        assert utc[0] == pd.Timestamp("2024-01-10 17:00", tz="UTC")
    # deep summer: NY and EU are UTC-4, fixed EST stays UTC-5
    for rid, h in ((NY, 16), (EU, 16), (EST, 17)):
        utc, _ = CI.label_to_utc(L("2024-07-10 12:00"), rid)
        assert utc[0] == pd.Timestamp(f"2024-07-10 {h}:00", tz="UTC")


def test_gap_weeks_separate_ny_from_eu():
    # 2024-03-10 .. 2024-03-31: US already on DST, EU not yet -> the rules disagree by one hour
    lab = L("2024-03-13 12:00")
    assert CI.label_to_utc(lab, NY)[0][0] == pd.Timestamp("2024-03-13 16:00", tz="UTC")
    assert CI.label_to_utc(lab, EU)[0][0] == pd.Timestamp("2024-03-13 17:00", tz="UTC")
    # 2024-10-27 .. 2024-11-03: EU back on standard time, US still on DST
    lab = L("2024-10-30 12:00")
    assert CI.label_to_utc(lab, NY)[0][0] == pd.Timestamp("2024-10-30 16:00", tz="UTC")
    assert CI.label_to_utc(lab, EU)[0][0] == pd.Timestamp("2024-10-30 17:00", tz="UTC")


# ----------------------------------------------------------------------------- DST transition weeks
def test_ny_spring_forward_week():
    labels = L("2024-03-10 01:00", "2024-03-10 03:00", "2024-03-11 12:00")
    utc, status = CI.label_to_utc(labels, NY)
    assert list(utc) == list(U("2024-03-10 06:00", "2024-03-10 07:00", "2024-03-11 16:00"))
    assert list(status) == ["OK"] * 3
    with pytest.raises(CI.ClockError, match="NONEXISTENT"):
        CI.label_to_utc(L("2024-03-10 02:00"), NY)
    utc, status = CI.label_to_utc(L("2024-03-10 02:00"), NY, nonexistent="flag")
    assert pd.isna(utc[0]) and status[0] == "NONEXISTENT"


def test_ny_fall_back_week():
    labels = L("2024-11-03 00:00", "2024-11-03 02:00", "2024-11-04 12:00")
    utc, _ = CI.label_to_utc(labels, NY)
    assert list(utc) == list(U("2024-11-03 04:00", "2024-11-03 07:00", "2024-11-04 17:00"))
    with pytest.raises(CI.ClockError, match="AMBIGUOUS"):
        CI.label_to_utc(L("2024-11-03 01:00"), NY)
    early, _ = CI.label_to_utc(L("2024-11-03 01:00"), NY, ambiguous="earlier")
    late, _ = CI.label_to_utc(L("2024-11-03 01:00"), NY, ambiguous="later")
    assert early[0] == pd.Timestamp("2024-11-03 05:00", tz="UTC")
    assert late[0] == pd.Timestamp("2024-11-03 06:00", tz="UTC")
    flagged, status = CI.label_to_utc(L("2024-11-03 01:00"), NY, ambiguous="flag")
    assert pd.isna(flagged[0]) and status[0] == "AMBIGUOUS"


def test_eu_spring_forward_and_fall_back_fall_on_saturday_evening_labels():
    # London 01:00 BST gap (2024-03-31) -> label Saturday 20:00 does not exist
    with pytest.raises(CI.ClockError, match="NONEXISTENT"):
        CI.label_to_utc(L("2024-03-30 20:00"), EU)
    utc, _ = CI.label_to_utc(L("2024-03-30 19:00", "2024-03-30 21:00"), EU)
    assert list(utc) == list(U("2024-03-31 00:00", "2024-03-31 01:00"))
    # London 01:00 repeated (2024-10-27) -> label Saturday 20:00 is ambiguous
    with pytest.raises(CI.ClockError, match="AMBIGUOUS"):
        CI.label_to_utc(L("2024-10-26 20:00"), EU)
    utc, _ = CI.label_to_utc(L("2024-10-26 19:00", "2024-10-26 21:00"), EU)
    assert list(utc) == list(U("2024-10-26 23:00", "2024-10-27 02:00"))


def test_transitions_fall_inside_the_weekend_closure():
    # every DST transition of NY and EU 2019-2025 lies between Friday 22:00 UTC and Sunday 21:00 UTC
    for rid in (NY, EU):
        tr = CI.transitions(rid, 2019, 2025)
        assert len(tr) == 14
        for t in tr:
            u = pd.Timestamp(t["utc"])
            closed = (u.dayofweek == 5) or (u.dayofweek == 6 and u.hour < 21) or (u.dayofweek == 4 and u.hour >= 22)
            assert closed, t
    assert CI.transitions(EST, 2019, 2025) == []


def test_pinned_transitions_match_the_tz_database():
    for rid in (NY, EU):
        assert CI.transitions(rid, 2019, 2025) == CI.PINNED_TRANSITIONS[rid]


# ----------------------------------------------------------------------------- weekly open / close
def test_weekly_open_and_close_ny_17h():
    # The FX week opens Sunday 17:00 New York and closes Friday 17:00 New York (last bar starts 16:00).
    winter_open, summer_open, gap_open = "2024-01-07 17:00", "2024-06-09 17:00", "2024-03-17 17:00"
    winter = CI.label_to_utc(L(winter_open, "2024-01-12 16:00"), NY)[0]
    summer = CI.label_to_utc(L(summer_open, "2024-06-14 16:00"), NY)[0]
    assert list(winter) == list(U("2024-01-07 22:00", "2024-01-12 21:00"))
    assert list(summer) == list(U("2024-06-09 21:00", "2024-06-14 20:00"))
    # In a gap week the EU rule needs a 16:00 label to reach the 21:00 UTC open.
    eu = CI.label_to_utc(L("2024-03-17 16:00"), EU)[0]
    assert eu[0] == pd.Timestamp("2024-03-17 21:00", tz="UTC")
    assert CI.label_to_utc(L(gap_open), NY)[0][0] == pd.Timestamp("2024-03-17 21:00", tz="UTC")


# ----------------------------------------------------------------------------- round trip
@pytest.mark.parametrize("rid", [NY, EU, EST])
def test_round_trip_every_hour_2019_2025(rid):
    utc = pd.date_range("2019-01-01", "2026-01-01", freq="h", tz="UTC", inclusive="left")
    labels, fold = CI.utc_to_label(utc, rid)
    back = CI.label_to_utc_with_fold(labels, fold, rid)
    assert back.equals(utc)
    # labels that are unique map back without any fold
    ok = ~pd.Index(labels).duplicated(keep=False)
    plain, status = CI.label_to_utc(labels[ok], rid)
    assert plain.equals(utc[ok]) and (status == "OK").all()


# ----------------------------------------------------------------------------- identity record
def test_identity_record_is_declared_and_digested():
    labels = pd.date_range("2024-01-07 17:00", periods=48, freq="h")
    utc, _ = CI.label_to_utc(labels, EU)
    rec = CI.identity_record(EU, source_sha256="a" * 64, labels=labels, utc=utc,
                             window=("2024-01-01", "2024-02-01"))
    for k in ("rule", "tzdata", "bar_convention_in", "bar_convention_out", "source", "mapping", "transitions",
              "transition_policy", "identity_sha256"):
        assert k in rec
    assert rec["rule"]["tz"] == "Europe/London" and rec["bar_convention_out"] == "UTC_BAR_START"
    assert rec["tzdata"]["version"]
    assert rec["mapping"]["digest_sha256"] == CI.mapping_digest(labels, utc)
    body = {k: v for k, v in rec.items() if k != "identity_sha256"}
    assert rec["identity_sha256"] == CI.canonical_sha256(body)
    json.dumps(rec)  # serialisable


def test_perturbation_changes_digests():
    labels = pd.date_range("2024-01-07 17:00", periods=48, freq="h")
    utc, _ = CI.label_to_utc(labels, EU)
    base = CI.identity_record(EU, "a" * 64, labels, utc, ("2024-01-01", "2024-02-01"))
    # one moved label -> different mapping digest and identity
    moved = pd.DatetimeIndex(list(labels[:10]) + [labels[10] + 100 * HOUR] + list(labels[11:]))
    utc2, _ = CI.label_to_utc(moved, EU)
    p1 = CI.identity_record(EU, "a" * 64, moved, utc2, ("2024-01-01", "2024-02-01"))
    assert p1["mapping"]["digest_sha256"] != base["mapping"]["digest_sha256"]
    assert p1["identity_sha256"] != base["identity_sha256"]
    # one-hour hand shift of the utc side is detected by the digest
    assert CI.mapping_digest(labels, utc + HOUR) != base["mapping"]["digest_sha256"]
    # a different source byte identity changes the identity, not the mapping
    p2 = CI.identity_record(EU, "b" * 64, labels, utc, ("2024-01-01", "2024-02-01"))
    assert p2["mapping"] == base["mapping"] and p2["identity_sha256"] != base["identity_sha256"]
    # a different rule changes the identity
    utc3, _ = CI.label_to_utc(labels, NY)
    p3 = CI.identity_record(NY, "a" * 64, labels, utc3, ("2024-01-01", "2024-02-01"))
    assert p3["identity_sha256"] != base["identity_sha256"]


# ----------------------------------------------------------------------------- synthetic sources
def _admitted_frame(utc_index, seed=0):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 1e-4, len(utc_index)))
    openp = np.r_[close[0], close[:-1]]
    high = np.maximum(openp, close) + 5e-5
    low = np.minimum(openp, close) - 5e-5
    return pd.DataFrame({"timestamp": utc_index, "available_at": utc_index + HOUR, "BidOpen": openp,
                         "BidHigh": high, "BidLow": low, "BidClose": close,
                         "source_id": "independent-hf:eurusd:x"})


def _week_hours(open_utc, hours=120):
    return pd.date_range(open_utc, periods=hours, freq="h", tz="UTC")


def _write_sources(tmp_path, rid=EU, shift_hours=0, drop_lake=(), seed=0):
    """Admitted root in UTC bar start; lake = same bars labelled by ``rid`` (+ optional clock defect)."""
    utc = _week_hours("2024-03-10 21:00").append(_week_hours("2024-07-07 21:00")).append(_week_hours("2024-11-03 22:00"))
    adm = _admitted_frame(utc, seed)
    labels, _ = CI.utc_to_label(utc, rid)
    labels = labels + pd.Timedelta(hours=shift_hours)
    lake = pd.DataFrame({"datetime": labels.tz_localize("UTC"), "open": adm.BidOpen.to_numpy(),
                         "high": adm.BidHigh.to_numpy(), "low": adm.BidLow.to_numpy(), "close": adm.BidClose.to_numpy()})
    if drop_lake:
        lake = lake.drop(index=list(drop_lake)).reset_index(drop=True)
    lp, ap = tmp_path / "1h.parquet", tmp_path / "eurusd.parquet"
    lake.to_parquet(lp, index=False)
    adm.to_parquet(ap, index=False)
    return lp, ap, utc


def test_map_lake_filters_by_mapped_time_and_never_reads_test(tmp_path):
    lp, ap, utc = _write_sources(tmp_path)
    m = CI.map_lake(lp, "2024-01-01", "2025-01-01", EU)
    assert m.index.name == "UTC_BAR_START" and m.index.tz is not None
    assert m.index.equals(utc)
    with pytest.raises(D.TestRowsError):
        CI.map_lake(lp, "2024-01-01", "2025-01-02", EU)
    # the label read bound is the label of the UTC end (2024-12-31 19:00 in winter)
    assert CI.label_bounds(EU, "2024-01-01", "2025-01-01")[1] == pd.Timestamp("2024-12-31 19:00")


def test_witness_exact_under_the_true_rule_and_refutes_others(tmp_path):
    lp, ap, utc = _write_sources(tmp_path, rid=EU)
    w = CI.witness(lp, ap, "2024-01-01", "2025-01-01", rules=[EU, NY, EST])
    eu, ny = w["rules"][EU], w["rules"][NY]
    assert eu["verdict"] == "EXACT_CLOCK_MAPPING"
    assert eu["totals"]["joined"] == len(utc) and eu["totals"]["ohlc_exact"] == len(utc)
    assert eu["totals"]["residual_clock"] == 0
    # the March gap week separates NY from EU: those rows land one hour off and are classified CLOCK
    assert ny["verdict"] == "CLOCK_RESIDUALS_PRESENT" and ny["totals"]["residual_clock"] > 0
    assert ny["us_eu_gap_periods"]["exact_in_us_eu_gap"] == 0 < eu["us_eu_gap_periods"]["exact_in_us_eu_gap"]
    assert w["rules"][EST]["verdict"] == "CLOCK_RESIDUALS_PRESENT"
    assert w["selected_rule"] == EU


def test_witness_flags_a_one_hour_clock_defect_as_clock_not_source(tmp_path):
    lp, ap, utc = _write_sources(tmp_path, rid=EU, shift_hours=1)
    w = CI.witness(lp, ap, "2024-01-01", "2025-01-01", rules=[EU])
    t = w["rules"][EU]["totals"]
    assert w["rules"][EU]["verdict"] != "EXACT_CLOCK_MAPPING"
    assert t["residual_clock"] + t["lake_only_clock"] > 0.9 * len(utc)


def test_witness_classifies_a_price_change_as_source(tmp_path):
    lp, ap, utc = _write_sources(tmp_path, rid=EU)
    lake = pd.read_parquet(lp)
    lake.loc[5, "high"] += 3e-4  # one vendor range differs, clock untouched
    lake.to_parquet(lp, index=False)
    w = CI.witness(lp, ap, "2024-01-01", "2025-01-01", rules=[EU])
    t = w["rules"][EU]["totals"]
    assert t["residual_source"] == 1 and t["residual_clock"] == 0
    assert w["rules"][EU]["verdict"] == "EXACT_CLOCK_MAPPING"  # the clock is exact; the vendor differs


def test_witness_counts_rows_in_one_source_only(tmp_path):
    lp, ap, utc = _write_sources(tmp_path, rid=EU, drop_lake=range(0, 30))
    w = CI.witness(lp, ap, "2024-01-01", "2025-01-01", rules=[EU])
    t = w["rules"][EU]["totals"]
    assert t["admitted_only"] == 30 and t["lake_only"] == 0
    assert w["coverage_by_half"]["admitted"]["2024H1"] - w["coverage_by_half"]["lake_mapped"][EU]["2024H1"] == 30


def test_witness_report_has_no_raw_rows(tmp_path):
    lp, ap, _ = _write_sources(tmp_path, rid=EU)
    w = CI.witness(lp, ap, "2024-01-01", "2025-01-01", rules=[EU])
    text = json.dumps(w)
    assert "1.10" not in text  # no prices
    assert "BidClose" not in text


# ----------------------------------------------------------------------------- joined view + adapter
def test_joined_view_is_keyed_by_asset_time_convention_source(tmp_path):
    lp, ap, utc = _write_sources(tmp_path, rid=EU)
    view = CI.joined_view(lp, ap, "2024-01-01", "2025-01-01", EU, asset="EURUSD",
                          lake_sha256="l" * 64, admitted_sha256="a" * 64)
    assert list(view.columns[:4]) == ["asset", "utc_bar_start", "bar_convention", "source"]
    assert not view.duplicated(["asset", "utc_bar_start", "bar_convention", "source"]).any()
    assert set(view.bar_convention) == {"UTC_BAR_START"}
    assert set(view.source) == {"admitted:" + "a" * 64, f"lake_mapped:{EU}:" + "l" * 64}
    wide = CI.comparison_view(view)
    assert wide["ohlc_equal"].all()


def test_bar_source_default_is_byte_identical_to_load_ohlc(tmp_path):
    lp, ap, utc = _write_sources(tmp_path, rid=EU)
    old = D.load_ohlc(lp, "2024-01-01", "2025-01-01")
    new = BS.load_ohlc(lp, "2024-01-01", "2025-01-01")
    assert new.equals(old)
    assert new.index.equals(old.index) and new.index.name == old.index.name and str(new.index.dtype) == str(old.index.dtype)
    assert list(new.dtypes) == list(old.dtypes)
    assert pd.util.hash_pandas_object(new, index=True).equals(pd.util.hash_pandas_object(old, index=True))
    cfg = {"default_source": "lake_label", "sources": {"lake_label": {"path": str(lp)}}}
    assert BS.load_ohlc_from_config(cfg, "2024-01-01", "2025-01-01").equals(old)


def test_bar_source_admitted_and_lake_mapped_agree_on_utc(tmp_path):
    lp, ap, utc = _write_sources(tmp_path, rid=EU)
    a = BS.load_ohlc(ap, "2024-01-01", "2025-01-01", source="admitted")
    m = BS.load_ohlc(lp, "2024-01-01", "2025-01-01", source="lake_mapped", clock_rule=EU)
    assert list(a.columns) == ["OPEN", "HIGH", "LOW", "CLOSE"] == list(m.columns)
    assert a.index.name == m.index.name == "DATE_TIME"
    pd.testing.assert_frame_equal(a, m)
    with pytest.raises(D.TestRowsError):
        BS.load_ohlc(ap, "2024-01-01", "2025-02-01", source="admitted")
    with pytest.raises(ValueError):
        BS.load_ohlc(ap, "2024-01-01", "2025-01-01", source="vendor_guess")


def test_bar_source_config_resolves_roots_and_checks_pins(tmp_path):
    lp, ap, utc = _write_sources(tmp_path, rid=EU)
    cfg = {"default_source": "lake_label",
           "sources": {"lake_label": {"path": "<FINANCIAL_DATA_ROOT>/1h.parquet"},
                       "lake_mapped": {"path": "<FINANCIAL_DATA_ROOT>/1h.parquet", "clock_rule": EU,
                                       "sha256": D.file_sha256(lp)},
                       "admitted": {"path": "<ADMITTED_ROOT>/eurusd.parquet", "sha256": D.file_sha256(ap)}}}
    roots = {"FINANCIAL_DATA_ROOT": str(tmp_path), "ADMITTED_ROOT": str(tmp_path)}
    a = BS.load_ohlc_from_config(cfg, "2024-01-01", "2025-01-01", source="admitted", roots=roots)
    m = BS.load_ohlc_from_config(cfg, "2024-01-01", "2025-01-01", source="lake_mapped", roots=roots)
    pd.testing.assert_frame_equal(a, m)
    cfg["sources"]["admitted"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="DIGEST"):
        BS.load_ohlc_from_config(cfg, "2024-01-01", "2025-01-01", source="admitted", roots=roots)


def test_shipped_config_declares_the_view_and_keeps_the_legacy_default():
    from pathlib import Path
    cfg = json.loads((Path(__file__).resolve().parents[2] / "config/short_horizon/eurusd_h1_bar_sources.json").read_text())
    assert cfg["default_source"] == "lake_label"
    assert set(cfg["sources"]) == {"lake_label", "lake_mapped", "admitted"}
    assert cfg["joined_view"]["key"] == ["asset", "utc_bar_start", "bar_convention", "source"]
    assert cfg["sources"]["lake_mapped"]["clock_rule"] in CI.RULES
    assert cfg["sources"]["admitted"]["sha256"].startswith("c80847bb")
    assert cfg["sources"]["lake_label"]["sha256"].startswith("c6628375")


def test_committed_identity_is_pinned_by_the_config_and_self_consistent():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / "config/short_horizon/eurusd_h1_bar_sources.json").read_text())
    spec = cfg["sources"]["lake_mapped"]
    rec = json.loads((root / spec["clock_identity"]).read_text())
    body = {k: v for k, v in rec.items() if k != "identity_sha256"}
    assert CI.canonical_sha256(body) == rec["identity_sha256"] == spec["clock_identity_sha256"]
    assert rec["rule"]["id"] == spec["clock_rule"] and rec["source"]["sha256"] == spec["sha256"]
    assert rec["transitions"]["matches_pinned"] and rec["transitions"]["instants"] == CI.PINNED_TRANSITIONS[rec["rule"]["id"]]
    assert rec["witness"]["verdict"] == "EXACT_CLOCK_MAPPING"
    assert rec["bar_convention_out"] == "UTC_BAR_START" and rec["transition_policy"] == spec["transition_policy"]
    w = json.loads((root / "docs/short_horizon/results/clock_identity/WITNESS.json").read_text())
    assert w["witness_sha256"] == rec["witness"]["sha256"]
    assert CI.canonical_sha256({k: v for k, v in w.items() if k != "witness_sha256"}) == w["witness_sha256"]
    assert w["window_utc"][1] <= "2025-01-01T00:00:00+00:00"


def test_verify_identity_recomputes_the_mapping(tmp_path):
    lp, ap, utc = _write_sources(tmp_path, rid=EU)
    m = CI.map_lake(lp, "2024-01-01", "2025-01-01", EU)
    rec = CI.identity_record(EU, D.file_sha256(lp), m["LABEL"], m.index, ("2024-01-01", "2025-01-01"))
    assert CI.verify_identity(rec, lp)
    bad = dict(rec, mapping=dict(rec["mapping"], digest_sha256="0" * 64))
    bad["identity_sha256"] = CI.canonical_sha256({k: v for k, v in bad.items() if k != "identity_sha256"})
    with pytest.raises(CI.ClockError, match="MAPPING_DIGEST"):
        CI.verify_identity(bad, lp)

"""Forward harness: calendar, fill-after-receipt rule, barrier outcome, probability parsing, idempotent decide."""
import json

import numpy as np
import pandas as pd

from tools import forward_llm_harness as F


def test_week_bounds_and_entry_window():
    o, c = F.week_bounds(pd.Timestamp("2026-10-21 12:00", tz="UTC"))                 # Wednesday
    assert str(o) == "2026-10-18 21:00:00+00:00" and str(c) == "2026-10-23 21:00:00+00:00"   # 17:00 EDT
    assert F.entry_allowed(pd.Timestamp("2026-10-21 12:00", tz="UTC"))[0]
    assert not F.entry_allowed(pd.Timestamp("2026-10-19 02:00", tz="UTC"))[0]        # first 12 h of the week
    assert not F.entry_allowed(pd.Timestamp("2026-10-23 12:00", tz="UTC"))[0]        # last 12 h
    assert not F.entry_allowed(pd.Timestamp("2026-10-24 12:00", tz="UTC"))[0]        # Saturday


def test_fill_is_never_an_open_already_passed_and_outcomes():
    s = np.arange(10) * 3600.0 + 1_800_000_000
    bars = pd.DataFrame({"bar_start_epoch": s, "open": 1.0, "high": 1.001, "low": 0.999, "close": 1.0})
    bars.loc[5, "high"] = 1.02
    receipt = s[2] + 1800                                                              # mid bar 2 -> fill at bar 3
    st, oc, fill, used = F.outcome(bars, receipt, 0.005, pd.Timestamp(s[-1] + 10 * 3600, unit="s", tz="UTC"))
    assert fill == s[3] and st == "FINAL" and oc == "UPPER"
    st, oc, _, _ = F.outcome(bars, s[-1] + 10, 0.005, pd.Timestamp(s[-1] + 99 * 3600, unit="s", tz="UTC"))
    assert st == "PENDING"


def test_parse_prob():
    assert F.parse_prob("0.62") == 0.62 and F.parse_prob("Probability: .4") == 0.4 and F.parse_prob("55%") == 0.55
    assert F.parse_prob("I cannot know") is None


def test_decide_is_idempotent_and_records_before_outcome(tmp_path, monkeypatch):
    s = (np.arange(120) * 3600.0 + 1_792_000_800).tolist()
    rng = np.random.default_rng(0)
    c = 1.12 * np.exp(rng.normal(0, 8e-4, 120).cumsum())
    snap = {"raw_sha256": "x", "received_at_epoch": 1_792_464_000.0,                        # a Wednesday, 2026-10-21 12:00 UTC
            "bars": [{"bar_start_epoch": t, "open": v, "high": v * 1.001, "low": v * 0.999, "close": v} for t, v in zip(s, c)]}
    (tmp_path / "snap.json").write_text(json.dumps(snap))
    (tmp_path / "status.json").write_text(json.dumps({"state": "READY", "received_at_epoch": snap["received_at_epoch"],
                                                      "raw_sha256": "x", "snapshot": str(tmp_path / "snap.json")}))
    monkeypatch.setattr(F, "now", lambda: snap["received_at_epoch"] + 10)
    monkeypatch.setattr(F, "ask_ollama", lambda m, p, h: ("0.55", m))
    args = type("A", (), {"status": str(tmp_path / "status.json"), "state": str(tmp_path / "st"), "calendar": "/nonexistent",
                          "models": ["fake"], "ollama": "", "max_age": 900})
    F.decide(args)
    F.decide(args)
    lines = (tmp_path / "st" / "DECISIONS.jsonl").read_text().splitlines()
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["entry_allowed"] and d["models"]["fake"]["p_up"] == 0.55 and d["prompt_sha256"]
    assert d["written_epoch"] < snap["received_at_epoch"] + 3600                                # before any later bar exists

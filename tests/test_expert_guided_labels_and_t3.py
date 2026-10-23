"""Per-agent signal labels in E3-M (rev 2 consumption; Musashi's instructions 2026-10-22), the pointwise extractor
type, the T3 manifest and the executable regularization rule. No data, no fit."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import signals as SG  # noqa: E402
from tools import expert_guided_clean_round as C  # noqa: E402
from tools import expert_guided_e3m as M  # noqa: E402
from tools import expert_guided_t3_manifest as T3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H = 3_600_000_000_000


class _Seg:
    def __init__(self, ff):
        self.force_flat = pd.Timestamp(ff, unit="ns", tz="UTC")


class _Cal:
    def __init__(self, n, ff_bar):
        self.segments = [_Seg(ff_bar)]
        self._n = n

    def assign(self, index):
        return np.zeros(self._n, np.int64)


def _fake_cx(pivots, conf, period_end_bar, n=12):
    """12 bars with a known path; pivots of the optimal expert with their confirmation bars; one period TRAIN ending
    after ``period_end_bar`` (availability = bar start + 1 h)."""
    c = np.array([1.0000, 1.0005, 0.9990, 1.0020, 1.0030, 1.0010, 0.9990, 0.9970, 0.9980, 0.9985, 0.9990, 0.9995])[:n]
    o = np.r_[c[0], c[:-1]]
    bars = pd.DataFrame({"OPEN": o, "HIGH": c + 0.0002, "LOW": c - 0.0002, "CLOSE": c},
                        index=pd.date_range("2021-01-04", periods=n, freq="h", tz="UTC"))
    v = bars.index.as_unit("ns").asi8
    from app.short_horizon.expert_guided import oracle as OR
    piv = [(b, k, lv) for b, k, lv in pivots]
    sup = OR.label_support_end(n, piv, conf)
    ds = type("DS", (), {})()
    ds.bars, ds.sigma, ds.pivots, ds.support_end = bars, np.full(n, 0.0010), piv, sup
    ds.cal = _Cal(n, v[-1] + 10 * H)
    ds.row_bar = np.arange(n)
    ds.period = np.array(["TRAIN"] * n, object)
    ds.eligible = np.ones(n, bool)
    end = pd.Timestamp(v[period_end_bar] + H, unit="ns", tz="UTC")
    ds.meta = {"periods": {"TRAIN": ["2021-01-04", end.strftime("%Y-%m-%dT%H:%M:%S")]}}
    ds.rows = lambda p: np.flatnonzero(ds.period == p)
    cx = type("CX", (), {})()
    cx.ds = ds
    cx.Xw = np.zeros((n, 3), np.float32)
    cx.h, cx.l, cx.c = bars["HIGH"].to_numpy(), bars["LOW"].to_numpy(), c
    cx.oracle = OR.LegOracle(c, piv)
    return cx


def test_label_available_at_waits_for_the_pivot_confirmation_not_the_pivot_price():
    # peak at bar 4 confirmed only at bar 9: the long horizon of bar 0 ends at 4, but its label is final at 9
    piv = [(2, -1, 0.9988), (4, 1, 1.0032), (7, -1, 0.9968)]
    cx = _fake_cx(piv, conf=[3, 9, 9], period_end_bar=11)
    lab = M.SignalLabels(cx, "opportunity", "hold_through", SG.SignalParams(cost_sigma=0.0))
    assert lab.hz[1]["bar"][0] == 4 and lab.avail_bar[1][0] == 9
    # a period that ends after the pivot's price (bar 5) but before its confirmation (bar 9) censors the row
    cx2 = _fake_cx(piv, conf=[3, 9, 9], period_end_bar=5)
    lab2 = M.SignalLabels(cx2, "opportunity", "hold_through", SG.SignalParams(cost_sigma=0.0))
    assert lab2.available(0, 0, 1) is False and lab2.flat_action(0, 0) is None
    assert lab.available(0, 0, 1) is True and lab.flat_action(0, 0)[0] == E.LONG


def test_build_sets_opportunity_mode_relabels_flat_and_positioned_rows_and_drops_censored():
    piv = [(2, -1, 0.9988), (4, 1, 1.0032), (7, -1, 0.9968), (10, 1, 0.9992)]
    cx = _fake_cx(piv, conf=[3, 5, 8, 11], period_end_bar=11)
    p = SG.SignalParams(cost_sigma=0.0, theta_open=1.0, sl_max=2.0, theta_close=0.5)
    lab = M.SignalLabels(cx, "opportunity", "hold_through", p)
    FLAT = np.zeros(9, np.float32); FLAT[-1] = 1.0
    POS = np.zeros(9, np.float32); POS[0] = 1.0
    vis = [(0, 0, FLAT, E.HOLD, 1.5, 2.0, None),                    # ZigZag says HOLD; the signal says LONG (3.2 sigma by the peak)
           (4, 4, POS, E.HOLD, 1.5, 2.0, (1, 1.0000, 0.9980, 1.0040, 11)),   # long at the peak: nothing more to gain -> CLOSE
           (2, 2, POS, E.HOLD, 1.5, 2.0, (1, 1.0000, 0.9980, 1.0040, 11))]   # long at the valley, SL 2 sigma below, peak ahead -> HOLD
    D = M.build_sets("FULL", vis, cx, False, labels=lab, label_mode="opportunity")
    assert D["open_l"][1].tolist() == [1] and D["open_s"][1].tolist() == [0]
    assert np.isclose(D["open_l"][2][0][0], 3.0) and np.isclose(D["open_l"][2][0][1], 1.2)   # TP head bounded at 3, SL = adverse 1.2
    assert D["close_l"][1].tolist() == [M.EXIT, 0] and lab.n["closer_used"] == 2 and lab.n["flat_used"] == 1
    # a tight SL (0.5 sigma) is reached by the 1.2-sigma dip before the peak -> CLOSE for the same market row
    D2 = M.build_sets("FULL", [(2, 2, POS, E.HOLD, 1.5, 2.0, (1, 1.0000, 0.9995, 1.0040, 11))], cx, False, labels=lab, label_mode="opportunity")
    assert D2["close_l"][1].tolist() == [M.EXIT]
    # censored (period ends before the horizon is final): the row is dropped, never filled
    cx3 = _fake_cx(piv, conf=[3, 5, 8, 11], period_end_bar=3)
    lab3 = M.SignalLabels(cx3, "opportunity", "hold_through", p)
    D3 = M.build_sets("FULL", vis[:1], cx3, False, labels=lab3, label_mode="opportunity")
    assert D3 == {} and lab3.n["flat_censored"] == 1
    # the default mode is byte-identical: the oracle label path is untouched (the LegOracle targets the NEXT pivot, the
    # valley at bar 2, so it opens SHORT at bar 0 where the opportunity signal opens LONG towards the larger peak)
    D0 = M.build_sets("FULL", vis[:1], cx, False)
    assert D0["open_s"][1].tolist() == [1] and D0["open_l"][1].tolist() == [0]


def test_both_mode_truth_table_and_optimum_rule_censoring():
    piv = [(2, -1, 0.9988), (4, 1, 1.0032), (7, -1, 0.9968)]
    cx = _fake_cx(piv, conf=[3, 5, 8], period_end_bar=11)
    p = SG.SignalParams(cost_sigma=0.0, theta_open=1.0, sl_max=2.0, theta_close=0.5)
    FLAT = np.zeros(9, np.float32); FLAT[-1] = 1.0
    POS = np.zeros(9, np.float32); POS[0] = 1.0
    lab = M.SignalLabels(cx, "both", "hold_through", p)
    # the ZigZag side wins when it names one, even against the signal's side
    D = M.build_sets("FULL", [(0, 0, FLAT, E.SHORT, 1.5, 2.0, None)], cx, True, labels=lab, label_mode="both")
    assert D["open_s"][1].tolist() == [1] and D["open_l"][1].tolist() == [0]
    # ZigZag HOLD -> the signal's side
    D = M.build_sets("FULL", [(0, 0, FLAT, E.HOLD, 1.5, 2.0, None)], cx, True, labels=lab, label_mode="both")
    assert D["open_l"][1].tolist() == [1]
    # closers: CLOSE if either rule says so. Long at the valley (bar 2), SL 0.9980: the signal says HOLD (peak ahead, dip
    # survivable); the exit rule says CLOSE when the TP (1.0040) is never reached before the SL at bar 7 -> OR = CLOSE;
    # with a TP reached at bar 4 (1.0025) both say HOLD
    D = M.build_sets("FULL", [(2, 2, POS, E.HOLD, 1.5, 2.0, (1, 1.0000, 0.9980, 1.0040, 11))], cx, False, labels=lab, label_mode="both")
    assert D["close_l"][1].tolist() == [M.EXIT] and lab.positioned_action(2, 2, (1, 1.0000, 0.9980, 1.0040, 11)) == E.HOLD
    D = M.build_sets("FULL", [(2, 2, POS, E.HOLD, 1.5, 2.0, (1, 1.0000, 0.9980, 1.0025, 11))], cx, False, labels=lab, label_mode="both")
    assert D["close_l"][1].tolist() == [0]
    # the optimum rule closes at the valley of bar 2 for a SHORT (the next pivot inside the window is a peak = adverse)
    lab_o = M.SignalLabels(cx, "opportunity", "optimum", p)
    assert lab_o.positioned_action(2, 2, (-1, 1.0010, 1.0030, 0.9980, 11)) == E.CLOSE
    # ... and is censored when no pivot lies inside the window (a 1-bar cap from bar 5: the valley at 7 is outside)
    lab_c = M.SignalLabels(cx, "opportunity", "optimum", SG.SignalParams(cost_sigma=0.0, theta_open=1.0, sl_max=2.0, theta_close=0.5, max_bars=1))
    assert lab_c.available(5, 5, 1) and lab_c.positioned_action(5, 5, (1, 1.0000, 0.9950, 1.0100, 11)) is None
    # a horizon truncated at the end of the data is never final: the ZigZag beyond it is not settled
    assert lab_o.available(8, 8, 1) is False and lab_o.positioned_action(8, 8, (1, 0.9970, 0.9950, 1.0000, 11)) is None
    with pytest.raises(ValueError):
        M.SignalLabels(cx, "opportunity", "greedy", p)


def test_e3m_parser_declares_the_label_flags_with_byte_identical_defaults():
    a = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c"])
    assert (a.labels, a.closer_rule, a.theta_open, a.theta_close) == ("zigzag", "hold_through", 0.5, 0.1)
    b = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c", "--labels", "both", "--closer-rule", "optimum"])
    assert (b.labels, b.closer_rule) == ("both", "optimum")
    with pytest.raises(SystemExit):
        M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c", "--labels", "oracle"])


def test_pointwise_group_extractor_has_no_temporal_mixing_and_fewer_parameters():
    import torch
    from app.short_horizon.expert_guided import temporal as T
    torch.manual_seed(0)
    conv = T.TemporalTrunk(4, groups=[[0, 1], [2, 3]], group_units=[2, 2], group_kinds=["conv", "conv"])
    pw = T.TemporalTrunk(4, groups=[[0, 1], [2, 3]], group_units=[2, 2], group_kinds=["conv", "pointwise"])
    n_conv = sum(p.numel() for p in conv.branch[1].parameters())
    n_pw = sum(p.numel() for p in pw.branch[1].parameters())
    assert n_pw == 2 * 2 + 2 and n_conv == 2 * 2 * 3 + 2
    # the pointwise branch output at time t depends on x[t] only
    x = torch.randn(1, 13, 4)
    y0 = pw.branch[1](x[:, :, 2:].transpose(1, 2))
    x2 = x.clone(); x2[:, :6, :] += 5.0
    y1 = pw.branch[1](x2[:, :, 2:].transpose(1, 2))
    assert torch.allclose(y0[:, :, 6:], y1[:, :, 6:]) and not torch.allclose(y0[:, :, :6], y1[:, :, :6])
    with pytest.raises(ValueError):
        T.TemporalTrunk(4, groups=[[0, 1], [2, 3]], group_units=[2, 2], group_kinds=["conv", "dense"])
    # the policy net carries the kinds and round-trips its state dict
    net = T.TemporalPolicyNet(4, 13, 2, n_reg=2, n_state=9, groups=[[0, 1], [2, 3]], group_units=[2, 2], group_kinds=["pointwise", "conv"])
    net2 = T.TemporalPolicyNet(4, 13, 2, n_reg=2, n_state=9, groups=[[0, 1], [2, 3]], group_units=[2, 2], group_kinds=["pointwise", "conv"])
    net2.load_state_dict(net.state_dict())
    assert net.trunk.group_kinds == ["pointwise", "conv"]


def _spec_and_groups(tmp_path):
    cols = ["a", "b", "c", "d", "e", "cal.x", "cal.y", "dly.z"]
    lst = tmp_path / "FEATURE_LIST_x.json"; lst.write_text(json.dumps({"features": cols}))
    grp = tmp_path / "FEATURE_GROUPS_x.json"
    grp.write_text(json.dumps({"groups": {"px": ["a", "b", "c"], "wavelet/x": ["d", "e"], "calendar": ["cal.x", "cal.y"], "dly.z": ["dly.z"]},
                               "units": {"px": 2, "wavelet/x": 2, "calendar": 2, "dly.z": 2}}))
    spec = tmp_path / "SPEC.json"
    spec.write_text(json.dumps({"model": {"units": 2, "depth": 2, "head": 16, "core": None, "max_epochs": 5}, "scaling": "fixed_train",
                                "nonstationary": "diff", "split_contract": "consolidated", "exec_contract": "sac", "min_orders_per_week": 2.0,
                                "regularization": {"candidates": {"base": {"dropout": 0.1, "wd": 1e-4}, "strong": {"dropout": 0.3, "wd": 1e-2}}}}))
    return spec, grp, lst


def test_t3_manifest_is_finite_stable_partitioned_and_split_by_parity(tmp_path):
    spec, grp, lst = _spec_and_groups(tmp_path)
    out = tmp_path / "t3"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_t3_manifest.py"), "emit", "--spec", str(spec), "--groups", str(grp),
                        "--list", str(lst), "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    man = json.loads((out / "T3_MANIFEST.json").read_text())
    units = man["units"]
    assert [u["id"] for u in units] == [0, 1, 2, 3, 4] and units[0]["role"] == "control"
    assert [u["kind"] for u in units[1:]] == ["per_feature", "per_feature", "pointwise", "pointwise"]   # wavelet per scale; calendar/daily pointwise; singleton pointwise
    assert [u["worker"] for u in units] == ["satoshi", "musashi", "satoshi", "musashi", "satoshi"]
    cols = json.loads(lst.read_text())["features"]
    for u in units:
        g = json.loads(Path(u["groups_file"]).read_text())
        idx, un, names, kinds = C.load_groups(u["groups_file"], cols)
        assert sorted(i for gg in idx for i in gg) == list(range(len(cols)))         # every unit partitions the list
        assert all(k in ("conv", "pointwise") for k in kinds)
    per = json.loads(Path(units[2]["groups_file"]).read_text())
    assert "wavelet/x/d" in per["groups"] and "wavelet/x" not in per["groups"]
    assert man["fixed"]["regularization"]["status"] == "DEPENDENCY"
    # stable identities: emitting again gives the same units
    r2 = subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_t3_manifest.py"), "emit", "--spec", str(spec), "--groups", str(grp),
                         "--list", str(lst), "--out", str(tmp_path / "t3b")], capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode == 0
    man2 = json.loads((tmp_path / "t3b" / "T3_MANIFEST.json").read_text())
    assert [u["identity"] for u in man2["units"]] == [u["identity"] for u in units]
    # the queue is refused without the T2 decision and written with it
    q = subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_t3_manifest.py"), "queue", "--manifest", str(out / "T3_MANIFEST.json"),
                        "--decision", str(tmp_path / "missing.json"), "--worker", "musashi", "--remote-root", "/r", "--store", "s.parquet",
                        "--out", str(tmp_path / "q.sh")], capture_output=True, text=True, cwd=ROOT)
    assert q.returncode != 0 and "T3_REGULARIZATION_DEPENDENCY_MISSING" in q.stderr
    (tmp_path / "dec.json").write_text(json.dumps({"how": "NON_DOMINATED_UNIQUE", "values": {"dropout": 0.3, "wd": 0.01}}))
    q = subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_t3_manifest.py"), "queue", "--manifest", str(out / "T3_MANIFEST.json"),
                        "--decision", str(tmp_path / "dec.json"), "--worker", "musashi", "--remote-root", "/r", "--store", "s.parquet",
                        "--out", str(tmp_path / "q.sh")], capture_output=True, text=True, cwd=ROOT)
    assert q.returncode == 0 and (tmp_path / "q.sh").read_text().count("expert_guided_e3m.py run") == 2   # ids 1 and 3
    assert "--dropout 0.3 --wd 0.01" in (tmp_path / "q.sh").read_text() and "--split-contract consolidated" in (tmp_path / "q.sh").read_text()


def _e3m(path, inner, train, dd, valid=True, support=("t1", "i1"), state="none", arm="consolidated"):
    seeds = {str(s): {"trading": {"TRAIN": {"sharpe_daily_sqrt365": tr, "max_drawdown_pct": 5.0},
                                  "INNER_A": {"sharpe_daily_sqrt365": i, "max_drawdown_pct": d},
                                  "OBS_2024": {"sharpe_daily_sqrt365": 99.0, "max_drawdown_pct": 0.0}}}     # absurd: must never be read
             for s, i, tr, d in zip((1, 2, 3), inner, train, dd)}
    path.mkdir(parents=True)
    (path / "E3M.json").write_text(json.dumps({"trials": {"FULL": seeds}, "decision": {"means": {"FULL": {"all_valid": valid}}},
                                               "meta": {"support_hash": {"TRAIN": support[0], "INNER_A": support[1]}},
                                               "agent_state": {"mode": state}, "split_contract": {"arm": arm},
                                               "exec_contract": "sac", "min_orders_per_week": 2.0}))


def _run_reg(tmp_path, spec, res, extra=()):
    out = tmp_path / "REGULARIZATION_DECISION.json"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_clean_round.py"), "regularize", "--spec", str(spec),
                        "--results", str(res), "--out", str(out), "--lists", "core15,shortlist_strict", *extra], capture_output=True, text=True, cwd=ROOT)
    return r, (json.loads(out.read_text()) if out.exists() else None)


def test_regularization_rule_is_non_dominated_with_a_declared_tie_break(tmp_path):
    spec, _, _ = _spec_and_groups(tmp_path)
    res = tmp_path / "res"
    # strong better on every component of both lists -> unique non-dominated
    _e3m(res / "gap_core15_none_base", (0.5, 0.6, 0.7), (3.0, 3.1, 3.2), (10, 11, 12))
    _e3m(res / "gap_core15_none_strong", (0.8, 0.8, 0.8), (2.0, 2.0, 2.0), (8, 8, 8))
    _e3m(res / "gap_shortlist_strict_none_base", (0.1, 0.2, 0.3), (5.0, 5.0, 5.0), (20, 20, 20), support=("t2", "i2"))
    _e3m(res / "gap_shortlist_strict_none_strong", (0.4, 0.4, 0.4), (3.0, 3.0, 3.0), (15, 15, 15), support=("t2", "i2"))
    r, d = _run_reg(tmp_path, spec, res, ["--apply"])
    assert r.returncode == 0, r.stderr
    assert d["chosen"] == "strong" and d["how"] == "NON_DOMINATED_UNIQUE" and d["non_dominated"] == ["strong"]
    s = json.loads(spec.read_text())["regularization"]
    assert (s["dropout"], s["wd"], s["status"]) == (0.3, 0.01, "DECIDED") and s["source"].startswith("T2 decision")
    # mixed: strong wins Sharpe by less than the seed dispersion and has the larger gap on one list -> both non-dominated,
    # tie-break 1 fails (difference within pooled std), tie-break 2 -> the smaller mean gap (base)
    res2 = tmp_path / "res2"
    _e3m(res2 / "gap_core15_none_base", (0.5, 0.7, 0.9), (2.0, 2.2, 2.4), (10, 10, 10))
    _e3m(res2 / "gap_core15_none_strong", (0.6, 0.8, 1.0), (2.5, 2.7, 2.9), (10, 10, 10))
    _e3m(res2 / "gap_shortlist_strict_none_base", (0.1, 0.3, 0.5), (3.0, 3.2, 3.4), (20, 20, 20), support=("t2", "i2"))
    _e3m(res2 / "gap_shortlist_strict_none_strong", (0.2, 0.4, 0.6), (3.5, 3.7, 3.9), (20, 20, 20), support=("t2", "i2"))
    spec2 = tmp_path / "SPEC2.json"; spec2.write_text(Path(spec).read_text())
    r, d = _run_reg(tmp_path, spec2, res2)
    assert r.returncode == 0 and sorted(d["non_dominated"]) == ["base", "strong"] and d["chosen"] == "base" and d["how"] == "TIE_BREAK_2_SMALLER_GAP"
    # an invalid setting on a list removes that list from the comparison; no common valid list -> base by default, never applied
    res3 = tmp_path / "res3"
    _e3m(res3 / "gap_core15_none_base", (0.5, 0.6, 0.7), (3.0, 3.1, 3.2), (10, 11, 12), valid=False)
    _e3m(res3 / "gap_core15_none_strong", (0.8, 0.8, 0.8), (2.0, 2.0, 2.0), (8, 8, 8))
    _e3m(res3 / "gap_shortlist_strict_none_base", (0.1, 0.2, 0.3), (5.0, 5.0, 5.0), (20, 20, 20), support=("t2", "i2"))
    _e3m(res3 / "gap_shortlist_strict_none_strong", (0.4, 0.4, 0.4), (3.0, 3.0, 3.0), (15, 15, 15), valid=False, support=("t2", "i2"))
    r, d = _run_reg(tmp_path, spec2, res3, ["--apply"])
    assert r.returncode != 0 and d["how"] == "DEFAULT_BY_VALIDITY" and d["chosen"] == "base" and d["compared_lists"] == []
    # a support mismatch between the compared settings is refused
    res4 = tmp_path / "res4"
    _e3m(res4 / "gap_core15_none_base", (0.5, 0.6, 0.7), (3.0, 3.1, 3.2), (10, 11, 12), support=("t1", "i1"))
    _e3m(res4 / "gap_core15_none_strong", (0.8, 0.8, 0.8), (2.0, 2.0, 2.0), (8, 8, 8), support=("t1", "iX"))
    r, _ = _run_reg(tmp_path, spec2, res4)
    assert r.returncode != 0 and "SUPPORT_MISMATCH" in r.stderr
    # an arm run with agent feedback or the wrong contract is refused
    res5 = tmp_path / "res5"
    _e3m(res5 / "gap_core15_none_base", (0.5, 0.6, 0.7), (3.0, 3.1, 3.2), (10, 11, 12), state="full")
    r, _ = _run_reg(tmp_path, spec2, res5)
    assert r.returncode != 0 and "ARM_AGENT_STATE" in r.stderr


@pytest.mark.slow
def test_e3m_cli_signal_labels_run_end_to_end_and_checkpoints_reload(tmp_path):
    """Musashi's instructions 2026-10-22: CLI -> labels/tensors -> round 0 -> DAgger -> checkpoint -> reload, with the
    replay contract, no feedback, `both` labels and the `optimum` closer; the saved modules reproduce their probe."""
    import os
    import torch
    from app.short_horizon.expert_guided import data as DG
    from app.short_horizon.expert_guided import temporal as T
    from tests.test_expert_guided_scaling import ADMITTED_DIR, SEL15_DIR, build_real_fixture
    if not (ADMITTED_DIR / "eurusd.parquet").exists() or not (SEL15_DIR / "train.csv").exists():
        pytest.skip("admitted file / sel15 CSVs not available on this host")
    adm, sel = build_real_fixture(tmp_path / "fx")
    out = tmp_path / "e3m"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    cmd = [sys.executable, str(ROOT / "tools" / "expert_guided_e3m.py"), "run", "--admitted", str(adm), "--csv-dir", str(sel),
           "--out", str(out), "--scaling", "fixed_train", "--max-epochs", "1", "--variants", "FULL", "--cpu", "--no-verify",
           "--agent-state", "none", "--exec-contract", "sac", "--min-orders-per-week", "2.0", "--labels", "both", "--closer-rule", "optimum",
           "--depth", "2", "--head", "16"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    e3 = json.loads((out / "E3M.json").read_text())
    lab = e3["labels"]
    assert lab["mode"] == "both" and lab["closer_rule"] == "optimum" and lab["params"]["theta_open"] == 0.5
    assert lab["counts"]["flat_used"] > 0 and lab["counts"]["closer_used"] > 0 and lab["profile_train"]["rows"] > 0
    assert lab["profile_train"]["open_long"] + lab["profile_train"]["open_short"] + lab["profile_train"]["nop"] + lab["profile_train"]["censored"] == lab["profile_train"]["rows"]
    assert e3["exec_contract"] == "sac" and e3["min_orders_per_week"] == 2.0 and e3["agent_state"]["mode"] == "none"
    for seed, t in e3["trials"]["FULL"].items():
        for p, m in t["trading"].items():
            assert m["max_entries_per_day"] <= 1 and "weeks_below_min_orders" in m
    # checkpoints reload and reproduce the probe logits recorded at save time
    ds, art = DG.load_with_scaler(out, adm, sel, verify=False)
    Xw = ds.windows.reshape(len(ds.windows), -1).astype(np.float32)
    nf, win = ds.windows.shape[2], ds.windows.shape[1]
    arch = e3["arch"]
    key = f"FULL_open_l_{list(e3['trials']['FULL'])[0]}"
    net = T.TemporalPolicyNet(nf, win, 2, n_reg=2, n_state=9, units=arch["units"], branch=arch["branch"], core=arch["core"],
                              depth=arch["depth"], head=arch["head"])
    net.load_state_dict(torch.load(out / f"{key}.pt", map_location="cpu"))
    net.eval()
    pr = e3["probe"][key]
    X = torch.tensor(np.stack([np.r_[Xw[r], np.zeros(9, np.float32)] for r in pr["rows"]]))
    with torch.no_grad():
        lg, _ = net(X)
    assert np.allclose(lg.numpy(), np.array(pr["logits"]), atol=1e-4)


def test_t2_verifier_reads_a_missing_labels_block_as_legacy_zigzag_and_rejects_other_modes(tmp_path):
    """Musashi's precision 2026-10-22: the pinned producer (186b131) does not emit `labels`; that is not a failure, but an
    explicit non-ZigZag or unknown mode is. Real fields only; no E3M.json is rewritten to satisfy a schema."""
    res = tmp_path / "out"
    _e3m(res / "gap_core15_none_base", (0.5, 0.6, 0.7), (3.0, 3.1, 3.2), (10, 11, 12))
    e = json.loads((res / "gap_core15_none_base" / "E3M.json").read_text())
    e["trials"]["FULL"] = {str(s): v for s, v in zip((20261019, 20261020, 20261021), e["trials"]["FULL"].values())}
    e["meta"]["scaling"] = {"policy": "fixed_train"}
    e["decision"]["means"]["FULL"].update(inner_a_sharpe_mean=0.6)
    unit = {"name": "gap_core15_none_base", "identity": "x", "agent_state": "none"}
    assert C.label_mode_of(e) == ("zigzag", "PRODUCER_PREDATES_FIELD") and C.verify_t2_arm(e, unit) == []
    assert C.verify_t2_arm(dict(e, labels={"mode": "zigzag"}), unit) == []
    assert C.verify_t2_arm(dict(e, labels={"mode": "opportunity"}), unit) == ["LABELS:opportunity:DECLARED"]
    assert C.verify_t2_arm(dict(e, labels={"mode": "oracle"}), unit) == ["LABELS:oracle:UNKNOWN_MODE"]
    assert "SPLIT:dev3y" in C.verify_t2_arm(dict(e, split_contract={"arm": "dev3y"}), unit)
    (res / "gap_core15_none_base" / "E3M.json").write_text(json.dumps(e))
    units = tmp_path / "U.json"
    units.write_text(json.dumps({"units": [unit, {"name": "gap_core15_none_strong", "identity": "y", "agent_state": "none"}]}))
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_clean_round.py"), "verify-t2", "--units", str(units), "--results", str(res)],
                       capture_output=True, text=True, cwd=ROOT)
    d = json.loads(r.stdout)
    assert r.returncode == 2 and [x["status"] for x in d["rows"]] == ["OK", "MISSING"] and d["rows"][0]["labels"] == "zigzag/PRODUCER_PREDATES_FIELD"

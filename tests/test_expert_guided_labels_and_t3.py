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
                                "store": {"path": str(tmp_path / "store.parquet"), "sha256": "0" * 64},
                                "nonstationary": "diff", "split_contract": "consolidated", "exec_contract": "sac", "min_orders_per_week": 2.0,
                                "regularization": {"candidates": {"base": {"dropout": 0.1, "wd": 1e-4}, "strong": {"dropout": 0.3, "wd": 1e-2}}}}))
    return spec, grp, lst


def _profiles(path, cols, acf1, acf24, within):
    hdr = ["column", "acf_lag1", "acf_lag24", "acf_first_insignificant_lag", "redundancy_within_family_max", "fold_stability_ratio", "constant_fit"]
    with open(path, "w") as f:
        f.write(",".join(hdr) + "\n")
        for c in cols:
            f.write(",".join([c, str(acf1[c]), str(acf24[c]), "5", str(within[c]), "0.1", "False"]) + "\n")


def _t3_inputs(tmp_path):
    spec, grp, lst = _spec_and_groups(tmp_path)
    s = json.loads(spec.read_text())
    s["regularization"].update(dropout=0.1, wd=1e-4, chosen="base", status="DECIDED", source="test")
    spec.write_text(json.dumps(s))
    cols = json.loads(lst.read_text())["features"]
    acf1 = {"a": 0.02, "b": 0.01, "c": 0.0, "d": 0.5, "e": 0.5, "cal.x": 0.99, "cal.y": 0.98, "dly.z": 0.97}
    acf24 = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.1, "e": 0.1, "cal.x": 0.99, "cal.y": 0.99, "dly.z": 0.8}
    within = {"a": 0.99, "b": 0.99, "c": 0.99, "d": 0.3, "e": 0.3, "cal.x": 0.7, "cal.y": 0.7, "dly.z": ""}
    prof = tmp_path / "PROFILES.csv"
    _profiles(prof, cols, acf1, acf24, within)
    return spec, grp, lst, prof


def _t3(tmp_path, *args):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_t3_manifest.py"), *map(str, args)], capture_output=True, text=True, cwd=ROOT)


def test_t3_v2_candidates_and_expectations_come_from_each_group_profile(tmp_path):
    spec, grp, lst, prof = _t3_inputs(tmp_path)
    out = tmp_path / "t3"
    r = _t3(tmp_path, "emit", "--spec", spec, "--groups", grp, "--list", lst, "--profiles", prof, "--profiles-manifest-sha256", "x", "--out", out)
    assert r.returncode == 0, r.stderr
    man = json.loads((out / "T3_MANIFEST.json").read_text())
    u = man["units"]
    # control + 3 multi-column groups x 2 alternatives + 1 single-column group x pointwise
    assert [x["id"] for x in u] == list(range(8)) and u[0]["role"] == "control"
    assert [(x["group"], x["kind"]) for x in u[1:]] == [("px", "pointwise"), ("px", "per_feature"), ("wavelet/x", "pointwise"),
                                                         ("wavelet/x", "per_feature"), ("calendar", "pointwise"), ("calendar", "per_feature"),
                                                         ("dly.z", "pointwise")]
    assert [x["worker"] for x in u] == ["satoshi", "musashi"] * 4
    ex = {x["group"]: x["expectation"] for x in u[1:]}
    assert ex["px"]["memory"] == "near_white" and ex["px"]["redundancy"] == "high" and ex["px"]["expected"] == ["conv >= pointwise", "conv >= per_feature"]
    assert ex["wavelet/x"]["redundancy"] == "low" and ex["wavelet/x"]["expected"] == ["per_feature >= conv"]
    assert ex["calendar"]["memory"] == "persistent" and "pointwise >= conv" in ex["calendar"]["expected"]
    cols = json.loads(lst.read_text())["features"]
    for x in u:
        idx, _, _, kinds = C.load_groups(out / "groups" / x["groups_file"], cols)
        assert sorted(i for g in idx for i in g) == list(range(len(cols)))
    # stable identities
    r2 = _t3(tmp_path, "emit", "--spec", spec, "--groups", grp, "--list", lst, "--profiles", prof, "--profiles-manifest-sha256", "x", "--out", tmp_path / "t3b")
    assert [x["identity"] for x in json.loads((tmp_path / "t3b" / "T3_MANIFEST.json").read_text())["units"]] == [x["identity"] for x in u]
    # the T2 decision is a hard dependency
    s = json.loads(spec.read_text()); s["regularization"].update(status="PENDING_T2", dropout=None); spec.write_text(json.dumps(s))
    r3 = _t3(tmp_path, "emit", "--spec", spec, "--groups", grp, "--list", lst, "--profiles", prof, "--profiles-manifest-sha256", "x", "--out", tmp_path / "t3c")
    assert r3.returncode != 0 and "T3_REGULARIZATION_NOT_DECIDED" in r3.stderr
    # typed queue per worker, no duplicates across workers
    for w in ("satoshi", "musashi"):
        q = _t3(tmp_path, "queue", "--manifest", out / "T3_MANIFEST.json", "--worker", w, "--out", tmp_path / f"q_{w}.sh", "--out-units", tmp_path / f"U_{w}.json")
        assert q.returncode == 0, q.stderr
    us = json.loads((tmp_path / "U_satoshi.json").read_text())["units"] + json.loads((tmp_path / "U_musashi.json").read_text())["units"]
    assert sorted(x["id"] for x in us) == list(range(8))
    sh = (tmp_path / "q_musashi.sh").read_text()
    assert sh.count("expert_guided_e3m.py run") == 4 and "--dropout 0.1 --wd 0.0001" in sh and "--split-contract consolidated" in sh and "--window 13 --stride 2" in sh


def _t3_result(path, sharpe, gap, params=100, support=("t", "i"), valid=True):
    seeds = {str(s): {"trading": {"INNER_A": {"sharpe_daily_sqrt365": x}, "TRAIN": {"sharpe_daily_sqrt365": x + g},
                                  "OBS_2024": {"sharpe_daily_sqrt365": 99.0}}} for s, x, g in zip((1, 2, 3), sharpe, gap)}
    path.mkdir(parents=True)
    (path / "E3M.json").write_text(json.dumps({"trials": {"FULL": seeds}, "decision": {"means": {"FULL": {"all_valid": valid}}},
                                               "params": {"FULL": {"m": params}}, "meta": {"support_hash": {"TRAIN": support[0], "INNER_A": support[1]}}}))


def test_t3_v2_decision_rule_and_combination(tmp_path):
    spec, grp, lst, prof = _t3_inputs(tmp_path)
    out = tmp_path / "t3"
    assert _t3(tmp_path, "emit", "--spec", spec, "--groups", grp, "--list", lst, "--profiles", prof, "--profiles-manifest-sha256", "x", "--out", out).returncode == 0
    man = json.loads((out / "T3_MANIFEST.json").read_text())
    res = tmp_path / "res"
    ctl = ((0.20, 0.25, 0.30), (1.0, 1.0, 1.0))
    for u in man["units"]:
        sharpe, gap = ctl
        if (u["group"], u["kind"]) == ("px", "per_feature"):
            sharpe = (0.80, 0.85, 0.90)                                 # beyond the pooled seed std, same gap -> replaces conv
        if (u["group"], u["kind"]) == ("calendar", "pointwise"):
            sharpe, gap = (0.80, 0.85, 0.90), (3.0, 3.0, 3.0)           # better inner but a much larger gap -> does not qualify
        if (u["group"], u["kind"]) == ("dly.z", "pointwise"):
            sharpe = (0.22, 0.27, 0.32)                                 # within the seed noise -> control stays
        _t3_result(res / u["name"], sharpe, gap)
    d = _t3(tmp_path, "decide", "--manifest", out / "T3_MANIFEST.json", "--results", res, "--out", tmp_path / "DEC.json")
    assert d.returncode == 0, d.stderr
    dec = {x["group"]: x["chosen"] for x in json.loads((tmp_path / "DEC.json").read_text())["groups"]}
    assert dec == {"px": "per_feature", "wavelet/x": "conv", "calendar": "conv", "dly.z": "conv"}
    c = _t3(tmp_path, "combine", "--manifest", out / "T3_MANIFEST.json", "--decision", tmp_path / "DEC.json")
    assert c.returncode == 0, c.stderr
    gC = json.loads((out / "groups" / "unit_C.json").read_text())
    assert "px" not in gC["groups"] and gC["groups"]["px/a"] == ["a"] and gC["kinds"]["calendar"] == "conv"
    # a unit on a different support is refused
    import shutil
    shutil.rmtree(res / man["units"][3]["name"])
    _t3_result(res / man["units"][3]["name"], *ctl, support=("t", "other"))
    d2 = _t3(tmp_path, "decide", "--manifest", out / "T3_MANIFEST.json", "--results", res, "--out", tmp_path / "DEC2.json")
    assert d2.returncode != 0 and "T3_SUPPORT_DIFFERS" in d2.stderr


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


def _verified(tmp_path, res):
    """A verification document for the synthetic arms (what verify-t2 writes when all_ok)."""
    rows = [{"unit": d.name, "status": "OK", "sha256": C.sha(d / "E3M.json")} for d in sorted(res.iterdir())]
    v = tmp_path / f"T2_VERIFY_{res.name}.json"
    v.write_text(json.dumps({"all_ok": True, "rows": rows, "producer": {"authenticated": True, "emits_labels": False}}))
    return v


def _run_reg(tmp_path, spec, res, extra=()):
    out = tmp_path / "REGULARIZATION_DECISION.json"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_clean_round.py"), "regularize", "--spec", str(spec),
                        "--results", str(res), "--out", str(out), "--lists", "core15,shortlist_strict",
                        "--verified", str(_verified(tmp_path, res)), *extra], capture_output=True, text=True, cwd=ROOT)
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


def test_regularize_refuses_a_result_that_is_not_the_verified_file(tmp_path):
    spec, _, _ = _spec_and_groups(tmp_path)
    res = tmp_path / "res"
    for lst, sup in (("core15", ("t1", "i1")), ("shortlist_strict", ("t2", "i2"))):
        _e3m(res / f"gap_{lst}_none_base", (0.5, 0.6, 0.7), (3.0, 3.1, 3.2), (10, 11, 12), support=sup)
        _e3m(res / f"gap_{lst}_none_strong", (0.8, 0.8, 0.8), (2.0, 2.0, 2.0), (8, 8, 8), support=sup)
    v = _verified(tmp_path, res)
    f = res / "gap_core15_none_strong" / "E3M.json"
    f.write_text(f.read_text().replace('"max_drawdown_pct": 8', '"max_drawdown_pct": 1'))      # swapped after the check
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_clean_round.py"), "regularize", "--spec", str(spec),
                        "--results", str(res), "--out", str(tmp_path / "D.json"), "--verified", str(v)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "ARM_NOT_THE_VERIFIED_RESULT:gap_core15_none_strong" in r.stderr
    v.write_text(json.dumps({"all_ok": False, "rows": [], "producer": {}}))
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_clean_round.py"), "regularize", "--spec", str(spec),
                        "--results", str(res), "--out", str(tmp_path / "D.json"), "--verified", str(v)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "T2_NOT_VERIFIED" in r.stderr


AUTH = {"authenticated": True, "emits_labels": False, "config_identity": "cfg"}


def test_label_mode_requires_an_authenticated_legacy_producer_musashi_c01():
    """Musashi's C01 witnesses as regressions: absence is legacy ZigZag ONLY with an authenticated producer that does not
    emit the field; null, a list, an unknown producer, or a field the pinned producer cannot emit are rejected explicitly."""
    e = {"trials": {}}
    assert C.label_mode_of(e, AUTH) == ("zigzag", C.LEGACY_ABSENT) and C.labels_ok(*C.label_mode_of(e, AUTH))
    assert C.label_mode_of(e, None) == ("absent", C.UNAUTH) and not C.labels_ok(*C.label_mode_of(e, None))      # arbitrary producer
    assert C.label_mode_of(e, {"authenticated": False}) == ("absent", C.UNAUTH)
    assert C.label_mode_of(dict(e, labels=None), AUTH) == ("null", C.MALFORMED)                                 # null is not absent
    assert C.label_mode_of(dict(e, labels=["zigzag"]), AUTH) == ("list", C.MALFORMED)                           # no AttributeError
    assert C.label_mode_of(dict(e, labels={"mode": 3})) == ("3", C.MALFORMED)
    assert C.label_mode_of(dict(e, labels={"mode": "zigzag"}), AUTH)[1] == "PRESENT_BUT_PRODUCER_DOES_NOT_EMIT"
    assert C.labels_ok(*C.label_mode_of(dict(e, labels={"mode": "zigzag"})))                                     # a new producer declares it
    assert not C.labels_ok(*C.label_mode_of(dict(e, labels={"mode": "opportunity"})))
    assert C.label_mode_of(dict(e, labels={"mode": "oracle"})) == ("oracle", C.UNKNOWN)


def _receipt_case(tmp_path):
    arm = tmp_path / "out" / "gap_core15_none_base"
    arm.mkdir(parents=True)
    files = {"E3M.json": b"{}", "SCALER.json": b"s", "PROBE.json": b"p"}
    files |= {f"FULL_{m}_{s}.pt": f"{m}{s}".encode() for m in ("open_l", "open_s", "close_l", "close_s") for s in ("20261019", "20261020", "20261021")}
    for n, b in files.items():
        (arm / n).write_bytes(b)
    cmd = "tools/expert_guided_e3m.py run --feature-list <musashi-root>/in/FEATURE_LIST_core15.json --dropout 0.1 --wd 1e-4"
    import hashlib
    unit = {"name": "gap_core15_none_base", "identity": hashlib.sha256(cmd.encode()).hexdigest()[:16], "command": cmd,
            "agent_state": "none", "list": "core15", "dropout": 0.1, "wd": 0.0001}
    rc = {"config_sha256": "cfg", "job": {"name": unit["name"], "peer_command": cmd, "peer_command_identity": unit["identity"],
                                         "dropout": 0.1, "wd": 0.0001, "state": "none", "list": "FEATURE_LIST_core15.json", "split": "consolidated"},
          "hashes": {n: C.sha(arm / n) for n in files}, "support_hash": {"TRAIN": "t", "INNER_A": "i"},
          "requested_features": ["a", "b"], "consumed_features": ["a", "b"]}
    e = {"meta": {"support_hash": {"TRAIN": "t", "INNER_A": "i"}}}
    return rc, unit, arm, e


def test_receipt_binds_identity_regularization_list_and_every_sealed_file_musashi_c02(tmp_path):
    """Musashi's C02 witness: the same result presented under an altered unit (name, identity, list, dropout, wd) must
    fail; so must a receipt from another config, a swapped checkpoint or a different support."""
    rc, unit, arm, e = _receipt_case(tmp_path)
    assert C.verify_t2_receipt(rc, unit, arm, e, AUTH, ["a", "b"]) == []
    bad = dict(unit, name="gap_shortlist_strict_none_strong", identity="0" * 16, list="shortlist_strict", dropout=0.3, wd=0.01)
    errs = C.verify_t2_receipt(rc, bad, arm, e, AUTH, ["a", "b"])
    for code in ("RECEIPT_NAME", "RECEIPT_IDENTITY", "RECEIPT_REGULARIZATION", "RECEIPT_LIST"):
        assert any(x.startswith(code) for x in errs), (code, errs)
    assert "RECEIPT_CONFIG_IDENTITY" in C.verify_t2_receipt(rc, unit, arm, e, dict(AUTH, config_identity="other"), ["a", "b"])
    (arm / "FULL_open_l_20261020.pt").write_bytes(b"swapped")
    assert "ARTIFACT_HASH:FULL_open_l_20261020.pt" in C.verify_t2_receipt(rc, unit, arm, e, AUTH, ["a", "b"])
    rc2, unit2, arm2, e2 = _receipt_case(tmp_path / "b")
    assert "RECEIPT_SUPPORT_DIFFERS" in C.verify_t2_receipt(rc2, unit2, arm2, {"meta": {"support_hash": {"TRAIN": "t", "INNER_A": "x"}}}, AUTH, ["a", "b"])
    assert "RECEIPT_COLUMNS_DIFFER_FROM_LIST" in C.verify_t2_receipt(rc2, unit2, arm2, e2, AUTH, ["a", "c"])
    del rc2["hashes"]["PROBE.json"]
    assert "RECEIPT_HASH_MISSING:PROBE.json" in C.verify_t2_receipt(rc2, unit2, arm2, e2, AUTH, ["a", "b"])


def test_producer_authentication_compares_the_pinned_source_with_the_science_commit(tmp_path):
    import hashlib
    blob = subprocess.run(["git", "show", "186b131:tools/expert_guided_e3m.py"], cwd=ROOT, capture_output=True)
    if blob.returncode != 0:
        pytest.skip("science commit not in this clone")
    cfg = {"pins": {"<root>/code/tools/expert_guided_e3m.py": hashlib.sha256(blob.stdout).hexdigest(), "<root>/a.tar.gz": "x"}}
    p = tmp_path / "CONFIG.json"
    p.write_text(json.dumps(cfg))
    pr = C.authenticate_producer(p, "186b131")
    assert pr["authenticated"] and pr["emits_labels"] is False and pr["config_identity"] == hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    cfg["pins"]["<root>/code/tools/expert_guided_e3m.py"] = "0" * 64
    p.write_text(json.dumps(cfg))
    assert C.authenticate_producer(p, "186b131")["error"] == "PINNED_SOURCE_DIFFERS_FROM_SCIENCE_COMMIT"
    pr_new = C.authenticate_producer(p, "e9291b7")
    assert pr_new["authenticated"] is False                           # pin does not match the newer source either


def test_producer_authentication_uses_an_exported_blob_when_git_is_absent(tmp_path, monkeypatch):
    import hashlib
    blob = subprocess.run(["git", "show", "186b131:tools/expert_guided_e3m.py"], cwd=ROOT, capture_output=True)
    if blob.returncode != 0:
        pytest.skip("science commit not in this clone")
    cfg = tmp_path / "CONFIG.json"
    cfg.write_text(json.dumps({"pins": {"<root>/code/tools/expert_guided_e3m.py": hashlib.sha256(blob.stdout).hexdigest()}}))
    d = tmp_path / "blobs" / "cafe123" / "tools"
    d.mkdir(parents=True)
    (d / "expert_guided_e3m.py").write_bytes(blob.stdout)
    pr = C.authenticate_producer(cfg, "cafe123", blobs=tmp_path / "blobs")            # unknown to git: the blob decides
    assert pr["authenticated"] and pr["science_source"] == "exported_blob" and pr["emits_labels"] is False
    (d / "expert_guided_e3m.py").write_bytes(blob.stdout + b"# altered")
    assert C.authenticate_producer(cfg, "cafe123", blobs=tmp_path / "blobs")["authenticated"] is False
    assert C.authenticate_producer(cfg, "cafe123")["error"] == "SCIENCE_COMMIT_UNKNOWN:cafe123"

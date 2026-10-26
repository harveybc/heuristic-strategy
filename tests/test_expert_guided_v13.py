"""Contract v1.3 (2026-10-25, after the v12 chain: seeds opening one side only and activity sitting on the 2.0 boundary):
per-side open genes, the side-balance constraint, the activity rule held with confidence (CI90 lower bound of trades/week
>= 2.0 on the calibration block, weekly-block bootstrap; owner: "Con confianza estadística"), the front-size diagnostic
with its budget escalation, the report fixes and the v1.3 stages. Synthetic data, no training."""
import copy
import json
import math
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.short_horizon.expert_guided import action_contract as ACT

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "docs/short_horizon/clean_round/contracts"


# ----------------------------------------------------------------------------- activity CI
def test_activity_interval_known_answers_and_determinism():
    c = ACT.activity_interval(np.full(27, 3.0))
    assert c == {"mean": 3.0, "lb": 3.0, "ub": 3.0, "weeks": 27}                          # no spread: the bound is the mean
    x = np.array([0.0, 4.0] * 13 + [2.0])                                                 # mean 2.0 with spread
    a, b = ACT.activity_interval(x), ACT.activity_interval(x)
    assert a == b and a["mean"] == pytest.approx(2.0) and a["lb"] < 2.0 < a["ub"]         # same seed -> same interval; fails the rule
    ok = ACT.activity_interval(np.array([3, 3, 2, 4, 3, 3, 4, 2, 3, 3] * 3, float))        # mean 3.0, small spread
    assert ok["mean"] == pytest.approx(3.0) and ok["lb"] >= 2.0
    # known answer: the bootstrap mean's spread ~ sd / sqrt(n); the 90 % bound ~ mean - 1.645 sd / sqrt(n)
    rng = np.random.default_rng(1)
    y = rng.poisson(3.0, 400).astype(float)
    k = ACT.activity_interval(y)
    assert k["lb"] == pytest.approx(y.mean() - 1.645 * y.std() / math.sqrt(len(y)), abs=0.03)
    e = ACT.activity_interval(np.zeros(0))
    assert math.isnan(e["lb"]) and e["weeks"] == 0


def _trade(ts):
    return {"entry_time": pd.Timestamp(ts, tz="UTC"), "net_usd": 1.0, "units": 1000, "entry_bar_abs": 1}


def test_weekly_entries_uses_the_engine_weeks_and_counts_empty_decision_weeks():
    tr = [_trade("2022-01-03 10:00"), _trade("2022-01-04 10:00"), _trade("2022-01-12 10:00")]   # ISO weeks 1, 1, 2
    w = ACT.weekly_entries({"weeks_with_decisions": 4, "n_entries": 3}, tr)
    assert sorted(w.tolist()) == [0.0, 0.0, 1.0, 2.0] and w.sum() == 3
    w2 = ACT.weekly_entries({"weeks_with_decisions": 4, "n_entries": 4}, tr)               # an entry without its trade record
    assert w2.sum() == 4 and len(w2) == 4


def window(opw, weeks, trades, nl, ns, ret=None):
    rng = np.random.default_rng(2)
    eq = 10_000.0 * np.cumprod(1 + (rng.normal(0.001, 0.01, 182) if ret is None else ret))
    return {"daily_equity": eq, "sharpe_daily_sqrt365": 1.0, "net_profit_usd": 300.0, "max_drawdown_pct": 4.0, "orders_per_week": opw,
            "weeks_with_decisions": weeks, "n_entries": len(trades), "n_entries_long": nl, "n_entries_short": ns}


def trades_by_week(counts):
    out = []
    for i, n in enumerate(counts):
        day = pd.Timestamp("2022-01-03", tz="UTC") + pd.Timedelta(weeks=i)
        out += [{"entry_time": day + pd.Timedelta(hours=j), "net_usd": 1.0, "units": 1000, "entry_bar_abs": 1} for j in range(int(n))]
    return out, np.array([np.nan, 1e-3])


def test_window_components_activity_ci_and_side_cap():
    tr, sig = trades_by_week([0, 4] * 13 + [2])                                           # 54 trades / 27 weeks = exactly 2.0
    base = ACT.window_components(window(2.0, 27, tr, 27, 27), tr, sig, 10_000.0)
    assert base["valid"] and "opw_ci_lb" not in base and "side_max_share" not in base     # v1.2 scoring unchanged (no new keys)
    c = ACT.window_components(window(2.0, 27, tr, 27, 27), tr, sig, 10_000.0, activity_ci=True, side_cap=0.9)
    assert not c["valid"] and "activity_ci90_lower_bound_below_2" in c["reasons"]         # on the boundary with spread: not met
    assert c["opw_ci_mean"] == pytest.approx(2.0) and c["opw_ci_lb"] < 2.0 and c["violation"] == pytest.approx(2.0 - c["opw_ci_lb"])
    tr3, sig3 = trades_by_week([3, 3, 2, 4, 3, 3, 4, 2, 3] * 3)                             # 81 / 27 = 3.0, small spread
    ok = ACT.window_components(window(3.0, 27, tr3, 41, 40), tr3, sig3, 10_000.0, activity_ci=True, side_cap=0.9)
    assert ok["valid"] and ok["opw_ci_lb"] >= 2.0 and ok["side_max_share"] == pytest.approx(41 / 81) and ok["violation"] == 0.0
    one = ACT.window_components(window(3.0, 27, tr3, 77, 4), tr3, sig3, 10_000.0, activity_ci=True, side_cap=0.9)
    assert not one["valid"] and "one_side_over_90pct" in one["reasons"] and one["violation"] == pytest.approx(77 / 81 - 0.9)


# ----------------------------------------------------------------------------- per-side genes: a dominant side no longer takes everything
N = 600


def dominant_table():
    """pL systematically above pS (the v12 one-sided failure): calibrated long scores 0.4-0.9, short scores 0.01-0.05."""
    rng = np.random.default_rng(3)
    t = {f: np.full(N, np.nan) for f in ACT.FIELDS}
    t["pL"], t["pS"] = rng.uniform(0.4, 0.9, N), rng.uniform(0.01, 0.05, N)
    for k in ("tpL", "tpS"):
        t[k] = np.full(N, 1.5)
    for k in ("slL", "slS"):
        t[k] = np.full(N, 1.0)
    t["cost"] = np.full(N, 0.1)
    t["pN"] = rng.uniform(0.0, 0.5, N)
    t["pCL"], t["pCS"] = rng.uniform(0.0, 1.0, N), rng.uniform(0.0, 1.0, N)
    return t


def side_counts(view, s, rows):
    pol = ACT.ContractPolicy(view, s)
    d = [pol.open_decision(int(r)) for r in rows]
    return sum(1 for x in d if x is not None and x[0] == ACT.E.LONG), sum(1 for x in d if x is not None and x[0] == ACT.E.SHORT)


def test_per_side_genes_open_both_sides_where_the_shared_quantile_opened_one():
    t, rows = dominant_table(), np.arange(N)
    g12 = ACT.v1_gene_map(t, rows).to_absolute({"open_rate": 0.2, "gain_quantile": 0.0, "close_rate": 0.3, "allow_reverse": 0.0, "nop_veto_rate": 0.0})
    nl, ns = side_counts(t, ACT.Strategy.from_genes(g12, False), rows)
    assert nl > 0 and ns == 0                                                              # v1.2: one shared quantile -> longs only
    g13 = ACT.v13_gene_map(t, rows).to_absolute({"open_rate_l": 0.1, "open_rate_s": 0.1, "gain_quantile_l": 0.0, "gain_quantile_s": 0.0,
                                                  "close_rate": 0.3, "allow_reverse": 0.0, "nop_veto_rate": 0.0})
    s13 = ACT.StrategyPS.from_genes(g13, False)
    nl, ns = side_counts(t, s13, rows)
    assert nl > 0 and ns > 0                                                               # v1.3: each side at its own rate
    assert ACT.strategy_from_dict(s13.to_dict()) == s13 and isinstance(ACT.strategy_from_dict(ACT.Strategy.from_genes(g12, False).to_dict()), ACT.Strategy)


def v13_small():
    d = json.loads((CONTRACTS / "ACTION_CONTRACT_v1_3.json").read_text())
    d["strategy"]["optimizer"]["final"] = {"population": 8, "generations": 2}
    d["strategy"]["optimizer"]["final_escalated"] = {"population": 12, "generations": 3}
    return d


def side_score(view, s):
    """A stand-in for the replay that applies the v1.3 rules from per-side entry counts (120 rows = one week)."""
    rows = np.flatnonzero(np.isfinite(view["pL"]))
    nl, ns = side_counts(view, s, rows)
    n = nl + ns
    opw = min(n / (len(rows) / 120.0), 5.0)
    lb = opw - 0.3
    share = max(nl, ns) / n if n else float("nan")
    reasons = ([] if n else ["zero_trades"]) + (["activity_ci90_lower_bound_below_2"] if lb < 2.0 else []) + (["one_side_over_90pct"] if n and share > 0.9 else [])
    sh = 1.0 + 0.001 * n if n else float("nan")
    return {"sharpe": sh, "se_lo": 0.5, "lcb": sh - 0.5 if n else float("nan"), "dd": 5.0, "opw": opw, "opd": opw / 5, "r_ann": 0.05, "trade_t": 1.0,
            "trade_mean_sigma": 0.1, "trade_tstat_reused": 1.0, "calmar": 1.0, "return_minus_dd": 0.01, "trades": n, "days": 30, "weeks_below_2": 0,
            "weeks_below_2_share": 0.0, "agent_closes": 3 if n else 0, "opw_ci_lb": lb, "side_max_share": share, "long": nl, "short": ns,
            "valid": not reasons, "violation": max(0.0, 2.0 - lb) + (max(0.0, share - 0.9) if n else 0.0), "reasons": reasons}


def test_v13_freeze_respects_the_side_cap_and_probes_the_side_balance_corner():
    t, rows = dominant_table(), np.arange(N)
    fr = ACT.freeze_strategy(t, rows, side_score, v13_small(), 1, "final", "sharpe_lcb")
    ch = fr["chosen"]
    assert fr["valid"] and ch["side_max_share"] <= 0.9 and ch["opw_ci_lb"] >= 2.0 and ch["long"] > 0 and ch["short"] > 0
    assert set(fr["strategy"]) == {"open_threshold_l", "open_threshold_s", "gain_margin_l", "gain_margin_s", "close_threshold", "allow_reverse", "nop_veto"}
    pts = [p["genes"] for p in fr["feasibility_probe"]["points"]]
    assert len(pts) == 2 ** 5 + 2 and pts[-1]["open_rate_l"] == pts[-1]["open_rate_s"] == 0.30          # the side-balance corner
    assert "VALID" in fr["feasibility_probe"]["rule"] and fr["activity_rule"] == ACT.ACTIVITY_RULE_TEXT
    allp = fr["runs"]["without_veto"]["points"] + fr["runs"]["with_veto"]["points"]
    assert any(not p["valid"] and "one_side_over_90pct" in p["reasons"] for p in allp)                  # violating points archived
    assert all(p["side_max_share"] <= 0.9 for p in fr["runs"]["without_veto"]["front_early_close"] + fr["runs"]["with_veto"]["front_early_close"])


def test_front_too_small_escalates_the_budget_once_and_is_reported():
    t, rows = dominant_table(), np.arange(N)
    fr = ACT.freeze_strategy(t, rows, side_score, v13_small(), 1, "final", "sharpe_lcb")   # a tiny budget leaves a small front
    dv = fr["fitness_diversity"]
    assert dv["budget_escalation"] is not None and dv["budget_escalation"]["from"] == {"population": 8, "generations": 2}
    assert dv["budget_escalation"]["to"] == {"population": 12, "generations": 3} and "FRONT_TOO_SMALL" in dv["budget_escalation"]["reason"]
    assert dv["selectable_front_points"] == dv["budget_escalation"]["front_after"]
    assert dv["flag"] == ("FRONT_TOO_SMALL" if dv["selectable_front_points"] < ACT.FRONT_MIN else None)
    assert 1 <= dv["distinct_points_chosen_across_variants"] <= dv["variants"] == 7
    big = v13_small()
    big["strategy"]["optimizer"]["final"] = {"population": 24, "generations": 8}
    fb = ACT.freeze_strategy(t, rows, side_score, big, 1, "final", "sharpe_lcb")
    if fb["fitness_diversity"]["selectable_front_points"] >= ACT.FRONT_MIN:               # a large enough front is never escalated
        assert fb["fitness_diversity"]["budget_escalation"] is None and fb["fitness_diversity"]["flag"] is None


def test_infeasible_v13_space_is_refused():
    t, rows = dominant_table(), np.arange(N)
    one_sided = lambda view, s: dict(side_score(view, s), valid=False, reasons=["one_side_over_90pct"])  # noqa: E731
    with pytest.raises(SystemExit, match="STRATEGY_SPACE_INFEASIBLE"):
        ACT.freeze_strategy(t, rows, one_sided, v13_small(), 1, "final", "sharpe_lcb")


def test_v12_strategy_doc_has_no_v13_keys():
    from tests.test_expert_guided_sanity_and_scale_adaptive import score_from_view, small, tiny_table
    v12 = small(json.loads((CONTRACTS / "ACTION_CONTRACT_v1_2.json").read_text()))
    b = ACT.freeze_strategy(tiny_table(), np.arange(400), score_from_view, v12, 1, "final", "sharpe_lcb")
    assert "fitness_diversity" not in b and "activity_rule" not in b and ACT.score_kwargs(v12) == {}


# ----------------------------------------------------------------------------- contracts
def test_v13_contracts_validate_and_refuse_partial_rules():
    from app.short_horizon.expert_guided import c2 as C2
    v = json.loads((CONTRACTS / "ACTION_CONTRACT_v1_3.json").read_text())
    assert ACT.validate(v) == [] and ACT.is_v13(v["strategy"]) and ACT.score_kwargs(v) == {"activity_ci": True, "side_cap": 0.9}
    for f in ("ACTION_CONTRACT_C2_v1_3.json", "ACTION_CONTRACT_C2G_v1_3.json"):
        d = json.loads((CONTRACTS / f).read_text())
        assert C2.validate(d) == [] and ACT.is_v13(d["strategy"]) and tuple(d["strategy"]["genes"]) == C2.GENES_Q13
    for mut, code in ((lambda d: d["strategy"].__setitem__("side_cap", 0.95), "V13_SIDE_CAP"),
                      (lambda d: d["strategy"].pop("activity_constraint"), "V13_PER_SIDE_GENES_WITHOUT_RULES"),
                      (lambda d: d["strategy"]["optimizer"].__setitem__("final_escalated", {"population": 8, "generations": 2}), "V13_ESCALATED_BUDGET"),
                      (lambda d: d["strategy"]["activity_ci"].__setitem__("draws", 100), "V13_ACTIVITY_CI")):
        x = copy.deepcopy(v)
        mut(x)
        x["identity"] = None
        assert any(code in e for e in ACT.validate(ACT.seal(x))), code


def test_c2_per_side_margins():
    from app.short_horizon.expert_guided import c2 as C2
    s = C2.StrategyC2PS(0.5, -0.5, 0.1, 0.9, False, None)
    t = {"net": np.array([[[0.2, 0.1], [0.0, -0.2]]]), "pn": np.array([0.0])}               # long net 0.2 < 0.5; short net 0.0 > -0.5
    o = C2.PolicyC2(t, s, cells=[(1.0, 1.0), (2.0, 2.0)]).open_decision(0)
    assert o is not None and o[0] == C2.E.SHORT
    assert C2.strategy_from_dict(s.to_dict()) == s and isinstance(C2.strategy_from_dict(C2.NO_TRADE.to_dict()), C2.StrategyC2)


# ----------------------------------------------------------------------------- reports
def test_separability_refusal_renders_without_keyerror_and_failed_units_show_informational_metrics(tmp_path):
    from tools import expert_guided_wf_report as R
    ref = {"verdict": "REFUSED_FAILED_SANITY", "failed_units": ["U"], "causes": [{"below_2_trades_per_week": 1}]}
    line = R.sep_line("wf2022/a_vs_b", ref)
    assert "REFUSED_FAILED_SANITY" in line and "below_2_trades_per_week" in line
    assert "Δ +0.50" in R.sep_line("k", {"delta": 0.5, "pooled_se": 1.0, "verdict": "NOT_SEPARABLE"})
    (tmp_path / "U").mkdir()
    (tmp_path / "U" / "SANITY.json").write_text(json.dumps({"status": "FAILED_SANITY", "ensemble": {"INNER_B": {
        "trades": 87, "orders_per_week": 1.64, "sharpe": 0.34, "agent_closes": 64, "side": {"long": 39, "short": 48}}}}))
    md = "\n".join(R.md_sanity([{"unit": "U", "sanity": "FAILED_SANITY", "sanity_causes": {"below_2": 1}}], tmp_path))
    assert "1.64/wk" in md and "NOT a result" in md and "39/48" in md


# ----------------------------------------------------------------------------- spec and launcher
def test_spec_v10_preregisters_the_v13_stages_and_emits_them(tmp_path):
    pytest.importorskip("torch")
    from tools import expert_guided_clean_round as C
    s = json.loads((ROOT / "docs/short_horizon/clean_round/CLEAN_ROUND_SPEC.json").read_text())
    assert s["identity"].endswith("-v11") and any(r.get("identity", "").endswith("-v10") for r in s["revisions"]) and s["superseded"]["WF_V12_"] == "SUPERSEDED_ONE_SIDED_AND_BOUNDARY_ACTIVITY"
    for k, f in (("walk_forward_v13", "ACTION_CONTRACT_v1_3.json"), ("walk_forward_c2_13", "ACTION_CONTRACT_C2_v1_3.json"),
                 ("walk_forward_c2g_13", "ACTION_CONTRACT_C2G_v1_3.json")):
        assert s[k]["action_contract"]["sha256"] == ACT.sha256_file(CONTRACTS / f) and "Con confianza estadística" in s[k]["activity_rule"]
    assert C.walk_forward_v12_errors(s, "walk_forward_v13", C.WF_V13_STAGE) == []
    assert C.walk_forward_c2_errors(s, "walk_forward_c2_13", C.WF_C2_13_STAGE) == []
    assert C.walk_forward_c2g_errors(s, "walk_forward_c2g_13", C.WF_C2G_13_STAGE, "walk_forward_c2_13", C.WF_C2_13_STAGE) == []
    x = copy.deepcopy(s)
    x["walk_forward_v13"]["action_contract"] = dict(s["walk_forward_v12"]["action_contract"])
    assert "WF_V13_CONTRACT_NOT_V13" in C.walk_forward_v12_errors(x, "walk_forward_v13", C.WF_V13_STAGE)
    e = copy.deepcopy(s)
    e["remote_root"] = "/r"
    for stage, key, emit, n in ((C.WF_V13_STAGE, "walk_forward_v13", C.emit_walk_forward_v1, 4), (C.WF_C2_13_STAGE, "walk_forward_c2_13", C.emit_walk_forward_c2, 2),
                                (C.WF_C2G_13_STAGE, "walk_forward_c2g_13", C.emit_walk_forward_c2, 2)):
        out = tmp_path / f"{stage}.sh"
        emit(e, types.SimpleNamespace(out=str(out)), key, stage)
        q = out.read_text()
        assert q.count(f"if [ ! -f $Q/STATE/{stage}_wf20") == n and f'log "STAGE_{stage}_DONE"' in q
        assert Path(e[key]["action_contract"]["path"]).name in q


# ----------------------------------------------------------------------------- the chain's first-seed / first-ensemble stop rule
def _run(args):
    import subprocess
    import sys
    return subprocess.run([sys.executable, str(ROOT / "tools/clean_round_chain/first_seed_check.py")] + args, capture_output=True, text=True)


def _ss(trades=79, opw=2.9, long=38, short=41, agent=6, exits=None):
    return {"trades": trades, "opw": opw, "long": long, "short": short, "agent_closes": agent,
            "exits": exits or {"TP": 20, "SL": 10, "FORCED_FLAT": 3, "AGENT_CLOSE": agent, "REVERSAL": 0}}


@pytest.mark.parametrize("bad, cause", [({"trades": 0, "opw": 0.0}, "zero_trades"), ({"opw": 1.9}, "below_2_trades_per_week"),
                                        ({"long": 80, "short": 0}, "one_side_over_90pct"), ({"agent": 0}, "zero_agent_closes"),
                                        ({"exits": {"TP": 0, "SL": 0, "FORCED_FLAT": 0, "AGENT_CLOSE": 79, "REVERSAL": 0}}, "single_exit_reason")])
def test_first_seed_check_passes_a_sane_seed_and_stops_on_each_cause(tmp_path, bad, cause):
    log = tmp_path / "u.log"
    log.write_text("noise\n" + json.dumps({"seed": 1, "seed_sanity": {"INNER_A": _ss(), "INNER_B": _ss()}}) + "\n")
    r = _run(["seed", "--log", str(log), "--unit", "U", "--poll", "0"])
    assert r.returncode == 0 and json.loads(r.stdout)["pass"] is True
    log.write_text(json.dumps({"seed": 1, "seed_sanity": {"INNER_A": _ss(), "INNER_B": _ss(**bad)}}) + "\n")
    r = _run(["seed", "--log", str(log), "--unit", "U", "--poll", "0"])
    assert r.returncode == 3 and cause in json.loads(r.stdout)["causes"]["INNER_B"]
    log.write_text("Traceback: STRATEGY_SPACE_INFEASIBLE\n")
    r = _run(["seed", "--log", str(log), "--unit", "U", "--pid", "999999999", "--poll", "0"])
    assert r.returncode == 4
    sj = tmp_path / "SANITY.json"
    sj.write_text(json.dumps({"causes": [{"cause": "below_2_trades_per_week", "where": "ensemble", "block": "INNER_B"}]}))
    assert _run(["ensemble", "--sanity", str(sj), "--unit", "U"]).returncode == 3
    sj.write_text(json.dumps({"causes": [{"cause": "below_2_trades_per_week", "where": "seed 2", "block": "INNER_B"}]}))
    assert _run(["ensemble", "--sanity", str(sj), "--unit", "U"]).returncode == 0

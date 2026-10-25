"""Output contract v1 (owner 2026-10-25): the pinned contract, its typed refusals, Platt calibration with a known answer,
equal-mass reliability, AUC, the sigma units of the cost, the strategy's decisions and its lower-confidence-bound choice,
the NOP labels' consistency with the openers, the seed-ensemble averaging, additivity of every existing mode, and the
NO-PEEK mutation: poisoning the decision block (INNER_B) leaves the best epochs, the Platt maps, the visitation
strategies and the frozen strategy identical."""
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from app.short_horizon.expert_guided import action_contract as ACT
from app.short_horizon.expert_guided import data as DG
from app.short_horizon.expert_guided import env as E

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "short_horizon" / "clean_round" / "contracts" / "ACTION_CONTRACT_v1.json"


def contract():
    return json.loads(CONTRACT.read_text())


# ----------------------------------------------------------------------------- the contract
def test_pinned_contract_validates_and_its_identity_covers_every_field():
    doc = contract()
    assert ACT.validate(doc) == [] and doc["identity"] == ACT.identity_of(doc)
    assert tuple(doc["specialists"]) == ACT.SPECIALISTS
    cfg = E.EnvConfig()
    for m in ACT.OPENERS:                                     # TP/SL bounds are the env's own
        ct = doc["specialists"][m]["continuous"]
        assert (ct["tp_sigma"]["lo"], ct["tp_sigma"]["hi"]) == cfg.tp_bounds and (ct["sl_sigma"]["lo"], ct["sl_sigma"]["hi"]) == cfg.sl_bounds
    assert ACT.encoding_audit(doc)["verdict"] == "PASS"
    bad = copy.deepcopy(doc)
    bad["strategy"]["gain_note"] = "x"                                  # any edit without resealing breaks the identity
    assert "ACTION_CONTRACT_IDENTITY" in ACT.validate(bad)


@pytest.mark.parametrize("mut, code", [
    (lambda d: d["specialists"].pop("nop"), "ACTION_CONTRACT_SPECIALISTS"),
    (lambda d: d["specialists"]["open_l"]["continuous"]["tp_sigma"].update(lo=0.1, hi=5.0), "ACTION_CONTRACT_TP_SL_BOUNDS"),
    (lambda d: d["specialists"]["open_s"]["continuous"]["sl_sigma"].update(mask="every row"), "ACTION_CONTRACT_TP_SL_MASK"),
    (lambda d: d["calibration"].update(method="temperature"), "ACTION_CONTRACT_CALIBRATION"),
    (lambda d: d["calibration"].update(fit_block="INNER_B"), "ACTION_CONTRACT_CALIBRATION"),
    (lambda d: d["strategy"].update(fit_block="INNER_B"), "ACTION_CONTRACT_STRATEGY_BLOCK"),
    (lambda d: d["strategy"].update(calibration_min_trades_per_week=2.5), "ACTION_CONTRACT_ACTIVITY_RULE"),
    (lambda d: d["strategy"]["optimizer"].update(constraint="penalty"), "ACTION_CONTRACT_CONSTRAINT_HANDLING"),
    (lambda d: d["strategy"].update(choice="highest Sharpe"), "ACTION_CONTRACT_CHOICE"),
    (lambda d: d["strategy"]["optimizer"]["final"].update(population=10), "ACTION_CONTRACT_OPTIMIZER"),
    (lambda d: d["blocks"].update(decision="INNER_A"), "ACTION_CONTRACT_BLOCKS"),
    (lambda d: d["blocks"].update(calibration_block_uses=["calibration"]), "ACTION_CONTRACT_BLOCKS"),
    (lambda d: d["constraints"].update(agent_state="full"), "ACTION_CONTRACT_CONSTRAINTS"),
])
def test_each_contract_mutation_is_refused_even_when_resealed(mut, code):
    d = contract()
    mut(d)
    d = ACT.seal(d)
    assert any(e.startswith(code) for e in ACT.validate(d)), ACT.validate(d)


def test_a_split_without_a_post_freeze_decision_block_is_refused():
    doc = contract()
    with pytest.raises(SystemExit, match="ACTION_CONTRACT_NO_POST_FREEZE_DECISION_BLOCK:consolidated"):
        ACT.require_decision_block(doc, DG.split_contract("consolidated"), "consolidated")
    for name in ("wf2022", "wf2023", "dev3y"):
        ACT.require_decision_block(doc, DG.split_contract(name), name)


def test_e3m_refuses_the_contract_before_reading_any_data(tmp_path):
    pytest.importorskip("torch")
    from tools import expert_guided_e3m as M
    base = ["run", "--admitted", str(tmp_path / "nope"), "--csv-dir", str(tmp_path / "nope"), "--out", str(tmp_path / "o"),
            "--scaling", "fixed_train", "--variants", "FULL", "--action-contract", str(CONTRACT)]
    for extra, code in ((["--split-contract", "consolidated", "--agent-state", "none"], "ACTION_CONTRACT_NO_POST_FREEZE_DECISION_BLOCK"),
                        (["--split-contract", "wf2022", "--agent-state", "minimal"], "ACTION_CONTRACT_REQUIRES_AGENT_STATE_NONE"),
                        (["--split-contract", "wf2022", "--agent-state", "none", "--aux-targets"], "ACTION_CONTRACT_V1_REFUSES")):
        with pytest.raises(SystemExit, match=code):
            M.run(M.build_parser().parse_args(base + extra))


# ----------------------------------------------------------------------------- calibration and metrics
def test_platt_recovers_a_known_affine_map_and_a_temperature_alone_cannot():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 200_000)
    y = (rng.random(len(z)) < ACT.sigmoid(0.5 * z + 0.8)).astype(int)        # prior shift b = 0.8
    a, b = ACT.fit_platt(z, y)
    assert abs(a - 0.5) < 0.02 and abs(b - 0.8) < 0.02
    ll = lambda p: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))        # noqa: E731
    best_temp = min(ll(ACT.sigmoid(z / T)) for T in np.linspace(0.5, 4.0, 141))
    assert ll(ACT.platt(z, a, b)) < best_temp - 0.01                          # the bias is what removes the shift
    assert ACT.fit_platt(z[:10], np.ones(10)) == (1.0, 0.0)                   # one class: nothing to calibrate


def test_reliability_uses_15_equal_mass_bins_and_auc_known_answers():
    rng = np.random.default_rng(1)
    p = rng.random(30_000)
    y = (rng.random(len(p)) < p).astype(int)                                  # perfectly calibrated
    r = ACT.reliability(p, y, 15)
    assert len(r["bins"]) == 15 and max(b["n"] for b in r["bins"]) - min(b["n"] for b in r["bins"]) <= 1
    assert r["ece"] < 0.02
    assert ACT.reliability(np.full(1000, 0.9), np.zeros(1000), 15)["ece"] == pytest.approx(0.9)
    assert ACT.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0 and ACT.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert ACT.auc([0.5] * 4, [0, 1, 0, 1]) == 0.5 and np.isnan(ACT.auc([0.1, 0.2], [1, 1]))


def test_cost_terms_are_both_sigma_multiples():
    sig = np.array([np.nan, 5e-4, 4e-4, 2e-4])
    row_bar = np.array([0, 1, 2, -1])
    c = ACT.row_cost_sigma(1e-4, sig, row_bar, 0.1)
    assert c[0] == pytest.approx(0.1 + 1e-4 / 5e-4) and c[1] == pytest.approx(0.1 + 1e-4 / 4e-4) and c[2] == pytest.approx(0.1 + 1e-4 / 2e-4)
    assert np.isnan(c[3])                                                       # no fill bar -> no cost -> the row is blank
    prof = E.EnvConfig().profile
    assert prof.round_trip_price() > 0 and prof.round_trip_price() < 1e-3       # price units, same as sigma


# ----------------------------------------------------------------------------- the strategy
def table(rows=4, **over):
    t = {f: np.full(rows, np.nan) for f in ACT.FIELDS}
    base = dict(pL=0.7, tpL=1.5, slL=1.0, pS=0.2, tpS=1.0, slS=1.0, pN=0.1, pCL=0.3, pCS=0.3, cost=0.15)
    base.update(over)
    for f, v in base.items():
        t[f][:rows - 1] = v                                                     # the last row stays blank
    return t


def act(t, s, pos=0, r=0):
    st = np.zeros(len(E.STATE_NAMES), np.float32)
    st[0] = pos
    return ACT.ContractPolicy(t, s)(None)(0, r, st, None)


def test_strategy_decisions_threshold_margin_veto_close_and_reverse():
    s = ACT.Strategy(open_threshold=0.5, gain_margin=0.0, close_threshold=0.5, allow_reverse=False, nop_veto=None)
    assert act(table(), s)[0] == E.LONG                                         # gain 0.7*1.5 - 0.3*1 - 0.15 = 0.6 > 0
    assert act(table(pL=0.4), s)[0] == E.HOLD                                   # below the open threshold
    assert act(table(), ACT.Strategy(0.5, 0.7, 0.5, False, None))[0] == E.HOLD  # below the gain margin
    assert act(table(pN=0.9), ACT.Strategy(0.5, 0.0, 0.5, False, 0.8))[0] == E.HOLD          # NOP veto
    assert act(table(pN=0.9), s)[0] == E.LONG                                                # veto disabled
    assert act(table(), s, pos=1)[0] == E.HOLD                                  # p_close 0.3 <= 0.5: hold
    assert act(table(pCL=0.8), s, pos=1)[0] == E.CLOSE
    assert act(table(pCS=0.8), s, pos=-1)[0] == E.CLOSE                         # reverse not allowed
    assert act(table(pCS=0.8), ACT.Strategy(0.5, 0.0, 0.5, True, None), pos=-1)[0] == E.LONG  # close-and-reverse
    assert act(table(pCL=0.8), ACT.Strategy(0.5, 0.0, 0.5, True, None), pos=1)[0] == E.CLOSE  # same side: just close
    with pytest.raises(ValueError, match="ACTION_CONTRACT_ROW_OUTSIDE_TABLE"):
        act(table(), s, r=3)                                                    # a blank row is never read


def test_restrict_blanks_every_row_outside_the_block_and_mean_tables():
    t = table(rows=5)
    v = ACT.restrict(t, [1, 2])
    assert np.isnan(v["pL"][0]) and v["pL"][1] == t["pL"][1] and np.isnan(v["cost"][3])
    one = ACT.mean_tables([t])
    assert all(np.array_equal(one[f], t[f], equal_nan=True) for f in ACT.FIELDS)          # N = 1 is the member itself
    t2 = table(rows=5, pL=0.3, tpL=2.5)
    m = ACT.mean_tables([t, t2])
    assert m["pL"][0] == pytest.approx(0.5) and m["tpL"][0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="ENSEMBLE_COST_MISMATCH"):
        ACT.mean_tables([t, table(rows=5, cost=0.3)])


def comp(sharpe=1.0, se=0.5, dd=2.0, opw=3.0, trades=40, valid=True, **k):
    c = {"sharpe": sharpe, "se_lo": se, "lcb": sharpe - se, "dd": dd, "opw": opw, "opd": opw / 5, "r_ann": 0.1, "trade_t": 2.0, "trade_tstat_reused": 2.0,
         "trade_mean_sigma": 0.1, "calmar": 5.0, "return_minus_dd": 0.08, "trades": trades, "days": 180, "valid": valid,
         "violation": 0.0 if valid else max(0.0, 2.0 - opw) + 0.5, "reasons": [] if valid else ["below_2_trades_per_week"]}
    c.update(k)
    return c


def test_optimizer_searches_sharpe_dd_and_each_fitness_picks_on_the_valid_front_deterministically():
    c = contract()
    c["strategy"]["optimizer"]["final"] = {"population": 16, "generations": 8}

    def score(view, s):                                                         # synthetic surface: Sharpe high at large thresholds,
        sh = 3.0 * s.open_threshold                                             # but the SE grows faster there (few trades),
        se = 0.2 + 4.0 * s.open_threshold ** 3                                  # and above 0.8 the activity margin fails
        return comp(sharpe=sh, se=se, dd=5.0 - s.close_threshold, valid=s.open_threshold < 0.8, r_ann=s.gain_margin, opw=2.5 + s.allow_reverse)

    random.seed(123)
    before = random.random()
    random.seed(123)
    t = table(rows=6)
    f1 = ACT.freeze_strategy(t, [0, 1, 2], score, c, seed=7)
    assert random.random() == before                                            # the global random state is restored
    f2 = ACT.freeze_strategy(t, [0, 1, 2], score, c, seed=7)
    assert f1["by_fitness"] == f2["by_fitness"]
    run = f1["runs"]["without_veto"]
    assert run["objectives"] == ["sharpe max", "max_drawdown_pct min"] and set(run["chosen"]) == set(ACT.FITNESS)
    assert run["evaluations"] == 16 * (8 + 1) and len(run["evaluation_log"]) == run["evaluations"]          # every evaluation, all generations
    assert any(not x["valid"] for x in run["points"]) and all(set(x["fitness"]) == set(ACT.FITNESS) for x in run["points"])  # infeasible points kept
    assert 0 < run["share_valid"] < 1
    for f in ACT.FITNESS:                                                       # the archive view of every variant
        av = f1["by_fitness"][f]["archive_view"]
        assert av["best_sharpe_unconstrained"]["sharpe"] >= max(x["sharpe"] for x in run["points"] if x["valid"])
        assert "NOT SELECTABLE" in av["note"]
    front = run["front"]
    assert all(x["valid"] for x in front)
    for f in ACT.FITNESS:                                                       # every variant's pick is its best on the front
        assert run["chosen"][f]["fitness"][f] == max(x["fitness"][f] for x in front)
    assert run["chosen"]["sharpe_lcb"]["genes"]["open_threshold"] < 0.8 and "nop_veto" not in run["chosen"]["sharpe_lcb"]["genes"]
    for x in front:                                                             # components are recorded for every front point
        assert {"sharpe", "se_lo", "lcb", "r_ann", "opd", "trade_tstat_reused", "calmar", "return_minus_dd", "dd", "opw", "trades"} <= set(x)
        assert set(x["fitness"]) == set(ACT.FITNESS)
    assert set(f1["term_scales"]) == set(ACT.SCALE_TERMS)


def test_composite_scales_bring_every_term_to_median_magnitude_one_on_the_reference_front():
    rng = np.random.default_rng(8)
    ref = [comp(sharpe=float(a), r_ann=float(b), opw=float(o), dd=float(d)) for a, b, o, d in
           zip(rng.normal(2, 1, 41), rng.normal(0.3, 0.2, 41), rng.uniform(2.5, 5, 41), rng.uniform(1, 9, 41))]
    ref.append(comp(sharpe=99.0, valid=False))                                  # invalid points never set a scale
    sc = ACT.term_scales(ref)
    valid = [p for p in ref if p["valid"]]
    for t, f in ACT.SCALE_TERMS.items():
        assert not sc[t]["fallback_used"] and sc[t]["c"] == pytest.approx(1 / sc[t]["s"])
        assert abs(float(np.median([abs(f(p)) * sc[t]["c"] for p in valid])) - 1.0) < 1e-9
    p = valid[0]
    assert ACT.fitness("balanced", p, sc) == pytest.approx(p["sharpe"] / sc["sharpe"]["s"] + p["r_ann"] / sc["r_ann"]["s"] + p["opd"] / sc["opd"]["s"] - p["dd"] / sc["dd"]["s"])
    assert ACT.fitness("sharpe_plus_return", p, sc) == pytest.approx(p["sharpe"] / sc["sharpe"]["s"] + p["r_ann"] / sc["r_ann"]["s"])
    zero = ACT.term_scales([comp(sharpe=0.0, r_ann=0.0, opw=0.0, dd=0.0)])      # below the floor: declared fallback, recorded
    assert all(zero[t]["fallback_used"] and zero[t]["s"] == ACT.FALLBACK_SCALES[t] for t in ACT.SCALE_TERMS)
    with pytest.raises(ValueError, match="FITNESS_SCALES_REQUIRED"):
        ACT.fitness("balanced", p)


def test_reused_fitnesses_match_their_sources():
    from app.walk_forward_optimizer import _sharpe_fitness
    rng = np.random.default_rng(9)
    r = rng.normal(0.2, 1.0, 40)
    assert ACT.trade_tstat(r) == _sharpe_fitness(None, [{"pnl": float(x)} for x in r], min_trades=15)
    assert ACT.trade_tstat(r) == pytest.approx(np.mean(r) / np.std(r) * np.sqrt(40))
    assert math.isnan(ACT.trade_tstat(r[:14]))                                 # rejected below 15 trades (source: -1e6): undefined, kept
    assert ACT.return_minus_dd(0.12, 8.0) == pytest.approx(0.12 - 1.0 * 8.0 / 100.0)   # agent-multi sac_agent.fitness, lambda 1
    assert ACT.DD_LAMBDA == 1.0 and ACT.TRADE_TSTAT_MIN_TRADES == 15


def test_freeze_keeps_the_nop_veto_only_when_it_strictly_improves_the_lower_bound():
    c = contract()
    c["strategy"]["optimizer"]["final"] = {"population": 8, "generations": 3}
    t = table(rows=6)
    fr = ACT.freeze_strategy(t, [0, 1, 2], lambda view, s: comp(), c, seed=1)
    assert fr["nop_veto_kept"] is False and fr["strategy"]["nop_veto"] is None   # a tie never keeps the veto
    assert set(fr["by_fitness"]) == set(ACT.FITNESS) and fr["primary_fitness"] == "sharpe_lcb"

    def veto_helps(view, s):
        lcb = 0.5 + (0.3 if s.nop_veto is not None and s.nop_veto < 0.6 else 0.0)
        return comp(sharpe=lcb + 0.5, se=0.5)

    fr = ACT.freeze_strategy(t, [0, 1, 2], veto_helps, c, seed=1)
    assert fr["nop_veto_kept"] is True and fr["strategy"]["nop_veto"] < 0.6
    fr = ACT.freeze_strategy(t, [0, 1, 2], lambda view, s: comp(opw=1.0, valid=False), c, seed=1)
    assert fr["strategy"] is None and fr["valid"] is False and "INVALID" in fr["note"]
    assert fr["runs"]["without_veto"]["points"] and fr["by_fitness"]["sharpe_lcb"]["archive_view"]["best_unconstrained_by_fitness"] is not None
    with pytest.raises(ValueError, match="STRATEGY_FITNESS_UNKNOWN"):
        ACT.freeze_strategy(t, [0, 1, 2], lambda view, s: comp(), c, seed=1, primary="profit")


# ----------------------------------------------------------------------------- fitness variants (known answers)
def window(net=500.0, days=182, sharpe=1.2, dd=4.0, opw=3.0, valid=True, ret=None):
    eq = np.full(days, 10_000.0 + net)
    if ret is not None:
        eq = 10_000.0 * np.cumprod(1 + np.asarray(ret))
    return {"daily_equity": eq, "sharpe_daily_sqrt365": sharpe, "net_profit_usd": net, "max_drawdown_pct": dd, "orders_per_week": opw, "valid": valid}


def trades_sigma(rets, units=1000, sigma=1e-3):
    """Trades whose per-unit net result / sigma of the fill bar equals ``rets`` (sigma multiples)."""
    return [{"net_usd": r * sigma * units, "units": units, "entry_bar_abs": 1} for r in rets], np.array([np.nan, sigma, np.nan])


def test_each_fitness_has_its_known_answer():
    rng = np.random.default_rng(5)
    tr, sig = trades_sigma([1.0, 2.0, 3.0] * 5)                                # 15 trades: F3's minimum
    c = ACT.window_components(window(ret=rng.normal(0.001, 0.01, 182), net=500.0, sharpe=1.2), tr, sig, 10_000.0)
    assert c["valid"] and c["trades"] == 15 and c["days"] == 182
    assert c["r_ann"] == pytest.approx(0.05 * 365 / 182)                         # R annualized to the block length
    assert c["opd"] == pytest.approx(3.0 / 5)
    x = np.array([1.0, 2.0, 3.0] * 5)
    assert c["trade_t"] == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(15))
    assert c["trade_tstat_reused"] == pytest.approx(x.mean() / x.std() * np.sqrt(15))  # the reused source: ddof 0
    assert c["calmar"] == pytest.approx(100 * c["r_ann"] / 4.0)
    unit = {t: {"s": 1.0, "c": 1.0} for t in ACT.SCALE_TERMS}                    # unit scales: the composites reduce to the raw sums
    assert ACT.fitness("sharpe_lcb", c) == pytest.approx(1.2 - c["se_lo"])
    assert ACT.fitness("sharpe_plus_return", c, unit) == pytest.approx(1.2 + c["r_ann"])
    assert ACT.fitness("sharpe_plus_return_plus_activity", c, unit) == pytest.approx(1.2 + c["r_ann"] + 0.6)
    assert ACT.fitness("balanced", c, unit) == pytest.approx(1.2 + c["r_ann"] + 0.6 - 4.0)
    assert ACT.fitness("trade_tstat", c) == pytest.approx(c["trade_tstat_reused"])
    assert ACT.fitness("calmar", c) == pytest.approx(c["calmar"])
    assert ACT.fitness("return_minus_dd", c) == pytest.approx(c["r_ann"] - 0.04)


def test_zero_trades_undefined_sharpe_and_low_activity_are_kept_infeasible_never_zero_or_minus_inf():
    rng = np.random.default_rng(6)
    ret = rng.normal(0, 0.01, 100)
    unit = {t: {"s": 1.0} for t in ACT.SCALE_TERMS}
    z = ACT.window_components(window(ret=ret, opw=0.0), [], np.array([1e-3]), 10_000.0)                       # zero trades
    assert not z["valid"] and z["trades"] == 0 and math.isnan(z["sharpe"]) and "zero_trades" in z["reasons"] and z["violation"] == pytest.approx(2.0)
    assert math.isnan(ACT.fitness("sharpe_lcb", z)) and math.isnan(ACT.fitness("balanced", z, unit))          # NaN, never 0 / -inf
    u = ACT.window_components(window(ret=ret, sharpe=float("nan")), *trades_sigma([1.0, 2.0]), 10_000.0)     # undefined Sharpe
    assert not u["valid"] and "sharpe_undefined" in u["reasons"] and u["violation"] == pytest.approx(1.0)
    lo = ACT.window_components(window(ret=ret, opw=1.6), *trades_sigma([1.0, 2.0]), 10_000.0)                # below exactly 2.0
    assert not lo["valid"] and lo["violation"] == pytest.approx(0.4) and math.isfinite(ACT.fitness("sharpe_lcb", lo))   # value recorded anyway
    ok = ACT.window_components(window(ret=ret, opw=2.0), *trades_sigma([1.0, 2.0]), 10_000.0)                # exactly 2.0: feasible, no margin
    assert ok["valid"] and ok["violation"] == 0.0
    c = ACT.window_components(window(ret=ret, dd=0.0), *trades_sigma([1.0, 2.0]), 10_000.0)                  # N = 2 < 15, DD = 0
    assert c["valid"] and math.isnan(ACT.fitness("trade_tstat", c)) and math.isnan(ACT.fitness("calmar", c))
    assert math.isfinite(ACT.fitness("sharpe_lcb", c)) and math.isfinite(ACT.fitness("return_minus_dd", c))


def test_constraint_domination_orders_feasible_then_by_violation_then_pareto():
    pts = [comp(sharpe=1.0, dd=3.0), comp(sharpe=2.0, dd=2.0), comp(sharpe=0.5, dd=1.0),                      # feasible
           comp(sharpe=9.0, dd=0.1, opw=1.9, valid=False), comp(sharpe=float("nan"), dd=0.0, opw=0.0, valid=False, violation=2.0),
           comp(sharpe=5.0, dd=0.5, opw=1.0, valid=False)]                                                       # infeasible
    enc = [ACT.search_objectives(p) for p in pts]
    pareto = lambda a, b: a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1])                       # noqa: E731
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i != j:
                assert pareto(enc[i], enc[j]) == ACT.cdominates(a, b), (i, j)
    assert ACT.cdominates(pts[2], pts[3])                                      # any feasible beats a high-Sharpe infeasible point
    assert ACT.cdominates(pts[3], pts[5]) and ACT.cdominates(pts[5], pts[4])   # smaller violation wins
    assert ACT.cdominates(pts[1], pts[0]) and not ACT.cdominates(pts[1], pts[2])
    assert all(math.isfinite(x) for e in enc for x in e)


def test_orders_per_day_is_bounded_by_the_daily_cap():
    for opw, opd in ((0.0, 0.0), (2.5, 0.5), (5.0, 1.0)):                       # 1 entry/day x 5 trading days
        assert ACT.window_components(window(opw=opw, ret=np.full(50, 0.001)), *trades_sigma([1.0, 2.0]), 10_000.0, 0.0)["opd"] == pytest.approx(opd)
    assert ACT.TRADING_DAYS_PER_WEEK == 5.0 and E.EnvConfig().max_entries_per_day >= 1


# ----------------------------------------------------------------------------- NOP labels
class FakeMR:
    def __init__(self, labs):
        self.labs = labs

    def label(self, bar, r):
        return self.labs.get(bar)


def test_nop_labels_are_consistent_with_the_openers_and_carry_their_weight():
    mr = FakeMR({0: (0, 1.0, None, "nop_hard", np.nan), 1: (0, 0.4, None, "nop_graded", 0.6), 2: (1, 0.8, 1.2, "open", 0.8),
                 3: (0, 0.7, None, "negative", -0.7), 4: (0, 1.0, None, "nop_hard", np.nan)})
    nl = ACT.NopLabels(mr)
    assert nl.label(0, 0, False) == (1, 1.0)
    assert nl.label(1, 1, False) == (1, 0.4)                                    # graded: its weight
    assert nl.label(2, 2, False) == (0, 1.0)                                    # some resolution trades well
    assert nl.label(3, 3, False) == (1, 0.7)
    assert nl.label(4, 4, True) == (0, 1.0)                                     # the arm opens: NOP = 0 whatever the ladder says
    assert nl.label(9, 9, False) is None                                        # censored: dropped
    assert nl.table["arm_open"]["nop_hard"] == 1 and nl.table["arm_no_open"]["censored"] == 1


def test_nop_information_detects_a_signal_beyond_the_openers():
    rng = np.random.default_rng(2)
    n = 4000
    y = rng.integers(0, 2, n)
    t = {f: np.full(n, np.nan) for f in ACT.FIELDS}
    t["pL"][:] = rng.random(n) * 0.5
    t["pS"][:] = rng.random(n) * 0.5                                           # openers carry no information on y
    t["pN"][:] = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.1, n), 0, 1)           # NOP does
    info = ACT.nop_information(t, np.arange(n), y)
    assert info["auc_pnop"] > 0.95 and abs(info["auc_one_minus_max_open"] - 0.5) < 0.05 and info["delta_logloss"] < -0.1
    assert sum(c["rows"] for c in info["joint_confusion"].values()) == n


# ----------------------------------------------------------------------------- network heads, additivity, end to end
def test_tp_sl_heads_are_bounded_inside_the_network():
    torch = pytest.importorskip("torch")
    from app.short_horizon.expert_guided import temporal as T
    net = T.TemporalPolicyNet(4, 13, 2, n_reg=2, n_state=9, units=2, reg_bounds=((0.25, 3.0), (0.5, 3.0)))
    x = torch.tensor(np.r_[np.full(13 * 4, 1e6), np.zeros(9)][None, :].repeat(3, 0), dtype=torch.float32)
    x[1] *= -1
    _, rg = net(x)
    assert float(rg[:, 0].min()) >= 0.25 and float(rg[:, 0].max()) <= 3.0 and float(rg[:, 1].min()) >= 0.5 and float(rg[:, 1].max()) <= 3.0


def _synthetic_v1(poison_inner_b=False, seed=20261019):
    torch = pytest.importorskip("torch")  # noqa: F841
    from app.short_horizon.expert_guided import learn as L
    from app.short_horizon.expert_guided import temporal as T
    from tests.test_expert_guided_ablation import _synthetic_dataset
    from tools import expert_guided_e3m as M
    from tools.expert_guided_e1t import Ctx
    ds, _ = _synthetic_dataset("dev3y", "fixed_train")
    cx = Ctx(ds)
    if poison_inner_b:
        rows = np.flatnonzero(ds.period == "INNER_B")
        cx.Xw[rows] = np.random.default_rng(99).normal(0, 1000, cx.Xw[rows].shape).astype(np.float32)
    SCFG, ECFG = M.effective_contracts("sac", 2.0)
    AC = contract()
    AC["strategy"]["optimizer"]["final"] = {"population": 8, "generations": 2}
    AC["strategy"]["optimizer"]["visitation"] = {"population": 4, "generations": 1}
    la = {"labels": "both", "closer_rule": "hold_through", "theta_open": 0.5, "theta_close": 0.1, "extra_experts": "", "multires_thresholds_json": ""}
    d = M.v1_context(cx, {"label_args": la}, ECFG, SCFG)
    nf, win, NS = ds.windows.shape[2], ds.windows.shape[1], len(E.STATE_NAMES)
    mods = {"open_l": (2, 2), "open_s": (2, 2), "close_l": (2, 0), "close_s": (2, 0), "nop": (2, 0)}

    def fit_module(v, m, D_tr, D_in, sd):
        nc, nr = mods[m]
        f = np.maximum(np.bincount(D_tr[1], minlength=nc) / len(D_tr[1]), 1e-3)
        w = 1 / np.sqrt(f)
        w = w / w[D_tr[1]].mean()
        model, _, info = L.fit(D_tr[0], D_tr[1], D_in[0], D_in[1], seed=sd, n_cls=nc, R=D_tr[2] if nr else None, Rm=D_tr[3] if nr else None,
                               Ri=D_in[2] if nr else None, Rmi=D_in[3] if nr else None, class_weight=w,
                               sample_weight=D_tr[4] if len(D_tr) > 4 else None, max_epochs=4, patience=2,
                               net=lambda: T.TemporalPolicyNet(nf, win, nc, n_reg=nr, n_state=NS, units=2))
        return T.TorchPolicy(model), info

    import tempfile
    d.update({"contract": AC, "mods": mods, "fit_module": fit_module, "SCFG": SCFG, "led": Path(tempfile.mkdtemp()) / "LEDGER.jsonl", "NS": NS,
              "rows_all": np.flatnonzero(cx.obs_ok), "cost": ACT.row_cost_sigma(SCFG.profile.round_trip_price(), ds.sigma, ds.row_bar, 0.1),
              "cal_rows": np.flatnonzero(cx.obs_ok & (ds.period == "INNER_A")), "exc": M.realized_excursions(cx)})
    return (cx, d) + M.v1_seed(cx, seed, d)


@pytest.mark.slow
def test_poisoning_the_decision_block_changes_nothing_that_is_fitted_or_chosen():
    cx0, d0, pols0, inf0, ex0, pol0, t0 = _synthetic_v1(False)
    cx1, d1, pols1, inf1, ex1, pol1, t1 = _synthetic_v1(True)
    assert ex0["best_epochs"] == ex1["best_epochs"]                             # early stopping
    assert {m: (c["a"], c["b"]) for m, c in ex0["calibration"].items()} == {m: (c["a"], c["b"]) for m, c in ex1["calibration"].items()}
    assert [r["visitation_strategy"] for r in ex0["visitation_rounds"]] == [r["visitation_strategy"] for r in ex1["visitation_rounds"]]
    same = lambda a, b: json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)   # noqa: E731  (NaN-aware)
    assert same(ex0["strategy"]["strategy"], ex1["strategy"]["strategy"]) and same(ex0["strategy"]["chosen"], ex1["strategy"]["chosen"])
    assert same({f: b["strategy"] for f, b in ex0["strategy"]["by_fitness"].items()}, {f: b["strategy"] for f, b in ex1["strategy"]["by_fitness"].items()})
    assert same(ex0["strategy"]["runs"]["with_veto"]["points"], ex1["strategy"]["runs"]["with_veto"]["points"])   # the whole archive
    ia = d0["cal_rows"]
    assert all(np.array_equal(t0[f][ia], t1[f][ia], equal_nan=True) for f in ACT.FIELDS)
    ib = np.flatnonzero(cx0.obs_ok & (cx0.ds.period == "INNER_B"))
    assert not np.allclose(t0["pL"][ib], t1["pL"][ib], equal_nan=True)         # the poison did reach INNER_B's outputs


@pytest.mark.slow
def test_v1_records_calibration_nop_consistency_and_tp_sl_diagnostics():
    cx, d, pols, infos, ex, pol, t = _synthetic_v1(False)
    assert set(ex["calibration"]) == set(ACT.SPECIALISTS)
    for c in ex["calibration"].values():
        assert len(c["after"]["reliability"]) <= 15 and set(c["before"]) >= {"auc", "brier", "ece"}
    assert set(ex["nop_label_consistency"]) == {"arm_open", "arm_no_open"}
    for m in ("open_l", "open_s"):
        x = ex["tp_sl_calibration_block"][m]
        assert x["rows"] > 0 and x["mask"].startswith("rows whose label is an OPEN")
    from tools import expert_guided_e3m as M
    rows0 = {}
    sets = M.v1_round0_sets(cx, "INNER_A", d, rows_out=rows0)
    assert all(len(rows0[m]) == len(sets[m][1]) for m in sets)                  # rows align with the samples
    lab = {m: dict() for m in ("open_l", "open_s", "nop")}
    for m in lab:
        for r, y in zip(rows0[m], sets[m][1]):
            lab[m].setdefault(r, set()).add(int(y))
    for r, ys in lab["nop"].items():                                            # consistency: NOP = 1 => no open label on that row
        if 1 in ys:
            assert 1 not in lab["open_l"].get(r, set()) and 1 not in lab["open_s"].get(r, set())
    assert len(sets["nop"]) == 5                                                # the NOP specialist carries its weights


def test_existing_modes_are_untouched_by_default():
    pytest.importorskip("torch")
    from tools import expert_guided_e3m as M
    a = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c"])
    assert a.action_contract == ""
    assert "nop" not in M.MODULES["FULL"]
    import inspect
    sig = inspect.signature(M.build_sets)
    assert sig.parameters["nop_labels"].default is None and sig.parameters["rows_out"].default is None


# ----------------------------------------------------------------------------- launcher: stage WF_V1
SPEC = ROOT / "docs" / "short_horizon" / "clean_round" / "CLEAN_ROUND_SPEC.json"


def _real_spec():
    s = json.loads(SPEC.read_text())
    paths = [s["recipe"]["path"], s["store"]["path"], s["walk_forward_v1"]["action_contract"]["path"]] + [f["recipe"]["path"] for f in s["walk_forward"]["folds"].values()]
    if not all(Path(p).exists() for p in paths):
        pytest.skip("the pinned spec's files are not on this machine")
    return s


def test_wf_v1_stage_validates_and_every_drift_is_refused():
    pytest.importorskip("torch")
    from tools import expert_guided_clean_round as C
    s = _real_spec()
    assert C.validate(s) == []
    for mut, code in ((lambda x: x["walk_forward_v1"]["action_contract"].update(sha256="0" * 64), "WF_V1_CONTRACT_DIGEST_MISMATCH"),
                      (lambda x: x["walk_forward_v1"]["action_contract"].update(identity="action_0000000000000000"), "WF_V1_CONTRACT_IDENTITY_MISMATCH"),
                      (lambda x: x["walk_forward_v1"]["action_contract"].update(path="/nonexistent.json"), "WF_V1_CONTRACT_MISSING"),
                      (lambda x: x["walk_forward_v1"].update(folds=["consolidated"]), "WF_V1_FOLD_REFUSED:consolidated"),
                      (lambda x: x["walk_forward_v1"]["arms"].append({"name": "x_v1", "base": "nope"}), "WF_V1_BASE_UNKNOWN"),
                      (lambda x: x["walk_forward_v1"].update(rule=""), "WF_V1_RULE_NOT_PREREGISTERED"),
                      (lambda x: x["stages"]["A"][0].update(action_contract="ACTION_CONTRACT_v1.json"), "ACTION_CONTRACT_NEEDS_DECISION_BLOCK")):
        x = copy.deepcopy(s)
        mut(x)
        assert any(e.startswith(code) for e in C.validate(x)), (code, C.validate(x))


def test_wf_v1_emission_checks_the_contract_digest_and_passes_the_contract(tmp_path):
    pytest.importorskip("torch")
    from tools import expert_guided_clean_round as C
    s = _real_spec()
    gate = tmp_path / "LAUNCH_EXPERT_GUIDED.json"
    gate.write_text("{}")
    s["gates"] = [{"path": str(gate)}]
    sp = tmp_path / "SPEC.json"
    sp.write_text(json.dumps(s))
    a = type("A", (), {"spec": str(sp), "stage": "WF_V1", "winner_from": None, "out": str(tmp_path / "q.sh"), "arms": None, "seeds": None, "out_suffix": None})()
    C.cmd_emit(a)
    q = (tmp_path / "q.sh").read_text()
    w = s["walk_forward_v1"]
    assert q.count("expert_guided_e3m.py run") == 4 and q.count("--action-contract $Q/contracts/ACTION_CONTRACT_v1.json") == 4
    assert q.count(f'{w["action_contract"]["sha256"]}  $Q/contracts/ACTION_CONTRACT_v1.json') == 4
    assert "--split-contract wf2022" in q and "--split-contract wf2023" in q and q.count(f"--seeds {w['seeds']}") == 4
    assert "WF_V1_wf2022_both_hold_v1" in q and "--labels both" in q and "STAGE_WF_V1_DONE" in q
    loc = C.localize(s, "/R")
    assert loc["walk_forward_v1"]["action_contract"]["path"] == "/R/contracts/ACTION_CONTRACT_v1.json"


@pytest.mark.slow
def test_seed_ensemble_v1_replays_each_seed_and_refits_one_calibration_and_strategy(tmp_path):
    torch = pytest.importorskip("torch")
    from tools import expert_guided_e3m as M
    from tools import expert_guided_seed_ensemble as SE
    cx, d, pols, infos, ex, pol, t = _synthetic_v1(False)
    seed = "20261019"
    for m, p in pols.items():
        torch.save(p.m.state_dict(), tmp_path / f"FULL_{m}_{seed}.pt")
    SCFG = d["SCFG"]
    periods = ["TRAIN", "INNER_A", "INNER_B"]
    trading = {p: M.summary(cx.rollout(p, pol(cx), SCFG)[0]) for p in periods}
    la = {"labels": "both", "closer_rule": "hold_through", "theta_open": 0.5, "theta_close": 0.1, "extra_experts": "", "multires_thresholds_json": ""}
    e3m = {"exec_contract": "sac", "min_orders_per_week": 2.0, "arch": {"units": 2, "branch": None, "core": None, "depth": 1, "head": 0},
           "recipe": None, "agent_state": {"mode": "none"},
           "action_contract": {"identity": d["contract"]["identity"], "sha256": "x", "contract": d["contract"], "label_args": la, "primary_fitness": "sharpe_lcb"},
           "trials": {"FULL": {seed: {"trading": trading, "action_contract": {"calibration": ex["calibration"], "strategy": ex["strategy"]}}}}}
    (tmp_path / "E3M.json").write_text(json.dumps(M.jsonable(e3m)))
    doc = SE.evaluate(tmp_path, cx.ds, periods)
    assert doc["verification"]["all_per_seed_replays_equal_run"]                # the member's own frozen policy reproduces the run
    assert set(doc["by_fitness"]) == set(ACT.FITNESS) and set(doc["ensemble_calibration"]) == set(ACT.SPECIALISTS)
    assert doc["arbiter"]["primary_fitness"] == "sharpe_lcb" and set(doc["ensemble"]) == set(periods)
    assert all("INNER_B" in v["windows"] for v in doc["by_fitness"].values())


# ----------------------------------------------------------------------------- report wf-v1 (generator test, owner's closure-table rule)
def _unit(out, name, ens_by_fitness=None, primary_ib=None):
    d = out / name
    d.mkdir(parents=True)
    tr = {p: {"sharpe_daily_sqrt365": 0.5, "max_drawdown_pct": 3.0, "orders_per_week": 3.0, "win_rate": 0.4, "valid": True} for p in ("TRAIN", "INNER_A", "INNER_B")}
    e = {"trials": {"FULL": {"1": {"trading": tr}}}, "params": {"FULL": {"open_l": 10}}, "meta": {"split_contract": {"arm": "wf2022"}, "support_hash": {"INNER_B": "h"}},
         "labels": {"mode": "both"}, "agent_state": {"mode": "none"}}
    (d / "E3M.json").write_text(json.dumps(e))

    def win(sh, se, opw, valid=True):
        return {"sharpe": sh, "se_lo": se, "sharpe_minus_se": sh - se, "days": 365, "bootstrap": {"low": sh - 2 * se, "high": sh + 2 * se},
                "metrics": {"max_drawdown_pct": 3.0, "orders_per_week": opw, "win_rate": 0.4, "valid": valid}}
    ib = win(*(primary_ib or (0.5, 0.4, 3.0)))
    ens = {"ensemble": {"INNER_B": ib}, "arbiter": {"arbiter": "x"}, "seeds": ["1"], "verification": {}, "per_seed": {}}
    if ens_by_fitness is not None:
        ens["by_fitness"] = {f: {"windows": {"INNER_B": win(*v)}, "strategy": {"x": 1}, "valid_on_calibration": True} for f, v in ens_by_fitness.items()}
        ens["ensemble_strategy"] = {"term_scales": {t: {"s": 1.0, "c": 1.0, "fallback_used": False} for t in ACT.SCALE_TERMS}}
    (d / "ENSEMBLE.json").write_text(json.dumps(ens))


def test_wf_v1_report_chooses_the_variant_on_wf2022_and_confirms_on_wf2023(tmp_path):
    from tools import expert_guided_wf_report as R
    out = tmp_path / "out"
    names = list(ACT.FITNESS)
    v22 = {f: (1.0, 0.5, 3.0) for f in names}
    v22["calmar"] = (3.0, 0.5, 3.0)                                             # best LCB on wf2022 ...
    v22["balanced"] = (9.0, 0.5, 1.5)                                           # ... a higher one is INVALID (< 2 trades/week)
    v23 = {f: (0.2, 0.5, 3.0) for f in names}
    v23["calmar"] = (1.0, 0.4, 2.5)                                             # confirmed on wf2023
    for fold, vv in (("wf2022", v22), ("wf2023", v23)):
        _unit(out, f"WF_V1_{fold}_both_hold_v1", vv)
        _unit(out, f"WF_{fold}_both_hold")
    spec = {"identity": "t", "walk_forward_v1": {"folds": ["wf2022", "wf2023"], "arms": [{"name": "both_hold_v1", "base": "both_hold"}],
                                                  "action_contract": {"identity": "action_x", "sha256": "0" * 64}, "rule": "r", "fitness_rule": "fr",
                                                  "fitness_variants": names}}
    sp = tmp_path / "S.json"
    sp.write_text(json.dumps(spec))
    a = type("A", (), {"results": str(out), "spec": str(sp), "out_json": str(tmp_path / "R.json"), "out_md": str(tmp_path / "R.md"), "date": "d"})()
    R.cmd_wf_v1(a)
    doc = json.loads((tmp_path / "R.json").read_text())
    rule = doc["fitness_variant_rule"]["both_hold_v1"]
    assert rule["chosen"] == "calmar" and rule["verdict"] == "CONFIRMED"
    rows = {(r["fitness"], r["fold"]): r for r in doc["fitness_rows"]}
    assert len(rows) == 2 * len(names) and rows[("balanced", "wf2022")]["verdict"].startswith("INVALID")
    md = (tmp_path / "R.md").read_text()
    assert "| both_hold_v1 | balanced | wf2022 |" in md and "Escalas" in md



# ----------------------------------------------------------------------------- encoding audit (E)
MUSASHI_LIKE = {"policy": {"decision": {"encoding": "intervals_of_continuous", "source_continuous": "tanh(mu) in [-1, 1]", "bins": [-0.6, -0.2, 0.2, 0.6],
                                        "n_classes": 5, "classes": ["close", "short", "hold", "long", "reverse"], "output_dim": 1, "loss": "sac"}}}


@pytest.mark.parametrize("mut, code", [
    (lambda d: d.update(specialists=MUSASHI_LIKE), "CATEGORICAL_FROM_CONTINUOUS:policy"),                       # one continuous output -> 5 bins
    (lambda d: d["specialists"]["open_l"]["decision"].update(encoding="thresholds_of_regression"), "CATEGORICAL_FROM_CONTINUOUS:open_l"),
    (lambda d: d["specialists"]["nop"]["decision"].update(bins=[0.5]), "CATEGORICAL_FROM_CONTINUOUS:nop"),
    (lambda d: d["specialists"]["open_s"]["continuous"]["tp_sigma"].update(encoding="softmax_ce"), "CONTINUOUS_THROUGH_SOFTMAX:open_s.tp_sigma"),
    (lambda d: d["specialists"]["close_l"]["decision"].update(output_dim=1), "HEAD_DIM_MISMATCH:close_l"),
    (lambda d: d["specialists"]["close_s"]["decision"].update(n_classes=3), "HEAD_DIM_MISMATCH:close_s"),
    (lambda d: d["specialists"]["open_l"]["continuous"]["sl_sigma"].update(output_dim=2), "HEAD_DIM_MISMATCH:open_l.sl_sigma"),
    (lambda d: d["specialists"]["open_l"]["continuous"]["sl_sigma"].update(lo=3.0, hi=0.5), "CONTINUOUS_BOUNDS:open_l.sl_sigma"),
    (lambda d: d["action_space"]["produced_by"].update(CLOSE=[["close_l", "maybe"]]), "ACTION_UNREACHABLE:CLOSE"),
    (lambda d: d["action_space"]["produced_by"].pop("OPEN_SHORT"), "ACTION_UNREACHABLE:OPEN_SHORT"),
    (lambda d: d["strategy"]["threshold_targets"].update(open_threshold="open_l.tp_sigma"), "CATEGORICAL_FROM_CONTINUOUS:strategy.open_threshold"),
])
def test_encoding_audit_refuses_musashis_design_and_every_encoding_drift(mut, code):
    d = contract()
    mut(d)
    d = ACT.seal(d)
    aud = ACT.encoding_audit(d)
    assert aud["verdict"] == "REFUSED" and any(e.startswith(code) for e in aud["errors"]), aud["errors"]
    assert any(e.startswith(code) for e in ACT.validate(d))                    # validate runs the audit


def test_head_shapes_are_checked_at_runtime():
    pytest.importorskip("torch")
    from app.short_horizon.expert_guided import temporal as T
    doc = contract()
    ok = {m: T.TemporalPolicyNet(4, 13, 2, n_reg=(2 if m in ACT.OPENERS else 0), n_state=9, units=2) for m in ACT.SPECIALISTS}
    assert all(v["status"] == "MATCH" for v in ACT.check_head_shapes(doc, ok).values())
    bad = dict(ok, nop=T.TemporalPolicyNet(4, 13, 3, n_state=9, units=2))      # 3 classes where the contract declares 2
    with pytest.raises(SystemExit, match="HEAD_SHAPE_MISMATCH:nop"):
        ACT.check_head_shapes(doc, bad)
    bad = dict(ok, open_l=T.TemporalPolicyNet(4, 13, 2, n_reg=1, n_state=9, units=2))   # one continuous head missing
    with pytest.raises(SystemExit, match="HEAD_SHAPE_MISMATCH:open_l"):
        ACT.check_head_shapes(doc, bad)
    bad = dict(ok, open_s=T.TemporalPolicyNet(4, 13, 2, n_reg=2, n_state=9, units=2, reg_bounds=((0.1, 5.0), (0.5, 3.0))))
    with pytest.raises(SystemExit, match="HEAD_SHAPE_MISMATCH:open_s:bounds"):
        ACT.check_head_shapes(doc, bad)


def test_closure_fill_writes_only_the_evidence_column():
    from tools import expert_guided_wf_report as R
    tpl = (ROOT / "docs" / "short_horizon" / "clean_round" / "OUTPUT_CONTRACT_V1_CLOSURE.md").read_text()
    assert all(f"<!-- EVIDENCE:{k} -->" in tpl for k in R.EVIDENCE_KEYS)
    ev = {k: f"value {k} | with a pipe" for k in R.EVIDENCE_KEYS}
    filled = R.fill_closure(tpl, ev)
    assert "<!-- EVIDENCE" not in filled and "value D3 / with a pipe" in filled
    for a_, b_ in zip(tpl.splitlines(), filled.splitlines()):                   # every other column unchanged
        if "<!-- EVIDENCE:" in a_:
            assert a_.rsplit("|", 2)[0] == b_.rsplit("|", 2)[0]
        else:
            assert a_ == b_

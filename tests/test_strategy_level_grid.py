"""DEAP stage 1 (the 144-point level grid, ruling 2026-10-29): the grid, the per-point documents, the world groups, the
aggregation (Deb: failing witnesses first, then Sharpe; an errored group counts as failing), the edge report, and the
mirror property test keeping the menu level genes of the base point (MENU_GENES_MISMATCH guard)."""
import json

from app.short_horizon.expert_guided import strategy_params as SP
from tools import strategy_level_grid as G
from tools import strategy_mirror_property as MP
from tools import strategy_param_deap as D


def test_grid_has_144_distinct_valid_full_f4_points():
    pts = G.grid_points()
    assert len(pts) == 144 == len({json.dumps(p, sort_keys=True) for p in pts})
    base = SP.default_doc()
    for p in pts:
        d = G.point_doc(base, p)
        assert d["exec_profile"] == "full_f4" and all(d[k] == v for k, v in p.items())
        assert {k: d[k] for k in d if k not in p} == {k: base[k] for k in base if k not in p}
    assert set(G.LEVEL_GRID) == set(D.MENU_LEVEL_KEYS)


def test_groups_keep_twin_pairs_together_and_aggregation_is_deb():
    sub = [{"subset_key": "A@q1.00"}, {"subset_key": "S_X@q1.00", "pair": "X"}, {"subset_key": "M_X@q1.00", "pair": "X"}]
    gs = G.groups_of(sub)
    assert [k for k, _ in gs] == ["A@q1.00", "X"] and len(gs[1][1]) == 2
    w = lambda k, sh, f: {"key": k, "sharpe": sh, "fails": f, "quality": 1.0, "activity": {"trades_per_week": 5.0, "reopen_1bar_share": 0.1}, "net_sigma_per_trade_median": 0.4, "burst_gate": None, "w17": None, "levels_clipped": 0}  # noqa: E731
    tasks = [{"point": 0, "group": "A@q1.00", "n_worlds": 1, "per_world": [w("A", 10.0, [])], "error": None},
             {"point": 0, "group": "X", "n_worlds": 2, "per_world": [w("S_X", 12.0, []), w("M_X", 14.0, ["W14_mirror"])], "error": None},
             {"point": 1, "group": "A@q1.00", "n_worlds": 1, "per_world": [w("A", 5.0, [])], "error": None},
             {"point": 1, "group": "X", "n_worlds": 2, "per_world": [], "error": "MenuGeneMismatch: x"},
             {"point": 2, "group": "A@q1.00", "n_worlds": 1, "per_world": [w("A", 1.0, [])], "error": None}]
    agg = G.aggregate(tasks, 2)
    assert set(agg) == {0, 1}                                         # point 2 is incomplete
    assert agg[0]["fitness_sharpe"] == 12.0 and agg[0]["n_failing_witnesses"] == 1 and not agg[0]["feasible"]
    assert agg[1]["n_failing_witnesses"] == 2 and not agg[1]["feasible"] and agg[1]["errors"]


def test_edge_report_flags_the_three_valued_extremes_only():
    e = G.edge_report({"a_sl": 0.75, "a_tp": 1.0, "b_sl": 0.5, "b_tp": 0.0, "buffer_pips": 3.0, "ref_p75": 0.0})
    assert e["a_sl"]["edge"] and not e["a_tp"]["edge"] and e["b_sl"]["at_max"] and not e["b_sl"]["edge"]


def test_mirror_property_random_genomes_keep_the_base_level_genes_under_full_f4():
    base = SP.default_doc()
    gs = MP.genomes(5, 20261028, None)
    assert len(gs) == 5
    for _, d in gs:
        assert all(d[k] == base[k] for k in D.MENU_LEVEL_KEYS)
    assert len({json.dumps({k: d[k] for k in D.GENOME if k not in D.MENU_LEVEL_KEYS}, sort_keys=True) for _, d in gs}) == 5


def test_judge_pairs_applies_the_full_f4_w14_rule_and_drops_the_control_world(monkeypatch):
    """full_f4 (BM.w14_menu_rule in the DEAP): the oracle's sealed W14 decides the form; DEFECT -> the side-swapped-reference
    mirror world governs W14 and E6; VALIDATED -> the sealed form; the Mswap world is judged, never scored."""
    calls = []

    def fake_check(x, y):
        calls.append((x, y))
        return {"pass": x == y, "n_a": 50, "n_b": 50}
    monkeypatch.setattr(D.BM, "mirror_check", fake_check)
    ep = {0: {"early": 0.5, "mean_R": 0.1, "n": 10}, 1: {"early": 0.5, "mean_R": 0.1, "n": 10}}

    def pair(o_s, o_m, s, m, sw):
        return [{"key": "S_X", "fails": [], "_pair": "X", "_role": "S", "_trades": s, "_episodes": ep, "_oracle_trades": o_s},
                {"key": "M_X", "fails": [], "_pair": "X", "_role": "M", "_trades": m, "_episodes": ep, "_oracle_trades": o_m},
                {"key": "Mswap_M_X", "fails": ["anything"], "_pair": "X", "_role": "Mswap", "_trades": sw, "_episodes": ep, "_oracle_trades": o_m}]
    res = pair("oa", "ob", "t", "u", "t")                             # oracle fails the sealed form (DEFECT): the swapped world governs and passes
    D.judge_pairs(res)
    assert [r["key"] for r in res] == ["S_X", "M_X"] and res[1]["fails"] == [] and res[1]["w14_control_status"] == "DEFECT"
    assert all(not any(k.startswith("_") for k in r) for r in res)
    res = pair("oa", "oa", "t", "u", "t")                             # oracle passes (VALIDATED): the sealed form governs and fails
    D.judge_pairs(res)
    assert res[1]["fails"] == ["W14_mirror"] and res[1]["w14_control_status"] == "VALIDATED"
    res = pair("oa", "ob", "t", "t", "v")                             # DEFECT and the swapped world differs: W14 fails on the governing form
    D.judge_pairs(res)
    assert res[1]["fails"] == ["W14_mirror"]


def test_swap_worlds_only_under_full_f4():
    sub = [{"subset_key": "S_X@q1.00", "pair": "X", "role": "S"}, {"subset_key": "M_X@q1.00", "pair": "X", "role": "M"}, {"subset_key": "A@q1.00"}]
    f4 = D.with_swap_worlds(sub, SP.default_doc())
    assert len(f4) == 4 and f4[-1]["role"] == "Mswap" and f4[-1]["swap_refs"] is True and f4[-1]["pair"] == "X"
    assert D.with_swap_worlds(sub, SP.default_doc("full_f3")) == sub


def test_stage2_genome_freezes_the_level_genes():
    assert set(D.GENOME_F4) - set(D.GENOME_F4_STAGE2) == set(D.MENU_LEVEL_KEYS) and {"z_veto", "burst_vol_ratio", "closer_adv_margin"} <= set(D.GENOME_F4_STAGE2)
    base = SP.validate({**SP.default_doc(), "a_sl": 1.5, "ref_p75": 1.0})
    try:
        D.STAGE["freeze_level_genes"] = True
        g = D.genome_of(base)
        assert g == D.GENOME_F4_STAGE2 and D.bounds(g)["burst_vol_ratio"] == [1.2, 2.5]
        doc = D.decode([1.0] * len(g), base)
        assert doc["a_sl"] == 1.5 and doc["ref_p75"] == 1.0 and doc["burst_vol_ratio"] == 2.5 and doc["open_rate_l"] == 1.0
    finally:
        D.STAGE["freeze_level_genes"] = False
    assert D.genome_of(base) == D.GENOME_F4


def test_an_infeasible_point_is_a_recorded_error_not_a_dead_worker(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise SystemExit("STRATEGY_SPACE_INFEASIBLE:no calibration-block reference rows for ['open_score']")
    monkeypatch.setattr(G.D, "CachedWorld", Boom)
    G._BASE.update({"AC": None, "cx": None, "SCFG0": None})
    monkeypatch.setattr(G.S, "with_variant", lambda s, v: s)
    r = G._task((4, "A@q1.00", [{"subset_key": "A@q1.00", "quality": 1.0}], SP.default_doc()))
    assert r["per_world"] == [] and r["error"].startswith("SystemExit: STRATEGY_SPACE_INFEASIBLE")
    agg = G.aggregate([r], 1)
    assert agg[4]["n_failing_witnesses"] == 1 and not agg[4]["feasible"]


def test_deap_worker_turns_an_infeasible_candidate_into_a_failing_world():
    class W:
        cell = {"subset_key": "S_X@q1.00", "quality": 1.0, "pair": "X", "role": "S"}
        released = False

        def evaluate(self, p):
            raise SystemExit("STRATEGY_SPACE_INFEASIBLE:x")

        def _release(self):
            W.released = True
    r = D._safe_evaluate(W(), {})
    assert r["fails"] == ["STRATEGY_SPACE_INFEASIBLE"] and r["sharpe"] == -10.0 and r["_role"] == "S" and W.released


def _boom_world(*a, **k):
    raise SystemExit("STRATEGY_SPACE_INFEASIBLE:no calibration-block reference rows for ['open_score']")


class _OkWorld:
    def __init__(self, cx, SCFG, AC, cell, seed):
        self.cell = cell

    def evaluate(self, doc):
        return {"key": self.cell["subset_key"], "sharpe": 1.0, "fails": [], "quality": 1.0, "opens": 3, "trades": 3, "early_closes": 0, "levels_clipped": 0,
                "activity": {"trades_per_week": 1.0, "reopen_1bar_share": 0.0}, "burst_gate": None, "w17": None, "trade_stats": {"net_sigma_per_trade_median": 0.3}}


def _dispatch_world(cx, SCFG, AC, cell, seed):
    return _boom_world() if cell.get("boom") else _OkWorld(cx, SCFG, AC, cell, seed)


def test_a_grid_task_raising_systemexit_leaves_the_pool_alive_and_records_the_error(monkeypatch, tmp_path):
    """A real fork pool (the stage-1 executor): one task's world raises SystemExit; every task returns, the pool survives,
    the error is recorded, and the report lists the point as INFEASIBLE with its reason, never ranked."""
    import multiprocessing as mp
    monkeypatch.setattr(G.D, "CachedWorld", _dispatch_world)
    monkeypatch.setattr(G.S, "with_variant", lambda s, v: s)
    G._BASE.update({"AC": None, "cx": None, "SCFG0": None})
    doc = SP.default_doc()
    tasks = [(0, "A@q1.00", [{"subset_key": "A@q1.00", "quality": 1.0}], doc),
             (1, "A@q1.00", [{"subset_key": "A@q1.00", "quality": 1.0, "boom": True}], doc),
             (2, "A@q1.00", [{"subset_key": "A@q1.00", "quality": 1.0}], doc)]
    pool = mp.get_context("fork").Pool(2, maxtasksperchild=8)
    try:
        out = list(pool.imap(G._task, tasks, chunksize=1))
        again = list(pool.imap(G._task, tasks[:1], chunksize=1))    # the pool still serves tasks afterwards
    finally:
        pool.close()
        pool.join()
    assert [r["point"] for r in out] == [0, 1, 2] and out[1]["error"].startswith("SystemExit: STRATEGY_SPACE_INFEASIBLE") and out[0]["error"] is None and again[0]["error"] is None
    d = tmp_path / "out"
    d.mkdir()
    pts = [G.grid_points()[i] for i in range(3)]
    (d / "RUN_META.json").write_text(json.dumps({"groups": 1, "points": 3, "point_list": pts}))
    (d / "TASKS.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    G.cmd_report(type("A", (), {"out": str(d)})())
    res = json.loads((d / "STAGE1_RESULT.json").read_text())
    assert [x["point"] for x in res["infeasible_points"]] == [1] and res["infeasible_points"][0]["status"] == "INFEASIBLE"
    assert "STRATEGY_SPACE_INFEASIBLE" in res["infeasible_points"][0]["reasons"][0]
    assert [r["point"] for r in res["ranking"]] == [0, 2] and all(r["status"] == "FEASIBLE" for r in res["ranking"])


def test_mirror_property_genomes_keep_a_non_canonical_launch_points_level_genes():
    point = {"a_sl": 0.75, "a_tp": 0.75, "ref_p75": 1.0}
    gs = MP.genomes(3, 20261028, point)
    assert gs[0][0] == "point" and all(d["a_sl"] == 0.75 and d["a_tp"] == 0.75 and d["ref_p75"] == 1.0 for _, d in gs)


def test_deap_pool_configures_the_level_genes_before_building_worlds(monkeypatch):
    seen = {}

    class Ctx:
        def Pipe(self):
            raise RuntimeError("stop after configure")
    monkeypatch.setattr(D.mp, "get_context", lambda k: Ctx())
    D._BASE.update({"AC": 1, "cx": 1, "SCFG": 1})
    doc = SP.validate({**SP.default_doc(), "a_sl": 1.5})
    try:
        D.Pool([{"subset_key": "A@q1.00"}], 1, "wf2022", "", "", "", doc)
    except RuntimeError:
        pass
    finally:
        D._BASE.clear()
    assert D.ZZ.PARAMS["a_sl"] == 1.5
    D.ZZ.configure(SP.default_doc())

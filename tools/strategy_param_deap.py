#!/usr/bin/env python
"""Strategy parameter optimisation by simulation (owner order 2026-10-27): DEAP with fitness = mean daily SHARPE over a
declared, stratified subset of behaviour-matrix worlds (every block / family, each world and its real-cost mirror, at
perfect heads AND at the operating quality 0.70); a candidate is FEASIBLE only if every sealed / effective witness
still passes on its perfect-head subset worlds and the quality-independent witnesses pass on the 0.70 worlds
(constraint domination by the number of failing witnesses; an infeasible point is never selected). No training: the
engine truth, the simulated heads and the ladder levels of every world are built ONCE per worker process; an
evaluation is the rollout of the parameterised C3Z strategy plus the witnesses. Population 16, 30 generations,
declared seed, elitism 2, every consumed coefficient of STRATEGY_PARAMS.json with its declared bounds as the genome.
Resumable (checkpoint per generation, archive of every evaluated point). The chosen point is re-verified on the FULL
matrix by the chain driver (tools/clean_round_chain/run_strategy_param_stage.sh).

CLI: subset (write the declared subset), measure (time one evaluation), run (the DEAP search), best (write the k-th
best feasible point as a STRATEGY_PARAMS file).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402
from app.short_horizon.expert_guided import strategy_zz as ZZ  # noqa: E402
from tools import strategy_behaviour_matrix as BM  # noqa: E402
from tools import strategy_hierarchy as H  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402

SEED = 20261027
POPULATION, GENERATIONS, ELITE = 16, 30, 2
OPERATING_QUALITY = 0.70
# the genome: every STRATEGY_PARAMS coefficient the C3Z strategy consumes; the others are fixed, with the reason
GENOME = ("open_rate_l", "nop_veto", "close_rem_rate_l", "close_rem_rate_s", "close_p_l", "close_p_s", "allow_reverse", "observe_bars",
          "rr_margin_sigma", "ev_from_net", "close_margin_floor", "risk_reward_floor", "burst_window_bars", "burst_vol_ratio", "zz_room_tp_mult",
          "zz_fine_h", "zz_timing_gate", "zz_nop_gate", "zz_close_room", "close_p_requires_rem_below", "zz_theta_f", "zz_counter_net_frac", "zz_post_flip_bars", "closer_mode", "closer_adv_margin", "tp_mode", "k_tp_protect", "zz_reverse_flip_bars", "z_close", "k_persist", "closer_filter",
          "a_sl", "b_sl", "buffer_pips", "a_tp", "b_tp", "rho", "ref_p75", "horizon_rule")   # v2: the business-contract genes
INTEGER = {"observe_bars", "burst_window_bars", "zz_fine_h", "zz_post_flip_bars", "zz_reverse_flip_bars", "k_persist"}
BINARY = {"allow_reverse", "ev_from_net", "zz_timing_gate", "zz_nop_gate", "closer_mode", "tp_mode", "closer_filter", "ref_p75", "horizon_rule"}
TIED = {"open_rate_s": "open_rate_l"}
FIXED = {"tp_a_l": "parametric TP / SL family: not consumed by C3Z (cell family)", "v_alpha_l": "parametric volume family: not consumed by C3Z (risk sizing)",
         "risk_pct": "execution contract (owner's sizing rule), not a strategy coefficient", "entry_cutoff_hours": "execution contract (Friday cutoff)",
         "max_entries_per_day": "execution contract (daily cap)", "confirmation_lag_bars": "a witness constant, not consumed by the strategy",
         "open_margin_eps": "numerical guard against float ties on the quantile sample", "forced_every_bars": "the exit audit's episode schedule, not a trading coefficient", "tp_grid": "the cell grid of the outcome heads", "sl_grid": "the cell grid of the outcome heads"}
# full_f4 (coordinator ruling 2026-10-29): the protective-TP genes are INERT under the decision menu (a contracted TP is never
# inflated) and are excluded from that profile's genome, so the search does not spend dimensions on them (declared, not silent)
from app.short_horizon.expert_guided import menu as _MENU  # noqa: E402
INERT_FULL_F4 = dict(_MENU.INERT_UNDER_MENU)
GENOME_F4 = tuple(k for k in GENOME if k not in INERT_FULL_F4) + ("z_veto",)   # z_veto: the veto gate (ruling 2026-10-29), consumed under full_f4 only


def bounds_table(genome: tuple, doc: dict | None = None) -> dict:
    """The bounds table a genome is encoded on (STRATEGY_PARAMS v2.2, ruling A): the parameter document's own profile table
    when a document is given (full_f4 -> strategy_params.BOUNDS_F4, open_rate up to 1.0); otherwise the full_f4 genome on
    BOUNDS_F4 and the full genome on the audited full_f3 table BOUNDS (byte-identical)."""
    if doc is not None:
        return SP.bounds_of(SP.exec_profile_of(doc))
    return SP.bounds_of("full_f4") if tuple(genome) == GENOME_F4 else SP.BOUNDS


def genome_of(doc: dict) -> tuple:
    """The genome a parameter document's profile searches: full_f4 without its inert genes, otherwise the full genome."""
    return GENOME_F4 if SP.exec_profile_of(doc) == "full_f4" else GENOME


def project(x: list, genome: tuple, base: dict | None = None) -> list:
    """A genome vector of the full GENOME projected on ``genome`` (warm starts from an archive of the full genome); a gene
    of ``genome`` absent from GENOME (z_veto) takes the base document's encoded value (0.5 without a base)."""
    if len(x) == len(genome):
        return list(x)
    if len(x) != len(GENOME):
        raise ValueError(f"GENOME_LENGTH:{len(x)}")
    tb = bounds_table(genome, base)
    def enc(k):
        if base is None or k not in base:
            return 0.5
        lo, hi = tb[k]
        return (float(base[k]) - lo) / (hi - lo)
    # v2.2: a full-genome vector is read on the same table encode / decode use for ``base`` (the document's profile table);
    # an archive encoded before v2.2 on the full_f3 table must be decoded with that table first (warm starts are not planned)
    return [x[GENOME.index(k)] if k in GENOME else enc(k) for k in genome]


QUALITY_INDEPENDENT = ("W16_sl_tp_every_order", "W17_no_open_in_burst", "W0_activity", "W12_missing", "W11_cutoff")


# ----------------------------------------------------------------------------- the declared subset
def build_subset(cells: list, per_level: int = 1, seed: int = SEED, mirror_at_quality: bool = False, mirrors: bool = False) -> list:
    """Stratified: per matrix level (every block / family) ``per_level`` base cells drawn with the declared seed; each
    world appears with its REAL-COST mirror (R_ = mirrored path, costs unchanged), at perfect heads and at quality 0.70."""
    rng = random.Random(seed)
    base = [c for c in cells if not c.get("mirror") and not c.get("sym_of")]
    by_level = {}
    for c in base:
        by_level.setdefault(c["level"], []).append(c)
    out = []
    for lvl in sorted(by_level):
        pick = rng.sample(by_level[lvl], min(per_level, len(by_level[lvl])))
        for c in pick:
            m = dict(c)
            m["id"], m["mirror"], m["mirror_real_cost_of"] = "R_" + c["id"], True, c["id"]
            m["legs_gen"], m["legs"], m["regime"] = list(c["legs"]), [H.MIRROR[x] for x in c["legs"]], H.MIRROR[c["regime"]]
            for w in ((c, m) if mirrors else (c,)):                  # both trade directions are already in every world's BUY and SELL episodes
                for q in ((1.0, OPERATING_QUALITY) if (mirror_at_quality or w is c) else (1.0,)):
                    x = dict(w)
                    x["quality"] = q
                    x["subset_key"] = f"{w['id']}@q{q:.2f}"
                    out.append(x)
    return out


def activity_stats(world: dict, trades: list) -> dict:
    """The activity next to the Sharpe (coordinator ruling 2026-10-29: Sharpe-only selection trades activity for Sharpe;
    the owner sees it): trades per week of the block, the share of re-opens one bar or less after the previous close."""
    a, b = H.period_bars(world["cx"].ds, BM.BLOCK)
    weeks = max((b - a) / (24 * 5), 1e-9)
    tr = sorted(trades, key=lambda t: int(t["entry_bar_abs"]))
    gaps = [int(tr[i + 1]["entry_bar_abs"]) - int(tr[i]["exit_bar_abs"]) for i in range(len(tr) - 1)]
    return {"trades_per_week": len(tr) / weeks, "reopen_1bar_share": (sum(1 for g in gaps if g <= 1) / len(gaps)) if gaps else float("nan")}


def trade_stats(world: dict, trades: list) -> dict:
    """The record-schema trade summary (BM.trade_summary): net sigma per trade mean / median, win rate, starting balance,
    SL / TP pips and holding bars p10 / p50 / p90; for the business-contract comparison and the reports (never a witness)."""
    return BM.trade_summary(world, trades)


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ----------------------------------------------------------------------------- one cached world and one evaluation
class CachedWorld:
    def __init__(self, cx, SCFG, AC, cell: dict, seed: int = 0):
        self.cell = cell
        w = BM.cell_world(cx, SCFG, AC, cell, seed)
        self.w = w
        genes0 = SP.genes_c3(SP.default_doc())
        _, view, cells = BM.strategy_for(w, genes0, AC)          # builds the view, the ladder (cached below) and the C3 gene map inputs
        self.view0, self.cells = view, cells
        bars = w.get("bars") if w.get("bars") is not None else w["cx"].ds.bars
        self.levels = ZZ.compact_levels(ZZ.ladder_levels(bars, w["cx"].ds.sigma))   # exact narrow dtypes, unread arrays dropped (memory)
        self.bars = bars
        cx2, truth, table = w["cx"], w["truth"], w["table"]
        self.cal_rows = np.flatnonzero(cx2.obs_ok & (cx2.ds.period == BM.CAL))
        ref = C3.closer_reference_sides(table, S.closer_src_of(truth))
        self.gm = C3.gene_map(view, self.cal_rows, ref, cells, truth.e_time, truth.cost)
        self.ov = BM.oracle_view(w, view)
        orl = BM.rollout_oracle(w, self.ov, cells)                     # the oracle does not depend on the candidate: once per world
        self.oroll = orl
        owit = BM.witnesses(cell, w, orl, cells)
        self.owit = owit
        self.defects = sorted(k for k, v in owit.items() if isinstance(v, dict) and "pass" in v and not k.endswith("_v2") and v["pass"] is False)   # the hold oracle's failures (logged)
        # O*: the second validity control (timing / exit witnesses, VALIDITY_SPEC); its run is shared with the exit layer's memo
        self.sroll = BM.rollout_oracle_star(w, self.ov, cells)
        self.swit = BM.witnesses(cell, w, self.sroll, cells)
        cl1 = view["closer"].with_quality(1.0)
        memo = w.setdefault("_exit_memo", {})
        memo["sroll"], memo["st_star"], memo["st_hold"] = self.sroll, BM.exit_stats(w, self.sroll, cl1), BM.exit_stats(w, orl, cl1)
        self.closer1 = cl1
        self.cache_base = dict(view["closer"].cache)                  # the closer's path / value cache after the controls; restored per evaluation
        for ro in (self.sroll, self.oroll):                           # memory (the 13G pool was OOM-killed at generation 8): the controls' per-tick logs are consumed above
            for k in ("closes", "log", "positions", "attr"):
                ro.pop(k, None)
        self.oracle_sharpe = float(orl["metrics"].get("sharpe_daily_sqrt365", float("nan")))
        self._release()                                               # memory: the build's per-tick caches (head rows, new paths) are never read again
        self._release_build(cx)
        # memory: what an evaluation never reads again (the raw head table -- the view holds its restricted copy -- and the
        # exploration visits, consumed by the gene map and the closer references above); ~270 MB per world before
        self.w.pop("table", None)
        self.w["truth"].xvis = {}

    def evaluate(self, params: dict) -> dict:
        bc = getattr(self.view0["closer"], "bc", None)
        if bc is not None:
            bc.rows.clear()                                           # the market-only head's rows: recomputed per evaluation (bounded memory)
        cache = self.view0["closer"].cache                            # shared by every quality copy: without the reset it grew ~20 MB per
        cache.clear()                                                 # evaluation per worker (new positions per candidate) until the 13G
        cache.update(self.cache_base)                                 # scope was OOM-killed at generation 8
        ZZ.configure(params)
        if "rr_margin_sigma" in params:
            C3.RR_MARGIN_SIGMA = float(params["rr_margin_sigma"])
        xv = self.w["scfg"].variant
        if xv is not None and getattr(xv, "sizing_mode", "") == "risk_budget" and "rho" in params:   # v2: the candidate's rho reaches the engine
            self.w["scfg"] = dataclasses.replace(self.w["scfg"], variant=dataclasses.replace(xv, rho=float(params["rho"])))
        genes = SP.genes_c3(params)
        strat = C3.StrategyC3.from_genes(self.gm.to_absolute(genes), "nop_veto" in genes)
        view = dict(self.view0)
        qh, q = ZZ.head_quantities(self.bars, self.w["cx"].ds.sigma, self.w.get("zz_quality", self.w.get("quality", 1.0)), self.w.get("zz_mode"), self.w.get("seed", 0), cached=self.levels)
        keep = np.isfinite(view["pn"])
        for k, v in ZZ.to_rows(qh, self.w["cx"].ds.row_bar).items():
            view[k] = np.where(keep.reshape((-1,) + (1,) * (np.ndim(v) - 1)), v, np.nan)
        view["risk_burst"] = ZZ.burst_rows(self.bars, self.w["cx"].ds.sigma, self.w["cx"].ds.row_bar, hold=ZZ.burst_hold_bars(self.w["truth"].menu is not None))
        if ZZ.PARAMS["ev_from_net"] >= 0.5:
            strat = dataclasses.replace(strat, open_margin=ZZ.net_open_margin(view, self.cal_rows, self.cells, float(genes["open_rate"])))
        self.w["ladder"] = q
        roll = BM.rollout_cell(self.w, strat, view, self.cells)
        wit = BM.witnesses(self.cell, self.w, roll, self.cells)
        perfect = float(self.cell["quality"]) >= 0.999
        if perfect and "W8_early_close_at_switch" in wit:
            wit["W8_early_close_at_switch_v3"] = BM.w8_switch_v3(self.w, roll, self.sroll, self.closer1)
        vres = BM.validity(wit, self.owit, self.swit, BM.expected_witness_ids(self.cell, self.w))   # VALIDITY_SPEC v2, identical to the matrix's governing verdict
        fails = [k for k in vres["failing"] if perfect or k in QUALITY_INDEPENDENT]
        v1 = BM.validity_v1_retired(wit, self.owit, self.swit)        # the retired v1 rule, recorded for the re-score delta only (never governs)
        fails_v1 = [k for k in v1["failing"] if perfect or k in QUALITY_INDEPENDENT]
        exit_fails, ep_summary = [], None
        if perfect:                                                   # the exit layer (sealed EXIT_SPEC) is a constraint: E0-E5, EP, E1', E2', E7, Z1, E8
            ex = BM.exit_layer_of(self.cell, self.w, strat, view, self.cells, roll, self.oroll, self.ov, light=True)
            exit_fails = list(ex["failing_effective"])
            fails += exit_fails
            ep_summary = ex.get("episode_summary")
        d7 = None
        if not perfect:                                               # QUALIFICATION_SPEC D7 as a constraint (coordinator ruling 2026-10-28): per side, E4 and
            st_s = BM.closer_d7_stats(self.w, roll, self.closer1)     # regret <= the information-limited Bayes closer's + 0.05, the same opener
            broll = BM.rollout_policy(self.w, BM.PolicyBayes, strat, view, self.cells)
            st_b = BM.closer_d7_stats(self.w, broll, self.closer1)
            d7f = BM.d7_failures(st_s, st_b)
            fails += d7f
            d7 = {side: {"E4": st_s[side]["E4"], "E4_bayes": st_b[side]["E4"], "regret": st_s[side]["regret_norm"], "regret_bayes": st_b[side]["regret_norm"]} for side in ("BUY", "SELL")}
        sh = float(roll["metrics"].get("sharpe_daily_sqrt365", float("nan")))
        out = {"key": self.cell["subset_key"], "sharpe": sh if math.isfinite(sh) else -10.0, "fails": fails, "opens": wit["summary"]["opens"],
               "net_usd": wit["summary"]["net_usd"], "quality": float(self.cell["quality"]), "d7": d7,
               "validity_v2": {k: vres[k] for k in ("not_evaluable", "unvalidated", "defects", "replay_conflicts", "evaluated")}, "fails_v1_retired": fails_v1 + exit_fails,
               "trades": len(roll["trades"]), "early_closes": sum(1 for t in roll["trades"] if t.get("exit_reason") == "EARLY"),
               "levels_clipped": int(roll["metrics"].get("n_levels_clipped", 0)) if isinstance(roll["metrics"], dict) else None,
               "trade_stats": trade_stats(self.w, roll["trades"]), "activity": activity_stats(self.w, roll["trades"]), "horizon_mix": BM.horizon_mix(roll, view, self.w)}
        if self.cell.get("pair"):                                     # a zero-cost twin / exact-mirror pair: W14 and E6 are judged in the worker
            out["_trades"], out["_episodes"], out["_pair"], out["_role"] = BM.annotate_sigma(self.w, roll["trades"]), ep_summary, self.cell["pair"], self.cell["role"]
        self._release()
        return out

    def _release_build(self, cx_base):
        """Memory after the build: what only the build reads. The world's own feature matrix (cx.Xw: the learned C2 /
        C3 heads' input; the simulated heads never read it) and, on a q < 1 world, the oracle view (read only by the
        perfect-head exit layer); on a perfect world the oracle view's arrays that equal the view's share them. Set to
        None, not emptied: a later reader fails loudly instead of reading zeros. The engine-truth tables stay (the
        witnesses read them, some behind an 'is not None' guard that would change the result silently)."""
        wcx = self.w["cx"]
        if wcx is not cx_base and getattr(wcx, "Xw", None) is not None and wcx.Xw is not getattr(cx_base, "Xw", None):
            object.__setattr__(wcx, "Xw", None)
        if float(self.cell["quality"]) < 0.999:
            self.ov = None
        else:                                                         # at perfect heads the oracle view's tables equal the view's: one copy
            for k, v in list(self.ov.items()):
                u = self.view0.get(k)
                if isinstance(v, np.ndarray) and isinstance(u, np.ndarray) and v is not u and v.shape == u.shape and v.dtype == u.dtype \
                        and np.array_equal(v, u, equal_nan=v.dtype.kind == "f"):
                    self.ov[k] = u
        gc.collect()
        _malloc_trim()

    def _release(self):
        """Memory (coordinator ruling 2026-10-28, <= 0.17 GB per world): between evaluations a world holds only what the
        next evaluation reads. The market-only head's rows and the closer's per-position path / value cache are
        recomputed on demand (deterministic: the same numbers), so they are dropped after every evaluation and after
        the build, the closer cache back to the controls' paths; the hierarchy ladder of the last candidate is rebuilt
        by the next evaluation; the freed heap goes back to the OS."""
        bc = getattr(self.view0["closer"], "bc", None)
        if bc is not None:
            bc.rows.clear()
        cache = self.view0["closer"].cache
        cache.clear()
        cache.update(self.cache_base)
        self.w["ladder"] = None
        gc.collect()
        _malloc_trim()


# ----------------------------------------------------------------------------- genome <-> parameter document
def bounds(genome: tuple = GENOME) -> dict:
    tb = bounds_table(genome)
    return {k: list(tb[k]) for k in genome}


def decode(x: list, base: dict, genome: tuple | None = None) -> dict:
    """The parameter document of a genome vector; the genome is the full one or the full_f4 one, told apart by length
    unless given."""
    g = genome or (GENOME if len(x) == len(GENOME) else (GENOME_F4 if len(x) == len(GENOME_F4) else None))
    if g is None:
        raise ValueError(f"GENOME_LENGTH:{len(x)}")
    d = dict(base)
    tb = bounds_table(g, base)
    for k, u in zip(g, x):
        lo, hi = tb[k]
        v = min(max(lo + (hi - lo) * min(max(float(u), 0.0), 1.0), lo), hi)   # clipped: at u = 1 the float sum overshot hi (rho 0.010000000000000002 was refused)
        if k in INTEGER:
            v = int(round(v))
        if k in BINARY:
            v = 1.0 if v >= 0.5 * (lo + hi) else float(lo)
        d[k] = v
    for k, src in TIED.items():
        d[k] = d[src]
    return SP.validate(d)


def encode(doc: dict, genome: tuple = GENOME) -> list:
    tb = bounds_table(genome, doc)
    return [(float(doc[k]) - tb[k][0]) / (tb[k][1] - tb[k][0]) for k in genome]


# ----------------------------------------------------------------------------- the worker pool (one process per share of worlds)
_BASE = {}


def judge_pairs(res: list) -> None:
    """W14 (exact mirror of the twin's ledger) and E6 (episode exit mix and mean R of the twin and its mirror, sides
    swapped) for the twin / mirror pairs of the subset, added to the mirror world's failures; the trades and episode
    summaries are dropped before the results travel to the parent."""
    by = {}
    for r in res:
        if r.get("_pair"):
            by.setdefault(r["_pair"], {})[r["_role"]] = r
    for key, pr in by.items():
        a, b = pr.get("S"), pr.get("M")
        if a is None or b is None:                                    # a pair split across workers: NOT_EVALUABLE, never PASS (VALIDITY_SPEC v2)
            for r in (a, b):
                if r is not None:
                    r["fails"].append("W14_mirror_NOT_EVALUABLE")
            continue
        w14 = BM.mirror_check(a["_trades"], b["_trades"])
        if not w14["pass"]:
            b["fails"].append("W14_mirror")
        ea, eb = a.get("_episodes"), b.get("_episodes")
        if ea and eb:
            g = lambda d, k: d.get(k, d.get(str(k)))  # noqa: E731
            dmax = max(abs(g(ea, 0)["early"] - g(eb, 1)["early"]), abs(g(ea, 1)["early"] - g(eb, 0)["early"]), abs(g(ea, 0)["mean_R"] - g(eb, 1)["mean_R"]),
                       abs(g(ea, 1)["mean_R"] - g(eb, 0)["mean_R"]), abs(g(ea, 0)["n"] - g(eb, 1)["n"]), abs(g(ea, 1)["n"] - g(eb, 0)["n"]))
            if not dmax <= 1e-9:
                b["fails"].append("E6_exit_mirror")
    for r in res:
        for k in ("_trades", "_episodes", "_pair", "_role"):
            r.pop(k, None)


def _worker(conn, share: list, fold: str, admitted: str, csv_dir: str, contract: str):
    import resource
    import warnings
    warnings.filterwarnings("ignore")
    AC, cx, SCFG = _BASE["AC"], _BASE["cx"], _BASE["SCFG"]             # loaded once in the parent, shared by fork (copy-on-write)
    worlds = []
    t0 = time.time()
    for c in share:
        worlds.append(CachedWorld(cx, SCFG, AC, c, 0))
    conn.send({"ready": [w.cell["subset_key"] for w in worlds], "build_sec": time.time() - t0, "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
               "defects": {w.cell["subset_key"]: w.defects for w in worlds}, "oracle_sharpe": {w.cell["subset_key"]: w.oracle_sharpe for w in worlds}})
    while True:
        msg = conn.recv()
        if msg is None:
            break
        t1 = time.time()
        res = [w.evaluate(msg) for w in worlds]
        judge_pairs(res)
        conn.send({"results": res, "sec": time.time() - t1, "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0})


class Pool:
    def __init__(self, subset: list, jobs: int, fold: str, admitted: str, csv_dir: str, contract: str, params_doc: dict | None = None):
        ctx = mp.get_context("fork")
        if not _BASE:
            _BASE["AC"] = S.contract_for_sim(contract)
            cx, SCFG = S.load_fold(fold, admitted, csv_dir)
            _BASE["cx"], _BASE["SCFG"] = cx, S.with_variant(SCFG, C3.exec_variant_of(params_doc))   # the profile of the parameter file (absent = full_f3, byte-identical)
        groups = {}
        for c in subset:                                              # a twin / mirror pair shares one worker (W14 / E6 are judged there)
            groups.setdefault(c.get("pair") or c["subset_key"], []).append(c)
        shares = [[] for _ in range(jobs)]
        for i, g in enumerate(sorted(groups.values(), key=len, reverse=True)):
            min(shares, key=len).extend(g)
        self.conns, self.procs = [], []
        for sh in shares:
            if not sh:
                continue
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, sh, fold, admitted, csv_dir, contract), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.ready = [c.recv() for c in self.conns]

    def evaluate(self, params: dict) -> dict:
        for c in self.conns:
            c.send(params)
        outs = [c.recv() for c in self.conns]
        res = [r for o in outs for r in o["results"]]
        n_fail = sum(len(r["fails"]) for r in res)
        sh = float(np.mean([r["sharpe"] for r in res]))
        act = {"trades_per_week_mean": float(np.mean([r["activity"]["trades_per_week"] for r in res])) if res else float("nan"),
               "reopen_1bar_share_median": float(np.nanmedian([r["activity"]["reopen_1bar_share"] for r in res])) if res else float("nan"),
               "net_sigma_per_trade_median": float(np.nanmedian([r["trade_stats"].get("net_sigma_per_trade_median", np.nan) for r in res])) if res else float("nan")}
        return {"fitness_sharpe": sh, "n_failing_witnesses": n_fail, "feasible": n_fail == 0, "per_world": res, "activity": act, "wall_sec": max(o["sec"] for o in outs),
                "worker_maxrss_mb": [round(o.get("maxrss_mb", 0.0)) for o in outs]}

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=30)


# ----------------------------------------------------------------------------- the search (constraint domination; elitism)
def better(a: dict, b: dict) -> bool:
    """Deb's constraint domination with one objective: fewer failing witnesses first, then the higher Sharpe."""
    if a["n_failing_witnesses"] != b["n_failing_witnesses"]:
        return a["n_failing_witnesses"] < b["n_failing_witnesses"]
    return a["fitness_sharpe"] > b["fitness_sharpe"]


def warm_population(archive: Path, base: dict, k: int) -> list:
    """The top ``k`` distinct points of an earlier run's archive by (failing witnesses, -Sharpe) under ITS rule; they are
    re-scored under the current rule in generation 0 (coordinator ruling 2026-10-27)."""
    recs = [json.loads(x) for x in archive.read_text().splitlines() if x.strip()]
    recs.sort(key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))
    out, seen = [], set()
    for r in recs:
        key = json.dumps(r["params"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(encode({**base, **r["params"]}))
        if len(out) == k:
            break
    return out


class DegenerateGen0(RuntimeError):
    pass


DEGENERACY_RULE = ("generation-0 degeneracy (coordinator order 2026-10-28, standing for every DEAP launch): the search stops "
                   "on its own after generation 0 when (a) every candidate has the same failing count > 0 AND the identical failing "
                   "(world, witness) set, or (b) one witness fails "
                   "in 100 % of the candidates in EVERY world where it was evaluated (a constraint no parameter vector can "
                   "move: the W14 verifier defect of run 4b failed 9 / 9 pairs in 16 / 16 candidates), or (c) a "
                   "NOT_EVALUABLE verdict appears in every candidate; (b') a witness failing in 100 % of the candidates in "
                   "only SOME worlds is reported, not a stop (a narrow warm start shares real failures)")


def _applicable_worlds(witness: str, per_world: list) -> list:
    """The worlds where a witness is evaluated, from the subset: W14 on the mirror worlds, D7 on the q < 1 worlds, any
    other witness on every world (so (b) never fires for a witness that fails only where it applies to a few)."""
    base = witness.replace("_NOT_EVALUABLE", "")
    if base.startswith("W14"):
        return sorted(w["key"] for w in per_world if w["key"].startswith("M_"))
    if base.startswith("D7_"):
        return sorted(w["key"] for w in per_world if float(w.get("quality", 1.0)) < 0.999)
    return sorted(w["key"] for w in per_world)


def degenerate_gen0(recs: list) -> dict:
    """DEGENERACY_RULE on the archive records of generation 0 (each with n_failing_witnesses and per_world
    [{key, quality, fails}])."""
    n = len(recs)
    out = {"rule": DEGENERACY_RULE, "candidates": n, "degenerate": False, "reasons": [], "every_candidate_some_worlds": []}
    if n < 2:
        out["reasons"].append("fewer than 2 candidates: not judged")
        return out
    counts = sorted({int(r["n_failing_witnesses"]) for r in recs})
    sets = {frozenset((w["key"], f) for w in r["per_world"] for f in w["fails"]) for r in recs}
    if len(counts) == 1:                                              # (a) only with failures AND one identical failing set (a near-identical
        if counts[0] > 0 and len(sets) == 1:                          # warm start can share a count by chance: then logged, never a stop)
            out["degenerate"] = True
            out["reasons"].append(f"(a) every candidate has the same failing count {counts[0]} and the identical failing set")
        else:
            out["same_count_logged_only"] = {"count": counts[0], "distinct_failing_sets": len(sets)}
    hits = {}
    for i, r in enumerate(recs):
        for w in r["per_world"]:
            for f in w["fails"]:
                hits.setdefault(f, {}).setdefault(w["key"], set()).add(i)
    for f, by_world in sorted(hits.items()):
        always = sorted(k for k, idx in by_world.items() if len(idx) == n)
        if not always:
            continue
        applicable = _applicable_worlds(f, recs[0]["per_world"])
        if "NOT_EVALUABLE" in f:
            out["degenerate"] = True
            out["reasons"].append(f"(c) {f} in every candidate ({len(always)} worlds)")
        elif applicable and set(applicable) <= set(always):
            out["degenerate"] = True
            out["reasons"].append(f"(b) {f} fails in {n} / {n} candidates in all {len(applicable)} worlds where it is evaluated")
        else:
            out["every_candidate_some_worlds"].append({"witness": f, "worlds_failing_in_every_candidate": len(always), "worlds_evaluated": len(applicable)})
    return out


def run_search(pool: Pool, base: dict, out: Path, population: int, generations: int, seed: int, log=print, init: list | None = None) -> dict:
    from deap import tools as DT
    rnd = random.Random(seed)
    G = genome_of(base)
    ck = out / "CHECKPOINT.pkl"
    arch = out / "ARCHIVE.jsonl"
    if ck.exists():
        st = pickle.loads(ck.read_bytes())
        pop, gen0 = st["pop"], st["gen"] + 1
        rnd.setstate(st["rng"])
        log(json.dumps({"resume_from_generation": st["gen"]}))
    else:
        if init:
            pop = [project(x, G, base) for x in init[:population]] + [[rnd.random() for _ in G] for _ in range(population - len(init[:population]))]   # warm start
            log(json.dumps({"warm_start_points": len(init[:population])}))
        else:
            pop = [encode(base, G)] + [[rnd.random() for _ in G] for _ in range(population - 1)]   # the canonical point is in generation 0
        pop = [{"x": x, **{k: v for k, v in _eval(pool, x, base, arch, 0).items()}} for x in pop]
        gen0 = 1
        ck.write_bytes(pickle.dumps({"pop": pop, "gen": 0, "rng": rnd.getstate()}))
        dg = degenerate_gen0([json.loads(x) for x in arch.read_text().splitlines() if x.strip() and json.loads(x).get("generation") == 0])
        log(json.dumps({"gen0_degeneracy": dg}))
        if dg["degenerate"]:                                          # the stage gate (coordinator order 2026-10-28): stop on our own, no signal
            (out / "FAILED_DEGENERATE_GEN0").write_text(json.dumps(dg, indent=1))
            raise DegenerateGen0(dg["reasons"])
    nb = len(G)
    for g in range(gen0, generations + 1):
        pop.sort(key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))
        elite = [dict(r) for r in pop[:ELITE]]
        kids = []
        while len(kids) < population - ELITE:
            def tour():
                a, b = rnd.sample(pop, 2)
                return a if better(a, b) else b
            p1, p2 = list(tour()["x"]), list(tour()["x"])
            random.seed(rnd.random())
            if rnd.random() < 0.9:
                p1, p2 = DT.cxSimulatedBinaryBounded(p1, p2, eta=15.0, low=[0.0] * nb, up=[1.0] * nb)
            for ch in (p1, p2):
                DT.mutPolynomialBounded(ch, eta=20.0, low=[0.0] * nb, up=[1.0] * nb, indpb=1.0 / nb)
                kids.append(list(ch))
        kids = kids[:population - ELITE]
        pop = elite + [{"x": x, **_eval(pool, x, base, arch, g)} for x in kids]
        best = min(pop, key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))
        log(json.dumps({"generation": g, "best_sharpe": best["fitness_sharpe"], "best_failing": best["n_failing_witnesses"],
                        "feasible_in_pop": sum(1 for r in pop if r["feasible"])}))
        ck.write_bytes(pickle.dumps({"pop": pop, "gen": g, "rng": rnd.getstate()}))
    pop.sort(key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))
    return {"best": pop[0], "population": pop}


def _eval(pool: Pool, x: list, base: dict, arch: Path, gen: int) -> dict:
    doc = decode(x, base)
    r = pool.evaluate(doc)
    rec = {"generation": gen, "params": {k: doc[k] for k in genome_of(base)}, "fitness_sharpe": r["fitness_sharpe"], "n_failing_witnesses": r["n_failing_witnesses"],
           "activity": r.get("activity"),                             # trades / week, 1-bar re-open share, net sigma per trade next to the Sharpe (ruling 2026-10-29)
           "feasible": r["feasible"], "wall_sec": r["wall_sec"], "worker_maxrss_mb": r.get("worker_maxrss_mb"), "per_world": r["per_world"]}
    with open(arch, "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
    return {"fitness_sharpe": r["fitness_sharpe"], "n_failing_witnesses": r["n_failing_witnesses"], "feasible": r["feasible"]}


# ----------------------------------------------------------------------------- CLI
def cmd_subset(a):
    cells = json.loads(Path(a.cells).read_text())
    sub = build_subset(cells, a.per_level, SEED)
    Path(a.out).write_text(json.dumps(sub, indent=1))
    lv = {}
    for c in sub:
        lv[c["level"]] = lv.get(c["level"], 0) + 1
    print(json.dumps({"subset_worlds": len(sub), "levels": len(lv), "per_level": lv}))


def cmd_measure(a):
    sub = json.loads(Path(a.subset).read_text())[: a.n]
    base = SP.load(a.params)
    t0 = time.time()
    pool = Pool(sub, a.jobs, a.fold, a.admitted, a.csv_dir, a.contract, base)
    build = time.time() - t0
    times = []
    for _ in range(a.reps):
        t1 = time.time()
        r = pool.evaluate(base)
        times.append(time.time() - t1)
    pool.close()
    import resource
    print(json.dumps({"worlds": len(sub), "jobs": a.jobs, "build_sec_total": build, "build_sec_per_worker": [x["build_sec"] for x in pool.ready],
                      "worker_maxrss_mb": [x["maxrss_mb"] for x in pool.ready], "parent_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                      "eval_wall_sec": times, "eval_sec_per_world_per_job": float(np.mean(times)) * a.jobs / max(len(sub), 1),
                      "canonical_fitness_sharpe": r["fitness_sharpe"], "canonical_failing": r["n_failing_witnesses"],
                      "failing_by_world": {x["key"]: x["fails"] for x in r["per_world"] if x["fails"]}}, default=float))


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if BM.stop_requested(out) and os.environ.get("BEHAVIOUR_MATRIX_ALLOWED") != "1":
        print("STOP flag present: nothing started")
        raise SystemExit(75)
    sub = json.loads(Path(a.subset).read_text())
    base = SP.load(a.params)
    meta = {"seed": SEED, "population": a.population, "generations": a.generations, "elite": ELITE, "genome": genome_of(base), "bounds": bounds(genome_of(base)), "inert_fixed": (INERT_FULL_F4 if genome_of(base) == GENOME_F4 else {}), "fixed": FIXED, "tied": TIED,
            "subset": a.subset, "subset_worlds": len(sub), "operating_quality": OPERATING_QUALITY, "quality_independent_constraints": QUALITY_INDEPENDENT,
            "fitness": "mean daily Sharpe (sqrt 365) over the subset worlds", "constraint": "number of failing sealed / effective witnesses (perfect heads, per-cell rule) incl. the sealed exit layer E0-E5 / EP / E1' / E2' / E7 / Z1 / E8, + quality-independent witnesses (0.70) + QUALIFICATION_SPEC D7 per side on the q < 1 worlds (E4 and regret <= the Bayes closer's + 0.05, phi 0.587)",
            "exit_spec_sha256": BM.EXIT_SPEC_SHA,
            "params_base_sha256": SP.sha256_of(a.params), "spec_v3_sha256": BM.SPEC_V3_SHA, "warm_archive": getattr(a, "warm_archive", None),
            "effective_rule": "VALIDITY_SPEC per world (hold oracle + O* controls, vacuous passes do not validate, W8 switch v3 on a replayed destructive-close conflict)"}
    (out / "RUN_META.json").write_text(json.dumps(meta, indent=1, default=list))
    log = lambda m: (print(m, flush=True), open(out / "PROGRESS.log", "a").write(m + "\n"))  # noqa: E731
    t0 = time.time()
    pool = Pool(sub, a.jobs, a.fold, a.admitted, a.csv_dir, a.contract, base)
    log(json.dumps({"pool_ready_sec": time.time() - t0, "workers": len(pool.ready), "worker_build_maxrss_mb": [round(x["maxrss_mb"]) for x in pool.ready],
                    "defects": {k: v for x in pool.ready for k, v in x["defects"].items() if v}}))
    init = warm_population(Path(a.warm_archive), base, a.population) if getattr(a, "warm_archive", None) else None
    try:
        res = run_search(pool, base, out, a.population, a.generations, SEED, log, init=init)
    except DegenerateGen0 as e:                                       # the stage reads FAILED_DEGENERATE_GEN0 and stops without a resume
        log(json.dumps({"FAILED_DEGENERATE_GEN0": list(e.args[0])}))
        pool.close()
        sys.exit(3)
    pool.close()
    (out / "RESULT.json").write_text(json.dumps({"best": res["best"], "population": res["population"], "best_params": {k: v for k, v in zip(genome_of(base), res["best"]["x"])},
                                                 "wall_sec": time.time() - t0}, indent=1, default=float))
    log(json.dumps({"done": True, "best_sharpe": res["best"]["fitness_sharpe"], "best_failing": res["best"]["n_failing_witnesses"]}))


def cmd_best(a):
    """The k-th best FEASIBLE evaluated point of the archive as a STRATEGY_PARAMS document (for the full-matrix check)."""
    base = SP.load(a.params)
    recs = [json.loads(x) for x in Path(a.archive).read_text().splitlines() if x.strip()]
    seen, feas = set(), []
    for r in sorted(recs, key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"])):
        key = json.dumps(r["params"], sort_keys=True)
        if key in seen or not r["feasible"]:
            continue
        seen.add(key)
        feas.append(r)
    if not feas and getattr(a, "allow_infeasible", False):             # the DEAP point is re-verified on the full matrix whatever it picks (coordinator 2026-10-27)
        for r in sorted(recs, key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"])):
            key = json.dumps(r["params"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                feas.append(r)
    if a.k >= len(feas):
        print(json.dumps({"no_feasible_point_at_rank": a.k, "feasible_points": len(feas)}))
        raise SystemExit(3)
    r = feas[a.k]
    doc = dict(base)
    doc.update(r["params"])
    for k, src in TIED.items():
        doc[k] = doc[src]
    SP.validate(doc)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({"rank": a.k, "fitness_sharpe": r["fitness_sharpe"], "n_failing_witnesses": r["n_failing_witnesses"], "feasible": r["feasible"], "generation": r["generation"], "out": a.out, "sha256": SP.sha256_of(a.out)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--cells", required=True)
    s.add_argument("--per-level", type=int, default=1)
    s.add_argument("--out", required=True)
    for name in ("measure", "run"):
        r = sub.add_parser(name)
        r.add_argument("--subset", required=True)
        r.add_argument("--params", default=str(ROOT / "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json"))
        r.add_argument("--fold", default="wf2022")
        r.add_argument("--admitted", required=True)
        r.add_argument("--csv-dir", required=True)
        r.add_argument("--contract", default=str(S.CONTRACT))
        r.add_argument("--jobs", type=int, default=10)
        if name == "measure":
            r.add_argument("--n", type=int, default=4)
            r.add_argument("--reps", type=int, default=2)
        else:
            r.add_argument("--population", type=int, default=POPULATION)
            r.add_argument("--generations", type=int, default=GENERATIONS)
            r.add_argument("--out", required=True)
            r.add_argument("--warm-archive", default=None, help="an earlier run's ARCHIVE.jsonl: generation 0 = its top points re-scored under the current rule")
    b = sub.add_parser("best")
    b.add_argument("--archive", required=True)
    b.add_argument("--params", default=str(ROOT / "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json"))
    b.add_argument("--k", type=int, default=0)
    b.add_argument("--out", required=True)
    b.add_argument("--allow-infeasible", action="store_true", help="when no point is feasible, rank every point by (failing witnesses, -Sharpe)")
    a = ap.parse_args()
    {"subset": cmd_subset, "measure": cmd_measure, "run": cmd_run, "best": cmd_best}[a.cmd](a)


if __name__ == "__main__":
    main()

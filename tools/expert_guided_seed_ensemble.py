#!/usr/bin/env python3
"""Seed ensemble of a finished E3-M arm (owner 2026-10-25): the N seed checkpoints of every module are averaged in
probability space into ONE policy (softmax probabilities averaged; regression heads averaged) and replayed with the
same ``Composed`` routing, env and replay contract as the run. No training. Per period: every seed's own frozen policy
replayed again (a verification: it must reproduce the E3M.json figure), the ensemble, and the uncertainty of each Sharpe
(``stats``: Lo SE, stationary-bootstrap interval).

Arbiter of the ensemble (declared before any result was read): the most frequent frozen arbiter among the seeds (ties
-> "prob"); for "prob" the rate is the median of the seeds' frozen rates and tau is re-derived on the selection block
(INNER_A) from the ensemble's own open scores, exactly as the run derives it per seed. Nothing is chosen on the
validation window (INNER_B of a walk-forward fold, or OBS_2024).

Outputs ENSEMBLE.json in the arm's out dir (or --out).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import stats as ST  # noqa: E402
from tools.expert_guided_e1t import Ctx, summary  # noqa: E402
from app.short_horizon.expert_guided import action_contract as ACT  # noqa: E402
from tools.expert_guided_e3m import Composed, effective_contracts  # noqa: E402

MODULES = ("open_l", "open_s", "close_l", "close_s")


class ProbEnsemble:
    """Callable with the TorchPolicy contract (x[None, :] -> (logits, reg)): log of the mean softmax across members (so
    the consumer's softmax returns the mean probability exactly) and the mean regression output. N = 1 reproduces the
    single member's probabilities and regressions."""

    def __init__(self, members):
        self.members = list(members)
        if not self.members:
            raise ValueError("ENSEMBLE_EMPTY")

    def __call__(self, x):
        P, R = [], []
        for m in self.members:
            lg, rg = m(x)
            lg = np.asarray(lg, float)
            z = lg - lg.max(axis=1, keepdims=True)
            p = np.exp(z)
            P.append(p / p.sum(axis=1, keepdims=True))
            R.append(None if rg is None else np.asarray(rg, float))
        p = np.mean(P, axis=0)
        lg = np.log(np.clip(p, 1e-300, None))
        rg = None if R[0] is None else np.mean(R, axis=0)
        return lg, rg


def load_module(out_dir: Path, e3m: dict, module: str, seed: str, n_feat: int, win: int):
    import torch
    from app.short_horizon.expert_guided import temporal as T
    arch = e3m["arch"]
    kw = dict(units=arch["units"], branch=arch.get("branch"), core=arch.get("core"), depth=arch["depth"], head=arch["head"])
    if arch.get("groups") is not None:
        kw.update(groups=arch["groups"], group_units=arch["group_units"], group_names=arch["group_names"], group_kinds=arch["group_kinds"])
    for k in ("trunk", "fusion_units", "core_blocks", "readout"):      # residual_tcn runs record their trunk in ARCH (default u: absent)
        if k in arch:
            kw[k] = arch[k]
    sd = torch.load(out_dir / f"FULL_{module}_{seed}.pt", map_location="cpu", weights_only=True)
    nc = int(sd["cls.weight"].shape[0])                # classes and regression heads come from the checkpoint itself
    nr = 0
    if "lo" in sd and "hi" in sd:                      # the regression bounds travel in the checkpoint (persistent buffers)
        kw["reg_bounds"] = tuple(zip(sd["lo"].tolist(), sd["hi"].tolist()))
        nr = len(sd["lo"])
    ns = int((e3m.get("n_state_by_module") or {}).get(module, len(E.STATE_NAMES)))     # C2G closers: + order geometry
    net = T.TemporalPolicyNet(n_feat, win, nc, n_reg=nr, n_state=ns, **kw)
    net.load_state_dict(sd, strict=True)
    return T.TorchPolicy(net)


def ensemble_arbiter(trials: dict) -> dict:
    arbs = [trials[s]["arbiter"]["arbiter"] for s in trials]
    cnt = Counter(arbs)
    top = max(cnt.values())
    cands = sorted(a for a, c in cnt.items() if c == top)
    arb = "prob" if "prob" in cands else cands[0]
    rate = None
    if arb == "prob":
        rates = [trials[s]["arbiter"].get("rate") for s in trials if trials[s]["arbiter"]["arbiter"] == "prob" and trials[s]["arbiter"].get("rate") is not None]
        rate = float(np.median(rates)) if rates else 0.06
    return {"arbiter": arb, "rate": rate, "members": arbs, "rule": "most frequent frozen arbiter (ties -> prob); prob rate = median of the seeds' rates; tau re-derived on INNER_A"}


def window_stats(m: dict, cfg) -> dict:
    ret = ST.daily_returns(m["daily_equity"], float(cfg.capital))
    d = ST.describe(ret, point=m["sharpe_daily_sqrt365"])
    s = summary(m)
    return {"metrics": s, "sharpe": d["sharpe"], "engine_sharpe": d["engine_sharpe"], "recomputed_equals_engine": d["recomputed_equals_engine"],
            "se_lo": d["se_lo"], "sharpe_minus_se": d["sharpe_minus_se"], "days": d["days"], "bootstrap": d["bootstrap"]}


def evaluate_v1(out_dir: Path, e3m: dict, ds, periods: list, seeds: list, cfg=None, from_members: bool = False) -> dict:
    """Seed ensemble of an action-contract-v1 arm: per seed, its calibrated table and its own frozen strategy are
    replayed again (verification against the run); the ENSEMBLE averages the seeds' calibrated probabilities and TP/SL,
    refits ONE Platt map per specialist on the ensemble over the calibration block (round-0 label rows of INNER_A,
    rebuilt deterministically from the run's label arguments), and re-optimises the strategy on the calibration block
    only (never a member's strategy). The decision between arms reads this ensemble on the decision block."""
    from tools import expert_guided_e3m as M
    ac = e3m["action_contract"]
    AC = ac["contract"]
    cb = AC["blocks"]["calibration"]
    cx = Ctx(ds)
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    n_feat, win, NS = ds.windows.shape[2], ds.windows.shape[1], len(E.STATE_NAMES)
    SCFG, ECFG = effective_contracts(e3m.get("exec_contract", "sac"), e3m.get("min_orders_per_week"))
    cfg = cfg or SCFG
    trials = e3m["trials"]["FULL"]
    rows_all = np.flatnonzero(cx.obs_ok)
    cost = ACT.row_cost_sigma(SCFG.profile.round_trip_price(), ds.sigma, ds.row_bar, AC["strategy"]["cost_sigma"])
    cal_rows = np.flatnonzero(cx.obs_ok & (ds.period == cb))
    doc = {"arm_dir": str(out_dir), "recipe": e3m.get("recipe"), "split_contract": ds.meta["split_contract"], "seeds": seeds, "periods": periods,
           "support_hash": ds.meta["support_hash"], "action_contract": {"identity": ac["identity"], "sha256": ac["sha256"]},
           "per_seed": {}, "ensemble": {}, "verification": {"per_seed_replay_equals_run": {}}}
    tables = []
    for s in seeds:
        pols = {m: load_module(out_dir, e3m, m, s, n_feat, win) for m in ACT.SPECIALISTS}
        doc.setdefault("head_shapes", {})[s] = ACT.check_head_shapes(AC, {m: p.m for m, p in pols.items()})
        cal = trials[s]["action_contract"]["calibration"]
        t = ACT.calibrate_table(ACT.raw_table(pols, cx.Xw, rows_all, NS, cost), cal)
        tables.append(t)
        st_s = ACT.strategy_of(trials[s]["action_contract"]["strategy"]) or M.NO_TRADE
        doc["per_seed"][s] = {}
        for p in periods:
            m = cx.rollout(p, ACT.ContractPolicy(t, st_s)(cx), cfg)[0]
            doc["per_seed"][s][p] = window_stats(m, cfg)
            run_v = trials[s]["trading"].get(p, {}).get("sharpe_daily_sqrt365")
            if run_v is not None:
                doc["verification"]["per_seed_replay_equals_run"][f"{s}/{p}"] = bool(abs(run_v - m["sharpe_daily_sqrt365"]) < 1e-9)
    ens = ACT.mean_tables(tables)
    # ONE Platt map per specialist refitted on the ensemble, calibration block only (round-0 label rows of INNER_A)
    d = M.v1_context(cx, ac, ECFG, SCFG, periods=(cb,))
    rows0 = {}
    sets = M.v1_round0_sets(cx, cb, d, rows_out=rows0)
    ens_raw = {f: v.copy() for f, v in ens.items()}
    cal_e = {}
    for m, f in ACT.PROB_OF.items():
        ens_raw[f] = np.where(np.isfinite(ens[f]), ACT.logit(np.nan_to_num(ens[f], nan=0.5)), np.nan)
        r = np.asarray(rows0.get(m, []), np.int64)
        y = np.asarray(sets[m][1], int) if m in sets else np.zeros(0, int)
        if len(r) != len(y):
            raise SystemExit(f"ENSEMBLE_V1_ROWS_MISALIGNED:{m}")
        z = ens_raw[f][r]
        a_, b_ = ACT.fit_platt(z, y)
        cal_e[m] = {"a": a_, "b": b_, "rows": int(len(y)), "before": ACT.head_metrics(ACT.sigmoid(z), y), "after": ACT.head_metrics(ACT.platt(z, a_, b_), y)}
    ens_t = ACT.calibrate_table(ens_raw, cal_e)
    min_opw = float(AC["strategy"]["calibration_min_trades_per_week"])
    sig = np.asarray(ds.sigma, float)

    def score_table(view, strat):
        m, trades = ACT.rollout_with_trades(cx, cb, ACT.ContractPolicy(view, strat)(cx), SCFG)
        return ACT.window_components(m, trades, sig, SCFG.capital, min_opw, **ACT.score_kwargs(AC))

    primary = ac.get("primary_fitness") or ACT.DEFAULT_FITNESS
    if from_members:
        from app.short_horizon.expert_guided import ensemble_freeze as EF
        mem = EF.strategy_from_members({str(s_): trials[s_]["action_contract"]["strategy"]["strategy"] for s_ in seeds
                                        if (trials[s_].get("action_contract") or {}).get("strategy", {}).get("strategy")})
        fr = {"strategy": mem["strategy"], "nop_veto_kept": bool(mem["strategy"].get("nop_veto", False)), "chosen": None, "from_members": mem, "valid": True}
        doc["arbiter"] = {"arbiter": "action_contract_v1", "primary_fitness": primary, "strategy": fr["strategy"], "nop_veto_kept": fr["nop_veto_kept"], "chosen": None,
                          "rule": EF.RULE, "member_seed": mem["seed"]}
    else:
        fr = ACT.freeze_strategy(ens_t, cal_rows, score_table, AC, min(int(x) for x in seeds), "final", primary)
        doc["arbiter"] = {"arbiter": "action_contract_v1", "primary_fitness": primary, "strategy": fr["strategy"], "nop_veto_kept": fr["nop_veto_kept"], "chosen": fr["chosen"],
                          "rule": "mean of the seeds' calibrated probabilities and TP/SL -> one Platt refit per specialist on INNER_A -> strategy re-optimised on INNER_A (never a member's)"}
    doc["ensemble_calibration"] = cal_e
    doc["ensemble_strategy"] = fr
    doc["by_fitness"] = {}
    for f, bf in fr["by_fitness"].items():                 # every fitness variant on the same ensemble (no refit)
        st_f = ACT.strategy_from_dict(bf["strategy"]) if bf["strategy"] is not None else M.NO_TRADE
        w = {}
        for p in periods:
            if p == "OBS_2024" and f != primary:
                continue                                    # 2024: one read, the arm's own variant only
            m = cx.rollout(p, ACT.ContractPolicy(ens_t, st_f)(cx), cfg)[0]
            w[p] = window_stats(m, cfg)
        ib = w.get("INNER_B", {}).get("metrics")
        av = bf["archive_view"]
        bu = av["best_unconstrained_by_fitness"]
        if bu is not None:                                  # the fitness without the activity rule: NOT SELECTABLE, reported
            cu = ACT.ContractPolicy(ens_t, ACT.strategy_from_dict(bu["strategy"]))
            bu["windows_not_selectable"] = {p: window_stats(cx.rollout(p, cu(cx), cfg)[0], cfg) for p in periods if p != "OBS_2024"}
        doc["by_fitness"][f] = {"strategy": bf["strategy"], "valid_on_calibration": bf["valid"], "nop_veto_kept": bf["nop_veto_kept"],
                                "chosen_components": bf["chosen"], "windows": w, "archive_view": av,
                                "out_of_sample_activity": None if ib is None else {"INNER_B_trades_per_week": ib["orders_per_week"], "valid": bool(ib["valid"] and ib["orders_per_week"] >= 2.0)}}
    doc["ensemble"] = doc["by_fitness"][primary]["windows"]
    if "INNER_B" in doc["ensemble"]:
        ib = doc["ensemble"]["INNER_B"]["metrics"]
        doc["out_of_sample_activity"] = {"INNER_B_trades_per_week": ib["orders_per_week"], "valid": ib["valid"],
                                         "owner_rule": "below 2 trades/week on INNER_B = INVALID"}
    doc["verification"]["all_per_seed_replays_equal_run"] = all(doc["verification"]["per_seed_replay_equals_run"].values())
    doc["verification"]["recomputed_sharpe_equals_engine"] = all(v["recomputed_equals_engine"] for s in doc["per_seed"].values() for v in s.values()) and \
        all(v["recomputed_equals_engine"] for v in doc["ensemble"].values())
    return doc


def evaluate_c2(out_dir: Path, e3m: dict, ds, periods: list, seeds: list, cfg=None, from_members: bool = False) -> dict:
    """Seed ensemble of a C2 arm (causal-ladder stage C2): per seed its calibrated table and its own frozen strategy are
    replayed again (verification); the ENSEMBLE averages the seeds' calibrated probabilities and predicted values (net
    grid, remaining), refits ONE Platt map per binary head on the saved round-0 INNER_A label rows (C2_SETS.npz) and
    re-optimises the strategy on the calibration block only, for every fitness variant."""
    from app.short_horizon.expert_guided import c2 as C2
    ac = e3m["action_contract"]
    AC = ac["contract"]
    cb = AC["blocks"]["calibration"]
    cx = Ctx(ds)
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    n_feat, win, NS = ds.windows.shape[2], ds.windows.shape[1], len(E.STATE_NAMES)
    SCFG, _ = effective_contracts(e3m.get("exec_contract", "sac"), e3m.get("min_orders_per_week"))
    cfg = cfg or SCFG
    trials = e3m["trials"]["FULL"]
    cal_rows = np.flatnonzero(cx.obs_ok & (ds.period == cb))
    doc = {"arm_dir": str(out_dir), "recipe": e3m.get("recipe"), "split_contract": ds.meta["split_contract"], "seeds": seeds, "periods": periods,
           "support_hash": ds.meta["support_hash"], "action_contract": {"identity": ac["identity"], "sha256": ac["sha256"], "schema": AC["schema"]},
           "per_seed": {}, "ensemble": {}, "verification": {"per_seed_replay_equals_run": {}}}
    tables = []
    for s in seeds:
        pols = {m: load_module(out_dir, e3m, m, s, n_feat, win) for m in C2.SPECIALISTS}
        doc.setdefault("head_shapes", {})[s] = ACT.check_head_shapes(AC, {m: p.m for m, p in pols.items()})
        t = C2.seed_table(pols, cx, NS, trials[s]["action_contract"]["calibration"], geo=C2.is_c2g(AC))
        tables.append(t)
        st_s = C2.strategy_of(trials[s]["action_contract"]["strategy"]) or C2.NO_TRADE
        doc["per_seed"][s] = {}
        for p in periods:
            m = cx.rollout(p, C2.PolicyC2(t, st_s)(cx), cfg)[0]
            doc["per_seed"][s][p] = window_stats(m, cfg)
            run_v = trials[s]["trading"].get(p, {}).get("sharpe_daily_sqrt365")
            if run_v is not None:
                doc["verification"]["per_seed_replay_equals_run"][f"{s}/{p}"] = bool(abs(run_v - m["sharpe_daily_sqrt365"]) < 1e-9)
    saved = C2.load_sets(out_dir / "C2_SETS.npz")
    ens_t, cal_e = C2.ensemble_calibration(C2.mean_tables(tables), cx, saved[cb])
    min_opw = float(AC["strategy"]["calibration_min_trades_per_week"])
    sig = np.asarray(ds.sigma, float)

    def score_table(view, strat):
        m, trades = ACT.rollout_with_trades(cx, cb, C2.PolicyC2(view, strat)(cx), SCFG)
        return ACT.window_components(m, trades, sig, SCFG.capital, min_opw, **ACT.score_kwargs(AC))

    primary = ac.get("primary_fitness") or ACT.DEFAULT_FITNESS
    closer_src = {m: {"rows": saved[cb][m]["rows"], "G": saved[cb][m].get("G")} for m in C2.CLOSERS if m in saved[cb]}
    if from_members:
        from app.short_horizon.expert_guided import ensemble_freeze as EF
        mem = EF.strategy_from_members({str(s_): trials[s_]["action_contract"]["strategy"]["strategy"] for s_ in seeds
                                        if (trials[s_].get("action_contract") or {}).get("strategy", {}).get("strategy")})
        fr = {"strategy": mem["strategy"], "nop_veto_kept": bool(mem["strategy"].get("nop_veto", False)), "chosen": None, "from_members": mem, "valid": True}
        doc["arbiter"] = {"arbiter": "action_contract_c2", "primary_fitness": primary, "strategy": fr["strategy"], "nop_veto_kept": fr["nop_veto_kept"], "chosen": None,
                          "rule": EF.RULE, "member_seed": mem["seed"]}
    else:
        fr = C2.freeze(ens_t, cal_rows, score_table, AC, min(int(x) for x in seeds), "final", primary, closer_src)
        doc["arbiter"] = {"arbiter": "action_contract_c2", "primary_fitness": primary, "strategy": fr["strategy"], "nop_veto_kept": fr["nop_veto_kept"], "chosen": fr["chosen"],
                          "rule": "mean of the seeds' calibrated probabilities and predicted values -> one Platt refit per binary head on INNER_A -> strategy re-optimised on INNER_A"}
    doc["ensemble_calibration"] = cal_e
    doc["ensemble_strategy"] = fr
    doc["by_fitness"] = {}
    for f, bf in fr["by_fitness"].items():
        st_f = C2.strategy_from_dict(bf["strategy"]) if bf["strategy"] is not None else C2.NO_TRADE
        w = {}
        for p in periods:
            if p == "OBS_2024" and f != primary:
                continue
            w[p] = window_stats(cx.rollout(p, C2.PolicyC2(ens_t, st_f)(cx), cfg)[0], cfg)
        ib = w.get("INNER_B", {}).get("metrics")
        av = bf["archive_view"]
        bu = av["best_unconstrained_by_fitness"]
        if bu is not None:
            cu = C2.PolicyC2(ens_t, C2.strategy_from_dict(bu["strategy"]))
            bu["windows_not_selectable"] = {p: window_stats(cx.rollout(p, cu(cx), cfg)[0], cfg) for p in periods if p != "OBS_2024"}
        doc["by_fitness"][f] = {"strategy": bf["strategy"], "valid_on_calibration": bf["valid"], "nop_veto_kept": bf["nop_veto_kept"],
                                "chosen_components": bf["chosen"], "windows": w, "archive_view": av, "note": bf.get("note"),
                                "out_of_sample_activity": None if ib is None else {"INNER_B_trades_per_week": ib["orders_per_week"], "valid": bool(ib["valid"] and ib["orders_per_week"] >= 2.0)}}
    doc["ensemble"] = doc["by_fitness"][primary]["windows"]
    if "INNER_B" in doc["ensemble"]:
        ib = doc["ensemble"]["INNER_B"]["metrics"]
        doc["out_of_sample_activity"] = {"INNER_B_trades_per_week": ib["orders_per_week"], "valid": ib["valid"], "owner_rule": "below 2 trades/week on INNER_B = INVALID"}
    doc["verification"]["all_per_seed_replays_equal_run"] = all(doc["verification"]["per_seed_replay_equals_run"].values())
    doc["verification"]["recomputed_sharpe_equals_engine"] = all(v["recomputed_equals_engine"] for s in doc["per_seed"].values() for v in s.values()) and \
        all(v["recomputed_equals_engine"] for v in doc["ensemble"].values())
    return doc


def evaluate_c3(out_dir: Path, e3m: dict, ds, periods: list, seeds: list, cfg=None) -> dict:
    """Seed ensemble of a C3 arm: per seed its calibrated table (3-class openers, nets, NOP, closers with geometry) and
    its own frozen strategy are replayed again under the run's exec variant (verification); the ENSEMBLE averages the
    seeds' calibrated probabilities / nets, refits ONE calibration per specialist on the saved calibration-block rows
    (C3_SETS.npz) and re-optimises the strategy on the selection block only, for every fitness variant."""
    from app.short_horizon.expert_guided import c3 as C3
    ac = e3m["action_contract"]
    AC = ac["contract"]
    B = e3m.get("blocks_used") or {}
    cb, sel = B.get("calibration", "INNER_A"), B.get("selection", "INNER_A")
    cx = Ctx(ds)
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    n_feat, win, NS = ds.windows.shape[2], ds.windows.shape[1], len(E.STATE_NAMES)
    SCFG0, _ = effective_contracts(e3m.get("exec_contract", "sac"), e3m.get("min_orders_per_week"))
    SCFG = C3.cfg_for(e3m, SCFG0)
    cfg = cfg or SCFG
    trials = e3m["trials"]["FULL"]
    rp = C3.run_params(e3m)
    caps = C3.cap_arrays(cx, SCFG)
    cal_rows = np.flatnonzero(cx.obs_ok & (ds.period == cb))
    view_rows = np.union1d(cal_rows, np.flatnonzero(cx.obs_ok & (ds.period == sel)))
    doc = {"arm_dir": str(out_dir), "recipe": e3m.get("recipe"), "split_contract": ds.meta["split_contract"], "seeds": seeds, "periods": periods,
           "support_hash": ds.meta["support_hash"], "action_contract": {"identity": ac["identity"], "sha256": ac["sha256"], "schema": AC["schema"]},
           "exec_variant": e3m["c3"].get("exec_variant"), "blocks": dict(B), "per_seed": {}, "ensemble": {}, "verification": {"per_seed_replay_equals_run": {}}}
    tables = []
    for s in seeds:
        pols = {m: load_module(out_dir, e3m, m, s, n_feat, win) for m in C3.SPECIALISTS}
        doc.setdefault("head_shapes", {})[s] = ACT.check_head_shapes(AC, {m: p.m for m, p in pols.items()})
        t = C3.seed_table(pols, cx, NS, trials[s]["action_contract"]["calibration"], rp["cells"], caps)
        tables.append(t)
        st_s = C3.strategy_from_dict(trials[s]["action_contract"]["strategy"]["strategy"]) if trials[s]["action_contract"]["strategy"].get("strategy") else C3.NO_TRADE
        doc["per_seed"][s] = {}
        for p in periods:
            m = cx.rollout(p, C3.policy(t, st_s, e3m)(cx), cfg)[0]
            doc["per_seed"][s][p] = window_stats(m, cfg)
            run_v = trials[s]["trading"].get(p, {}).get("sharpe_daily_sqrt365")
            if run_v is not None:
                doc["verification"]["per_seed_replay_equals_run"][f"{s}/{p}"] = bool(abs(run_v - m["sharpe_daily_sqrt365"]) < 1e-9)
    saved = C2.load_sets(out_dir / "C3_SETS.npz")
    ens_t, cal_e = C3.ensemble_calibration(C3.mean_tables(tables), cx, saved[cb])
    min_opw = float(AC["strategy"]["calibration_min_trades_per_week"])
    sig = np.asarray(ds.sigma, float)

    def score_table(view, strat):
        m, trades = ACT.rollout_with_trades(cx, sel, C3.PolicyC3(view, strat, rp["cells"], rp["e_time"], rp["cost"])(cx), SCFG)
        return ACT.window_components(m, trades, sig, SCFG.capital, min_opw, **ACT.score_kwargs(AC))

    primary = ac.get("primary_fitness") or ACT.DEFAULT_FITNESS
    closer_src = {m: {"rows": saved[cb][m]["rows"], "G": saved[cb][m].get("G")} for m in C3.CLOSERS if m in saved[cb]}
    fr = C3.freeze(ens_t, view_rows, score_table, AC, min(int(x) for x in seeds), "final", primary, closer_src, rp["cells"], rp["e_time"], rp["cost"], cal_rows)
    st_e = C3.strategy_from_dict(fr["strategy"]) if fr.get("strategy") else C3.NO_TRADE
    benefit = C3.early_close_benefit(cx, ens_t, st_e, SCFG, rp["cells"], rp["e_time"], rp["cost"], dict(C3.EARLY_CLOSE_BENEFIT, block=sel))
    w_ = benefit["with_early_close"]
    from app.short_horizon.expert_guided import exec_variants as XV
    ok, reason = XV.early_close_ok(float(w_["n_close_AGENT_CLOSE"]) / max(int(w_["trades"]), 1), benefit["lower_bound"], C3.EARLY_CLOSE_MIN_SHARE)
    benefit.update({"early_close_ok": ok, "reason": reason, "beneficial": ok})
    if not ok and fr.get("strategy"):
        fr["strategy"] = st_e.no_early_close().to_dict()
        for bf in fr["by_fitness"].values():
            if bf.get("strategy"):
                bf["strategy"] = C3.strategy_from_dict(bf["strategy"]).no_early_close().to_dict()
    doc["arbiter"] = {"arbiter": "action_contract_c3", "primary_fitness": primary, "strategy": fr["strategy"], "nop_veto_kept": fr["nop_veto_kept"], "chosen": fr["chosen"],
                      "early_close_disabled": not ok, "early_close_benefit": benefit,
                      "rule": "mean of the seeds' calibrated probabilities and nets -> one calibration per specialist on the calibration block -> strategy re-optimised on the selection block"}
    doc["ensemble_calibration"] = cal_e
    doc["ensemble_strategy"] = fr
    doc["by_fitness"] = {}
    for f, bf in fr["by_fitness"].items():
        st_f = C3.strategy_from_dict(bf["strategy"]) if bf["strategy"] is not None else C3.NO_TRADE
        w = {}
        for p in periods:
            if p == "OBS_2024" and f != primary:
                continue
            w[p] = window_stats(cx.rollout(p, C3.policy(ens_t, st_f, e3m)(cx), cfg)[0], cfg)
        ib = w.get("INNER_B", {}).get("metrics")
        av = bf["archive_view"]
        bu = av["best_unconstrained_by_fitness"]
        if bu is not None:
            cu = C3.policy(ens_t, bu["strategy"], e3m)
            bu["windows_not_selectable"] = {p: window_stats(cx.rollout(p, cu(cx), cfg)[0], cfg) for p in periods if p != "OBS_2024"}
        doc["by_fitness"][f] = {"strategy": bf["strategy"], "valid_on_calibration": bf["valid"], "nop_veto_kept": bf["nop_veto_kept"],
                                "chosen_components": bf["chosen"], "windows": w, "archive_view": av, "note": bf.get("note"),
                                "out_of_sample_activity": None if ib is None else {"INNER_B_trades_per_week": ib["orders_per_week"], "valid": bool(ib["valid"] and ib["orders_per_week"] >= 2.0)}}
    doc["ensemble"] = doc["by_fitness"][primary]["windows"]
    if "INNER_B" in doc["ensemble"]:
        ib = doc["ensemble"]["INNER_B"]["metrics"]
        doc["out_of_sample_activity"] = {"INNER_B_trades_per_week": ib["orders_per_week"], "valid": ib["valid"], "owner_rule": "below 2 trades/week on INNER_B = INVALID"}
    doc["verification"]["all_per_seed_replays_equal_run"] = all(doc["verification"]["per_seed_replay_equals_run"].values())
    doc["verification"]["recomputed_sharpe_equals_engine"] = all(v["recomputed_equals_engine"] for s in doc["per_seed"].values() for v in s.values()) and \
        all(v["recomputed_equals_engine"] for v in doc["ensemble"].values())
    return doc


def evaluate(out_dir: Path, ds, periods: list, seeds: list | None = None, cfg=None, selection: str = "INNER_A", from_members: bool = False) -> dict:
    e3m = json.loads((out_dir / "E3M.json").read_text())
    from app.short_horizon.expert_guided import c2 as C2
    from app.short_horizon.expert_guided import c3 as C3
    if C3.is_c3((e3m.get("action_contract") or {}).get("contract", {})):            # C3 (2026-10-26)
        tr = e3m["trials"]["FULL"]
        return evaluate_c3(out_dir, e3m, ds, periods, [str(s) for s in (seeds or list(tr))], cfg)
    if C2.is_c2((e3m.get("action_contract") or {}).get("contract", {})):            # C2 and C2G
        tr = e3m["trials"]["FULL"]
        return evaluate_c2(out_dir, e3m, ds, periods, [str(s) for s in (seeds or list(tr))], cfg, from_members=from_members)
    if e3m.get("action_contract"):
        tr = e3m["trials"]["FULL"]
        return evaluate_v1(out_dir, e3m, ds, periods, [str(s) for s in (seeds or list(tr))], cfg, from_members=from_members)
    trials = e3m["trials"]["FULL"]
    seeds = [str(s) for s in (seeds or list(trials))]
    missing = [s for s in seeds if s not in trials]
    if missing:
        raise SystemExit(f"ENSEMBLE_SEEDS_NOT_IN_RUN:{missing}")
    cx = Ctx(ds)
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    n_feat, win = ds.windows.shape[2], ds.windows.shape[1]
    mask = E.state_mask(e3m["agent_state"]["mode"])
    SCFG, _ = effective_contracts(e3m.get("exec_contract", "sac"), e3m.get("min_orders_per_week"))
    cfg = cfg or SCFG
    pols = {s: {m: load_module(out_dir, e3m, m, s, n_feat, win) for m in MODULES} for s in seeds}
    doc = {"arm_dir": str(out_dir), "recipe": e3m.get("recipe"), "split_contract": ds.meta["split_contract"], "seeds": seeds, "periods": periods,
           "support_hash": ds.meta["support_hash"], "per_seed": {}, "ensemble": {}, "arbiter": ensemble_arbiter({s: trials[s] for s in seeds}),
           "verification": {"per_seed_replay_equals_run": {}}}
    for s in seeds:
        a = trials[s]["arbiter"]
        c = Composed("FULL", pols[s], a["arbiter"], a.get("tau"), mask=mask)
        doc["per_seed"][s] = {}
        for p in periods:
            m = cx.rollout(p, c(cx), cfg)[0]
            doc["per_seed"][s][p] = window_stats(m, cfg)
            run_v = trials[s]["trading"].get(p, {}).get("sharpe_daily_sqrt365")
            if run_v is not None:
                doc["verification"]["per_seed_replay_equals_run"][f"{s}/{p}"] = bool(abs(run_v - m["sharpe_daily_sqrt365"]) < 1e-9)
    ens = {m: ProbEnsemble([pols[s][m] for s in seeds]) for m in MODULES}
    arb = doc["arbiter"]
    tau = None
    if arb["arbiter"] == "prob":
        pm = Composed("FULL", ens, "prob", mask=mask).open_scores(cx, cx.ds.rows(selection))
        tau = float(np.quantile(pm, 1 - arb["rate"]))
    arb["tau"] = tau
    c = Composed("FULL", ens, arb["arbiter"], tau, mask=mask)
    for p in periods:
        m = cx.rollout(p, c(cx), cfg)[0]
        doc["ensemble"][p] = window_stats(m, cfg)
    doc["verification"]["all_per_seed_replays_equal_run"] = all(doc["verification"]["per_seed_replay_equals_run"].values())
    doc["verification"]["recomputed_sharpe_equals_engine"] = all(v["recomputed_equals_engine"] for s in doc["per_seed"].values() for v in s.values()) and \
        all(v["recomputed_equals_engine"] for v in doc["ensemble"].values())
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm-dir", required=True, help="E3-M out dir of the arm (E3M.json + FULL_<module>_<seed>.pt)")
    ap.add_argument("--extra-arm-dir", action="append", default=[], help="seed-extension out dirs of the same arm (pooled members)")
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--recipe", required=True, help="the recipe the arm ran on (fold recipe for WF units)")
    ap.add_argument("--split-contract", required=True)
    ap.add_argument("--periods", default="TRAIN,INNER_A")
    ap.add_argument("--strategy-from-members", action="store_true",
                    help="protocol v11 (A5-8): the ensemble strategy is a frozen MEMBER strategy (closest to the gene-wise median); no re-search")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rec = json.loads(Path(a.recipe).read_text())
    feats = [c["store_column"] for c in rec["tensor"]["columns"]]
    ds = DG.load_store(a.admitted, a.store, feats, scaling="fixed_train", contract=a.split_contract, recipe=rec)
    out_dir = Path(a.arm_dir)
    if a.extra_arm_dir:
        # pooled seeds: merge the extension dirs' checkpoints and trials into a scratch view of the arm (hard links, read-only)
        import tempfile
        # the scratch view lives next to the arm (same filesystem) so hard links work; symlink when they cannot
        view = Path(tempfile.mkdtemp(prefix="ens_pool_", dir=out_dir.parent))

        def link(f):
            try:
                (view / f.name).hardlink_to(f)
            except OSError:
                (view / f.name).symlink_to(f.resolve())
        e = json.loads((out_dir / "E3M.json").read_text())
        for f in out_dir.glob("FULL_*.pt"):
            link(f)
        for d in a.extra_arm_dir:
            d = Path(d)
            ep = json.loads((d / "E3M.json").read_text())
            if ep.get("recipe") != e.get("recipe") or ep["meta"]["support_hash"] != e["meta"]["support_hash"]:
                raise SystemExit(f"POOL_MISMATCH:{d}")
            e["trials"]["FULL"].update(ep["trials"]["FULL"])
            for f in d.glob("FULL_*.pt"):
                link(f)
        (view / "E3M.json").write_text(json.dumps(e))
        out_dir = view
    try:
        doc = evaluate(out_dir, ds, [p.strip() for p in a.periods.split(",") if p.strip()], from_members=bool(getattr(a, "strategy_from_members", False)))
    finally:
        if a.extra_arm_dir:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)     # the scratch view only; the arm dirs are untouched
    doc["arm_dir"] = a.arm_dir
    doc["pooled_dirs"] = a.extra_arm_dir
    out = Path(a.out) if a.out else Path(a.arm_dir) / "ENSEMBLE.json"
    out.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({"out": str(out), "seeds": doc["seeds"], "arbiter": doc["arbiter"],
                      "ensemble": {p: {"sharpe": round(v["sharpe"], 3), "se": round(v["se_lo"], 3), "valid": v["metrics"]["valid"]} for p, v in doc["ensemble"].items()},
                      "per_seed_equal_run": doc["verification"]["all_per_seed_replays_equal_run"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""PZ4 (docs/short_horizon/clean_round/PZ_PREREGISTRO_2026_10_30.md section 4): the quality ladder and the formal-versus-
pragmatic contrast. Per (level world, seed, quality) the world is built ONCE (behaviour-matrix cell_world + the C3Z
strategy_for, full_f4) and four arms run on it: C3Z (the canonical point, or the chain's champion by --params), PZ,
PZ-6h (PZ restricted to h = 6) and the trivial base (the side of p_up[6 h] at every flat tick, the h = 6 order of the
C3Z menu at the median references and baseline genes, no early close). Per arm: daily Sharpe, the median / mean net in
sigma per trade, the unit-size profit (sum of net sigma per trade), trades per week, the exit mix, the close regret
against O* over EVERY trade (P* = the best executable close of the trade's engine path or its natural exit; units of
the initial cash risk), the native counterfactual of every early close; the C3Z net head's MAE / naive and the PZ heads'
report (coverage, MAE / naive, Brier / naive).

CLI: run (resumable, one JSON per world, STOP flag between worlds), report (tables, the paired cluster bootstrap with
the Holm column, the pre-registered decision rule, plots).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.short_horizon.expert_guided import action_contract as ACT  # noqa: E402
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402
from app.short_horizon.expert_guided import strategy_pz as PZ  # noqa: E402
from app.short_horizon.expert_guided import strategy_zz as ZZ  # noqa: E402
from app.short_horizon.expert_guided import outcomes as OC  # noqa: E402
from tools import strategy_behaviour_matrix as BM  # noqa: E402
from tools import strategy_hierarchy as H  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402

ARMS = ("C3Z", "PZ", "PZ6", "TRIVIAL")
QUALITIES = (0.55, 0.62, 0.70, 0.80, 1.00)
DECISION_Q = 0.70
BOOT_SEED, BOOT_N, ALPHA = 20261028, 10000, 0.05


def _bar_masks(w):
    cx = w["cx"]
    rb = np.asarray(cx.ds.row_bar, np.int64)
    per = np.asarray(cx.ds.period)
    nb = len(w["bars"]) if w.get("bars") is not None else len(cx.ds.bars)
    out = []
    for p in (BM.CAL, BM.BLOCK):
        m = np.zeros(nb, bool)
        bb = rb[np.flatnonzero(cx.obs_ok & (per == p))]
        m[bb[(bb >= 0) & (bb < nb)]] = True
        out.append(m)
    return out


def regret_all_trades(w, trades, closer) -> dict:
    """Per trade (P* - realized net) / initial cash risk, P* = max(best executable close of the engine path, natural exit)."""
    env = closer.env
    sig = np.asarray(w["cx"].ds.sigma, float)
    reg = []
    for t in trades:
        k = int(t["entry_bar_abs"])
        u = int(t["units"])
        pos = E.Position(int(t["direction"]), u, k, float(t["entry_price"]), float(t["stop_loss"]), float(t["take_profit"]), float(sig[k]),
                         int(t.get("segment", 0) or 0), env._fill_cost(u, k, "MARKET"))
        path = closer.path(pos)
        risk = abs(float(t["entry_price"]) - float(t["stop_loss"])) * u
        if risk <= 0:
            continue
        pstar = max(float(np.max(path["P"])) if len(path["P"]) else -np.inf, float(path["hold_net"]))
        reg.append((pstar - float(t["net_usd"])) / risk)
    r = np.asarray(reg, float)
    return {"n": int(len(r)), "mean": float(r.mean()) if len(r) else float("nan"), "median": float(np.median(r)) if len(r) else float("nan")}


def arm_metrics(w, roll, closer) -> dict:
    trades = BM.annotate_sigma(w, roll["trades"])
    m = roll["metrics"]
    a, b = H.period_bars(w["cx"].ds, BM.BLOCK)
    weeks = max((b - a) / (24 * 5), 1e-9)
    ns = np.asarray([t["net_sigma"] for t in trades], float)
    mix = {k: float(np.mean([t["exit_reason"] == k for t in trades])) if trades else float("nan") for k in ("TP", "SL", "TIME", "EARLY")}
    return {"sharpe": float(m.get("sharpe_daily_sqrt365", float("nan"))), "trades": len(trades), "trades_per_week": len(trades) / weeks,
            "net_sigma_median": float(np.nanmedian(ns)) if len(ns) else float("nan"), "net_sigma_mean": float(np.nanmean(ns)) if len(ns) else float("nan"),
            "profit_sigma_unit": float(np.nansum(ns)) if len(ns) else 0.0, "exit_mix": mix, "regret_vs_Ostar": regret_all_trades(w, trades, closer),
            "exit_counterfactual": BM.exit_counterfactual(w, roll, closer), "trade_summary": BM.trade_summary(w, trades),
            "levels_clipped": int(m.get("n_levels_clipped", 0)), "failed_sanity": len(trades) == 0}


def _roll(w, Pol, view, strat, cells, scfg):
    truth = w["truth"]
    pol = Pol(view, strat, cells, truth.e_time, truth.cost, observe_bars=0)
    m, trades = ACT.rollout_with_trades(w["cx"], BM.BLOCK, pol(w["cx"]), scfg)
    return {"metrics": m, "trades": trades, "positions": getattr(pol, "positions", {}), "attr": getattr(pol, "attr", [])}


def run_world(cx, SCFG, AC, cell: dict, seed: int, q: float, genes_c3: dict, pz_genes: dict) -> dict:
    t0 = time.time()
    cq = dict(cell, quality=float(q))
    BM.STRATEGY, H.MENU_ORDERS_FN = "c3z", None
    w = BM.cell_world(cx, SCFG, AC, cq, seed)
    strat, view, cells = BM.strategy_for(w, genes_c3, AC)
    cl1 = view["closer"].with_quality(1.0)
    out = {"id": cell["id"], "seed": seed, "quality": float(q), "arms": {}}
    rc = BM.rollout_cell(w, strat, view, cells)
    out["arms"]["C3Z"] = arm_metrics(w, rc, cl1) | {"head_mae": BM.head_mae(w, view), "horizon_mix": BM.horizon_mix(rc, view, w)}
    # the PZ heads at the world's quality, on the same world
    cal_b, ev_b = _bar_masks(w)
    bars = w["bars"] if w.get("bars") is not None else w["cx"].ds.bars
    lab = PZ.expert_labels(bars)
    hd = PZ.heads(lab, q, seed, cal_b)
    rep = PZ.head_report(lab, hd, ev_b, cal_b)
    scfg = w["scfg"]
    xv = scfg.variant
    if xv is not None and getattr(xv, "sizing_mode", "") == "risk_budget" and abs(float(xv.rho) - float(pz_genes["rho"])) > 1e-12:
        scfg = dataclasses.replace(scfg, variant=dataclasses.replace(xv, rho=float(pz_genes["rho"])))   # PZ's rho gene reaches the engine
    rb = np.asarray(w["cx"].ds.row_bar, np.int64)
    sobj = PZ.strategy_object(pz_genes)
    for arm, hz, Pol in (("PZ", PZ.HORIZONS, PZ.PolicyPZ), ("PZ6", (6,), PZ.PolicyPZ), ("TRIVIAL", PZ.HORIZONS, PZ.PolicyTrivial)):
        v = dict(view)
        xo = PZ.exit_offsets(OC.make_env(w["cx"], w["scfg"]), rb) if PZ.DESIGN_VERSION != "1.0" else None
        v.update(PZ.view_entries(bars, w["cx"].ds.sigma, rb, w["table"]["cost_row"], hd, pz_genes, horizons=hz, exit_offset=xo))
        v["menu_stamp"] = None
        if arm == "TRIVIAL":
            v.update(PZ.trivial_orders(bars, w["cx"].ds.sigma, rb))
        r = _roll(w, Pol, v, sobj, cells, scfg if arm != "TRIVIAL" else w["scfg"])
        out["arms"][arm] = arm_metrics(w, r, cl1) | {"horizon_mix": BM.horizon_mix(r, v, w)}
    out["pz_heads"] = rep
    out["sec"] = round(time.time() - t0, 1)
    return out


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = json.loads(Path(a.cells).read_text())
    k, n = (int(x) for x in a.chunk.split("/"))
    tasks = [(c, s, q) for q in [float(x) for x in a.qualities.split(",")] for c in cells for s in [int(x) for x in a.seeds.split(",")]]
    mine = [t for i, t in enumerate(tasks) if i % n == k]
    AC = S.contract_for_sim(a.contract)
    cx, SCFG = S.load_fold(a.fold, a.admitted, a.csv_dir)
    doc = SP.load(a.params)
    SCFG = S.with_variant(SCFG, C3.exec_variant_of(doc))
    ZZ.configure(doc)
    genes = SP.genes_c3(doc)
    pz_genes = PZ.load_genes(a.pz_params) if a.pz_params else PZ.validate_genes({})
    PZ.DESIGN_VERSION = a.pz_design or PZ.DESIGN_VERSION
    meta = {"pz_design_version": PZ.DESIGN_VERSION, "params": str(a.params), "params_sha256": SP.sha256_of(a.params), "pz_genes": pz_genes, "pz_genes_sha256": PZ.genes_sha(pz_genes),
            "pz_code_sha256": PZ.code_sha(), "strategy_code_sha256": BM.STRATEGY_CODE_SHA, "cells": a.cells, "label": a.label}
    (out / "RUN_META.json").write_text(json.dumps(meta, indent=1))
    for c, s, q in mine:
        if (out / "STOP").exists():
            print(json.dumps({"STOP": True}), flush=True)
            break
        p = out / f"W_{c['id']}_s{s}_q{q:.2f}.json"
        if p.exists():
            continue
        try:
            rec = run_world(cx, SCFG, AC, c, s, q, genes, pz_genes)
        except SystemExit as e:
            rec = {"id": c["id"], "seed": s, "quality": q, "error": str(e)}
        rec["meta"] = {k_: meta[k_] for k_ in ("params_sha256", "pz_genes_sha256", "label", "pz_design_version")}
        p.write_text(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"world": p.name, "sec": rec.get("sec"), "sharpe": {k_: v["sharpe"] for k_, v in (rec.get("arms") or {}).items()}}), flush=True)


# ----------------------------------------------------------------------------- report
def paired_bootstrap(diff: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"n": int(len(d)), "mean": float("nan"), "ci95": [float("nan")] * 2, "p": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    bm = d[idx].mean(axis=1)
    p = min(1.0, 2.0 * min(float(np.mean(bm <= 0)), float(np.mean(bm >= 0))))
    return {"n": int(len(d)), "mean": float(d.mean()), "ci95": [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))], "p": p}


def holm(ps: list) -> list:
    m = len(ps)
    order = sorted(range(m), key=lambda i: (ps[i] if np.isfinite(ps[i]) else 2.0))
    adj = [float("nan")] * m
    run = 0.0
    for rank, i in enumerate(order):
        if not np.isfinite(ps[i]):
            continue
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def contrast(recs: list, q: float) -> dict:
    """The pre-registered family at quality q: {PZ, PZ6, TRIVIAL} - C3Z per world on Sharpe and on the close regret."""
    by = [r for r in recs if abs(r["quality"] - q) < 1e-9 and "arms" in r]
    rows = []
    for arm in ("PZ", "PZ6", "TRIVIAL"):
        for met in ("sharpe", "regret"):
            diffs, excl = [], 0
            for r in by:
                a, c = r["arms"][arm], r["arms"]["C3Z"]
                if a["failed_sanity"] or c["failed_sanity"]:
                    excl += 1
                    continue
                x = (a["sharpe"] - c["sharpe"]) if met == "sharpe" else (a["regret_vs_Ostar"]["mean"] - c["regret_vs_Ostar"]["mean"])
                diffs.append(x)
            b = paired_bootstrap(np.asarray(diffs))
            rows.append({"arm": arm, "metric": met, "worlds": len(by), "excluded_failed_sanity": excl, **b})
    adj = holm([x["p"] for x in rows])
    for x, p in zip(rows, adj):
        x["p_holm"] = p
    return {"quality": q, "family": rows}


def decision(con: dict, matrix_pass: bool | None) -> dict:
    get = {(x["arm"], x["metric"]): x for x in con["family"]}
    sh, rg = get[("PZ", "sharpe")], get[("PZ", "regret")]
    c3_better_sharpe = bool(np.isfinite(sh["mean"]) and sh["mean"] < 0 and np.isfinite(sh["p_holm"]) and sh["p_holm"] < ALPHA)
    c3_better_regret = bool(np.isfinite(rg["mean"]) and rg["mean"] > 0 and np.isfinite(rg["p_holm"]) and rg["p_holm"] < ALPHA)
    if matrix_pass is None:
        verdict = "PENDING_MATRIX"
    elif not matrix_pass:
        verdict = "PZ_NOT_ADOPTED_MATRIX"
    elif c3_better_sharpe or c3_better_regret:
        verdict = "C3Z_KEPT"
    else:
        verdict = "PZ_ADOPTED"
    return {"matrix_pass": matrix_pass, "c3z_significantly_better_sharpe": c3_better_sharpe, "c3z_significantly_better_regret": c3_better_regret, "verdict": verdict}


def summarize(recs: list) -> list:
    rows = []
    for q in sorted({r["quality"] for r in recs if "arms" in r}):
        for arm in ARMS:
            xs = [r["arms"][arm] for r in recs if "arms" in r and abs(r["quality"] - q) < 1e-9]
            ok = [x for x in xs if not x["failed_sanity"]]
            med = lambda f: float(np.nanmedian([f(x) for x in ok])) if ok else float("nan")  # noqa: E731
            row = {"quality": q, "arm": arm, "worlds": len(xs), "failed_sanity": len(xs) - len(ok), "sharpe_median": med(lambda x: x["sharpe"]),
                   "sharpe_mean": float(np.nanmean([x["sharpe"] for x in ok])) if ok else float("nan"),
                   "net_sigma_per_trade_median": med(lambda x: x["net_sigma_median"]), "profit_sigma_unit_median": med(lambda x: x["profit_sigma_unit"]),
                   "trades_per_week_median": med(lambda x: x["trades_per_week"]), "early_share_median": med(lambda x: x["exit_mix"]["EARLY"]),
                   "sl_share_median": med(lambda x: x["exit_mix"]["SL"]), "regret_mean_median": med(lambda x: x["regret_vs_Ostar"]["mean"])}
            if arm == "C3Z":
                row["head_mae_over_naive_median"] = med(lambda x: x["head_mae"]["mae_over_naive"])
            rows.append(row)
        hp = [r["pz_heads"] for r in recs if "pz_heads" in r and abs(r["quality"] - q) < 1e-9]
        if hp:
            f = lambda k1, k2: float(np.nanmedian([h[k1].get(k2, np.nan) for h in hp]))  # noqa: E731
            rows.append({"quality": q, "arm": "PZ_HEADS", "worlds": len(hp), "fav_cov90": f("fav", "coverage_90"), "fav_cov50": f("fav", "coverage_50_q25_q75"),
                         "adv_below_q75": f("adv", "share_below_q75"), "adv_cov90": f("adv", "coverage_90"), "tau_cov90": f("tau", "coverage_90"),
                         "fav_mae_over_naive": f("fav", "mae_over_naive"), "adv_mae_over_naive": f("adv", "mae_over_naive"), "tau_mae_over_naive": f("tau", "mae_over_naive"),
                         "p_up_brier_over_naive": f("p_up", "brier_over_naive"), "p_up_accuracy": f("p_up", "accuracy"), "p_up_ece": f("p_up", "ece")})
    return rows


ARM_COLORS = {"C3Z": "#2a78d6", "PZ": "#eb6834", "PZ6": "#1baf7a", "TRIVIAL": "#eda100"}   # the dataviz reference categorical slots 1-4 (adjacent pairs validated)
ARM_LABELS = {"C3Z": "C3Z formal", "PZ": "PZ pragmatic", "PZ6": "PZ-6h", "TRIVIAL": "trivial base"}


def x_axis(row_q: dict, arm: str, heads: dict | None, kind: str) -> float:
    """The owner's plot rule: the arm's OWN outputs' error. C3Z: its net head's MAE / naive; PZ arms: the mean MAE / naive
    of the fav / adv / tau families (declared); kind 'mae_over_naive' only (MAE in different units is not comparable)."""
    if arm == "C3Z":
        return row_q.get("head_mae_over_naive_median", float("nan"))
    if heads is None:
        return float("nan")
    return float(np.nanmean([heads["fav_mae_over_naive"], heads["adv_mae_over_naive"], heads["tau_mae_over_naive"]]))


def plots(rep: dict, out: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = rep["summary"]
    heads = {r["quality"]: r for r in rows if r["arm"] == "PZ_HEADS"}
    files = []
    for met, ylab, fname in (("sharpe_median", "daily Sharpe (median over worlds)", "pz4_sharpe_vs_mae_naive.png"),
                             ("profit_sigma_unit_median", "unit-size profit, sum of net sigma per trade (median over worlds)", "pz4_profit_vs_mae_naive.png")):
        fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        for arm in ARMS:
            pts = sorted(((x_axis(r, arm, heads.get(r["quality"]), "mae_over_naive"), r[met], r["quality"]) for r in rows if r["arm"] == arm), key=lambda t: t[2])
            pts = [p for p in pts if np.isfinite(p[0]) and np.isfinite(p[1])]
            if not pts:
                continue
            xs, ys, qs = zip(*pts)
            ax.plot(xs, ys, color=ARM_COLORS[arm], lw=2, marker="o", ms=6, label=ARM_LABELS[arm], markeredgecolor="#fcfcfb", markeredgewidth=1.5)
            ax.annotate(ARM_LABELS[arm], (xs[0], ys[0]), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#52514e")
            for x_, y_, q_ in pts:
                if abs(q_ - DECISION_Q) < 1e-9:
                    ax.annotate("q0.70", (x_, y_), textcoords="offset points", xytext=(4, -10), fontsize=7, color="#52514e")
        ax.set_xlabel("MAE / naive of the arm's own outputs (q1.00 at 0; q0.55 rightmost)", fontsize=9, color="#52514e")
        ax.set_ylabel(ylab, fontsize=9, color="#52514e")
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#b5b4ad")
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"PZ4 ({rep.get('label')}): {ylab.split(' (')[0]} against the outputs' error", fontsize=10, color="#0b0b0b", loc="left")
        fig.tight_layout()
        fig.savefig(out / fname, facecolor=fig.get_facecolor())
        plt.close(fig)
        files.append(fname)
    return files


def cmd_report(a):
    recs = [json.loads(p.read_text()) for p in sorted(Path(a.results).glob("W_*.json"))]
    errs = [r for r in recs if "error" in r]
    matrix_pass = None if a.matrix_pass is None else (a.matrix_pass == "yes")
    con = contrast(recs, DECISION_Q)
    others = [contrast(recs, q) for q in QUALITIES if q != DECISION_Q and any(abs(r["quality"] - q) < 1e-9 for r in recs)]
    rep = {"records": len(recs), "errors": [(r["id"], r["seed"], r["quality"], r["error"][:200]) for r in errs], "summary": summarize([r for r in recs if "arms" in r]),
           "contrast_q070": con, "contrast_other_q_descriptive": others, "decision": decision(con, matrix_pass), "label": a.label,
           "rule": "PZ_PREREGISTRO_2026_10_30 section 4"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep["plots"] = plots(rep, out)
    (out / "PZ4_REPORT.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"records": len(recs), "errors": len(errs), "decision": rep["decision"]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cells", required=True)
    r.add_argument("--qualities", default=",".join(f"{q:.2f}" for q in QUALITIES))
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--chunk", default="0/1")
    r.add_argument("--fold", default="wf2022")
    r.add_argument("--admitted", required=True)
    r.add_argument("--csv-dir", required=True)
    r.add_argument("--contract", default=str(S.CONTRACT))
    r.add_argument("--params", required=True)
    r.add_argument("--pz-params", default=None)
    r.add_argument("--pz-design", default=None, choices=PZ.DESIGN_VERSIONS)
    r.add_argument("--label", default="canonical")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--matrix-pass", default=None, choices=("yes", "no"))
    p.add_argument("--label", default="canonical")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

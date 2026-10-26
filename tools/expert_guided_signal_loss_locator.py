#!/usr/bin/env python3
"""Signal-loss locator (pre-registered in docs/short_horizon/clean_round/SIGNAL_LOSS_LOCATOR_PREREGISTRATION_2026_10_26.md):
WHERE in our pipeline is predictive signal lost? Stage by stage on the SAME rows, targets and walk-forward folds:

  S1   HistGradientBoosting on the decision-bar recipe features + lags 2/6/12/24 rows   (already seen; not blind)
  S1b  S1 + causal rolling means (72, 168 rows) and rolling std (24, 168 rows)
  S2   HistGradientBoosting on the flattened recipe WINDOW exactly as the network sees it (diagnostic, never a model)
  S3   our trained heads per seed, raw;  S4 the same after Platt;  S5 the seed ensemble (mean of calibrated outputs)
  S6   the strategy's opened trades (seed-ensemble primary strategy), precision against target (a)

Verdicts are per stage (SIGNAL_PRESENT / ABSENT_AT_THIS_STAGE) and per drop (SUSPECT_INTERNAL_DEFECT when the paired
drop CI is entirely < 0). Nothing here is a statement about the market. TEST closed.

  run     one fold: stages S1-S5, S6 and the diagnostics; writes LOCATOR_<fold>.json
  report  both folds -> SIGNAL_LOSS_LOCATOR.{json,md}
"""
from __future__ import annotations

import argparse
import gzip
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import expert_guided_signal_audit as SA  # noqa: E402

REF = (1.5, 1.5)
B_CELLS = ((1.5, 1.5), (1.5, 2.5), (2.0, 2.5))
D_CELL, D_K = (1.5, 2.5), (3, 8)
ROLL_MEAN, ROLL_STD = (72, 168), (24, 168)
AUC_LO_MIN, SPEARMAN_LO_MIN = SA.AUC_LO_MIN, SA.SPEARMAN_LO_MIN
FAMILIES = {"C2": "WF_C2_13_{f}_zz_feat_c2_13", "C2G": "WF_C2G_13_{f}_zz_feat_c2g_13",
            "V13zz": "WF_V13_{f}_zz_hold_v13", "V13both": "WF_V13_{f}_both_hold_v13"}
ENSEMBLE_TRADES = "{unit}__ensemble_sharpe_lcb__INNER_B.csv.gz"
WINDOW_HOURS = 26


# ----------------------------------------------------------------------------- pure helpers
def rolling_features(C: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Causal rolling summaries over the last k observed rows (current row included); invalid rows are NaN and do not
    count; a summary needs at least k/2 valid rows. Columns: mean@72, mean@168, std@24, std@168 (each x n_features)."""
    import pandas as pd
    X = np.where(valid[:, None], C, np.nan)
    df = pd.DataFrame(X)
    out = [df.rolling(k, min_periods=max(2, k // 2)).mean().to_numpy() for k in ROLL_MEAN]
    out += [df.rolling(k, min_periods=max(2, k // 2)).std().to_numpy() for k in ROLL_STD]
    return np.concatenate(out, axis=1).astype(np.float32)


def stage_metric(kind: str, y, s, base: float | None):
    """One stage's metrics on index set idx (classification: AUC, Brier skill if s is a probability, top-10 % precision;
    regression: Spearman, top-10 % realized mean)."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    prob = kind != "regression" and np.nanmin(s) >= 0 and np.nanmax(s) <= 1

    def m(idx):
        yy, ss = y[idx], s[idx]
        mk = SA.top_mask(ss, 0.10)
        if kind == "regression":
            return {"spearman": SA.spearman(yy, ss), "top10": float(np.mean(yy[mk]))}
        out = {"auc": SA.auc(yy.astype(int), ss), "top10": float(np.mean(yy[mk]))}
        if prob and base is not None:
            out["brier_skill"] = SA.skill(SA.brier(yy, ss), SA.brier(yy, np.full(len(yy), base)))
        return out
    return m


def paired_bootstrap(fns: dict, day, n=SA.N_BOOT, block=SA.BLOCK_DAYS, seed=SA.SEED) -> dict:
    """Every stage evaluated on the SAME day draws: {stage: {metric: [draws]}} for paired drops."""
    u, groups = SA.day_groups(day)
    rng = np.random.default_rng(seed)
    draws = {k: {} for k in fns}
    for _ in range(n):
        di = SA.stationary_indices(len(u), block, rng)
        idx = np.concatenate([groups[i] for i in di])
        for k, fn in fns.items():
            for mk, v in fn(idx).items():
                draws[k].setdefault(mk, []).append(v)
    return draws


def ci(v) -> list:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [float("nan"), float("nan")]


def drop(draws: dict, a: str, b: str, metric: str) -> dict | None:
    """b - a on paired draws; SUSPECT_INTERNAL_DEFECT when the CI is entirely < 0."""
    if a not in draws or b not in draws or metric not in draws[a] or metric not in draws[b]:
        return None
    d = np.asarray(draws[b][metric], float) - np.asarray(draws[a][metric], float)
    c = ci(d)
    return {"from": a, "to": b, "metric": metric, "delta_ci": c, "delta_median": float(np.nanmedian(d)),
            "flag": "SUSPECT_INTERNAL_DEFECT" if c[1] < 0 else ("GAIN" if c[0] > 0 else "NO_SIGNIFICANT_CHANGE")}


def stage_verdict(kind: str, cis: dict) -> str:
    if kind == "regression":
        return "SIGNAL_PRESENT" if cis.get("spearman", [float("nan")])[0] > SPEARMAN_LO_MIN else "ABSENT_AT_THIS_STAGE"
    return "SIGNAL_PRESENT" if cis.get("auc", [float("nan")])[0] > AUC_LO_MIN else "ABSENT_AT_THIS_STAGE"


def mean_stage(per_seed_fns: list):
    """Mean over seeds of each seed's metric on the same draw (the S3 / S4 'per-seed head' stage)."""
    def m(idx):
        vals = [f(idx) for f in per_seed_fns]
        return {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
    return m


def first_vs_best(day_of_row, side_rows_by_day: dict, opened_rows, score, y) -> dict:
    """Per day with an opened trade: target value of the opened row vs the best-scored row of the same side and day."""
    opened, best = [], []
    for r in opened_rows:
        d = day_of_row[r]
        cand = side_rows_by_day.get(d)
        if cand is None or not len(cand):
            continue
        bi = cand[int(np.nanargmax(score[cand]))]
        opened.append(y[r])
        best.append(y[bi])
    return {"days": len(opened), "precision_opened": float(np.mean(opened)) if opened else float("nan"),
            "precision_best_same_day": float(np.mean(best)) if best else float("nan"),
            "pairs": [list(map(float, opened)), list(map(float, best))]}


def grid_round(x, grid) -> np.ndarray:
    g = np.asarray(grid, float)
    x = np.asarray(x, float)
    return g[np.argmin(np.abs(x[:, None] - g[None, :]), axis=1)] if len(x) else x


# ----------------------------------------------------------------------------- run one fold
def cmd_run(a):
    import torch  # noqa: F401
    from app.short_horizon.expert_guided import action_contract as ACT
    from app.short_horizon.expert_guided import c2 as C2
    from tools.expert_guided_seed_ensemble import load_module
    t0 = time.time()
    fold = a.split_contract
    ds, cx, SCFG, t, _tx, names, E, OC = _build(a)
    bar_ns = ds.bars.index.as_unit("ns").asi8
    day_bar = bar_ns // (86400 * 10**9)
    row_bar = np.asarray(ds.row_bar, np.int64)
    elig = np.flatnonzero(t["eligible"] & ds.row_ok)
    periods = ("TRAIN", "INNER_A", "INNER_B")
    nf, win = ds.windows.shape[2], ds.windows.shape[1]
    NS = len(E.STATE_NAMES)
    Ccur = ds.windows[:, -1, :]
    R = rolling_features(Ccur, np.asarray(ds.row_ok, bool))

    def F_S1(rows):
        return SA.features(ds, rows)

    def F_S1b(rows):
        return np.concatenate([SA.features(ds, rows), R[rows]], axis=1)

    def F_S2(rows):
        return ds.windows[rows].reshape(len(rows), -1).astype(np.float32)

    # ---- targets (same rows for every stage)
    targets = {}
    ci_ref = OC.cell_index(REF)
    for si, side in enumerate(("long", "short")):
        per = {}
        for p in periods:
            rows = elig[ds.period[elig] == p]
            ok = OC.available(cx, t["exit_bar"][rows, si, ci_ref], t["censored"][rows, si, ci_ref], p) & np.isfinite(t["net_sigma"][rows, si, ci_ref])
            rows = rows[ok]
            per[p] = (rows, (t["hit"][rows, si, ci_ref] == SA.HIT_TP).astype(float))
        targets[f"a_tp_first_{side}_tp1.5_sl1.5"] = {"kind": "a", "side": si, "per": per}
        for cell in B_CELLS:
            ci_ = OC.cell_index(cell)
            per = {}
            for p in periods:
                rows = elig[ds.period[elig] == p]
                ok = OC.available(cx, t["exit_bar"][rows, si, ci_], t["censored"][rows, si, ci_], p) & np.isfinite(t["net_sigma"][rows, si, ci_])
                rows = rows[ok]
                per[p] = (rows, t["net_sigma"][rows, si, ci_].astype(float))
            targets[f"b_net_{side}_tp{cell[0]:g}_sl{cell[1]:g}"] = {"kind": "regression", "side": si, "cell": ci_, "per": per}
        ci_d = OC.cell_index(D_CELL)
        for k in D_K:
            r0, r, y = SA.exit_decisions(elig, row_bar, t["hit"][elig, si, ci_d], t["net_sigma"][elig, si, ci_d], t["exit_bar"][elig, si, ci_d], k)
            per = {}
            for p in periods:
                m = (ds.period[r] == p) & ds.row_ok[r] & OC.available(cx, t["exit_bar"][r0, si, ci_d], t["censored"][r0, si, ci_d], p)
                per[p] = (r[m], y[m].astype(float))
            targets[f"d_sl_first_{side}_k{k}"] = {"kind": "d", "side": si, "per": per}

    # ---- network heads: raw / calibrated tables per family and seed on the union of scored rows
    score_rows = np.unique(np.concatenate([tg["per"][p][0] for tg in targets.values() for p in ("INNER_B",)]))
    heads, diag_fit = {}, {}
    for fam, pat in FAMILIES.items():
        unit = pat.format(f=fold)
        ud = Path(a.results) / unit
        if not (ud / "E3M.json").exists():
            heads[fam] = {"missing": unit}
            continue
        e3m = json.loads((ud / "E3M.json").read_text())
        trials = e3m["trials"]["FULL"]
        seeds = list(trials)
        fam_rec = {"unit": unit, "seeds": seeds, "raw": [], "cal": []}
        diag_fit[fam] = {"unit": unit, "params": e3m.get("params"), "fits": {s: trials[s].get("fits") for s in seeds}}
        is_c2 = fam.startswith("C2")
        for s in seeds:
            cal = trials[s]["action_contract"]["calibration"]
            if is_c2:
                mods = C2.SPECIALISTS if fam == "C2" else ("open_l", "open_s", "nop")
                pols = {m: load_module(ud, e3m, m, s, nf, win) for m in mods}
                X = np.c_[cx.Xw[score_rows], np.zeros((len(score_rows), NS), np.float32)].astype(np.float32)
                raw = {"z_ptp": np.full((len(cx.Xw), 2), np.nan), "net": np.full((len(cx.Xw), 2, len(OC.CELLS)), np.nan),
                       "z_psl": np.full((len(cx.Xw), 2), np.nan)}
                for m in ("open_l", "open_s"):
                    z, rr = ACT.module_outputs(pols[m], X)
                    raw["z_ptp"][score_rows, C2.SIDE_OF[m]] = z
                    raw["net"][score_rows, C2.SIDE_OF[m], :] = rr
                if fam == "C2":
                    for m in ("close_l", "close_s"):
                        z, _ = ACT.module_outputs(pols[m], X)
                        raw["z_psl"][score_rows, C2.SIDE_OF[m]] = z
                calt = {"ptp": np.full_like(raw["z_ptp"], np.nan), "psl": np.full_like(raw["z_psl"], np.nan), "net": raw["net"]}
                for m in ("open_l", "open_s"):
                    calt["ptp"][:, C2.SIDE_OF[m]] = np.where(np.isfinite(raw["z_ptp"][:, C2.SIDE_OF[m]]),
                                                             ACT.platt(np.nan_to_num(raw["z_ptp"][:, C2.SIDE_OF[m]]), cal[m]["a"], cal[m]["b"]), np.nan)
                if fam == "C2":
                    for m in ("close_l", "close_s"):
                        calt["psl"][:, C2.SIDE_OF[m]] = np.where(np.isfinite(raw["z_psl"][:, C2.SIDE_OF[m]]),
                                                                 ACT.platt(np.nan_to_num(raw["z_psl"][:, C2.SIDE_OF[m]]), cal[m]["a"], cal[m]["b"]), np.nan)
            else:
                pols = {m: load_module(ud, e3m, m, s, nf, win) for m in ("open_l", "open_s")}
                X = np.c_[cx.Xw[score_rows], np.zeros((len(score_rows), NS), np.float32)].astype(np.float32)
                raw = {"z_ptp": np.full((len(cx.Xw), 2), np.nan)}
                calt = {"ptp": np.full((len(cx.Xw), 2), np.nan)}
                for m in ("open_l", "open_s"):
                    z, _ = ACT.module_outputs(pols[m], X)
                    raw["z_ptp"][score_rows, C2.SIDE_OF[m]] = z
                    calt["ptp"][score_rows, C2.SIDE_OF[m]] = ACT.platt(z, cal[m]["a"], cal[m]["b"])
            fam_rec["raw"].append(raw)
            fam_rec["cal"].append(calt)
        heads[fam] = fam_rec
        print(json.dumps({"fold": fold, "family": fam, "seeds": len(seeds), "s": round(time.time() - t0, 1)}), flush=True)

    # ---- stages per target
    out = {"fold": fold, "window_hours": WINDOW_HOURS, "targets": {}, "diagnostics": {}}
    for name, tg in targets.items():
        kind, si = tg["kind"], tg["side"]
        rtr, ytr = tg["per"]["TRAIN"]
        ra, ya = tg["per"]["INNER_A"]
        rb, yb = tg["per"]["INNER_B"]
        if len(rtr) < 200 or len(ra) < 50 or len(rb) < 50:
            out["targets"][name] = {"kind": kind, "skipped": "too few rows"}
            continue
        base = float(np.mean(ytr)) if kind != "regression" else None
        scores, info = {}, {}
        for st, F in (("S1", F_S1), ("S1b", F_S1b), ("S2", F_S2)):
            Xtr, Xa = F(rtr), F(ra)
            if kind == "regression":
                fn, inf = SA.fit_regressor("hgb", Xtr, ytr, Xa, ya)
            else:
                fn, inf = SA.fit_classifier("hgb", Xtr, ytr.astype(int), Xa, ya.astype(int))
            scores[st] = fn(F(rb))
            if st == "S1" and kind != "regression":
                inf["train_auc_in_sample"] = SA.auc(ytr.astype(int), fn(Xtr))
                inf["inner_a_auc"] = SA.auc(ya.astype(int), fn(Xa))
            if st == "S1" and kind == "regression":
                inf["train_spearman_in_sample"] = SA.spearman(ytr, fn(Xtr))
            inf["features"] = int(Xtr.shape[1])
            info[st] = inf
        fams = {}
        for fam, fr in heads.items():
            if "missing" in fr:
                continue
            if kind == "a":
                raw_s = [r_["z_ptp"][rb, si] for r_ in fr["raw"]]
                cal_s = [c_["ptp"][rb, si] for c_ in fr["cal"]]
            elif kind == "regression":
                if "net" not in fr["raw"][0]:
                    continue
                raw_s = [r_["net"][rb, si, tg["cell"]] for r_ in fr["raw"]]
                cal_s = raw_s
            else:
                if fam != "C2":
                    continue
                raw_s = [r_["z_psl"][rb, si] for r_ in fr["raw"]]
                cal_s = [c_["psl"][rb, si] for c_ in fr["cal"]]
            if not np.all(np.isfinite(raw_s[0])):
                continue
            fams[fam] = (raw_s, cal_s)
        dd = day_bar[row_bar[rb]]
        fns = {st: stage_metric(kind, yb, scores[st], base) for st in scores}
        for fam, (raw_s, cal_s) in fams.items():
            fns[f"S3:{fam}"] = mean_stage([stage_metric(kind, yb, s_, None) for s_ in raw_s])
            if kind != "regression":
                fns[f"S4:{fam}"] = mean_stage([stage_metric(kind, yb, s_, base) for s_ in cal_s])
            fns[f"S5:{fam}"] = stage_metric(kind, yb, np.mean(cal_s, axis=0), base)
        idx = np.arange(len(yb))
        point = {k: f(idx) for k, f in fns.items()}
        draws = paired_bootstrap(fns, dd)
        cis = {k: {m: ci(v) for m, v in d.items()} for k, d in draws.items()}
        metric = "spearman" if kind == "regression" else "auc"
        drops = [drop(draws, "S1", "S1b", metric), drop(draws, "S1", "S2", metric)]
        for fam in fams:
            drops += [drop(draws, "S2", f"S3:{fam}", metric), drop(draws, f"S3:{fam}", f"S5:{fam}", metric),
                      drop(draws, "S1", f"S5:{fam}", metric)]
        rec = {"kind": kind, "n": {p: int(len(tg["per"][p][0])) for p in periods}, "base_rate": base, "info": info,
               "point": point, "ci": cis, "verdicts": {k: stage_verdict(kind, cis[k]) for k in cis},
               "drops": [d_ for d_ in drops if d_]}
        # ---- S6 + first-vs-best (target a only)
        if kind == "a":
            ymap = np.full(len(row_bar), np.nan)
            ymap[rb] = yb
            s6 = {}
            for fam, unit_pat in FAMILIES.items():
                unit = unit_pat.format(f=fold)
                tp = Path(a.trades) / ENSEMBLE_TRADES.format(unit=unit)
                if not tp.exists():
                    continue
                with gzip.open(tp, "rt") as fh:
                    tr = list(csv.DictReader(fh))
                side_dir = 1 if si == 0 else -1
                dec_bars = [int(x["entry_bar"]) - 1 for x in tr if int(x["direction"]) == side_dir]
                orows = [int(cx.rob[b]) for b in dec_bars if 0 <= b < len(cx.rob) and int(cx.rob[b]) >= 0]
                orows = [r for r in orows if np.isfinite(ymap[r])]
                k = len(orows)
                if k == 0:
                    s6[fam] = {"opened_with_label": 0}
                    continue
                prec_open = float(np.nanmean(ymap[orows]))
                pos_in_b = {r: i for i, r in enumerate(rb)}
                res = {"opened_with_label": k, "precision_opened": prec_open, "base_rate_inner_b": float(np.mean(yb))}
                ranked = [("S1", scores["S1"])] + ([(f"S5:{fam}", np.mean(fams[fam][1], axis=0))] if fam in fams else [])
                for st_name, sc in ranked:
                    mk = SA.top_mask(sc, k / len(yb))
                    res[f"precision_top_k_{st_name}"] = float(np.mean(yb[mk]))
                # paired-by-day bootstrap of precision(opened) - precision(top-k of S1)
                odays = day_bar[row_bar[np.asarray(orows)]]
                u, groups = SA.day_groups(np.concatenate([dd, odays]))
                rng = np.random.default_rng(SA.SEED)
                ally = np.concatenate([yb, ymap[orows]])
                isopen = np.r_[np.zeros(len(yb), bool), np.ones(len(orows), bool)]
                allsc = np.r_[scores["S1"], np.full(len(orows), np.nan)]
                dv = []
                for _ in range(SA.N_BOOT):
                    di = SA.stationary_indices(len(u), SA.BLOCK_DAYS, rng)
                    ii = np.concatenate([groups[j] for j in di])
                    io, ib = ii[isopen[ii]], ii[~isopen[ii]]
                    if len(io) == 0 or len(ib) == 0:
                        continue
                    mk = SA.top_mask(allsc[ib], len(io) / len(ib))
                    dv.append(float(np.mean(ally[io])) - float(np.mean(ally[ib][mk])))
                res["delta_opened_minus_S1_topk_ci"] = ci(dv)
                res["flag_vs_S1"] = "SUSPECT_INTERNAL_DEFECT" if res["delta_opened_minus_S1_topk_ci"][1] < 0 else "NO_SIGNIFICANT_LOSS"
                # first vs best signal of the day (daily cap)
                rows_by_day = {}
                for r in rb:
                    rows_by_day.setdefault(int(day_bar[row_bar[r]]), []).append(r)
                rows_by_day = {d_: np.asarray(v, np.int64) for d_, v in rows_by_day.items()}
                dor = {r: int(day_bar[row_bar[r]]) for r in orows}
                full_s1 = np.full(len(row_bar), np.nan)
                full_s1[rb] = scores["S1"]
                fvb = {"S1": first_vs_best(dor, rows_by_day, orows, full_s1, ymap)}
                if fam in fams:
                    full_s5 = np.full(len(row_bar), np.nan)
                    full_s5[rb] = np.mean(fams[fam][1], axis=0)
                    fvb[f"S5:{fam}"] = first_vs_best(dor, rows_by_day, orows, full_s5, ymap)
                for kk, vv in fvb.items():
                    pr = np.asarray(vv.pop("pairs"), float)
                    if pr.size and pr.shape[1] > 1:
                        rng2 = np.random.default_rng(SA.SEED)
                        bs = []
                        for _ in range(SA.N_BOOT):
                            ii = rng2.integers(0, pr.shape[1], pr.shape[1])
                            bs.append(float(np.mean(pr[1, ii]) - np.mean(pr[0, ii])))
                        vv["delta_best_minus_opened_ci"] = ci(bs)
                res["first_vs_best_same_day"] = fvb
                s6[fam] = res
            rec["S6"] = s6
        out["targets"][name] = rec
        print(json.dumps({"fold": fold, "target": name, "S1": point["S1"].get(metric), "S2": point["S2"].get(metric),
                          "S5": {k[3:]: v.get(metric) for k, v in point.items() if k.startswith("S5:")}, "s": round(time.time() - t0, 1)}), flush=True)

    # ---- diagnostics
    dg = out["diagnostics"]
    dg["training_dynamics"] = {}
    for fam, d_ in diag_fit.items():
        be = []
        for s, fits in d_["fits"].items():
            for m, fi in (fits or {}).items():
                be.append({"seed": s, "module": m, "epochs": fi.get("epochs"), "best_epoch": fi.get("best_epoch"),
                           "stop": fi.get("stop_reason"), "params": fi.get("params")})
        early = [x for x in be if x["best_epoch"] is not None and x["best_epoch"] <= 3]
        dg["training_dynamics"][fam] = {"unit": d_["unit"], "heads": be, "share_best_epoch_le_3": len(early) / max(len(be), 1),
                                        "learning_curves": "UNMEASURED: E3-M stores epochs / best epoch / stop reason only; per-epoch train vs INNER_A losses need a re-fit"}
    rows_train = {}
    for fam, pat in FAMILIES.items():
        led = Path(a.results) / pat.format(f=fold) / "TRIAL_LEDGER.jsonl"
        if not led.exists():
            continue
        for line in led.read_text().splitlines():
            e = json.loads(line)
            if e.get("event") == "FIT" and e.get("round", 0) == 0:
                rows_train.setdefault(fam, {}).setdefault(e["module"], []).append(int(e.get("rows_train", 0)))
    dg["params_vs_rows"] = {fam: {m: {"params": (diag_fit.get(fam, {}).get("params") or {}).get("FULL", {}).get(m),
                                     "rows_train_round0_median": int(np.median(v))} for m, v in mm.items()} for fam, mm in rows_train.items()}
    dg["grid_bounds"] = {}
    for fam, pat in FAMILIES.items():
        tp = Path(a.trades) / ENSEMBLE_TRADES.format(unit=pat.format(f=fold))
        if not tp.exists():
            continue
        with gzip.open(tp, "rt") as fh:
            tr = list(csv.DictReader(fh))
        dtp = np.asarray([float(x["dist_tp_sigma"]) for x in tr])
        dsl = np.asarray([float(x["dist_sl_sigma"]) for x in tr])
        gtp, gsl = grid_round(dtp, OC.GRID_TP), grid_round(dsl, OC.GRID_SL)
        dg["grid_bounds"][fam] = {"trades": len(tr), "tp_sigma_p10_p50_p90": [float(np.quantile(dtp, q)) for q in (0.1, 0.5, 0.9)] if len(tr) else None,
                                  "sl_sigma_p10_p50_p90": [float(np.quantile(dsl, q)) for q in (0.1, 0.5, 0.9)] if len(tr) else None,
                                  "share_tp_at_grid_max": float(np.mean(gtp == max(OC.GRID_TP))) if len(tr) else None,
                                  "share_sl_at_grid_max": float(np.mean(gsl == max(OC.GRID_SL))) if len(tr) else None,
                                  "share_tp_at_grid_min": float(np.mean(gtp == min(OC.GRID_TP))) if len(tr) else None,
                                  "share_sl_at_grid_min": float(np.mean(gsl == min(OC.GRID_SL))) if len(tr) else None}
    dur = {}
    for cell in (REF,) + B_CELLS[1:]:
        ci_ = OC.cell_index(cell)
        for si, side in enumerate(("long", "short")):
            rows = elig[np.isin(ds.period[elig], ("TRAIN", "INNER_A", "INNER_B"))]
            eb = t["exit_bar"][rows, si, ci_]
            ok = eb >= 0
            fill = row_bar[rows] + 1
            h = (bar_ns[np.clip(eb, 0, len(bar_ns) - 1)] - bar_ns[np.clip(fill, 0, len(bar_ns) - 1)]) / 3.6e12
            h = h[ok]
            dur[f"tp{cell[0]:g}_sl{cell[1]:g}_{side}"] = {"median_h": float(np.median(h)), "p90_h": float(np.quantile(h, 0.9)),
                                                          "share_longer_than_window": float(np.mean(h > WINDOW_HOURS))}
    dg["window_vs_horizons"] = {"window_hours": WINDOW_HOURS, "resolution_hours": dur}
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({"out": a.out, "seconds": out["seconds"]}))


def _build(a):
    from app.short_horizon.expert_guided import data as DG
    from app.short_horizon.expert_guided import env as E
    from app.short_horizon.expert_guided import outcomes as OC
    from tools.expert_guided_e1t import Ctx
    from tools.expert_guided_e3m import effective_contracts
    rec = json.loads(Path(a.recipe).read_text())
    feats = [c["store_column"] for c in rec["tensor"]["columns"]]
    names = [c.get("tensor_name", c["store_column"]) for c in rec["tensor"]["columns"]]
    ds = DG.load_store(a.admitted, a.store, feats, scaling="fixed_train", contract=a.split_contract, recipe=rec)
    cx = Ctx(ds)
    SCFG, _ = effective_contracts("sac", 2.0)
    t = OC.load_table(a.outcomes, cx, SCFG)
    return ds, cx, SCFG, t, None, names, E, OC


# ----------------------------------------------------------------------------- report
def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _c(c):
    return "[—]" if not c else f"[{_f(c[0])}, {_f(c[1])}]"


def markdown(doc: dict) -> str:
    L = ["# Signal-loss locator: where in our pipeline is signal lost? (2026-10-26)", "",
         f"Pre-registration: `{doc['prereg']}`. Same rows / targets / folds for every stage; INNER_B scored once per fold; "
         "paired day-block bootstrap (block 5, 2 000 draws, seed 20261026). Verdicts are per stage, never about the market. "
         "S1 numbers were already seen before the pre-registration (declared); everything else is blind. TEST closed.", "",
         "## Stage table (INNER_B; AUC for a/d, Spearman for b; point [95 % CI] · verdict)", "",
         "| target | fold | S1 | S1b | S2 | S3 (per-seed mean) | S5 (ensemble) | suspect drops |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in sorted({n for fd in doc["folds"] for n in fd["targets"]}):
        for fd in doc["folds"]:
            r = fd["targets"].get(name)
            if not r or "skipped" in r:
                continue
            m = "spearman" if r["kind"] == "regression" else "auc"

            def cell(st):
                if st not in r["point"]:
                    return "—"
                v = r["verdicts"][st]
                return f"{_f(r['point'][st].get(m))} {_c(r['ci'][st].get(m))} {'✓' if v == 'SIGNAL_PRESENT' else '✗'}"
            s3 = "; ".join(f"{k[3:]} {cell(k)}" for k in r["point"] if k.startswith("S3:"))
            s5 = "; ".join(f"{k[3:]} {cell(k)}" for k in r["point"] if k.startswith("S5:"))
            sus = "; ".join(f"{d['from']}→{d['to']} Δ {_c(d['delta_ci'])}" for d in r["drops"] if d["flag"] == "SUSPECT_INTERNAL_DEFECT")
            L.append(f"| {name} | {fd['fold']} | {cell('S1')} | {cell('S1b')} | {cell('S2')} | {s3 or '—'} | {s5 or '—'} | {sus or '—'} |")
    L += ["", "✓ = SIGNAL_PRESENT at that stage (AUC lower CI > 0.52, Spearman lower CI > 0.02); ✗ = ABSENT_AT_THIS_STAGE. "
          "S4 (calibrated heads) has the same AUC as S3 by construction (Platt is monotone); its Brier skill is in the JSON.", ""]
    L += ["## S6: the strategy's opened trades vs the same target (a: TP before SL at 1.5σ/1.5σ), INNER_B", "",
          "| target | fold | family | opened (labelled) | precision opened | top-k S1 | top-k S5 | Δ opened − S1 top-k [CI] | base rate | first vs best-of-day (S5): opened / best [Δ CI] |",
          "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |"]
    for fd in doc["folds"]:
        for name, r in fd["targets"].items():
            for fam, s in (r.get("S6") or {}).items():
                if not s.get("opened_with_label"):
                    L.append(f"| {name} | {fd['fold']} | {fam} | 0 | — | — | — | — | — | — |")
                    continue
                fv = (s.get("first_vs_best_same_day") or {}).get(f"S5:{fam}") or {}
                L.append(f"| {name} | {fd['fold']} | {fam} | {s['opened_with_label']} | {_f(s['precision_opened'])} | {_f(s.get('precision_top_k_S1'))} | "
                         f"{_f(s.get(f'precision_top_k_S5:{fam}'))} | {_c(s.get('delta_opened_minus_S1_topk_ci'))} {s.get('flag_vs_S1', '')} | {_f(s['base_rate_inner_b'])} | "
                         f"{_f(fv.get('precision_opened'))} / {_f(fv.get('precision_best_same_day'))} {_c(fv.get('delta_best_minus_opened_ci'))} |")
    L += ["", "## Diagnostics", ""]
    for fd in doc["folds"]:
        dg = fd["diagnostics"]
        L.append(f"### {fd['fold']}")
        for fam, td in dg.get("training_dynamics", {}).items():
            bes = [h["best_epoch"] for h in td["heads"] if h["best_epoch"] is not None]
            L.append(f"- training ({fam}): best epoch median {np.median(bes) if bes else '—'} (min {min(bes) if bes else '—'}, max {max(bes) if bes else '—'}); "
                     f"share ≤ 3: {_f(td['share_best_epoch_le_3'])}; {td['learning_curves']}")
        for fam, mm in dg.get("params_vs_rows", {}).items():
            L.append(f"- params vs rows ({fam}): " + "; ".join(f"{m} {v['params']} / {v['rows_train_round0_median']}" for m, v in mm.items()))
        for fam, gb in dg.get("grid_bounds", {}).items():
            L.append(f"- TP/SL of opened trades ({fam}, {gb['trades']} trades): TP σ p10/p50/p90 {gb['tp_sigma_p10_p50_p90']}, SL σ {gb['sl_sigma_p10_p50_p90']}; "
                     f"share at grid max TP {_f(gb['share_tp_at_grid_max'])}, SL {_f(gb['share_sl_at_grid_max'])}; at min TP {_f(gb['share_tp_at_grid_min'])}, SL {_f(gb['share_sl_at_grid_min'])}")
        wv = dg.get("window_vs_horizons") or {}
        if wv:
            L.append(f"- window {wv['window_hours']} h vs resolution hours: " + "; ".join(f"{k} median {_f(v['median_h'], 1)} p90 {_f(v['p90_h'], 1)} (> window {_f(v['share_longer_than_window'], 2)})"
                                                                                          for k, v in wv["resolution_hours"].items()))
        for name, r in fd["targets"].items():
            s1 = (r.get("info") or {}).get("S1") or {}
            if "train_auc_in_sample" in s1:
                L.append(f"- S1 capacity vs drift {name}: TRAIN AUC (in-sample) {_f(s1['train_auc_in_sample'])}, INNER_A {_f(s1.get('inner_a_auc'))}, INNER_B {_f(r['point']['S1'].get('auc'))}")
        L.append("")
    return "\n".join(L) + "\n"


def cmd_report(a):
    folds = [json.loads(Path(p).read_text()) for p in a.folds]
    doc = {"prereg": "docs/short_horizon/clean_round/SIGNAL_LOSS_LOCATOR_PREREGISTRATION_2026_10_26.md", "folds": folds}
    Path(a.out_json).write_text(json.dumps(doc, indent=1))
    Path(a.out_md).write_text(markdown(doc))
    sus = [(fd["fold"], n, d["from"], d["to"]) for fd in folds for n, r in fd["targets"].items() for d in r.get("drops", []) if d["flag"] == "SUSPECT_INTERNAL_DEFECT"]
    print(json.dumps({"suspect_drops": sus[:60], "n_suspect": len(sus)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    for k in ("admitted", "store", "recipe", "split_contract", "outcomes", "results", "trades", "out"):
        r.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
    r.set_defaults(fn=cmd_run)
    rp = sp.add_parser("report")
    rp.add_argument("--folds", nargs="+", required=True)
    rp.add_argument("--out-json", required=True)
    rp.add_argument("--out-md", required=True)
    rp.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

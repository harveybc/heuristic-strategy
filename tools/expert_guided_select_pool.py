#!/usr/bin/env python3
"""In-model feature selection over the layer-1 pool (owner 2026-10-22): selection WITH a model, not with a lag-0
univariate filter. DEVELOPMENT; TRAIN 2020-2022 only; no 2023/2024 row is read here.

Rows: observable flat origins of TRAIN (entry gate open). Features: the store's columns with the documented
preprocessing decision applied (``f@diff`` / ``f@asinh`` replace the raw level where the diagnostics decided so).
Labels (three tasks, all hindsight, labels only):
  opener   the expert's anticipatory 3-class pivot label (NONE / valley=LONG / peak=SHORT)   -> what the openers learn
  hit      a +/-2 sigma barrier is touched before the 72-bar timeout                         -> activity
  up       upper barrier first, among hits                                                   -> direction
Model: HistGradientBoosting (depth 4, min leaf 200, 300 iterations, lr 0.05) under PURGED blocked CV (6 contiguous
folds, 72-bar embargo on both sides). Importance: GROUPED permutation (all columns of a group permuted together on
the validation fold; loss increase vs the unpermuted fold), then per-feature permutation inside the groups that
matter. A group matters when its mean loss increase over folds is positive with a lower 90 % bound above 0.
Outputs FEATURE_LIST_top.json (top-K features by per-feature importance summed over the three tasks, K = 40),
FEATURE_LIST_core15.json (the baseline), FEATURE_LIST_core15_plus_top.json, SELECTION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from tools.expert_guided_e1t import STUDENT_CFG, Ctx  # noqa: E402
from tools.expert_guided_e3m import collect  # noqa: E402

CORE15 = DG.FEATURES
FOLDS, EMBARGO, K_TOP = 6, 72, 40


def subgroup(c: str) -> str:
    base = c.split("@")[0]
    if base.startswith(("fx.", "dly.", "fac.", "kal.", "cot.", "btc.")):
        parts = base.split(".")
        return ".".join(parts[:2]) if len(parts) > 2 else parts[0]
    if base.startswith("px."):
        return "px." + base.split(".")[1].rstrip("_0123456789") if "_" in base else "px"
    if base.startswith(("cal.", "hod_", "dow_")):
        return "calendar"
    if base.startswith(("cdl.", "ev.", "ssa.", "act.")):
        return base.split(".")[0]
    if "_w" in base and any(k in base for k in ("hurst", "variogram", "box_")):
        return "fractal"
    return "inventory"


def purged_folds(n: int, k: int = FOLDS, embargo: int = EMBARGO):
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        va = np.arange(edges[i], edges[i + 1])
        tr = np.r_[np.arange(0, max(edges[i] - embargo, 0)), np.arange(min(edges[i + 1] + embargo, n), n)]
        yield tr, va


def fit_eval(X, y, tr, va, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    m = HistGradientBoostingClassifier(max_depth=4, min_samples_leaf=200, max_iter=300, learning_rate=0.05, random_state=seed)
    m.fit(X[tr], y[tr])
    return m, log_loss(y[va], m.predict_proba(X[va]), labels=np.unique(y))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-top", type=int, default=K_TOP)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((Path(a.store).parent / "MANIFEST.json").read_text())
    dec = man["decisions"]
    cols = [f"{c}@{d.lower()}" if d != "NONE" else c for c, d in dec.items() if c not in ("OPEN", "HIGH", "LOW", "CLOSE")]
    store = pd.read_parquet(a.store, columns=["timestamp"] + cols)
    sts = pd.DatetimeIndex(pd.to_datetime(store["timestamp"], utc=True))
    ds = DG.load(a.admitted, a.csv_dir)                      # labels and the admitted grid (scaling irrelevant here)
    cx = Ctx(ds)
    cx.o, cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    v = ds.bars.index.as_unit("ns").asi8
    seg = ds.cal.assign(ds.bars.index)
    ff = np.array([s.force_flat.value for s in ds.cal.segments], np.int64)
    flat_bar = np.array([int(np.searchsorted(v + DG.H_NS, ff[int(seg[b])], side="right")) - 1 if seg[b] >= 0 else len(v) for b in range(len(v))])
    hold = lambda *x: (E.HOLD, 1.5, 2.0)  # noqa: E731
    vis = [vv for vv in collect(cx, "TRAIN", hold, STUDENT_CFG)[1] if vv[2][-1] > 0.5]
    bars_t = np.array([vv[1] for vv in vis], int)
    # barrier outcomes (hit / up) and the opener label at those bars; labels must be final before 2023-01-01
    from tools.expert_guided_e4q import candidate_results  # 32-candidate results; use the (long 2/2) and (short 2/2) slots
    rows = np.array([vv[0] for vv in vis], int)
    cx.flat_bar = flat_bar
    Y, fin = candidate_results(cx, rows, int(np.searchsorted(v, DG._utc_ns("2023-01-01"))))
    from tools.expert_guided_e4q import CANDS
    jl, js = CANDS.index((1, 2.0, 2.0)), CANDS.index((-1, 2.0, 2.0))
    hit = np.where(fin, ((Y[:, jl] > 0) | (Y[:, js] > 0)).astype(float), np.nan)
    up = np.where(fin & (hit == 1), (Y[:, jl] > 0).astype(float), np.nan)
    opener = ds.y_hind[bars_t].astype(float)
    sup = ds.support_end[bars_t]
    opener[(sup < 0) | (sup >= int(np.searchsorted(v, DG._utc_ns("2023-01-01"))))] = np.nan
    # features at the decision bar (store row = bar start)
    pos = pd.Series(np.arange(len(sts)), index=sts)
    idx = pos.reindex(ds.bars.index[bars_t]).to_numpy()
    ok = np.isfinite(idx)
    Xall = store[cols].to_numpy(float)[idx[ok].astype(int)]
    med = np.nanmedian(Xall, 0)
    Xall = np.where(np.isfinite(Xall), Xall, med)             # HGB handles NaN, but keep the permutation clean
    groups = {c: subgroup(c) for c in cols}
    gnames = sorted(set(groups.values()))
    results = {}
    for task, y in (("opener", opener[ok]), ("hit", hit[ok]), ("up", up[ok])):
        m_ = np.isfinite(y)
        X, yy = Xall[m_], y[m_].astype(int)
        base_losses, gimp, fimp = [], {g: [] for g in gnames}, {c: [] for c in cols}
        for fi, (tr, va) in enumerate(purged_folds(len(yy))):
            model, base = fit_eval(X, yy, tr, va, fi)
            base_losses.append(base)
            rng = np.random.default_rng(fi)
            from sklearn.metrics import log_loss
            for g in gnames:
                cidx = [i for i, c in enumerate(cols) if groups[c] == g]
                Xp = X[va].copy()
                Xp[:, cidx] = Xp[rng.permutation(len(va))][:, cidx]
                gimp[g].append(log_loss(yy[va], model.predict_proba(Xp), labels=np.unique(yy)) - base)
            for i, c in enumerate(cols):
                Xp = X[va].copy()
                Xp[:, i] = Xp[rng.permutation(len(va)), i]
                fimp[c].append(log_loss(yy[va], model.predict_proba(Xp), labels=np.unique(yy)) - base)
        prior = np.bincount(yy) / len(yy)
        prior_ll = float(-np.mean(np.log(prior[yy])))
        gt = {g: {"mean": float(np.mean(v_)), "lo90": float(np.mean(v_) - 1.645 * np.std(v_, ddof=1) / np.sqrt(len(v_)))} for g, v_ in gimp.items()}
        results[task] = {"rows": int(len(yy)), "class_counts": np.bincount(yy).tolist(), "cv_log_loss": float(np.mean(base_losses)),
                         "prior_log_loss": prior_ll, "cv_minus_prior": float(np.mean(base_losses) - prior_ll),
                         "groups": gt, "groups_that_matter": sorted([g for g, t in gt.items() if t["lo90"] > 0], key=lambda g: -gt[g]["mean"]),
                         "features": {c: float(np.mean(v_)) for c, v_ in fimp.items()}}
        print(json.dumps({"task": task, "rows": int(len(yy)), "cv_minus_prior": round(results[task]["cv_minus_prior"], 4),
                          "groups_that_matter": results[task]["groups_that_matter"][:8], "s": round(time.time() - t0)}), flush=True)
    # a task whose purged-CV loss is not below the prior carries no out-of-fold information: its permutation
    # importances measure overfit reliance, not signal, so it does not contribute to the ranking
    informative = [t for t in results if results[t]["cv_minus_prior"] < 0]
    score = {c: sum(results[t]["features"][c] for t in informative) for c in cols}
    top = [c for c, _ in sorted(score.items(), key=lambda kv: -kv[1])[: a.k_top]]
    core = [c for c in cols if c.split("@")[0] in CORE15]
    lists = {"top": top, "core15": core, "core15_plus_top": core + [c for c in top if c not in core]}
    for k, feats in lists.items():
        (out / f"FEATURE_LIST_{k}.json").write_text(json.dumps({"name": k, "features": feats, "store_sha256": man["file_sha256"], "rule": __doc__}, indent=1))
    (out / "SELECTION.json").write_text(json.dumps({"results": results, "informative_tasks": informative, "score": score, "lists": lists,
                                                    "wall": time.time() - t0}, indent=1))
    L = ["# In-model selection over the layer-1 pool (TRAIN 2020-2022, purged CV)", "",
         f"Tasks that beat the prior out of fold (the only ones that rank features): {informative}", ""]
    for t, r in results.items():
        L += [f"## task {t}: rows {r['rows']}, CV log-loss − prior = {r['cv_minus_prior']:+.4f}", "",
              "| group | mean Δloss | lo90 | matters |", "|---|---|---|---|"]
        for g, v_ in sorted(r["groups"].items(), key=lambda kv: -kv[1]["mean"]):
            L.append(f"| {g} | {v_['mean']:+.5f} | {v_['lo90']:+.5f} | {'yes' if v_['lo90'] > 0 else ''} |")
        L.append("")
    L += [f"## Top features (summed permutation Δloss over the informative tasks {informative})", "", "| rank | feature | Δloss sum |", "|---|---|---|"]
    for i, c in enumerate(top, 1):
        L.append(f"| {i} | {c} | {score[c]:+.5f} |")
    (out / "SELECTION.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"top10": top[:10], "wall": round(time.time() - t0)}))


if __name__ == "__main__":
    main()

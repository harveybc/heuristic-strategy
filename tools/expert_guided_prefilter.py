#!/usr/bin/env python3
"""Strict pre-filter before the floating selection (owner 2026-10-26): tighten the EXISTING metric + causal-stage
evidence of selection v5 so the SBFS starts from roughly half of the 121 recipe-v2 candidates. Only existing
artifacts and TRAIN-row measurements are used; no model is fitted here. Every threshold is declared in RULES before
the data is read, and the sensitivity grid reports the count under the alternative thresholds.

Rules (applied in this order, each on the survivors of the previous one):
  R1 constants: TRAIN std == 0 (or not finite)                                 -> drop
  R2 duplicates: R^2 >= R2_T between two columns (TRAIN, complete rows) joins a cluster; one representative per
     cluster is kept: the column whose v5 family has the larger mean opener3 gain (RANKING.csv, hgb); tie -> the
     column that comes first in the recipe order; coherent blocks (sin/cos pairs, rg.gmm.p0/p1/p2) move together
  R3 date/regime proxies: single-feature adversarial AUC (TRAIN 2020-2022 vs 2023-H1, both inside TRAIN) >= R3_T
     -> drop, except declared calendar encodings (hod/dow/cal.* sin-cos and session dummies), kept by design
  R4 family evidence: families whose mean delta_loss (hgb, RANKING.csv) is <= 0 on EVERY task -> drop all columns;
     families RESCUED_INTERACTION are kept whatever their gains
  R5 recorded parent-vs-component contrast stage (CONTRASTS_PARENT_COMPONENTS.csv): a component column whose paired
     gain over the family's 'none' baseline (same task/family/ctx/fold, hgb) is <= 0 on BOTH informative tasks
     (opener3 and hit) -> drop; a coherent block is dropped only when every member is <= 0 on both tasks. v5 has NO
     per-column causal or lagged-dependence verdict (its decision evidence kinds are block_permutation, interaction,
     retrained_contrast at family level), so this is the only column-level stage evidence that exists; stated as such
  R6 within-family redundancy: inside a surviving family with k > 4 columns, keep the number of columns equal to the
     PCA component count explaining R6_V of the family's TRAIN variance, chosen by variance-weighted squared loading
     (blocks as units), never below 2 columns
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RULES = {"R2_T": 0.99, "R3_T": 0.75, "R6_V": 0.99, "informative_tasks": ("opener3", "hit"), "model": "hgb",
         "train_end": "2023-07-01", "r3_split": "2023-01-01", "r3_calendar_prefixes": ("hod_", "dow_", "cal.")}
GRID = {"R3_T": (0.70, 0.75, 0.80), "R6_V": (0.98, 0.99, 0.995)}
# component arm name -> recipe column, per decomposition family (only columns present in the recipe are mapped)
COMPONENT_MAP = {
    "wavelet/log_close": {"C:A4": "wv.lc.A4@diff", "C:D1": "wv.lc.D1", "C:D2": "wv.lc.D2", "C:D3": "wv.lc.D3", "C:D4": "wv.lc.D4",
                          "C:energy_D1": "wv.lc.energy_D1@diff", "C:energy_D2": "wv.lc.energy_D2@log", "C:energy_D3": "wv.lc.energy_D3@log", "C:energy_D4": "wv.lc.energy_D4@log"},
    "wavelet/logret_1": {"C:A4": "wv.r1.A4", "C:D1": "wv.r1.D1", "C:D2": "wv.r1.D2", "C:D3": "wv.r1.D3", "C:D4": "wv.r1.D4",
                         "C:energy_D1": "wv.r1.energy_D1@diff", "C:energy_D2": "wv.r1.energy_D2@log", "C:energy_D3": "wv.r1.energy_D3@log", "C:energy_D4": "wv.r1.energy_D4@log"},
    "hilbert/logret_1": {"C:phase_cos": "hb.r1.phase_cos", "C:phase_sin": "hb.r1.phase_sin", "C:inst_freq": "hb.r1.inst_freq"},
    "hilbert/log_close": {"C:inst_freq": "hb.lc_detr.inst_freq"},
    "strength/fx.eur_strength.ret_1": {"C:phase_cos": "hb.eur.phase_cos", "C:phase_sin": "hb.eur.phase_sin", "C:inst_freq": "hb.eur.inst_freq"},
    "strength/fx.usd_strength.ret_1": {"C:A4": "wv.usd.A4"},
    "kalman/log_close": {"C:filtered_level_minus_observation": "kf.lc.level_minus_close"},
    "ssa/BidClose": {"C:ssa.trend_minus_close_pips": "ssa.trend_minus_close_pips", "C:ssa.c2_pips": "ssa.c2_pips", "C:ssa.c3_pips": "ssa.c3_pips"},
    "gmm/logret_1+rv_24+log_range": {"C:posterior_state_0_(vol_rank_0)": "rg.gmm.p0", "C:posterior_state_1_(vol_rank_1)": "rg.gmm.p1",
                                     "C:posterior_state_2_(vol_rank_2)": "rg.gmm.p2", "C:posterior_entropy": "rg.gmm.entropy"},
}


# ----------------------------------------------------------------------------------------------- structure helpers
def family_of(group: str) -> str:
    return group.split("/")[0] if group.startswith("px.close_loc/") else group


def coherent_blocks(cols) -> dict:
    """column -> block id; sin/cos pairs and the rg.gmm.p0/p1/p2 triplet are one block; everything else is alone."""
    blocks = {}
    for c in cols:
        if c.endswith("_sin") or c.endswith("_cos"):
            blocks[c] = c[:-4]
        elif c in ("rg.gmm.p0", "rg.gmm.p1", "rg.gmm.p2"):
            blocks[c] = "rg.gmm.p"
        else:
            blocks[c] = c
    return blocks


def expand_blocks(drop: set, blocks: dict, cols) -> set:
    ids = {blocks[c] for c in drop}
    return {c for c in cols if blocks[c] in ids}


# ----------------------------------------------------------------------------------------------- rules (pure)
def r1_constants(std: dict) -> set:
    return {c for c, s in std.items() if s is None or not np.isfinite(s) or s == 0}


def r2_duplicates(cols, r2: np.ndarray, fam: dict, fam_gain: dict, blocks: dict, threshold: float):
    """Union-find clusters at r2 >= threshold; keep one representative per cluster. Returns (dropped, clusters)."""
    n = len(cols)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if r2[i, j] >= threshold:
                parent[find(i)] = find(j)
    clusters = defaultdict(list)
    for i, c in enumerate(cols):
        clusters[find(i)].append(c)
    dropped, out = set(), []
    for members in clusters.values():
        if len(members) < 2:
            continue
        keep = sorted(members, key=lambda c: (-fam_gain.get(fam[c], 0.0), cols.index(c)))[0]
        drop = {c for c in members if c != keep}
        out.append({"members": members, "keep": keep, "drop": sorted(drop)})
        dropped |= drop
    return expand_blocks(dropped, blocks, cols) - {m["keep"] for m in out}, out


def r3_proxies(auc: dict, threshold: float, calendar_prefixes) -> set:
    return {c for c, a in auc.items() if a is not None and a >= threshold and not c.startswith(tuple(calendar_prefixes))}


def r4_family_evidence(fam: dict, task_gain: dict, status: dict) -> set:
    """task_gain: family -> {task: mean delta_loss}; drop families <= 0 on every task unless RESCUED_INTERACTION."""
    dead = {f for f, g in task_gain.items() if g and all(v <= 0 for v in g.values()) and status.get(f) != "RESCUED_INTERACTION"}
    return {c for c, f in fam.items() if f in dead}


def r5_components(comp_gain: dict, blocks: dict, cols, tasks) -> set:
    """comp_gain: column -> {task: mean paired gain}; drop when <= 0 on every informative task; blocks need every
    member dead."""
    dead = {c for c, g in comp_gain.items() if all(g.get(t, 0.0) <= 0 for t in tasks)}
    out = set()
    for c in dead:
        members = [m for m in cols if blocks[m] == blocks[c]]
        if all(m in dead for m in members):
            out |= set(members)
    return out


def r6_pca(fam_cols: dict, Z: dict, blocks: dict, variance: float) -> tuple:
    """fam_cols: family -> surviving columns (recipe order); Z: column -> standardized TRAIN vector (same rows).
    Keeps, per family with > 4 columns, k = #components for `variance`, chosen by block importance; floor 2."""
    dropped, detail = set(), {}
    for f, cs in fam_cols.items():
        if len(cs) <= 4:
            continue
        M = np.column_stack([Z[c] for c in cs])
        M = M - M.mean(0)
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
        var = s ** 2 / np.sum(s ** 2)
        k = int(np.searchsorted(np.cumsum(var), variance) + 1)
        k = max(2, min(k, len(cs)))
        imp = {c: float(np.sum((Vt[:, i] ** 2) * var)) for i, c in enumerate(cs)}
        ids = sorted({blocks[c] for c in cs}, key=lambda b: -sum(imp[c] for c in cs if blocks[c] == b))
        keep = []
        for b in ids:
            if len(keep) >= k:
                break
            keep += [c for c in cs if blocks[c] == b]
        drop = [c for c in cs if c not in keep]
        dropped |= set(drop)
        detail[f] = {"k": k, "kept": keep, "dropped": drop, "importance": imp, "cum_var": [float(x) for x in np.cumsum(var)]}
    return dropped, detail


# ----------------------------------------------------------------------------------------------- artifact readers
def read_ranking(path, model):
    agg = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["model"] == model:
            agg[(r["family"], r["task"])].append(float(r["delta_loss"]))
    task_gain = defaultdict(dict)
    for (f, t), v in agg.items():
        task_gain[f][t] = float(np.mean(v))
    return dict(task_gain)


def read_component_gains(path, model, tasks):
    rows = [r for r in csv.DictReader(open(path)) if r["model"] == model and r["task"] in tasks]
    base = {(r["task"], r["family"], r["ctx"], r["fold"]): float(r["val_loss"]) for r in rows if r["arm"] == "none"}
    g = defaultdict(list)
    for r in rows:
        if r["arm"].startswith("C:") and (r["task"], r["family"], r["ctx"], r["fold"]) in base:
            g[(r["family"], r["arm"], r["task"])].append(base[(r["task"], r["family"], r["ctx"], r["fold"])] - float(r["val_loss"]))
    out = defaultdict(dict)
    for (f, arm, t), v in g.items():
        col = COMPONENT_MAP.get(f, {}).get(arm)
        if col:
            out[col][t] = float(np.mean(v))
    return dict(out)


def measure_train(store, cols, train_end, r3_split):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(store, columns=cols + ["timestamp"])
    ts = pd.to_datetime(df.pop("timestamp"), utc=True)
    m = ts < pd.Timestamp(train_end, tz="UTC")
    X, ts = df[m], ts[m]
    std = {c: float(X[c].std(skipna=True)) for c in cols}
    full = X.dropna()
    Zf = (full - full.mean()) / full.std().replace(0, np.nan)
    Zf = Zf.fillna(0.0)
    C = np.corrcoef(Zf.to_numpy().T)
    r2 = np.nan_to_num(C ** 2)
    np.fill_diagonal(r2, 0.0)
    y = (ts >= pd.Timestamp(r3_split, tz="UTC")).to_numpy().astype(int)
    auc = {}
    for c in cols:
        v = X[c].to_numpy()
        ok = np.isfinite(v)
        if ok.sum() < 100 or len(set(y[ok])) < 2 or np.nanstd(v) == 0:
            auc[c] = None
            continue
        a = roc_auc_score(y[ok], v[ok])
        auc[c] = float(max(a, 1 - a))
    Z = {c: Zf[c].to_numpy() for c in cols}
    return {"rows_train": int(len(X)), "rows_complete": int(len(full)), "std": std, "r2": r2, "auc": auc, "Z": Z}


# ----------------------------------------------------------------------------------------------- pipeline
def run(cols, fam, status, task_gain, comp_gain, meas, r2_t, r3_t, r6_v, tasks, calendar_prefixes):
    blocks = coherent_blocks(cols)
    fam_gain = {f: g.get("opener3", 0.0) for f, g in task_gain.items()}
    reasons, surv = {}, list(cols)

    def apply(name, drop):
        nonlocal surv
        for c in surv:
            if c in drop:
                reasons[c] = name
        surv = [c for c in surv if c not in drop]
        return len(surv)

    counts = {"start": len(surv)}
    counts["R1"] = apply("R1_constant", r1_constants({c: meas["std"][c] for c in surv}))
    idx = [cols.index(c) for c in surv]
    r2d, clusters = r2_duplicates(surv, meas["r2"][np.ix_(idx, idx)], fam, fam_gain, blocks, r2_t)
    counts["R2"] = apply("R2_duplicate", r2d)
    counts["R3"] = apply("R3_date_proxy", r3_proxies({c: meas["auc"][c] for c in surv}, r3_t, calendar_prefixes))
    counts["R4"] = apply("R4_family_no_evidence", r4_family_evidence({c: fam[c] for c in surv}, task_gain, status))
    counts["R5"] = apply("R5_component_no_gain", r5_components({c: g for c, g in comp_gain.items() if c in surv}, blocks, surv, tasks))
    fam_cols = defaultdict(list)
    for c in surv:
        fam_cols[fam[c]].append(c)
    r6d, r6detail = r6_pca(dict(fam_cols), {c: meas["Z"][c] for c in surv}, blocks, r6_v)
    counts["R6"] = apply("R6_family_redundancy", r6d)
    return {"counts": counts, "survivors": surv, "reasons": reasons, "r2_clusters": clusters, "r6": r6detail}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--selection", required=True, help="selection_v5 directory")
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = json.load(open(a.recipe))
    cols = [c["store_column"] for c in rec["tensor"]["columns"]]
    groups = rec["representation"]["groups"]["groups"]
    fam = {c: family_of(g) for g, cs in groups.items() for c in cs}
    dec = json.load(open(Path(a.selection) / "SHORTLIST.json"))["decisions"]
    status = {f: v["status"] for f, v in dec.items()}
    task_gain = read_ranking(Path(a.selection) / "RANKING.csv", RULES["model"])
    comp_gain = read_component_gains(Path(a.selection) / "CONTRASTS_PARENT_COMPONENTS.csv", RULES["model"], RULES["informative_tasks"])
    meas = measure_train(a.store, cols, RULES["train_end"], RULES["r3_split"])
    res = run(cols, fam, status, task_gain, comp_gain, meas, RULES["R2_T"], RULES["R3_T"], RULES["R6_V"], RULES["informative_tasks"], RULES["r3_calendar_prefixes"])
    grid = {}
    for r3t in GRID["R3_T"]:
        for r6v in GRID["R6_V"]:
            grid[f"R3_T={r3t}|R6_V={r6v}"] = run(cols, fam, status, task_gain, comp_gain, meas, RULES["R2_T"], r3t, r6v, RULES["informative_tasks"], RULES["r3_calendar_prefixes"])["counts"]["R6"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"rule": RULES, "recipe_identity": rec.get("identity"), "rows_train": meas["rows_train"], "rows_complete": meas["rows_complete"],
           "counts": res["counts"], "candidates": res["survivors"], "n": len(res["survivors"]),
           "dropped": {c: {"reason": res["reasons"][c], "family": fam[c], "adv_auc": meas["auc"][c], "train_std": meas["std"][c], "component_gain": comp_gain.get(c)} for c in cols if c not in res["survivors"]},
           "r2_clusters": res["r2_clusters"], "r6_detail": {f: {k: v for k, v in d.items() if k != "importance"} for f, d in res["r6"].items()},
           "adv_auc_top": sorted([(c, a) for c, a in meas["auc"].items() if a], key=lambda t: -t[1])[:10],
           "component_gains": comp_gain, "family_task_gain": task_gain, "family_status": status, "sensitivity": grid}
    doc["sha256_candidates"] = hashlib.sha256(json.dumps(res["survivors"]).encode()).hexdigest()
    (out / "CANDIDATES_STRICT.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"counts": res["counts"], "n": len(res["survivors"]), "sensitivity": grid}, indent=1))


if __name__ == "__main__":
    main()

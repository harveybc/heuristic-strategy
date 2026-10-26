#!/usr/bin/env python3
"""Strict pre-filter before the floating selection, v2 (owner 2026-10-26; Musashi's eight corrections applied).

Two classes of discard:
  DEFINITIVE (removed from the SBFS universe) only for demonstrated defects:
    D1 constants: TRAIN std == 0 among finite values (finite count, missing share and per-block constancy recorded);
       a column with no finite value is ABSENT (its own class), not constant
    D2 exact duplicates: max |x - y| == 0, or an affine identity y = a*x + b with max residual < EXACT_RESIDUAL;
       one representative per exact cluster (ordering rule below), the others are DEFINITIVE
  REVERSIBLE (tier B = excluded from the initial backward set, INCLUDED in the floating reinclusion pool):
    B1 near duplicates: Pearson^2 >= NEAR_R2 on the common support (per-period stability of the R^2 recorded)
    B2 temporal drift ALERT: single-column AUC(2020-2022 vs 2023-H1, both inside TRAIN) >= DRIFT_AUC AND the column
       is a raw LEVEL (not a change / ratio / oscillator / calendar encoding, see LEVEL_EXCLUDE_TOKENS); per-year
       origin/stability recorded; never automatic for calendar encodings or volatility ratios
    B3 family evidence: families with no positive mean delta_loss on any task (RANKING.csv, hgb) are ordered last and
       go to tier B; otherwise family gains ORDER the columns, they never exclude
    B4 (not applied): the parent-vs-component contrast (CONTRASTS_PARENT_COMPONENTS.csv) ORDERS components inside
       their family; it excludes nothing. v5 has no per-column relationships verdict with power/lag assumptions,
       so no "no dependence" discard exists
    B5 within-family redundancy: greedy FORWARD selection of original columns (coherent blocks as units) until
       FAMILY_VAR of the family's STANDARDIZED TRAIN variance (each column z-scored on the common support, so price
       units cannot dominate) is explained by projection on the selected columns; the unselected columns are tier B;
       variance != trading utility, stated
Every statistic is computed on the FIT block only (TRAIN 2020-01-01 -> 2023-07-01; the drift split is inside it) on
one frozen common support (rows complete in every candidate column; its hash is recorded). 60 is a budget guidance,
never a target reached by tuning: every threshold is enumerated with its sensitivity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RULES = {"EXACT_RESIDUAL": 1e-9, "NEAR_R2": 0.99, "DRIFT_AUC": 0.75, "FAMILY_VAR": 0.99, "FAMILY_MIN_COLS": 2, "FAMILY_B5_MIN_SIZE": 5,
         "informative_tasks": ("opener3", "hit"), "model": "hgb", "train_end": "2023-07-01", "drift_split": "2023-01-01",
         "periods": (("2020", "2020-01-01", "2021-01-01"), ("2021", "2021-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"), ("2023H1", "2023-01-01", "2023-07-01"))}
GRID = {"NEAR_R2": (0.95, 0.99, 0.999), "DRIFT_AUC": (0.70, 0.75, 0.80), "FAMILY_VAR": (0.90, 0.98, 0.99, 0.995)}
# named profiles, declared before any run: the budget profile feeds the SBFS start set (tier A); everything it moves
# goes to tier B (reinclusion pool), nothing is deleted. Search cost grows about quadratically with |A|.
PROFILES = {"strict_v2_default": {"NEAR_R2": 0.99, "FAMILY_VAR": 0.99},
            "strict_v2_budget": {"NEAR_R2": 0.95, "FAMILY_VAR": 0.90}}
SBFS_START_PROFILE = "strict_v2_budget"
# a column is a raw LEVEL unless its name carries one of these tokens (changes, ratios, oscillators, encodings, bounded)
LEVEL_EXCLUDE_TOKENS = ("@diff", "ret", "chg", "change", "ratio", "logret", "z156", "_sin", "_cos", "session", "hours_", "energy", "phase", "inst_freq",
                        "slope", "entropy", "rg.gmm.p", "p_hit", "close_loc", "px.er_", "rv_", "position", ".D1", ".D2", ".D3", ".D4", "@asinh",
                        "RSI", "CCI", "ROC", "Stochastic", "WilliamsR", "price_minus_ema", "minus_close")
CALENDAR_PREFIXES = ("hod_", "dow_", "cal.")


def family_of(group: str) -> str:
    return group.split("/")[0] if group.startswith("px.close_loc/") else group


def coherent_blocks(cols) -> dict:
    blocks = {}
    for c in cols:
        if c.endswith("_sin") or c.endswith("_cos"):
            blocks[c] = c[:-4]
        elif c in ("rg.gmm.p0", "rg.gmm.p1", "rg.gmm.p2"):
            blocks[c] = "rg.gmm.p"
        else:
            blocks[c] = c
    return blocks


def is_level(col: str) -> bool:
    return not any(t in col for t in LEVEL_EXCLUDE_TOKENS) and not col.startswith(CALENDAR_PREFIXES)


# ----------------------------------------------------------------------------------------------- definitive rules
def d1_constants(X: dict, blocks_per_col: dict | None = None):
    """X: column -> raw TRAIN vector (NaN allowed). Returns (constant set, absent set, record)."""
    const, absent, rec = set(), set(), {}
    for c, v in X.items():
        v = np.asarray(v, float)
        fin = np.isfinite(v)
        n = int(fin.sum())
        rec[c] = {"finite": n, "missing_share": float(1 - n / len(v)) if len(v) else 1.0}
        if n == 0:
            absent.add(c)
            rec[c]["class"] = "ABSENT"
            continue
        s = float(np.std(v[fin]))
        rec[c]["std"] = s
        if blocks_per_col is not None:
            rec[c]["constant_blocks"] = int(sum(1 for b in blocks_per_col[c] if np.isfinite(b).sum() and np.std(b[np.isfinite(b)]) == 0))
        if s == 0:
            const.add(c)
            rec[c]["class"] = "CONSTANT"
    return const, absent, rec


def affine_residual(x, y):
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.max(np.abs(A @ coef - y))), [float(coef[0]), float(coef[1])]


def representative(members, cols, fam, fam_rank):
    """ordering rule: the column of the family ranked first by family gain; tie -> first in the recipe order."""
    return sorted(members, key=lambda c: (fam_rank.get(fam[c], 1e9), cols.index(c)))[0]


def d2_exact_duplicates(cols, Z: dict, fam, fam_rank, blocks, residual):
    """Z: column -> vector on the common support. Exact = max|diff| == 0 or affine residual < residual."""
    n = len(cols)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            x, y = Z[cols[i]], Z[cols[j]]
            if np.max(np.abs(x - y)) == 0:
                pairs.append((cols[i], cols[j], "identical", 0.0))
                parent[find(i)] = find(j)
                continue
            if np.std(x) > 0:
                r, coef = affine_residual(x, y)
                if r < residual:
                    pairs.append((cols[i], cols[j], f"affine y={coef[0]:.6g}x+{coef[1]:.6g}", r))
                    parent[find(i)] = find(j)
    clusters = defaultdict(list)
    for i, c in enumerate(cols):
        clusters[find(i)].append(c)
    dropped, out = set(), []
    for m in clusters.values():
        if len(m) < 2:
            continue
        keep = representative(m, cols, fam, fam_rank)
        out.append({"members": m, "keep": keep, "definitive": sorted(set(m) - {keep})})
        dropped |= set(m) - {keep}
    return dropped, out, pairs


# ----------------------------------------------------------------------------------------------- reversible rules
def b1_near_duplicates(cols, Z: dict, Zp: dict, fam, fam_rank, blocks, threshold):
    """Pearson^2 on the common support; the representative stays in tier A, the others go to tier B; the per-period
    R^2 is recorded for stability. Blocks move together."""
    n = len(cols)
    M = np.column_stack([Z[c] for c in cols])
    C = np.corrcoef(M.T)
    R2 = np.nan_to_num(C ** 2)
    np.fill_diagonal(R2, 0.0)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if R2[i, j] >= threshold:
                per = {p: float(np.corrcoef(Zp[p][cols[i]], Zp[p][cols[j]])[0, 1] ** 2) for p in Zp}
                pairs.append({"a": cols[i], "b": cols[j], "r2": float(R2[i, j]), "r2_per_period": per})
                parent[find(i)] = find(j)
    clusters = defaultdict(list)
    for i, c in enumerate(cols):
        clusters[find(i)].append(c)
    tier_b, out = set(), []
    for m in clusters.values():
        if len(m) < 2:
            continue
        keep = representative(m, cols, fam, fam_rank)
        ids = {blocks[c] for c in m if c != keep}
        b = {c for c in cols if blocks[c] in ids and c != keep}
        out.append({"members": m, "keep": keep, "tier_b": sorted(b)})
        tier_b |= b
    return tier_b, out, pairs


def b2_drift_alert(auc: dict, auc_per_year: dict, threshold):
    """auc: column -> AUC(2020-22 vs 2023-H1); only raw LEVEL columns at/above the threshold go to tier B."""
    flagged, rec = set(), {}
    for c, a in auc.items():
        rec[c] = {"auc": a, "level": is_level(c), "per_year": auc_per_year.get(c)}
        if a is not None and a >= threshold and is_level(c):
            flagged.add(c)
            rec[c]["tier_b"] = True
    return flagged, rec


def b3_family_order(fam: dict, task_gain: dict, status: dict):
    """families ordered by their best task gain; families with no positive gain on any task go to tier B."""
    best = {f: max(g.values()) if g else float("-inf") for f, g in task_gain.items()}
    order = sorted(set(fam.values()), key=lambda f: -best.get(f, float("-inf")))
    rank = {f: i for i, f in enumerate(order)}
    dead = {f for f in set(fam.values()) if best.get(f, float("-inf")) <= 0 and status.get(f) != "RESCUED_INTERACTION"}
    return rank, {c for c, f in fam.items() if f in dead}, best


def b5_forward_variance(fam_cols: dict, Z: dict, blocks: dict, target, min_cols, min_size):
    """Greedy forward selection of original columns (blocks as units) until the family's TRAIN variance explained by
    projection on the selected columns reaches `target`. Returns (tier_b set, detail)."""
    tier_b, detail = set(), {}
    for f, cs in fam_cols.items():
        if len(cs) < min_size:
            continue
        M = np.column_stack([Z[c] for c in cs])
        M = M - M.mean(0)
        total = float(np.sum(M ** 2))
        ids = []
        for c in cs:
            if blocks[c] not in ids:
                ids.append(blocks[c])
        chosen, trace = [], []
        while ids:
            best = None
            for b in ids:
                cand = chosen + [c for c in cs if blocks[c] == b]
                S = M[:, [cs.index(c) for c in cand]]
                coef, *_ = np.linalg.lstsq(S, M, rcond=None)
                ev = 1 - float(np.sum((M - S @ coef) ** 2)) / total
                if best is None or ev > best[0]:
                    best = (ev, b)
            ev, b = best
            chosen += [c for c in cs if blocks[c] == b]
            ids.remove(b)
            trace.append({"block": b, "explained": ev})
            if ev >= target and len(chosen) >= min_cols:
                break
        drop = [c for c in cs if c not in chosen]
        tier_b |= set(drop)
        detail[f] = {"kept": chosen, "tier_b": drop, "trace": trace, "note": "variance explained, not trading utility"}
    return tier_b, detail


# ----------------------------------------------------------------------------------------------- artifact readers
def read_ranking(path, model):
    agg = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["model"] == model:
            agg[(r["family"], r["task"])].append(float(r["delta_loss"]))
    out = defaultdict(dict)
    for (f, t), v in agg.items():
        out[f][t] = float(np.mean(v))
    return dict(out)


def read_component_order(path, model, tasks):
    """component arm gains over the paired 'none' baseline: ORDERING information only."""
    rows = [r for r in csv.DictReader(open(path)) if r["model"] == model and r["task"] in tasks]
    base = {(r["task"], r["family"], r["ctx"], r["fold"]): float(r["val_loss"]) for r in rows if r["arm"] == "none"}
    g = defaultdict(list)
    for r in rows:
        k = (r["task"], r["family"], r["ctx"], r["fold"])
        if (r["arm"].startswith("C:") or r["arm"].startswith("P_") or r["arm"] == "Cgroup") and k in base:
            g[(r["family"], r["arm"])].append(base[k] - float(r["val_loss"]))
    return {f"{f}|{a}": float(np.mean(v)) for (f, a), v in g.items()}


def measure(store, cols, rules):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(store, columns=cols + ["timestamp"])
    ts = pd.to_datetime(df.pop("timestamp"), utc=True)
    m = ts < pd.Timestamp(rules["train_end"], tz="UTC")
    X, ts = df[m].reset_index(drop=True), ts[m].reset_index(drop=True)
    raw = {c: X[c].to_numpy(float) for c in cols}
    blocks_per_col = {c: [X[c][(ts >= pd.Timestamp(a, tz="UTC")) & (ts < pd.Timestamp(b, tz="UTC"))].to_numpy(float) for _, a, b in rules["periods"]] for c in cols}
    ok = np.isfinite(X.to_numpy(float)).all(axis=1)
    common = X[ok]
    ts_c = ts[ok]
    support_hash = hashlib.sha256(np.ascontiguousarray(ts_c.astype("int64").to_numpy()).tobytes()).hexdigest()[:16]
    Z = {c: common[c].to_numpy(float) for c in cols}
    Zp = {}
    for name, a, b in rules["periods"]:
        mm = (ts_c >= pd.Timestamp(a, tz="UTC")) & (ts_c < pd.Timestamp(b, tz="UTC"))
        Zp[name] = {c: common[c][mm].to_numpy(float) for c in cols}
    y = (ts_c >= pd.Timestamp(rules["drift_split"], tz="UTC")).to_numpy().astype(int)
    auc, auc_year = {}, {}
    for c in cols:
        v = Z[c]
        if np.std(v) == 0 or len(set(y)) < 2:
            auc[c] = None
            continue
        a = roc_auc_score(y, v)
        auc[c] = float(max(a, 1 - a))
        per = {}
        names = [p[0] for p in rules["periods"]]
        for i in range(len(names) - 1):
            xa, xb = Zp[names[i]][c], Zp[names[i + 1]][c]
            if len(xa) and len(xb) and (np.std(xa) > 0 or np.std(xb) > 0):
                aa = roc_auc_score(np.r_[np.zeros(len(xa)), np.ones(len(xb))], np.r_[xa, xb])
                per[f"{names[i]}->{names[i+1]}"] = float(max(aa, 1 - aa))
        auc_year[c] = per
    return {"rows_train": int(len(X)), "rows_common": int(ok.sum()), "support_hash": support_hash, "raw": raw, "blocks_per_col": blocks_per_col,
            "Z": Z, "Zp": Zp, "auc": auc, "auc_year": auc_year}


# ----------------------------------------------------------------------------------------------- pipeline
def run(cols, fam, status, task_gain, meas, near_r2, drift_auc, family_var, exact_residual, min_cols, min_size):
    blocks = coherent_blocks(cols)
    fam_rank, b3_cols, best_gain = b3_family_order(fam, task_gain, status)
    const, absent, d1rec = d1_constants({c: meas["raw"][c] for c in cols}, meas["blocks_per_col"])
    universe = [c for c in cols if c not in const | absent]
    d2drop, d2clusters, d2pairs = d2_exact_duplicates(universe, meas["Z"], fam, fam_rank, blocks, exact_residual)
    universe = [c for c in universe if c not in d2drop]
    b1, b1clusters, b1pairs = b1_near_duplicates(universe, meas["Z"], meas["Zp"], fam, fam_rank, blocks, near_r2)
    b2, b2rec = b2_drift_alert({c: meas["auc"][c] for c in universe}, meas["auc_year"], drift_auc)
    b3 = {c for c in universe if c in b3_cols}
    tier_a_pre = [c for c in universe if c not in b1 | b2 | b3]
    fam_cols = defaultdict(list)
    for c in tier_a_pre:
        fam_cols[fam[c]].append(c)
    Zs = {c: (meas["Z"][c] - meas["Z"][c].mean()) / (meas["Z"][c].std() or 1.0) for c in tier_a_pre}
    b5, b5detail = b5_forward_variance(dict(fam_cols), Zs, blocks, family_var, min_cols, min_size)
    tier_a = [c for c in tier_a_pre if c not in b5]
    tier_b = [c for c in universe if c not in tier_a]
    reasons = {}
    for c in cols:
        if c in absent: reasons[c] = "ABSENT"
        elif c in const: reasons[c] = "D1_constant"
        elif c in d2drop: reasons[c] = "D2_exact_duplicate"
        elif c in b1: reasons[c] = "B1_near_duplicate"
        elif c in b2: reasons[c] = "B2_drift_alert_level"
        elif c in b3: reasons[c] = "B3_family_no_evidence"
        elif c in b5: reasons[c] = "B5_family_redundancy"
    counts = {"start": len(cols), "ABSENT": len(absent), "D1_constant": len(const), "D2_exact_duplicate": len(d2drop),
              "B1_near_duplicate": len(b1), "B2_drift_alert_level": len(b2), "B3_family_no_evidence": len(b3), "B5_family_redundancy": len(b5),
              "tier_A": len(tier_a), "tier_B": len(tier_b), "universe_after_definitive": len(universe)}
    return {"counts": counts, "tier_a": tier_a, "tier_b": tier_b, "reasons": reasons, "d1": d1rec, "d2_clusters": d2clusters, "d2_pairs": d2pairs,
            "b1_clusters": b1clusters, "b1_pairs": b1pairs, "b2": b2rec, "b3_family_rank": fam_rank, "b3_best_gain": best_gain, "b5": b5detail}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--selection", required=True)
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
    comp_order = read_component_order(Path(a.selection) / "CONTRASTS_PARENT_COMPONENTS.csv", RULES["model"], RULES["informative_tasks"])
    meas = measure(a.store, cols, RULES)
    R = RULES
    profiles = {}
    for name, pv in PROFILES.items():
        profiles[name] = run(cols, fam, status, task_gain, meas, pv["NEAR_R2"], R["DRIFT_AUC"], pv["FAMILY_VAR"], R["EXACT_RESIDUAL"], R["FAMILY_MIN_COLS"], R["FAMILY_B5_MIN_SIZE"])
    res = profiles[SBFS_START_PROFILE]
    sens = {}
    for k, vals in GRID.items():
        for v in vals:
            kw = {"near_r2": R["NEAR_R2"], "drift_auc": R["DRIFT_AUC"], "family_var": R["FAMILY_VAR"]}
            kw[{"NEAR_R2": "near_r2", "DRIFT_AUC": "drift_auc", "FAMILY_VAR": "family_var"}[k]] = v
            r = run(cols, fam, status, task_gain, meas, kw["near_r2"], kw["drift_auc"], kw["family_var"], R["EXACT_RESIDUAL"], R["FAMILY_MIN_COLS"], R["FAMILY_B5_MIN_SIZE"])
            sens[f"{k}={v}"] = {"tier_A": r["counts"]["tier_A"], "tier_B": r["counts"]["tier_B"]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "expert_guided_prefilter.v2", "rules": {k: v for k, v in R.items() if k != "periods"}, "periods": R["periods"], "level_exclude_tokens": LEVEL_EXCLUDE_TOKENS,
           "profiles": {n: {"thresholds": PROFILES[n], "counts": profiles[n]["counts"], "tier_A": profiles[n]["tier_a"], "tier_B": profiles[n]["tier_b"], "reasons": profiles[n]["reasons"],
                            "b1_clusters": profiles[n]["b1_clusters"], "b5_detail": profiles[n]["b5"],
                            "sha256_tier_A": hashlib.sha256(json.dumps(profiles[n]["tier_a"]).encode()).hexdigest(), "sha256_tier_B": hashlib.sha256(json.dumps(profiles[n]["tier_b"]).encode()).hexdigest()} for n in PROFILES},
           "sbfs_start_profile": SBFS_START_PROFILE, "sbfs_start_reason": "search cost grows about quadratically with |tier A|; tier B stays in the floating reinclusion pool, nothing is deleted",
           "recipe_identity": rec.get("identity"), "fit_block": ["2020-01-01", R["train_end"]], "rows_train": meas["rows_train"], "rows_common_support": meas["rows_common"], "support_hash": meas["support_hash"],
           "counts": res["counts"], "tier_A": res["tier_a"], "tier_B": res["tier_b"],
           "definitive": {c: res["reasons"][c] for c in cols if res["reasons"].get(c, "").startswith(("D", "ABSENT"))},
           "reasons": res["reasons"], "d1_record": res["d1"], "d2_clusters": res["d2_clusters"], "d2_pairs": res["d2_pairs"],
           "b1_clusters": res["b1_clusters"], "b1_pairs": res["b1_pairs"], "b2_drift": res["b2"], "b3_family_rank": res["b3_family_rank"], "b3_best_gain": res["b3_best_gain"],
           "b4_component_order_not_applied": comp_order, "b5_detail": res["b5"], "sensitivity": sens,
           "sha256_tier_A": hashlib.sha256(json.dumps(res["tier_a"]).encode()).hexdigest(), "sha256_tier_B": hashlib.sha256(json.dumps(res["tier_b"]).encode()).hexdigest()}
    (out / "CANDIDATES_STRICT.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"profiles": {n: profiles[n]["counts"] for n in PROFILES}, "sbfs_start_profile": SBFS_START_PROFILE, "sensitivity": sens}, indent=1))


if __name__ == "__main__":
    main()

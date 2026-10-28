#!/usr/bin/env python3
"""Fold scopes of the feature selection and the affected-unit map (Musashi's final joint audit A1, 2026-10-23).

  scopes    every unit of a selection run builds its purged contiguous folds over ITS OWN row set, so a fold's
            evaluation block moves by a few rows between units. One producer scope per (k_folds, fold): the excluded
            span is the UNION over all units of [first evaluation row - embargo, last evaluation row + embargo] (bar
            times), so a producer fitted once per fold never fits on any unit's evaluation or embargo rows. A consumer
            scope (fit window only, no exclusion) is added for the downstream consumers.
  affected  given the fold-local producer report (predictor tools/fold_local_producers.py build), a unit is AFFECTED
            when any column it reads (its columns or its row-defining columns, derived encodings included) changes
            in the scope of its fold; every other unit is provably unchanged and is reused from the ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def engine(a, ledger: Path):
    from tools import pool_selection_v2 as PS
    tmp = Path(tempfile.mkdtemp(prefix="selscopes_"))
    shutil.copy(ledger, tmp / "LEDGER.jsonl")
    ns = argparse.Namespace(store=a.store, catalog=a.catalog, admitted=a.admitted, csv_dir=a.csv_dir, fit_end=a.fit_end,
                            out=str(tmp), stage="all", shard="0/1")
    return PS, PS.Engine(ns)


def unit_rows(PS, e, task, mdl, cols, row_cols):
    rc = row_cols or cols
    sel = e.row_set(task, rc)
    b = e.bars[sel]
    if mdl == "temporal":
        rci = [e.cols.index(c) for c in rc]
        winok = np.isfinite(e.M[:, rci]).all(1)
        cs = np.concatenate([[0], np.cumsum(~winok)])
        keep = (b - PS.WIN_SEL + 1 >= 0) & (cs[b + 1] - cs[np.maximum(b - PS.WIN_SEL + 1, 0)] == 0)
        b = b[keep]
    return b


def derived_sources(e) -> dict:
    """derived encoding column -> the pool columns it is computed from."""
    src = {}
    for f, enc in e.encodings.items():
        for v, cols in enc["variants"].items():
            for c in cols:
                if c.startswith("enc.gmm."):
                    src[c] = [x for x in e.pool_cols if x.startswith("rg.gmm.p")]
                elif c.startswith("enc.hmm."):
                    src[c] = [x for x in e.pool_cols if x.startswith("rg.hmm.p")]
                elif c.startswith("enc.hod."):
                    src[c] = ["hod_sin", "hod_cos"]
                elif c.startswith("enc.ev."):
                    src[c] = [x for x in e.pool_cols if x.startswith("ev.")]
                elif c.startswith("enc.px.range_"):
                    src[c] = [x for x in e.pool_cols if x.split("@")[0] == c[len("enc."):].rsplit(".", 1)[0]]
                elif c == "enc.close.rel480":
                    src[c] = []                       # from the admitted CLOSE, not a fitted producer
    return src


def cmd_scopes(a):
    PS, e = engine(a, Path(a.ledger))
    units = e.arms()
    spans = {}
    ts = e.ts_ns
    for key, stage, task, mdl, cols, k, K, pg, meta in units:
        b = unit_rows(PS, e, task, mdl, cols, meta.get("row_cols"))
        n = len(b)
        edges = np.linspace(0, n, K + 1).astype(int)
        lo, hi = max(edges[k] - PS.EMBARGO, 0), min(edges[k + 1] + PS.EMBARGO, n) - 1
        s = spans.setdefault((K, k), [ts[b[lo]], ts[b[hi]], 0])
        s[0], s[1], s[2] = min(s[0], ts[b[lo]]), max(s[1], ts[b[hi]]), s[2] + 1
    import pandas as pd
    iso = lambda v: str(pd.Timestamp(int(v), unit="ns", tz="UTC"))  # noqa: E731
    scopes = [{"name": "consumer", "kind": "consumer", "fit_window": [a.consumer_fit[0], a.consumer_fit[1]], "exclude": [],
               "diag_window": ["2020-01-01", "2023-01-01"],
               "note": "consumers: fit 2020-01-01 .. 2023-07-01 by availability; the inner block 2023 S2, 2024 and TEST are never fitted on"}]
    for (K, k), (lo, hi, n) in sorted(spans.items()):
        scopes.append({"name": f"sel_K{K}_f{k}", "kind": "selection", "k_folds": K, "fold": k, "units": n,
                       "fit_window": ["2020-01-01", a.fit_end], "exclude": [[iso(lo), iso(hi)]], "diag_window": ["2020-01-01", "2023-01-01"]})
    doc = {"schema": "selection_fold_scopes.v1", "ledger": Path(a.ledger).name, "ledger_sha256": hashlib.sha256(Path(a.ledger).read_bytes()).hexdigest(),
           "units": len(units), "embargo_rows": PS.EMBARGO, "rule": __doc__.split("scopes")[1].split("affected")[0].strip(), "scopes": scopes}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({"units": len(units), "scopes": [(s["name"], s.get("exclude")) for s in scopes]}, indent=1))


def cmd_affected(a):
    PS, e = engine(a, Path(a.ledger))
    rep = json.loads(Path(a.report).read_text())
    changed = {name: set(v["changed"]) for name, v in rep["scopes"].items()}
    # a column whose finite mask moves in ANY scope moves the shared support of every unit that reads it (the engine
    # keeps one support: finite in the pool and in every scope), so it is changed for every fold
    support_moved = set()
    for name in rep["scopes"]:
        r = json.loads((Path(a.report).parent / f"SCOPE_{name}.json").read_text())
        support_moved |= {c for c, v in r["columns"].items() if v.get("finite_mask_identical") is False}
    for name in changed:
        changed[name] |= support_moved
    src = derived_sources(e)
    units = e.arms()
    out, counts = [], {}
    for key, stage, task, mdl, cols, k, K, pg, meta in units:
        reads = set(cols) | set(meta.get("row_cols") or [])
        expanded = set(reads)
        for c in reads:
            expanded |= set(src.get(c, []))
        hit = sorted(expanded & changed[f"sel_K{K}_f{k}"])
        aff = bool(hit)
        out.append({"key": key, "stage": stage, "task": task, "model": mdl, "k_folds": K, "fold": k, "affected": aff,
                    "changed_columns_read": hit[:12], "n_changed_read": len(hit)})
        c = counts.setdefault(stage, {"affected": 0, "unaffected": 0})
        c["affected" if aff else "unaffected"] += 1
    doc = {"schema": "selection_affected_units.v1", "report_sha256": hashlib.sha256(Path(a.report).read_bytes()).hexdigest(),
           "counts": counts, "support_moved_columns": sorted(support_moved), "units": out,
           "note": "stages are dependent: informative tasks come from the base stage and rescue pairs from the contrast "
                   "decisions, so a changed upstream decision re-plans the downstream stage (new keys are new units)"}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps(counts, indent=1))


def add_engine_args(p):
    p.add_argument("--store", action="append", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--admitted", required=True)
    p.add_argument("--csv-dir", required=True)
    p.add_argument("--fit-end", default="2024-01-01")
    p.add_argument("--ledger", required=True, help="the selection ledger whose units are scoped / classified")
    p.add_argument("--out", required=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scopes"); add_engine_args(s); s.add_argument("--consumer-fit", nargs=2, default=["2020-01-01", "2023-07-01"])
    f = sub.add_parser("affected"); add_engine_args(f); f.add_argument("--report", required=True)
    a = ap.parse_args()
    sys.exit({"scopes": cmd_scopes, "affected": cmd_affected}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()

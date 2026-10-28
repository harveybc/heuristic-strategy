#!/usr/bin/env python3
"""Per-column input justification for a consumer feature list (owner 2026-10-22: every preprocessing, every selected
input and every encoding must be justified by measured metrics, nothing hidden). Reads only artifacts and the store
(no fit, no model). For every column of ``--feature-list``:

  family / parent / component / real algorithm      from the generated FEATURE_CATALOG (the producer's row)
  preprocessing decision + its metric               NONE / DIFF / ASINH with ADF p, KPSS p, skew, kurtosis (TRAIN)
  selection evidence                                the v3 family decision (status, retained arm, evidence items)
  encoding                                          the encoding family's v3 decision and whether the consumer uses it
  scaling                                           fixed TRAIN z: mean/std on the declared fit period, max |z| in the
                                                    fit period and in each later period (outliers are KEPT, not clipped)

Writes INPUT_JUSTIFICATION_<name>.csv / .md and FEATURE_GROUPS_<name>.json (one small extractor per selection family).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.pool_selection_v2 import load_catalog  # noqa: E402

PERIODS = {"fit": ("2020-01-01", "2023-07-01"), "inner_2023H2": ("2023-07-01", "2024-01-01"),
           "validation_2024": ("2024-01-01", "2025-01-01")}        # consolidated contract (scaler fit = fit block)
ENC_FAMILY = (("hod_", "hod"), ("rg.gmm.", "regime_gmm"), ("rg.hmm.", "regime_hmm"), ("ev.", "event_clock"),
              ("px.range_", "range_scale"), ("CLOSE@diff", "price_level"))
STORE_PRIORITY = ("cross_asset_r4", "l2", "layer1_pool")


def enc_family(col):
    for pref, fam in ENC_FAMILY:
        if col.startswith(pref):
            return fam
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feature-list", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--selection-dir", required=True, help="selection v3 output (SHORTLIST.json, ENCODING.csv)")
    ap.add_argument("--store", required=True, help="full pool parquet the consumer reads")
    ap.add_argument("--encoding-adopted", default="no", choices=("no", "yes"))
    ap.add_argument("--encoding-evidence", default="consumer arm strict+encodings INNER_A -1.38 vs strict -1.13 (no gain)")
    ap.add_argument("--group-units", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lst = json.loads(Path(a.feature_list).read_text())
    name, cols = lst["name"], list(lst["features"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(a.catalog)
    sel = Path(a.selection_dir)
    dec = json.loads((sel / "SHORTLIST.json").read_text())["decisions"]
    enc = pd.read_csv(sel / "ENCODING.csv") if (sel / "ENCODING.csv").exists() else pd.DataFrame()
    enc_dec = enc.groupby("family").decision.first().to_dict() if len(enc) else {}
    fam_of = load_catalog(a.catalog, cols).set_index("column")["family"].to_dict()
    # store values for the scaling report
    df = pd.read_parquet(a.store, columns=["timestamp"] + cols)
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    if (ts >= pd.Timestamp("2025-01-01", tz="UTC")).any():
        raise SystemExit("TEST_ROWS_IN_STORE")
    msk = {k: (ts >= pd.Timestamp(s, tz="UTC")) & (ts < pd.Timestamp(e, tz="UTC")) for k, (s, e) in PERIODS.items()}
    rows = []
    for c in cols:
        base = c.split("@")[0]
        cr = cat[(cat.column == base) & (cat.generation_status == "GENERATED")]
        if len(cr):
            cr = cr.assign(_p=cr.store.map(lambda s_: STORE_PRIORITY.index(s_) if s_ in STORE_PRIORITY else 9)).sort_values("_p").iloc[0]
            met = json.loads(cr.preprocessing_metric_json) if isinstance(cr.preprocessing_metric_json, str) else {}
        else:
            cr, met = None, {}
        x = df[c].to_numpy(float)
        fit = x[msk["fit"]]
        mu, sd = np.nanmean(fit), np.nanstd(fit)
        sd = sd if sd > 0 else 1.0
        z = {k: float(np.nanmax(np.abs((x[m] - mu) / sd))) if np.isfinite(x[m]).any() else float("nan") for k, m in msk.items()}
        fam = fam_of.get(c, "")
        d = dec.get(fam, {})
        ev = d.get("evidence", [])
        ef = enc_family(c)
        rows.append({
            "column": c, "family": fam, "parent": None if cr is None else cr.parent, "component": None if cr is None else cr.component,
            "algorithm": None if cr is None else cr.algorithm, "source_store": None if cr is None else cr.store,
            "preprocessing": "DIFF" if c.endswith("@diff") else "ASINH" if c.endswith("@asinh") else (None if cr is None else cr.preprocessing_decision),
            "adf_p": met.get("adf_p"), "kpss_p": met.get("kpss_p"), "skew": met.get("skew"), "kurtosis": met.get("kurtosis"),
            "preprocessing_reason": met.get("reason") or met.get("note"),
            "selection_status": d.get("status"), "retained_arm": d.get("retained_arm"), "n_evidence": len(ev),
            "evidence": "; ".join(f"{e['kind']}:{e.get('model', '')}:{e.get('arm', '')}:{e.get('task')} +{e['mean']:.4f}" for e in ev[:3]),
            "encoding_family": ef, "encoding_v3_decision": enc_dec.get(ef, "") if ef else "",
            "encoding_used_by_consumer": ("base encoding" if ef and a.encoding_adopted == "no" else ef and "adopted variant") or "",
            "scaling": "fixed TRAIN z (fit 2020-01-01..2023-07-01), no clipping",
            "train_mean": float(mu), "train_std": float(sd), "max_abs_z_fit": z["fit"], "max_abs_z_inner_2023H2": z["inner_2023H2"],
            "max_abs_z_validation_2024": z["validation_2024"], "nan_share_fit": float(np.isnan(fit).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(out / f"INPUT_JUSTIFICATION_{name}.csv", index=False)
    groups = {}
    for c, f in zip(t.column, t.family):
        groups.setdefault(f, []).append(c)
    (out / f"FEATURE_GROUPS_{name}.json").write_text(json.dumps(
        {"name": name, "rule": "one small causal extractor per selection family (catalog-driven family keys)",
         "groups": groups, "units": {g: a.group_units for g in groups}}, indent=1))
    heavy = t[(t.max_abs_z_fit > 10) | (t.max_abs_z_validation_2024 > 10)]
    unsel = t[~t.selection_status.isin(["RETAINED", "RESCUED_INTERACTION"])]
    L = [f"# Input justification: `{name}` ({len(t)} columns, {len(groups)} extractor groups)", "",
         "Every column below reaches the consumer exactly as listed: store value -> (decided transform, already applied in the store) -> "
         "fixed TRAIN z-score with the constants of the fit block -> window. Nothing else: no rolling statistic, no clipping, no "
         "winsorizing, no NaN fill (a window with a non-finite value is excluded and counted), no agent feedback (the consumer runs "
         "with agent state `none`; `minimal` is a declared control).", "",
         f"- Preprocessing decisions: {t.preprocessing.value_counts(dropna=False).to_dict()} (rule: ADF p > 0.05 and KPSS p < 0.05 on TRAIN -> DIFF; positive and strongly skewed -> ASINH only if it reduces the measured skew; otherwise NONE).",
         f"- Selection: {t.selection_status.value_counts(dropna=False).to_dict()}" + ("" if unsel.empty else f" — columns without a retained family: {unsel.column.tolist()} (core15 control inputs, kept as the inherited reference)"),
         f"- Encoding: v3 encoding contrasts gave <= 0.02 nats, inconsistent across models/tasks; consumer evidence: {a.encoding_evidence}. "
         f"Adopted: {a.encoding_adopted} -> every column keeps its base encoding (sin/cos clock, regime probabilities, linear event hours, raw ranges, differenced level).",
         f"- Heavy tails KEPT (not clipped): {len(heavy)} columns reach |z| > 10 in the fit block or in 2024; listed in the table (max |z|). "
         "This is measured information, not an error; any transform of them requires its own matched contrast.", "",
         "| column | family | preprocessing (metric) | selection | encoding | TRAIN mean / std | max abs z fit / 2023H2 / 2024 |", "|---|---|---|---|---|---|---|"]
    f_ = lambda v: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.3g}" if isinstance(v, float) else str(v))  # noqa: E731
    for r in t.itertuples():
        pre = f"{r.preprocessing} (ADF p {f_(r.adf_p)}, KPSS p {f_(r.kpss_p)}, skew {f_(r.skew)}, kurt {f_(r.kurtosis)})"
        selc = f"{r.selection_status} {r.retained_arm or ''} [{r.n_evidence}]"
        L.append(f"| {r.column} | {r.family} | {pre} | {selc} | {r.encoding_used_by_consumer or '-'} | {f_(r.train_mean)} / {f_(r.train_std)} | "
                 f"{f_(r.max_abs_z_fit)} / {f_(r.max_abs_z_inner_2023H2)} / {f_(r.max_abs_z_validation_2024)} |")
    (out / f"INPUT_JUSTIFICATION_{name}.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"list": name, "columns": len(t), "groups": len(groups), "heavy_tail_columns": len(heavy),
                      "preprocessing": t.preprocessing.value_counts(dropna=False).to_dict(),
                      "selection": t.selection_status.value_counts(dropna=False).to_dict()}))


if __name__ == "__main__":
    main()

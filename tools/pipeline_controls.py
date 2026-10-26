#!/usr/bin/env python3
"""PIPELINE CONTROLS (DIAGNOSTIC ONLY, 2026-10-26): positive and negative controls run through the SAME code path as the
real clean-round experiments (recipe-bound E3-M, contract v1.3 and C2 v1.3, walk-forward fold wf2022), so that a failure
to learn / exploit a signal KNOWN to exist is located inside the pipeline instead of being read as "the inputs carry no
signal".

Every artifact written here is DIAGNOSTIC: the derived stores, manifests and recipes carry ``DIAGNOSTIC`` in their
identity (``recipe_DIAGNOSTIC_<KIND>_<16 hex>``), ``scientific_release: false`` and a ``diagnostic`` block; they live in
their own directory tree and are never mixed with real results. The originals (store, manifest, recipe, admitted bars)
are only read, never modified. No TEST row (>= 2025-01-01) is read: the store ends 2024-12-31 and the admitted reader
filters at read time.

Controls (``build`` kinds):
  cheat        one input column (default ``act.p_hit``, the retired in-sample producer) replaced by a LEAKED, perfectly
               informative feature: the realized first touch of close[b] +/- 1 causal sigma (sigma of the fill bar, the
               engine's own scale) from bar b + 1 up to the session's forced flat: +1 up first (long 1s/1s TP before SL),
               -1 down first (short TP before SL), 0 neither / both inside one bar. ``--noise s`` adds N(0, s^2) to it.
  sign12       the same column replaced by the sign of the next 12 h log return close[b+12] / close[b] (alternative leak).
  noise_inputs every recipe column replaced by i.i.d. N(0, 1) (NaN pattern kept, so the eligible rows and the C2 outcome
               table stamp are those of the real store): negative control (b).
  synthetic    a synthetic H1 market on the REAL admitted timestamps (calendar, closures, forced flats unchanged):
               GARCH(1,1) hourly volatility with the real hour-of-week profile, 12 sub-steps per bar for HIGH/LOW, and a
               PLANTED causal pattern: an observable AR(1) factor f_b (half-life ``--half-life`` hours, N(0,1) stationary)
               written into ``act.p_hit`` drives the drift of every following bar, E[r_{b+1} | f_b] = kappa * sigma_h * f_b.
               The price-difference columns (OPEN/HIGH/LOW/CLOSE@diff, RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN,
               px.logret_k) are recomputed on the synthetic prices; every other column keeps its real values (realistic
               distractors, uninformative about the synthetic path).
Negative control (a), labels shuffled within days, is a RUN option (``run-e3m --shuffle-labels-within-day``): every label
set the pipeline builds (v1.3 ``build_sets``, C2 ``sets``) has its labels / regression targets / weights permuted among the
rows of the same UTC decision day, module by module; inputs and evaluation are untouched.

``run-e3m`` / ``run-tool`` call the pipeline's own entry points in-process (``tools/expert_guided_e3m.py``,
``tools/expert_guided_trading_report.py``) with two declared, optional diagnostic hooks: ``--pin-admitted`` replaces the
admitted-bar digest pin by the synthetic bars' digest (every verification still runs, against that identity), and the
label shuffle above. Nothing else in the code path changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon import zigzag as Z  # noqa: E402
from tools import expert_guided_wf_recipe as WR  # noqa: E402

DIAG = "DIAGNOSTIC"
STORE_NAME = "full_pool_eurusd_h1.parquet"
CHEAT_COLUMN = "act.p_hit"
SYN_COLUMN = "act.p_hit"


# ----------------------------------------------------------------------------- helpers
def sha_json(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def recipe_columns(rec: dict) -> list:
    return [c["store_column"] for c in rec["tensor"]["columns"]]


def read_store(store: Path, cols: list) -> pd.DataFrame:
    """The base store restricted to timestamp, OHLC and the recipe's columns, rows < TEST (read-time filter)."""
    want = ["timestamp", "OPEN", "HIGH", "LOW", "CLOSE"] + [c for c in cols if c not in ("OPEN", "HIGH", "LOW", "CLOSE")]
    df = pd.read_parquet(store, columns=want, filters=[("timestamp", "<", pd.Timestamp(DG.TEST_CLOSED_FROM, tz="UTC"))])
    if pd.to_datetime(df["timestamp"], utc=True).max() >= pd.Timestamp(DG.TEST_CLOSED_FROM, tz="UTC"):
        raise SystemExit("TEST_ROWS_PRESENT")
    return df.reset_index(drop=True)


def base_dataset(a, rec):
    """The real dataset through the real recipe (bars, calendar, sigma), used to compute leaked / planted columns."""
    return DG.load_store(a.admitted, a.store, recipe_columns(rec), scaling="fixed_train", contract=rec["split"]["arm"], recipe=rec)


def forced_flat_bar(ds) -> np.ndarray:
    v = ds.bars.index.as_unit("ns").asi8
    seg = ds.cal.assign(ds.bars.index)
    ff = np.array([s.force_flat.value for s in ds.cal.segments], np.int64)
    n = len(v)
    return np.array([int(np.searchsorted(v + DG.H_NS, ff[int(seg[b])], side="right")) - 1 if seg[b] >= 0 else n - 1 for b in range(n)])


def first_touch(h, l, c, sigma, ffb, k_up=1.0, k_dn=1.0) -> np.ndarray:
    """Per bar b: +1 when close[b] + k_up * sigma[b+1] is touched before close[b] - k_dn * sigma[b+1] in bars b+1..ffb[b],
    -1 for the opposite, 0 when neither is touched or both inside one bar (ambiguous), 0 when sigma is undefined."""
    n = len(c)
    out = np.zeros(n)
    for b in range(n - 1):
        s = sigma[b + 1]
        if not (np.isfinite(s) and s > 0):
            continue
        up, dn = c[b] + k_up * s, c[b] - k_dn * s
        for k in range(b + 1, min(int(ffb[b]), n - 1) + 1):
            hu, hd = h[k] >= up, l[k] <= dn
            if hu and hd:
                break
            if hu:
                out[b] = 1.0
                break
            if hd:
                out[b] = -1.0
                break
    return out


def bar_values_to_store(df: pd.DataFrame, ds, per_bar: np.ndarray) -> np.ndarray:
    """Map a per-bar array onto the store rows (store timestamp = bar start)."""
    v = ds.bars.index.as_unit("ns").asi8
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    pos = np.searchsorted(v, ts)
    ok = (pos < len(v)) & (v[np.minimum(pos, len(v) - 1)] == ts)
    out = np.zeros(len(df))
    out[ok] = per_bar[pos[ok]]
    return out


def write_derived(out: Path, kind: str, df: pd.DataFrame, base_store: Path, base_rec: dict, base_rec_path: Path, note: dict,
                  admitted_sha: str | None = None) -> dict:
    """Derived store + MANIFEST + recipe (constants re-fitted on the fold's fit block with the loader's rules), all DIAGNOSTIC."""
    (out / "store").mkdir(parents=True, exist_ok=True)
    (out / "recipe").mkdir(parents=True, exist_ok=True)
    sp = out / "store" / STORE_NAME
    df.to_parquet(sp, index=False)
    man = json.loads((base_store.parent / "MANIFEST.json").read_text())
    cols = recipe_columns(base_rec)
    man_d = {"schema": man["schema"], "source_sha256": admitted_sha or man["source_sha256"], "rows": int(len(df)),
             "file": STORE_NAME, "file_sha256": DG.sha256(sp), "test_read": False, "features": [c for c in df.columns if c != "timestamp"],
             "columns": list(df.columns), "diagnostic": {"kind": kind, "status": DIAG, "never_a_result": True,
                                                        "derived_from_manifest_sha256": DG.sha256(base_store.parent / "MANIFEST.json"),
                                                        "derived_from_store_sha256": man.get("file_sha256"), **note}}
    (out / "store" / "MANIFEST.json").write_text(json.dumps(man_d, indent=1))
    rec = copy.deepcopy(base_rec)
    rec["store"] = dict(rec["store"], sha256=man_d["file_sha256"], manifest_sha256=DG.sha256(out / "store" / "MANIFEST.json"))
    fold = rec["split"]["arm"]
    scl = copy.deepcopy(base_rec["scaler"])
    scl.update(WR.fold_constants(sp, base_rec, fold))
    scl["derived"] = f"{DIAG}: constants re-fitted on the {fold} fit block of the derived store with the loader's rules"
    rec["scaler"] = scl
    rec["scientific_release"] = False
    rec["scope"] = f"{DIAG} pipeline control '{kind}' derived from {base_rec['identity']} (fold {fold}); never a result, never a release"
    rec["identity"] = f"recipe_{DIAG}_{kind.upper()}_" + sha_json({k: rec[k] for k in WR.IDENT_KEYS})[:16]
    rec["derived_from"] = {"identity": base_rec["identity"], "sha256": DG.sha256(base_rec_path), "deviation": ["store.diagnostic_columns", "scaler.refit_on_derived_store"],
                           "fold": fold}
    rec["diagnostic"] = {"kind": kind, "status": DIAG, **note}
    changed = [cols[j] for j in range(len(cols)) if not (np.isclose(scl["mean"][j], base_rec["scaler"]["mean"][j]) and np.isclose(scl["std"][j], base_rec["scaler"]["std"][j]))]
    rp = out / "recipe" / f"CONSUMER_RECIPE_{fold}_{DIAG}_{kind.upper()}.json"
    rp.write_text(json.dumps(rec, indent=1))
    doc = {"status": DIAG, "kind": kind, "store": str(sp), "store_sha256": man_d["file_sha256"], "recipe": str(rp), "recipe_sha256": DG.sha256(rp),
           "identity": rec["identity"], "columns_with_changed_constants": changed, "note": note}
    (out / "DIAGNOSTIC.json").write_text(json.dumps(doc, indent=1))
    return doc


# ----------------------------------------------------------------------------- builders
def build_cheat(a):
    rec = json.loads(Path(a.recipe).read_text())
    ds = base_dataset(a, rec)
    df = read_store(Path(a.store), recipe_columns(rec))
    h, l, c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    sig = np.asarray(ds.sigma, float)
    if a.kind == "sign12":
        per_bar = np.zeros(len(c))
        per_bar[:-12] = np.sign(np.log(c[12:] / c[:-12]))
        what = "sign of the next 12 h log return, close[b+12] / close[b]"
    else:
        per_bar = first_touch(h, l, c, sig, forced_flat_bar(ds), 1.0, 1.0)
        what = "first touch of close[b] +/- 1 sigma (sigma of the fill bar b+1) in bars b+1..forced flat: +1 up first, -1 down first, 0 neither/ambiguous"
    leak = bar_values_to_store(df, ds, per_bar)
    rng = np.random.default_rng(a.seed)
    val = leak + (rng.normal(0.0, a.noise, len(leak)) if a.noise > 0 else 0.0)
    nan = ~np.isfinite(df[a.column].to_numpy(float))
    val[nan] = np.nan                                     # NaN pattern kept: same eligible rows, same C2 outcome stamp
    df[a.column] = val
    kind = a.kind + (f"_noise{a.noise:g}" if a.noise > 0 else "")
    note = {"column_replaced": a.column, "leaked_feature": what, "noise_sd": a.noise, "noise_seed": a.seed,
            "leak_value_counts_rows": {str(k): int(v) for k, v in zip(*np.unique(leak[~nan], return_counts=True))},
            "column_name_kept": "the tensor name is unchanged so the recipe's groups / model resolve identically"}
    print(json.dumps(write_derived(Path(a.out), kind, df, Path(a.store), rec, Path(a.recipe), note)))


def build_noise_inputs(a):
    rec = json.loads(Path(a.recipe).read_text())
    cols = recipe_columns(rec)
    df = read_store(Path(a.store), cols)
    rng = np.random.default_rng(a.seed)
    for col in cols:
        x = df[col].to_numpy(float)
        z = rng.normal(0.0, 1.0, len(x))
        z[~np.isfinite(x)] = np.nan
        df[col] = z
    note = {"columns_replaced": "all recipe columns", "values": "i.i.d. N(0,1)", "seed": a.seed, "nan_pattern": "kept"}
    print(json.dumps(write_derived(Path(a.out), "noise_inputs", df, Path(a.store), rec, Path(a.recipe), note)))


def synth_prices(ts: pd.DatetimeIndex, real_close: np.ndarray, seed: int, kappa: float, half_life: float, substeps: int = 12):
    """GARCH(1,1) hourly log returns with the real hour-of-week volatility profile and a planted AR(1) drift factor.
    Returns (open, high, low, close, f) on the given bar starts."""
    rng = np.random.default_rng(seed)
    n = len(ts)
    r_real = np.r_[0.0, np.diff(np.log(real_close))]
    how = (ts.dayofweek * 24 + ts.hour).to_numpy()
    gap = np.r_[False, np.diff(ts.as_unit("ns").asi8) > 3 * DG.H_NS]
    prof = pd.Series(np.abs(r_real)).groupby(how).mean().reindex(range(168)).fillna(np.abs(r_real).mean()).to_numpy()
    prof = prof / prof[how].mean()
    sd_h = float(np.std(r_real[~gap]))                    # unconditional hourly vol of the real series (gaps excluded)
    rho = 0.5 ** (1.0 / half_life)
    alpha, beta = 0.05, 0.93
    garch_c = (1 - alpha - beta)
    f = np.zeros(n)
    eta = rng.normal(size=n)
    for i in range(1, n):
        f[i] = rho * f[i - 1] + np.sqrt(1 - rho ** 2) * eta[i]
    h2 = 1.0
    o, hi, lo, c = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    p = float(real_close[0])
    eps_prev = 0.0
    for i in range(n):
        h2 = garch_c + alpha * eps_prev ** 2 + beta * h2
        s = sd_h * np.sqrt(h2) * prof[how[i]]
        drift = kappa * s * (f[i - 1] if i > 0 else 0.0)  # E[r_i | f_{i-1}]: the factor at the decision bar drives the next bar
        if gap[i]:
            p *= np.exp(rng.normal(0.0, 3 * s))           # weekend / closure gap
        z = rng.normal(size=substeps)
        eps_prev = float(z.sum() / np.sqrt(substeps))
        path = p * np.exp(np.cumsum(drift / substeps + s * z / np.sqrt(substeps)))
        o[i], c[i] = p, path[-1]
        hi[i], lo[i] = max(p, path.max()), min(p, path.min())
        p = c[i]
    return o, hi, lo, c, f, {"sd_hourly": sd_h, "rho": rho, "garch": [garch_c, alpha, beta], "kappa": kappa, "half_life_h": half_life,
                             "substeps": substeps, "gap_rows": int(gap.sum())}


def build_synthetic(a):
    from app.short_horizon import clock_identity as CI
    rec = json.loads(Path(a.recipe).read_text())
    cols = recipe_columns(rec)
    adm = Path(a.admitted)
    raw = pd.read_parquet(adm / "eurusd.parquet", filters=[("timestamp", "<", pd.Timestamp(DG.TEST_CLOSED_FROM, tz="UTC"))])
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(pd.to_datetime(raw["timestamp"], utc=True))
    o, h, l, c, f, gen = synth_prices(ts, raw["BidClose"].to_numpy(float), a.seed, a.kappa, a.half_life)
    out = Path(a.out)
    (out / "admitted").mkdir(parents=True, exist_ok=True)
    syn = raw.copy()
    syn["BidOpen"], syn["BidHigh"], syn["BidLow"], syn["BidClose"] = o, h, l, c
    if "source_id" in syn:
        syn["source_id"] = f"{DIAG}_SYNTHETIC:seed{a.seed}"
    syn.to_parquet(out / "admitted" / "eurusd.parquet", index=False)
    (out / "admitted" / "CLOSURE_INTERVALS.json").write_bytes((adm / "CLOSURE_INTERVALS.json").read_bytes())
    adm_sha = DG.sha256(out / "admitted" / "eurusd.parquet")
    CI.read_admitted(out / "admitted" / "eurusd.parquet", "2020-01-01", DG.TEST_CLOSED_FROM)    # the pipeline's reader accepts it
    df = read_store(Path(a.store), cols)
    sts = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    v = ts.as_unit("ns").asi8
    pos = np.searchsorted(v, sts)
    if not ((pos < len(v)) & (v[np.minimum(pos, len(v) - 1)] == sts)).all():
        raise SystemExit("SYNTHETIC_STORE_JOIN")
    O, H, L, C = o[pos], h[pos], l[pos], c[pos]
    df["OPEN"], df["HIGH"], df["LOW"], df["CLOSE"] = O, H, L, C
    lc = np.log(C)
    recomputed = []

    def put(col, x):
        if col in df.columns:
            nan = ~np.isfinite(df[col].to_numpy(float))
            x = np.asarray(x, float).copy()
            x[nan] = np.nan
            df[col] = x
            recomputed.append(col)
    for k, arr in (("OPEN@diff", O), ("HIGH@diff", H), ("LOW@diff", L), ("CLOSE@diff", C)):
        put(k, np.r_[np.nan, np.diff(arr)])
    put("RAW_BAR_LOG_RETURN", np.log(C / O))
    put("RAW_CLOSE_LOG_RETURN", np.r_[np.nan, np.diff(lc)])
    for k in (1, 2, 4, 6, 12, 24, 48):
        put(f"px.logret_{k}", np.r_[np.full(k, np.nan), lc[k:] - lc[:-k]])
    nanp = ~np.isfinite(df[SYN_COLUMN].to_numpy(float))
    fv = f[pos].copy()
    fv[nanp] = np.nan
    df[SYN_COLUMN] = fv
    # theoretical drift of the next h bars in units of the 24 h causal sigma, for the record
    note = {"generator": gen, "seed": a.seed, "planted_column": SYN_COLUMN,
            "planted_rule": "E[log r_{b+1} | f_b] = kappa * sigma_h(b+1) * f_b; f AR(1), N(0,1) stationary, observable after bar b",
            "expected_drift_next_h_bars_per_unit_f": {str(hh): float(a.kappa * sum((0.5 ** (1 / a.half_life)) ** k for k in range(hh))) for hh in (6, 12, 24)},
            "expected_drift_units": "multiples of the hourly vol sigma_h", "recomputed_columns": recomputed,
            "other_columns": "real values kept: realistic distractors, uninformative about the synthetic path",
            "admitted_sha256": adm_sha, "timestamps": "the real admitted bar starts (< 2025-01-01), calendar and closures unchanged"}
    np.save(out / "admitted" / f"PLANTED_FACTOR_{DIAG}.npy", f)
    print(json.dumps(write_derived(out, "synthetic", df, Path(a.store), rec, Path(a.recipe), note, admitted_sha=adm_sha)))


KEEP_GROUPS_CLEAN = ("calendar", "ev", "px.hours_since_session_open")


def build_synthetic_clean(a):
    """Synthetic market with the distractors removed: from the synthetic store, every column that is neither recomputed
    on the synthetic prices, nor the planted factor, nor a clock column (groups ``KEEP_GROUPS_CLEAN``) is set to 0 (NaN
    pattern kept; the loader's std==0 rule scales it by 1). Separates 'the network cannot learn a moderate signal' from
    'the network is drowned by 100+ uninformative columns'."""
    rec = json.loads(Path(a.recipe).read_text())
    syn = json.loads((Path(a.store).parent / "MANIFEST.json").read_text())
    keep = set(syn["diagnostic"].get("recomputed_columns", [])) | {SYN_COLUMN}
    for g in KEEP_GROUPS_CLEAN:
        keep |= set(rec["representation"]["groups"]["groups"][g])
    cols = recipe_columns(rec)
    df = read_store(Path(a.store), cols)
    zeroed = []
    for col in cols:
        if col in keep or col in ("OPEN", "HIGH", "LOW", "CLOSE"):
            continue
        x = df[col].to_numpy(float)
        df[col] = np.where(np.isfinite(x), 0.0, np.nan)
        zeroed.append(col)
    note = {"from": str(a.store), "kept": sorted(keep), "zeroed_columns": len(zeroed), "planted_column": SYN_COLUMN,
            "generator": syn["diagnostic"].get("generator")}
    print(json.dumps(write_derived(Path(a.out), "synthetic_clean", df, Path(a.store), rec, Path(a.recipe), note, admitted_sha=syn["source_sha256"])))


# ----------------------------------------------------------------------------- run hooks
SHUFFLE_SCOPE = {"scope": "day"}


def _day_groups(ds, rows):
    day = (np.asarray(ds.row_ts, np.int64)[np.asarray(rows, np.int64)] // (24 * DG.H_NS))
    return np.zeros_like(day) if SHUFFLE_SCOPE["scope"] == "global" else day


def _permute_within(groups, rng):
    idx = np.arange(len(groups))
    out = idx.copy()
    for g in np.unique(groups):
        k = idx[groups == g]
        out[k] = rng.permutation(k)
    return out


def install_label_shuffle(seed: int, ledger: list, scope: str = "day"):
    """Negative control (a): every label set the pipeline builds is permuted within UTC decision days (labels, targets, masks,
    weights move together; inputs stay). ``scope="global"``: permuted across the whole set (one period per call), the
    corrected control when labels are autocorrelated over ~a day (a within-day permutation keeps most of the association)."""
    from app.short_horizon.expert_guided import c2 as C2
    import tools.expert_guided_e3m as E3
    rng = np.random.default_rng(seed)
    SHUFFLE_SCOPE["scope"] = scope
    orig_bs, orig_c2 = E3.build_sets, C2.sets

    def shuffle_pack(ds, D, rows):
        out = {}
        for m, tup in D.items():
            r = rows.get(m)
            if r is None or len(r) != len(tup[1]):
                raise SystemExit(f"SHUFFLE_ROWS_UNAVAILABLE:{m}")
            perm = _permute_within(_day_groups(ds, r), rng)
            out[m] = (tup[0],) + tuple(np.asarray(t)[perm] for t in tup[1:])
            ledger.append({"module": m, "rows": int(len(r)), "moved": int((perm != np.arange(len(perm))).sum())})
        return out

    def build_sets(variant, vis, cx, from_expert, *args, rows_out=None, **kw):
        ro = {}
        D = orig_bs(variant, vis, cx, from_expert, *args, rows_out=ro, **kw)
        if rows_out is not None:
            for m, v in ro.items():
                rows_out.setdefault(m, []).extend(v)
        return shuffle_pack(cx.ds, D, ro)

    def sets(cx, et, xt, period, nopl, n_state, *args, **kw):
        D, rows = orig_c2(cx, et, xt, period, nopl, n_state, *args, **kw)
        return shuffle_pack(cx.ds, D, rows), rows

    orig_exit = C2.OC.exit_sets

    def exit_sets(cx, xt, period, side_index):           # DAgger closer rows (C2 rounds > 0)
        r, y, R = orig_exit(cx, xt, period, side_index)
        perm = _permute_within(_day_groups(cx.ds, r), rng)
        ledger.append({"module": f"exit_sets/{side_index}", "rows": int(len(r)), "moved": int((perm != np.arange(len(perm))).sum())})
        return r, np.asarray(y)[perm], np.asarray(R)[perm]

    E3.build_sets = build_sets
    C2.sets = sets
    C2.OC.exit_sets = exit_sets


def run_e3m(a, rest):
    import tools.expert_guided_e3m as E3
    if a.pin_admitted:
        DG.PINS["admitted_sha256"] = a.pin_admitted
    led = []
    if a.shuffle_labels_within_day is not None:
        install_label_shuffle(a.shuffle_labels_within_day, led)
    elif a.shuffle_labels_global is not None:
        install_label_shuffle(a.shuffle_labels_global, led, scope="global")
    args = E3.build_parser().parse_args(rest)
    if DIAG not in str(args.out):
        raise SystemExit("CONTROL_OUT_MUST_BE_LABELLED_DIAGNOSTIC")
    E3.run(args)
    out = Path(args.out)
    p = out / "E3M.json"
    doc = json.loads(p.read_text())
    doc["diagnostic_control"] = {"status": DIAG, "pin_admitted": a.pin_admitted, "shuffle_labels_within_day": a.shuffle_labels_within_day, "shuffle_labels_global": a.shuffle_labels_global,
                                 "shuffle_ledger_samples": len(led), "shuffle_moved_share": (sum(x["moved"] for x in led) / max(1, sum(x["rows"] for x in led))) if led else None}
    p.write_text(json.dumps(doc, indent=1))


def run_tool(a, rest):
    if a.pin_admitted:
        DG.PINS["admitted_sha256"] = a.pin_admitted
    import runpy
    tool = rest[0]
    sys.argv = [tool] + rest[1:]
    runpy.run_path(str(ROOT / tool), run_name="__main__")


# ----------------------------------------------------------------------------- references and summaries
def _load_ds(recipe_path, store, admitted, pin=""):
    if pin:
        DG.PINS["admitted_sha256"] = pin
    rec = json.loads(Path(recipe_path).read_text())
    return DG.load_store(admitted, store, recipe_columns(rec), scaling="fixed_train", contract=rec["split"]["arm"], recipe=rec), rec


def column_by_row(ds, store, column) -> np.ndarray:
    """Raw (unscaled) store value of ``column`` per dataset row (row_ts = store timestamp + 1 h)."""
    df = pd.read_parquet(store, columns=["timestamp", column])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("ns").astype("int64").to_numpy() + DG.H_NS
    m = dict(zip(ts.tolist(), df[column].to_numpy(float).tolist()))
    return np.array([m.get(int(t), np.nan) for t in ds.row_ts], float)


def auc(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, int)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(x).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oracle_reference(a):
    """Theoretical ceiling through the SAME engine and replay contract: the policy that reads the planted / leaked column
    directly (never a network): open sign(x) when |x| >= q (q a quantile of |x| on the calibration block) at a grid cell;
    the cell and q are chosen on INNER_A by Sharpe among valid points, INNER_B / TRAIN reported. Also the feature-level
    AUC of the raw column against each head's target (C2 sets saved by the run when ``--c2-sets`` is given; the ZigZag
    hindsight class on eligible rows for v1.3)."""
    from tools.expert_guided_e1t import Ctx
    from tools.expert_guided_e3m import effective_contracts
    from app.short_horizon.expert_guided import env as E
    from app.short_horizon.expert_guided import oracle as OR
    ds, rec = _load_ds(a.recipe, a.store, a.admitted, a.pin_admitted)
    cx = Ctx(ds)
    SCFG, _ = effective_contracts("sac", 2.0)
    x = column_by_row(ds, a.store, a.column)
    cells = [(1.0, 1.0), (1.5, 1.5), (1.0, 1.5), (1.5, 1.0), (2.0, 1.5), (2.0, 2.0), (0.75, 0.75)]
    rows_a = np.flatnonzero(cx.obs_ok & (ds.period == "INNER_A"))
    absx = np.abs(x[rows_a][np.isfinite(x[rows_a])])
    qs = sorted({float(np.quantile(absx, 1 - r)) for r in (0.05, 0.1, 0.2, 0.33, 0.5)} | ({0.5} if a.discrete else set()))
    from tools.expert_guided_e1t import summary

    def pol(q, tp, sl):
        def f(bar, r, st, env):
            if int(st[0]) != 0:
                return E.HOLD, tp, sl
            v = x[r]
            if not np.isfinite(v) or abs(v) < q or v == 0:
                return E.HOLD, tp, sl
            return (E.LONG if v > 0 else E.SHORT), tp, sl
        return f
    grid = []
    for q in qs:
        for tp, sl in cells:
            m = summary(cx.rollout("INNER_A", pol(q, tp, sl), SCFG)[0])
            grid.append({"q": q, "tp": tp, "sl": sl, "INNER_A": {k: m.get(k) for k in ("sharpe_daily_sqrt365", "net_profit_usd", "orders_per_week", "win_rate", "profit_factor", "valid")}})
    ok = [g for g in grid if g["INNER_A"]["valid"] and np.isfinite(g["INNER_A"]["sharpe_daily_sqrt365"] or np.nan)]
    best = max(ok or grid, key=lambda g: g["INNER_A"]["sharpe_daily_sqrt365"] if np.isfinite(g["INNER_A"]["sharpe_daily_sqrt365"] or np.nan) else -1e9)
    chosen = {p: summary(cx.rollout(p, pol(best["q"], best["tp"], best["sl"]), SCFG)[0]) for p in ("TRAIN", "INNER_A", "INNER_B")}
    fixed = {f"tp{tp:g}_sl{sl:g}": {p: {k: v for k, v in summary(cx.rollout(p, pol(qs[0] if a.discrete else 0.0, tp, sl), SCFG)[0]).items()
                                         if k in ("sharpe_daily_sqrt365", "net_profit_usd", "orders_per_week", "win_rate", "profit_factor", "n_close_TP", "n_close_SL", "n_close_FORCED_FLAT", "n_entries_long", "n_entries_short")}
                                     for p in ("INNER_A", "INNER_B")} for tp, sl in ((1.0, 1.0), (1.5, 1.5))}
    feat = {"note": "AUC of the raw column (sign-oriented) against: round-0 labels = the expert opens only AT a hindsight pivot bar "
                    "(y_hind); DAgger labels = the LegOracle direction to the next pivot (d_star) and its open rule (remaining >= 1 sigma)"}
    lo = cx.oracle
    sig1 = np.asarray(ds.sigma, float)
    for p in ("INNER_A", "INNER_B"):
        rr = ds.rows(p)
        b = ds.row_bar[rr]
        yh = ds.y_hind[b]
        rem = lo.remaining[b] / sig1[np.minimum(b + 1, len(sig1) - 1)]
        feat[p] = {"rows": int(len(rr)), "round0_pivot_long_vs_rest": auc(x[rr], (yh == OR.LONG_CLASS).astype(int)),
                   "round0_pivot_short_vs_rest": auc(-x[rr], (yh == OR.SHORT_CLASS).astype(int)),
                   "dagger_dstar_up_vs_down": auc(x[rr], (lo.d_star[b] > 0).astype(int)),
                   "dagger_open_long_vs_rest": auc(x[rr], ((lo.d_star[b] > 0) & (rem >= 1.0)).astype(int)),
                   "dagger_open_short_vs_rest": auc(-x[rr], ((lo.d_star[b] < 0) & (rem >= 1.0)).astype(int)),
                   "pivot_bar_share": float(np.mean(yh != 0)), "dagger_open_share": float(np.mean(rem >= 1.0))}
    if a.c2_sets:
        z = np.load(a.c2_sets, allow_pickle=False)
        for p in ("INNER_A", "INNER_B"):
            for m, sgn in (("open_l", 1), ("open_s", -1), ("close_l", -1), ("close_s", 1)):
                kr, ky = f"{p}|{m}|rows", f"{p}|{m}|y"
                if kr in z.files and ky in z.files:
                    feat[p][f"c2_{m}"] = auc(sgn * x[z[kr]], z[ky])
    doc = {"status": DIAG, "column": a.column, "grid": grid, "chosen_on_INNER_A": best, "chosen_trading": chosen, "fixed_cells_all_signals": fixed,
           "feature_level_auc": feat, "policy": "flat: open sign(x) when |x| >= q at (tp, sl); never closes early; the engine and the sac replay contract"}
    Path(a.out).write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({"chosen": best, "INNER_B": {k: chosen["INNER_B"].get(k) for k in ("sharpe_daily_sqrt365", "net_profit_usd", "orders_per_week", "win_rate", "profit_factor")},
                      "feature_auc": feat}, default=float))


def summarize(a):
    """One compact DIAGNOSTIC record per control unit: head metrics (INNER_A calibration block, INNER_B frozen), the frozen
    strategy genes, trading per seed and period from the trading-detail report (per side, exits, PF, expectancy, Sharpe, DD)."""
    e = json.loads((Path(a.unit_dir) / "E3M.json").read_text())
    tr = json.loads(Path(a.trading).read_text()) if a.trading and Path(a.trading).exists() else None
    c2 = "c2" in e
    out = {"status": DIAG, "unit": Path(a.unit_dir).name, "contract": e["action_contract"]["identity"], "c2": c2,
           "recipe": e.get("recipe"), "diagnostic_control": e.get("diagnostic_control"), "seeds": {}}
    for seed, rec in e["trials"]["FULL"].items():
        ac = rec["action_contract"]
        heads = {}
        for m, c in ac["calibration"].items():
            ib = ac["reported_only_INNER_B"]["heads"].get(m, {})
            ibc = ib.get("calibrated") or {}
            heads[m] = {"INNER_A": {"auc": c["after"]["auc"], "brier": c["after"]["brier"], "n": c["after"]["n"], "pos": c["after"]["positives"]},
                        "INNER_B": {"auc": ibc.get("auc"), "brier": ibc.get("brier"), "n": ibc.get("n"), "pos": ibc.get("positives")}}
            if c2:
                rg = (ac.get("heads_calibration_block", {}).get(m, {}) or {}).get("regression")
                rgb = ib.get("regression")
                if rg:
                    heads[m]["INNER_A"]["reg_spearman"] = rg["mean_spearman"]
                if rgb:
                    heads[m]["INNER_B"]["reg_spearman"] = rgb["mean_spearman"]
        st = ac["strategy"]
        s = {"heads": heads, "strategy": st.get("strategy"), "chosen": st.get("chosen"), "nop_veto_kept": st.get("nop_veto_kept"),
             "gene_values": (st.get("chosen") or {}).get("genes"), "best_epochs": ac.get("best_epochs"),
             "fits": {m: {k: f.get(k) for k in ("epochs", "best_epoch", "stop_reason")} for m, f in rec["fits"].items()},
             "engine": {p: {k: rec["trading"][p].get(k) for k in ("net_profit_usd", "sharpe_daily_sqrt365", "max_drawdown_pct", "trades", "win_rate",
                                                                    "profit_factor", "orders_per_week", "valid", "n_entries_long", "n_entries_short",
                                                                    "n_close_TP", "n_close_SL", "n_close_FORCED_FLAT", "n_close_AGENT_CLOSE", "n_close_REVERSAL")}
                        for p in rec["trading"]}}
        if tr:
            pol = tr["policies"].get(f"seed/{seed}") or tr["policies"].get(f"seed_{seed}") or next((v for k, v in tr["policies"].items() if str(seed) in k), None)
            if pol:
                s["detail"] = {p: {"trades_per_week": d["activity"].get("orders_per_week") if isinstance(d.get("activity"), dict) else None,
                                   "buy_share": d["direction"]["buy_share"], "sell_share": d["direction"]["sell_share"],
                                   "BUY": {k: d["direction"]["BUY"].get(k) for k in ("trades", "win_rate", "profit_factor", "expectancy_usd", "expectancy_sigma", "net_usd")},
                                   "SELL": {k: d["direction"]["SELL"].get(k) for k in ("trades", "win_rate", "profit_factor", "expectancy_usd", "expectancy_sigma", "net_usd")},
                                   "exits": {r: x["count"] for r, x in d["exits"].items()}, "equity": d["equity"],
                                   "costs": d["costs"], "correctness_fail": d["correctness_fail"]} for p, d in pol.items()}
        out["seeds"][seed] = s
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({sd: {"heads": {m: (round(h["INNER_A"]["auc"] or float("nan"), 3), round(h["INNER_B"]["auc"] or float("nan"), 3)) for m, h in v["heads"].items()},
                           "engine": {p: (round(x["net_profit_usd"], 0), round(x["sharpe_daily_sqrt365"], 2), round(x["orders_per_week"], 2), x["n_entries_long"], x["n_entries_short"])
                                      for p, x in v["engine"].items()}} for sd, v in out["seeds"].items()}, default=float))


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def probe_heads(a):
    """Stage locator data/window -> heads: rank correlation of every trained head output (logit margin, and for C2 the best
    predicted net per side) with the planted / leaked column, per period, on the observable rows (state zeros, as the
    contract). A head that ignores a column the label depends on shows ~0 here even when the column predicts the label."""
    from tools.expert_guided_e1t import Ctx
    from tools import expert_guided_seed_ensemble as SE
    from app.short_horizon.expert_guided import action_contract as ACT
    from app.short_horizon.expert_guided import env as E
    ds, rec = _load_ds(a.recipe, a.store, a.admitted, a.pin_admitted)
    cx = Ctx(ds)
    e = json.loads((Path(a.unit_dir) / "E3M.json").read_text())
    x = column_by_row(ds, a.store, a.column)
    nf, win, NS = ds.windows.shape[2], ds.windows.shape[1], len(E.STATE_NAMES)
    out = {"status": DIAG, "unit": Path(a.unit_dir).name, "column": a.column, "seeds": {}}
    for seed in e["trials"]["FULL"]:
        res = {}
        for m in ("open_l", "open_s", "nop", "close_l", "close_s"):
            pol = SE.load_module(Path(a.unit_dir), e, m, seed, nf, win)
            for p in ("TRAIN", "INNER_A", "INNER_B"):
                rows = np.flatnonzero(cx.obs_ok & (ds.period == p))
                X = np.c_[cx.Xw[rows], np.zeros((len(rows), NS), np.float32)].astype(np.float32)
                z, rg = ACT.module_outputs(pol, X)
                d = res.setdefault(m, {}).setdefault(p, {"rows": int(len(rows))})
                d["spearman_logit_vs_column"] = spearman(z, x[rows])
                if rg is not None and rg.shape[1] > 2:
                    d["spearman_best_net_vs_column"] = spearman(rg.max(1), x[rows])
                elif rg is not None and rg.shape[1] >= 1:
                    d["spearman_reg0_vs_column"] = spearman(rg[:, 0], x[rows])
        out["seeds"][seed] = res
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({sd: {m: {p: round(v.get("spearman_logit_vs_column", float("nan")), 3) for p, v in r.items()} for m, r in rs.items()} for sd, rs in out["seeds"].items()}))


def ablate_strategy(a):
    """Stage locator heads -> strategy: the run's own trained heads and Platt maps (rebuilt from the checkpoints), replayed
    with (1) the frozen strategy (must reproduce the run) and (2) the same strategy with every early close disabled (positions
    end only at TP / SL / forced flat; no reverse). Nothing is fitted; INNER_B is read only to report."""
    import dataclasses
    from tools.expert_guided_e1t import Ctx, summary
    from tools.expert_guided_e3m import effective_contracts
    from tools import expert_guided_seed_ensemble as SE
    from app.short_horizon.expert_guided import action_contract as ACT
    from app.short_horizon.expert_guided import c2 as C2
    from app.short_horizon.expert_guided import env as E
    ds, rec = _load_ds(a.recipe, a.store, a.admitted, a.pin_admitted)
    cx = Ctx(ds)
    e = json.loads((Path(a.unit_dir) / "E3M.json").read_text())
    SCFG, _ = effective_contracts(e.get("exec_contract", "sac"), e.get("min_orders_per_week"))
    nf, win, NS = ds.windows.shape[2], ds.windows.shape[1], len(E.STATE_NAMES)
    c2 = "c2" in e
    keys = ("net_profit_usd", "sharpe_daily_sqrt365", "max_drawdown_pct", "trades", "win_rate", "profit_factor", "orders_per_week",
            "n_entries_long", "n_entries_short", "n_close_TP", "n_close_SL", "n_close_FORCED_FLAT", "n_close_AGENT_CLOSE", "n_close_REVERSAL")
    out = {"status": DIAG, "unit": Path(a.unit_dir).name, "c2": c2, "seeds": {}}
    rows_all = np.flatnonzero(cx.obs_ok)
    for seed, tr in e["trials"]["FULL"].items():
        pols = {m: SE.load_module(Path(a.unit_dir), e, m, seed, nf, win) for m in ACT.SPECIALISTS}
        cal = tr["action_contract"]["calibration"]
        sdoc = tr["action_contract"]["strategy"]["strategy"]
        if c2:
            table = C2.seed_table(pols, cx, NS, cal)
            st = C2.strategy_from_dict(sdoc)
            variants = {"frozen": st, "no_early_close": dataclasses.replace(st, close_margin=1e9, close_p=2.0, allow_reverse=False)}
            mk = lambda s_: C2.PolicyC2(table, s_)  # noqa: E731
        else:
            cost = ACT.row_cost_sigma(SCFG.profile.round_trip_price(), ds.sigma, ds.row_bar, e["action_contract"]["contract"]["strategy"]["cost_sigma"])
            table = ACT.calibrate_table(ACT.raw_table(pols, cx.Xw, rows_all, NS, cost), cal)
            st = ACT.strategy_from_dict(sdoc)
            variants = {"frozen": st, "no_early_close": dataclasses.replace(st, close_threshold=2.0, allow_reverse=False)}
            mk = lambda s_: ACT.ContractPolicy(table, s_)  # noqa: E731
        res = {}
        for name, s_ in variants.items():
            res[name] = {p: {k: summary(cx.rollout(p, mk(s_)(cx), SCFG)[0]).get(k) for k in keys} for p in ("INNER_A", "INNER_B")}
        res["frozen_reproduces_run"] = {p: bool(np.isclose(res["frozen"][p]["net_profit_usd"], tr["trading"][p]["net_profit_usd"])) for p in ("INNER_A", "INNER_B")}
        out["seeds"][seed] = res
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({sd: {n: {p: (round(v["sharpe_daily_sqrt365"], 2), round(v["net_profit_usd"]), round(v["orders_per_week"], 2), v["n_close_AGENT_CLOSE"])
                               for p, v in r[n].items()} for n in ("frozen", "no_early_close")} | {"repro": r["frozen_reproduces_run"]}
                      for sd, r in out["seeds"].items()}, default=float))


def learner_check(a):
    """Stage locator labels -> learner, outside the network: the SAME C2 entry labels (TP before SL at the reference cell,
    from the run's stamped outcome table) fitted on TRAIN by simple learners and scored on INNER_A / INNER_B:
    (1) the planted / leaked column alone (rank AUC, no fit), (2) L2 logistic regression on the last window slot of all
    tensor columns (the scaled recipe tensor), (3) the same on the full flattened window, (4) gradient boosting on the
    last slot. If (2)-(4) recover what (1) shows while the network does not, the loss is inside the network fit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from tools.expert_guided_e1t import Ctx
    from app.short_horizon.expert_guided import outcomes as OC
    ds, rec = _load_ds(a.recipe, a.store, a.admitted, a.pin_admitted)
    cx = Ctx(ds)
    et = OC.load_table(a.outcomes)
    x = column_by_row(ds, a.store, a.column)
    last = ds.windows[:, -1, :].astype(np.float64)
    full = cx.Xw
    out = {"status": DIAG, "column": a.column, "sides": {}}
    for si, name, sgn in ((0, "open_l", 1), (1, "open_s", -1)):
        sets = {p: OC.entry_sets(cx, et, p, si)[:2] for p in ("TRAIN", "INNER_A", "INNER_B")}
        r_tr, y_tr = sets["TRAIN"]
        res = {"rows": {p: int(len(v[0])) for p, v in sets.items()}, "positive_rate": {p: float(np.mean(v[1])) for p, v in sets.items()}}
        res["column_alone_auc"] = {p: auc(sgn * x[r], y) for p, (r, y) in sets.items()}
        for tag, M, mk in (("logreg_last_slot", last, lambda: LogisticRegression(C=0.05, max_iter=2000)),
                           ("logreg_full_window", full, lambda: LogisticRegression(C=0.005, max_iter=2000)),
                           ("gbm_last_slot", last, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0))):
            mdl = mk().fit(M[r_tr], y_tr)
            res[tag] = {p: auc(mdl.predict_proba(M[r])[:, 1], y) for p, (r, y) in sets.items()}
        # the value heads (what the C2 open rule reads): realized net per cell, rank correlation (mean over the 12 cells)
        from sklearn.linear_model import Ridge
        nets = {p: OC.entry_sets(cx, et, p, si)[2] for p in ("TRAIN", "INNER_A", "INNER_B")}
        res["net_column_alone_spearman"] = {p: float(np.mean([spearman(sgn * x[sets[p][0]], nets[p][:, j]) for j in range(nets[p].shape[1])])) for p in sets}
        rg = Ridge(alpha=100.0).fit(last[r_tr], nets["TRAIN"])
        res["net_ridge_last_slot_spearman"] = {p: float(np.mean([spearman(rg.predict(last[sets[p][0]])[:, j], nets[p][:, j]) for j in range(nets[p].shape[1])])) for p in sets}
        out["sides"][name] = res
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: {t: ({p: round(v, 3) for p, v in d.items()} if isinstance(d, dict) else d) for t, d in r.items() if t not in ("rows", "positive_rate")}
                      for k, r in out["sides"].items()}))


def table(a):
    """Markdown tables (DIAGNOSTIC) from the SUMMARY_DIAGNOSTIC_<unit>.json records: heads, trading per period, genes."""
    def f(x, d=2):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
    H, T, G = [], [], []
    for u in a.units:
        sm = json.loads((Path(a.dir) / f"SUMMARY_DIAGNOSTIC_{u}.json").read_text())
        for seed, s in sm["seeds"].items():
            hd = s["heads"]
            H.append(f"| {u} | " + " | ".join(f"{f(hd[m]['INNER_A']['auc'])} / {f(hd[m]['INNER_B']['auc'])} ({f(hd[m]['INNER_A']['brier'], 3)} / {f(hd[m]['INNER_B']['brier'], 3)})"
                                               for m in ("open_l", "open_s", "close_l", "close_s", "nop")) + " |")
            for p in a.periods:
                e = s["engine"][p]
                dt = (s.get("detail") or {}).get(p) or {}
                eq = dt.get("equity", {})
                B, S = dt.get("BUY", {}), dt.get("SELL", {})
                ex = dt.get("exits", {})
                T.append(f"| {u} | {p} | {f(e['orders_per_week'])} | {f(dt.get('buy_share', float('nan')) * 100 if dt else None, 0)} / {f(dt.get('sell_share', float('nan')) * 100 if dt else None, 0)} | "
                         f"{f(B.get('win_rate', float('nan')) * 100 if B.get('trades') else None, 0)} / {f(S.get('win_rate', float('nan')) * 100 if S.get('trades') else None, 0)} | "
                         f"{f(e['profit_factor'])} | {f(eq.get('expectancy_sigma'), 3)} | {f(e['sharpe_daily_sqrt365'])} | {f(e['max_drawdown_pct'], 1)} | {f(e['net_profit_usd'], 0)} | "
                         f"{ex.get('TP', e['n_close_TP'])}/{ex.get('SL', e['n_close_SL'])}/{ex.get('TIME', e['n_close_FORCED_FLAT'])}/{ex.get('AGENT_CLOSE', e['n_close_AGENT_CLOSE'])}/{ex.get('REVERSAL', e['n_close_REVERSAL'])} |")
            G.append(f"| {u} | `{json.dumps(s.get('gene_values'))}` | `{json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in (s.get('strategy') or {}).items()})}` |")
    out = ["### Heads: calibrated AUC INNER_A / INNER_B (Brier INNER_A / INNER_B)", "",
           "| unit | open_l | open_s | close_l | close_s | nop |", "|---|---|---|---|---|---|"] + H + [
          "", "### Trading (engine, sac replay contract)", "",
          "| unit | period | trades/wk | % buy / sell | win % buy / sell | PF | expectancy (sigma) | Sharpe | max DD % | net USD | exits TP/SL/TIME/AGENT/REV |",
          "|---|---|---|---|---|---|---|---|---|---|---|"] + T + ["", "### Frozen strategy genes (chosen on INNER_A)", "", "| unit | genes | strategy |", "|---|---|---|"] + G
    Path(a.out).write_text("\n".join(out) + "\n")
    print(a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("kind", choices=["cheat", "sign12", "noise_inputs", "synthetic", "synthetic_clean"])
    b.add_argument("--recipe", required=True)
    b.add_argument("--store", required=True)
    b.add_argument("--admitted", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--column", default=CHEAT_COLUMN)
    b.add_argument("--noise", type=float, default=0.0)
    b.add_argument("--seed", type=int, default=20261026)
    b.add_argument("--kappa", type=float, default=0.10)
    b.add_argument("--half-life", type=float, default=12.0)
    for name in ("run-e3m", "run-tool"):
        r = sub.add_parser(name)
        r.add_argument("--pin-admitted", default="")
        if name == "run-e3m":
            r.add_argument("--shuffle-labels-within-day", type=int, default=None)
            r.add_argument("--shuffle-labels-global", type=int, default=None)
    o = sub.add_parser("oracle")
    o.add_argument("--recipe", required=True)
    o.add_argument("--store", required=True)
    o.add_argument("--admitted", required=True)
    o.add_argument("--column", default=CHEAT_COLUMN)
    o.add_argument("--discrete", action="store_true")
    o.add_argument("--c2-sets", default="")
    o.add_argument("--pin-admitted", default="")
    o.add_argument("--out", required=True)
    h = sub.add_parser("probe-heads")
    for k in ("--recipe", "--store", "--admitted", "--unit-dir", "--out"):
        h.add_argument(k, required=True)
    h.add_argument("--column", default=CHEAT_COLUMN)
    h.add_argument("--pin-admitted", default="")
    x = sub.add_parser("ablate-strategy")
    for k in ("--recipe", "--store", "--admitted", "--unit-dir", "--out"):
        x.add_argument(k, required=True)
    x.add_argument("--pin-admitted", default="")
    lc = sub.add_parser("learner-check")
    for k in ("--recipe", "--store", "--admitted", "--outcomes", "--out"):
        lc.add_argument(k, required=True)
    lc.add_argument("--column", default=CHEAT_COLUMN)
    lc.add_argument("--pin-admitted", default="")
    tb = sub.add_parser("table")
    tb.add_argument("--dir", required=True)
    tb.add_argument("--units", nargs="+", required=True)
    tb.add_argument("--periods", nargs="+", default=["INNER_A", "INNER_B"])
    tb.add_argument("--out", required=True)
    m = sub.add_parser("summarize")
    m.add_argument("--unit-dir", required=True)
    m.add_argument("--trading", default="")
    m.add_argument("--out", required=True)
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if a.cmd == "build":
        if DIAG not in a.out:
            raise SystemExit("CONTROL_OUT_MUST_BE_LABELLED_DIAGNOSTIC")
        {"cheat": build_cheat, "sign12": build_cheat, "noise_inputs": build_noise_inputs, "synthetic": build_synthetic, "synthetic_clean": build_synthetic_clean}[a.kind](a)
    elif a.cmd == "oracle":
        oracle_reference(a)
    elif a.cmd == "probe-heads":
        probe_heads(a)
    elif a.cmd == "ablate-strategy":
        ablate_strategy(a)
    elif a.cmd == "learner-check":
        learner_check(a)
    elif a.cmd == "table":
        table(a)
    elif a.cmd == "summarize":
        summarize(a)
    elif a.cmd == "run-e3m":
        run_e3m(a, rest)
    else:
        run_tool(a, rest)


if __name__ == "__main__":
    main()

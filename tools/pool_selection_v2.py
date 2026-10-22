#!/usr/bin/env python3
"""Pool selection v2: multivariate, parent/component-aware feature selection over the complete EURUSD H1 candidate
pool (auditor's brief 2026-10-21 §3, §6; owner 2026-10-22: selection WITH the model, no univariate veto, no loss of
useful input information).

Protocol (declared before any result is read; see ``plan``):
  rows      observable flat origins inside TRAIN (consolidated contract 2020-01-01 .. fit_end, default 2024-01-01)
            with the entry gate open; every label FINAL before fit_end; TEST never read.
  tasks     opener3 (anticipatory hindsight pivot NONE/LONG/SHORT), hit (+/-2 sigma barrier before 72 bars),
            up_given_hit (direction among hits), sl_first_long / sl_first_short (hypothetical order at next open,
            TP 2 sigma / SL 2 sigma: SL or negative forced flat before TP), net_long / net_short (net result in sigma,
            regression). One single target does not define utility: a feature is judged on every task.
  CV        purged, contiguous folds inside TRAIN (6 for the base ranking, 3 for the contrasts), embargo 96 bars on
            both sides (label support <= 72 bars + forced flat); the temporal model early-stops on the last 15 % of
            each fold's training rows (own embargo). Scaling for the temporal model is fitted on the fold's training
            rows only (HGB is invariant to monotone rescaling, so it reads the raw decided columns).
  models    M1 HistGradientBoosting (depth 4, min leaf 200, 300 iters, lr 0.05) on the decision-bar vector;
            M2 the lane's causal temporal trunk (per-feature causal conv, units 2, window 24 completed bars at
            stride 1) + linear head. Two complementary inductive biases; neither alone eliminates a family.
  ranking   grouped BLOCK permutation on the validation fold (contiguous 72-row blocks permuted jointly for every
            column — and every window slot — of the group; 3 seeds); reported per model and task.
  contrasts per family F, in two fixed contexts (ctx=full: every other column of the pool; ctx=core15: the 15
            inherited inputs): without F vs with F, retrained. For a decomposition family with a declared parent P:
            none / P / Cj (each component) / coherent subgroups / Cgroup / P+Cgroup, same rows (explicit
            intersection, timestamp hash published together with each family's native coverage), same folds,
            seeds and budget. Model size (parameters / columns) is recorded for every arm.
  rescue    bounded interaction rescue: pairs of families not retained individually, ctx=core15, retained when the
            pair beats both singletons and the context with lo90 > 0.
  decision  (declared) a family is RETAINED when, in any task and context, the retrained with-vs-without gain has
            a one-sided 90 % lower bound above zero, or its block-permutation importance does in any model; the
            retained columns of a decomposition family are those of the best arm whose gain over `none` has
            lo90 > 0 (so detail-only, approximation-only or parent+components are all admissible); everything
            else is NOT_RETAINED_WITH_THIS_EVIDENCE — never "useless". Tasks whose base CV loss is not below
            the prior with the full pool are reported as NOT_INFORMATIVE_WITH_THIS_POOL and do not rank alone.
Outputs (``--out``): BUDGET.json (units and estimate, written by ``plan`` before ``run``), LEDGER.jsonl (one line
per unit, resumable), RANKING.csv, CONTRASTS_PARENT_COMPONENTS.csv, FAMILY_DECISIONS.csv, SHORTLIST.json,
SUPPORT.json (row counts, timestamp hashes, native coverage per family), SELECTION_V2.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from tools.expert_guided_e1t import STUDENT_CFG, Ctx  # noqa: E402
from tools.expert_guided_e3m import collect  # noqa: E402
from tools.expert_guided_e4q import CANDS, candidate_results  # noqa: E402

CORE15 = list(DG.FEATURES)
EMBARGO, BLOCK = 96, 72
MIN_COVERAGE = 0.5
FOLDS_BASE, FOLDS_CONTRAST = 6, 3
WIN_SEL = 24
SEEDS = (20261022, 20261023, 20261024)
HGB = dict(max_depth=4, min_samples_leaf=200, max_iter=300, learning_rate=0.05)
TASKS = {"opener3": "cls", "hit": "cls", "up_given_hit": "cls", "sl_first_long": "cls", "sl_first_short": "cls",
         "net_long": "reg", "net_short": "reg"}
EST_SECONDS = {"hgb": 10.0, "temporal": 120.0}
# parent -> the pool columns that ARE the original variable (after its own documented preprocessing decision)
PARENT_COLUMNS = {"log_close": ["CLOSE@diff", "px.logret_1"], "logret_1": ["px.logret_1"],
                  "log_range": ["RAW_LOG_RANGE"], "hl_pips": ["cdl.range_pips@asinh"]}
# component subgroups recognised inside a decomposition family (by substring of the component name)
SUBGROUPS = {"details": ("D1", "D2", "D3", "D4", "detail"), "approximation": ("A4", "approx", "level", "trend"),
             "energy": ("energy",), "phase": ("phase", "cos", "sin", "freq"), "amplitude": ("amp",),
             "state": ("prob", "entropy", "state", "surprise"), "innovation": ("innov",), "slope": ("slope",)}


# ----------------------------------------------------------------------------- catalog / families
def family_of(col: str) -> str:
    base = col.split("@")[0]
    if base.startswith(("fx.", "dly.", "fac.", "kal.", "cot.", "btc.", "wav.", "hil.", "mt.", "reg.", "fd.")):
        parts = base.split(".")
        return ".".join(parts[:2]) if len(parts) > 2 else parts[0]
    if base.startswith("px."):
        return "px." + base.split(".")[1].rstrip("_0123456789") if "_" in base else "px"
    if base.startswith(("cal.", "hod_", "dow_")):
        return "calendar"
    if base.startswith(("cdl.", "ev.", "ssa.", "act.")):
        return base.split(".")[0]
    if any(k in base for k in ("hurst", "variogram", "box_")):
        return "fractal"
    # the inherited technical inventory, split by what each indicator measures (fallback when no catalog is given)
    if base in ("OPEN", "HIGH", "LOW", "CLOSE") or base.startswith("RAW_"):
        return "inv.raw"
    if base in ("RSI", "Stochastic_K", "Stochastic_D", "WilliamsR", "CCI", "ROC", "BB_position", "price_minus_ema"):
        return "inv.momentum"
    if base in ("MACD", "MACD_Histogram", "MACD_Signal", "ADX", "DI_plus", "DI_minus"):
        return "inv.trend"
    if base in ("ATR", "ATR_ratio", "BB_Width", "rolling_std_24"):
        return "inv.volatility"
    return "inventory"


def load_catalog(path: str | None, cols: list) -> pd.DataFrame:
    """One row per pool column: column, family, parent, component. From FEATURE_CATALOG.csv when given (another
    lane generates it), otherwise derived from the column names (parent = None: no parent/component contrasts)."""
    if path:
        cat = pd.read_csv(path)
        cat = cat[cat["column"].isin(cols)].copy()
        cat["parent"] = cat["parent"].where(cat["parent"].notna() & (cat["parent"] != ""), None)
        cat["component"] = cat["component"].where(cat["component"].notna(), "")
        missing = sorted(set(cols) - set(cat["column"]))
        rows = [{"column": c, "family": family_of(c), "parent": None, "component": ""} for c in missing]
        return pd.concat([cat[["column", "family", "parent", "component"]], pd.DataFrame(rows)], ignore_index=True)
    return pd.DataFrame([{"column": c, "family": family_of(c), "parent": None, "component": ""} for c in cols])


def decided_columns(store_dir: Path) -> list:
    man = json.loads((store_dir / "MANIFEST.json").read_text())
    dec = man.get("decisions", {})
    if not dec:
        return [c for c in man["columns"] if c != "timestamp"]
    return [f"{c}@{d.lower()}" if d != "NONE" else c for c, d in dec.items()]


def load_pool(stores: list) -> tuple[pd.DataFrame, dict]:
    frames, pins = [], {}
    for s in stores:
        p = Path(s)
        d = p.parent if p.is_file() else p
        f = p if p.is_file() else next(d.glob("*.parquet"))
        cols = [c for c in decided_columns(d) if c not in ("OPEN", "HIGH", "LOW", "CLOSE")]
        df = pd.read_parquet(f)
        cols = [c for c in cols if c in df.columns]
        ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
        frames.append(df[cols].set_index(ts))
        pins[str(f)] = DG.sha256(f)
    pool = pd.concat(frames, axis=1)
    pool = pool.loc[:, ~pool.columns.duplicated()]
    return pool, pins


# ----------------------------------------------------------------------------- rows and labels
def build_rows(admitted, csv_dir, fit_end):
    ds = DG.load(admitted, csv_dir)                     # labels, calendar and the admitted grid only
    cx = Ctx(ds)
    cx.o, cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    v = ds.bars.index.as_unit("ns").asi8
    seg = ds.cal.assign(ds.bars.index)
    ff = np.array([s.force_flat.value for s in ds.cal.segments], np.int64)
    cx.flat_bar = np.array([int(np.searchsorted(v + DG.H_NS, ff[int(seg[b])], side="right")) - 1 if seg[b] >= 0
                            else len(v) for b in range(len(v))])
    hold = lambda *x: (E.HOLD, 1.5, 2.0)  # noqa: E731
    vis = [vv for vv in collect(cx, ("2020-01-01", fit_end), hold, STUDENT_CFG)[1] if vv[2][-1] > 0.5]
    rows = np.array([vv[0] for vv in vis], int)
    bars = np.array([vv[1] for vv in vis], int)
    end_bar = int(np.searchsorted(v, DG._utc_ns(fit_end)))
    Y, fin = candidate_results(cx, rows, end_bar)
    jl, js = CANDS.index((1, 2.0, 2.0)), CANDS.index((-1, 2.0, 2.0))
    nan = np.full(len(rows), np.nan)
    lab = {}
    lab["hit"] = np.where(fin, ((Y[:, jl] > 0) | (Y[:, js] > 0)).astype(float), nan)
    lab["up_given_hit"] = np.where(fin & (lab["hit"] == 1), (Y[:, jl] > 0).astype(float), nan)
    lab["sl_first_long"] = np.where(fin, (Y[:, jl] < 0).astype(float), nan)
    lab["sl_first_short"] = np.where(fin, (Y[:, js] < 0).astype(float), nan)
    lab["net_long"] = np.where(fin, Y[:, jl], nan)
    lab["net_short"] = np.where(fin, Y[:, js], nan)
    op = ds.y_hind[bars].astype(float)
    sup = ds.support_end[bars]
    op[(sup < 0) | (sup >= end_bar)] = np.nan
    lab["opener3"] = op
    return ds, bars, lab


def purged_folds(n: int, k: int, embargo: int = EMBARGO):
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        va = np.arange(edges[i], edges[i + 1])
        tr = np.r_[np.arange(0, max(edges[i] - embargo, 0)), np.arange(min(edges[i + 1] + embargo, n), n)]
        yield i, tr, va


def block_perm(n: int, rng, block: int = BLOCK) -> np.ndarray:
    nb = int(np.ceil(n / block))
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(b * block, min((b + 1) * block, n)) for b in order])
    return idx


def ts_hash(ts_ns: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(ts_ns, dtype=np.int64).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------- models
def prior_loss(kind, y_tr, y_va):
    if kind == "reg":
        return float(np.mean((y_va - y_tr.mean()) ** 2))
    classes = np.unique(np.r_[y_tr, y_va])
    p = np.array([(y_tr == c).mean() for c in classes])
    p = np.clip(p, 1e-6, 1)
    return float(-np.mean(np.log(p[np.searchsorted(classes, y_va)])))


def loss_of(kind, y, pred, classes=None):
    if kind == "reg":
        return float(np.mean((y - pred) ** 2))
    from sklearn.metrics import log_loss
    return float(log_loss(y, pred, labels=classes))


class HGBModel:
    name = "hgb"

    def __init__(self, kind, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        cls = HistGradientBoostingRegressor if kind == "reg" else HistGradientBoostingClassifier
        self.kind, self.m = kind, cls(random_state=seed, **HGB)

    def fit(self, X, y, Xi=None, yi=None):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X) if self.kind == "reg" else self.m.predict_proba(X)

    def n_params(self):
        """Tree nodes over all iterations (the model's effective size)."""
        return int(sum(p.nodes.shape[0] for it in self.m._predictors for p in it))


class TemporalModel:
    """Causal temporal trunk (units 2) + linear head on windows of WIN_SEL completed bars at stride 1."""
    name = "temporal"

    def __init__(self, kind, seed, n_feat, n_out, win=WIN_SEL, epochs=40, patience=5):
        import torch
        from app.short_horizon.expert_guided.temporal import TemporalTrunk
        torch.manual_seed(seed)
        self.torch, self.kind, self.win = torch, kind, win
        self.trunk = TemporalTrunk(n_feat, units=2, dropout=0.1)
        self.head = torch.nn.Linear(self.trunk.out_dim, n_out)
        self.epochs, self.patience = epochs, patience

    def params(self):
        return list(self.trunk.parameters()) + list(self.head.parameters())

    def n_params(self):
        return int(sum(p.numel() for p in self.params()))

    def _forward(self, w):
        return self.head(self.trunk(w))

    def _loss(self, out, y):
        t = self.torch
        return t.nn.functional.mse_loss(out[:, 0], y) if self.kind == "reg" else t.nn.functional.cross_entropy(out, y.long())

    def fit(self, W, y, Wi, yi, batch=256):
        t = self.torch
        opt = t.optim.AdamW(self.params(), lr=1e-3, weight_decay=1e-4)
        W, y, Wi, yi = (t.tensor(a, dtype=t.float32) for a in (W, y, Wi, yi))
        best, best_state, bad = np.inf, None, 0
        n = len(W)
        g = t.Generator().manual_seed(0)
        for ep in range(self.epochs):
            self.trunk.train()
            perm = t.randperm(n, generator=g)
            for i in range(0, n, batch):
                j = perm[i:i + batch]
                opt.zero_grad()
                self._loss(self._forward(W[j]), y[j]).backward()
                opt.step()
            self.trunk.eval()
            with t.no_grad():
                vl = float(self._loss(self._forward(Wi), yi))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = [p.detach().clone() for p in self.params()]
            else:
                bad += 1
                if bad >= self.patience:
                    break
        with t.no_grad():
            for p, b in zip(self.params(), best_state):
                p.copy_(b)
        self.epochs_run = ep + 1
        return self

    def predict(self, W, batch=2048):
        t = self.torch
        self.trunk.eval()
        outs = []
        with t.no_grad():
            for i in range(0, len(W), batch):
                o = self._forward(t.tensor(W[i:i + batch], dtype=t.float32))
                outs.append(o[:, 0].numpy() if self.kind == "reg" else t.softmax(o, 1).numpy())
        return np.concatenate(outs)


def windows_for(Z: np.ndarray, bars: np.ndarray, win: int = WIN_SEL) -> np.ndarray:
    idx = bars[:, None] - np.arange(win - 1, -1, -1)[None, :]
    return Z[np.clip(idx, 0, len(Z) - 1)].astype(np.float32)


# ----------------------------------------------------------------------------- the engine
class Engine:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.pool, self.pins = load_pool(a.store)
        self.cols = list(self.pool.columns)
        self.cat = load_catalog(a.catalog, self.cols)
        self.fam = dict(zip(self.cat["column"], self.cat["family"]))
        self.ds, self.bars, self.labels = build_rows(a.admitted, a.csv_dir, a.fit_end)
        grid = self.ds.bars.index
        self.M = self.pool.reindex(grid).to_numpy(float)          # bar grid x columns (NaN where absent)
        self.ts_ns = grid.as_unit("ns").asi8
        # declared rule: a column finite on fewer than MIN_COVERAGE of the labelled rows cannot join the pool's
        # row intersection (it would empty it); it is reported as INSUFFICIENT_COVERAGE, not judged
        lab_any = np.zeros(len(self.bars), bool)
        for y in self.labels.values():
            lab_any |= np.isfinite(y)
        cov = np.isfinite(self.M[self.bars][lab_any]).mean(0)
        self.low_coverage = {c: float(cov[i]) for i, c in enumerate(self.cols) if cov[i] < MIN_COVERAGE}
        keep = [i for i, c in enumerate(self.cols) if c not in self.low_coverage]
        self.M, self.cols = self.M[:, keep], [self.cols[i] for i in keep]
        self.cat = self.cat[self.cat["column"].isin(self.cols)]
        self.families = {}
        for c in self.cols:
            self.families.setdefault(self.fam[c], []).append(c)
        self.decomp = {}                                           # family -> {"parent": [cols], "components": {name: col}}
        for f, g in self.cat.groupby("family"):
            par = [p for p in g["parent"].dropna().unique()]
            if par:
                pc = [c for p in par for c in PARENT_COLUMNS.get(p, [p]) if c in self.cols and self.fam.get(c) != f]
                comps = {(r.component or r.column): r.column for r in g.itertuples()}
                self.decomp[f] = {"parent": par, "parent_cols": pc, "components": comps}
        self.informative = None
        self.ledger = self.out / "LEDGER.jsonl"
        self.done = set()
        if self.ledger.exists():
            for line in self.ledger.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if "error" not in r:                       # failed units are retried on resume
                        self.done.add(r["key"])

    # ---- row sets
    def row_set(self, task, cols):
        """Rows of the task whose columns are all finite at the decision bar (and inside the window for the
        temporal model, which only needs the window columns finite: enforced in ``unit``)."""
        y = self.labels[task]
        ci = [self.cols.index(c) for c in cols]
        ok = np.isfinite(y) & np.isfinite(self.M[self.bars][:, ci]).all(1)
        return np.where(ok)[0]

    def native_coverage(self, task):
        y = self.labels[task]
        base = np.isfinite(y)
        out = {}
        for f, cs in self.families.items():
            ci = [self.cols.index(c) for c in cs]
            out[f] = float((np.isfinite(self.M[self.bars][:, ci]).all(1) & base).sum() / max(base.sum(), 1))
        return out

    # ---- one unit of work: fit on fold, score, optional block permutation of groups
    def unit(self, task, model_name, cols, fold, k_folds, seed, perm_groups=None):
        kind = TASKS[task]
        sel = self.row_set(task, cols)
        ci = [self.cols.index(c) for c in cols]
        y = self.labels[task][sel]
        b = self.bars[sel]
        X = self.M[b][:, ci]
        if model_name == "temporal":
            winok = np.isfinite(self.M[:, ci]).all(1)
            cs = np.concatenate([[0], np.cumsum(~winok)])
            keep = (b - WIN_SEL + 1 >= 0) & (cs[b + 1] - cs[np.maximum(b - WIN_SEL + 1, 0)] == 0)
            sel, y, b, X = sel[keep], y[keep], b[keep], X[keep]
        folds = list(purged_folds(len(y), k_folds))
        _, tr, va = folds[fold]
        classes = np.unique(y) if kind == "cls" else None
        t0 = time.time()
        if model_name == "hgb":
            m = HGBModel(kind, seed).fit(X[tr], y[tr])
            pred = m.predict(X[va])
            train_loss = loss_of(kind, y[tr], m.predict(X[tr]), classes)
            feat = lambda idx: X[idx]  # noqa: E731
            apply_perm = lambda A, cidx, p: (A.__setitem__((slice(None), cidx), A[p][:, cidx]) or A)  # noqa: E731
        else:
            n_in = len(tr)
            cut = int(n_in * 0.85)
            tr_fit, tr_in = tr[:max(cut - EMBARGO, 1)], tr[cut:]
            mu, sd = np.nanmean(self.M[b[tr_fit]][:, ci], 0), np.nanstd(self.M[b[tr_fit]][:, ci], 0)
            sd = np.where(sd > 0, sd, 1.0)
            Z = (self.M[:, ci] - mu) / sd
            W = windows_for(Z, b)
            n_out = 1 if kind == "reg" else len(classes)
            yy = y if kind == "reg" else np.searchsorted(classes, y)
            m = TemporalModel(kind, seed, len(cols), n_out).fit(W[tr_fit], yy[tr_fit], W[tr_in], yy[tr_in])
            pred = m.predict(W[va])
            train_loss = loss_of(kind, y[tr_fit], m.predict(W[tr_fit]), classes)
            feat = lambda idx: W[idx]  # noqa: E731
            apply_perm = lambda A, cidx, p: (A.__setitem__((slice(None), slice(None), cidx), A[p][:, :, cidx]) or A)  # noqa: E731
        va_loss = loss_of(kind, y[va], pred, classes)
        res = {"task": task, "kind": kind, "model": model_name, "fold": fold, "k_folds": k_folds, "seed": seed,
               "n_cols": len(cols), "rows_train": int(len(tr)), "rows_val": int(len(va)),
               "support_hash": ts_hash(self.ts_ns[b]), "val_hash": ts_hash(self.ts_ns[b[va]]),
               "train_loss": train_loss, "val_loss": va_loss, "prior_loss": prior_loss(kind, y[tr], y[va]),
               "n_params": m.n_params(), "seconds": round(time.time() - t0, 1)}
        if model_name == "temporal":
            res["epochs"] = m.epochs_run
        if perm_groups:
            imp = {}
            for g, gcols in perm_groups.items():
                cidx = [cols.index(c) for c in gcols if c in cols]
                if not cidx:
                    continue
                vals = []
                for s in SEEDS:
                    rng = np.random.default_rng(s + fold)
                    A = feat(va).copy()
                    A = apply_perm(A, cidx, block_perm(len(va), rng))
                    vals.append(loss_of(kind, y[va], m.predict(A), classes) - va_loss)
                imp[g] = float(np.mean(vals))
            res["perm"] = imp
        return res

    # ---- units
    def informative_tasks(self):
        """Declared rule: a task is informative when the HGB base CV loss (6 folds, full pool) is below the prior."""
        if self.informative is not None:
            return self.informative
        rows = [json.loads(l) for l in self.ledger.read_text().splitlines() if l.strip()] if self.ledger.exists() else []
        base = [r for r in rows if r.get("stage") == "base" and r["model"] == "hgb"]
        inf = []
        for t in TASKS:
            rr = [r for r in base if r["task"] == t]
            if len(rr) == FOLDS_BASE and np.mean([r["val_loss"] - r["prior_loss"] for r in rr]) < 0:
                inf.append(t)
        self.informative = inf
        return inf

    def arms(self):
        """Every unit key -> (stage, task, model, cols, fold, k_folds, perm_groups, meta). Deterministic order."""
        units = []
        full = self.cols
        groups = {f: cs for f, cs in self.families.items()}
        for t in TASKS:
            for mdl in ("hgb", "temporal"):
                for k in range(FOLDS_BASE):
                    units.append((f"base|{t}|{mdl}|{k}", "base", t, mdl, full, k, FOLDS_BASE, groups,
                                  {"family": "ALL", "ctx": "full", "arm": "all"}))
        inf = self.informative_tasks() if self.a.stage in ("contrast", "rescue", "all") else []
        core = [c for c in full if c.split("@")[0] in CORE15]
        for t in inf:
            for k in range(FOLDS_CONTRAST):
                units.append((f"ctx|{t}|core15|{k}", "contrast", t, "hgb", core, k, FOLDS_CONTRAST, None,
                              {"family": "NONE", "ctx": "core15", "arm": "none"}))
                units.append((f"ctx|{t}|full|{k}", "contrast", t, "hgb", full, k, FOLDS_CONTRAST, None,
                              {"family": "NONE", "ctx": "full", "arm": "all"}))
                for f, cs in self.families.items():
                    dec = self.decomp.get(f)
                    pc = dec["parent_cols"] if dec else []
                    fam_all = cs + pc
                    for ctx_name, ctx in (("full", [c for c in full if c not in fam_all]),
                                          ("core15", [c for c in core if c not in fam_all])):
                        arms = {"none": ctx, "with": ctx + cs}
                        if dec:
                            comps = dec["components"]
                            arms = {"none": ctx, "P": ctx + pc, "Cgroup": ctx + cs, "P+Cgroup": ctx + pc + cs}
                            for name, col in comps.items():
                                arms[f"C:{name}"] = ctx + [col]
                            for sg, keys in SUBGROUPS.items():
                                sub = [col for name, col in comps.items() if any(kk.lower() in name.lower() for kk in keys)]
                                if sub and len(sub) < len(comps):
                                    arms[f"S:{sg}"] = ctx + sub
                        for arm, cols in arms.items():
                            if arm == "none" and ctx_name == "core15" and not fam_all:
                                continue
                            units.append((f"con|{t}|{f}|{ctx_name}|{arm}|{k}", "contrast", t, "hgb", cols, k,
                                          FOLDS_CONTRAST, None, {"family": f, "ctx": ctx_name, "arm": arm}))
        if self.a.stage in ("rescue", "all"):
            for t in inf:
                for f1, f2 in self.rescue_pairs():
                    for k in range(FOLDS_CONTRAST):
                        cols = core + [c for c in self.families[f1] + self.families[f2] if c not in core]
                        units.append((f"res|{t}|{f1}+{f2}|{k}", "rescue", t, "hgb", cols, k, FOLDS_CONTRAST, None,
                                      {"family": f"{f1}+{f2}", "ctx": "core15", "arm": "pair"}))
        return units

    def rescue_pairs(self, budget=20):
        """Pairs among families NOT retained individually (declared budget 20), ordered by the sum of their
        core15-context mean gains (largest first, even when not significant)."""
        dec = self.decisions(quiet=True)
        cand = [f for f, d in dec.items() if d["status"] == "NOT_RETAINED_WITH_THIS_EVIDENCE"]
        score = {f: dec[f].get("best_gain_core15", 0.0) for f in cand}
        pairs = sorted(combinations(cand, 2), key=lambda p: -(score[p[0]] + score[p[1]]))
        return pairs[:budget]

    # ---- run
    def run(self):
        units = self.arms()
        shard, nshard = (int(x) for x in self.a.shard.split("/"))
        todo = [u for i, u in enumerate(units) if i % nshard == shard and u[0] not in self.done]
        print(json.dumps({"units_total": len(units), "shard": self.a.shard, "todo": len(todo)}), flush=True)
        for key, stage, task, mdl, cols, k, kf, pg, meta in todo:
            try:
                res = self.unit(task, mdl, cols, k, kf, SEEDS[0], pg)
            except Exception as ex:  # a failed unit is recorded, never silently skipped
                import traceback
                res = {"task": task, "model": mdl, "fold": k, "error": f"{type(ex).__name__}: {ex}",
                       "traceback": traceback.format_exc().splitlines()[-4:]}
            res.update({"key": key, "stage": stage, **meta, "cols_hash": hashlib.sha256("|".join(cols).encode()).hexdigest()[:12]})
            with self.ledger.open("a") as fh:
                fh.write(json.dumps(res) + "\n")
            self.done.add(key)
            print(json.dumps({k_: res.get(k_) for k_ in ("key", "val_loss", "prior_loss", "seconds", "error")}), flush=True)

    def plan(self):
        self.a.stage = "all"
        self.informative = list(TASKS)   # the plan counts every task; the informative rule is applied at run time
        units = self.arms()
        est = sum(EST_SECONDS[u[3]] for u in units)
        counts = {}
        for u in units:
            counts[u[1]] = counts.get(u[1], 0) + 1
        doc = {"declared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "fit_end": self.a.fit_end,
               "rows_by_task": {t: int(np.isfinite(y).sum()) for t, y in self.labels.items()},
               "columns": len(self.cols), "families": {f: len(c) for f, c in self.families.items()},
               "excluded_insufficient_coverage": self.low_coverage, "min_coverage": MIN_COVERAGE,
               "decomposition_families": {f: {"parent": d["parent"], "parent_cols": d["parent_cols"],
                                              "components": list(d["components"])} for f, d in self.decomp.items()},
               "units": counts, "units_upper_bound": len(units), "estimated_cpu_hours_upper_bound": round(est / 3600, 1),
               "folds": {"base": FOLDS_BASE, "contrast": FOLDS_CONTRAST, "embargo_bars": EMBARGO, "block_bars": BLOCK},
               "models": {"hgb": HGB, "temporal": {"units": 2, "window": WIN_SEL, "stride": 1, "optim": "AdamW 1e-3 wd 1e-4",
                                                   "dropout": 0.1, "early_stop": "last 15% of fold train rows, patience 5"}},
               "seeds": SEEDS, "rescue_budget_pairs": 20, "decision_rule": __doc__.split("decision")[1].split("Outputs")[0].strip(),
               "stores": self.pins, "catalog": self.a.catalog}
        (self.out / "BUDGET.json").write_text(json.dumps(doc, indent=1))
        (self.out / "SUPPORT.json").write_text(json.dumps({t: {"rows": int(np.isfinite(y).sum()),
                                                               "native_coverage": self.native_coverage(t)}
                                                           for t, y in self.labels.items()}, indent=1))
        print(json.dumps({k: doc[k] for k in ("rows_by_task", "columns", "units", "estimated_cpu_hours_upper_bound")}))

    # ---- aggregation
    def rows(self):
        return [json.loads(l) for l in self.ledger.read_text().splitlines() if l.strip()] if self.ledger.exists() else []

    @staticmethod
    def lo90(vals):
        v = np.array(vals, float)
        return float(v.mean() - 1.645 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("-inf")

    def decisions(self, quiet=False):
        rows = [r for r in self.rows() if "error" not in r]
        base = [r for r in rows if r["stage"] == "base"]
        con = [r for r in rows if r["stage"] == "contrast"]
        out = {}
        # permutation ranking per family (any model, any task)
        perm = {}
        for r in base:
            for g, v in r.get("perm", {}).items():
                perm.setdefault((g, r["model"], r["task"]), []).append(v)
        for f in self.families:
            d = {"family": f, "status": "NOT_RETAINED_WITH_THIS_EVIDENCE", "evidence": [], "retained_cols": [],
                 "best_gain_core15": 0.0}
            for (g, mdl, t), vals in perm.items():
                if g == f and len(vals) == FOLDS_BASE and self.lo90(vals) > 0 and t in (self.informative or TASKS):
                    d["evidence"].append({"kind": "block_permutation", "model": mdl, "task": t, "mean": float(np.mean(vals)),
                                          "lo90": self.lo90(vals)})
            for ctx in ("full", "core15"):
                for t in TASKS:
                    none = {r["fold"]: r["val_loss"] for r in con if r["family"] == f and r["ctx"] == ctx and r["arm"] == "none" and r["task"] == t}
                    if not none:
                        ref = {r["fold"]: r["val_loss"] for r in con if r["family"] == "NONE" and r["ctx"] == ctx and r["task"] == t}
                        none = ref
                    arms = {}
                    for r in con:
                        if r["family"] == f and r["ctx"] == ctx and r["task"] == t and r["arm"] != "none":
                            arms.setdefault(r["arm"], {})[r["fold"]] = r
                    for arm, by_fold in arms.items():
                        folds = sorted(set(by_fold) & set(none))
                        if len(folds) < FOLDS_CONTRAST:
                            continue
                        gains = [none[k] - by_fold[k]["val_loss"] for k in folds]
                        rec = {"kind": "retrained_contrast", "ctx": ctx, "task": t, "arm": arm, "mean": float(np.mean(gains)),
                               "lo90": self.lo90(gains), "n_cols": by_fold[folds[0]]["n_cols"]}
                        if ctx == "core15":
                            d["best_gain_core15"] = max(d["best_gain_core15"], rec["mean"])
                        if rec["lo90"] > 0:
                            d["evidence"].append(rec)
            if d["evidence"]:
                d["status"] = "RETAINED"
                cons = [e for e in d["evidence"] if e["kind"] == "retrained_contrast"]
                if f in self.decomp and cons:
                    best = max(cons, key=lambda e: e["mean"])
                    d["retained_arm"] = best["arm"]
                    d["retained_cols"] = self.arm_columns(f, best["arm"])
                else:
                    d["retained_cols"] = list(self.families[f])
            out[f] = d
        res = [r for r in rows if r["stage"] == "rescue"]
        for r in res:
            pass
        pairs = {}
        for r in res:
            pairs.setdefault((r["family"], r["task"]), {})[r["fold"]] = r["val_loss"]
        for (fp, t), by_fold in pairs.items():
            f1, f2 = fp.split("+")
            ref = {r["fold"]: r["val_loss"] for r in con if r["family"] == "NONE" and r["ctx"] == "core15" and r["task"] == t}
            s1 = {r["fold"]: r["val_loss"] for r in con if r["family"] == f1 and r["ctx"] == "core15" and r["task"] == t and r["arm"] in ("with", "Cgroup", "P+Cgroup")}
            s2 = {r["fold"]: r["val_loss"] for r in con if r["family"] == f2 and r["ctx"] == "core15" and r["task"] == t and r["arm"] in ("with", "Cgroup", "P+Cgroup")}
            folds = sorted(set(by_fold) & set(ref) & set(s1) & set(s2))
            if len(folds) < FOLDS_CONTRAST:
                continue
            g_pair = [ref[k] - by_fold[k] for k in folds]
            g1, g2 = [ref[k] - s1[k] for k in folds], [ref[k] - s2[k] for k in folds]
            if self.lo90(g_pair) > 0 and np.mean(g_pair) > max(np.mean(g1), np.mean(g2)):
                for f in (f1, f2):
                    if out[f]["status"] != "RETAINED":
                        out[f].update({"status": "RESCUED_INTERACTION", "retained_cols": list(self.families[f])})
                    out[f]["evidence"].append({"kind": "interaction", "pair": fp, "task": t, "mean": float(np.mean(g_pair)),
                                               "lo90": self.lo90(g_pair), "singletons": [float(np.mean(g1)), float(np.mean(g2))]})
        if not quiet:
            print(json.dumps({f: d["status"] for f, d in out.items()}))
        return out

    def arm_columns(self, f, arm):
        dec, cs = self.decomp[f], self.families[f]
        if arm == "P":
            return list(dec["parent_cols"])
        if arm == "Cgroup" or arm == "with":
            return list(cs)
        if arm == "P+Cgroup":
            return list(dec["parent_cols"]) + list(cs)
        if arm.startswith("C:"):
            return [dec["components"][arm[2:]]]
        if arm.startswith("S:"):
            keys = SUBGROUPS[arm[2:]]
            return [col for name, col in dec["components"].items() if any(k.lower() in name.lower() for k in keys)]
        return list(cs)

    def aggregate(self):
        self.a.stage = "all"
        inf = self.informative_tasks()
        rows = self.rows()
        base = [r for r in rows if r["stage"] == "base" and "error" not in r]
        rank = []
        for r in base:
            for g, v in r.get("perm", {}).items():
                rank.append({"family": g, "model": r["model"], "task": r["task"], "fold": r["fold"], "delta_loss": v,
                             "val_loss": r["val_loss"], "prior_loss": r["prior_loss"], "train_loss": r["train_loss"],
                             "n_params": r["n_params"]})
        pd.DataFrame(rank).to_csv(self.out / "RANKING.csv", index=False)
        con = [r for r in rows if r["stage"] in ("contrast", "rescue")]
        pd.DataFrame([{k: v for k, v in r.items() if k != "perm"} for r in con]).to_csv(
            self.out / "CONTRASTS_PARENT_COMPONENTS.csv", index=False)
        dec = self.decisions()
        pd.DataFrame([{"family": f, "status": d["status"], "n_cols": len(self.families[f]), "retained_cols": len(d["retained_cols"]),
                       "retained_arm": d.get("retained_arm", ""), "n_evidence": len(d["evidence"]),
                       "best_gain_core15": d["best_gain_core15"]} for f, d in dec.items()]).to_csv(self.out / "FAMILY_DECISIONS.csv", index=False)
        core = [c for c in self.cols if c.split("@")[0] in CORE15]
        short = []
        for f, d in dec.items():
            short += [c for c in d["retained_cols"] if c not in short]
        lists = {"core15": core, "shortlist": short, "core15_plus_shortlist": core + [c for c in short if c not in core]}
        (self.out / "SHORTLIST.json").write_text(json.dumps({"lists": lists, "informative_tasks": inf, "stores": self.pins,
                                                             "decisions": dec}, indent=1))
        errors = [r for r in rows if "error" in r]
        L = ["# Pool selection v2", "", f"Columns {len(self.cols)} in {len(self.families)} families; units done {len(rows)}; failed {len(errors)}.",
             f"Informative tasks (HGB base CV loss below prior): {inf}", ""]
        L += ["## Base losses (full pool)", "", "| task | model | CV loss | prior | train loss | params |", "|---|---|---|---|---|---|"]
        for t in TASKS:
            for mdl in ("hgb", "temporal"):
                rr = [r for r in base if r["task"] == t and r["model"] == mdl]
                if rr:
                    L.append(f"| {t} | {mdl} | {np.mean([r['val_loss'] for r in rr]):.4f} | {np.mean([r['prior_loss'] for r in rr]):.4f} | "
                             f"{np.mean([r['train_loss'] for r in rr]):.4f} | {rr[0]['n_params']} |")
        L += ["", "## Family decisions", "", "| family | cols | status | retained arm | evidence |", "|---|---|---|---|---|"]
        for f, d in sorted(dec.items(), key=lambda kv: (kv[1]["status"], kv[0])):
            ev = "; ".join(f"{e['kind']}:{e.get('task')}:{e.get('arm', e.get('model', ''))} +{e['mean']:.4f} (lo90 {e['lo90']:.4f})" for e in d["evidence"][:4])
            L.append(f"| {f} | {len(self.families[f])} | {d['status']} | {d.get('retained_arm', '')} | {ev} |")
        if errors:
            L += ["", "## Failed units", ""] + [f"- {r['key']}: {r['error']}" for r in errors[:30]]
        (self.out / "SELECTION_V2.md").write_text("\n".join(L) + "\n")
        print(json.dumps({"shortlist": len(short), "retained_families": sum(d["status"] != "NOT_RETAINED_WITH_THIS_EVIDENCE" for d in dec.values()),
                          "families": len(dec), "errors": len(errors)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("plan", "run", "aggregate"))
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--store", action="append", required=True, help="store directory (with MANIFEST.json) or parquet; repeatable")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-end", default="2024-01-01")
    ap.add_argument("--stage", default="all", choices=("base", "contrast", "rescue", "all"))
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    eng = Engine(a)
    getattr(eng, a.cmd)()


if __name__ == "__main__":
    main()

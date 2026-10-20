#!/usr/bin/env python3
"""Predictive validation of a timing feature selection through the pivot-detector harness (CPU, DEVELOPMENT).

Expert-demonstration / hindsight-guided learning: the label is the ZigZag expert's FUTURE pivot (T1, primary,
anticipatory); every input is causal (bars <= the completed origin bar).  Usefulness established here is for
this PROXY (pivot classification + the trivial reversal strategy), not for an RL policy.

Periods (plan audit 2026-10-19, F01/F02/F06):
  TRAIN_2020_2022    fit only
  INNER_A_2023H1     early stopping, temperature, regularization-cell choice, operating threshold
  INNER_B_2023H2     scored with every choice frozen (never updates a choice)
  REOBSERVED_2024    PREVIOUSLY OBSERVED development / reconfirmation (the expert's CHOICE and earlier
                     detectors already reported 2024); read once by ``reconfirm`` after ``freeze`` has sealed
                     the candidate; never a pristine holdout.  2023 is JOINT development: the expert's
                     parameters were chosen with 2023.
Labels arrive purged/censored from the predictor driver's support ledger (cuts include 2023-07-01).

Stages:  develop (one shard = one feature set x one arm; fits every regularization cell x seed; never reads
2024 rows), freeze (cell, temperature, threshold on INNER_A; INNER_B scored; FROZEN.json sealed),
reconfirm (2024 with the frozen models/choices only), seal (summary, trial ledger, read ledger).

Monitor (owner's rule, F06 measurement fix): score = 0.5 * (TRAIN loss + INNER_A loss), both evaluated in
inference mode at the same checkpoint with the same TRAIN class weights; the running training loss is
logged separately and never enters the score.  Patience 10, min_delta 1e-8, best weights restored; budget,
best epoch / iteration and stop cause logged.

Arms and the declared regularization grid (owner decision 2026-10-19; the default cell is primary):
  m1_dmo_class  13 slots x 2 h, per-feature causal Conv1D branches (grouped conv), flatten fusion + origin
                time encoding, Dense core, DROPOUT after the core, softmax head; AdamW
  m2_gru        GRU(32) on the 48-bar window (no recurrent dropout), DROPOUT on the GRU output, Dense(16),
                softmax; AdamW
                grid {dropout 0, 0.1, 0.2} x {weight_decay 1e-4, 1e-3}; primary (0.1, 1e-4); no BatchNorm,
                no LayerNorm
  m3_hgb        HistGradientBoosting, max_depth 4, min_samples_leaf 200, learning_rate 0.05, max_iter 400,
                l2_regularization in {0, 1, 10}; primary l2 0; the same monitor every 10 iterations
  Cell rule (pre-declared): lowest mean INNER_A loss at the restored checkpoint across seeds, ties by the
  smaller mean TRAIN-INNER gap; a non-primary winner is recorded as a replacement in the trial ledger.

Strategy: the sealed adapter ``tools.short_horizon_zigzag_study.Ctx.run`` (TP 2 sigma, SL 2 sigma,
RealisticProfile costs, order rules, <= 4 orders/day, validity >= 3 orders/week).  An evaluator-side
REOPEN GUARD drops a signal whose fill bar precedes the segment's operational start (12 h after the
weekly open): the pinned ``zigzag.signal_blocks`` path does not apply that rule (``blocks.build_blocks``
does).  The frequency rule only marks validity; it never adds an entry.
"""
from __future__ import annotations

import argparse
import hashlib
import types
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import short_horizon_pivot_detector as PD  # noqa: E402
from tools import short_horizon_zigzag_study as STUDY  # noqa: E402
from tools.short_horizon_zigzag_study import Ctx, params_of  # noqa: E402
from app.short_horizon import zigzag as Z  # noqa: E402

try:                                            # the v2 expert's executable fill path (abbfaa6); absent before it
    from app.short_horizon import zigzag_opt as ZO  # noqa: E402
except ImportError:  # pragma: no cover
    ZO = None
_EXEC = {"executable": False, "cap": 4, "zo": None}

EVAL_PERIODS = {"TRAIN_2020_2022": ("2020-01-01", "2023-01-01"), "INNER_A_2023H1": ("2023-01-01", "2023-07-01"),
                "INNER_B_2023H2": ("2023-07-01", "2024-01-01"), "REOBSERVED_2024": ("2024-01-01", "2025-01-01")}
STUDY.PERIODS.update(EVAL_PERIODS)            # runtime registration of the evaluation splits (no adapter edit)
if ZO is not None:
    ZO.PERIODS.update(EVAL_PERIODS)
DEVELOP_PERIODS = ("TRAIN_2020_2022", "INNER_A_2023H1", "INNER_B_2023H2")
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
ARMS = ("m1_dmo_class", "m2_gru", "m3_hgb")
WINDOW, SLOTS, STRIDE = 48, 13, 2
SEEDS = (6102026, 6102027, 6102028)
TRAINING = dict(batch_size=256, learning_rate=1e-3, patience=10, min_delta=1e-8, max_epochs=200,
                hgb_max_iter=400, hgb_eval_every=10, hgb_learning_rate=0.05, hgb_max_depth=4, hgb_min_samples_leaf=200)
GRID = {
    "m1_dmo_class": [{"dropout": d, "weight_decay": w} for d in (0.0, 0.1, 0.2) for w in (1e-4, 1e-3)],
    "m2_gru": [{"dropout": d, "weight_decay": w} for d in (0.0, 0.1, 0.2) for w in (1e-4, 1e-3)],
    "m3_hgb": [{"l2_regularization": v} for v in (0.0, 1.0, 10.0)],
}
PRIMARY = {"m1_dmo_class": {"dropout": 0.1, "weight_decay": 1e-4}, "m2_gru": {"dropout": 0.1, "weight_decay": 1e-4},
           "m3_hgb": {"l2_regularization": 0.0}}
SHARPE_DEFINITION = ("mean / sd (ddof 1) of daily returns of the marked-to-market equity on every calendar day of the "
                     "period (closures forward-filled), x sqrt(365); NaN when ruined")
BOOT = dict(block="ISO calendar week", reps=2000, seed=20261019)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canon(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def cell_id(cell: dict) -> str:
    return "_".join(f"{k}{v:g}" for k, v in sorted(cell.items()))


def _utc(s):
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _set_threads():
    try:
        import tensorflow as tf
        n = int(os.environ.get("TIMING_EVAL_THREADS", "4"))
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:  # noqa: BLE001 - threads already initialised in this process
        pass


# --------------------------------------------------------------------------------------- inputs
def windows(z: np.ndarray, rows: np.ndarray, window: int = WINDOW) -> np.ndarray:
    idx = rows[:, None] + np.arange(-window + 1, 1)[None, :]
    return z[idx]


def time_encoding(ts: pd.DatetimeIndex) -> np.ndarray:
    h, d = np.asarray(ts.hour, float), np.asarray(ts.dayofweek, float)
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)]).astype(np.float32)


def period_of(ts: pd.DatetimeIndex) -> np.ndarray:
    period = np.full(len(ts), "", object)
    for name, (a, b) in EVAL_PERIODS.items():
        period[(ts >= _utc(a)) & (ts < _utc(b))] = name
    return period


def input_identity(ev: Path) -> str:
    """The full evaluation-input identity: the parsed FEATURE_SETS contract (sets, expert, strategy incl. the daily
    cap, splits, source, eligibility digest) and the bytes of every input file it names."""
    ev = Path(ev)
    doc = json.loads((ev / "FEATURE_SETS.json").read_text())
    files = {n: sha(ev / n) for n in ("features_all.parquet", "targets_all.parquet", "ELIGIBILITY.parquet") if (ev / n).exists()}
    return canon({"feature_sets": doc, "files": files})


def load_inputs(ev: Path, *, include_2024: bool):
    """Features/targets (and the shared eligibility mask when the contract declares one) with digests verified;
    without ``include_2024`` no row at or after 2024-01-01 is kept.  Returns (sets_doc, feats, targ, ts, elig)."""
    sets_doc = json.loads((ev / "FEATURE_SETS.json").read_text())
    if sha(ev / "features_all.parquet") != sets_doc["features_sha256"] or sha(ev / "targets_all.parquet") != sets_doc["targets_sha256"]:
        raise SystemExit("EVAL_INPUT_DIGEST")
    feats = pd.read_parquet(ev / "features_all.parquet")
    targ = pd.read_parquet(ev / "targets_all.parquet")
    ts = pd.DatetimeIndex(feats["timestamp"])
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    if (ts >= pd.Timestamp("2025-01-01", tz="UTC")).any():
        raise SystemExit("TEST_ROWS_FORBIDDEN")
    elig = None
    spec = sets_doc.get("eligibility")
    if spec:
        if sha(ev / spec["file"]) != spec["file_sha256"]:
            raise SystemExit("ELIGIBILITY_DIGEST")
        em = pd.read_parquet(ev / spec["file"])
        ets = pd.DatetimeIndex(em["timestamp"]).tz_convert("UTC")
        if not np.array_equal(ets.asi8, ts.asi8):
            raise SystemExit("ELIGIBILITY_ROWS_MISALIGNED")
        elig = em[spec["column"]].to_numpy(bool)
    if not include_2024:
        keep = ts < pd.Timestamp("2024-01-01", tz="UTC")
        feats, targ, ts = feats.loc[keep].reset_index(drop=True), targ.loc[keep].reset_index(drop=True), ts[keep]
        elig = elig[np.asarray(keep)] if elig is not None else None
    return sets_doc, feats, targ, ts, elig


def common_rows(feats, targ, ts, sets_doc, k, elig=None):
    """Identical rows for every set and arm: a full finite 48-bar window over the UNION of compared columns,
    a labelled T1 (support ledger already purged/censored), inside a declared period.  Also returns, per set,
    the rows that set alone would have had (to report what the longest warm-up costs)."""
    sets = {n: c for n, c in sets_doc["sets"].items() if c}
    union = sorted({c for cols in sets.values() for c in cols})
    y = targ[sets_doc.get("target_column") or f"timing_pivot_k{k}"].to_numpy(float)   # v5: family-chosen column
    period = period_of(ts)

    if elig is not None and (np.isfinite(y) & ~elig).any():
        raise SystemExit("ELIGIBILITY_LABEL_MISMATCH: a finite label sits on an ineligible origin")
    gate = elig if elig is not None else np.ones(len(y), bool)

    def ok_for(cols):
        fin = np.isfinite(feats[cols].to_numpy(float)).all(axis=1)
        win = pd.Series(fin).rolling(WINDOW).min().fillna(0).to_numpy().astype(bool)
        return win & np.isfinite(y) & (period != "") & gate

    common = ok_for(union)
    alone = {n: ok_for(cols) for n, cols in sets.items()}
    loss = {n: {p: int((alone[n] & (period == p)).sum() - (common & (period == p)).sum()) for p in EVAL_PERIODS} for n in sets}
    return common, period, y, sets, loss


# --------------------------------------------------------------------------------------- monitor
class OwnerMonitor:
    """score = 0.5 * (TRAIN loss + INNER_A loss), both in inference mode at the same checkpoint, same weighting."""

    def __init__(self, cw, y_tr, y_in, patience, min_delta, budget):
        self.cw, self.y_tr, self.y_in = cw, y_tr, y_in
        self.patience, self.min_delta, self.budget = patience, min_delta, budget
        self.best, self.best_step, self.wait, self.best_state, self.history = np.inf, -1, 0, None, []
        self.stop_cause = "BUDGET_EXHAUSTED"

    def wloss(self, p, y):
        p = np.clip(np.asarray(p, float), 1e-9, 1)
        w = self.cw[y]
        return float(np.sum(w * -np.log(p[np.arange(len(y)), y])) / np.sum(w))

    def step(self, step, p_tr, p_in, state_getter, running=None) -> bool:
        tr, inn = self.wloss(p_tr, self.y_tr), self.wloss(p_in, self.y_in)
        score = 0.5 * (tr + inn)
        self.history.append(dict(step=int(step), train_eval_loss=tr, inner_a_loss=inn, score=score, gap=inn - tr,
                                 running_train_loss=running))
        if score < self.best - self.min_delta:
            self.best, self.best_step, self.wait, self.best_state = score, int(step), 0, state_getter()
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stop_cause = "PATIENCE"
            return True
        return False

    def summary(self):
        b = next(h for h in self.history if h["step"] == self.best_step) if self.best_step >= 0 else {}
        return dict(best_step=self.best_step, steps_run=len(self.history), budget=self.budget, patience=self.patience,
                    min_delta=self.min_delta, stop_cause=self.stop_cause, best_score=self.best,
                    best_inner_a_loss=b.get("inner_a_loss"), best_train_eval_loss=b.get("train_eval_loss"),
                    best_gap=b.get("gap"), history=self.history)


def class_weights(y_tr):
    counts = np.bincount(y_tr, minlength=3).astype(float)
    return (1.0 / np.maximum(counts, 1)) * counts.sum() / 3.0


def _keras_fit(model, x_tr, y_tr, x_in, y_in, cw, seed):
    import tensorflow as tf
    keras = tf.keras
    mon = OwnerMonitor(cw, y_tr, y_in, TRAINING["patience"], TRAINING["min_delta"], TRAINING["max_epochs"])

    class _CB(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            p_tr = self.model.predict(x_tr, verbose=0, batch_size=4096)
            p_in = self.model.predict(x_in, verbose=0, batch_size=4096)
            if mon.step(epoch, p_tr, p_in, self.model.get_weights, running=float((logs or {}).get("loss", np.nan))):
                self.model.stop_training = True

    model.fit(x_tr, y_tr, sample_weight=cw[y_tr], batch_size=TRAINING["batch_size"], epochs=TRAINING["max_epochs"],
              verbose=0, shuffle=True, callbacks=[_CB()])
    if mon.best_state is not None:
        model.set_weights(mon.best_state)
    return model, mon.summary()


def _compile(model, cell):
    import tensorflow as tf
    keras = tf.keras
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=TRAINING["learning_rate"], weight_decay=cell["weight_decay"]),
                  loss="sparse_categorical_crossentropy")
    return model


def build_dmo_class(F, n_time, cell, filters=4, core=32):
    import tensorflow as tf
    keras = tf.keras
    inp = keras.Input(shape=(SLOTS, F))
    tin = keras.Input(shape=(n_time,))
    b = keras.layers.Conv1D(filters * F, 3, padding="causal", groups=F, activation="relu")(inp)
    b = keras.layers.Conv1D(filters * F, 3, padding="causal", groups=F, activation="relu")(b)
    fused = keras.layers.Concatenate()([keras.layers.Flatten()(b), tin])
    h = keras.layers.Dense(core, activation="relu")(fused)
    h = keras.layers.Dropout(cell["dropout"])(h)
    return _compile(keras.Model([inp, tin], keras.layers.Dense(3, activation="softmax")(h)), cell)


def build_gru(F, cell):
    import tensorflow as tf
    keras = tf.keras
    model = keras.Sequential([keras.Input(shape=(WINDOW, F)), keras.layers.GRU(32, recurrent_dropout=0.0),
                              keras.layers.Dropout(cell["dropout"]), keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(3, activation="softmax")])
    return _compile(model, cell)


class StagedHGB:
    """A boosting model restored to its monitored best iteration."""

    def __init__(self, clf, n_iter):
        self.clf, self.n_iter = clf, int(n_iter)

    def predict(self, x, verbose=0, batch_size=0):
        p = None
        for i, p in enumerate(self.clf.staged_predict_proba(x), start=1):
            if i == self.n_iter:
                return p
        return p


def fit_hgb(x_tr, y_tr, x_in, y_in, seed, cell):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cw = class_weights(y_tr)
    clf = HistGradientBoostingClassifier(learning_rate=TRAINING["hgb_learning_rate"], max_depth=TRAINING["hgb_max_depth"],
                                         min_samples_leaf=TRAINING["hgb_min_samples_leaf"], max_leaf_nodes=None,
                                         l2_regularization=cell["l2_regularization"], max_iter=TRAINING["hgb_max_iter"],
                                         early_stopping=False, random_state=int(seed))
    clf.fit(x_tr, y_tr, sample_weight=cw[y_tr])
    mon = OwnerMonitor(cw, y_tr, y_in, TRAINING["patience"], TRAINING["min_delta"], TRAINING["hgb_max_iter"])
    every = TRAINING["hgb_eval_every"]
    for i, (p_tr, p_in) in enumerate(zip(clf.staged_predict_proba(x_tr), clf.staged_predict_proba(x_in)), start=1):
        if i % every:
            continue
        if mon.step(i, p_tr, p_in, lambda i=i: i):
            break
    return StagedHGB(clf, mon.best_state or TRAINING["hgb_max_iter"]), mon.summary()


def arm_inputs(arm, Z, tenc, rows):
    W = windows(Z, rows)
    if arm == "m1_dmo_class":
        return [W[:, WINDOW - 1 - STRIDE * (SLOTS - 1):: STRIDE, :], tenc[rows]]
    if arm == "m2_gru":
        return W
    return W.reshape(len(rows), -1)


def fit_arm(arm, cell, xa_tr, y_tr, xa_in, y_in, seed, F):
    if arm == "m3_hgb":
        return fit_hgb(xa_tr, y_tr, xa_in, y_in, seed, cell)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(int(seed))
    model = build_dmo_class(F, 4, cell) if arm == "m1_dmo_class" else build_gru(F, cell)
    return _keras_fit(model, xa_tr, y_tr, xa_in, y_in, class_weights(y_tr), seed)


def save_model(model, path: Path, arm):
    """One file per fit: the suffix is APPENDED to the full key (``Path.with_suffix`` would cut a cell id such as
    ``weight_decay0.0001`` at its decimal point and make different fits overwrite one file)."""
    path = path.parent / (path.name + (".pkl" if arm == "m3_hgb" else ".keras"))
    if path.exists():
        raise SystemExit(f"MODEL_FILE_EXISTS: {path.name}")
    if arm == "m3_hgb":
        path.write_bytes(pickle.dumps(model))
    else:
        model.save(path)
    return path


def load_model(path: Path, arm):
    if arm == "m3_hgb":
        return pickle.loads(path.read_bytes())
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def scaler(feats, cols, rows_train):
    X = feats[cols].to_numpy(float)
    mu, sd = X[rows_train].mean(0), X[rows_train].std(0)
    return mu, np.where(sd > 0, sd, 1.0)


def scaled(feats, cols, mu, sd):
    Z = ((feats[cols].to_numpy(float) - mu) / sd).astype(np.float32)
    Z[~np.isfinite(Z)] = 0.0          # only outside the identical rows' windows (they are finite by construction)
    return Z


# --------------------------------------------------------------------------------------- stage: develop
def develop(a):
    _set_threads()
    ev, out = Path(a.eval_input), Path(a.out)
    sets_doc, feats, targ, ts, elig = load_inputs(ev, include_2024=False)
    k = int(sets_doc["k_bars"])
    common, period, y_raw, sets, loss = common_rows(feats, targ, ts, sets_doc, k, elig)
    cols = sets[a.set]
    rows = np.flatnonzero(common)
    per = period[rows]
    y = y_raw[rows].astype(np.int64)
    tr, ia = per == "TRAIN_2020_2022", per == "INNER_A_2023H1"
    mu, sd = scaler(feats, cols, rows[tr])
    Z = scaled(feats, cols, mu, sd)
    xa = arm_inputs(a.arm, Z, time_encoding(ts), rows)
    sel = (lambda m: [v[m] for v in xa]) if isinstance(xa, list) else (lambda m: xa[m])
    shard = out / "shards" / f"{a.set}__{a.arm}"
    (shard / "models").mkdir(parents=True, exist_ok=True)
    fits, preds = {}, {}
    for cell in GRID[a.arm]:
        for seed in a.seeds:
            key = f"{cell_id(cell)}__s{seed}"
            t0 = time.time()
            model, fit = fit_arm(a.arm, cell, sel(tr), y[tr], sel(ia), y[ia], seed, len(cols))
            fit.update(cell=cell, seed=seed, wall_seconds=time.time() - t0, primary=cell == PRIMARY[a.arm])
            mpath = save_model(model, shard / "models" / key, a.arm)
            fit["model_file"], fit["model_sha256"] = mpath.name, sha(mpath)
            preds[key] = model.predict(xa, verbose=0, batch_size=4096).astype(np.float32)
            fits[key] = fit
            print(json.dumps({"shard": shard.name, "fit": key, "best_step": fit["best_step"], "stop": fit["stop_cause"],
                              "inner_a": fit["best_inner_a_loss"], "s": round(fit["wall_seconds"], 1)}), flush=True)
    np.savez_compressed(shard / "develop_predictions.npz", rows=rows, y=y, period=per.astype(str),
                        **{k_: v for k_, v in preds.items()})
    doc = dict(stage="develop", set=a.set, arm=a.arm, columns=cols, scaler={"mean": mu.tolist(), "sd": sd.tolist(), "fit_on": "TRAIN_2020_2022 identical rows"},
               grid=GRID[a.arm], primary=PRIMARY[a.arm], seeds=list(a.seeds), training=TRAINING, fits=fits,
               rows_by_period={p: int((per == p).sum()) for p in DEVELOP_PERIODS},
               common_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(), rows_lost_to_union_warmup=loss[a.set],
               predictions_sha256=sha(shard / "develop_predictions.npz"), periods_read=list(DEVELOP_PERIODS),
               reads_2024=False, eval_input={"features_sha256": sets_doc["features_sha256"], "targets_sha256": sets_doc["targets_sha256"]},
               input_identity=input_identity(ev), eligibility_applied=elig is not None)
    (shard / "DEVELOP.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")


# --------------------------------------------------------------------------------------- scoring helpers
def reopen_guard(signals, ctx, delay_bars: int = 1):
    """Drop a signal whose fill bar (signal bar + 1 + entry delay) precedes its segment's operational start."""
    v = ctx.bars.index
    seg_of = ctx.cal.assign(v)
    kept, dropped = [], 0
    for b, d in signals:
        f = b + 1 + delay_bars
        if f >= len(v):
            kept.append((b, d))
            continue
        seg = ctx.cal.segments[int(seg_of[f])]
        if v[f] < seg.operational_start:
            dropped += 1
            continue
        kept.append((b, d))
    return kept, dropped


def per_direction(P, y, sig, origin_bar, mask):
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    yy, PP = y[mask], P[mask]
    out = {"rows": int(mask.sum()), "log_loss": float(log_loss(yy, PP, labels=[0, 1, 2]))}
    bar_y = {int(b): int(v) for b, v, m in zip(origin_bar, y, mask) if m}
    for c, name, dsign in ((1, "valley_long", 1), (2, "peak_short", -1)):
        pos = yy == c
        out[f"{name}_support"] = int(pos.sum())
        out[f"{name}_base_rate_incl_none"] = float(pos.mean()) if len(yy) else math.nan
        out[f"{name}_pr_auc"] = float(average_precision_score(pos, PP[:, c])) if pos.any() else math.nan
        out[f"{name}_roc_auc"] = float(roc_auc_score(pos, PP[:, c])) if pos.any() and (~pos).any() else math.nan
        s = [(t, d) for t, d in sig if d == dsign and t in bar_y]
        hits = sum(bar_y[t] == c for t, _ in s)
        out[f"{name}_signals"] = len(s)
        out[f"{name}_precision"] = float(hits / len(s)) if s else math.nan
        out[f"{name}_recall"] = float(hits / pos.sum()) if pos.any() else math.nan
    return out


def trade_breakdown(res, ctx):
    tr = res.trades
    ff = {seg.segment_id: seg.force_flat for seg in ctx.cal.segments}
    seg_ids = [s.segment_id for s in ctx.cal.segments]
    forced = sum(1 for t in tr if t["exit_reason"] == "TIME" and pd.Timestamp(t["exit_time"]) >= ff[seg_ids[int(t["segment"])]])
    entries = [pd.Timestamp(t["entry_time"]) for t in tr]
    # close-and-reverse: a TIME exit that is not a forced flat, followed in the same segment within 2 bars by an
    # entry in the opposite direction (the pinned adapter closes at the block end and reopens at the next fill)
    reverse = sum(1 for t, nxt in zip(tr, tr[1:]) if t["exit_reason"] == "TIME" and int(nxt["segment"]) == int(t["segment"])
                  and pd.Timestamp(t["exit_time"]) < ff[seg_ids[int(t["segment"])]] and nxt["direction"] == -t["direction"]
                  and int(nxt["entry_bar_abs"]) - int(t["exit_bar_abs"]) <= 2)
    m = res.metrics
    return {"entries_filled": len(entries), "closes_tp": m.get("exit_tp"), "closes_sl": m.get("exit_sl"),
            "closes_time": m.get("exit_time"), "closes_forced_flat": int(forced), "close_and_reverse": int(reverse),
            "orders_submitted": m.get("orders_submitted"), "refused_entries": m.get("refused_entries"),
            "unfilled_orders": m.get("unfilled_orders"), "capped_orders": m.get("capped_orders"),
            "below_min_units": m.get("below_min_units"), "margin_refused_orders": m.get("margin_refused_orders"),
            "cost_spread_usd": m.get("cost_spread_usd"), "cost_slippage_usd": m.get("cost_slippage_usd"),
            "cost_commission_usd": m.get("cost_commission_usd"), "cost_financing_usd": m.get("cost_financing_usd"),
            "gross_usd": m.get("gross_usd"), "ambiguous_stop_first": m.get("ambiguous_stop_first")}


def calendar_from_closures(closures, index, reopen_hours=12.0, preclose_hours=12.0):
    """The adapter calendar for the admitted source: sessions = complements of the declared closure intervals,
    with the business rules (entry delay after open, forced flat before close, no extra cutoff)."""
    from app.short_horizon import calendar as CAL
    cl = sorted((pd.Timestamp(a).tz_convert("UTC"), pd.Timestamp(b).tz_convert("UTC")) for a, b in closures)
    lo, hi = index[0], index[-1] + pd.Timedelta(hours=1)
    segs = []
    for (a0, b0), (a1, b1) in zip(cl, cl[1:]):
        if b0 < hi and a1 > lo:
            segs.append({"segment_id": f"ADM-{b0:%Y%m%dT%H}", "market_open": str(b0), "market_close": str(a1)})
    cfg = {"calendar_id": "ADMITTED_CLOSURE_INTERVALS", "version": "v1", "segments": segs,
           "entry_delay_after_open_hours": float(reopen_hours), "force_flat_before_close_hours": float(preclose_hours),
           "entry_cutoff_before_flat_hours": 0}
    return CAL.load_calendar(cfg)


def make_ctx(sets_doc, a, ts):
    """Bars + calendar for the strategy.  Admitted source: bars through the clock lane's ``bar_source``
    (admitted, UTC bar start, digest-pinned) and the calendar from the same CLOSURE_INTERVALS.json (pinned).
    Legacy lake source: the pinned ``Ctx``."""
    src = sets_doc.get("source") or {}
    if src.get("kind") != "admitted":
        return Ctx(a.data)
    from app.short_horizon import bar_source as BS
    if not a.closures:
        raise SystemExit("ADMITTED_SOURCE_NEEDS_CLOSURES")
    if sha(a.data) != src["sha256"] or sha(a.closures) != src["closures_sha256"]:
        raise SystemExit("ADMITTED_SOURCE_DIGEST")
    bars = BS.load_ohlc(a.data, ts[0], pd.Timestamp("2025-01-01", tz="UTC"), source="admitted")
    bars.index = pd.DatetimeIndex(bars.index).tz_convert("UTC").rename("DATE_TIME")
    bc = sets_doc.get("business_calendar") or {}
    cal = calendar_from_closures(json.loads(Path(a.closures).read_text()), bars.index, bc.get("reopen_hours", 12), bc.get("preclose_hours", 12))
    if not (sets_doc.get("strategy") or {}).get("executable"):
        raise SystemExit("ADMITTED_SOURCE_NEEDS_EXECUTABLE_PATH")
    return types.SimpleNamespace(bars=bars, cal=cal, h=bars.HIGH.to_numpy(float), l=bars.LOW.to_numpy(float), digest=src["sha256"])


def check_frozen_inputs(frozen: dict, ev: Path):
    """Refuse any drift of the evaluation inputs (expert, strategy incl. cap, sets, population, labels, data,
    eligibility) since the freeze, BEFORE any data or model is loaded."""
    want = frozen.get("input_identity")
    if want is None:
        raise SystemExit("FROZEN_WITHOUT_INPUT_IDENTITY")
    got = input_identity(ev)
    if got != want:
        raise SystemExit(f"INPUT_IDENTITY_DRIFT: frozen {want[:12]} != current {got[:12]}")


def classification_baselines(Ps, y, prior, mask):
    """Ensemble (seed-mean) probabilities against the constant TRAIN-prior classifier on the same rows:
    log-loss and its difference, per-direction PR-AUC with the base rate (its no-skill baseline, NONE in the
    denominator), Brier score and expected calibration error (10 equal-width bins)."""
    from sklearn.metrics import average_precision_score, log_loss
    P = np.mean(Ps, axis=0)[mask]
    yy = y[mask]
    pr = np.tile(prior, (len(yy), 1))
    out = {"rows": int(mask.sum()), "ensemble_log_loss": float(log_loss(yy, P, labels=[0, 1, 2])),
           "train_prior_log_loss": float(log_loss(yy, pr, labels=[0, 1, 2]))}
    out["log_loss_minus_train_prior"] = out["ensemble_log_loss"] - out["train_prior_log_loss"]
    for c, name in ((1, "valley"), (2, "peak")):
        pos = (yy == c).astype(float)
        p = P[:, c]
        bins = np.clip((p * 10).astype(int), 0, 9)
        ece = sum(abs(pos[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
        out.update({f"{name}_pr_auc": float(average_precision_score(pos, p)) if pos.any() else np.nan,
                    f"{name}_pr_baseline_base_rate": float(pos.mean()), f"{name}_brier": float(np.mean((p - pos) ** 2)),
                    f"{name}_brier_train_prior": float(np.mean((prior[c] - pos) ** 2)), f"{name}_ece": float(ece)})
    return out


def setup_expert(sets_doc, ctx, zigzag_run):
    """Expert pivots and the strategy rule.  From FEATURE_SETS.json when the driver wrote an ``expert`` block
    (module digest verified against the loaded zigzag module); otherwise the legacy v1 CHOICE.json."""
    exp, strat = sets_doc.get("expert"), sets_doc.get("strategy") or {}
    if exp:
        actual = sha(Path(Z.__file__))
        if actual != exp["module_sha256"]:
            raise SystemExit(f"EXPERT_MODULE_DIGEST: loaded zigzag {actual[:12]} != driver's {exp['module_sha256'][:12]}")
        pp = exp["params"]
        p = Z.Params(depth=int(pp["depth"]), deviation_pips=float(pp["deviation_pips"]), backstep=int(pp["backstep"]))
        k_tp = float(strat.get("k_tp", 2.0))
    else:
        p = params_of(json.loads((Path(zigzag_run) / "CHOICE.json").read_text())["params"])
        k_tp = float(pd.read_csv(Path(zigzag_run) / "confirmed.csv").query("chosen").k_tp.iloc[0])
    _EXEC["executable"] = bool(strat.get("executable", False))
    _EXEC["cap"] = int(strat.get("max_orders_per_day", 4))
    if _EXEC["executable"]:
        if ZO is None:
            raise SystemExit("EXECUTABLE_PATH_ABSENT: zigzag_opt not importable")
        _EXEC["zo"] = ZO.Evaluator(ctx.bars, ctx.cal)
    return p, Z.hindsight_pivots(ctx.h, ctx.l, p), k_tp


def strategy(ctx, piv, Pc, origin_bar, tau, k, k_tp, period):
    sig = PD.signals_from(Pc, origin_bar, tau, k)
    if _EXEC["executable"]:
        # owner's executable path: decision after the completed bar b, fill at the open of b + 1
        # (executable_profile, entry_delay 0); entries before the operational start are refused by the adapter
        out, res = _EXEC["zo"].run(sig, piv, period, "causal", k_tp=k_tp, cap=_EXEC["cap"], executable=True)
        m = {kk: res.metrics[kk] for kk in STUDY.KEEP}
        m.update(valid=out["valid"], rule_skipped_orders=out["rule_skipped_orders"],
                 refused_before_operational_start=out["refused_before_operational_start"], fill_path="EXECUTABLE_B_PLUS_1")
        dropped = 0
    else:
        sig, dropped = reopen_guard(sig, ctx)
        m, res = ctx.run(sig, piv, period, mode="causal", k_tp=k_tp)
        m = {kk: v for kk, v in m.items() if kk != "period"}
        m["fill_path"] = "LEGACY_B_PLUS_2_WITH_REOPEN_GUARD"
    m.update(trade_breakdown(res, ctx), signals_emitted=len(sig), signals_dropped_reopen_guard=dropped)
    return m, res, sig


def daily_returns(res, initial: float = None):
    """Daily returns of the marked-to-market equity, the first day measured against the INITIAL capital (the
    execution accounting's convention), not against the first daily close."""
    d = res.daily_equity
    base = float(STUDY.CAPITAL if initial is None else initial)
    v = d.to_numpy(float)
    return pd.Series(np.diff(np.r_[base, v]) / np.r_[base, v[:-1]], index=d.index)


def _daily_returns_legacy(res):
    d = res.daily_equity
    return pd.Series(np.diff(np.r_[d.iloc[0], d.to_numpy()]) / np.r_[d.iloc[0], d.to_numpy()[:-1]], index=d.index)


def block_bootstrap(stat_fn, frame: pd.DataFrame, *, reps=BOOT["reps"], seed=BOOT["seed"]):
    """Resample whole ISO weeks with replacement (dependence-preserving within a week); ``frame`` is indexed by
    timestamp; returns the 2.5 / 50 / 97.5 % quantiles of ``stat_fn`` over replicates."""
    idx = pd.DatetimeIndex(frame.index)
    wk = idx.isocalendar().year.astype(str).to_numpy() + "-" + idx.isocalendar().week.astype(str).to_numpy()
    groups = {}
    for i, w in enumerate(wk):
        groups.setdefault(w, []).append(i)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        ii = np.concatenate([groups[keys[j]] for j in pick])
        vals.append(stat_fn(frame.iloc[ii]))
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return {"q025": float(np.quantile(vals, 0.025)), "q500": float(np.quantile(vals, 0.5)), "q975": float(np.quantile(vals, 0.975)),
            "reps": int(len(vals)), "block": BOOT["block"]} if len(vals) else {"reps": 0}


def sharpe365(r):
    r = np.asarray(r, float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0


def deflated_sharpe(sr_daily, n_obs, skew, kurt, trial_srs):
    """Bailey & Lopez de Prado DSR (supplement): P[SR > SR0] with SR0 the expected maximum of N trial Sharpes."""
    from scipy.stats import norm
    trial_srs = np.asarray([s for s in trial_srs if np.isfinite(s)], float)
    N = len(trial_srs)
    if N < 2 or n_obs < 3:
        return {"dsr": math.nan, "trials": N}
    var = trial_srs.var(ddof=1)
    g = 0.5772156649
    sr0 = math.sqrt(var) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * math.e)))
    den = math.sqrt(max(1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2, 1e-12))
    z = (sr_daily - sr0) * math.sqrt(n_obs - 1) / den
    return {"dsr": float(norm.cdf(z)), "trials": N, "sr0_daily": float(sr0), "sr_daily": float(sr_daily)}


# --------------------------------------------------------------------------------------- stage: freeze
def choose_cell(fits, arm):
    by = {}
    for key, f in fits.items():
        by.setdefault(cell_id(f["cell"]), []).append(f)
    rows = []
    for cid, fs in by.items():
        rows.append({"cell": fs[0]["cell"], "cell_id": cid, "mean_inner_a_loss": float(np.mean([f["best_inner_a_loss"] for f in fs])),
                     "mean_gap": float(np.mean([f["best_gap"] for f in fs])), "seeds": len(fs), "primary": fs[0]["cell"] == PRIMARY[arm]})
    rows.sort(key=lambda r: (r["mean_inner_a_loss"], r["mean_gap"]))
    win = rows[0]
    return win, rows, {"rule": "lowest mean INNER_A loss at the restored checkpoint across seeds, ties by smaller mean gap",
                       "winner": win["cell_id"], "primary": cell_id(PRIMARY[arm]), "replaced_primary": not win["primary"]}


def freeze(a):
    ev, out = Path(a.eval_input), Path(a.out)
    cur_identity = input_identity(ev)
    sets_doc, feats, targ, ts, elig = load_inputs(ev, include_2024=False)
    k = int(sets_doc["k_bars"])
    ctx = make_ctx(sets_doc, a, ts)
    p, piv, k_tp = setup_expert(sets_doc, ctx, a.zigzag_run)
    lake_ts = ctx.bars.index.as_unit("s").asi8
    if not np.array_equal(lake_ts[: len(ts)], ts.as_unit("s").asi8):
        raise SystemExit("EVAL_INPUT_GRID_DIFFERS_FROM_LAKE")
    frozen, scored, trials, baselines = {"shards": {}}, [], [], []
    for shard in sorted((out / "shards").iterdir()):
        dev = json.loads((shard / "DEVELOP.json").read_text())
        if sha(shard / "develop_predictions.npz") != dev["predictions_sha256"]:
            raise SystemExit("DEVELOP_PREDICTIONS_DIGEST:" + shard.name)
        if dev.get("input_identity") is not None and dev["input_identity"] != cur_identity:
            raise SystemExit("SHARD_INPUT_IDENTITY_DRIFT:" + shard.name)
        z = np.load(shard / "develop_predictions.npz", allow_pickle=False)
        rows, y, per = z["rows"], z["y"], z["period"].astype(str)
        win, cell_table, rule = choose_cell(dev["fits"], dev["arm"])
        ia, ib, trn = per == "INNER_A_2023H1", per == "INNER_B_2023H2", per == "TRAIN_2020_2022"
        seeds = sorted(f["seed"] for f in dev["fits"].values() if cell_id(f["cell"]) == win["cell_id"])
        temps, cal = {}, {}
        for s in seeds:
            P = z[f"{win['cell_id']}__s{s}"].astype(float)
            temps[s] = PD.temperature(P, y, ia)
            cal[s] = PD.calibrate(P, temps[s])
        # threshold on INNER_A (mean Sharpe over seeds, valid first); every attempt recorded in the trial ledger
        grid = []
        for tau in THRESHOLDS:
            ms = []
            for s in seeds:
                m, _, _ = strategy(ctx, piv, cal[s], rows, tau, k, k_tp, "INNER_A_2023H1")
                ms.append(m)
                trials.append(dict(set=dev["set"], arm=dev["arm"], cell=win["cell_id"], seed=s, threshold=tau, period="INNER_A_2023H1",
                                   sharpe=m["sharpe_daily_sqrt365"], valid=m["valid"], entries=m["entries_filled"]))
            grid.append(dict(threshold=tau, sharpe=float(np.nanmean([m["sharpe_daily_sqrt365"] for m in ms])),
                             valid=bool(all(m["valid"] for m in ms))))
        gv = [g for g in grid if g["valid"]]
        tau = max(gv or grid, key=lambda g: (g["sharpe"], -g["threshold"]))["threshold"]
        key = f"{dev['set']}__{dev['arm']}"
        models = {s: dev["fits"][f"{win['cell_id']}__s{s}"] for s in seeds}
        frozen["shards"][key] = dict(set=dev["set"], arm=dev["arm"], cell=win["cell"], cell_choice=rule, cell_table=cell_table,
                                     temperatures={str(s): t for s, t in temps.items()}, threshold=tau, threshold_grid_inner_a=grid,
                                     models={str(s): {"file": m["model_file"], "sha256": m["model_sha256"]} for s, m in models.items()},
                                     columns=dev["columns"], scaler=dev["scaler"], scaler_sha256=canon(dev["scaler"]),
                                     common_rows_sha256=dev["common_rows_sha256"], develop_sha256=sha(shard / "DEVELOP.json"))
        prior = np.bincount(y[trn], minlength=3) / trn.sum()
        frozen["shards"][key]["train_prior"] = prior.tolist()
        for pname, mask in (("INNER_A_2023H1", ia), ("INNER_B_2023H2", ib)):
            baselines.append(dict(set=dev["set"], arm=dev["arm"], period=pname, **classification_baselines([cal[s] for s in seeds], y, prior, mask)))
        # scoring with every choice frozen: TRAIN (diagnostic), INNER_A (where chosen), INNER_B (frozen)
        for pname, mask in (("TRAIN_2020_2022", trn), ("INNER_A_2023H1", ia), ("INNER_B_2023H2", ib)):
            for s in seeds:
                m, res, sig = strategy(ctx, piv, cal[s], rows, tau, k, k_tp, pname)
                cls = per_direction(cal[s], y, sig, rows, mask)
                scored.append(dict(set=dev["set"], arm=dev["arm"], cell=win["cell_id"], seed=s, threshold=tau, period=pname,
                                   role={"TRAIN_2020_2022": "FIT_DIAGNOSTIC", "INNER_A_2023H1": "CHOICE_BLOCK",
                                         "INNER_B_2023H2": "FROZEN_SCORE"}[pname], **cls, **m))
                if pname == "INNER_B_2023H2":
                    pd.DataFrame({"r": daily_returns(res)}).to_parquet(shard / f"daily_INNER_B_2023H2_s{s}.parquet")
        np.savez_compressed(shard / "frozen_calibrated_develop.npz", rows=rows, y=y, period=per,
                            **{f"s{s}": cal[s].astype(np.float32) for s in seeds})
    frozen.update(input_identity=cur_identity, eligibility_applied=elig is not None, stage="freeze", expert={"params": p.__dict__, "k_tp": k_tp, "fill_path": "EXECUTABLE_B_PLUS_1" if _EXEC["executable"]
                                          else "LEGACY_B_PLUS_2_WITH_REOPEN_GUARD", "max_orders_per_day": _EXEC["cap"]},
                  thresholds_attempted=list(THRESHOLDS), periods_read=list(DEVELOP_PERIODS), reads_2024=False,
                  rules={"cell": "lowest mean INNER_A loss, ties by gap", "threshold": "max mean INNER_A Sharpe over seeds among valid, else max",
                         "temperature": "per seed on INNER_A (log grid 0.25..8)"})
    frozen["freeze_sha256"] = canon(frozen)
    (out / "FROZEN.json").write_text(json.dumps(frozen, indent=1, default=float) + "\n")
    pd.DataFrame(scored).to_csv(out / "scores_develop.csv", index=False)
    pd.DataFrame(trials).to_csv(out / "trials_threshold_inner_a.csv", index=False)
    pd.DataFrame(baselines).to_csv(out / "classification_baselines.csv", index=False)
    uncertainty(out, ["INNER_B_2023H2"], ev)
    print(json.dumps({"freeze_sha256": frozen["freeze_sha256"], "shards": {k_: (v["cell"], v["threshold"]) for k_, v in frozen["shards"].items()}}))


# --------------------------------------------------------------------------------------- stage: reconfirm
def reconfirm(a):
    _set_threads()
    ev, out = Path(a.eval_input), Path(a.out)
    frozen = json.loads((out / "FROZEN.json").read_text())
    if canon({k: v for k, v in frozen.items() if k != "freeze_sha256"}) != frozen["freeze_sha256"]:
        raise SystemExit("FROZEN_DIGEST")
    check_frozen_inputs(frozen, ev)                       # before any data or model is loaded
    sets_doc, feats, targ, ts, elig = load_inputs(ev, include_2024=True)
    k = int(sets_doc["k_bars"])
    common, period, y_raw, sets, loss = common_rows(feats, targ, ts, sets_doc, k, elig)
    for key, fz in frozen["shards"].items():              # population and scaler re-derived and compared before loading
        z = np.load(out / "shards" / key / "develop_predictions.npz", allow_pickle=False)
        if hashlib.sha256(z["rows"].tobytes()).hexdigest() != fz["common_rows_sha256"]:
            raise SystemExit("POPULATION_DRIFT:" + key)
        rows_tr = np.flatnonzero(common & (period == "TRAIN_2020_2022"))
        mu, sd = scaler(feats, fz["columns"], rows_tr)
        if canon({"mean": mu.tolist(), "sd": sd.tolist(), "fit_on": fz["scaler"]["fit_on"]}) != fz["scaler_sha256"]:
            raise SystemExit("SCALER_DRIFT:" + key)
    ctx = make_ctx(sets_doc, a, ts)
    p, piv, k_tp = setup_expert(sets_doc, ctx, a.zigzag_run)
    if not np.array_equal(ctx.bars.index.as_unit("s").asi8, ts.as_unit("s").asi8):
        raise SystemExit("EVAL_INPUT_GRID_DIFFERS_FROM_LAKE")
    rows = np.flatnonzero(common & (period == "REOBSERVED_2024"))
    y = y_raw[rows].astype(np.int64)
    mask = np.ones(len(rows), bool)
    scored = []
    tenc = time_encoding(ts)
    baselines = []
    for key, fz in frozen["shards"].items():
        shard = out / "shards" / key
        Z = scaled(feats, fz["columns"], np.asarray(fz["scaler"]["mean"]), np.asarray(fz["scaler"]["sd"]))
        xa = arm_inputs(fz["arm"], Z, tenc, rows)
        probs = []
        for s, m in fz["models"].items():
            path = shard / "models" / m["file"]
            if sha(path) != m["sha256"]:
                raise SystemExit("FROZEN_MODEL_DIGEST:" + m["file"])
            model = load_model(path, fz["arm"])
            Pc = PD.calibrate(model.predict(xa, verbose=0, batch_size=4096), fz["temperatures"][s])
            met, res, sig = strategy(ctx, piv, Pc, rows, fz["threshold"], k, k_tp, "REOBSERVED_2024")
            scored.append(dict(set=fz["set"], arm=fz["arm"], cell=cell_id(fz["cell"]), seed=int(s), threshold=fz["threshold"],
                               period="REOBSERVED_2024", role="PREVIOUSLY_OBSERVED_RECONFIRMATION",
                               **per_direction(Pc, y, sig, rows, mask), **met))
            pd.DataFrame({"r": daily_returns(res)}).to_parquet(shard / f"daily_REOBSERVED_2024_s{s}.parquet")
            np.savez_compressed(shard / f"reobserved_2024_s{s}.npz", rows=rows, y=y, P=Pc.astype(np.float32))
            probs.append(Pc)
        if "train_prior" in fz:
            baselines.append(dict(set=fz["set"], arm=fz["arm"], period="REOBSERVED_2024",
                                  **classification_baselines(probs, y, np.asarray(fz["train_prior"]), mask)))
    pd.DataFrame(scored).to_csv(out / "scores_reobserved_2024.csv", index=False)
    if baselines:
        b0 = out / "classification_baselines.csv"
        prev = pd.read_csv(b0) if b0.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame(baselines)], ignore_index=True).to_csv(b0, index=False)
    (out / "RECONFIRM.json").write_text(json.dumps(dict(stage="reconfirm", freeze_sha256=frozen["freeze_sha256"],
                                                        rows=int(len(rows)), rows_lost_to_union_warmup={n: v["REOBSERVED_2024"] for n, v in loss.items()},
                                                        decisions_changed=False), indent=1) + "\n")
    uncertainty(out, ["REOBSERVED_2024"], ev)


# --------------------------------------------------------------------------------------- uncertainty
def uncertainty(out: Path, periods, eval_input):
    """Paired candidate - baseline differences per arm on identical rows, weekly block bootstrap: seed-mean
    T1 log-loss per row and seed-mean daily strategy return Sharpe.  Trial family declared alongside."""
    out = Path(out)
    frozen = json.loads((out / "FROZEN.json").read_text())
    sets = sorted({v["set"] for v in frozen["shards"].values()})
    arms = sorted({v["arm"] for v in frozen["shards"].values()})
    doc = json.loads((out / "UNCERTAINTY.json").read_text()) if (out / "UNCERTAINTY.json").exists() else {}
    base = "admitted_core_15_553fc047"
    for pname in periods:
        res_p = {}
        for arm in arms:
            for cand in [s for s in sets if s != base]:
                ll, rr = {}, {}
                for sname in (cand, base):
                    key = f"{sname}__{arm}"
                    fz = frozen["shards"].get(key)
                    if fz is None:
                        continue
                    shard = out / "shards" / key
                    probs, days = [], []
                    for s in fz["models"]:
                        if pname == "INNER_B_2023H2":
                            z = np.load(shard / "frozen_calibrated_develop.npz", allow_pickle=True)
                            m = z["period"].astype(str) == pname
                            rows_, y_, P_ = z["rows"][m], z["y"][m], z[f"s{s}"][m]
                        else:
                            z = np.load(shard / f"reobserved_2024_s{s}.npz")
                            rows_, y_, P_ = z["rows"], z["y"], z["P"]
                        probs.append(P_.astype(float))
                        days.append(pd.read_parquet(shard / f"daily_{pname}_s{s}.parquet")["r"])
                    Pm = np.mean(probs, axis=0)
                    ll[sname] = pd.Series(-np.log(np.clip(Pm[np.arange(len(y_)), y_], 1e-9, 1)), index=rows_)
                    rr[sname] = pd.concat(days, axis=1).mean(axis=1)
                if len(ll) < 2:
                    continue
                if not ll[cand].index.equals(ll[base].index):
                    raise SystemExit("ROWS_NOT_IDENTICAL")
                bars_idx = pd.read_parquet(Path(eval_input) / "features_all.parquet", columns=["timestamp"])["timestamp"]
                tix = pd.DatetimeIndex(bars_idx.iloc[ll[cand].index.to_numpy()])
                dl = pd.DataFrame({"d": (ll[cand] - ll[base]).to_numpy()}, index=tix)
                dr = pd.DataFrame({"c": rr[cand], "b": rr[base]}).dropna()
                res_p[f"{cand}_minus_{base}__{arm}"] = {
                    "rows": int(len(dl)), "mean_logloss_diff": float(dl["d"].mean()),
                    "logloss_diff_ci": block_bootstrap(lambda f: float(f["d"].mean()), dl),
                    "sharpe_candidate": sharpe365(dr["c"]), "sharpe_baseline": sharpe365(dr["b"]),
                    "sharpe_diff_ci": block_bootstrap(lambda f: sharpe365(f["c"]) - sharpe365(f["b"]), dr),
                    "note": "negative log-loss difference favours the candidate; seed-mean predictions and returns"}
        doc[pname] = res_p
    doc["trial_family"] = {"sets": sets, "arms": arms, "cells": {a: len(GRID[a]) for a in arms}, "seeds": len(SEEDS),
                           "thresholds": len(THRESHOLDS), "bootstrap": BOOT,
                           "selection_awareness": "subsets chosen on TRAIN only (never 2023/2024); cell, temperature and threshold "
                                                  "chosen on INNER_A; INNER_B and 2024 scored frozen; DSR supplement in SEAL"}
    (out / "UNCERTAINTY.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")


# --------------------------------------------------------------------------------------- stage: seal
def seal(a):
    out = Path(a.out)
    frozen = json.loads((out / "FROZEN.json").read_text())
    sets_doc = json.loads((Path(a.eval_input) / "FEATURE_SETS.json").read_text())   # target column / k (v5 family)
    dev = pd.read_csv(out / "scores_develop.csv")
    rec = pd.read_csv(out / "scores_reobserved_2024.csv") if (out / "scores_reobserved_2024.csv").exists() else pd.DataFrame()
    allsc = pd.concat([dev, rec], ignore_index=True)
    cols = [c for c in allsc.columns if c not in ("set", "arm", "cell", "seed", "threshold", "period", "role")
            and pd.api.types.is_numeric_dtype(allsc[c])]
    summ = allsc.groupby(["set", "arm", "cell", "threshold", "period", "role"])[cols].mean().reset_index()
    summ["seeds"] = allsc.groupby(["set", "arm", "cell", "threshold", "period", "role"]).seed.nunique().to_numpy()
    no_trade = [dict(set="NO_TRADE_COMPARATOR", arm="-", cell="-", threshold=np.nan, period=p, role="COMPARATOR",
                     sharpe_daily_sqrt365=0.0, net_profit_usd=0.0, entries_filled=0, max_drawdown_pct=0.0, valid=False)
                for p in EVAL_PERIODS]
    summ = pd.concat([summ, pd.DataFrame(no_trade)], ignore_index=True)
    summ.to_csv(out / "summary.csv", index=False)
    trials = pd.read_csv(out / "trials_threshold_inner_a.csv")
    dsr = {}
    for key, fz in frozen["shards"].items():
        for pname in ("INNER_B_2023H2", "REOBSERVED_2024"):
            files = sorted((out / "shards" / key).glob(f"daily_{pname}_s*.parquet"))
            if not files:
                continue
            r = pd.concat([pd.read_parquet(f)["r"] for f in files], axis=1).mean(axis=1).to_numpy()
            sr = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
            from scipy.stats import kurtosis, skew
            tsr = trials[(trials.set == fz["set"]) & (trials.arm == fz["arm"])].sharpe / math.sqrt(365)
            dsr[f"{key}__{pname}"] = deflated_sharpe(sr, len(r), float(skew(r)), float(kurtosis(r, fisher=False)), tsr)
    exp = frozen.get("expert") or {"params": json.loads((Path(a.zigzag_run) / "CHOICE.json").read_text())["params"]}
    trial_ledger = {"expert_configs": 1, "expert_params": exp["params"], "fill_path": exp.get("fill_path"), "k_tp": exp.get("k_tp"),
                    "k_bars": sets_doc.get("k_bars", 2),
                    "selection_targets": sets_doc.get("selection_targets") or ["timing_pivot_k2", "timing_valley_k2", "timing_peak_k2"],
                    "evaluation_target": (sets_doc.get("target_column") or "timing_pivot_k2") + " (T1, primary)", "feature_sets": sorted({v["set"] for v in frozen["shards"].values()}),
                    "arms": sorted({v["arm"] for v in frozen["shards"].values()}), "regularization_grid": GRID, "primary_cells": PRIMARY,
                    "seeds": list(SEEDS), "thresholds": list(THRESHOLDS),
                    "cell_choices": {k_: v["cell_choice"] for k_, v in frozen["shards"].items()},
                    "threshold_choices": {k_: v["threshold"] for k_, v in frozen["shards"].items()},
                    "threshold_trials_rows": int(len(trials))}
    bl = out / "classification_baselines.csv"
    seal_doc = dict(label="DEVELOPMENT_TIMING_PROXY_PREDICTIVE_VALIDATION", evaluator="heuristic-strategy tools/short_horizon_timing_eval.py",
                    comparison=("two feature sets under a shared recipe and context: identical eligible rows, grid, seeds, budget, monitor "
                                "and strategy; m1 also receives 4 shared origin time encodings (hour and weekday sin/cos of the bar start); "
                                "model capacity scales with the number of features, so this is not a matched-parameter comparison"),
                    classification_baselines=pd.read_csv(bl).to_dict(orient="records") if bl.exists() else None,
                    input_identity=frozen.get("input_identity"), eligibility_applied=frozen.get("eligibility_applied"),
                    proxy_only=True, periods=EVAL_PERIODS, roles={"TRAIN_2020_2022": "fit", "INNER_A_2023H1": "choices (joint development)",
                                                                  "INNER_B_2023H2": "frozen score (joint development)",
                                                                  "REOBSERVED_2024": "previously observed reconfirmation, never pristine"},
                    freeze_sha256=frozen["freeze_sha256"], sharpe_definition=SHARPE_DEFINITION, monitor=OwnerMonitor.__doc__,
                    training=TRAINING, trial_ledger=trial_ledger, deflated_sharpe_supplement=dsr,
                    uncertainty=json.loads((out / "UNCERTAINTY.json").read_text()), summary=summ.to_dict(orient="records"),
                    files={n: sha(out / n) for n in ("FROZEN.json", "scores_develop.csv", "summary.csv", "UNCERTAINTY.json")
                           + (("classification_baselines.csv",) if bl.exists() else ())
                           + (("scores_reobserved_2024.csv", "RECONFIRM.json") if (out / "RECONFIRM.json").exists() else ())},
                    test_read=False)
    (out / "SEAL.json").write_text(json.dumps(seal_doc, indent=1, default=float) + "\n")
    print(json.dumps({"seal_sha256": sha(out / "SEAL.json")}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("develop", "freeze", "reconfirm", "seal"))
    ap.add_argument("--eval-input", required=True, help="directory with features_all.parquet, targets_all.parquet, FEATURE_SETS.json")
    ap.add_argument("--data", required=False, help="lake EUR/USD H1 parquet (pinned digest; freeze/reconfirm)")
    ap.add_argument("--zigzag-run", default=str(ROOT / "docs/short_horizon/results/zigzag"))
    ap.add_argument("--closures", help="admitted CLOSURE_INTERVALS.json (freeze/reconfirm on the admitted source)")
    ap.add_argument("--out", required=True, help="evaluation directory (shards/, FROZEN.json, SEAL.json)")
    ap.add_argument("--set", help="develop: feature set name from FEATURE_SETS.json")
    ap.add_argument("--arm", choices=list(ARMS), help="develop: arm")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    a = ap.parse_args(argv)
    {"develop": develop, "freeze": freeze, "reconfirm": reconfirm, "seal": seal}[a.stage](a)


if __name__ == "__main__":
    main()

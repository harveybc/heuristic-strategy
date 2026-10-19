#!/usr/bin/env python3
"""Predictive validation of a timing feature selection through the pivot-detector harness (CPU, DEVELOPMENT).

For each named feature set in FEATURE_SETS.json (predictor ``tools/timing_selection.py`` writes it next to
``features_all.parquet`` / ``targets_all.parquet``) and each ARM, a causal classifier is trained on TRAIN
with the owner's early-stopping rule (monitor = 0.5 * (class-weighted running TRAIN loss + the same loss on
INNER 2023 purged), patience 10, min_delta 1e-8, best weights restored), temperature-calibrated on INNER,
scored by AUC / calibrated log-loss / precision-recall at the operating threshold, and by the trivial
reversal strategy's Sharpe through the sealed adapter (``tools.short_horizon_zigzag_study.Ctx.run``:
TP = k_tp * sigma, SL = 2 sigma, realistic costs, order/frequency rules).  The threshold is chosen on INNER
2023 (mean Sharpe over seeds, valid first); CONFIRM 2024 is reported once at that threshold.  Every set and
arm is scored on IDENTICAL rows (rows with a full finite 48-bar window over the union of all compared
columns, labelled and not purged).  TEST (2025+) is never read.

Arms (the pivot-detector experiment's three, plus a per-bar probe):
  m0_mlp        per-bar MLP(64,32) on the current bar's features
  m1_dmo_class  I6-DMO-class network: 13 slots x 2 h (t-24..t), per-feature causal Conv1D branches
                (grouped convolution = one branch per feature), weightless flatten fusion, origin-time
                encoding (hour/weekday sin-cos), Dense core, softmax head.  "Class" = same wiring idea as
                tools.i6dmo.model.build_dmo, not that module (which needs the sealed population).
  m2_gru        GRU(32) -> Dense(16) on the raw 48-bar hourly window
  m3_hgb        HistGradientBoostingClassifier on the flattened 48-bar window, monitored every 10
                iterations with the same two loss terms (TRAIN term from staged probabilities)
Scaling: TRAIN-only z-score per feature (declared; not the RL rolling z-score).

Targets: y in {0 none, 1 valley within (t, t+k], 2 peak within (t, t+k]} from the hindsight ZigZag expert
(labels only).  Rows whose label window crosses a split cut are purged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import short_horizon_pivot_detector as PD  # noqa: E402
from tools.short_horizon_zigzag_study import Ctx, params_of  # noqa: E402

PERIODS = ("DECISION_2020_2022", "VALIDATION_2023", "CONFIRM_2024")
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
ARMS = ("m0_mlp", "m1_dmo_class", "m2_gru", "m3_hgb")
WINDOW, SLOTS, STRIDE = 48, 13, 2
TRAINING = dict(batch_size=256, learning_rate=1e-3, weight_decay=1e-4, patience=10, min_delta=1e-8, max_epochs=200,
                hgb_max_iter=400, hgb_eval_every=10, hgb_learning_rate=0.05, hgb_max_leaf_nodes=31)
HOUR = 3600


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc(s):
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# --------------------------------------------------------------------------------------- data
def design_matrix(features: pd.DataFrame, cols: list[str], lags: int) -> np.ndarray:
    """Row t = [x_t, x_{t-1}, ..., x_{t-lags}] of the named columns (causal: rows <= t only)."""
    x = features[cols].to_numpy(float)
    parts = [x]
    for j in range(1, lags + 1):
        parts.append(np.vstack([np.full((j, x.shape[1]), np.nan), x[:-j]]))
    return np.hstack(parts)


def windows(z: np.ndarray, rows: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """(len(rows), window, F): bars t-window+1..t for each row t (rows must be >= window-1)."""
    idx = rows[:, None] + np.arange(-window + 1, 1)[None, :]
    return z[idx]


def split_rows(ts: pd.DatetimeIndex, splits: dict, k: int):
    period = np.full(len(ts), "", object)
    cuts = []
    for name, key in (("DECISION_2020_2022", "train"), ("VALIDATION_2023", "inner"), ("CONFIRM_2024", "confirm")):
        a, b = _utc(splits[key][0]), _utc(splits[key][1])
        period[(ts >= a) & (ts < b)] = name
        cuts.append(b)
    purged = np.zeros(len(ts), bool)
    for c in cuts:
        purged |= (ts >= c - pd.Timedelta(hours=k)) & (ts < c)
    return period, purged


def time_encoding(ts: pd.DatetimeIndex) -> np.ndarray:
    h, d = np.asarray(ts.hour, float), np.asarray(ts.dayofweek, float)
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)]).astype(np.float32)


# --------------------------------------------------------------------------------------- owner's monitor
class _OwnerMonitor:
    """monitor = 0.5 * (running TRAIN weighted loss + INNER weighted log-loss); patience / min_delta / best-restore
    semantics as in tools/short_horizon_pivot_detector_train.py (paired-mean form of agent-multi's PairedMonitor)."""

    def __init__(self, keras, x_inner, y_inner, w_inner, patience, min_delta):
        self.x_inner, self.y_inner, self.w_inner = x_inner, y_inner, w_inner
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.wait, self.best_weights, self.history = np.inf, -1, 0, None, []

    def weighted_loss(self, p):
        p = np.clip(p, 1e-9, 1)
        ll = -np.log(p[np.arange(len(self.y_inner)), self.y_inner])
        return float(np.sum(self.w_inner * ll) / np.sum(self.w_inner))

    def inner_loss(self, model):
        return self.weighted_loss(model.predict(self.x_inner, verbose=0, batch_size=4096))

    def step(self, epoch, train, inner, weights_getter):
        """Returns True when patience is exhausted."""
        score = 0.5 * (train + inner)
        self.history.append(dict(epoch=epoch, train_running_loss=train, inner_weighted_loss=inner, score=score, loss_gap=inner - train))
        if score < self.best - self.min_delta:
            self.best, self.best_epoch, self.wait, self.best_weights = score, epoch, 0, weights_getter()
            return False
        self.wait += 1
        return self.wait >= self.patience

    def on_epoch_end(self, epoch, logs, model):
        if self.step(epoch, float(logs["loss"]), self.inner_loss(model), model.get_weights):
            model.stop_training = True


def _compile(keras, model):
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"]),
                  loss="sparse_categorical_crossentropy")
    return model


def _fit_keras(keras, model, x_tr, y_tr, x_in, y_in, cw):
    mon = _OwnerMonitor(keras, x_in, y_in, cw[y_in], TRAINING["patience"], TRAINING["min_delta"])

    class _CB(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mon.on_epoch_end(epoch, logs or {}, self.model)

    model.fit(x_tr, y_tr, sample_weight=cw[y_tr], batch_size=TRAINING["batch_size"], epochs=TRAINING["max_epochs"],
              verbose=0, shuffle=True, callbacks=[_CB()])
    if mon.best_weights is not None:
        model.set_weights(mon.best_weights)
    return model, dict(best_epoch=mon.best_epoch, epochs=len(mon.history), history=mon.history, class_weights=cw.tolist(),
                       parameters=int(model.count_params()))


def class_weights(y_tr):
    counts = np.bincount(y_tr, minlength=3).astype(float)
    return (1.0 / np.maximum(counts, 1)) * counts.sum() / 3.0


def fit_mlp(x_tr, y_tr, x_in, y_in, seed: int, hidden=(64, 32)):
    import tensorflow as tf
    keras = tf.keras
    keras.utils.set_random_seed(int(seed))
    model = keras.Sequential([keras.Input(shape=(x_tr.shape[1],)), *[keras.layers.Dense(h, activation="relu") for h in hidden],
                              keras.layers.Dense(3, activation="softmax")])
    return _fit_keras(keras, _compile(keras, model), x_tr, y_tr, x_in, y_in, class_weights(y_tr))


def fit_dmo_class(x_tr, y_tr, x_in, y_in, seed: int, filters=4, core=32):
    """x = [slots (n,13,F), time (n,4)]; per-feature causal Conv1D branches = grouped Conv1D with F groups."""
    import tensorflow as tf
    keras = tf.keras
    keras.utils.set_random_seed(int(seed))
    slots, tenc = x_tr
    F = slots.shape[2]
    inp = keras.Input(shape=(slots.shape[1], F))
    tin = keras.Input(shape=(tenc.shape[1],))
    b = keras.layers.Conv1D(filters * F, 3, padding="causal", groups=F, activation="relu")(inp)
    b = keras.layers.Conv1D(filters * F, 3, padding="causal", groups=F, activation="relu")(b)
    fused = keras.layers.Concatenate()([keras.layers.Flatten()(b), tin])
    h = keras.layers.Dense(core, activation="relu")(fused)
    out = keras.layers.Dense(3, activation="softmax")(h)
    model = _compile(keras, keras.Model([inp, tin], out))
    return _fit_keras(keras, model, list(x_tr), y_tr, list(x_in), y_in, class_weights(y_tr))


def fit_gru(x_tr, y_tr, x_in, y_in, seed: int):
    import tensorflow as tf
    keras = tf.keras
    keras.utils.set_random_seed(int(seed))
    model = keras.Sequential([keras.Input(shape=x_tr.shape[1:]), keras.layers.GRU(32), keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(3, activation="softmax")])
    return _fit_keras(keras, _compile(keras, model), x_tr, y_tr, x_in, y_in, class_weights(y_tr))


class _StagedHGB:
    def __init__(self, clf, n_iter):
        self.clf, self.n_iter = clf, n_iter

    def predict(self, x, verbose=0, batch_size=0):
        for i, p in enumerate(self.clf.staged_predict_proba(x), start=1):
            if i == self.n_iter:
                return p
        return p


def fit_hgb(x_tr, y_tr, x_in, y_in, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cw = class_weights(y_tr)
    clf = HistGradientBoostingClassifier(learning_rate=TRAINING["hgb_learning_rate"], max_leaf_nodes=TRAINING["hgb_max_leaf_nodes"],
                                         max_iter=TRAINING["hgb_max_iter"], early_stopping=False, random_state=int(seed))
    clf.fit(x_tr, y_tr, sample_weight=cw[y_tr])
    mon = _OwnerMonitor(None, x_in, y_in, cw[y_in], TRAINING["patience"], TRAINING["min_delta"])
    w_tr = cw[y_tr]
    stopped = None
    every = TRAINING["hgb_eval_every"]
    for i, (p_tr, p_in) in enumerate(zip(clf.staged_predict_proba(x_tr), clf.staged_predict_proba(x_in)), start=1):
        if i % every:
            continue
        p_tr = np.clip(p_tr, 1e-9, 1)
        train = float(np.sum(w_tr * -np.log(p_tr[np.arange(len(y_tr)), y_tr])) / w_tr.sum())
        if mon.step(i, train, mon.weighted_loss(p_in), lambda i=i: i):
            stopped = i
            break
    best = mon.best_weights or TRAINING["hgb_max_iter"]
    return _StagedHGB(clf, best), dict(best_iteration=best, stopped_at=stopped, history=mon.history, class_weights=cw.tolist())


# --------------------------------------------------------------------------------------- evaluation
def arm_inputs(arm, Z, tenc, rows, lags):
    if arm == "m0_mlp":
        return Z[rows] if not lags else np.hstack([Z[rows - j] for j in range(0, lags + 1)])
    W = windows(Z, rows)
    if arm == "m1_dmo_class":
        return [W[:, WINDOW - 1 - STRIDE * (SLOTS - 1):: STRIDE, :], tenc[rows]]
    if arm == "m2_gru":
        return W
    return W.reshape(len(rows), -1)


FITTERS = {"m0_mlp": fit_mlp, "m1_dmo_class": fit_dmo_class, "m2_gru": fit_gru, "m3_hgb": fit_hgb}


def evaluate_set(name, cols, arms, feats, ts, y_all, period, ctx, piv, k, k_tp, seeds, lags, common):
    X = feats[cols].to_numpy(float)
    keep = common.copy()
    rows = np.flatnonzero(keep)
    tr = period[rows] == "DECISION_2020_2022"
    mu, sd = X[rows[tr]].mean(0), X[rows[tr]].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = ((X - mu) / sd).astype(np.float32)
    Z[~np.isfinite(Z)] = 0.0                                   # outside the common rows only (never inside a used window)
    tenc = time_encoding(ts)
    y = y_all[rows]
    per = period[rows]
    inn = per == "VALIDATION_2023"
    base = np.bincount(y[tr], minlength=3) / tr.sum()
    out_rows, fits, preds = [], {}, {}
    for arm in arms:
        xa = arm_inputs(arm, Z, tenc, rows, lags)
        sel = (lambda m: [v[m] for v in xa]) if isinstance(xa, list) else (lambda m: xa[m])
        for seed in seeds:
            t0 = time.time()
            model, fit = FITTERS[arm](sel(tr), y[tr], sel(inn), y[inn], seed)
            fit["wall_seconds"] = time.time() - t0
            fits[f"{name}:{arm}:s{seed}"] = fit
            P = model.predict(xa, verbose=0, batch_size=4096)
            T = PD.temperature(P, y, inn)
            Pc = PD.calibrate(P, T)
            preds[(name, arm, seed)] = dict(origin_bar=rows, y=y, period=per, P=Pc, T=T)
            for tau in THRESHOLDS:
                sig = PD.signals_from(Pc, rows, tau, k)
                for pname in PERIODS:
                    mm = per == pname
                    m, _ = ctx.run(sig, piv, pname, mode="causal", k_tp=k_tp)
                    prec, rec, n_sig = PD.prec_recall(sig, y, rows, mm)
                    m = {kk: v for kk, v in m.items() if kk != "period"}
                    out_rows.append(dict(set=name, arm=arm, seed=seed, temperature=T, threshold=tau, period=pname, signals=n_sig,
                                         precision_at_threshold=prec, recall_at_threshold=rec, **PD.metrics(Pc, y, mm, base), **m))
    return out_rows, fits, preds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-input", required=True, help="directory with features_all.parquet, targets_all.parquet, FEATURE_SETS.json")
    ap.add_argument("--data", required=True, help="lake EUR/USD H1 parquet (pinned digest)")
    ap.add_argument("--zigzag-run", default=str(ROOT / "docs/short_horizon/results/zigzag"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="*", default=None, help="subset of set names (default: all)")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[6102026, 6102027, 6102028])
    ap.add_argument("--lags", type=int, default=0, help="m0 only: lagged bars appended to the per-bar vector")
    a = ap.parse_args(argv)
    ev = Path(a.eval_input)
    sets_doc = json.loads((ev / "FEATURE_SETS.json").read_text())
    feats = pd.read_parquet(ev / "features_all.parquet")
    targ = pd.read_parquet(ev / "targets_all.parquet")
    if sha(ev / "features_all.parquet") != sets_doc["features_sha256"] or sha(ev / "targets_all.parquet") != sets_doc["targets_sha256"]:
        raise SystemExit("EVAL_INPUT_DIGEST")
    k = int(sets_doc["k_bars"])
    ts = pd.DatetimeIndex(feats["timestamp"])
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    if (ts >= pd.Timestamp("2025-01-01", tz="UTC")).any():
        raise SystemExit("TEST_ROWS_FORBIDDEN")
    y_raw = targ[f"timing_pivot_k{k}"].to_numpy(float)
    period, purged = split_rows(ts, sets_doc["splits"], k)
    ctx = Ctx(a.data)
    p = params_of(json.loads((Path(a.zigzag_run) / "CHOICE.json").read_text())["params"])
    piv = ctx.pivots(p)
    k_tp = float(pd.read_csv(Path(a.zigzag_run) / "confirmed.csv").query("chosen").k_tp.iloc[0])
    lake_ts = ctx.bars.index.as_unit("s").asi8
    if not np.array_equal(lake_ts, ts.as_unit("s").asi8):
        raise SystemExit("EVAL_INPUT_GRID_DIFFERS_FROM_LAKE")
    sets = {n: c for n, c in sets_doc["sets"].items() if (a.sets is None or n in a.sets)}
    missing = {n: [c for c in cols if c not in feats.columns] for n, cols in sets.items()}
    sets = {n: cols for n, cols in sets.items() if cols and not missing[n]}
    # identical rows for every set and arm: full finite 48-bar window over the union of all compared columns,
    # labelled, not purged, inside a declared period
    all_cols = sorted({c for cols in sets.values() for c in cols})
    fin = np.isfinite(feats[all_cols].to_numpy(float)).all(axis=1)
    win_ok = pd.Series(fin).rolling(WINDOW).min().fillna(0).to_numpy().astype(bool)
    common = win_ok & np.isfinite(y_raw) & ~purged & (period != "")
    y_all = np.where(np.isfinite(y_raw), y_raw, 0).astype(np.int64)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, fits, preds = [], {}, {}
    for name, cols in sets.items():
        r, f, pr = evaluate_set(name, cols, a.arms, feats, ts, y_all, period, ctx, piv, k, k_tp, a.seeds, a.lags, common)
        rows += r
        fits.update(f)
        preds.update(pr)
        pd.DataFrame(rows).to_csv(out / "timing_eval.csv", index=False)      # partial progress is durable
    det = pd.DataFrame(rows)
    det.to_csv(out / "timing_eval.csv", index=False)
    choice, summary = {}, []
    for name in sets:
        for arm in a.arms:
            v = det[(det.set == name) & (det.arm == arm) & (det.period == "VALIDATION_2023")]
            g = v.groupby("threshold").agg(sharpe=("sharpe_daily_sqrt365", "mean"), valid=("valid", "min")).reset_index()
            gv = g[g.valid.astype(bool)]
            tau = float((gv if not gv.empty else g).sort_values("sharpe", ascending=False).threshold.iloc[0])
            choice[f"{name}:{arm}"] = tau
            for pname in PERIODS:
                d = det[(det.set == name) & (det.arm == arm) & (det.threshold == tau) & (det.period == pname)]
                summary.append(dict(set=name, arm=arm, n_features=len(sets[name]), threshold=tau, period=pname, seeds=len(d),
                                    rows=int((common & (period == pname)).sum()),
                                    **{f"{c}_mean": float(d[c].mean()) for c in ("sharpe_daily_sqrt365", "max_drawdown_pct", "trades_per_week",
                                                                                  "precision_at_threshold", "recall_at_threshold", "auc_valley",
                                                                                  "auc_peak", "log_loss", "log_loss_base_rate", "win_rate")},
                                    sharpe_sd=float(d.sharpe_daily_sqrt365.std(ddof=0)), valid_all=bool(d.valid.all())))
    summ = pd.DataFrame(summary)
    summ.to_csv(out / "timing_eval_summary.csv", index=False)
    frames = [pd.DataFrame(dict(set=n, arm=ar, seed=s, origin_ts=ts.as_unit("s").asi8[d["origin_bar"]], period=d["period"], y=d["y"],
                                p_none=d["P"][:, 0], p_valley=d["P"][:, 1], p_peak=d["P"][:, 2])) for (n, ar, s), d in preds.items()]
    pred_path = out / "timing_eval_predictions.csv.gz"
    pd.concat(frames).to_csv(pred_path, index=False, float_format="%.6f")
    (out / "RUN.json").write_text(json.dumps(dict(fits=fits, training=TRAINING, lags=a.lags, seeds=a.seeds, arms=a.arms), indent=1, default=float) + "\n")
    seal = dict(label="DEVELOPMENT_TIMING_SELECTION_PREDICTIVE_VALIDATION", evaluator="heuristic-strategy tools/short_horizon_timing_eval.py",
                arms={ar: ARMS_DOC[ar] for ar in a.arms}, scaling="TRAIN-only z-score per feature",
                early_stopping="monitor = 0.5*(running TRAIN weighted loss + INNER 2023 weighted log-loss), patience 10, min_delta 1e-8, best restored; m3 every 10 iterations",
                chosen_on="VALIDATION_2023 (threshold by mean Sharpe over seeds, valid first)", confirm="CONFIRM_2024 once",
                identical_rows={pn: int((common & (period == pn)).sum()) for pn in PERIODS}, sets={n: cols for n, cols in sets.items()},
                missing_columns=missing, operating_threshold=choice, k_bars=k, k_tp=k_tp, m_sl=2.0, lake_sha256=ctx.digest,
                expert_params=p.__dict__, features_sha256=sets_doc["features_sha256"], targets_sha256=sets_doc["targets_sha256"],
                predictions_sha256=sha(pred_path), files={n: sha(out / n) for n in ("timing_eval.csv", "timing_eval_summary.csv", "RUN.json")},
                summary=summary, test_read=False)
    (out / "SEAL.json").write_text(json.dumps(seal, indent=1, default=float) + "\n")
    print(summ[summ.period != "DECISION_2020_2022"][["set", "arm", "n_features", "threshold", "period", "auc_valley_mean", "auc_peak_mean",
                                                         "log_loss_mean", "log_loss_base_rate_mean", "precision_at_threshold_mean",
                                                         "sharpe_daily_sqrt365_mean", "trades_per_week_mean", "valid_all"]].to_string(index=False))


ARMS_DOC = {
    "m0_mlp": "per-bar MLP(64,32) on the current bar's features",
    "m1_dmo_class": "13 slots x 2 h window, per-feature causal Conv1D branches (grouped conv, 4 filters, 2 layers), flatten fusion + origin time encoding, Dense(32) core, softmax",
    "m2_gru": "GRU(32) -> Dense(16) -> softmax on the raw 48-bar hourly window",
    "m3_hgb": "HistGradientBoostingClassifier (lr 0.05, 31 leaves, <= 400 iterations) on the flattened 48-bar window, owner's monitor every 10 iterations",
}


if __name__ == "__main__":
    main()

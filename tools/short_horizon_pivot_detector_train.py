#!/usr/bin/env python3
"""Pivot-detector GPU stage (DEVELOPMENT): trains m1/m2/m3 on the two input arms and writes sealed
per-origin probabilities. Runs INSIDE a staged predictor tree (PYTHONPATH = that tree) on the admitted
GPU role; never reads TEST (the population and INPUT.npz end 2024-12-31; the labels file is DEVELOPMENT).

Arms (same 19 features, same FIT-only z-score of the sealed population):
  A  13-slot x 2 h market-time windows of the sealed MG2m population (x, offsets_hours, closure_crossed)
  B  raw 48-bar hourly window of INPUT.npz rows with ts <= origin (causal availability)
Targets: y in {0 none, 1 valley within (t, t+k], 2 peak within (t, t+k]} from the hindsight ZigZag labels
file (built on the coordinator from the lake H1 file), joined by origin timestamp; rows whose label
window crosses a split cut are purged.
Models: m1 = I6-DMO recipe adapted (per-feature causal Conv1D branches, weightless fusion, physical-time
encoder, core; softmax head), m2 = small GRU on arm B, m3 = HistGradientBoosting on the flattened arm B.
Recipe (m1, m2): AdamW 1e-3 / 1e-4, batch 256, class-weighted CE, max 200 epochs, 3 seeds.
Early stopping (owner's corrected formula; the paired-mean form mirrors agent-multi
agent_plugins/paired_early_stopping.py@e119e8ad): (a) TRAIN term = the class-weighted running loss that
training itself produces at the end of each epoch (no re-evaluation of TRAIN in inference mode);
(b) INNER term = the same class-weighted log-loss on INNER 2023 (purged), evaluated at the end of each
epoch; monitor = 0.5 * (a + b), patience 10, min_delta 1e-8, best weights restored. As cheap diagnostics
(NOT part of the monitor) each epoch also records, on INNER only, the trivial reversal strategy's Sharpe
(TP 2 sigma, SL 2 sigma, realistic costs, order/frequency rules), trades/week, log-loss and AUC.
2024 is a single confirmation: never evaluated here, never used for stopping or selection.
m3 (boosting) is monitored every 10 iterations on its staged probabilities with the same two loss terms.
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

from tools.i6dmo import model as M  # noqa: E402  (determinism flags BEFORE tensorflow)
import tensorflow as tf  # noqa: E402
from tools.i6dmo import design as D, train as T  # noqa: E402

keras = tf.keras
HS = Path(os.environ.get("PIVOT_HS_DIR", "/var/tmp/pivot-detector-20261019/hs"))   # shipped heuristic-strategy code + lake
sys.path.insert(0, str(HS))                       # hs/app (the strategy package) resolves first; tools.i6dmo never imports app


def _load_hs(name):
    """The staged predictor tree owns the regular package ``tools``; the strategy's tool modules are loaded by
    path and registered under their own ``tools.`` names so their mutual imports resolve."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tools." + name, HS / "tools" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tools." + name] = mod
    spec.loader.exec_module(mod)
    return mod


_load_hs("short_horizon_oracle")
_STUDY = _load_hs("short_horizon_zigzag_study")
Ctx, params_of = _STUDY.Ctx, _STUDY.params_of

MONITOR_TAU = 0.5
MONITOR_PERIODS = {"SELECT_2023": "VALIDATION_2023"}
K_BARS = 2
TRAINING = dict(batch_size=256, learning_rate=0.001, weight_decay=0.0001, patience=10, min_delta=1e-8, max_epochs=200)
CUTS = {"TRAIN_2020_2022": ("2020-01-01", "2023-01-01"), "SELECT_2023": ("2023-01-01", "2024-01-01"),
        "CONFIRM_2024": ("2024-01-01", "2025-01-01")}
HOUR = 3600


def epoch(s):
    from datetime import datetime, timezone
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cgroup_peak():
    try:
        line = next(l for l in Path("/proc/self/cgroup").read_text().splitlines() if l.startswith("0::"))
        return int(Path("/sys/fs/cgroup" + line[3:].strip(), "memory.peak").read_text().strip())
    except (OSError, StopIteration, ValueError):
        return None


# ------------------------------------------------------------------------------------------- data
def load_arms(args):
    design = D.load_design(args.design)
    effective = D.effective_config(design)
    features = list(effective["feature_manifest"]["features"])
    pop = T.load_matched_population(args.population, args.population_digest, effective["horizons_hours"], features,
                                    input_dir=args.input_dir)
    if list(pop.scaler.feature_names) != features:
        raise SystemExit("SCALER_FEATURE_ORDER")
    mean, std = np.asarray(pop.scaler.mean, float), np.asarray(pop.scaler.std, float)
    parts = []
    for split in ("FIT", "INNER", "DEVELOPMENT"):
        t = T.arm_split(pop, effective["design"], split)
        parts.append(dict(split=split, origin_ts=np.asarray(t.row_id, np.int64), x=np.asarray(t.x, float),
                          offsets=np.asarray(t.offsets_hours, np.float32), closure=np.asarray(t.closure_crossed, np.float32)))
    origin_ts = np.concatenate([p["origin_ts"] for p in parts])
    order = np.argsort(origin_ts, kind="stable")
    xa = ((np.concatenate([p["x"] for p in parts]) - mean) / std).astype(np.float32)[order]
    offsets = np.concatenate([p["offsets"] for p in parts])[order]
    closure = np.concatenate([p["closure"] for p in parts])[order]
    pop_split = np.concatenate([[p["split"]] * len(p["origin_ts"]) for p in parts])[order]
    origin_ts = origin_ts[order]
    if len(np.unique(origin_ts)) != len(origin_ts):
        raise SystemExit("DUPLICATE_ORIGINS")
    # arm B: 48 hourly rows with ts <= origin, from the verified INPUT.npz (same features, same scaler)
    with np.load(Path(args.input_dir) / "INPUT.npz", allow_pickle=False) as z:
        ts, xin = np.asarray(z["ts"], np.int64), np.asarray(z["x"], float)
    if xin.shape[1] != len(features):
        raise SystemExit("INPUT_FEATURE_COUNT")
    idx = np.searchsorted(ts, origin_ts, side="right") - 1
    ok_b = (idx >= 47) & (ts[np.clip(idx, 0, len(ts) - 1)] == origin_ts)
    win = np.stack([np.arange(i - 47, i + 1) for i in np.clip(idx, 47, len(ts) - 1)])
    xb = ((xin[win] - mean) / std).astype(np.float32)
    finite_b = np.isfinite(xb).all(axis=(1, 2))
    ok_b &= finite_b                                   # a window with a non-finite raw value is not an arm-B row
    # labels (hindsight ZigZag, DEVELOPMENT) joined by origin timestamp; purge windows crossing a cut
    with np.load(args.labels, allow_pickle=False) as z:
        bar_ts, y_bar, valid_bar = np.asarray(z["bar_ts"], np.int64), np.asarray(z["y"], np.int64), np.asarray(z["valid"], bool)
    pos = {int(t): i for i, t in enumerate(bar_ts)}
    li = np.array([pos.get(int(t), -1) for t in origin_ts])
    ok_l = li >= 0
    y = np.where(ok_l, y_bar[np.clip(li, 0, len(y_bar) - 1)], -1)
    ok_l &= np.where(ok_l, valid_bar[np.clip(li, 0, len(valid_bar) - 1)], False)
    cuts = [epoch(b) for _, b in CUTS.values()]
    purged = np.zeros(len(origin_ts), bool)
    for c in cuts:
        purged |= (origin_ts >= c - K_BARS * HOUR) & (origin_ts < c)
    period = np.full(len(origin_ts), "", object)
    for name, (a, b) in CUTS.items():
        period[(origin_ts >= epoch(a)) & (origin_ts < epoch(b))] = name
    keep = ok_b & ok_l & ~purged & (period != "")
    sel = np.flatnonzero(keep)
    meta = dict(features=features, population_digest=pop.digest, scaler_feature_names=list(pop.scaler.feature_names),
                scaler_n_rows=int(pop.scaler.n_rows), scaler_fit_row_digest=pop.scaler.fit_row_digest,
                input_sha256=sha(Path(args.input_dir) / "INPUT.npz"), labels_sha256=sha(args.labels),
                design_seal_sha256=design.get("seal_sha256"), origins_total=int(len(origin_ts)),
                dropped=dict(no_arm_b=int((~ok_b).sum()), arm_b_nonfinite=int((~finite_b).sum()), no_label=int((~ok_l).sum()),
                             purged_at_cut=int(purged.sum())),
                kept=int(len(sel)), pop_split_counts={s: int((pop_split[sel] == s).sum()) for s in ("FIT", "INNER", "DEVELOPMENT")},
                period_counts={p: int((period[sel] == p).sum()) for p in CUTS},
                class_counts={p: np.bincount(y[sel][period[sel] == p], minlength=3).tolist() for p in CUTS},
                k_bars=K_BARS, model_config_sha256=hashlib.sha256(json.dumps(effective["model"], sort_keys=True).encode()).hexdigest())
    return dict(origin_ts=origin_ts[sel], xa=xa[sel], offsets=offsets[sel], closure=closure[sel], xb=xb[sel], y=y[sel],
                period=period[sel], pop_split=pop_split[sel], effective=effective, meta=meta)


# ------------------------------------------------------------------------------------------- models
def m1_classifier(effective, seed):
    """The I6-DMO network wired exactly as model.build_dmo (per-feature causal Conv1D branches via the registered
    dmo_branch component, weightless sequence_concat fusion, DynamicPhysicalTimeEncoder, dmo_core) with a
    softmax pivot head in place of the regression heads; same seeding (set_random_seed before the build)."""
    from predictor_plugins.modular_temporal import FeatureSelect
    c = M.normalize_model_config(dict(effective["model"], seed=int(seed)))
    M.enable_determinism()
    keras.utils.set_random_seed(c["seed"])
    steps, n = c["slots"], len(c["features"])
    grid = tuple(range(steps))
    inputs = keras.Input((steps, n), name="observations")
    time_inputs = [keras.Input((steps,), name="offsets_hours"), keras.Input((steps,), name="closure_crossed")]
    sequences = []
    for i, feature in enumerate(c["features"]):
        name = f"branch_{i:02d}"
        comp = M.dmo_branch(input_shape=(steps, 1), time_grid=grid, output_steps=steps, name=name, params=M._branch_params(c))
        model = M._validate_component(comp, [(steps, 1)], grid, c["branch"]["filters"])
        sequences.append(model(FeatureSelect([i], name="select_" + name)(inputs)))
    factory, _ = M._resolve("fusion", {"plugin": "sequence_concat"}, {"fusion": "modular.fusion"})
    shapes = [tuple(s.shape[1:]) for s in sequences]
    fusion = M._validate_component(factory(input_shapes=shapes, time_grid=grid, name="sequence_fusion", params={}), shapes, grid,
                                   sum(s[1] for s in shapes))
    fused = fusion(sequences)
    encoded = M.DynamicPhysicalTimeEncoder(min_adjacent_hours=float(c["market_time"]["spacing_open_market_hours"]),
                                           name="physical_time_encoder")([fused, *time_inputs])
    core_shape = tuple(encoded.shape[1:])
    core = M._validate_component(M.dmo_core(input_shape=core_shape, time_grid=grid, name="temporal_core", params=M._core_params(c)),
                                 [core_shape], grid, c["core"]["filters"][-1])
    latent = core(encoded)
    out = keras.layers.Dense(3, activation="softmax", name="pivot_head")(keras.layers.Flatten()(latent))
    model = keras.Model([inputs, *time_inputs], out, name="pivot_m1")
    return model, int(model.count_params())


def m2_classifier(n_features, seed):
    keras.utils.set_random_seed(int(seed))
    inp = keras.Input((48, n_features), name="window48")
    h = keras.layers.GRU(32, name="gru")(inp)
    h = keras.layers.Dense(16, activation="relu")(h)
    out = keras.layers.Dense(3, activation="softmax", name="pivot_head")(h)
    return keras.Model(inp, out, name="pivot_m2")


def class_weights(y):
    counts = np.bincount(y, minlength=3).astype(float)
    w = counts.sum() / (3.0 * np.maximum(counts, 1))
    return w / w.mean()


class StrategyMonitor:
    """Owner's monitor: strategy Sharpe (inference) on TRAIN and INNER per epoch; score = -0.5*(S_a+S_b)."""

    def __init__(self, data, lake):
        self.ctx = Ctx(str(lake))
        self.piv = self.ctx.pivots(params_of(json.loads((HS / "docs/short_horizon/results/zigzag/CHOICE.json").read_text())["params"]))
        import pandas as pd
        self.k_tp = float(pd.read_csv(HS / "docs/short_horizon/results/zigzag/confirmed.csv").query("chosen").k_tp.iloc[0])
        bar_index = {int(t): i for i, t in enumerate(self.ctx.bars.index.as_unit("s").asi8)}
        self.origin_bar = np.array([bar_index.get(int(t), -1) for t in data["origin_ts"]])
        if (self.origin_bar < 0).any():
            raise SystemExit("ORIGIN_NOT_IN_LAKE")
        self.y, self.period = data["y"], data["period"]
        self.masks = {p: self.period == p for p in MONITOR_PERIODS}

    def signals(self, P, mask):
        sig, last = [], -10 ** 9
        order = np.argsort(self.origin_bar)
        for j in order:
            if not mask[j]:
                continue
            t = int(self.origin_bar[j])
            if t - last < K_BARS:
                continue
            if P[j, 1] >= MONITOR_TAU and P[j, 1] >= P[j, 2]:
                sig.append((t, 1)); last = t
            elif P[j, 2] >= MONITOR_TAU and P[j, 2] > P[j, 1]:
                sig.append((t, -1)); last = t
        return sig

    def inner_loss(self, P, w):
        """Class-weighted log-loss on INNER 2023 (the same loss training minimizes) — the INNER monitor term."""
        from sklearn.metrics import log_loss
        m = self.masks["SELECT_2023"]
        return float(log_loss(self.y[m], P[m], labels=[0, 1, 2], sample_weight=w[self.y[m]]))

    def inner_diagnostics(self, P):
        """Cheap INNER-only diagnostics, NOT part of the monitor: strategy Sharpe, trades/week, AUC, plain log-loss."""
        from sklearn.metrics import log_loss, roc_auc_score
        m = self.masks["SELECT_2023"]
        res, _ = self.ctx.run(self.signals(P, m), self.piv, MONITOR_PERIODS["SELECT_2023"], mode="causal", k_tp=self.k_tp)
        yy = self.y[m]
        return dict(inner_sharpe=float(res["sharpe_daily_sqrt365"]), inner_trades_per_week=float(res["trades_per_week"]),
                    inner_valid=bool(res["valid"]), inner_log_loss=float(log_loss(yy, P[m], labels=[0, 1, 2])),
                    inner_auc_valley=float(roc_auc_score(yy == 1, P[m, 1])), inner_auc_peak=float(roc_auc_score(yy == 2, P[m, 2])))

    def score(self, train_running_loss, P, w):
        b = self.inner_loss(P, w)
        a = float(train_running_loss)
        return dict(score=0.5 * (a + b) if np.isfinite(a) and np.isfinite(b) else np.inf, train_running_loss=a,
                    inner_weighted_loss=b, loss_gap=b - a)


def fit_classifier(model, X_tr, y_tr, X_all, monitor, seed, max_epochs, progress):
    """AdamW recipe; owner's early stop on -0.5*(Sharpe_TRAIN_eval + Sharpe_INNER), patience 10, min_delta 1e-8."""
    keras.utils.set_random_seed(int(seed))
    w = class_weights(y_tr)
    sw_tr = w[y_tr].astype(np.float32)
    opt = keras.optimizers.AdamW(learning_rate=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", weighted_metrics=[])
    best, best_epoch, best_w, stale, hist, stop = np.inf, 0, None, 0, [], "max_epochs"
    t0 = time.monotonic()
    for ep in range(1, max_epochs + 1):
        h = model.fit(X_tr, y_tr, sample_weight=sw_tr, batch_size=TRAINING["batch_size"], epochs=1, shuffle=True, verbose=0)
        P = model.predict(X_all, batch_size=4096, verbose=0)               # one inference pass: INNER term + diagnostics
        ev = monitor.score(h.history["loss"][-1], P, w)                    # TRAIN term = the epoch's running loss
        ev.update(monitor.inner_diagnostics(P))
        hist.append(dict(epoch=ep, **ev))
        if ev["score"] < best - TRAINING["min_delta"]:
            best, best_epoch, best_w, stale = ev["score"], ep, [np.array(v) for v in model.get_weights()], 0
        else:
            stale += 1
            if stale >= TRAINING["patience"]:
                stop = "patience"
                break
        progress(ep, ev["score"])
    if best_w is not None:
        model.set_weights(best_w)
    return dict(epochs=len(hist), best_epoch=best_epoch, best_score=best, stop_reason=stop,
                monitor="0.5*(train_running_weighted_loss + inner_weighted_log_loss); inner strategy Sharpe is a diagnostic only",
                diagnostic_threshold=MONITOR_TAU, seconds=time.monotonic() - t0, history=hist, class_weights=w.tolist())


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    data = load_arms(args)
    y, per = data["y"], data["period"]
    tr, se = per == "TRAIN_2020_2022", per == "SELECT_2023"
    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "gpu" and not gpus:
        raise SystemExit("NO_GPU_VISIBLE")
    max_epochs = 1 if args.measure else TRAINING["max_epochs"]
    seeds = [6102026 + i for i in range(1 if args.measure else args.seeds)]
    XA = [data["xa"], data["offsets"], data["closure"]]
    XB = data["xb"]
    monitor = StrategyMonitor(data, HS / "market_data/forex/g10/eurusd/1h.parquet")
    fits = {}
    log = lambda m, s, ep, mon: print(json.dumps(dict(model=m, seed=s, epoch=ep, monitor=mon)), flush=True)  # noqa: E731
    with tf.device("/GPU:0" if args.device == "gpu" else "/CPU:0"):
        for seed in seeds:
            model, n_params = m1_classifier(data["effective"], seed)
            r = fit_classifier(model, [a[tr] for a in XA], y[tr], XA, monitor, seed, max_epochs,
                               lambda ep, mon, s=seed: log("m1", s, ep, mon))
            P = model.predict(XA, batch_size=4096, verbose=0)
            np.savez_compressed(out / f"pred_m1_s{seed}.npz", origin_ts=data["origin_ts"], y=y, period=per.astype(str), probs=P)
            fits[f"m1_s{seed}"] = dict(r, params=int(model.count_params()), backbone_params=n_params)
            keras.backend.clear_session()
            model = m2_classifier(XB.shape[2], seed)
            r = fit_classifier(model, XB[tr], y[tr], XB, monitor, seed, max_epochs, lambda ep, mon, s=seed: log("m2", s, ep, mon))
            P = model.predict(XB, batch_size=4096, verbose=0)
            np.savez_compressed(out / f"pred_m2_s{seed}.npz", origin_ts=data["origin_ts"], y=y, period=per.astype(str), probs=P)
            fits[f"m2_s{seed}"] = dict(r, params=int(model.count_params()))
            keras.backend.clear_session()
    from sklearn.ensemble import HistGradientBoostingClassifier
    flat = XB.reshape(len(XB), -1)
    for seed in seeds:
        t0 = time.monotonic()
        w = class_weights(y[tr])
        clf = HistGradientBoostingClassifier(max_iter=10 if args.measure else 400, learning_rate=0.05, max_leaf_nodes=31,
                                             early_stopping=False, random_state=int(seed))
        clf.fit(flat[tr], y[tr], sample_weight=w[y[tr]])
        best, best_it, stale, hist, stop, best_P = np.inf, 0, 0, [], "max_iter", None
        from sklearn.metrics import log_loss
        for it, P in enumerate(clf.staged_predict_proba(flat), start=1):   # same two loss terms, every 10 iterations
            if it % 10:
                continue
            train_loss = log_loss(y[tr], P[tr], labels=[0, 1, 2], sample_weight=w[y[tr]])   # boosting exposes no running loss
            ev = monitor.score(train_loss, P, w)
            ev.update(monitor.inner_diagnostics(P))
            hist.append(dict(iteration=it, **ev))
            if ev["score"] < best - TRAINING["min_delta"]:
                best, best_it, stale, best_P = ev["score"], it, 0, P
            else:
                stale += 1
                if stale >= TRAINING["patience"]:
                    stop = "patience"
                    break
        P = best_P if best_P is not None else clf.predict_proba(flat)
        np.savez_compressed(out / f"pred_m3_s{seed}.npz", origin_ts=data["origin_ts"], y=y, period=per.astype(str), probs=P)
        fits[f"m3_s{seed}"] = dict(iterations_fitted=int(clf.n_iter_), best_iteration=int(best_it), best_score=float(best), stop_reason=stop,
                                   monitor="0.5*(train_weighted_log_loss + inner_weighted_log_loss) every 10 iterations (TRAIN term from staged probabilities)", history=hist,
                                   seconds=time.monotonic() - t0, class_weights=w.tolist())
    import importlib.metadata as im
    doc = dict(schema="pivot_detector_gpu_run.v1", measure=bool(args.measure), seeds=seeds, device=args.device,
               gpu_devices=[d.name for d in gpus], gpu_name=(tf.config.experimental.get_device_details(gpus[0]).get("device_name") if gpus else None),
               tensorflow=tf.__version__, keras=im.version("keras"), sklearn=im.version("scikit-learn"),
               training=TRAINING, k_bars=K_BARS, cuts=CUTS, wall_seconds=time.time() - t_wall, cgroup_peak_bytes=cgroup_peak(),
               fits=fits, data=data["meta"], test_read=False, label="DEVELOPMENT_PIVOT_DETECTOR_GPU_STAGE")
    (out / "RUN.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != "SEAL.json"}
    (out / "SEAL.json").write_text(json.dumps(dict(files=files), indent=1) + "\n")
    print(json.dumps(dict(wall=doc["wall_seconds"], peak=doc["cgroup_peak_bytes"], kept=data["meta"]["kept"], fits={k: {kk: v[kk] for kk in v if kk in ("epochs", "best_epoch", "stop_reason", "best_iteration", "best_score")} for k, v in fits.items()})))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--population-digest", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--measure", action="store_true", help="1 epoch / 5 iterations, one seed: the memory measurement run")
    run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Learned causal pivot detector on the REAL feature arms (DEVELOPMENT): coordinator-side stages.

  labels    hindsight ZigZag (Depth 3 / Deviation 36.3 / Backstep 2) on the pinned lake EUR/USD H1 file ->
            per-bar 3-class targets y in {none, valley within (t,t+k], peak within (t,t+k]}, k = 2 -> labels_h1.npz
            (shipped to the GPU role; the GPU stage is tools/short_horizon_pivot_detector_train.py)
  evaluate  sealed per-origin probabilities from the GPU stage -> temperature calibration on 2023, calibrated
            log-loss vs base rate, AUC, precision/recall at the operating threshold chosen on 2023 by the
            reversal strategy's Sharpe (TP = 2 sigma, SL = 2 sigma, realistic costs, order/frequency rules),
            confirmed once on 2024 -> detector_real.csv, DETECTOR_REAL_SEAL.json
  sanity    oracle -> noise: Sharpe as detector precision falls from 1.0 to the base rate (true k-bar labels
            diluted with random false signals; 3 seeds) -> oracle_noise.csv
  figures   points on the gap figure (zigzag_brecha_detector.png) and the oracle-noise figure
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import zigzag as Z  # noqa: E402
from tools.short_horizon_zigzag_study import CAPITAL, KEEP, Ctx, PERIODS, params_of, per  # noqa: E402,F401

K_BARS = 2
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
PERIOD_MAP = {"TRAIN_2020_2022": "DECISION_2020_2022", "SELECT_2023": "VALIDATION_2023", "CONFIRM_2024": "CONFIRM_2024"}
PRECISIONS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choice_params(run):
    return params_of(json.loads((run / "CHOICE.json").read_text())["params"])


# --------------------------------------------------------------------------------------- labels
def labels(args):
    ctx = Ctx(args.data)
    p = choice_params(Path(args.zigzag_run))
    piv = ctx.pivots(p)
    n = len(ctx.bars)
    y = np.zeros(n, np.int64)
    for i, kind, _ in piv:
        a, b = max(0, i - K_BARS), i - 1
        y[a:b + 1] = 1 if kind == Z.VALLEY else 2
    valid = np.arange(n) + K_BARS < n                  # the (t, t+k] window exists inside the loaded series
    bar_ts = ctx.bars.index.as_unit("s").asi8
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "labels_h1.npz", bar_ts=bar_ts, y=y, valid=valid, close=ctx.bars.CLOSE.to_numpy(float),
                        high=ctx.bars.HIGH.to_numpy(float))
    doc = dict(label="HINDSIGHT_ZIGZAG_TARGETS_DEVELOPMENT_ONLY", params=p.__dict__, k_bars=K_BARS, bars=int(n),
               pivots=len(piv), data_sha256=ctx.digest, zigzag_module_sha256=sha(ROOT / "app/short_horizon/zigzag.py"),
               class_counts=np.bincount(y, minlength=3).tolist(), window=[str(ctx.bars.index[0]), str(ctx.bars.index[-1])],
               labels_sha256=sha(out / "labels_h1.npz"), test_read=False,
               note="pivots are the final repainted ZigZag over 2019-12..2024-12 (hindsight by construction; labels only)")
    (out / "LABELS.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc))


# --------------------------------------------------------------------------------------- evaluation
def temperature(P, y, mask):
    """Post-hoc temperature on the 2023 selection rows: p_T ~ p^(1/T), T on a log grid by log-loss."""
    from sklearn.metrics import log_loss
    best, best_T = np.inf, 1.0
    logp = np.log(np.clip(P[mask], 1e-9, 1))
    for T in np.exp(np.linspace(np.log(0.25), np.log(8.0), 81)):
        q = np.exp(logp / T)
        q /= q.sum(axis=1, keepdims=True)
        ll = log_loss(y[mask], q, labels=[0, 1, 2])
        if ll < best:
            best, best_T = ll, float(T)
    return best_T


def calibrate(P, T):
    q = np.exp(np.log(np.clip(P, 1e-9, 1)) / T)
    return q / q.sum(axis=1, keepdims=True)


def signals_from(P, origin_bar, tau, k):
    """[(bar, direction)] with one signal per k bars (the study's rule); classes by argmax of valley/peak >= tau."""
    sig, last = [], -10 ** 9
    for j in np.argsort(origin_bar):
        t = int(origin_bar[j])
        if t - last < k:
            continue
        if P[j, 1] >= tau and P[j, 1] >= P[j, 2]:
            sig.append((t, 1)),
            last = t
        elif P[j, 2] >= tau and P[j, 2] > P[j, 1]:
            sig.append((t, -1))
            last = t
    return sig


def metrics(P, y, mask, base):
    from sklearn.metrics import log_loss, roc_auc_score
    yy = y[mask]
    return dict(auc_valley=roc_auc_score(yy == 1, P[mask, 1]) if (yy == 1).any() else math.nan,
                auc_peak=roc_auc_score(yy == 2, P[mask, 2]) if (yy == 2).any() else math.nan,
                log_loss=log_loss(yy, P[mask], labels=[0, 1, 2]),
                log_loss_base_rate=log_loss(yy, np.tile(base, (mask.sum(), 1)), labels=[0, 1, 2]),
                base_rate_valley=float(base[1]), base_rate_peak=float(base[2]), n=int(mask.sum()))


def prec_recall(sig, y, origin_bar, mask):
    """Precision of emitted signals and recall of labelled pivot origins, on the masked origins."""
    bar_y = {int(b): int(v) for b, v, m in zip(origin_bar, y, mask) if m}
    sel = [(t, d) for t, d in sig if t in bar_y]
    hit = [bar_y[t] == (1 if d > 0 else 2) for t, d in sel]
    positives = sum(1 for v in bar_y.values() if v > 0)
    return (float(np.mean(hit)) if hit else math.nan, float(sum(hit) / positives) if positives else math.nan, len(sel))


def evaluate(args):
    run = Path(args.zigzag_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(args.data)
    piv = ctx.pivots(choice_params(run))
    k_tp = float(pd.read_csv(run / "confirmed.csv").query("chosen").k_tp.iloc[0])
    bar_index = {int(t): i for i, t in enumerate(ctx.bars.index.as_unit("s").asi8)}
    gpu = Path(args.gpu_out)
    seal = json.loads((gpu / "SEAL.json").read_text())
    for name, digest in seal["files"].items():
        if sha(gpu / name) != digest:
            raise SystemExit("GPU_OUTPUT_DIGEST:" + name)
    runinfo = json.loads((gpu / "RUN.json").read_text())
    rows, preds = [], {}
    for path in sorted(gpu.glob("pred_m*_s*.npz")):
        model, seed = path.stem.split("_")[1], int(path.stem.split("_s")[1])
        with np.load(path, allow_pickle=False) as z:
            origin_ts, y, period, P = z["origin_ts"], z["y"], z["period"].astype(str), z["probs"]
        period = np.array([PERIOD_MAP[p] for p in period])
        origin_bar = np.array([bar_index.get(int(t), -1) for t in origin_ts])
        if (origin_bar < 0).any():
            raise SystemExit("ORIGIN_NOT_IN_LAKE")
        masks = {name: period == name for name in PERIODS}
        T = temperature(P, y, masks["VALIDATION_2023"])
        Pc = calibrate(P, T)
        base = np.bincount(y[masks["DECISION_2020_2022"]], minlength=3) / masks["DECISION_2020_2022"].sum()
        preds[(model, seed)] = dict(origin_ts=origin_ts, y=y, period=period, origin_bar=origin_bar, P=Pc, T=T)
        for tau in THRESHOLDS:
            sig = signals_from(Pc, origin_bar, tau, K_BARS)
            for name in PERIODS:
                m, _ = ctx.run(sig, piv, name, mode="causal", k_tp=k_tp)
                prec, rec, n_sig = prec_recall(sig, y, origin_bar, masks[name])
                m = {k: v for k, v in m.items() if k != "period"}
                rows.append(dict(model=model, seed=seed, temperature=T, threshold=tau, period=name, signals=n_sig,
                                 precision_at_threshold=prec, recall_at_threshold=rec, **metrics(Pc, y, masks[name], base), **m))
    tag = ("_" + args.tag) if args.tag else ""
    det = pd.DataFrame(rows)
    det.to_csv(out / f"detector_real{tag}.csv", index=False)
    # operating threshold per model: best 2023 Sharpe (valid first), mean over seeds; 2024 reported once at that threshold
    choice, summary = {}, []
    for model in sorted(det.model.unique()):
        v = det[(det.model == model) & (det.period == "VALIDATION_2023")]
        g = v.groupby("threshold").agg(sharpe=("sharpe_daily_sqrt365", "mean"), valid=("valid", "min")).reset_index()
        gv = g[g.valid.astype(bool)]
        tau = float((gv if not gv.empty else g).sort_values("sharpe", ascending=False).threshold.iloc[0])
        choice[model] = tau
        for name in PERIODS:
            d = det[(det.model == model) & (det.threshold == tau) & (det.period == name)]
            summary.append(dict(model=model, threshold=tau, period=name, seeds=len(d),
                                **{f"{c}_mean": float(d[c].mean()) for c in ("sharpe_daily_sqrt365", "max_drawdown_pct", "trades_per_week",
                                                                              "precision_at_threshold", "recall_at_threshold", "auc_valley",
                                                                              "auc_peak", "log_loss", "log_loss_base_rate", "win_rate")},
                                sharpe_sd=float(d.sharpe_daily_sqrt365.std(ddof=0)), valid_all=bool(d.valid.all())))
    pd.DataFrame(summary).to_csv(out / f"detector_real_summary{tag}.csv", index=False)
    # sealed predictions of the chosen operating points (calibrated)
    frames = []
    for (model, seed), d in preds.items():
        frames.append(pd.DataFrame(dict(model=model, seed=seed, origin_ts=d["origin_ts"], period=d["period"], y=d["y"],
                                        p_none=d["P"][:, 0], p_valley=d["P"][:, 1], p_peak=d["P"][:, 2])))
    pred_path = out / f"detector_real_predictions{tag}.csv.gz"
    pd.concat(frames).to_csv(pred_path, index=False, float_format="%.6f")
    doc = dict(label="DEVELOPMENT_REAL_FEATURE_PIVOT_DETECTOR", predictions_sha256=sha(pred_path), gpu_seal=seal["files"],
               gpu_run=dict((k, runinfo[k]) for k in ("gpu_name", "tensorflow", "keras", "sklearn", "seeds", "wall_seconds",
                                                       "cgroup_peak_bytes", "training", "k_bars")),
               data=runinfo["data"], operating_threshold=choice, temperatures={f"{m}_s{s}": d["T"] for (m, s), d in preds.items()},
               k_tp=k_tp, m_sl=2.0, chosen_on="VALIDATION_2023 (threshold by mean Sharpe over seeds, valid first)",
               confirm="CONFIRM_2024 once", lake_sha256=ctx.digest, expert_params=choice_params(run).__dict__, test_read=False)
    doc["input_mode"] = runinfo.get("input_mode", "manifest19_population")
    (out / f"DETECTOR_REAL_SEAL{tag}.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(json.dumps(dict(choice=choice, summary=[s for s in summary if s["period"] != "DECISION_2020_2022"]), default=float))


# --------------------------------------------------------------------------------------- oracle -> noise
def sanity(args):
    run = Path(args.zigzag_run)
    out = Path(args.out)
    ctx = Ctx(args.data)
    piv = ctx.pivots(choice_params(run))
    k_tp = float(pd.read_csv(run / "confirmed.csv").query("chosen").k_tp.iloc[0])
    with np.load(Path(args.gpu_out) / "pred_m1_s6102026.npz", allow_pickle=False) as z:  # the run's own origins
        origin_ts, y, period = z["origin_ts"], z["y"], z["period"].astype(str)
    period = np.array([PERIOD_MAP[p] for p in period])
    bar_index = {int(t): i for i, t in enumerate(ctx.bars.index.as_unit("s").asi8)}
    origin_bar = np.array([bar_index[int(t)] for t in origin_ts])
    rows = []
    for name in ("VALIDATION_2023", "CONFIRM_2024"):
        m = period == name
        pos = np.flatnonzero(m & (y > 0))
        neg = np.flatnonzero(m & (y == 0))
        base = len(pos) / m.sum()
        for p in list(PRECISIONS) + [round(base, 3)]:
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                n_false = int(round(len(pos) * (1 - p) / p))
                false = rng.choice(neg, size=min(n_false, len(neg)), replace=False)
                sig = [(int(origin_bar[j]), 1 if y[j] == 1 else -1) for j in pos]
                sig += [(int(origin_bar[j]), int(rng.choice([1, -1]))) for j in false]
                sig.sort()
                # TP = 2 sigma (causal) only: the study's hindsight next-pivot TP is undefined for origin-based
                # (pre-pivot) signals, whose "next pivot" is the labelled one at the entry level itself.
                mm, _ = ctx.run(sig, piv, name, mode="causal", k_tp=k_tp)
                mm = {k: v for k, v in mm.items() if k != "period"}
                rows.append(dict(period=name, precision=p, seed=seed, signals=len(sig), true_signals=len(pos),
                                 false_signals=len(false), base_rate=base, tp_mode="TP_2SIGMA_CAUSAL", **mm))
    pd.DataFrame(rows).to_csv(out / ("oracle_noise" + (("_" + args.tag) if args.tag else "") + ".csv"), index=False)
    print(pd.DataFrame(rows).groupby(["period", "tp_mode", "precision"]).sharpe_daily_sqrt365.mean().to_string())


# --------------------------------------------------------------------------------------- figures
def figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run, out = Path(args.zigzag_run), Path(args.out)
    conf = pd.read_csv(run / "confirmed.csv")
    conf = conf[conf.chosen]
    smoke = pd.read_csv(run / "detector.csv")
    sseal = json.loads((run / "DETECTOR_SEAL.json").read_text())
    smoke = smoke[(smoke.k_bars == sseal["k_bars"]) & (smoke.threshold == sseal["threshold"])]
    summ = pd.read_csv(out / "detector_real_summary.csv")
    tag = ("_" + args.tag) if args.tag else ""
    summ2 = pd.read_csv(out / f"detector_real_summary{tag}.csv") if args.tag and (out / f"detector_real_summary{tag}.csv").exists() else None
    periods = ["DECISION_2020_2022", "VALIDATION_2023", "CONFIRM_2024"]
    es = {"DECISION_2020_2022": "Decisión 2020–2022", "VALIDATION_2023": "Validación 2023", "CONFIRM_2024": "Confirmación 2024 (una vez)"}
    series = [("(a) experto con retrospectiva (NO causal)", [float(conf[conf.period == p].hind_sharpe_daily_sqrt365.iloc[0]) for p in periods], "#5b3f8c"),
              ("(b) confirmado causalmente", [float(conf[conf.period == p].conf_sharpe_daily_sqrt365.iloc[0]) for p in periods], "#1f7a7a"),
              ("(c) sonda MLP sin ventana, rasgos sustitutos", [float(smoke[smoke.period == p].sharpe_daily_sqrt365.iloc[0]) for p in periods], "#b35a2e")]
    labels_m = {"m1": "(d) m1 receta I6-DMO (ventana 13×2 h, 19 rasgos reales)", "m2": "(e) m2 GRU (48 barras horarias)", "m3": "(f) m3 gradient boosting (48 barras aplanadas)"}
    cols = {"m1": "#1f4e9c", "m2": "#2e8b57", "m3": "#8a6d1f"}
    for mdl in sorted(summ.model.unique()):
        series.append((labels_m[mdl] + f" · umbral {summ[summ.model == mdl].threshold.iloc[0]:.1f} · media de {int(summ[summ.model == mdl].seeds.iloc[0])} semillas",
                       [float(summ[(summ.model == mdl) & (summ.period == p)].sharpe_daily_sqrt365_mean.iloc[0]) for p in periods], cols[mdl]))
    labels_g = {"m1": "(g) m1 receta I6-DMO · 15 entradas núcleo admitido (13 obs × paso 2, z-score causal 256)",
                "m2": "(h) m2 GRU · 15 entradas núcleo admitido", "m3": "(i) m3 gradient boosting · 15 entradas núcleo admitido"}
    cols_g = {"m1": "#7aa6e0", "m2": "#7fcf9b", "m3": "#d4b35a"}
    if summ2 is not None:
        for mdl in sorted(summ2.model.unique()):
            series.append((labels_g[mdl] + f" · umbral {summ2[summ2.model == mdl].threshold.iloc[0]:.1f}",
                           [float(summ2[(summ2.model == mdl) & (summ2.period == p)].sharpe_daily_sqrt365_mean.iloc[0]) for p in periods], cols_g[mdl]))
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
    xs, w = np.arange(len(periods)), 0.8 / len(series)
    for i, (lab, vals, col) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        ax.bar(xs + off, vals, w, color=col, label=lab)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}".replace(".", ","), (x + off, v + (0.3 if v >= 0 else -0.9)), ha="center", fontsize=7)
    ax.axhline(0, color="#6b7378", linewidth=1)
    ax.set_xticks(xs, [es[p] for p in periods])
    ax.set_ylabel("Sharpe anualizado (reversión trivial en pivotes; TP 2·σ̂ salvo (a); 1:100; costes ×1)")
    ax.set_title("Brecha experto–causal con los detectores reales (rasgos de Musashi, población MG2m) · DESARROLLO", loc="left", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out / "zigzag_brecha_detector.png", dpi=130)
    fig.savefig(out / "zigzag_brecha_detector.pdf")
    plt.close(fig)
    on = pd.read_csv(out / ("oracle_noise" + tag + ".csv")) if args.tag and (out / ("oracle_noise" + tag + ".csv")).exists() else pd.read_csv(out / "oracle_noise.csv")
    if summ2 is not None:
        summ = summ2
        cols = cols_g
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    for ax, name in zip(axs, ("VALIDATION_2023", "CONFIRM_2024")):
        for mode, col in (("TP_2SIGMA_CAUSAL", "#1f7a7a"),):
            g = on[(on.period == name) & (on.tp_mode == mode)].groupby("precision").agg(y=("sharpe_daily_sqrt365", "mean"), sd=("sharpe_daily_sqrt365", "std")).sort_index()
            ax.errorbar(g.index, g.y, yerr=g.sd.fillna(0), marker="o", color=col, label=mode)
        for mdl in sorted(summ.model.unique()):
            r = summ[(summ.model == mdl) & (summ.period == name)].iloc[0]
            ax.scatter([r.precision_at_threshold_mean], [r.sharpe_daily_sqrt365_mean], marker="*", s=160, color=cols[mdl], zorder=5, label=f"{mdl} real")
        ax.axhline(0, color="#6b7378", linewidth=1)
        ax.set_xlabel("Precisión del detector (señales correctas / señales emitidas)")
        ax.set_ylabel("Sharpe anualizado")
        ax.set_title(f"Oráculo → ruido · {es[name]}", loc="left", fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"oraculo_ruido_precision{tag}.png", dpi=130)
    plt.close(fig)
    # per-epoch curves (overfitting diagnostic): TRAIN running loss vs INNER weighted loss, their mean, INNER Sharpe
    runinfo = json.loads((Path(args.gpu_out) / "RUN.json").read_text())
    fits = runinfo["fits"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), facecolor="white", sharex="col")
    for col, mdl in enumerate(("m1", "m2", "m3")):
        for key, fit in sorted(fits.items()):
            if not key.startswith(mdl + "_"):
                continue
            h = pd.DataFrame(fit["history"])
            x = h["epoch"] if "epoch" in h else h["iteration"]
            seed = key.split("_s")[1]
            axs[0, col].plot(x, h.train_running_loss, "--", linewidth=1, label=f"TRAIN running loss · s{seed}")
            axs[0, col].plot(x, h.inner_weighted_loss, "-", linewidth=1.2, label=f"INNER 2023 weighted loss · s{seed}")
            axs[0, col].plot(x, h.score, ":", linewidth=1.6, label=f"monitor (mean) · s{seed}")
            axs[0, col].axvline(fit.get("best_epoch", fit.get("best_iteration")), color="#999", linewidth=0.8)
            axs[1, col].plot(x, h.inner_sharpe, linewidth=1.2, label=f"INNER Sharpe (diagnostic) · s{seed}")
        axs[0, col].set_title(f"{mdl}: pérdida ponderada por época (gap TRAIN–INNER)", loc="left", fontsize=9)
        axs[1, col].set_title(f"{mdl}: Sharpe INNER 2023 por época (diagnóstico, no selecciona)", loc="left", fontsize=9)
        axs[1, col].axhline(0, color="#6b7378", linewidth=0.8)
        axs[1, col].set_xlabel("época (m1, m2) / iteración (m3)")
        for ax in axs[:, col]:
            ax.legend(fontsize=6)
    fig.suptitle("Curvas de entrenamiento: término TRAIN = pérdida en curso del propio entrenamiento; término INNER = misma pérdida en 2023 purgado; parada en la media · DESARROLLO", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"curvas_entrenamiento{tag}.png", dpi=130)
    plt.close(fig)
    rows = []
    for key, fit in sorted(fits.items()):
        for h in fit["history"]:
            rows.append(dict(fit=key, **{k: v for k, v in h.items()}))
    pd.DataFrame(rows).to_csv(out / f"training_curves{tag}.csv", index=False)
    print("figures written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("labels", "evaluate", "sanity", "figures"))
    ap.add_argument("--data", required=True, help="lake EUR/USD H1 parquet (pinned digest)")
    ap.add_argument("--zigzag-run", default=str(ROOT / "docs/short_horizon/results/zigzag"))
    ap.add_argument("--out", default=str(ROOT / "docs/short_horizon/results/pivot_detector"))
    ap.add_argument("--gpu-out", help="fetched GPU-stage output directory (pred_*.npz, RUN.json, SEAL.json)")
    ap.add_argument("--tag", default="", help="suffix for a second run's outputs (e.g. sel15)")
    a = ap.parse_args()
    {"labels": labels, "evaluate": evaluate, "sanity": sanity, "figures": figures}[a.stage](a)


if __name__ == "__main__":
    main()

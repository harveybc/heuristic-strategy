#!/usr/bin/env python3
"""Figures and tables for the ZigZag hindsight-expert study (DEVELOPMENT, doctoral style)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.analysis import zero_crossing  # noqa: E402
from tools.short_horizon_price_path_report import GRID, INK, PURPLE, RUST, TEAL, es  # noqa: E402

PER_ES = {"DECISION_2020_2022": "Decisión 2020-2022", "VALIDATION_2023": "Validación 2023", "CONFIRM_2024": "Confirmación 2024 (una vez)"}


def fig_front(front, choice, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor="white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8)
    acc = front[front.frequency_rule == "ACCEPTED"]
    rej = front[front.frequency_rule != "ACCEPTED"]
    sc = ax.scatter(acc.dec_max_drawdown_pct, acc.dec_sharpe_daily_sqrt365, c=acc.dec_trades_per_week, cmap="viridis", s=70,
                    edgecolors=INK, linewidths=0.6, label="punto del frente (aceptado por frecuencia)")
    if not rej.empty:
        ax.scatter(rej.dec_max_drawdown_pct, rej.dec_sharpe_daily_sqrt365, facecolors="none", edgecolors=RUST, s=80,
                   label="rechazado (< 3 órdenes/semana o > 4/día)")
    p = choice["params"]
    ch = front[(front.depth == p["depth"]) & (front.deviation_pips == p["deviation_pips"]) & (front.backstep == p["backstep"])]
    if not ch.empty:
        ax.scatter(ch.dec_max_drawdown_pct, ch.dec_sharpe_daily_sqrt365, marker="*", s=320, facecolors="none", edgecolors=RUST,
                   linewidths=2, label=f"elegido por Sharpe 2023: Depth {p['depth']}, Deviation {es(p['deviation_pips'], 1)} pips, Backstep {p['backstep']}")
    for _, r in front.iterrows():
        ax.annotate(f"{int(r.depth)}/{r.deviation_pips:.0f}/{int(r.backstep)}", (r.dec_max_drawdown_pct, r.dec_sharpe_daily_sqrt365),
                    fontsize=7, color=INK, xytext=(4, 3), textcoords="offset points")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("órdenes por semana (2020-2022)", color=INK)
    ax.set_xlabel("Máxima caída (% del capital) · 2020-2022", color=INK)
    ax.set_ylabel("Sharpe anualizado · 2020-2022", color=INK)
    ax.set_title("Frente de Pareto NSGA-II del experto ZigZag con retrospectiva (Depth / Deviation / Backstep) · "
                 "HINDSIGHT_NON_CAUSAL_EXPERT · DESARROLLO", fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8, loc="lower right", frameon=True, edgecolor="#d0d0d0")
    fig.tight_layout()
    fig.savefig(out / "zigzag_frente_pareto.png", dpi=130)
    fig.savefig(out / "zigzag_frente_pareto.pdf")
    plt.close(fig)


def fig_gap(conf, det, seal, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = conf[conf.chosen]
    d = det[(det.k_bars == seal["k_bars"]) & (det.threshold == seal["threshold"])]
    periods = list(PER_ES)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axhline(0, color="#6b7378", linewidth=1.0)
    xs = np.arange(len(periods))
    w = 0.26
    a = [float(c[c.period == p].hind_sharpe_daily_sqrt365.iloc[0]) for p in periods]
    b = [float(c[c.period == p].conf_sharpe_daily_sqrt365.iloc[0]) for p in periods]
    e = [float(d[d.period == p].sharpe_daily_sqrt365.iloc[0]) for p in periods]
    ax.bar(xs - w, a, w, color=PURPLE, label="(a) ZigZag con retrospectiva = experto (NO causal; TP = siguiente pivote)")
    ax.bar(xs, b, w, color=TEAL, label=f"(b) ZigZag confirmado causalmente (mismos parámetros; retardo mediano {es(c.lag_median.iloc[0], 0)} barras; TP = {es(c.k_tp.iloc[0], 1)}·σ̂)")
    ax.bar(xs + w, e, w, color=RUST, label=f"(c) SMOKE_PROBE_WEAK_DETECTOR (MLP sin ventana, k = {seal['k_bars']} barras, umbral {es(seal['threshold'], 1)}, rasgos sustitutos, una semilla) — no es evidencia del techo alcanzable")
    for x, (va, vb, ve) in enumerate(zip(a, b, e)):
        ax.annotate(f"brecha (a)−(b) = {es(va - vb, 1)}", (x, max(va, 0) + 0.6), ha="center", fontsize=9, color=INK)
        for off, v in ((-w, va), (0, vb), (w, ve)):
            ax.annotate(es(v, 2), (x + off, v + (0.3 if v >= 0 else -0.9)), ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs, [PER_ES[p] for p in periods])
    ax.set_ylabel("Sharpe anualizado (estrategia trivial de reversión en pivotes; 1:100; costes ×1)", color=INK)
    ax.set_title("Información perdida por el retardo de confirmación: la brecha que una política aprendida debe cerrar · DESARROLLO",
                 fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8, loc="center right", frameon=True, edgecolor="#d0d0d0")
    fig.tight_layout()
    fig.savefig(out / "zigzag_brecha.png", dpi=130)
    fig.savefig(out / "zigzag_brecha.pdf")
    plt.close(fig)


def fig_noise(noise, conf, det, seal, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(17, 6.5), facecolor="white")
    bes = {}
    for ax, period in zip(axs, ("VALIDATION_2023", "CONFIRM_2024")):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.axhline(0, color="#6b7378", linewidth=1.0)
        d = noise[noise.period == period]
        for J, col in zip(sorted(d.jitter.unique()), (PURPLE, TEAL, "#1f4e9c", RUST)):
            g = d[d.jitter == J].groupby("delay").agg(x=("timing_error_bars", "mean"), y=("sharpe_daily_sqrt365", "mean"),
                                                    sd=("sharpe_daily_sqrt365", "std"), valid=("valid", "min")).sort_values("x")
            ax.plot(g.x, g.y, color=col, linewidth=2.0 if J == 0 else 1.4, label=f"retardo d con jitter ±{int(J)} barras")
            if J > 0:
                ax.fill_between(g.x, g.y - g.sd.fillna(0), g.y + g.sd.fillna(0), color=col, alpha=0.12, linewidth=0)
            ok = g.valid.to_numpy().astype(bool)
            ax.scatter(g.x[ok], g.y[ok], color=col, s=28, zorder=4)
            ax.scatter(g.x[~ok], g.y[~ok], facecolors="none", edgecolors=col, s=40, zorder=4)
            if J == 0:
                bes[period] = zero_crossing(g.x.to_numpy(), g.y.to_numpy())
        if bes.get(period) is not None:
            ax.axvline(bes[period], color=RUST, linestyle=":", linewidth=1.6, label=f"Equilibrio: {es(bes[period], 2)} barras de error")
        c = conf[conf.chosen & (conf.period == period)].iloc[0]
        ax.scatter([c.lag_mean], [c.conf_sharpe_daily_sqrt365], marker="s", s=120, facecolors="none", edgecolors=INK, linewidths=2,
                   zorder=6, label=f"(b) confirmado: retardo medio {es(c.lag_mean, 1)} barras, TP = {es(c.k_tp, 1)}·σ̂")
        ax.set_xlabel("Error medio de sincronización del pivote (barras de 1 h)", color=INK)
        ax.set_ylabel("Sharpe anualizado", color=INK)
        ax.set_title(PER_ES[period], fontsize=11, color=INK, loc="left")
        ax.legend(fontsize=7.5, loc="upper right", frameon=True, edgecolor="#d0d0d0")
    fig.suptitle("Sharpe frente al error de sincronización de los pivotes (experto con retrospectiva retrasado y perturbado; "
                 "TP = siguiente pivote) · DESARROLLO · 1:100 · costes ×1", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "zigzag_ruido_temporal.png", dpi=130)
    fig.savefig(out / "zigzag_ruido_temporal.pdf")
    plt.close(fig)
    # detection accuracy figure
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.axhline(0, color="#6b7378", linewidth=1.0)
    for period, col in (("VALIDATION_2023", PURPLE), ("CONFIRM_2024", TEAL)):
        for k, mk in ((2, "o"), (4, "^")):
            g = det[(det.period == period) & (det.k_bars == k) & (det.signals > 0)].sort_values("precision_at_threshold")
            ok = g.valid.to_numpy().astype(bool)
            ax.plot(g.precision_at_threshold, g.sharpe_daily_sqrt365, color=col, linewidth=1.2, alpha=0.7)
            ax.scatter(g.precision_at_threshold[ok], g.sharpe_daily_sqrt365[ok], marker=mk, color=col, s=70, zorder=4,
                       label=f"{PER_ES[period]} · k = {k} barras")
            ax.scatter(g.precision_at_threshold[~ok], g.sharpe_daily_sqrt365[~ok], marker=mk, facecolors="none", edgecolors=col, s=80, zorder=4)
    base = float(det[det.k_bars == seal["k_bars"]].base_rate_valley.iloc[0]) * 2
    ax.axvline(base, color=RUST, linestyle="--", linewidth=1.4, label=f"tasa base de pivote en k barras ({es(base, 2)})")
    ax.set_xlabel("Precisión del detector en el umbral (pivote correcto dentro de k barras)", color=INK)
    ax.set_ylabel("Sharpe anualizado (reversión; TP = k_tp·σ̂)", color=INK)
    ax.set_title("Sharpe frente a la precisión de detección de pivotes — SMOKE_PROBE_WEAK_DETECTOR (puntos huecos: inválidos por frecuencia) · DESARROLLO",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=True, edgecolor="#d0d0d0")
    fig.tight_layout()
    fig.savefig(out / "zigzag_deteccion.png", dpi=130)
    fig.savefig(out / "zigzag_deteccion.pdf")
    plt.close(fig)
    return bes


def tables(front, choice, conf, lags, noise, det, seal, man):
    md = ["# ZigZag hindsight-expert study — tables (generated)\n",
          "DEVELOPMENT. The hindsight ZigZag is NON-CAUSAL (repaints): it is the expert for demonstrations, never a result. "
          "Trivial reversal strategy on EUR/USD H1 2020-2024 (lake), rel_volume 0,10, 1:100, costs ×1, SL = 2·σ̂ (causal 24 h vol), "
          "≥ 3 orders/week and ≤ 4/day.\n",
          f"## Pareto front (NSGA-II on 2020-2022, {choice['population']} individuals, {choice['generations']} generations; "
          f"{choice['front_points']} points, {choice['front_accepted']} accepted by the frequency rule)\n",
          "| Depth | Deviation (pips) | Backstep | Pivots | Dec. Sharpe | Dec. max DD % | Dec. orders/wk | Val. 2023 Sharpe | Val. orders/wk | Frequency rule |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in front.iterrows():
        md.append(f"| {int(r.depth)} | {es(r.deviation_pips, 1)} | {int(r.backstep)} | {int(r.pivots)} | {es(r.dec_sharpe_daily_sqrt365, 2)} | "
                  f"{es(r.dec_max_drawdown_pct, 1)} | {es(r.dec_trades_per_week, 1)} | {es(r.val_sharpe_daily_sqrt365, 2)} | "
                  f"{es(r.val_trades_per_week, 1)} | {r.frequency_rule} |")
    p = choice["params"]
    cf = choice["confirm_2024"]
    md.append(f"\nChosen (best 2023 Sharpe among accepted): Depth {p['depth']}, Deviation {es(p['deviation_pips'], 1)} pips, Backstep {p['backstep']}. "
              f"Confirmation 2024 (hindsight expert): Sharpe {es(cf['sharpe_daily_sqrt365'], 2)}, max DD {es(cf['max_drawdown_pct'], 1)} %, "
              f"{es(cf['trades_per_week'], 1)} orders/wk, TP share {es(cf['share_exit_tp'], 2)}. Leverage 1:10 / 1:50 Sharpe: "
              + " / ".join(es(v["sharpe_daily_sqrt365"], 2) for v in choice["confirm_2024_leverage"].values()) + ".\n")
    md.append("## Hindsight (a) vs causal-confirmed (b): the confirmation-lag gap\n")
    md.append("| Params | Lag median / mean / p90 (bars) | k_tp | Period | (a) Sharpe | (a) orders/wk | (b) Sharpe | (b) orders/wk | (b) TP / SL / time | Gap (a)−(b) |")
    md.append("|---|---|---:|---|---:|---:|---:|---:|---|---:|")
    for _, r in conf.iterrows():
        md.append(f"| {r.params}{' (chosen)' if r.chosen else ''} | {es(r.lag_median, 0)} / {es(r.lag_mean, 1)} / {es(r.lag_p90, 0)} | {es(r.k_tp, 1)} | {r.period} | "
                  f"{es(r.hind_sharpe_daily_sqrt365, 2)} | {es(r.hind_trades_per_week, 1)} | {es(r.conf_sharpe_daily_sqrt365, 2)} | {es(r.conf_trades_per_week, 1)} | "
                  f"{es(r.conf_share_exit_tp, 2)} / {es(r.conf_share_exit_sl, 2)} / {es(r.conf_share_exit_time, 2)} | "
                  f"{es(r.hind_sharpe_daily_sqrt365 - r.conf_sharpe_daily_sqrt365, 2)} |")
    md.append("\nk_tp chosen on 2020-2022 per parameter set: " + "; ".join(
        f"{k}: " + ", ".join(f"{kk} → {es(v, 2)}" for kk, v in lg["k_tp_scores_2020_2022"].items()) for k, lg in lags.items()) + "\n")
    md.append("## Timing-noise study (chosen params; hindsight TP; mean over seeds)\n")
    md.append("| Period | Delay d | Jitter ± | Timing error (bars) | Sharpe | Orders/wk | Valid |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    g = noise.groupby(["period", "delay", "jitter"]).agg(x=("timing_error_bars", "mean"), s=("sharpe_daily_sqrt365", "mean"),
                                                        t=("trades_per_week", "mean"), v=("valid", "min")).reset_index()
    for _, r in g.iterrows():
        md.append(f"| {r.period} | {int(r.delay)} | {int(r.jitter)} | {es(r.x, 2)} | {es(r.s, 2)} | {es(r.t, 1)} | {'sí' if r.v else 'no'} |")
    md.append("\n## SMOKE_PROBE_WEAK_DETECTOR (MLP 64-32 without lookback window, class-weighted CE, 30 epochs, one seed; trained 2020-2022, chosen 2023, confirmed 2024) — not evidence about the achievable ceiling\n")
    md.append(f"Features: {len(seal['features'])} causal lake features ({seal['feature_source']}). Device {seal['gpu_name'] or seal['device']}, "
              f"peak VRAM {es(seal['peak_vram_mib'], 0)} MiB, training wall {es(seal['wall_seconds'], 1)} s. Predictions sealed sha256 `{seal['sha256'][:16]}…`.\n")
    md.append("| k | Threshold | Period | Signals | Precision | AUC valley / peak | Log-loss (base rate) | Sharpe | Orders/wk | Valid |")
    md.append("|---:|---:|---|---:|---:|---|---|---:|---:|---|")
    for _, r in det.iterrows():
        md.append(f"| {int(r.k_bars)} | {es(r.threshold, 1)} | {r.period} | {int(r.signals)} | {es(r.precision_at_threshold, 3) if np.isfinite(r.precision_at_threshold) else '—'} | "
                  f"{es(r.auc_valley, 3)} / {es(r.auc_peak, 3)} | {es(r.log_loss, 3)} ({es(r.log_loss_base_rate, 3)}) | {es(r.sharpe_daily_sqrt365, 2)} | "
                  f"{es(r.trades_per_week, 1)} | {'sí' if r.valid else 'no'} |")
    md.append(f"\n## Demonstration dataset\n\n`demonstrations/expert_trades.csv` — {man['rows']} expert trades 2020-2024, sha256 `{man['sha256']}`, "
              f"label **{man['label']}**, params {man['params']}, data sha256 `{man['data_sha256'][:16]}…`.\n")
    return "\n".join(md) + "\n"


def build(run: Path, pub: Path):
    pub.mkdir(parents=True, exist_ok=True)
    front = pd.read_csv(run / "front.csv")
    choice = json.loads((run / "CHOICE.json").read_text())
    conf = pd.read_csv(run / "confirmed.csv")
    lags = json.loads((run / "LAGS.json").read_text())
    noise = pd.read_csv(run / "noise.csv")
    det = pd.read_csv(run / "detector.csv")
    seal = json.loads((run / "DETECTOR_SEAL.json").read_text())
    man = json.loads((run / "demonstrations" / "MANIFEST.json").read_text())
    fig_front(front, choice, pub)
    fig_gap(conf, det, seal, pub)
    bes = fig_noise(noise, conf, det, seal, pub)
    (pub / "TABLAS_ZIGZAG.md").write_text(tables(front, choice, conf, lags, noise, det, seal, man))
    for name in ("front.csv", "CHOICE.json", "confirmed.csv", "LAGS.json", "noise.csv", "detector.csv", "DETECTOR_SEAL.json"):
        shutil.copy2(run / name, pub / name)
    (pub / "demonstrations").mkdir(exist_ok=True)
    for name in ("expert_trades.csv", "MANIFEST.json"):
        shutil.copy2(run / "demonstrations" / name, pub / "demonstrations" / name)
    (pub / "BREAK_EVENS.json").write_text(json.dumps(bes, indent=1) + "\n")
    print(json.dumps({"break_evens": bes}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--publish", type=Path, default=ROOT / "docs/short_horizon/results/zigzag")
    a = ap.parse_args()
    build(a.run, a.publish)


if __name__ == "__main__":
    main()

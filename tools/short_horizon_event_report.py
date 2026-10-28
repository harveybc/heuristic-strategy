#!/usr/bin/env python3
"""Figures and tables for the causal event-window study (DEVELOPMENT, FIXTURE_RESEARCH_ONLY inputs).

Publishes aggregates only: no fixture row, actual, forecast or timestamp of a single release leaves the
run directory. Figures: Sharpe / net profit / max drawdown versus each oracle error axis (doctoral
style: Spanish labels, decimal comma, seed mean with a +-1 SD band, naive lines, frequency rule with
invalid levels hollow) and the real rule as markers where it trades.
"""
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

AXIS_ES = {"drift_pips": ("MAE de la deriva post-anuncio (pips)", "naive_drift"),
           "direction_accuracy": ("Acierto de dirección de la regla de signo", "half"),
           "vol_x_mean20": ("MAE de la volatilidad × naïve MEAN20", "one"),
           "half_life_bars": ("MAE de la vida media (barras de 1 h)", None)}
PERIOD_ES = {"DECISION_2011_2017": "Decisión 2011-2017 (FIXTURE)", "CONFIRM_2018_2020": "Confirmación 2018-2020 (FIXTURE)"}
METRICS = (("sharpe_daily_sqrt365", "Sharpe anualizado"), ("net_profit_usd", "Beneficio neto (USD, escala simétrica log)"),
           ("max_drawdown_usd", "Máxima caída (USD)"))


def level_stats(c, axis, period):
    d = c[(c.axis == axis) & (c.period == period)]
    g = d.groupby("level")
    out = pd.DataFrame({"n_valid": g.valid.sum(), "tpw": g.trades_per_week.mean(), "x": g["level"].first()})
    for m, _ in METRICS:
        out[f"{m}_mean"], out[f"{m}_sd"] = g[m].mean(), g[m].std()
    out["valid"] = out.n_valid >= 3
    if axis == "drift_pips":
        out["x"] = g.drift_mae_pips.mean()
    elif axis == "direction_accuracy":
        out["x"] = g.direction_accuracy.mean()
    elif axis == "vol_x_mean20":
        out["x"] = g.vol_mae_x_mean20.mean()
    elif axis == "half_life_bars":
        out["x"] = g.half_life_mae_bars.mean()
    return out.sort_values("x")


def figure(c, real, period, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axes_names = list(AXIS_ES)
    fig, axs = plt.subplots(3, 4, figsize=(20, 12), facecolor="white")
    naive_drift = float(c[(c.period == period)].naive_drift_mae_pips.mean())
    bes = {}
    for j, axis in enumerate(axes_names):
        st = level_stats(c, axis, period)
        for i, (m, label) in enumerate(METRICS):
            ax = axs[i, j]
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.axhline(0, color="#6b7378", linewidth=1.0)
            x, y, sd = st.x.to_numpy(), st[f"{m}_mean"].to_numpy(), st[f"{m}_sd"].fillna(0).to_numpy()
            ax.plot(x, y, color=TEAL, linewidth=2.2)
            ax.fill_between(x, y - sd, y + sd, color=TEAL, alpha=0.15, linewidth=0)
            ok = st.valid.to_numpy()
            ax.scatter(x[ok], y[ok], color=TEAL, s=40, zorder=4)
            ax.scatter(x[~ok], y[~ok], facecolors="none", edgecolors=TEAL, s=55, zorder=4, linewidths=1.5,
                       label="nivel inválido (< 3 órdenes/semana o ruina)" if (~ok).any() else None)
            if m == "net_profit_usd":
                ax.set_yscale("symlog", linthresh=1000)
            kind = AXIS_ES[axis][1]
            if kind == "naive_drift":
                ax.axvline(naive_drift, color=PURPLE, linestyle="--", linewidth=1.5, label=f"Naïve (deriva 0): MAE {es(naive_drift, 1)} pips")
            elif kind == "half":
                ax.axvline(0.5, color=PURPLE, linestyle="--", linewidth=1.5, label="Moneda (0,5)")
            elif kind == "one":
                ax.axvline(1.0, color=PURPLE, linestyle="--", linewidth=1.5, label="Naïve MEAN20 (1,0×)")
            if m == "sharpe_daily_sqrt365":
                order = np.argsort(x) if axis != "direction_accuracy" else np.argsort(-x)
                be = zero_crossing(x[order], y[order])
                bes[axis] = be
                if be is not None:
                    ax.axvline(be, color=RUST, linestyle=":", linewidth=1.6, label=f"Equilibrio: {es(be, 2)}")
                rr = real[(real.period == period) & (real.arm == "NEXT_BAR_UNGATED") & (real.event_spread == "5pips")
                          & (real.leverage == 100.0)]
                if axis == "drift_pips" and not rr.empty and rr.trades.iloc[0] > 0:
                    r = rr.iloc[0]
                    ax.scatter([r.drift_mae_pips], [r.sharpe_daily_sqrt365], marker="D", s=90, facecolors="none",
                               edgecolors=RUST, linewidths=1.8, zorder=6,
                               label="Regla real SIN puerta causal (diagnóstico; inválida por frecuencia)")
                if axis == "direction_accuracy" and not rr.empty and rr.trades.iloc[0] > 0:
                    r = rr.iloc[0]
                    ax.scatter([r.direction_accuracy], [r.sharpe_daily_sqrt365], marker="D", s=90, facecolors="none",
                               edgecolors=RUST, linewidths=1.8, zorder=6)
            if i == 2:
                ax.set_xlabel(AXIS_ES[axis][0], fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(label, fontsize=10, color=INK)
            if axis == "direction_accuracy":
                ax.invert_xaxis()
            if i == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7.5, loc="best", frameon=True, edgecolor="#d0d0d0")
    fig.suptitle(f"Estrategia causal de ventana de evento — oráculo degradado por eje · {PERIOD_ES[period]} · DESARROLLO · "
                 "RF direccional tras el anuncio, 1:100, costes ×1, spread de anuncio 5 pips (SUPUESTO)\n"
                 "La regla real con puerta causal NO opera: ninguna familia muestra efecto significativo "
                 "de la sorpresa sobre la deriva en 2011-2017", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    stem = f"evento_oraculo_{'decision' if period.startswith('DECISION') else 'confirmacion'}"
    fig.savefig(outdir / f"{stem}.png", dpi=120)
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)
    return stem, bes


def tables(c, real, fam, contrib, prep, choice):
    md = ["# Causal event-window strategy — tables (generated)\n",
          "DEVELOPMENT · FIXTURE_RESEARCH_ONLY inputs (aggregates only). Decision 2011-2017, confirmation 2018-2020 "
          "(fixture ends 2020-04-29). RF directional order after the release with SL/TP, owner sizing, 1:100, "
          "costs ×1, release-hour spread 5 pips (ASSUMED), ≥ 3 trades/week rule.\n"]
    md.append("## Oracle degradation (mean of 5 seeds; Sharpe / net USD / max DD USD / trades per week / valid)\n")
    for period in ("DECISION_2011_2017", "CONFIRM_2018_2020"):
        md.append(f"### {period}\n")
        md.append("| Axis | Level | x achieved | Sharpe | Net USD | Max DD USD | Trades/wk | TP share | Valid seeds |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for axis in list(AXIS_ES) + ["entry_latency_bars"]:
            d = c[(c.axis == axis) & (c.period == period)]
            for lv, g in d.groupby("level"):
                x = {"drift_pips": g.drift_mae_pips, "direction_accuracy": g.direction_accuracy,
                     "vol_x_mean20": g.vol_mae_x_mean20, "half_life_bars": g.half_life_mae_bars,
                     "entry_latency_bars": g.level}[axis].mean()
                md.append(f"| {axis} | {es(lv, 2)} | {es(x, 2)} | {es(g.sharpe_daily_sqrt365.mean(), 2)} | "
                          f"{es(g.net_profit_usd.mean(), 0)} | {es(g.max_drawdown_usd.mean(), 0)} | "
                          f"{es(g.trades_per_week.mean(), 1)} | {es(g.share_exit_tp.mean(), 2)} | {int(g.valid.sum())}/{len(g)} |")
        md.append("")
    md.append("## Real rule (surprise sign × family sign map, fitted drift / vol / half-life)\n")
    md.append(f"Grid chosen on 2011-2017 per arm: `{json.dumps(choice)}`\n")
    md.append("| Arm | Period | Spread | Leverage | Trades | Trades/wk | Dir. acc. | Sharpe | Net USD | Max DD USD | TP / SL / time | Rule-skipped | Valid |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|")
    r = real[real.stage != "decision"]
    for _, x in r.iterrows():
        md.append(f"| {x.arm} | {x.period} | {x.event_spread} | 1:{int(x.leverage)} | {int(x.trades)} | {es(x.trades_per_week, 2)} | "
                  f"{es(x.direction_accuracy, 3) if np.isfinite(x.direction_accuracy) else '—'} | "
                  f"{es(x.sharpe_daily_sqrt365, 2) if x.trades > 0 else '—'} | {es(x.net_profit_usd, 0)} | {es(x.max_drawdown_usd, 0)} | "
                  f"{es(x.share_exit_tp, 2)} / {es(x.share_exit_sl, 2)} / {es(x.share_exit_time, 2)} | {int(x.rule_skipped_orders)} | "
                  f"{'sí' if x.valid else 'no'} |")
    md.append("")
    md.append("## Family estimates on 2011-2017 (slope of the post-release drift on the standardized surprise, through the origin)\n")
    md.append("| Family | n | Window (bars) | Slope (pips per unit surprise) | p-value | Sign | Tradable (p < 0,05, n ≥ 20) | Mean |drift| (pips) |")
    md.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for _, x in fam.sort_values("p_value").iterrows():
        md.append(f"| {x.family} | {int(x.n)} | {int(x.window)} | {es(x.slope * 1e4, 2)} | {es(x.p_value, 3)} | {int(x.sign)} | "
                  f"{'sí' if x.tradable else 'no'} | {es(x.mean_abs_drift * 1e4, 1)} |")
    md.append("")
    if contrib is not None and not contrib.empty:
        md.append("## Per-family contribution (chosen configurations, 5 pips, 1:100)\n")
        md.append("| Arm | Period | Family | Trades | Net USD | TP | SL |")
        md.append("|---|---|---|---:|---:|---:|---:|")
        for _, x in contrib.sort_values(["arm", "period", "net_usd"]).iterrows():
            md.append(f"| {x.arm} | {x.period} | {x.family} | {int(x.trades)} | {es(x.net_usd, 0)} | {int(x.tp)} | {int(x.sl)} |")
        md.append("")
    md.append("## Fixture coverage (aggregates)\n")
    md.append(f"- Clock of the hourly fixture: offset {prep['clock']['offset_hours']} h from the NFP jump "
              f"(peak label hour {prep['clock']['peak_label_hour']}, vol ratio {es(prep['clock']['vol_ratio_peak_vs_other_hours'], 2)} "
              f"vs other hours on {prep['clock']['days']} NFP days) — {prep['clock']['status']}.")
    md.append(f"- NFP release-bar |move| / same hour one day earlier: {es(prep['nfp_release_bar_vol_ratio_vs_previous_day'], 2)} "
              f"(NFP POC reported 1,07 with its own normalisation; drift n.s. in both).")
    md.append("- USD/EUR high-impact rows with numeric actual and forecast, per year:\n")
    md.append("| Year | USD | EUR |")
    md.append("|---|---:|---:|")
    yrs = prep["numeric_universe_by_year"]
    for y in sorted(yrs["USD"]):
        md.append(f"| {y} | {yrs['USD'][y]} | {yrs['EUR'].get(y, 0)} |")
    md.append("")
    return "\n".join(md) + "\n"


def build(run: Path, pub: Path):
    pub.mkdir(parents=True, exist_ok=True)
    c = pd.concat([pd.read_csv(p) for p in sorted(run.glob("cells_*of*.csv"))], ignore_index=True)
    real = pd.read_csv(run / "real.csv")
    fam = pd.read_csv(run / "families.csv")
    contrib = pd.read_csv(run / "contributions.csv") if (run / "contributions.csv").exists() else None
    prep = json.loads((run / "PREP.json").read_text())
    choice = json.loads((run / "CHOICE.json").read_text())
    stems, bes = [], {}
    for period in ("DECISION_2011_2017", "CONFIRM_2018_2020"):
        s, b = figure(c, real, period, pub)
        stems.append(s)
        bes[period] = b
    (pub / "TABLAS_EVENTO.md").write_text(tables(c, real, fam, contrib, prep, choice))
    c.to_csv(pub / "oracle_cells.csv", index=False, float_format="%.6g")
    real.to_csv(pub / "real_rule.csv", index=False, float_format="%.6g")
    fam.to_csv(pub / "families_decision.csv", index=False, float_format="%.6g")
    if contrib is not None:
        contrib.to_csv(pub / "contributions.csv", index=False, float_format="%.6g")
    for name in ("PREP.json", "CHOICE.json"):
        shutil.copy2(run / name, pub / name)
    (pub / "BREAK_EVENS.json").write_text(json.dumps(bes, indent=1) + "\n")
    print(json.dumps({"figures": stems, "break_evens": bes}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--publish", type=Path, default=ROOT / "docs/short_horizon/results/causal_event")
    a = ap.parse_args()
    build(a.run, a.publish)


if __name__ == "__main__":
    main()

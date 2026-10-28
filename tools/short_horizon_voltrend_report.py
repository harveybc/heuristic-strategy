#!/usr/bin/env python3
"""Figures and tables for the trend + volatility -> TP/SL ablation (DEVELOPMENT).

Per block size and period: Sharpe versus drift MAE, direction accuracy and sigma MAE x MEAN20, each with
the degradation curve around the oracles (doctoral style: Spanish labels, decimal comma, naive lines,
invalid points hollow) and a marker for every (trend model x vol model) cell at the a-priori defaults.
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

TREND_MK = {"t0": ("o", "#6b7378"), "t1": ("s", "#1d2b36"), "t2": ("^", "#9c2f7a"), "t3": ("D", "#b9562e"), "t4": ("*", "#237a75")}
VOL_FILL = {"MEAN20": "none", "HAR": "fill", "ORACLE": "half"}
TREND_ES = {"t0": "t0 deriva cero (moneda)", "t1": "t1 AR(3)+HAR exógena", "t2": "t2 ridge multiescala", "t3": "t3 DLinear", "t4": "t4 oráculo"}
PERIOD_ES = {"VALIDATION_2023": "Decisión 2023", "CONFIRM_2024": "Confirmación 2024 (una vez)"}
AXES = (("curve_trend_blend", "drift_mae_pips", "MAE de la deriva del bloque (pips)", "zero"),
        ("curve_direction", "direction_accuracy", "Acierto de dirección", "half"),
        ("curve_vol_blend", "sigma_mae_x_mean20", "MAE de σ̂ × naïve MEAN20", "one"))


def figure(c, B, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(19, 10.5), facecolor="white")
    bes = {}
    for i, period in enumerate(("VALIDATION_2023", "CONFIRM_2024")):
        d = c[(c.block_hours == B) & (c.period == period)]
        for j, (stage, xcol, xlabel, naive) in enumerate(AXES):
            ax = axs[i, j]
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.axhline(0, color="#6b7378", linewidth=1.0)
            cv = d[d.stage == stage].groupby("level").agg(x=(xcol, "mean"), y=("sharpe", "mean"), sd=("sharpe", "std"),
                                                          valid=("valid", "min")).sort_values("x")
            x, y, sd = cv.x.to_numpy(), cv.y.to_numpy(), cv.sd.fillna(0).to_numpy()
            ax.plot(x, y, color=TEAL, linewidth=2.0, label="oráculo degradado (otro canal ideal)")
            ax.fill_between(x, y - sd, y + sd, color=TEAL, alpha=0.15, linewidth=0)
            ok = cv.valid.to_numpy().astype(bool)
            ax.scatter(x[ok], y[ok], color=TEAL, s=30, zorder=4)
            ax.scatter(x[~ok], y[~ok], facecolors="none", edgecolors=TEAL, s=45, zorder=4)
            order = np.argsort(x) if stage != "curve_direction" else np.argsort(-x)
            be = zero_crossing(x[order], y[order])
            bes[(period, stage)] = be
            if be is not None:
                ax.axvline(be, color=RUST, linestyle=":", linewidth=1.5, label=f"Equilibrio: {es(be, 2)}")
            if naive == "zero":
                nv = float(d[d.stage == "ablation_defaults"].zero_drift_mae_pips.mean())
                ax.axvline(nv, color=PURPLE, linestyle="--", linewidth=1.4, label=f"Naïve deriva 0: {es(nv, 1)} pips")
            elif naive == "half":
                ax.axvline(0.5, color=PURPLE, linestyle="--", linewidth=1.4, label="Moneda (0,5)")
            else:
                ax.axvline(1.0, color=PURPLE, linestyle="--", linewidth=1.4, label="Naïve MEAN20 (1,0×)")
            ab = d[d.stage == "ablation_defaults"]
            for _, r in ab.iterrows():
                mk, col = TREND_MK[r.trend_model]
                fill = VOL_FILL[r.vol_model]
                ax.scatter([r[xcol]], [r.sharpe if r.valid else 0.0], marker=mk, s=95 if mk != "*" else 150, zorder=6,
                           facecolors=col if fill == "fill" else ("none" if fill == "none" else "#ffffff"),
                           edgecolors=col, linewidths=1.6 if r.valid else 0.8, alpha=1.0 if fill != "half" else 0.9,
                           label=f"{TREND_ES[r.trend_model]} · σ̂ {r.vol_model}" if i == 0 and j == 0 else None)
            if stage == "curve_direction":
                ax.invert_xaxis()
            ax.set_xlabel(xlabel, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"Sharpe anualizado · {PERIOD_ES[period]}", fontsize=10, color=INK)
            if i == 0 and j == 0:
                ax.legend(fontsize=6.8, loc="upper right", frameon=True, edgecolor="#d0d0d0", ncol=2)
            elif j > 0 or i > 0:
                h, l = ax.get_legend_handles_labels()
                keep = [(a, b) for a, b in zip(h, l) if "Equilibrio" in b or "Naïve" in b or "Moneda" in b or "oráculo" in b]
                if keep:
                    ax.legend(*zip(*keep), fontsize=7.5, loc="best", frameon=True, edgecolor="#d0d0d0")
    fig.suptitle(f"Tendencia + volatilidad → TP/SL (VolTrendStrategy en valores a priori: m = 2, c_tp = 1, w = 1, θ = 0) — "
                 f"bloques de {B} h · DESARROLLO · 1:100 · costes ×1\nMarcadores: relleno = σ̂ HAR, hueco = σ̂ MEAN20, "
                 "blanco = σ̂ oráculo; forma = modelo de tendencia; Sharpe 0 e inválido = hueco en el eje",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    stem = f"tendencia_vol_B{B}"
    fig.savefig(out / f"{stem}.png", dpi=120)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    return stem, {f"{p}|{s}": v for (p, s), v in bes.items()}


def tables(c, fronts, results, prep):
    md = ["# Trend + volatility -> TP/SL — tables (generated)\n",
          "DEVELOPMENT. VolTrendStrategy at the a-priori defaults (m = 2, c_tp = 1, w_trend = 1, theta = 0, additive TP, "
          "pos = 1), weekly causal refits (t1, t2, HAR), DLinear fit on 2020-2022 with the ridge alpha chosen on 2023, "
          "2023 decision / 2024 confirmation, leverage 1:100, costs x1, frequency rule >= 3/week.\n"]
    for B in (24, 6):
        md.append(f"## Ablation at defaults · B = {B} h\n")
        md.append("| Trend | Vol | Period | Dir. acc. (coin 0,5) | Drift MAE pips (zero-drift naive) | σ̂ MAE ×MEAN20 | Sharpe | Net USD | Max DD USD | Orders/wk | TP / SL / time | Rule-skipped | Valid |")
        md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|")
        d = c[(c.stage == "ablation_defaults") & (c.block_hours == B)]
        for _, r in d.sort_values(["trend_model", "vol_model", "period"]).iterrows():
            md.append(f"| {r.trend_model} | {r.vol_model} | {r.period} | {es(r.direction_accuracy, 3)} | {es(r.drift_mae_pips, 1)} ({es(r.zero_drift_mae_pips, 1)}) | "
                      f"{es(r.sigma_mae_x_mean20, 3)} | {es(r.sharpe, 2)} | {es(r.net, 0)} | {es(r.max_dd_usd, 0)} | {es(r.orders_per_week, 2)} | "
                      f"{es(r.share_exit_tp, 2)} / {es(r.share_exit_sl, 2)} / {es(r.share_exit_time, 2)} | {int(r.rule_skipped_orders)} | {'sí' if r.valid else 'no'} |")
        md.append("")
        md.append(f"### c_tp sensitivity on 2023 (Sharpe) · B = {B} h\n")
        s = c[(c.stage == "c_tp_sensitivity") & (c.block_hours == B)].pivot_table(index=["trend_model", "vol_model"], columns="c_tp", values="sharpe")
        md.append("| Trend | Vol | " + " | ".join(f"c_tp = {es(x, 1)}" for x in s.columns) + " | clipped TP (c_tp = 1) |")
        md.append("|---|---|" + "---:|" * (len(s.columns) + 1))
        cl = c[(c.stage == "tp_form_clipped") & (c.block_hours == B)].set_index(["trend_model", "vol_model"]).sharpe
        for (tm, vm), row in s.iterrows():
            md.append(f"| {tm} | {vm} | " + " | ".join(es(v, 2) for v in row) + f" | {es(cl.get((tm, vm), float('nan')), 2)} |")
        md.append("")
        md.append(f"### Leverage appendix (net USD / Sharpe) · B = {B} h\n")
        lv = c[(c.block_hours == B) & c.stage.isin(["ablation_defaults", "leverage"])]
        lv = lv[lv.trend_model.isin(["t0", "t4"]) | (lv.vol_model == "HAR")]
        md.append("| Trend | Vol | Period | 1:10 | 1:50 | 1:100 |")
        md.append("|---|---|---|---|---|---|")
        for (tm, vm, pe), g in lv.groupby(["trend_model", "vol_model", "period"]):
            g = g.set_index("leverage")
            cells = [f"{es(g.loc[l].net, 0)} / {es(g.loc[l].sharpe, 2)}" if l in g.index else "—" for l in (10, 50, 100)]
            md.append(f"| {tm} | {vm} | {pe} | " + " | ".join(cells) + " |")
        md.append("")
        md.append(f"## NSGA-II on 2020-2022 (optimized row) · B = {B} h\n")
        r = results[B]
        md.append(f"Population {r['population']}, generations {r['generations_done']}, front {r['front_points']} points, "
                  f"{r['front_valid_both']} valid on both 2020-2022 and 2023. Objectives: maximize Sharpe, minimize max drawdown %; "
                  "invalid (frequency, cap, ruin) penalized at Sharpe −10 / DD 100.\n")
        md.append("| Trend | Vol | TP form | c_tp | m | w | θ | Dec. Sharpe | Dec. DD % | Dec. orders/wk | Dec. valid | Val. 2023 Sharpe | Val. valid |")
        md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|")
        for _, x in fronts[B].iterrows():
            md.append(f"| {x.trend_model} | {x.vol_model} | {x.tp_form} | {es(x.c_tp, 2)} | {es(x.m, 2)} | {es(x.w_trend, 2)} | {es(x.theta, 3)} | "
                      f"{es(x.dec_sharpe, 2)} | {es(x.dec_max_dd_pct, 1)} | {es(x.dec_orders_per_week, 2)} | {'sí' if x.dec_valid else 'no'} | "
                      f"{es(x.val_sharpe, 2)} | {'sí' if x.val_valid else 'no'} |")
        ch = r["chosen"]
        cf = ch["confirm_2024"]
        md.append(f"\nChosen point (best 2023 Sharpe on the front; manifest `{ch['manifest_sha256'][:16]}…`): "
                  f"confirmation 2024 Sharpe {es(cf['sharpe'], 2)}, net {es(cf['net'], 0)} USD, max DD {es(cf['max_dd'], 1)} %, "
                  f"{es(cf['orders_per_week'], 2)} orders/wk, valid {'sí' if cf['valid'] else 'no'}.\n")
    md.append("## DLinear ridge alpha chosen on 2023 (drift MAE, price units)\n")
    for B, v in prep["blocks"].items():
        md.append(f"- B = {B} h: " + ", ".join(f"alpha {a}: {es(float(s) * 1e4, 2)} pips" for a, s in v["dlinear_alpha_scores_2023"].items())
                  + f" → alpha {v['dlinear_alpha']}")
    return "\n".join(md) + "\n"


def build(run: Path, pub: Path):
    pub.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(run / "cells.csv")
    fronts = {B: pd.read_csv(run / f"front_B{B}.csv") for B in (24, 6)}
    results = {B: json.loads((run / f"nsga_B{B}_result.json").read_text()) for B in (24, 6)}
    prep = json.loads((run / "PREP.json").read_text())
    stems, bes = [], {}
    for B in (24, 6):
        s, b = figure(c, B, pub)
        stems.append(s)
        bes[B] = b
    (pub / "TABLAS_TENDENCIA_VOL.md").write_text(tables(c, fronts, results, prep))
    c.to_csv(pub / "cells.csv", index=False, float_format="%.6g")
    for B in (24, 6):
        fronts[B].to_csv(pub / f"front_B{B}.csv", index=False, float_format="%.6g")
        shutil.copy2(run / f"nsga_B{B}_result.json", pub / f"nsga_B{B}_result.json")
    shutil.copy2(run / "PREP.json", pub / "PREP.json")
    (pub / "BREAK_EVENS.json").write_text(json.dumps(bes, indent=1) + "\n")
    from app.short_horizon import voltrend as VT
    (pub / "VOLTREND_SCHEMA.json").write_text(json.dumps(VT.SCHEMA, indent=1) + "\n")
    (pub / "DEFAULTS_MANIFEST.json").write_text(VT.manifest_json(VT.defaults()) + "\n")
    print(json.dumps({"figures": stems, "break_evens": bes}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--publish", type=Path, default=ROOT / "docs/short_horizon/results/trend_vol")
    a = ap.parse_args()
    build(a.run, a.publish)


if __name__ == "__main__":
    main()

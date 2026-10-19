#!/usr/bin/env python3
"""Report for the calendar / implied-vol range experiment (DEVELOPMENT).

Sharpe vs range MAE x MEAN20 (RF-M with a coin-flip direction): the A1r/A3r break-even curves of the
oracle/noise study (2023 from the inner-year run, 2024 from the main grid) with a marker for every real
predictor scored through the sealed adapter (filled = coin direction, hollow = logistic direction).
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
from tools.short_horizon_price_path_report import INK, MUTED, GRID, PURPLE, RUST, TEAL, es  # noqa: E402
from tools.short_horizon_vstudy_report import curve  # noqa: E402

MODEL_STYLE = {"HAR": ("o", "#1d2b36"), "HARX": ("s", "#9c2f7a"), "GBM": ("D", "#b9562e")}


def model_family(tag):
    return "GBM" if tag.startswith("GBM") else tag


def f2(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else es(v, nd)


def figure(sc, vcells, B, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.4), sharey=True, facecolor="white")
    for ax, (period, split) in zip(axes, (("2023", "INNER_2023"), ("2024", "VALIDATION_2024"))):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.axhline(0, color="#6b7378", linewidth=1.1)
        for fam, label, col, ls, lw in (("a1r", "A1r: mezcla hacia MEAN20 (requisito a citar)", PURPLE, "-", 2.0),
                                        ("a3r", "A3r: mezcla + error (lectura menos estricta)", TEAL, "-", 2.6)):
            r = curve(vcells, B, split, "RFM_COIN", fam)
            if r is None:
                continue
            c, _ = r
            ax.plot(c.rr, np.where(c.valid, c.sharpe, np.nan), color=col, linestyle=ls, linewidth=lw, marker="o",
                    markersize=4, label=label)
        ax.axvline(1.0, color=PURPLE, linestyle="--", linewidth=1.6, label="Naïve MEAN20 (1,0×)")
        ax.axvline(0.85, color=RUST, linestyle=":", linewidth=1.8, label="Objetivo del experimento (0,85×)")
        d = sc[(sc.block_hours == B) & (sc.period == period) & (sc.leverage_scenario == 100.0)]
        for _, r in d.iterrows():
            mk, col = MODEL_STYLE[model_family(r.model)]
            filled = r.direction == "coin"
            y = r.sharpe_daily_sqrt365 if r.valid else 0.0
            ax.scatter([r.mae_r_ratio_mean20], [y], marker=mk, s=110, zorder=5, linewidths=1.6, edgecolors=col,
                       facecolors=col if filled else "none",
                       label=f"{r.model} · dirección {'al azar' if filled else 'logística'}"
                             + ("" if r.valid else " — inválida (Sharpe no reportado)"))
        ax.set_title(f"{'Decisión: 2023 (ajuste 2020-2022, purgado)' if period == '2023' else 'Confirmación: VALIDACIÓN 2024 (una vez)'}",
                     fontsize=11, color=INK, loc="left")
        ax.set_xlabel("MAE del rango R̂ × naïve MEAN20 (mismas filas)", fontsize=11, color=INK)
        ax.set_xlim(0, 1.6)
    axes[0].set_ylabel("Sharpe anualizado (RF-M; retornos diarios; 365 días)", fontsize=11, color=INK)
    h, l = axes[1].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a, b in zip(h, l):
        if b not in seen:
            seen.add(b)
            hh.append(a)
            ll.append(b)
    fig.legend(hh, ll, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.80, 0.92), frameon=True, edgecolor="#d0d0d0")
    fig.suptitle(f"Predictores reales de rango sobre la curva de equilibrio — bloques de {B} h · DESARROLLO · "
                 "apalancamiento 1:100 · costes ×1", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 0.79, 0.95))
    stem = f"calendario_rango_B{B}"
    fig.savefig(outdir / f"{stem}.png", dpi=130)
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)
    return stem


def tables(sc, choices, ev, inv):
    md = ["# Calendar / implied-vol range experiment — tables (generated)\n",
          "DEVELOPMENT. Decision on 2023 (fit 2020–2022, weekly purged refits); 2024 used once for confirmation "
          "through the sealed adapter. Range MAE ratios on the same rows; RF-M at leverage 1:100, costs ×1, "
          "owner sizing, SL/TP rule, ≥ 3 trades/week, stop-out at ruin.\n"]
    for B in (24, 6):
        ch = choices.get(B)
        if ch:
            md.append(f"**B = {B}: GBM capacity chosen on 2023:** `{json.dumps(ch['chosen'])}` "
                      f"(2023 scores: {', '.join(f'{k} {v * 1e4:.2f} pips' for k, v in ch['scores_2023'].items())}).\n")
        for period in ("2023", "2024"):
            d = sc[(sc.block_hours == B) & (sc.period == period) & (sc.leverage_scenario == 100.0)]
            if d.empty:
                continue
            md.append(f"## B = {B} h · {period}\n")
            md.append("| Model | Direction | R̂ MAE ×MEAN20 (×pers.) | Û / D̂ ×MEAN20 | Dir. acc. | Log-loss | Net USD | "
                      "Sharpe | Max DD USD | Trades/wk | Valid |")
            md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for _, r in d.sort_values(["model", "direction"]).iterrows():
                md.append(f"| {r.model} | {r.direction} | {f2(r.mae_r_ratio_mean20, 3)} ({f2(r.mae_r_ratio_persist, 3)}) | "
                          f"{f2(r.mae_u_ratio_mean20, 3)} / {f2(r.mae_d_ratio_mean20, 3)} | {f2(r.direction_accuracy, 3)} | "
                          f"{f2(r.log_loss, 4) if r.direction == 'logit' else '— (hard coin)'} | {es(r.net_profit_usd, 0)} | "
                          f"{f2(r.sharpe_daily_sqrt365) if r.valid else 'inválida'} | "
                          f"{es(r.max_drawdown_usd, 0)} | {f2(r.trades_per_week, 1)} | "
                          f"{'sí' if r.valid else 'no'} |")
            md.append("")
    md.append("## Leverage appendix (RF-M, coin direction; net USD / Sharpe)\n")
    md.append("| B | Period | Model | 1:10 | 1:50 | 1:100 |")
    md.append("|---|---|---|---|---|---|")
    d = sc[sc.direction == "coin"]
    for (B, period, model), g in d.groupby(["block_hours", "period", "model"]):
        g = g.set_index("leverage_scenario")
        cells = [f"{es(g.loc[lv].net_profit_usd, 0)} / {f2(g.loc[lv].sharpe_daily_sqrt365) if g.loc[lv].valid else 'inválida'}"
                 for lv in (10.0, 50.0, 100.0)]
        md.append(f"| {B} | {period} | {model} | " + " | ".join(cells) + " |")
    md.append("")
    if ev is not None:
        md.append("## Descriptive event diagnostic (calendar fixture period 2020-01 .. 2021-04; TRAIN; no decision)\n")
        md.append("| B | Blocks | n | Mean range (pips) | HAR range MAE ×MEAN20 |")
        md.append("|---|---|---:|---:|---:|")
        for _, r in ev.iterrows():
            md.append(f"| {r.block_hours} | {r.blocks} | {r.n} | {es(r.mean_range_pips, 1)} | {es(r.har_mae_r_ratio_mean20, 3)} |")
        md.append("")
    return "\n".join(md) + "\n"


def build(run: Path, vrun: Path, pub: Path):
    pub.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(run / "scores.csv", dtype={"period": str})
    vmain = pd.read_csv(vrun / "cells_merged.csv")
    vinner = pd.concat([pd.read_csv(p) for p in sorted(vrun.glob("cells_inner_*.csv"))], ignore_index=True)
    vcells = pd.concat([vmain[vmain.strategy == "RFM_COIN"], vinner], ignore_index=True)
    choices = {B: json.loads((run / f"B{B}" / "CHOICE.json").read_text()) for B in (24, 6)
               if (run / f"B{B}" / "CHOICE.json").exists()}
    ev = pd.read_csv(run / "event_diagnostic.csv") if (run / "event_diagnostic.csv").exists() else None
    inv = json.loads((run / "INVENTORY.json").read_text())
    stems = [figure(sc, vcells, B, pub) for B in (24, 6)]
    (pub / "TABLAS_CALENDARIO_RANGO.md").write_text(tables(sc, choices, ev, inv))
    sc.to_csv(pub / "scores.csv", index=False, float_format="%.8g")
    for name in ("INVENTORY.json", "fixture_high_impact_by_month.csv", "fixture_high_impact_by_family_usd_eur.csv",
                 "event_diagnostic.csv"):
        if (run / name).exists():
            shutil.copy2(run / name, pub / name)
    for B in (24, 6):
        if (run / f"B{B}" / "CHOICE.json").exists():
            shutil.copy2(run / f"B{B}" / "CHOICE.json", pub / f"CHOICE_B{B}.json")
    seals = {str(p.relative_to(run)): json.loads(p.read_text())["sha256"] for p in sorted(run.glob("B*/*/*_seal/SEAL.json"))}
    (pub / "SEALS.json").write_text(json.dumps(seals, indent=1) + "\n")
    print(json.dumps({"figures": stems, "rows": len(sc)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--vstudy", type=Path, required=True)
    ap.add_argument("--publish", type=Path, default=ROOT / "docs/short_horizon/results/calendar_range")
    a = ap.parse_args()
    build(a.run, a.vstudy, a.publish)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sharpe / net profit / trades versus the MAE of the predicted price path (owner style).

Reads only <run>/cells.csv written by ``tools/short_horizon_oracle.py run --grid pathx``.
Spanish labels with decimal comma, mean of seeds as a line, ±1 SD band plus thin seed
lines, dashed Naïve lines with their MAE, ideal annotation, first sign change marked
and a text box with the run conditions. DEVELOPMENT oracle/noise study, not a model.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.analysis import zero_crossing  # noqa: E402
from tools.short_horizon_oracle import LABEL, PATH_SIGMAS  # noqa: E402

TEAL, BLUE = "#237a75", "#1f4e9c"            # RF-M solid, RF-L dashed
PURPLE, RUST, INK, MUTED, GRID = "#5f5a8c", "#b9562e", "#1d2b36", "#52514e", "#e8e8e6"
STYLE = {"RF-M": (TEAL, "-"), "RF-L": (BLUE, "--")}
AXES = {24: (6, 24), 6: (3, 6)}              # primary and second horizon per block size
SPLIT_ES = {"VALIDATION_2024": ("VALIDACIÓN 2024", "01/01/2024 - 31/12/2024", "validacion"),
            "TRAIN_2020_2023": ("ENTRENAMIENTO 2020-2023", "01/01/2020 - 31/12/2023", "entrenamiento")}
BLOCK_ES = {24: "día hábil (B = 24 h)", 6: "bloques de 6 h (B = 6)"}
BLOCK_TITLE = {24: "Día hábil (B = 24 h)", 6: "Bloques de 6 h (B = 6)"}
TABLE_H = (1, 2, 3, 6, 12, 24)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def es(v, nd=4):
    """Decimal comma, dot thousands."""
    s = f"{v:,.{nd}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def stats(df, B, split, cost, strategy):
    d = df[(df.block_hours == B) & (df.split == split) & (df.cost_mult == cost) & (df.strategy == strategy)]
    g = d.groupby("level")
    cols = {"sharpe": "sharpe_daily_sqrt365", "net": "net_profit_usd", "trades": "trades",
            "dd": "max_drawdown_usd"}
    out = pd.concat({k: g[c].agg(["mean", "std", "min", "max"]) for k, c in cols.items()}, axis=1)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    seeds = {k: d.pivot_table(index="level", columns="seed", values=c) for k, c in cols.items()}
    return out.sort_index(), seeds


def mae_table(df, B, split):
    """Mean achieved MAE per horizon and per sigma (identical predictions for RF-M and RF-L)."""
    d = df[(df.block_hours == B) & (df.split == split) & (df.strategy == "RF-M") & (df.cost_mult == 1.0)]
    hs = [h for h in TABLE_H if h <= B]
    g = d.groupby("level")
    t = pd.DataFrame({f"mae_h{h}": g[f"path_mae_h{h}"].mean() for h in hs})
    t["mae_mean"] = g["path_mae"].mean()
    naive = {f"h{h}": float(d[f"naive_mae_h{h}"].iloc[0]) for h in hs}
    naive["mean"] = float(d["naive_path_mae"].iloc[0])
    for h in hs:
        t[f"ratio_h{h}"] = t[f"mae_h{h}"] / naive[f"h{h}"]
    t["ratio_mean"] = t["mae_mean"] / naive["mean"]
    return t.sort_index(), naive


def monotone_check(t, cols):
    """Each axis must increase with sigma, so every axis is a monotone function of the primary.

    Returns the largest decrease found along sigma (0 = monotone) and the largest relative
    deviation of each horizon's MAE from the primary one (sigma > 0).
    """
    out = {}
    base = t[cols[0]]
    for c in cols:
        v = t[c].to_numpy()
        out[f"max_decrease_{c}"] = float(max(0.0, np.max(np.maximum.accumulate(v) - v)))
        nz = base > 0
        out[f"max_rel_dev_{c}_vs_{cols[0]}"] = float(np.max(np.abs(t[c][nz] / base[nz] - 1))) if nz.any() else 0.0
    out["monotone"] = all(out[f"max_decrease_{c}"] == 0.0 for c in cols)
    return out


def figure(df, B, split, metric, outdir, cost=1.0):
    """x is drawn in sqrt(MAE) so the region around the naive is readable; every axis
    gets explicit ticks at exact MAE values (no scale composition)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    hp, hs = AXES[B]
    t, naive = mae_table(df, B, split)
    pcol, scol = f"mae_h{hp}", f"mae_h{hs}"
    check = monotone_check(t, [pcol, scol, "mae_mean"])
    xp = t[pcol].to_numpy()
    u = np.sqrt(xp)                                   # plotting coordinate
    split_es, period, _ = SPLIT_ES[split]
    ylabel = {"sharpe": "Sharpe anualizado (retornos diarios; 365 días)", "net": "Beneficio neto (USD)",
              "trades": "Operaciones por corrida"}[metric]
    title = {"sharpe": "Sharpe frente al error de las predicciones de precio",
             "net": "Beneficio frente al error de las predicciones de precio",
             "trades": "Operaciones frente al error de las predicciones de precio"}[metric]
    fig, ax = plt.subplots(figsize=(14, 9.6), facecolor="white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axhline(0, color="#6b7378", linewidth=1.2)
    ideal, crossings = {}, {}
    for st in ("RF-M", "RF-L"):
        col, ls = STYLE[st]
        s, seeds = stats(df, B, split, cost, st)
        s = s.loc[t.index]
        m, sd = s[f"{metric}_mean"].to_numpy(), np.nan_to_num(s[f"{metric}_std"].to_numpy())
        ax.fill_between(u, m - sd, m + sd, color=col, alpha=0.12, linewidth=0,
                        label=f"{st}: dispersión entre 5 semillas (±1 DE)")
        for seed in seeds[metric].columns:
            ax.plot(u, seeds[metric].loc[t.index, seed], color=col, alpha=0.22, linewidth=0.8, linestyle=ls)
        ax.plot(u, m, color=col, linestyle=ls, linewidth=2.6, marker="o", markersize=5,
                label=f"{st}: media de 5 semillas" + (" (entrada a mercado)" if st == "RF-M" else " (orden límite)"))
        ideal[st] = m[0]
        if metric != "trades":
            pos = np.flatnonzero(m <= 0)
            if len(pos) and pos[0] > 0:
                i = pos[0]
                ax.axvspan(u[i - 1], u[i], color=col, alpha=0.10, linewidth=0)
                crossings[st] = zero_crossing(xp, m)
            else:
                crossings[st] = None
    ax.axhline(0, color=PURPLE, linestyle=":", linewidth=1.4,
               label="Estrategia con predicciones de persistencia (sin operaciones)")
    inv_s = lambda v: np.interp(v, t[scol].to_numpy(), xp)        # noqa: E731
    inv_m = lambda v: np.interp(v, t["mae_mean"].to_numpy(), xp)  # noqa: E731
    for k, (h, colr, xpos) in enumerate(((hp, PURPLE, naive[f"h{hp}"]), (hs, RUST, float(inv_s(naive[f"h{hs}"]))))):
        ax.axvline(np.sqrt(xpos), color=colr, linestyle="--", linewidth=2.4,
                   label=f"Naïve h = {h} (persistencia): igualdad de MAE")
        ax.text(np.sqrt(xpos), 0.985 - 0.11 * k, f"  Naïve h = {h}\n  MAE = {es(naive[f'h{h}'], 5)}",
                transform=ax.get_xaxis_transform(), color=colr, fontsize=13, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))
    if metric != "trades":
        unit = "Sharpe" if metric == "sharpe" else "USD"
        txt = "Predicción ideal sin ruido\n" + "\n".join(f"{st}: {es(ideal[st], 2)} {unit}" for st in ("RF-M", "RF-L"))
        ax.annotate(txt, xy=(u[0], ideal["RF-M"]), xytext=(0.30, 0.74), textcoords="axes fraction", fontsize=12.5,
                    color=INK, va="top", arrowprops=dict(arrowstyle="->", color=RUST, lw=1.5))
        k = 0
        for st in ("RF-M", "RF-L"):
            col = STYLE[st][0]
            if crossings.get(st) is not None:
                ax.annotate(f"Primer cambio de signo observado ({st})", xy=(np.sqrt(crossings[st]), 0),
                            xytext=(0.50, 0.30 - 0.08 * k), textcoords="axes fraction", fontsize=11.5, color=INK,
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
            else:
                ax.text(0.50, 0.30 - 0.08 * k, f"{st}: sin cambio de signo en la rejilla (abstención)",
                        transform=ax.transAxes, fontsize=11.5, color=INK)
            k += 1
    # three stacked x axes with exact ticks
    cand = np.array([0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05])
    top_x = xp.max()
    nv = {"p": naive[f"h{hp}"], "s": naive[f"h{hs}"], "m": naive["mean"]}
    ticks_p = cand[cand <= top_x * 1.001]
    ax.set_xticks(np.sqrt(ticks_p), [f"{es(v, 4)}\n(×{es(v / nv['p'], 1)})" for v in ticks_p])
    ax.set_xlim(-0.004, np.sqrt(top_x) * 1.02)
    ax.set_xlabel(f"MAE de la predicción de precio a h = {hp} (USD/EUR)  ·  entre paréntesis: × naïve h = {hp}",
                  fontsize=12, color=INK)
    if check["monotone"]:
        for k, (inv, col, key, lab) in enumerate(((inv_s, scol, "s", f"MAE a h = {hs} (USD/EUR) (× naïve h = {hs})"),
                                                   (inv_m, "mae_mean", "m", f"MAE medio h = 1..{B} (USD/EUR) (× naïve medio)")),
                                                  start=1):
            sec = ax.secondary_xaxis("bottom", functions=(lambda v: v, lambda v: v))
            sec.spines["bottom"].set_position(("outward", 64 * k))
            vals = cand[cand <= t[col].max() * 1.001]
            sec.set_xticks(np.sqrt(inv(vals)), [f"{es(v, 4)}\n(×{es(v / nv[key], 1)})" for v in vals])
            sec.tick_params(labelsize=8.5, colors=MUTED)
            sec.set_xlabel(lab, fontsize=11, color=MUTED)
    ax.tick_params(labelsize=9.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: es(v, 0 if metric != "sharpe" else 1)))
    ax.set_ylabel(ylabel, fontsize=13, color=INK)
    n_blocks = int(df[(df.block_hours == B) & (df.split == split)].evaluated_blocks.iloc[0])
    tr = df[(df.block_hours == B) & (df.split == split) & (df.cost_mult == cost) & df.strategy.isin(["RF-M", "RF-L"])].trades
    box = "\n".join([period, f"{n_blocks} orígenes ({BLOCK_ES[B]})", "Capital inicial: 10.000 USD",
                     "Riesgo 1% por operación; nocional máx. 5×", f"{int(tr.min())}-{int(tr.max())} operaciones por corrida",
                     "TP/SL intravela; stop primero si ambos", "Spread: 2 pips (½ por lado)",
                     "Deslizamiento: 1 pip/lado", "Comisión: 3,5 USD/lote/lado",
                     "Swap: 10 USD/lote/rollover; plano al cierre", f"Costes ×{es(cost, 0)}; supuestos, no verificados",
                     "Ejes x en escala raíz cuadrada"])
    ax.text(0.985, 0.975, box, transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=INK,
            multialignment="left", bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#b8c4cc"))
    if metric == "trades":
        ax.text(0.30, 0.30, "Nota: el máximo y el mínimo de una trayectoria ruidosa están sesgados hacia afuera;\n"
                "las órdenes límite de RF-L dejan de ejecutarse (abstención). Con rangos muy amplios el\n"
                "dimensionamiento por riesgo redondea las posiciones de RF-M a cero.",
                transform=ax.transAxes, ha="left", va="top", fontsize=10.5, color=INK, multialignment="left")
    if metric == "trades":
        ax.legend(fontsize=9.5, loc="upper left", bbox_to_anchor=(0.30, 0.78), frameon=True, edgecolor="#d0d0d0")
    else:
        ax.legend(fontsize=9.5, loc="upper right", bbox_to_anchor=(1.0, 0.56), frameon=True, edgecolor="#d0d0d0")
    fig.suptitle(f"{title}\n{BLOCK_TITLE[B]} · {split_es} · DESARROLLO · estudio oráculo/ruido, "
                 "no es un resultado de modelo", fontsize=16, color=INK)
    fig.tight_layout()
    stem = f"barrido_trayectoria_{metric}_B{B}_{SPLIT_ES[split][2]}"
    fig.savefig(outdir / f"{stem}.png", dpi=150)
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)
    return stem, check, crossings


def break_even(df, B, split):
    hp, hs = AXES[B]
    t, naive = mae_table(df, B, split)
    rows = []
    for st in ("RF-M", "RF-L"):
        for cost in (1.0, 2.0):
            s, seeds = stats(df, B, split, cost, st)
            s = s.loc[t.index]
            row = {"block_hours": B, "split": split, "strategy": st, "cost_mult": cost}
            for key, col in (("net_mean", "net_mean"), ("net_all_seeds", "net_min"), ("sharpe_mean", "sharpe_mean")):
                sig = zero_crossing(t.index.to_numpy(), s[col].to_numpy())
                row[f"{key}_sigma"] = sig
                if sig is not None:
                    for name, c, nvk in ((f"h{hp}", f"mae_h{hp}", f"h{hp}"), (f"h{hs}", f"mae_h{hs}", f"h{hs}"),
                                         ("mean", "mae_mean", "mean")):
                        v = float(np.interp(sig, t.index.to_numpy(), t[c].to_numpy()))
                        row[f"{key}_mae_{name}"] = v
                        row[f"{key}_ratio_{name}"] = v / naive[nvk]
            row["trades_at_last_level"] = float(s["trades_mean"].iloc[-1])
            row["net_at_last_level"] = float(s["net_mean"].iloc[-1])
            rows.append(row)
    return rows


def level_table(df, B, split):
    t, naive = mae_table(df, B, split)
    out = t.copy()
    out.insert(0, "sigma", out.index)
    for st in ("RF-M", "RF-L"):
        for cost in (1.0, 2.0):
            s, _ = stats(df, B, split, cost, st)
            tag = f"{st}_x{cost:g}"
            for k in ("sharpe", "net", "trades", "dd"):
                out[f"{tag}_{k}"] = s.loc[t.index, f"{k}_mean"].to_numpy()
    return out.reset_index(drop=True), naive


def markdown_tables(df, splits):
    lines = ["# Tablas por nivel de ruido — trayectoria de precios (generado)\n",
             f"Etiqueta `{LABEL}`. Medias de 5 semillas. MAE en USD/EUR; ratio = MAE / MAE del naïve "
             "(persistencia) en el mismo horizonte y las mismas filas. Generado por "
             "`tools/short_horizon_price_path_report.py`.\n"]
    for B in (24, 6):
        for split in splits:
            tab, naive = level_table(df, B, split)
            hs = [h for h in TABLE_H if h <= B]
            lines.append(f"## {BLOCK_ES[B]} · {SPLIT_ES[split][0]}\n")
            lines.append("Naïve (persistencia): " + ", ".join(f"h{h} {es(naive[f'h{h}'], 5)}" for h in hs) +
                         f", medio {es(naive['mean'], 5)}.\n")
            head = ["σ"] + [f"MAE h{h}" for h in hs] + ["MAE medio"] + [f"×naïve h{h}" for h in hs] + ["×naïve medio"]
            lines.append("| " + " | ".join(head) + " |")
            lines.append("|" + "---:|" * len(head))
            for _, r in tab.iterrows():
                lines.append("| " + " | ".join([es(r.sigma, 5)] + [es(r[f"mae_h{h}"], 5) for h in hs] +
                                                [es(r.mae_mean, 5)] + [es(r[f"ratio_h{h}"], 2) for h in hs] +
                                                [es(r.ratio_mean, 2)]) + " |")
            lines.append("")
            head = ["σ"] + [f"{st} ×{c} {k}" for st in ("RF-M", "RF-L") for c in (1, 2)
                            for k in ("Sharpe", "neto USD", "oper.", "DD USD")]
            lines.append("| " + " | ".join(head) + " |")
            lines.append("|" + "---:|" * len(head))
            for _, r in tab.iterrows():
                cells = [es(r.sigma, 5)]
                for st in ("RF-M", "RF-L"):
                    for c in (1, 2):
                        tag = f"{st}_x{c}"
                        cells += [es(r[f"{tag}_sharpe"], 2), es(r[f"{tag}_net"], 0), es(r[f"{tag}_trades"], 0),
                                  es(r[f"{tag}_dd"], 0)]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def build(run: Path, pub: Path):
    df = pd.read_csv(run / "cells.csv")
    if set(df.label) != {LABEL} or set(df.grid) != {"pathx"}:
        raise SystemExit("unexpected population")
    pub.mkdir(parents=True, exist_ok=True)
    figures, checks, be = [], {}, []
    for B in (24, 6):
        for split in ("VALIDATION_2024", "TRAIN_2020_2023"):
            for metric in ("sharpe", "net", "trades"):
                stem, check, _ = figure(df, B, split, metric, pub)
                figures.append(stem)
                checks[f"B{B}_{split}"] = check
            be += break_even(df, B, split)
    tables = []
    for B in (24, 6):
        for split in ("VALIDATION_2024", "TRAIN_2020_2023"):
            tab, _ = level_table(df, B, split)
            tables.append(tab.assign(block_hours=B, split=split))
    pd.concat(tables).to_csv(pub / "tabla_por_nivel_de_ruido.csv", index=False, float_format="%.8g")
    (pub / "TABLAS_POR_NIVEL.md").write_text(markdown_tables(df, ("VALIDATION_2024", "TRAIN_2020_2023")))
    (pub / "break_even_trayectoria.json").write_text(json.dumps(be, indent=1, default=float) + "\n")
    (pub / "verificacion_ejes_monotonos.json").write_text(json.dumps(checks, indent=1) + "\n")
    with open(run / "cells.csv", "rb") as src, open(pub / "pathx_cells.csv.gz", "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0, filename="") as dst:
        shutil.copyfileobj(src, dst)
    receipts = sorted(run.glob("receipt_pathx_*.json"))
    manifest = {"label": LABEL, "cells_csv_sha256": sha(run / "cells.csv"), "rows": int(len(df)),
                "data_sha256": json.loads(receipts[0].read_text())["data_sha256"],
                "backtest_cpu_seconds": round(sum(json.loads(p.read_text())["cpu_seconds"] for p in receipts), 1),
                "chunks": len(receipts), "sigmas": list(PATH_SIGMAS), "figures": figures, "test_rows_read": False}
    (pub / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"checks": checks, "break_even": be}, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--publish", type=Path, default=ROOT / "docs/short_horizon/results/price_path")
    a = ap.parse_args()
    build(a.run, a.publish)


if __name__ == "__main__":
    main()

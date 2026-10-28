#!/usr/bin/env python3
"""Realism pass figures and tables: model-like path errors x realistic execution, frequency rule.

Reads only <run>/cells.csv of ``tools/short_horizon_oracle.py run --grid pathr``. Spanish
labels, decimal comma. Three panels per figure, one per consumed error (MAE at h_p, at h_s
and the mean over the path): with several error families the horizon-to-horizon mapping
differs per family, so stacked shared axes would be wrong. Owner frequency rule: a seed is
valid with >= 3 trades per business week (variant 4) and <= 4 orders per business day
(variant 2, enforced in the strategy); a level needs >= 3 valid seeds; invalid points are
drawn hollow grey and never define a sign change or a break-even.
DEVELOPMENT oracle/noise study, not a model result.
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
from app.short_horizon.analysis import frequency_validity, zero_crossing  # noqa: E402
from tools.short_horizon_oracle import A3_PAIRS, A1_EPS, LABEL  # noqa: E402
from tools.short_horizon_price_path_report import (AXES, BLOCK_TITLE, BLOCK_ES, SPLIT_ES, INK, MUTED,  # noqa: E402
                                                   GRID, PURPLE, RUST, TEAL, es)

CURVES = [  # (key, sweep, family, label, colour, linestyle, width)
    ("legacy_iid", "legacy", "iid", "i.i.d., costes anteriores ×1 (referencia)", "#9a9a96", ":", 1.6),
    ("iid", "realistic", "iid", "i.i.d. — cota superior optimista", "#6b7378", "--", 1.8),
    ("a1", "realistic", "a1", "A1: mezcla hacia persistencia", PURPLE, "-", 1.8),
    ("a2", "realistic", "a2", "A2: error de paseo aleatorio en h", RUST, "-", 1.8),
    ("a3", "realistic", "a3", "A3: A1 + A2 (realista, para citar)", TEAL, "-", 3.0),
]
MIN_WEEK, MIN_WEEK_STRICT, MAX_DAY, MAX_DAY_STRICT = 3.0, 4.0, 4, 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve(df, B, split, strategy, sweep, family, cost, min_week=MIN_WEEK, max_day=MAX_DAY):
    d = df[(df.block_hours == B) & (df.split == split) & (df.strategy == strategy) & (df.sweep == sweep)
           & (df.family == family) & (df.cost_mult == cost)].copy()
    if d.empty:
        return None
    ok, lv = frequency_validity(d, min_per_week=min_week, max_per_day=max_day)
    d["valid_seed"] = ok
    hp, hs = AXES[B]
    rows = []
    for level, g in d.groupby("level"):
        gv = g[g.valid_seed]
        rows.append({"level": level, "valid": bool(lv[level]), "n_valid": int(len(gv)),
                     "mae_p": g[f"path_mae_h{hp}"].mean(), "mae_s": g[f"path_mae_h{hs}"].mean(),
                     "mae_m": g["path_mae"].mean(),
                     "mae_h": {h: g[f"path_mae_h{h}"].mean() for h in (1, 2, 3, 6, 12, 24) if h <= B},
                     "sharpe": gv.sharpe_daily_sqrt365.mean() if len(gv) else np.nan,
                     "sharpe_sd": gv.sharpe_daily_sqrt365.std() if len(gv) > 1 else 0.0,
                     "sharpe_all": g.sharpe_daily_sqrt365.mean(),
                     "net": gv.net_profit_usd.mean() if len(gv) else np.nan,
                     "net_sd": gv.net_profit_usd.std() if len(gv) > 1 else 0.0,
                     "net_all": g.net_profit_usd.mean(), "net_min": gv.net_profit_usd.min() if len(gv) else np.nan,
                     "trades": g.trades.mean(), "trades_sd": g.trades.std() if len(g) > 1 else 0.0,
                     "trades_all": g.trades.mean(),
                     "tpw": g.trades_per_week.mean(), "tpw_min": g.trades_per_week.min(),
                     "max_day": int(g.max_orders_per_day.max()), "dd": gv.max_drawdown_usd.mean() if len(gv) else np.nan})
    c = pd.DataFrame(rows).sort_values("mae_p").reset_index(drop=True)
    nv = {k: float(d[f"naive_mae_h{h}"].iloc[0]) for k, h in (("p", hp), ("s", hs))}
    nv["m"] = float(d["naive_path_mae"].iloc[0])
    return c, nv


def valid_prefix(c):
    """Levels in MAE order up to (excluding) the first invalid one: the only part a result may use."""
    bad = np.flatnonzero(~c.valid.to_numpy())
    return c.iloc[: bad[0]] if len(bad) else c


def break_even(c, key="net"):
    pre = valid_prefix(c)
    out = {"validity_frontier_mae_p": float(c.mae_p.iloc[len(pre)]) if len(pre) < len(c) else None}
    if len(pre) < 2:
        out.update({"mae_p": None, "reason": "fewer than two valid levels"})
        return out
    x = zero_crossing(pre.mae_p, pre[key])
    out["mae_p"] = x
    if x is not None:
        out["mae_s"] = float(np.interp(x, pre.mae_p, pre.mae_s))
        out["mae_m"] = float(np.interp(x, pre.mae_p, pre.mae_m))
        out["tpw_at"] = float(np.interp(x, pre.mae_p, pre.tpw))
    else:
        out["reason"] = "no sign change while valid"
    allseeds = zero_crossing(pre.mae_p, pre.net_min) if key == "net" else None
    out["all_seeds_mae_p"] = allseeds
    return out


MARKER_SHAPE = {"path": "X", "PERSIST": "v", "MEAN20": "s", "EWMA": "D", "HAR": "*"}
MARKER_COLOUR = {"path": "#6b7378", "BASE": "#1d2b36", "LOGIT": "#9c2f7a"}


def draw_markers(axes, markers, B, split, strategy, metric):
    """Sealed baseline predictors (weekly refit + persistence path) at their achieved path MAE."""
    if markers is None:
        return
    mk = markers[(markers.block_hours == B) & (markers.split == split) & (markers.strategy == strategy)
                 & (markers.cost_mult == 1.0) & markers.cadence.isin(["weekly", "n/a"])]
    ycol = {"sharpe": "sharpe_daily_sqrt365", "net": "net_profit_usd", "trades": "trades"}[metric]
    for ax, xcol in zip(axes, ("path_mae_hp", "path_mae_hs", "path_mae_mean")):
        for _, r in mk.iterrows():
            colour = MARKER_COLOUR[r.order]
            ax.scatter([r[xcol]], [r[ycol]], marker=MARKER_SHAPE[r.excursion], s=120 if r.excursion == "HAR" else 70,
                       facecolors=colour if r.valid else "none", edgecolors=colour, linewidths=1.4, zorder=5,
                       label=(f"Predictor {r.predictor}" + (" (semanal)" if r.cadence == "weekly" else "")
                              + ("" if r.valid else " — inválido"))
                       if ax is axes[0] else None)


def figure(df, B, split, strategy, metric, outdir, markers=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    hp, hs = AXES[B]
    split_es, period, tag = SPLIT_ES[split]
    fig, axes = plt.subplots(1, 3, figsize=(25, 9.8 if markers is not None else 8.6), sharey=True, facecolor="white")
    ylabel = {"sharpe": "Sharpe anualizado (retornos diarios; 365 días)", "net": "Beneficio neto (USD)",
              "trades": "Operaciones por corrida"}[metric]
    xcols = (("mae_p", "p", f"MAE a h = {hp} (USD/EUR)"), ("mae_s", "s", f"MAE a h = {hs} (USD/EUR)"),
             ("mae_m", "m", f"MAE medio h = 1..{B} (USD/EUR)"))
    invalid_labelled = False
    curves = {}
    for key, sweep, fam, label, col, ls, lw in CURVES:
        r = curve(df, B, split, strategy, sweep, fam, 1.0)
        if r is not None:
            curves[key] = r
    for ax, (xc, nk, xl) in zip(axes, xcols):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.axhline(0, color="#6b7378", linewidth=1.1)
        for key, sweep, fam, label, col, ls, lw in CURVES:
            if key not in curves:
                continue
            c, nv = curves[key]
            u = c[xc].to_numpy()
            y = c[metric].to_numpy() if metric != "trades" else c.trades_all.to_numpy()
            valid = c.valid.to_numpy()
            yv = np.where(valid, y, np.nan)
            ax.plot(u, yv, color=col, linestyle=ls, linewidth=lw, marker="o", markersize=4 if lw < 2.5 else 6,
                    label=label if ax is axes[0] else None)
            if key == "a3":
                sd = c[f"{metric}_sd"].to_numpy() if metric != "trades" else c.trades_sd.to_numpy()
                ax.fill_between(u, yv - sd, yv + sd, color=col, alpha=0.13, linewidth=0,
                                label="A3: dispersión entre semillas válidas (±1 DE)" if ax is axes[0] else None)
            inv = ~valid
            if inv.any():
                yi = c.sharpe_all if metric == "sharpe" else (c.net_all if metric == "net" else c.trades_all)
                ax.scatter(u[inv], yi.to_numpy()[inv], facecolors="none", edgecolors="#9a9a96", s=34, linewidths=1.2,
                           label=("inválida: < 3 operaciones/semana" if not invalid_labelled and ax is axes[0] else None),
                           zorder=3)
                invalid_labelled = invalid_labelled or ax is axes[0]
            if metric != "trades" and key in ("a3", "iid"):
                pre = valid_prefix(c)
                ym = pre[metric].to_numpy()
                pos = np.flatnonzero(ym <= 0)
                if len(pos) and pos[0] > 0:
                    i = pos[0]
                    ax.axvspan(pre[xc].iloc[i - 1], pre[xc].iloc[i], color=col, alpha=0.10, linewidth=0)
                    if ax is axes[0]:
                        x0 = zero_crossing(pre[xc], pre[metric])
                        ax.annotate(f"Primer cambio de signo observado ({'A3' if key == 'a3' else 'i.i.d.'})",
                                    xy=(x0, 0), xytext=(0.42, 0.30 if key == "a3" else 0.20),
                                    textcoords="axes fraction", fontsize=10.5, color=INK,
                                    arrowprops=dict(arrowstyle="->", color=col, lw=1.3))
        c0, nv = curves["a3"]
        ax.axvline(nv[nk], color=RUST if nk == "s" else PURPLE, linestyle="--", linewidth=2.2,
                   label="Naïve (persistencia): igualdad de MAE" if ax is axes[0] else None)
        ax.text(nv[nk], 0.985, f"  Naïve\n  MAE = {es(nv[nk], 5)}", transform=ax.get_xaxis_transform(),
                color=RUST if nk == "s" else PURPLE, fontsize=11.5, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))
        # x range = the model-like families (A1, A2, A3); the i.i.d. tails continue off-range
        xmax = max(curves[k][0][xc].max() for k in ("a1", "a2", "a3") if k in curves) * 1.06
        ax.set_xlim(-0.02 * xmax, xmax)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _, n=nv[nk]: f"{es(v, 4)}\n(×{es(v / n, 1)})"))
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_xlabel(f"{xl}\nentre paréntesis: × naïve del mismo horizonte", fontsize=11, color=INK)
    draw_markers(axes, markers, B, split, strategy, metric)
    axes[0].set_ylabel(ylabel, fontsize=12.5, color=INK)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: es(v, 0 if metric != "sharpe" else 1)))
    if metric != "trades":
        c, _ = curves["a3"]
        unit = "Sharpe" if metric == "sharpe" else "USD"
        old = curves["legacy_iid"][0]
        axes[0].annotate(f"Predicción ideal sin ruido\n{strategy}, costes realistas: {es(c[metric].iloc[0], 2)} {unit}\n"
                         f"(costes anteriores: {es(old[metric].iloc[0], 2)} {unit})",
                         xy=(0, c[metric].iloc[0]), xytext=(0.30, 0.86), textcoords="axes fraction", fontsize=11,
                         color=INK, va="top", arrowprops=dict(arrowstyle="->", color=RUST, lw=1.4))
    d = df[(df.block_hours == B) & (df.split == split) & (df.strategy == strategy) & (df.sweep == "realistic")
           & (df.cost_mult == 1.0)]
    box = "\n".join([period, f"{int(d.eligible_blocks.iloc[0])} orígenes; {int(d.weeks_with_eligible_blocks.iloc[0])} semanas",
                     f"{BLOCK_TITLE[B]}; estrategia {strategy}",
                     "Capital inicial: 10.000 USD", "Riesgo 1%; nocional máx. 5×",
                     "Válido: ≥ 3 oper./semana, ≤ 4 órdenes/día",
                     "Latencia: 1 barra", "Límites: cruce ≥ 1 pip",
                     "Spread: 0,3 pip (núcleo) a 3 pips (rollover)",
                     "Deslizamiento: 0,2 pip mercado; 0,5 pip stop",
                     "Comisión: 3,5 USD/lote/lado",
                     "Swap: largo −7, corto +2,5 USD/lote/noche",
                     "Swap ×3 en el tercer rollover de la semana",
                     "TP/SL intravela; stop primero si ambos",
                     "Costes ×1; supuestos, no verificados con bróker",
                     "Eje x: rango de A1-A2-A3; las colas i.i.d.",
                     "continúan fuera (ver figuras i.i.d.)"])
    fig.text(0.815, 0.88, box, ha="left", va="top", fontsize=10, color=INK, multialignment="left",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#b8c4cc"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5 if markers is not None else 10, loc="upper left",
               bbox_to_anchor=(0.815, 0.42), frameon=True, edgecolor="#d0d0d0")
    title = {"sharpe": "Sharpe frente al error de las predicciones de precio — modelo de error y costes realistas",
             "net": "Beneficio frente al error de las predicciones de precio — modelo de error y costes realistas",
             "trades": "Operaciones frente al error de las predicciones de precio — modelo de error y costes realistas"}[metric]
    fig.suptitle(f"{title}\n{BLOCK_TITLE[B]} · {strategy} · {split_es} · DESARROLLO · estudio oráculo/ruido, "
                 "no es un resultado de modelo", fontsize=15, color=INK)
    fig.tight_layout(rect=(0, 0, 0.805, 1))
    stem = f"barrido_trayectoria_realista_{metric}_B{B}_{tag}_{strategy.replace('-', '')}"
    fig.savefig(outdir / f"{stem}.png", dpi=130)
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)
    return stem


def comparison(df):
    """Owner table: ideal Sharpe and break-even under (i) old costs/i.i.d., (ii) realistic/i.i.d.,
    (iii) realistic/A3, plus A1 and A2, both frequency variants, x1 and x2."""
    rows = []
    for B in (24, 6):
        hp, hs = AXES[B]
        for split in ("VALIDATION_2024", "TRAIN_2020_2023"):
            for st in ("RF-M", "RF-L"):
                for name, sweep, fam, cost in (("(i) costes anteriores / i.i.d.", "legacy", "iid", 1.0),
                                               ("(ii) realistas / i.i.d.", "realistic", "iid", 1.0),
                                               ("(iii) realistas / A3", "realistic", "a3", 1.0),
                                               ("realistas / A1", "realistic", "a1", 1.0),
                                               ("realistas / A2", "realistic", "a2", 1.0),
                                               ("(ii) realistas ×2 / i.i.d.", "realistic", "iid", 2.0),
                                               ("(iii) realistas ×2 / A3", "realistic", "a3", 2.0),
                                               ("realistas ×1 / A3, ≤ 2 órdenes/día", "realistic_cap2", "a3", 1.0)):
                    if sweep == "realistic_cap2" and B != 6:
                        continue
                    for min_week in (MIN_WEEK, MIN_WEEK_STRICT):
                        max_day = MAX_DAY_STRICT if sweep == "realistic_cap2" else MAX_DAY
                        r = curve(df, B, split, st, sweep, fam, cost, min_week=min_week, max_day=max_day)
                        if r is None:
                            continue
                        c, nv = r
                        be = break_even(c, "net")
                        bs = break_even(c, "sharpe")
                        ideal = c.iloc[0]
                        rows.append({"block_hours": B, "split": split, "strategy": st, "case": name,
                                     "min_trades_per_week": min_week, "max_orders_per_day": max_day,
                                     "ideal_sharpe": float(ideal.sharpe) if ideal.valid else None,
                                     "ideal_trades_per_week": float(ideal.tpw),
                                     "break_even_net_mae_p": be.get("mae_p"),
                                     "break_even_net_ratio_p": None if be.get("mae_p") is None else be["mae_p"] / nv["p"],
                                     "break_even_net_ratio_s": None if be.get("mae_p") is None else be["mae_s"] / nv["s"],
                                     "break_even_net_ratio_m": None if be.get("mae_p") is None else be["mae_m"] / nv["m"],
                                     "break_even_sharpe_ratio_p": None if bs.get("mae_p") is None else bs["mae_p"] / nv["p"],
                                     "all_seeds_ratio_p": None if be.get("all_seeds_mae_p") is None else be["all_seeds_mae_p"] / nv["p"],
                                     "validity_frontier_ratio_p": None if be.get("validity_frontier_mae_p") is None
                                     else be["validity_frontier_mae_p"] / nv["p"],
                                     "trades_per_week_at_break_even": be.get("tpw_at"),
                                     "reason": be.get("reason"), "h_p": hp, "h_s": hs})
    return rows


def a1_naive_point(df):
    out = []
    for B in (24, 6):
        for split in ("VALIDATION_2024", "TRAIN_2020_2023"):
            for st in ("RF-M", "RF-L"):
                for cost in (1.0, 2.0):
                    d = df[(df.block_hours == B) & (df.split == split) & (df.strategy == st) & (df.sweep == "realistic")
                           & (df.family == "a1") & (df.level == 0.0) & (df.cost_mult == cost)]
                    out.append({"block_hours": B, "split": split, "strategy": st, "cost_mult": cost,
                                "mae_ratio_mean": float(d.path_mae_ratio_to_naive.mean()),
                                "trades": float(d.trades.mean()), "trades_per_week": float(d.trades_per_week.mean()),
                                "net_usd": float(d.net_profit_usd.mean()), "sharpe": float(d.sharpe_daily_sqrt365.mean())})
    return out


def level_table(df):
    rows = []
    fams = {"legacy": ["iid"], "realistic": ["iid", "a1", "a2", "a3"], "realistic_cap2": ["iid", "a1", "a2", "a3"]}
    for B in (24, 6):
        for split in ("VALIDATION_2024", "TRAIN_2020_2023"):
            for sweep, fl in fams.items():
                if sweep == "realistic_cap2" and B != 6:
                    continue
                for fam in fl:
                    for st in ("RF-M", "RF-L"):
                        for cost in ((1.0, 2.0) if sweep == "realistic" else (1.0,)):
                            r = curve(df, B, split, st, sweep, fam, cost,
                                      max_day=MAX_DAY_STRICT if sweep == "realistic_cap2" else MAX_DAY)
                            if r is None:
                                continue
                            c, nv = r
                            for _, x in c.iterrows():
                                row = {"block_hours": B, "split": split, "profile": sweep, "family": fam, "strategy": st,
                                       "cost_mult": cost, "level": x.level,
                                       "a3_w": A3_PAIRS[int(x.level)][0] if fam == "a3" else None,
                                       "a3_step": A3_PAIRS[int(x.level)][1] if fam == "a3" else None,
                                       "valid_3_per_week": x.valid, "valid_seeds": x.n_valid,
                                       "orders_per_week_mean": x.tpw, "orders_per_week_min_seed": x.tpw_min,
                                       "max_orders_per_day": x.max_day, "mae_mean": x.mae_m,
                                       "ratio_mean": x.mae_m / nv["m"], "sharpe_valid": x.sharpe,
                                       "net_valid": x.net, "trades": x.trades_all, "max_dd_valid": x.dd}
                                for h, v in x.mae_h.items():
                                    row[f"mae_h{h}"] = v
                                    row[f"ratio_h{h}"] = v / float(df[(df.block_hours == B) & (df.split == split)][f"naive_mae_h{h}"].iloc[0])
                                rows.append(row)
    return pd.DataFrame(rows)


def markdown(cmp_rows, a1):
    def f(v, nd=2):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else es(v, nd)
    lines = ["# Tabla comparativa — realismo (generado)\n",
             f"Etiqueta `{LABEL}`. Break-even leído en la curva media de semillas válidas (beneficio neto), solo "
             "sobre niveles válidos y antes del primer nivel inválido. Ratios: MAE / naïve (persistencia) del mismo "
             "horizonte en las mismas filas. «—»: sin cambio de signo mientras la configuración es válida.\n"]
    for B in (24, 6):
        for split in ("VALIDATION_2024", "TRAIN_2020_2023"):
            for min_week in (MIN_WEEK, MIN_WEEK_STRICT):
                sub = [r for r in cmp_rows if r["block_hours"] == B and r["split"] == split
                       and r["min_trades_per_week"] == min_week]
                if not sub:
                    continue
                hp, hs = sub[0]["h_p"], sub[0]["h_s"]
                lines.append(f"## {BLOCK_TITLE[B]} · {SPLIT_ES[split][0]} · válido con ≥ {es(min_week, 0)} "
                             "operaciones/semana\n")
                lines.append(f"| Estrategia | Caso | Sharpe ideal | Break-even ×naïve h{hp} | ×naïve h{hs} | ×naïve medio "
                             f"| todas las semillas ×naïve h{hp} | frontera de validez ×naïve h{hp} | oper./semana en break-even |")
                lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
                for r in sub:
                    lines.append(f"| {r['strategy']} | {r['case']} | {f(r['ideal_sharpe'])} | "
                                 f"{f(r['break_even_net_ratio_p'])} | {f(r['break_even_net_ratio_s'])} | "
                                 f"{f(r['break_even_net_ratio_m'])} | {f(r['all_seeds_ratio_p'])} | "
                                 f"{f(r['validity_frontier_ratio_p'])} | {f(r['trades_per_week_at_break_even'])} |")
                lines.append("")
    lines.append("## A1 con w = 0 (naïve puro + ε de 0,5 pip)\n")
    lines.append("| B | Split | Estrategia | Costes | MAE/naïve | Operaciones | Oper./semana | Neto USD | Sharpe |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for r in a1:
        lines.append(f"| {r['block_hours']} | {SPLIT_ES[r['split']][0]} | {r['strategy']} | ×{es(r['cost_mult'], 0)} | "
                     f"{es(r['mae_ratio_mean'], 2)} | {es(r['trades'], 1)} | {es(r['trades_per_week'], 2)} | "
                     f"{es(r['net_usd'], 0)} | {es(r['sharpe'], 2)} |")
    return "\n".join(lines) + "\n"


def build(run: Path, pub: Path, markers_path=None):
    markers = pd.read_csv(markers_path, keep_default_na=False, na_values=[""]) if markers_path else None
    df = pd.read_csv(run / "cells.csv")
    if set(df.label) != {LABEL} or set(df.grid) != {"pathr"}:
        raise SystemExit("unexpected population")
    pub.mkdir(parents=True, exist_ok=True)
    stems = []
    for B in (24, 6):
        for split in ("VALIDATION_2024", "TRAIN_2020_2023"):
            for st in ("RF-M", "RF-L"):
                for metric in ("sharpe", "net", "trades"):
                    stems.append(figure(df, B, split, st, metric, pub, markers))
    cmp_rows = comparison(df)
    a1 = a1_naive_point(df)
    (pub / "comparacion_realismo.json").write_text(json.dumps({"comparison": cmp_rows, "a1_w0": a1}, indent=1,
                                                              default=float) + "\n")
    (pub / "TABLA_COMPARATIVA_REALISMO.md").write_text(markdown(cmp_rows, a1))
    level_table(df).to_csv(pub / "tabla_realismo_por_nivel.csv", index=False, float_format="%.8g")
    with open(run / "cells.csv", "rb") as src, open(pub / "pathr_cells.csv.gz", "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0, filename="") as dst:
        shutil.copyfileobj(src, dst)
    receipts = sorted(run.glob("receipt_pathr_*.json"))
    manifest = {"label": LABEL, "cells_csv_sha256": sha(run / "cells.csv"), "rows": int(len(df)),
                "data_sha256": json.loads(receipts[0].read_text())["data_sha256"],
                "backtest_cpu_seconds": round(sum(json.loads(p.read_text())["cpu_seconds"] for p in receipts), 1),
                "chunks": len(receipts), "a3_pairs": A3_PAIRS, "a1_eps": A1_EPS, "figures": stems,
                "frequency_rule": {"min_trades_per_week": [MIN_WEEK, MIN_WEEK_STRICT],
                                   "max_orders_per_day": [MAX_DAY, MAX_DAY_STRICT], "min_valid_seeds": 3},
                "test_rows_read": False}
    (pub / "MANIFEST_REALISMO.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"figures": len(stems)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--publish", type=Path, default=ROOT / "docs/short_horizon/results/price_path_realism")
    ap.add_argument("--markers", type=Path, help="baseline predictor results CSV (adapter scores)")
    a = ap.parse_args()
    build(a.run, a.publish, a.markers)


if __name__ == "__main__":
    main()

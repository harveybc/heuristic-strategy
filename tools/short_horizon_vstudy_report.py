#!/usr/bin/env python3
"""Figures and tables for the direction-free V1/V2 study (DEVELOPMENT oracle/noise study).

x axes: the error of the VOLATILITY predictor the strategies consume -- MAE of U_hat, D_hat and of
the range R_hat = U_hat + D_hat -- each labelled with its multiple of the MEAN20 naive; vertical
lines at the MEAN20 and persistence naive MAE. Frequency rule (>= 3 trades/week, variant 4;
<= 4 orders/day), ruin and >= 3 valid seeds applied before any mean, sign change or break-even.
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
from tools.short_horizon_price_path_report import INK, MUTED, GRID, PURPLE, RUST, TEAL, es  # noqa: E402

SPLIT_ES = {"VALIDATION_2024": ("VALIDACIÓN 2024", "validacion"),
            "TRAIN_2020_2023": ("ENTRENAMIENTO 2020-2023", "entrenamiento")}
STRAT_ES = {"V1": "V1 ruptura OCO + stop dinámico", "V2": "V2 reversión al ancla",
            "RFM_COIN": "RF-M, orden al azar", "RFL_COIN": "RF-L, orden al azar"}
FAMILIES = [("iid", "i.i.d. — cota superior optimista", "#6b7378", "--", 1.8),
            ("a1r", "A1r: mezcla hacia el naïve MEAN20", PURPLE, "-", 1.8),
            ("a3r", "A3r: A1r + error que decrece con la habilidad (realista)", TEAL, "-", 3.0)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validity(d, min_week=3.0, max_day=4):
    ok = (d.trades_per_week >= min_week) & (d.max_orders_per_day <= max_day) & (~d.ruined.astype(bool))
    return ok


def curve(df, B, split, strategy, family, lev=100.0, cost=1.0, min_week=3.0):
    d = df[(df.block_hours == B) & (df.split == split) & (df.strategy == strategy) & (df.family == family)
           & (df.leverage_scenario == lev) & (df.cost_mult == cost)].copy()
    if d.empty:
        return None
    d["valid_seed"] = validity(d, min_week)
    rows = []
    for level, g in d.groupby("level"):
        gv = g[g.valid_seed]
        rows.append({"level": level, "valid": len(gv) >= 3, "n_valid": len(gv),
                     "ru": (g.mae_u / g.naive_mae_u).mean(), "rd": (g.mae_d / g.naive_mae_d).mean(),
                     "rr": (g.mae_r / g.naive_mae_r).mean(), "rr_p": (g.mae_r / g.persist_mae_r).mean(),
                     "mae_u": g.mae_u.mean(), "mae_d": g.mae_d.mean(), "mae_r": g.mae_r.mean(),
                     "sharpe": gv.sharpe_daily_sqrt365.mean() if len(gv) else np.nan,
                     "sharpe_sd": gv.sharpe_daily_sqrt365.std() if len(gv) > 1 else 0.0,
                     "sharpe_all": g.sharpe_daily_sqrt365.mean(),
                     "net": gv.net_profit_usd.mean() if len(gv) else np.nan,
                     "net_min": gv.net_profit_usd.min() if len(gv) else np.nan,
                     "net_all": g.net_profit_usd.mean(),
                     "mult": (gv.end_equity_usd / 10_000).median() if len(gv) else np.nan,
                     "dd_pct": gv.max_drawdown_pct.mean() if len(gv) else np.nan,
                     "dd_pct_all": g.max_drawdown_pct.mean(),
                     "sortino": gv.sortino_daily_sqrt365.mean() if len(gv) else np.nan,
                     "worst_week": gv.worst_week_return.mean() if len(gv) else np.nan,
                     "tpw": g.trades_per_week.mean(), "tpw_min": g.trades_per_week.min(),
                     "tp": g.share_exit_tp.mean(), "sl": g.share_exit_sl.mean(), "time": g.share_exit_time.mean(),
                     "margin_pct": g.max_margin_used_pct.max(), "expo": g.max_exposure_x_balance.max(),
                     "margin_calls": int(g.margin_call_events.sum()), "ruined": int(g.ruined.astype(bool).sum()),
                     "rule_skipped": g.rule_skipped_orders.mean()})
    c = pd.DataFrame(rows).sort_values("rr").reset_index(drop=True)
    naive = {"u": float(d.naive_mae_u.iloc[0]), "d": float(d.naive_mae_d.iloc[0]), "r": float(d.naive_mae_r.iloc[0]),
             "pu": float(d.persist_mae_u.iloc[0]), "pd": float(d.persist_mae_d.iloc[0]),
             "pr": float(d.persist_mae_r.iloc[0])}
    return c, naive


def break_even(c, key="net"):
    bad = np.flatnonzero(~c.valid.to_numpy())
    n = bad[0] if len(bad) else len(c)
    pre = c.iloc[:n]
    out = {"frontier_rr": float(c.rr.iloc[n]) if n < len(c) else None}
    x = zero_crossing(pre.rr, pre[key]) if len(pre) >= 2 else None
    out["rr"] = x
    if x is not None:
        out["rr_p"] = float(np.interp(x, pre.rr, pre.rr_p))
        out["ru"] = float(np.interp(x, pre.rr, pre.ru))
        out["rd"] = float(np.interp(x, pre.rr, pre.rd))
        out["tpw"] = float(np.interp(x, pre.rr, pre.tpw))
    out["all_seeds_rr"] = zero_crossing(pre.rr, pre.net_min) if len(pre) >= 2 else None
    out["ideal_valid"] = bool(c.valid.iloc[0])
    return out


def figure(df, B, split, strategy, metric, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(1, 3, figsize=(24, 8.6), sharey=True, facecolor="white")
    ycol = {"sharpe": "sharpe", "net": "mult", "maxdd": "dd_pct"}[metric]
    ylab = {"sharpe": "Sharpe anualizado (retornos diarios; 365 días)",
            "net": "Capital final / inicial (mediana de semillas válidas; escala log)",
            "maxdd": "Máxima caída del capital (%)"}[metric]
    curves = {f: curve(df, B, split, strategy, f) for f, *_ in FAMILIES}
    curves["a3r_x2"] = curve(df, B, split, strategy, "a3r", cost=2.0)
    first_inv = True
    for ax, (xk, nk, pk, xl) in zip(axes, (("mae_u", "u", "pu", "MAE de Û (USD/EUR)"), ("mae_d", "d", "pd", "MAE de D̂ (USD/EUR)"),
                                         ("mae_r", "r", "pr", "MAE del rango R̂ = Û + D̂ (USD/EUR)"))):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        if metric != "maxdd":
            ax.axhline(0 if metric == "sharpe" else 1, color="#6b7378", linewidth=1.1)
        for (fam, label, col, ls, lw) in FAMILIES + [("a3r_x2", "A3r con costes ×2", TEAL, ":", 2.0)]:
            r = curves.get(fam)
            if r is None:
                continue
            c, nv = r
            x = c[xk].to_numpy()
            y = c[ycol].to_numpy()
            v = c.valid.to_numpy()
            ax.plot(x, np.where(v, y, np.nan), color=col, linestyle=ls, linewidth=lw, marker="o",
                    markersize=5 if lw > 2.5 else 4, label=label if ax is axes[0] else None)
            if fam == "a3r" and metric == "sharpe":
                sd = c.sharpe_sd.to_numpy()
                ax.fill_between(x, np.where(v, y - sd, np.nan), np.where(v, y + sd, np.nan), color=col, alpha=0.13,
                                linewidth=0, label="A3r: ±1 DE entre semillas válidas" if ax is axes[0] else None)
            if (~v).any():
                yi = c[{"sharpe": "sharpe_all", "net": "net_all", "maxdd": "dd_pct_all"}[metric]].to_numpy()
                if metric == "net":
                    yi = 1 + yi / 10_000
                ax.scatter(x[~v], yi[~v], facecolors="none", edgecolors="#9a9a96", s=34, linewidths=1.2, zorder=3,
                           label="inválida: < 3 oper./semana o ruina" if (first_inv and ax is axes[0]) else None)
                first_inv = first_inv and not (ax is axes[0])
            if fam == "a3r" and metric != "maxdd":
                bad = np.flatnonzero(~v)
                n = bad[0] if len(bad) else len(c)
                key = "sharpe" if metric == "sharpe" else "net"
                x0 = zero_crossing(c[xk].iloc[:n], c[key].iloc[:n]) if n >= 2 else None
                if x0 is not None:
                    ax.axvline(x0, color=col, linestyle=":", linewidth=1.5)
                    if ax is axes[0]:
                        ax.annotate("Primer cambio de signo observado (A3r)", xy=(x0, 0 if metric == "sharpe" else 1),
                                    xytext=(0.45, 0.25), textcoords="axes fraction", fontsize=10.5, color=INK,
                                    arrowprops=dict(arrowstyle="->", color=col, lw=1.3))
        c, nv = curves["a3r"]
        for val, colr, name in ((nv[nk], PURPLE, "Naïve MEAN20"), (nv[pk], RUST, "Naïve persistencia")):
            ax.axvline(val, color=colr, linestyle="--", linewidth=2.0)
            ax.text(val, 0.98 if name.endswith("MEAN20") else 0.86, f"  {name}\n  MAE = {es(val, 5)}",
                    transform=ax.get_xaxis_transform(), color=colr, fontsize=10.5, va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _, n=nv[nk]: f"{es(v, 4)}\n(×{es(v / n, 2)})"))
        ax.set_xlabel(f"{xl}\nentre paréntesis: × naïve MEAN20 del mismo predictor", fontsize=11, color=INK)
        ax.set_xlim(left=-0.02 * max(c[xk].max(), nv[pk]))
    if metric == "net":
        ys = np.concatenate([np.asarray(ln.get_ydata(), float) for ax in axes for ln in ax.get_lines()
                             if len(ln.get_ydata()) > 2] or [np.array([])])
        ys = ys[np.isfinite(ys)]
        if len(ys) and (ys > 0).all() and ys.max() / ys.min() > 20:
            axes[0].set_yscale("log")
        else:
            axes[0].set_ylabel("Capital final / inicial (mediana de semillas válidas)")
    if not any(np.isfinite(np.asarray(ln.get_ydata(), float)).any() for ax in axes for ln in ax.get_lines()
               if len(ln.get_ydata()) > 2):
        axes[1].text(0.5, 0.5, "Ningún nivel cumple la frecuencia mínima de operación:\nno se reporta como resultado.",
                     transform=axes[1].transAxes, ha="center", va="center", fontsize=13, color=INK)
    axes[0].set_ylabel(ylab, fontsize=12, color=INK)
    if metric != "net":
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: es(v, 1 if metric == "sharpe" else 0)))
    ideal = curves["iid"][0].iloc[0]
    d = df[(df.block_hours == B) & (df.split == split) & (df.strategy == strategy) & (df.leverage_scenario == 100)
           & (df.cost_mult == 1)]
    box = "\n".join([SPLIT_ES[split][0], f"{int(d.eligible_blocks.iloc[0])} orígenes; bloques de {B} h",
                     f"Estrategia {STRAT_ES[strategy]}", "Capital inicial 10.000 USD; compuesto",
                     "rel_volume 0,10 del saldo; percentual_order_size 1,0",
                     "Apalancamiento 1:100 (escenario por defecto)",
                     "Válido: ≥ 3 oper./semana, ≤ 4 órdenes/día, sin ruina",
                     "SL y TP efectivos en toda orden", "Latencia 1 barra; límites: cruce ≥ 1 pip",
                     "Spread 0,3-3 pips por hora; comisión 3,5 USD/lote/lado",
                     "Swap largo −7 / corto +2,5 USD/lote/noche",
                     f"Ideal: {es(ideal.tpw, 1)} oper./sem.; salidas TP {es(100 * ideal.tp, 0)} %, "
                     f"SL {es(100 * ideal.sl, 0)} %, tiempo {es(100 * ideal.time, 0)} %",
                     "Costes supuestos, no verificados con bróker"])
    fig.text(0.815, 0.88, box, ha="left", va="top", fontsize=9.5, color=INK, multialignment="left",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#b8c4cc"))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.815, 0.36), frameon=True, edgecolor="#d0d0d0")
    title = {"sharpe": "Sharpe", "net": "Capital final", "maxdd": "Máxima caída"}[metric]
    fig.suptitle(f"{title} frente al error del predictor de volatilidad — {STRAT_ES[strategy]}\n"
                 f"Bloques de {B} h · {SPLIT_ES[split][0]} · DESARROLLO · estudio oráculo/ruido, no es un resultado de modelo",
                 fontsize=15, color=INK)
    fig.tight_layout(rect=(0, 0, 0.805, 1))
    stem = f"barrido_volatilidad_{metric}_B{B}_{SPLIT_ES[split][1]}_{strategy}"
    fig.savefig(outdir / f"{stem}.png", dpi=130)
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)
    return stem


def leverage_figure(df, B, split, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True, facecolor="white")
    for ax, st in zip(axes, ("V1", "V2")):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.axhline(0, color="#6b7378", linewidth=1.1)
        for lev, col, ls in ((10.0, "#9c2f7a", "-"), (50.0, RUST, "--"), (100.0, TEAL, "-")):
            r = curve(df, B, split, st, "a3r", lev=lev)
            if r is None:
                continue
            c, nv = r
            ax.plot(c.rr, np.where(c.valid, c.sharpe, np.nan), color=col, linestyle=ls, marker="o", markersize=4,
                    linewidth=2.2 if lev == 100 else 1.6, label=f"1:{int(lev)} (nocional {int(lev / 10)}× saldo)")
        ax.axvline(1.0, color=PURPLE, linestyle="--", linewidth=1.6)
        if not any(np.isfinite(np.asarray(ln.get_ydata(), float)).any() for ln in ax.get_lines()
                   if len(ln.get_ydata()) > 2):
            ax.set_ylim(-1, 1)
            ax.text(0.5, 0.5, "Ningún nivel cumple la frecuencia mínima:\nno se reporta como resultado.",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12, color=INK)
        ax.set_title(STRAT_ES[st], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("MAE del rango R̂ × naïve MEAN20 (familia A3r)", fontsize=11, color=INK)
        ax.legend(fontsize=9.5, frameon=False)
    axes[0].set_ylabel("Sharpe anualizado (semillas válidas)", fontsize=11, color=INK)
    fig.suptitle(f"Escenarios de apalancamiento de la cuenta — bloques de {B} h · {SPLIT_ES[split][0]} · DESARROLLO · "
                 "estudio oráculo/ruido", fontsize=13, color=INK)
    fig.tight_layout()
    stem = f"apalancamiento_B{B}_{SPLIT_ES[split][1]}"
    fig.savefig(outdir / f"{stem}.png", dpi=130)
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)
    return stem


def f2(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else es(v, nd)


def tables(df, sens):
    rows, md = [], ["# Estudio V1/V2 sin dirección — tablas (generado)\n",
                    "DESARROLLO; estudio oráculo/ruido, no es un resultado de modelo. Break-even leído en la media de "
                    "semillas válidas (beneficio neto), solo sobre niveles válidos antes del primer inválido. "
                    "Ratios: MAE del predictor / MAE del naïve MEAN20 (entre paréntesis: / persistencia) en las mismas "
                    "filas. «—»: sin cambio de signo mientras es válida.\n"]
    for B in (24, 6):
        for split in ("VALIDATION_2024", "TRAIN_2020_2023"):
            md.append(f"## Bloques de {B} h · {SPLIT_ES[split][0]}\n")
            md.append("| Estrategia | Familia | Apal. | Costes | Sharpe ideal | oper./sem. ideal | Break-even R̂ ×MEAN20 (×pers.) "
                      "| Û / D̂ ×MEAN20 | todas las semillas | frontera validez | TP / SL / tiempo % ideal | margen máx. % | "
                      "exposición máx. | margin calls | ruinas |")
            md.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
            for st in ("V1", "V2", "RFM_COIN", "RFL_COIN"):
                for fam in ("iid", "a1r", "a3r"):
                    for lev, cost in ((100.0, 1.0), (50.0, 1.0), (10.0, 1.0), (100.0, 2.0)):
                        if fam != "a3r" and (lev, cost) != (100.0, 1.0):
                            continue
                        r = curve(df, B, split, st, fam, lev, cost)
                        if r is None:
                            continue
                        c, nv = r
                        be = break_even(c)
                        i0 = c.iloc[0]
                        rows.append({"block_hours": B, "split": split, "strategy": st, "family": fam, "leverage": lev,
                                     "cost_mult": cost, "ideal_sharpe": i0.sharpe if i0.valid else None,
                                     "ideal_trades_per_week": i0.tpw, **{f"be_{k}": v for k, v in be.items()},
                                     "ideal_tp_share": i0.tp, "ideal_sl_share": i0.sl, "ideal_time_share": i0.time,
                                     "max_margin_pct": c.margin_pct.max(), "max_exposure": c.expo.max(),
                                     "margin_calls": int(c.margin_calls.sum()), "ruined_runs": int(c.ruined.sum())})
                        md.append(f"| {st} | {fam} | 1:{int(lev)} | ×{int(cost)} | "
                                  f"{f2(i0.sharpe) if i0.valid else 'inválida'} | {f2(i0.tpw, 1)} | "
                                  f"{f2(be['rr'])} ({f2(be.get('rr_p'))}) | {f2(be.get('ru'))} / {f2(be.get('rd'))} | "
                                  f"{f2(be['all_seeds_rr'])} | {f2(be['frontier_rr'])} | "
                                  f"{es(100 * i0.tp, 0)} / {es(100 * i0.sl, 0)} / {es(100 * i0.time, 0)} | "
                                  f"{f2(c.margin_pct.max(), 1)} | {f2(c.expo.max(), 1)}× | {int(c.margin_calls.sum())} | "
                                  f"{int(c.ruined.sum())} |")
            md.append("")
    if sens is not None and len(sens):
        md.append("## Sensibilidad de multiplicadores — solo 2023 (A3r; apalancamiento 1:100; costes ×1)\n")
        md.append("Los multiplicadores principales se fijaron a priori; esta tabla solo muestra la sensibilidad en 2023 "
                  "(nunca 2024).\n")
        md.append("| B | Estrategia | Parámetros | Nivel A3r (w) | Sharpe medio | Neto medio USD | oper./sem. | válido (≥ 3 semillas) |")
        md.append("|---|---|---|---|---:|---:|---:|---|")
        for (B, st, prm, lv), g in sens.groupby(["block_hours", "strategy", "params", "level"]):
            ok = validity(g)
            from tools.short_horizon_vstudy import A3
            md.append(f"| {B} | {st} | `{prm}` | {es(A3[int(lv)][0], 1)} | "
                      f"{f2(g[ok].sharpe_daily_sqrt365.mean()) if ok.sum() >= 3 else 'inválida'} | "
                      f"{f2(g[ok].net_profit_usd.mean(), 0) if ok.sum() >= 3 else '—'} | {f2(g.trades_per_week.mean(), 1)} | "
                      f"{'sí' if ok.sum() >= 3 else 'no'} |")
        md.append("")
    return rows, "\n".join(md) + "\n"


def build(run: Path, pub: Path):
    df = pd.concat([pd.read_csv(p) for p in sorted(run.glob("cells_main_*.csv"))], ignore_index=True)
    sens_files = sorted(run.glob("cells_sens_*.csv"))
    sens = pd.concat([pd.read_csv(p) for p in sens_files], ignore_index=True) if sens_files else None
    pub.mkdir(parents=True, exist_ok=True)
    stems = []
    for B in (24, 6):
        for split in ("VALIDATION_2024", "TRAIN_2020_2023"):
            for st in ("V1", "V2"):
                for metric in ("sharpe", "net", "maxdd"):
                    stems.append(figure(df, B, split, st, metric, pub))
            stems.append(leverage_figure(df, B, split, pub))
    rows, md = tables(df, sens)
    (pub / "TABLAS_V1_V2.md").write_text(md)
    (pub / "break_even_v1_v2.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    merged = run / "cells_merged.csv"
    df.to_csv(merged, index=False)
    with open(merged, "rb") as src, open(pub / "vstudy_cells.csv.gz", "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0, filename="") as dst:
        shutil.copyfileobj(src, dst)
    if sens is not None:
        sens.to_csv(pub / "sensibilidad_2023.csv", index=False, float_format="%.8g")
    receipts = sorted(run.glob("receipt_*.json"))
    (pub / "MANIFEST.json").write_text(json.dumps({
        "cells_sha256": sha(merged), "rows": len(df),
        "cpu_seconds": round(sum(json.loads(p.read_text())["cpu_seconds"] for p in receipts), 1),
        "chunks": len(receipts), "figures": stems, "data_sha256": json.loads(receipts[0].read_text())["data_sha256"],
        "sizing": {"rel_volume": 0.10, "percentual_order_size": 1.0, "basis": "margin", "leverage_scenarios": [10, 50, 100]},
        "test_rows_read": False}, indent=1) + "\n")
    print(json.dumps({"figures": len(stems), "rows": len(df)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--publish", type=Path, default=ROOT / "docs/short_horizon/results/volatility_strategies")
    a = ap.parse_args()
    build(a.run, a.publish)


if __name__ == "__main__":
    main()

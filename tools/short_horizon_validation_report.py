#!/usr/bin/env python3
"""Figures and break-even tables for business-day RF-L degradation on VALIDATION 2024.

Reads only <run>/cells.csv and <run>/equity_val24_*.npz written by
``tools/short_horizon_oracle.py run --grid val24``. DEVELOPMENT oracle/noise
study, not a model result.
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
from tools.short_horizon_oracle import LABEL  # noqa: E402
from tools.short_horizon_report import style, INK, MUTED  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated slots 1-5
GRAY = "#8a8984"
SUBTITLE = "EURUSD business-day RF-L · DEVELOPMENT · VALIDATION 2024 · oracle/noise study, not a model result"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve(df, sweep, cost, by):
    g = df[(df.sweep == sweep) & (df.cost_mult == cost)].groupby(by)
    out = g.agg(net=("net_profit_usd", "mean"), net_lo=("net_profit_usd", "min"), net_hi=("net_profit_usd", "max"),
                sharpe=("sharpe_daily_sqrt365", "mean"), sharpe_lo=("sharpe_daily_sqrt365", "min"),
                sharpe_hi=("sharpe_daily_sqrt365", "max"), dd=("max_drawdown_usd", "mean"),
                dd_lo=("max_drawdown_usd", "min"), dd_hi=("max_drawdown_usd", "max"),
                ratio=("range_mae_ratio_to_naive", "mean"), mae_h=("mae_high_pips", "mean"),
                mae_l=("mae_low_pips", "mean"), ft=("first_touch_accuracy", "mean"),
                brier=("brier_calibrated", "mean"), pf=("profit_factor", "mean"), trades=("trades", "mean"),
                win=("win_rate", "mean"), end=("end_equity_usd", "mean"), seeds=("seed", "nunique")).reset_index()
    return out


def baseline(df, name, cost):
    return df[(df.sweep == "baseline") & (df.strategy == name) & (df.cost_mult == cost)].iloc[0]


def degradation_figure(df, x_of, xlabel, path, title, invert=False, secondary=None, symlog=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 11), sharex=True, facecolor="#fcfcfb")
    panels = [("net", "net profit, USD"), ("sharpe", "Sharpe, daily returns × √365"), ("dd", "max drawdown, USD")]
    naive = {m: baseline(df, "NAIVE_MEAN20_RF-L", m) for m in (1.0, 2.0)}
    crossings = {}
    for (key, ylabel), ax in zip(panels, axes):
        style(ax)
        for col, cost in zip(SERIES, (1.0, 2.0)):
            c = x_of(cost)
            ax.fill_between(c.x, c[f"{key}_lo"], c[f"{key}_hi"], color=col, alpha=0.18, linewidth=0)
            ax.plot(c.x, c[key], color=col, linewidth=2, marker="o", markersize=4,
                    label=f"RF-L, costs ×{cost:g} (mean, band = min–max of 5 seeds)")
            if key == "net":
                crossings[cost] = zero_crossing(c.x, c.net)
                if crossings[cost] is not None:
                    ax.axvline(crossings[cost], color=col, linestyle=":", linewidth=1.5)
                    ax.annotate(f"break-even ×{cost:g}: {crossings[cost]:.2f}", (crossings[cost], 0),
                                xytext=(4, 10), textcoords="offset points", fontsize=8, color=INK)
        ref = {"net": "net_profit_usd", "sharpe": "sharpe_daily_sqrt365", "dd": "max_drawdown_usd"}[key]
        ax.axhline(0, color=GRAY, linewidth=1.2, linestyle="--")
        ax.annotate("NO_TRADE (0)", (1, 0), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                    textcoords="offset points", ha="right", fontsize=8, color=MUTED)
        vals = {c: naive[c][ref] for c in (1.0, 2.0)}
        for cost, ls in ((1.0, "-."), (2.0, ":")):
            v = vals[cost]
            other = vals[3.0 - cost]
            ax.axhline(v, color=GRAY, linewidth=1.0, linestyle=ls)
            ax.annotate(f"naive-range RF-L ×{cost:g} ({v:,.1f})" if key == "sharpe" else
                        f"naive-range RF-L ×{cost:g} ({v:,.0f})", (0.5, v), xycoords=("axes fraction", "data"),
                        xytext=(0, 4 if v >= other else -11), textcoords="offset points", ha="center",
                        fontsize=8, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        if symlog:
            ax.set_xscale("symlog", linthresh=1.0)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right" if not invert else "lower left", bbox_to_anchor=None if not invert else (0.0, 0.12))
    none = [f"×{c:g}" for c in (1.0, 2.0) if crossings.get(c) is None]
    axes[0].set_title("net profit" + (f" — mean never crosses zero on this grid ({', '.join(none)})" if none else ""),
                      fontsize=9, color=INK, loc="left")
    axes[-1].set_xlabel(xlabel, fontsize=9, color=MUTED)
    if invert:
        axes[-1].invert_xaxis()
    if secondary:
        scale, label, pips = secondary
        top = axes[0].twiny()
        top.set_xscale(axes[0].get_xscale(), **({"linthresh": 1.0} if symlog else {}))
        top.set_xlim(axes[0].get_xlim())
        top.set_xticks([x / scale for x in pips], [f"{x:g}" for x in pips])
        top.set_xlabel(label, fontsize=9, color=MUTED)
        top.tick_params(colors=MUTED, labelsize=8)
        top.spines["top"].set_color("#e4e3df")
    fig.suptitle(f"{title}\n{SUBTITLE}", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return crossings


def joint_figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", ["#8f2121", "#e34948", "#f0efec", "#5598e7", "#104281"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), facecolor="#fcfcfb")
    contours = {}
    for ax, cost in zip(axes, (1.0, 2.0)):
        j = curve(df, "joint", cost, ["level", "p_flip"])
        levels = sorted(j.level.unique())
        ps = sorted(j.p_flip.unique())
        net = j.pivot(index="p_flip", columns="level", values="net").loc[ps, levels].to_numpy() / 1000
        ratio = j.groupby("level").ratio.mean().loc[levels].to_numpy()
        acc = j.groupby("p_flip").ft.mean().loc[ps].to_numpy()
        norm = TwoSlopeNorm(vcenter=0.0, vmin=min(net.min(), -0.5), vmax=net.max())
        im = ax.imshow(net, cmap=cmap, norm=norm, aspect="auto", origin="upper")
        ax.contour(np.arange(len(levels)), np.arange(len(ps)), net, levels=[0.0], colors=INK, linewidths=2)
        for i in range(len(ps)):
            for k in range(len(levels)):
                ax.text(k, i, f"{net[i, k]:.1f}", ha="center", va="center", fontsize=7,
                        color="#fcfcfb" if abs(norm(net[i, k]) - 0.5) > 0.35 else INK)
        ax.set_xticks(range(len(levels)), [f"{r:.2f}\n({lv:g})" for r, lv in zip(ratio, levels)], fontsize=7)
        ax.set_yticks(range(len(ps)), [f"{a:.2f}" for a in acc], fontsize=8)
        ax.set_xlabel("achieved range MAE / naive MEAN20 MAE  (σ in pips)", fontsize=9, color=MUTED)
        ax.set_ylabel("achieved first-touch accuracy (non-ambiguous blocks)", fontsize=9, color=MUTED)
        ax.set_title(f"mean net profit, k USD, 5 seeds · costs ×{cost:g} · black line = 0", fontsize=9,
                     color=INK, loc="left")
        fig.colorbar(im, ax=ax, shrink=0.8).set_label("k USD", fontsize=8, color=MUTED)
        contours[cost] = [(acc[i], zero_crossing(ratio, net[i]), ) for i in range(len(ps))]
    fig.suptitle(f"Joint degradation: Gaussian range noise × first-touch flips\n{SUBTITLE}", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return contours


def equity_figure(run, df, picks, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    curves = {}
    for p in sorted(run.glob("equity_val24_*.npz")):
        with np.load(p) as z:
            curves.update({k: z[k] for k in z.files})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor="#fcfcfb")
    for ax, zoom in zip(axes, (False, True)):
        style(ax)
        for col, (name, cell) in zip(SERIES, picks):
            if zoom and cell.get("hide_in_zoom"):
                continue
            key = "|".join(map(str, ("val24", 24, cell["strategy"], cell["family"], cell["level"], cell["p_flip"],
                                     cell["seed"], 1.0, cell["sweep"], "VALIDATION_2024")))
            y = curves[key]
            x = pd.date_range(pd.Timestamp(int(curves[key + "|t0"][0]), tz="UTC"), periods=len(y), freq="D")
            ax.plot(x, (y - 10_000) / 1000, color=col, linewidth=1.8)
            ax.annotate(name, (x[-1], (y[-1] - 10_000) / 1000), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
        ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.2)
        ax.annotate("NO_TRADE", (0.01, 0), xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", fontsize=8, color=MUTED)
        loc = mdates.AutoDateLocator(maxticks=7)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.set_ylabel("cumulative net P&L, k USD (start 10,000)", fontsize=9, color=MUTED)
        ax.set_title("all curves" if not zoom else "zoom: without ideal and light noise", fontsize=9,
                     color=INK, loc="left")
    fig.suptitle(f"2024 daily marked-to-market equity incl. flat days (seed 42, costs ×1)\n{SUBTITLE}",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fraction_level(c, x, frac, ideal):
    return zero_crossing(c[x], c.net - frac * ideal)


def build(run: Path, pub: Path):
    df = pd.read_csv(run / "cells.csv")
    if set(df.label) != {LABEL} or set(df.split) != {"VALIDATION_2024"}:
        raise SystemExit("unexpected population")
    pub.mkdir(parents=True, exist_ok=True)
    naive_pips = float(df.naive_mean20_mae_pips.dropna().iloc[0])

    def range_x(cost):
        c = curve(df, "range", cost, "level").sort_values("level")
        return c.assign(x=c.ratio)

    def ft_x(cost):
        c = curve(df, "firsttouch", cost, "p_flip").sort_values("p_flip")
        return c.assign(x=c.ft)

    cross_r = degradation_figure(df, range_x, "achieved range MAE / naive MEAN20 MAE (symlog: linear below 1)",
                                 pub / "validation_degradation_range.png",
                                 "Range noise only (Gaussian on predicted high and low; first touch perfect)",
                                 secondary=(naive_pips, f"achieved range MAE, pips (naive MEAN20 = {naive_pips:.1f} pips)",
                                            (5, 10, 18, 50, 100, 200, 470)),
                                 symlog=True)
    cross_f = degradation_figure(df, ft_x, "achieved first-touch accuracy (non-ambiguous blocks)",
                                 pub / "validation_degradation_firsttouch.png",
                                 "First-touch flips only (range perfect)", invert=True)
    contours = joint_figure(df, pub / "validation_degradation_joint.png")

    ideal = {m: curve(df, "range", m, "level").set_index("level").loc[0.0] for m in (1.0, 2.0)}
    # equity picks: ideal, light, medium, near break-even (joint cell with |mean net| smallest), naive
    j = curve(df, "joint", 1.0, ["level", "p_flip"])
    nb = j.iloc[(j.net.abs()).argmin()]
    picks = [("ideal", dict(strategy="RF-L", family="abs", level=0.0, p_flip=0.0, seed=42, sweep="range",
                            hide_in_zoom=True)),
             ("light: σ 5 pips", dict(strategy="RF-L", family="abs", level=5.0, p_flip=0.0, seed=42, sweep="range",
                                      hide_in_zoom=True)),
             ("medium: σ 30 pips", dict(strategy="RF-L", family="abs", level=30.0, p_flip=0.0, seed=42,
                                        sweep="range")),
             (f"near break-even: σ {nb.level:g} pips, p {nb.p_flip:g}",
              dict(strategy="RF-L", family="abs", level=float(nb.level), p_flip=float(nb.p_flip), seed=42,
                   sweep="joint")),
             ("naive-range RF-L", dict(strategy="NAIVE_MEAN20_RF-L", family="none", level=0.0, p_flip=0.0, seed=0,
                                       sweep="baseline"))]
    equity_figure(run, df, picks, pub / "validation_equity_2024.png")

    rows = []
    for name, sweep, by in (("range", "range", "level"), ("firsttouch", "firsttouch", "p_flip")):
        for m in (1.0, 2.0):
            c = curve(df, sweep, m, by).sort_values(by)
            for _, r in c.iterrows():
                rows.append({"sweep": name, "cost_mult": m, "sigma_pips": r.level if by == "level" else 0.0,
                             "p_flip": r.p_flip if by == "p_flip" else 0.0, **{k: r[k] for k in (
                                 "ratio", "mae_h", "mae_l", "ft", "brier", "net", "net_lo", "net_hi", "end", "dd",
                                 "sharpe", "pf", "trades", "win", "seeds")}})
    for m in (1.0, 2.0):
        c = curve(df, "joint", m, ["level", "p_flip"])
        for _, r in c.iterrows():
            rows.append({"sweep": "joint", "cost_mult": m, "sigma_pips": r.level, "p_flip": r.p_flip,
                         **{k: r[k] for k in ("ratio", "mae_h", "mae_l", "ft", "brier", "net", "net_lo", "net_hi",
                                              "end", "dd", "sharpe", "pf", "trades", "win", "seeds")}})
    summary = pd.DataFrame(rows)
    summary.to_csv(pub / "validation_degradation_summary.csv", index=False, float_format="%.6g")

    be = {"naive_mean20_mae_pips": naive_pips,
          "ideal": {str(m): {"net": float(ideal[m].net), "sharpe": float(ideal[m].sharpe)} for m in (1.0, 2.0)},
          "range_zero_crossing_ratio": {str(m): cross_r[m] for m in (1.0, 2.0)},
          "firsttouch_zero_crossing_accuracy": {str(m): cross_f[m] for m in (1.0, 2.0)},
          "range_ratio_at_fraction_of_ideal": {}, "firsttouch_accuracy_at_fraction_of_ideal": {},
          "joint_zero_contour_ratio_by_accuracy": {str(m): [[float(a), x] for a, x in contours[m]] for m in (1.0, 2.0)},
          "naive_baseline": {str(m): {k: float(baseline(df, "NAIVE_MEAN20_RF-L", m)[k]) for k in (
              "net_profit_usd", "sharpe_daily_sqrt365", "max_drawdown_usd", "first_touch_accuracy",
              "range_mae_ratio_to_naive", "trades")} for m in (1.0, 2.0)}}
    for m in (1.0, 2.0):
        cr = range_x(m)
        cf = ft_x(m)
        be["range_ratio_at_fraction_of_ideal"][str(m)] = {
            f: fraction_level(cr, "x", f, ideal[m].net) for f in (0.5, 0.1)}
        be["firsttouch_accuracy_at_fraction_of_ideal"][str(m)] = {
            f: fraction_level(cf, "x", f, ideal[m].net) for f in (0.5, 0.1)}
    (pub / "validation_break_even.json").write_text(json.dumps(be, indent=1, default=float) + "\n")
    with open(run / "cells.csv", "rb") as src, open(pub / "validation_cells.csv.gz", "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0, filename="") as dst:
        shutil.copyfileobj(src, dst)
    receipts = sorted(run.glob("receipt_val24_*.json"))
    manifest = {"label": LABEL, "cells_csv_sha256": sha(run / "cells.csv"), "rows": int(len(df)),
                "data_sha256": json.loads(receipts[0].read_text())["data_sha256"],
                "backtest_cpu_seconds": round(sum(json.loads(p.read_text())["cpu_seconds"] for p in receipts), 1),
                "chunks": len(receipts), "split": "VALIDATION_2024", "test_rows_read": False}
    (pub / "VALIDATION_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(be, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--publish", type=Path, default=ROOT / "docs/short_horizon/results/validation_2024")
    a = ap.parse_args()
    build(a.run, a.publish)


if __name__ == "__main__":
    main()

"""Report of the strategy simulation sweeps (tools/strategy_output_simulation.py): pooled tables per variant x quality,
break-even quality per head family, robustness grid, gene sensitivity, and the PNG curves (one axis per figure, fixed
categorical colours, oracle ceiling and chance floor on every panel). Writes SWEEP_*.csv/json, PNGs and the Markdown
fragments the design documents include."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# the validated default categorical palette (dataviz skill, light surface), fixed order; <= 6 series per figure
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GROUPS = {"quality": ["base", "block", "regime", "onesided", "closer_only", "opener_only"],
          "execution": ["base", "cost2", "cost3", "spike", "missing10"]}
PALETTE = {v: SLOTS[i] for g in GROUPS.values() for i, v in enumerate(g) if v != "base"}
PALETTE["base"] = SLOTS[0]
ORDER = ["base", "block", "regime", "onesided", "closer_only", "opener_only", "cost2", "cost3", "spike", "missing10"]
METRICS = [("dec_sharpe", "Sharpe anualizado (año de decisión)"), ("dec_net", "Neto USD (año de decisión, 10 000 iniciales)"),
           ("dec_exp", "Expectativa neta por trade (σ)"), ("dec_opw", "Órdenes por semana"), ("dec_dd", "Drawdown máximo %"),
           ("dec_early_close_value", "Valor del cierre temprano USD (con − sin)"), ("dec_win", "Tasa de acierto"),
           ("buy_win", "Acierto compras"), ("sell_win", "Acierto ventas")]


def pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, level): means over seeds x folds, the min lower CI of the expectancy, counts of refusals / failed checks."""
    rows = []
    for (v, lv), g in df.groupby(["variant", "level"]):
        ok = g[g["refused"].isna()]
        r = {"variant": v, "level": lv, "n": int(len(g)), "refused": int(g["refused"].notna().sum()),
             "checks_failed": int(ok["checks_failed"].fillna(0).sum()), "early_close_disabled": int(ok["early_close_disabled"].sum()),
             "auc_dec": float(ok["auc_dec"].mean()) if len(ok) else np.nan, "mae_over_naive_dec": float(ok["mae_over_naive_dec"].mean()) if len(ok) else np.nan,
             "spearman_dec": float(ok["spearman_dec"].mean()) if len(ok) else np.nan}
        for k, _ in METRICS:
            r[k] = float(ok[k].mean()) if len(ok) and ok[k].notna().any() else np.nan
            r[k + "_min"] = float(ok[k].min()) if len(ok) and ok[k].notna().any() else np.nan
            r[k + "_max"] = float(ok[k].max()) if len(ok) and ok[k].notna().any() else np.nan
        r["dec_exp_lo_min"] = float(ok["dec_exp_lo"].min()) if len(ok) and ok["dec_exp_lo"].notna().any() else np.nan
        r["dec_exp_lo_mean"] = float(ok["dec_exp_lo"].mean()) if len(ok) and ok["dec_exp_lo"].notna().any() else np.nan
        r["share_sharpe_positive"] = float((ok["dec_sharpe"] > 0).mean()) if len(ok) else np.nan
        r["dec_trades"] = float(ok["dec_trades"].mean()) if len(ok) else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["variant", "level"]).reset_index(drop=True)


def break_even(p: pd.DataFrame, variant: str, col: str = "dec_exp_lo_mean") -> dict:
    """The lowest quality level at which the mean lower CI of the per-trade expectancy is > 0 AND stays > 0 at every
    higher level of the ladder (monotone break-even); None when no level qualifies."""
    g = p[p["variant"] == variant].sort_values("level")
    lv = g["level"].to_numpy()
    x = g[col].to_numpy(float)
    ok = np.isfinite(x) & (x > 0)
    be = None
    for i in range(len(lv)):
        if ok[i] and ok[i:].all():
            be = float(lv[i])
            break
    first = float(lv[np.argmax(ok)]) if ok.any() else None
    return {"variant": variant, "break_even_level": be, "first_positive_level": first, "levels": lv.tolist(), "lower_ci": [None if not math.isfinite(v) else float(v) for v in x],
            "oracle_ceiling": {k: (None if not math.isfinite(float(g[k].iloc[-1])) else float(g[k].iloc[-1])) for k in ("dec_sharpe", "dec_net", "dec_exp")} if len(g) else {},
            "chance_floor": {k: (None if not math.isfinite(float(g[k].iloc[0])) else float(g[k].iloc[0])) for k in ("dec_sharpe", "dec_net", "dec_exp")} if len(g) else {}}


def plot_curves(p: pd.DataFrame, out: Path, variants: list, metric: str, label: str, xcol: str = "level", xlabel: str = "AUC de las cabezas binarias (= Spearman de las regresiones)", tag: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    ax.set_facecolor("#fcfcfb")
    for v in variants:
        g = p[p["variant"] == v].sort_values(xcol)
        if not len(g):
            continue
        x = g[xcol].to_numpy(float)
        y = g[metric].to_numpy(float)
        ax.plot(x, y, color=PALETTE.get(v, "#52514e"), linewidth=2, marker="o", markersize=5, label=v)
        if metric + "_min" in g and metric + "_max" in g:
            ax.fill_between(x, g[metric + "_min"].to_numpy(float), g[metric + "_max"].to_numpy(float), color=PALETTE.get(v, "#52514e"), alpha=0.12, linewidth=0)
    base = p[p["variant"] == "base"].sort_values("level")
    if len(base):
        ax.axhline(float(base[metric].iloc[-1]), color="#0b0b0b", linewidth=1, linestyle="--", label="techo: oráculo (calidad 1.0, base)")
        ax.axhline(float(base[metric].iloc[0]), color="#52514e", linewidth=1, linestyle=":", label="suelo: azar (calidad 0.5, base)")
    if metric in ("dec_sharpe", "dec_net", "dec_exp", "dec_early_close_value"):
        ax.axhline(0.0, color="#c3c2b7", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(label)
    ax.grid(True, color="#e7e6e2", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    name = f"{metric}_vs_{xcol}{tag}.png"
    fig.savefig(out / name)
    plt.close(fig)
    return name


def md_table(p: pd.DataFrame, variant: str) -> str:
    g = p[p["variant"] == variant].sort_values("level")
    h = "| calidad (AUC / ρ) | AUC medida (decisión) | MAE/naive neto | Sharpe | neto USD | expect. σ [IC inf. medio] | órdenes/sem | DD % | acierto compras / ventas | cierre temprano USD | fallos de chequeo | rechazos | n |"
    s = [h, "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    f = lambda x, d=2: ("—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}")  # noqa: E731
    for _, r in g.iterrows():
        s.append(f"| {r['level']:.2f} | {f(r['auc_dec'], 3)} | {f(r['mae_over_naive_dec'], 2)} | {f(r['dec_sharpe'])} | {f(r['dec_net'], 0)} | {f(r['dec_exp'], 3)} [{f(r['dec_exp_lo_mean'], 3)}] | "
                 f"{f(r['dec_opw'])} | {f(r['dec_dd'], 1)} | {f(r['buy_win'])} / {f(r['sell_win'])} | {f(r['dec_early_close_value'], 0)} | {int(r['checks_failed'])} | {int(r['refused'])} | {int(r['n'])} |")
    return "\n".join(s)


def gene_table(runs: list) -> str:
    rows = []
    for r in runs:
        for s in r["per_seed"]:
            st = s.get("strategy_effective") or s.get("strategy")
            if not st:
                continue
            rows.append({"variante": r["variant"], "calidad": r["level"], "semilla": s["seed"], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}})
    if not rows:
        return "(sin estrategias congeladas)"
    d = pd.DataFrame(rows)
    cols = list(d.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def sensitivity_table(runs: list) -> str:
    rows = []
    for r in runs:
        for s in r["per_seed"]:
            for g in (s.get("sensitivity") or []):
                rows.append({"variante": r["variant"], "calidad": r["level"], "semilla": s["seed"], **g})
    if not rows:
        return "(sin análisis de sensibilidad)"
    d = pd.DataFrame(rows)
    agg = d.groupby("gene").agg(d_sharpe_minus=("sharpe_minus_delta", "mean"), d_sharpe_plus=("sharpe_plus_delta", "mean"),
                                d_opw_minus=("opw_minus_delta", "mean"), d_opw_plus=("opw_plus_delta", "mean"), n=("gene", "size")).reset_index()
    out = ["| gen (cuantil) | Δ Sharpe (−0.1) | Δ Sharpe (+0.1) | Δ órdenes/sem (−0.1) | Δ órdenes/sem (+0.1) | n |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in agg.iterrows():
        out.append(f"| {r['gene']} | {r['d_sharpe_minus']:+.2f} | {r['d_sharpe_plus']:+.2f} | {r['d_opw_minus']:+.2f} | {r['d_opw_plus']:+.2f} | {int(r['n'])} |")
    return "\n".join(out)


FITNESS_ORDER = ["sharpe_lcb", "sharpe_plus_return", "sharpe_plus_return_plus_activity", "trade_tstat", "calmar", "return_minus_dd", "balanced"]
FITNESS_LABEL = {"sharpe_lcb": "F0 Sharpe − EE", "sharpe_plus_return": "F1 Sharpe + retorno", "sharpe_plus_return_plus_activity": "F2 Sharpe + retorno + órdenes/día (propietario)",
                 "trade_tstat": "F3 t por trade", "calmar": "F4 Calmar", "return_minus_dd": "F5 retorno − DD", "balanced": "F6 balanceada"}


def fitness_rows(runs: list) -> pd.DataFrame:
    rows = []
    for r in runs:
        for s in r["per_seed"]:
            for f, v in (s.get("by_fitness") or {}).items():
                d = v.get("decision") or {}
                rows.append({"fold": r["fold"], "variant": r["variant"], "gate": r.get("gate", "contract"), "family": r.get("family", "c3"), "level": r["level"], "seed": s["seed"], "fitness": f,
                             "valid_choice": bool(v.get("valid")), "dec_sharpe": d.get("sharpe_daily_sqrt365"), "dec_net": d.get("net_profit_usd"),
                             "dec_exp": d.get("expectancy"), "dec_exp_lo": d.get("expectancy_low"), "dec_opw": d.get("orders_per_week"), "dec_valid": d.get("valid"),
                             "sel_sharpe": (v.get("selection") or {}).get("sharpe_daily_sqrt365")})
    return pd.DataFrame(rows)


def fitness_table(fr: pd.DataFrame) -> tuple:
    """Per (scenario, level): the fitness with the best mean decision expectancy / Sharpe / net over seeds and folds, the
    share of (fold, seed) cells each fitness wins, and the pooled ranking (the production recommendation's evidence)."""
    if not len(fr):
        return "(sin evaluación por fitness)", {}
    fr = fr[fr["valid_choice"]]
    win = {}
    lines = ["| familia | escenario | calidad | mejor por expectativa | mejor por Sharpe | mejor por neto | estabilidad: fitness ganadora (expect.) por celda fold×semilla |", "| --- | --- | ---: | --- | --- | --- | --- |"]
    for (fam, v, lv), g in fr.groupby(["family", "variant", "level"]):
        m = g.groupby("fitness")[["dec_exp", "dec_sharpe", "dec_net"]].mean()
        best = {k: (m[k].idxmax() if m[k].notna().any() else None) for k in ("dec_exp", "dec_sharpe", "dec_net")}
        cells = g.groupby(["fold", "seed"])
        counts = {}
        for _, c in cells:
            c = c.dropna(subset=["dec_exp"])
            if len(c):
                w = c.loc[c["dec_exp"].idxmax(), "fitness"]
                counts[w] = counts.get(w, 0) + 1
                win[w] = win.get(w, 0) + 1
        stab = ", ".join(f"{FITNESS_LABEL.get(k, k).split(' ')[0]} {n}" for k, n in sorted(counts.items(), key=lambda x: -x[1]))
        lines.append(f"| {fam} | {v} | {lv:.2f} | {FITNESS_LABEL.get(best['dec_exp'], best['dec_exp'])} | {FITNESS_LABEL.get(best['dec_sharpe'], best['dec_sharpe'])} | {FITNESS_LABEL.get(best['dec_net'], best['dec_net'])} | {stab} |")
    pooled = fr.groupby("fitness")[["dec_exp", "dec_sharpe", "dec_net", "dec_opw"]].mean().reindex(FITNESS_ORDER)
    val = fr.groupby("fitness")["dec_valid"].mean().reindex(FITNESS_ORDER)
    lines += ["", "| fitness | expectativa media (decisión) | Sharpe medio | neto medio USD | órdenes/sem | válida fuera de bloque (≥ 2/sem) | celdas ganadas |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for f in FITNESS_ORDER:
        if f in pooled.index and pd.notna(pooled.loc[f, "dec_exp"]):
            lines.append(f"| {FITNESS_LABEL[f]} | {pooled.loc[f, 'dec_exp']:.3f} | {pooled.loc[f, 'dec_sharpe']:.2f} | {pooled.loc[f, 'dec_net']:.0f} | {pooled.loc[f, 'dec_opw']:.2f} | {100 * val.loc[f]:.0f} % | {win.get(f, 0)} |")
    return "\n".join(lines), {"wins": win, "pooled": pooled.reset_index().to_dict("records")}


def coverage_table(df: pd.DataFrame) -> str:
    if "family" not in df:
        df = df.assign(family="c3")
    g = df.groupby(["variant", "gate", "family", "level"]).agg(seeds=("seed", "nunique"), folds=("fold", "nunique"), refused=("refused", lambda x: int(x.notna().sum()))).reset_index()
    lines = ["| escenario | puerta | familia | calidad | folds | semillas | rechazos |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for _, r in g.iterrows():
        lines.append(f"| {r['variant']} | {r['gate']} | {r['family']} | {r['level']:.2f} | {int(r['folds'])} | {int(r['seeds'])} | {int(r['refused'])} |")
    return "\n".join(lines)


def quarter_table(runs: list) -> str:
    rows = []
    for r in runs:
        if r["variant"] != "base" or r.get("gate") != "perbar":
            continue
        for s in r["per_seed"]:
            for q in (s.get("quarters") or []):
                rows.append({"level": r["level"], "vol": q["vol_tercile"], "trend": q["trend_tercile"], "sharpe": q["sharpe"], "exp": q["expectancy"], "net": q["net_usd"], "opw": q["opw"]})
    if not rows:
        return "(sin desglose por régimen)"
    d = pd.DataFrame(rows)
    lines = ["| régimen (vol × tendencia) | calidad | Sharpe medio | expectativa media σ | neto medio USD | órdenes/sem | trimestres×semillas |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (vol, tr, lv), g in d.groupby(["vol", "trend", "level"]):
        lines.append(f"| vol {vol} / {tr} | {lv:.2f} | {g['sharpe'].mean():.2f} | {g['exp'].mean():.3f} | {g['net'].mean():.0f} | {g['opw'].mean():.2f} | {len(g)} |")
    return "\n".join(lines)


def family_table(df: pd.DataFrame) -> str:
    """The TP/SL x volume families compared on the decision block (base scenario, perbar gate): pooled over seeds and folds."""
    d = df[(df["variant"] == "base") & (df["gate"] == "perbar")]
    if not len(d) or "family" not in d:
        return "(sin comparación de familias)"
    lines = ["| familia (TP/SL, volumen) | calidad | Sharpe | neto USD | expect. σ [IC inf. medio] | órdenes/sem | DD % | cierre temprano USD | rechazos | n |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (fam, lv), g in d.groupby(["family", "level"]):
        ok = g[g["refused"].isna()]
        f = lambda c, k=2: ("—" if not len(ok) or ok[c].isna().all() else f"{ok[c].mean():.{k}f}")  # noqa: E731
        lines.append(f"| {fam} | {lv:.2f} | {f('dec_sharpe')} | {f('dec_net', 0)} | {f('dec_exp', 3)} [{f('dec_exp_lo', 3)}] | {f('dec_opw')} | {f('dec_dd', 1)} | {f('dec_early_close_value', 0)} | {int(g['refused'].notna().sum())} | {len(g)} |")
    return "\n".join(lines)


def coefficient_table(runs: list) -> str:
    """Optimised coefficient values of the FULL families per scenario x quality: mean, min and max over seeds and folds
    (the stability across seeds), with the frozen quantile thresholds."""
    rows = []
    for r in runs:
        if r.get("family", "c3") == "c3":
            continue
        for s in r["per_seed"]:
            g = s.get("chosen_genes")
            if not g:
                continue
            rows.append({"family": r["family"], "variant": r["variant"], "level": r["level"], "fold": r["fold"], "seed": s["seed"], **{k: float(v) for k, v in g.items()}})
    if not rows:
        return "(sin coeficientes optimizados)"
    d = pd.DataFrame(rows)
    genes = [c for c in d.columns if c not in ("family", "variant", "level", "fold", "seed")]
    lines = ["| familia | escenario | calidad | n | " + " | ".join(genes) + " |", "| --- | --- | ---: | ---: | " + " | ".join("---" for _ in genes) + " |"]
    for (fam, v, lv), g in d.groupby(["family", "variant", "level"]):
        cells = [f"{g[k].mean():.2f} [{g[k].min():.2f}, {g[k].max():.2f}]" for k in genes]
        lines.append(f"| {fam} | {v} | {lv:.2f} | {len(g)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


PATH_LABEL = {"base": "camino real sin intervención", "shock_2sigma": "salto persistente 2σ contra la posición", "shock_3sigma": "salto 3σ", "shock_4sigma": "salto 4σ",
              "spread_x5_24h": "spread ×5 durante 24 h", "missing_runs_6_24h": "rachas sin observación de 6–24 h", "vol_x2_in_heads": "ruido de las cabezas ×2 (motor intacto)",
              "regime_splice_trend_range_trend": "régimen tendencia → rango → tendencia (empalme real)"}
MECH_LABEL = {"early_close": "cierre temprano", "nop_veto": "veto NOP", "best_of_day": "puerta mejor del día", "volume": "fórmula de volumen"}


def intervention_table(ints: list) -> str:
    """Per family x level x path x mechanism: loss-limiting ratio (DD with / without) with its CI, the net difference and the
    verdict (TOLERANT when the CI upper bound < 1); plus the strategy's own loss under each intervention vs the base path."""
    if not ints:
        return "(sin batería de intervenciones)"
    lines = ["| familia | calidad | camino | mecanismo | ratio DD con/sin [IC 95 %] | Δ neto medio USD (con − sin) | veredicto | caminos |", "| --- | ---: | --- | --- | --- | ---: | --- | ---: |"]
    for d in sorted(ints, key=lambda x: (x["family"], x["level"], x["fold"])):
        for path, mechs in d["summary"].items():
            for mech, r in mechs.items():
                if mech == "_path":
                    continue
                lines.append(f"| {d['family']} | {d['level']:.2f} | {PATH_LABEL.get(path, path)} ({d['fold']}) | {MECH_LABEL.get(mech, mech)} | {r['ratio']:.2f} [{r['low']:.2f}, {r['high']:.2f}] | {r['net_diff_mean']:+.0f} | {r['verdict']} | {r['n']} |")
    lines += ["", "| familia | calidad | camino | neto medio USD (todo activado) | DD medio USD | neto base | DD base | caminos |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for d in sorted(ints, key=lambda x: (x["family"], x["level"], x["fold"])):
        for path, mechs in d["summary"].items():
            p = mechs.get("_path") or {}
            lines.append(f"| {d['family']} | {d['level']:.2f} | {PATH_LABEL.get(path, path)} ({d['fold']}) | {p.get('net_mean', float('nan')):.0f} | {p.get('dd_usd_mean', float('nan')):.0f} | {p.get('base_net_mean', float('nan')):.0f} | {p.get('base_dd_usd_mean', float('nan')):.0f} | {p.get('paths', 0)} |")
    return "\n".join(lines)


BANDS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.001)]
BAND_RULE = ("the SBFS evaluator measures, per subset, the OUT-OF-FOLD AUC of the TP-first opener heads on the CALIBRATION block (mean of the two "
             "sides) and picks the band containing it; the SL-first closer AUC and the NOP AUC are reported next to it; the decision block is never read; "
             "a measured AUC below 0.50 maps to the lowest band and the strategy is flagged NOT_SELECTABLE_BELOW_CHANCE")


def freeze_params(runs: list, stage_a: dict | None, family: str, gate: str = "perbar", variant: str = "base") -> dict:
    """The frozen parameter table by quality band: for each band, the Stage-B optimum = the median over seeds and folds of
    the chosen genes of the runs whose ACHIEVED calibration-block opener AUC falls in the band (Stage-A robust box applied
    as the gene bounds when present); bands without a run take the nearest run level and say so."""
    rows = []
    for r in runs:
        if r.get("family", "c3") != family or r.get("gate") != gate or r["variant"] != variant:
            continue
        for s in r["per_seed"]:
            g = s.get("chosen_genes")
            ach = ((s.get("simulation") or {}).get("achieved") or {}).get("calibration") or {}
            auc = np.nanmean([ach.get("auc_tp_first_side0", np.nan), ach.get("auc_tp_first_side1", np.nan)])
            if g and np.isfinite(auc):
                rows.append({"level": r["level"], "fold": r["fold"], "seed": s["seed"], "auc_cal": float(auc), "genes": g, "strategy": s.get("strategy_effective"), "stage_b": str(r.get("tag", "")).startswith("stageB"),
                             "dec_exp": ((s.get("decision") or {}).get("expectancy_sigma") or {}).get("expectancy"),
                             "dec_sharpe": ((s.get("decision") or {}).get("metrics") or {}).get("sharpe_daily_sqrt365")})
    table = []
    box = (stage_a or {}).get("robust_region", {}).get("box") or {}
    for lo, hi in BANDS:
        inb = [x for x in rows if lo <= x["auc_cal"] < hi]
        src = "runs_in_band"
        if not inb and rows:
            mid = (lo + hi) / 2
            nearest = min(rows, key=lambda x: abs(x["auc_cal"] - mid))["level"]
            inb = [x for x in rows if x["level"] == nearest]
            src = f"nearest_level_{nearest:.2f}"
        if not inb:
            table.append({"band": [lo, min(hi, 1.0)], "source": "no_run", "n": 0, "genes": None})
            continue
        genes = {}
        for k in sorted({k for x in inb for k in x["genes"]}):              # the NOP veto gene is absent when the veto was dropped
            vals = np.array([x["genes"].get(k, np.nan) for x in inb], float)
            if not np.isfinite(vals).any():
                continue
            med = float(np.nanmedian(vals))
            if k in box and not any(x["stage_b"] for x in inb):
                med = float(min(max(med, box[k][0]), box[k][1]))
            genes[k] = {"median": med, "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)), "n": int(np.isfinite(vals).sum())}
        table.append({"band": [lo, min(hi, 1.0)], "source": src, "n": len(inb), "levels": sorted({x["level"] for x in inb}), "stage_b": bool(all(x["stage_b"] for x in inb)), "genes": genes,
                      "decision_expectancy_mean": float(np.nanmean([x["dec_exp"] for x in inb if x["dec_exp"] is not None])) if any(x["dec_exp"] is not None for x in inb) else None,
                      "decision_sharpe_mean": float(np.nanmean([x["dec_sharpe"] for x in inb if x["dec_sharpe"] is not None])) if any(x["dec_sharpe"] is not None for x in inb) else None,
                      "absolute_strategy_example": inb[0]["strategy"]})
    return {"schema": "strategy_params_frozen.v1", "family": family, "gate": gate, "scenario_source": variant, "band_rule": BAND_RULE, "bands": table,
            "stage_a_robust_box": box, "note": "quantile genes are mapped to absolute thresholds on each subset's own calibration block by the frozen gene map; "
                                              "the coefficients (tp/sl/volume) and observe_bars are used as they are; the cutoff is the execution contract's (24 h)"}


def stage_tables(runs: list, stage_a_docs: dict) -> str:
    """Stage A (Pareto set, robust region box) and Stage B (A-vs-B parameter shift, out-of-block gain / loss with CI)."""
    lines = []
    for fam, sa in stage_a_docs.items():
        rr = sa.get("robust_region") or {}
        lines += [f"### Etapa A — familia `{fam}`: {sa.get('leaves')} hojas, {sa.get('evaluations')} evaluaciones, {sa.get('valid_points')} válidas, frente de Pareto {len(sa.get('pareto') or [])} puntos, región robusta {rr.get('points')} puntos", "",
                  "| gen | caja robusta [mín, máx] | óptimo A (mejor media) |", "| --- | --- | ---: |"]
        best = (sa.get("best_mean") or {}).get("genes") or {}
        for g, b in (rr.get("box") or {}).items():
            lines.append(f"| {g} | [{b[0]:.3f}, {b[1]:.3f}] | {best.get(g, float('nan')):.3f} |")
        lines += ["", "| Pareto: media compuesta | CVaR-10 % |", "| ---: | ---: |"] + [f"| {p['mean']:.3f} | {p['cvar10']:.3f} |" for p in (sa.get("pareto") or [])[:12]] + [""]
        if sa.get("per_regime_best") or rr.get("per_regime_best"):
            prb = rr.get("per_regime_best") or {}
            lines += ["| régimen L1 | mejor media compuesta en el régimen |", "| --- | ---: |"] + [f"| {k} | {v:.3f} |" for k, v in prb.items()] + [""]
    rows = []
    for r in runs:
        if not str(r.get("tag", "")).startswith("stageB"):
            continue
        for s in r["per_seed"]:
            sa = s.get("stage_a_optimum") or {}
            d = (s.get("decision") or {}).get("metrics") or {}
            ex = (s.get("decision") or {}).get("expectancy_sigma") or {}
            rows.append({"family": r.get("family", "c3"), "fold": r["fold"], "level": r["level"], "seed": s["seed"],
                         "B_exp": ex.get("expectancy"), "B_lo": ex.get("low"), "B_hi": ex.get("high"), "B_sharpe": d.get("sharpe_daily_sqrt365"), "B_net": d.get("net_profit_usd"),
                         "A_exp": (sa.get("decision") or {}).get("expectancy"), "A_lo": (sa.get("decision") or {}).get("expectancy_low"), "A_hi": (sa.get("decision") or {}).get("expectancy_high"),
                         "A_sharpe": (sa.get("decision") or {}).get("sharpe_daily_sqrt365"), "A_net": (sa.get("decision") or {}).get("net_profit_usd"),
                         "genes_B": s.get("chosen_genes") or {}, "genes_A": sa.get("genes") or {}})
    if rows:
        d = pd.DataFrame(rows)
        lines += ["### Etapa B — óptimo real dentro de la caja de A frente al óptimo de A replicado en el mismo bloque de decisión", "",
                  "| familia | fold | calidad | semilla | expect. B [IC] | expect. A [IC] | Δ (B − A) | Sharpe B / A | neto B / A USD |", "| --- | --- | ---: | ---: | --- | --- | ---: | --- | --- |"]
        for _, r in d.iterrows():
            f = lambda x: ("—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.3f}")  # noqa: E731
            dl = (r["B_exp"] - r["A_exp"]) if (r["B_exp"] is not None and r["A_exp"] is not None) else float("nan")
            lines.append(f"| {r['family']} | {r['fold']} | {r['level']:.2f} | {r['seed']} | {f(r['B_exp'])} [{f(r['B_lo'])}, {f(r['B_hi'])}] | {f(r['A_exp'])} [{f(r['A_lo'])}, {f(r['A_hi'])}] | {dl:+.3f} | {f(r['B_sharpe'])} / {f(r['A_sharpe'])} | {r['B_net']:.0f} / {r['A_net']:.0f} |")
        lines += ["", "Desplazamiento de parámetros A → B (media sobre semillas y folds por familia y calidad):", ""]
        for (fam, lv), g in d.groupby(["family", "level"]):
            keys = sorted({k for x in g["genes_B"] for k in x})
            lines += [f"| familia {fam}, calidad {lv:.2f} | " + " | ".join(keys) + " |", "| --- | " + " | ".join("---" for _ in keys) + " |"]
            lines.append("| A | " + " | ".join(f"{np.mean([x.get(k, np.nan) for x in g['genes_A']]):.3f}" for k in keys) + " |")
            lines.append("| B | " + " | ".join(f"{np.mean([x.get(k, np.nan) for x in g['genes_B']]):.3f}" for k in keys) + " |")
            lines.append("")
        dd = d.dropna(subset=["B_exp", "A_exp"])
        if len(dd):
            diff = (dd["B_exp"] - dd["A_exp"]).to_numpy()
            rng = np.random.default_rng(20261026)
            bs = np.array([rng.choice(diff, len(diff)).mean() for _ in range(2000)])
            lines += [f"Ganancia fuera de bloque de la re-optimización (B − A, expectativa σ, media sobre {len(diff)} celdas): {diff.mean():+.3f} [IC 95 % {np.quantile(bs, 0.025):+.3f}, {np.quantile(bs, 0.975):+.3f}]", ""]
    return "\n".join(lines) if lines else "(sin etapas A/B)"


def hierarchy_table(catalog: list, leaves_done: dict) -> str:
    if not catalog:
        return "(sin catálogo)"
    lines = ["| hoja | nivel | régimen | vol | fina | variante L4 | calidad | ejecución L5 | comportamiento esperado | testigo | hecha (semillas) |", "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |"]
    for x in catalog:
        done = leaves_done.get(x["id"])
        lines.append(f"| {x['id']} | {x['level']} | {x['regime']} | {x['vol']} | {x['fine']} | {x['variant']} | {x['quality']:.2f} | {json.dumps(x.get('exec') or {})} | {x['expected']} | {x['witness']} | {('sí (' + str(done) + ')') if done else 'no'} |")
    return "\n".join(lines)


def write(df: pd.DataFrame, runs: list, out: Path, docs: str, ints: list | None = None, stage_a_docs: dict | None = None, catalog: list | None = None,
          leaves_done: dict | None = None, contracts_dir: Path | None = None):
    if "family" not in df:
        df = df.assign(family="c3")
    if "tag" not in df:
        df = df.assign(tag="")
    runs_b = [r for r in runs if str(r.get("tag", "")).startswith("stageB")]
    runs = [r for r in runs if not str(r.get("tag", "")).startswith("stageB")]   # the sweep tables never mix the Stage-B points in
    df = df[df["tag"].fillna("").astype(str) == ""]
    df_c3 = df[df["family"] == "c3"]
    p = pooled(df_c3)
    p.to_csv(out / "SWEEP_POOLED.csv", index=False)
    variants = [v for v in ORDER if v in set(p["variant"])]
    figs = {}
    for metric, label in METRICS:
        for gname, gvars in GROUPS.items():
            vs = [v for v in gvars if v in set(p["variant"])]
            if vs:
                figs[f"{metric}:{gname}"] = plot_curves(p, out, vs, metric, label, tag=f"_{gname}")
    figs["mae"] = plot_curves(p, out, ["base"], "dec_sharpe", "Sharpe (año de decisión)", xcol="mae_over_naive_dec", xlabel="MAE / MAE naive de la regresión del neto (año de decisión)")
    figs["mae_net"] = plot_curves(p, out, ["base"], "dec_net", "Neto USD (año de decisión)", xcol="mae_over_naive_dec", xlabel="MAE / MAE naive de la regresión del neto (año de decisión)")
    be = {v: break_even(p, v) for v in variants}
    (out / "BREAK_EVEN.json").write_text(json.dumps(be, indent=1))
    frag = ["## Tablas por variante (media sobre semillas y folds; año de decisión congelado)", ""]
    for v in variants:
        frag += [f"### Variante `{v}`", "", md_table(p, v), ""]
    frag += ["## Umbral de rentabilidad por familia", "", "| variante | nivel de equilibrio (IC inferior medio > 0 y se mantiene) | primer nivel positivo | techo oráculo Sharpe / neto / expect. | suelo azar Sharpe / neto / expect. |",
             "| --- | ---: | ---: | --- | --- |"]
    for v, b in be.items():
        oc, cf = b["oracle_ceiling"], b["chance_floor"]
        g = lambda d, k: ("—" if d.get(k) is None else f"{d[k]:.2f}")  # noqa: E731
        frag.append(f"| {v} | {('—' if b['break_even_level'] is None else f'{b[chr(98)+chr(114)+chr(101)+chr(97)+chr(107)+chr(95)+chr(101)+chr(118)+chr(101)+chr(110)+chr(95)+chr(108)+chr(101)+chr(118)+chr(101)+chr(108)]:.2f}')} | "
                    f"{('—' if b['first_positive_level'] is None else f'{b[chr(102)+chr(105)+chr(114)+chr(115)+chr(116)+chr(95)+chr(112)+chr(111)+chr(115)+chr(105)+chr(116)+chr(105)+chr(118)+chr(101)+chr(95)+chr(108)+chr(101)+chr(118)+chr(101)+chr(108)]:.2f}')} | "
                    f"{g(oc, 'dec_sharpe')} / {g(oc, 'dec_net')} / {g(oc, 'dec_exp')} | {g(cf, 'dec_sharpe')} / {g(cf, 'dec_net')} / {g(cf, 'dec_exp')} |")
    fr = fitness_rows(runs)
    fr.to_csv(out / "FITNESS_ROWS.csv", index=False)
    ft, fsum = fitness_table(fr)
    (out / "FITNESS_SUMMARY.json").write_text(json.dumps(fsum, indent=1, default=float))
    frag += ["", "## Qué fitness usar: cada variante optimizada en el bloque de selección y puntuada UNA vez en el bloque de decisión", "", ft, ""]
    frag += ["## Cobertura del catálogo (escenario × puerta × familia × calidad → folds, semillas, rechazos)", "", coverage_table(df), ""]
    frag += ["## Régimen de mercado: trimestres del año de decisión por terciles de volatilidad y tendencia (base, puerta por barra)", "", quarter_table(runs), ""]
    frag += ["", "## Genes congelados por configuración", "", gene_table(runs), "", "## Sensibilidad de los genes (±0.1 en el cuantil, bloque de selección)", "", sensitivity_table(runs), ""]
    frag += ["## Batería de intervenciones y contrafactuales (bloque de decisión; genes congelados en selección)", "", intervention_table(ints or []), ""]
    frag += ["## Jerarquía de escenarios (catálogo) y cobertura de las hojas sintéticas", "", hierarchy_table(catalog or [], leaves_done or {}), ""]
    frag += ["## Optimización en dos etapas", "", stage_tables(runs_b, stage_a_docs or {}), ""]
    import hashlib
    frozen_all = {}
    for fam in ("c3", "param_param", "cell_param", "param_risk"):
        fz = freeze_params(runs_b or runs, (stage_a_docs or {}).get(fam), fam)
        if any(b.get("genes") for b in fz["bands"]):
            frozen_all[fam] = fz
    if frozen_all and contracts_dir is not None:
        contracts_dir.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"schema": "strategy_params_frozen.v1", "families": frozen_all, "primary_family": ("param_param" if "param_param" in frozen_all else "c3")}, indent=1, sort_keys=True, default=float)
        sha = hashlib.sha256(body.encode()).hexdigest()
        doc = json.loads(body)
        doc["sha256_body"] = sha
        (contracts_dir / "STRATEGY_PARAMS_FROZEN_v1.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        frag += ["## Parámetros congelados por banda de calidad (contrato `STRATEGY_PARAMS_FROZEN_v1.json`)", "", f"sha256 del cuerpo: `{sha}`; regla de banda: {BAND_RULE}", ""]
        for fam, fz in frozen_all.items():
            keys = sorted({k for b in fz["bands"] if b.get("genes") for k in b["genes"]})
            frag += [f"### familia `{fam}`", "", "| banda AUC | fuente | n | " + " | ".join(keys) + " |", "| --- | --- | ---: | " + " | ".join("---" for _ in keys) + " |"]
            for b in fz["bands"]:
                if b.get("genes"):
                    frag.append(f"| [{b['band'][0]:.2f}, {b['band'][1]:.2f}) | {b['source']} | {b['n']} | " + " | ".join((f"{b['genes'][k]['median']:.3f}" if k in b["genes"] else "—") for k in keys) + " |")
            frag.append("")
    frag += ["## Figuras", ""] + [f"- [{label} — variantes de calidad]({docs}/{figs[m + ':quality']})" for m, label in METRICS if m + ":quality" in figs] \
         + [f"- [{label} — variantes de ejecución]({docs}/{figs[m + ':execution']})" for m, label in METRICS if m + ":execution" in figs] \
         + [f"- [Sharpe vs MAE/naive]({docs}/{figs['mae']})", f"- [Neto vs MAE/naive]({docs}/{figs['mae_net']})", ""]
    (out / "SWEEP_FRAGMENT.md").write_text("\n".join(frag))
    return p, be, figs

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


def write(df: pd.DataFrame, runs: list, out: Path, docs: str):
    if "family" not in df:
        df = df.assign(family="c3")
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
    frag += ["## Figuras", ""] + [f"- [{label} — variantes de calidad]({docs}/{figs[m + ':quality']})" for m, label in METRICS if m + ":quality" in figs] \
         + [f"- [{label} — variantes de ejecución]({docs}/{figs[m + ':execution']})" for m, label in METRICS if m + ":execution" in figs] \
         + [f"- [Sharpe vs MAE/naive]({docs}/{figs['mae']})", f"- [Neto vs MAE/naive]({docs}/{figs['mae_net']})", ""]
    (out / "SWEEP_FRAGMENT.md").write_text("\n".join(frag))
    return p, be, figs

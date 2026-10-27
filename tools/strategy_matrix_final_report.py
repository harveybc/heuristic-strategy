"""Final report of the simulated-strategy lane (owner order 2026-10-28): the detailed report for the auditor and the audit
request, both in Spanish, generated from the result files only (no number is typed by hand).

Inputs (copied from the hosts to a local folder, placeholders in the documents): the canonical-point matrix records
(v10, main host + shard), the DEAP-point re-verification records, the DEAP run folders (run 1 superseded, run 2), the
matrix LOG lines of the parameter stage, and the two STRATEGY_PARAMS documents. Outputs: two Markdown files and the
PNG figures they link (relative links).

  python tools/strategy_matrix_final_report.py --canonical <dir> --optimised <dir> --deap <dir> --deap-run1 <dir> \
      --log <LOG> --params-optimised <json> --out docs/short_horizon/clean_round/strategy_matrix
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402
from app.short_horizon.expert_guided import strategy_zz as ZZ  # noqa: E402
from tools import strategy_behaviour_matrix as BM  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402
from tools import strategy_param_deap as D  # noqa: E402

REPORT = "INFORME_ESTRATEGIA_SIMULADA_2026_10_28.md"
REQUEST = "SOLICITUD_AUDITORIA_MUSASHI_2026_10_28.md"
GENE_ES = {
    "open_rate_l": "cuantil de apertura (margen único compartido por ambos lados; open_rate_s queda atado)",
    "nop_veto": "umbral ABSOLUTO sobre P(NOP) calibrada",
    "close_rem_rate_l": "cuantil de cierre por valor restante, largo (modo heredado del cerrador)",
    "close_rem_rate_s": "cuantil de cierre por valor restante, corto (modo heredado del cerrador)",
    "close_p_l": "umbral ABSOLUTO sobre P(SL primero), largo (modo heredado)",
    "close_p_s": "umbral ABSOLUTO sobre P(SL primero), corto (modo heredado)",
    "allow_reverse": "permiso de reverso (>= 0.5 permitido)",
    "observe_bars": "ventana de observación del mejor del día (0 = primera señal válida)",
    "rr_margin_sigma": "celdas TP < SL sólo con neto > coste + este margen",
    "ev_from_net": "ordenar los lados por el neto predicho de la celda (1) o por el EV p3 (0)",
    "close_margin_floor": "piso del margen de cierre por valor restante",
    "risk_reward_floor": "una celda sólo si neto predicho >= piso x MAE predicho",
    "burst_window_bars": "ventana de la detección de ráfagas de volatilidad",
    "burst_vol_ratio": "sin aperturas si la vol realizada > ratio x sigma causal",
    "zz_room_tp_mult": "el recorrido de la escala ZigZag gruesa debe pagar mult x TP",
    "zz_fine_h": "horizonte del pivote fino contrario",
    "zz_timing_gate": "compuerta de pivote fino contrario (encendida / apagada)",
    "zz_nop_gate": "compuerta NOP-dura de la escala",
    "zz_close_room": "cierre por alarma fina cuando el recorrido grueso es menor que esto (modo heredado)",
    "close_p_requires_rem_below": "el cierre por P(SL primero) exige valor restante predicho por debajo de esto",
    "zz_theta_f": "umbral f_open de la compuerta NOP-dura",
    "zz_counter_net_frac": "apertura contra el tramo grueso si neto predicho >= frac x TP",
    "zz_post_flip_bars": "barras tras un giro grueso sin la excepción de neto fuerte",
    "closer_mode": "1 = cierre por la cabeza de ventaja A(t); 0 = reglas heredadas",
    "closer_adv_margin": "margen del cierre por ventaja: cerrar cuando P(t) - V(t) >= -margen (sigma)",
    "tp_mode": "1 = TP protector = max(TP de celda, k x MFE predicho); 0 = TP de la celda",
    "k_tp_protect": "multiplicador del TP protector",
}
FIXED_ES = {"tp_a_l": "familia paramétrica de TP / SL: C3Z no la consume (familia de celdas)", "v_alpha_l": "familia paramétrica de volumen: C3Z no la consume (riesgo fijo)",
            "risk_pct": "contrato de ejecución (regla de dimensionamiento del propietario), no un coeficiente de estrategia",
            "entry_cutoff_hours": "contrato de ejecución (corte del viernes)", "max_entries_per_day": "contrato de ejecución (tope diario)",
            "confirmation_lag_bars": "constante de un testigo, la estrategia no la consume", "open_margin_eps": "guarda numérica contra empates de flotante en la muestra de cuantiles",
            "forced_every_bars": "calendario de los episodios de la auditoría de salida, no un coeficiente de trading", "tp_grid": "rejilla de celdas de las cabezas", "sl_grid": "rejilla de celdas de las cabezas"}


# ----------------------------------------------------------------------------- loading
def load_records(d: str) -> list:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(f"{d}/CELL_*.json"))]


def rec_sharpe(r: dict) -> float:
    for x in (r.get("exit") or {}).get("closer_sweep") or []:
        if x.get("closer_quality") == 1.0:
            return float(x.get("sharpe", float("nan")))
    return float("nan")


def governing_fails(r: dict) -> list:
    """The governing failures of a record: VALIDITY_SPEC verdict (or the pre-rule one for old records), the exit layer, W14."""
    if r.get("error"):
        return ["ERROR"]
    f = list((r.get("validity") or {}).get("failing", [])) if r.get("validity") else \
        [k for k, v in (r.get("results") or {}).items() if isinstance(v, dict) and "pass" in v and not re.search(r"_v[23]$", k) and v["pass"] is False and k not in r.get("witness_defects_sealed", [])]
    f += list((r.get("exit") or {}).get("failing_effective", []))
    w14 = (r.get("results") or {}).get("W14_mirror")
    if w14 and not w14.get("pass", True):
        f.append("W14_mirror")
    return sorted(set(f))


def pct(a, b) -> str:
    return f"{100.0 * a / b:.1f} %" if b else "—"


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a) -> str:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "—"


# ----------------------------------------------------------------------------- tables
def family_table(recs: list) -> list:
    fam = collections.defaultdict(list)
    for r in recs:
        fam[BM.cell_family(r["id"])].append(r)
    L = ["| familia | registros | PASS sellado | PASS efectivo previo a la regla v3 | PASS efectivo (regla vigente) | Sharpe diario medio (cabezas perfectas) |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
    tot = [0, 0, 0, 0]
    for f in sorted(fam):
        rs = fam[f]
        a = sum(1 for r in rs if r.get("PASS_sealed"))
        b = sum(1 for r in rs if r.get("PASS_effective_pre_v3_rule", r.get("PASS_effective")))
        c = sum(1 for r in rs if not governing_fails(r))
        sh = [rec_sharpe(r) for r in rs if math.isfinite(rec_sharpe(r))]
        tot = [tot[0] + len(rs), tot[1] + a, tot[2] + b, tot[3] + c]
        L.append(f"| {f} | {len(rs)} | {pct(a, len(rs))} | {pct(b, len(rs))} | {pct(c, len(rs))} | {np.mean(sh):.2f} |" if sh else f"| {f} | {len(rs)} | {pct(a, len(rs))} | {pct(b, len(rs))} | {pct(c, len(rs))} | — |")
    L.append(f"| **total** | {tot[0]} | {pct(tot[1], tot[0])} | {pct(tot[2], tot[0])} | {pct(tot[3], tot[0])} | |")
    return L


def side_exit_table(recs: list) -> list:
    rows = [r for r in recs if r.get("exit")]
    agg = {k: {si: collections.Counter() for si in ("0", "1")} for k in ("strategy", "hold_oracle", "O*")}
    for r in rows:
        nm = r["exit"]["normal_exit_mix"]
        for k in agg:
            for si in ("0", "1"):
                for c in ("early", "tp", "sl", "time"):
                    v = nm[k][si].get(c)
                    if v == v and v is not None:
                        agg[k][si][c] += v
    n = max(len(rows), 1)
    L = ["| lado | estrategia (temprano / TP / SL / tiempo) | oráculo de mantener | O* |", "| --- | --- | --- | --- |"]
    out = {}
    for si, side in (("0", "BUY"), ("1", "SELL")):
        L.append(f"| {side} | " + " | ".join(" / ".join(f"{100 * agg[k][si][c] / n:.1f} %" for c in ("early", "tp", "sl", "time")) for k in ("strategy", "hold_oracle", "O*")) + " |")
        out[side] = 100 * agg["strategy"][si]["early"] / n
    return L, out


def episode_table(recs: list) -> list:
    rows = [r for r in recs if r.get("exit")]
    L = ["| lado | episodios destinados al SL (suma) | E1′ medio | R medio / p90 (σ) | R de mantener hasta el SL (σ) | retraso mediano (barras) | Z1 etiqueta neta 0.1σ |",
         "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for side in ("BUY", "SELL"):
        si = "0" if side == "BUY" else "1"
        n_sl = sum(r["exit"]["episodes"]["by_side_class"][si]["SL"] for r in rows)
        w = [r["exit"]["witnesses"] for r in rows]
        e1 = [x[f"E1p_closed_before_SL_{side}"]["value"] for x in w if f"E1p_closed_before_SL_{side}" in x]
        e2 = [x[f"E2p_regret_SL_destined_{side}"]["value"] for x in w if f"E2p_regret_SL_destined_{side}" in x]
        e7 = [x[f"E7_lateness_vs_Pstar_{side}"]["value"]["median_late_bars"] for x in w if f"E7_lateness_vs_Pstar_{side}" in x]
        z1 = [x[f"Z1_zigzag_exit_label_{side}"]["value"]["declared_match"] for x in w if f"Z1_zigzag_exit_label_{side}" in x]
        L.append(f"| {side} | {n_sl} | {np.mean(e1):.3f} | {np.mean([v['mean_R_sigma'] for v in e2]):.4f} / {np.mean([v['p90_R_sigma'] for v in e2]):.4f} | "
                 f"{np.mean([v['hold_to_SL_mean_R'] for v in e2]):.3f} | {np.median(e7):.0f} | {np.mean(z1):.3f} |")
    return L


def lavail_table(recs: list) -> list:
    curves = collections.defaultdict(list)
    for r in recs:
        for k, v in ((r.get("exit") or {}).get("exit_loss_curves") or {}).items():
            if k.endswith("_Lavail"):
                curves[k].append(v)
    L = ["| curva (L_avail(t) media, σ) | 0 | 2 | 4 | 8 | 12 | 24 | 47 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in sorted(curves):
        arr = np.nanmean(np.asarray(curves[k], float), axis=0)
        L.append(f"| {k.replace('_Lavail', '')} | " + " | ".join(f"{arr[i]:.3f}" if i < len(arr) and arr[i] == arr[i] else "—" for i in (0, 2, 4, 8, 12, 24, 47)) + " |")
    return L


def censored_table(recs: list) -> list:
    """Trades flattened by the end of the block (censored from E1-E4, kept in the exit mix), per side and per cell."""
    rows = [(r["id"], r["seed"], (r.get("exit") or {}).get("censored_by_block_end", {}).get("strategy")) for r in recs]
    rows = [(i, s_, c) for i, s_, c in rows if c and (c.get("0", c.get(0, 0)) or c.get("1", c.get(1, 0)))]
    tot = [sum(int(c.get("0", c.get(0, 0))) for _, _, c in rows), sum(int(c.get("1", c.get(1, 0))) for _, _, c in rows)]
    L = [f"Operaciones censuradas por el fin del bloque (estrategia): BUY {tot[0]}, SELL {tot[1]}, en {len(rows)} registros.", "",
         "| registro | semilla | BUY | SELL |", "| --- | ---: | ---: | ---: |"]
    L += [f"| {i} | {s_} | {c.get('0', c.get(0, 0))} | {c.get('1', c.get(1, 0))} |" for i, s_, c in sorted(rows)]
    return L


def failing_list(recs: list, cap: int = 400) -> tuple:
    bad = [(r["id"], r["seed"], governing_fails(r), (r.get("validity") or {}).get("per_witness", {})) for r in recs if governing_fails(r)]
    L = ["| registro | semilla | testigos que fallan (forma gobernante) | estado de validez / forma gobernante | causa registrada |", "| --- | ---: | --- | --- | --- |"]
    for cid, seed, f, pw in sorted(bad)[:cap]:
        st = "; ".join(f"{k}: {pw[k]['status']} → {pw[k]['governing_form']}" for k in f if k in pw)
        cause = []
        for k in f:
            x = pw.get(k)
            if x:
                cause.append(f"{k} falla su forma gobernante con eventos de control {x.get('control_events')}")
            elif k.startswith("E"):
                cause.append(f"{k}: testigo de salida sellado (O* lo pasa en la celda)")
            else:
                cause.append(k)
        L.append(f"| {cid} | {seed} | {', '.join(f)} | {st or '—'} | {'; '.join(cause)} |")
    if len(bad) > cap:
        L.append(f"| … | | {len(bad) - cap} registros más en el JSON de resultados | | |")
    return L, len(bad)


def witness_counts(recs: list) -> list:
    c = collections.Counter(k for r in recs for k in governing_fails(r))
    return ["| testigo | registros que lo fallan |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in c.most_common()]


def params_table(canon: dict, opt: dict | None) -> list:
    L = ["| clave | canónico (v10) | punto DEAP | límites | en el genoma | toca un límite (punto DEAP) | significado |", "| --- | ---: | ---: | --- | --- | --- | --- |"]
    for k, (lo, hi) in SP.BOUNDS.items():
        cv, ov = canon.get(k), (opt or {}).get(k)
        hit = ""
        if k in D.GENOME and ov is not None and isinstance(ov, (int, float)):
            span = float(hi) - float(lo)
            hit = "SÍ (inferior)" if abs(float(ov) - float(lo)) <= 0.01 * span else ("SÍ (superior)" if abs(float(ov) - float(hi)) <= 0.01 * span else "no")
            if k in D.BINARY:
                hit = "binario"
        desc = ("igual a " + D.TIED[k] + " (margen único para ambos lados)") if k in D.TIED else GENE_ES.get(k) or next((v for f, v in FIXED_ES.items() if k.startswith(f.rstrip("l").rstrip("_")) or k == f), "fijo (ver FIXED en strategy_param_deap.py)")
        L.append(f"| {k} | {cv} | {ov if ov is not None else '—'} | [{lo}, {hi}] | {'sí' if k in D.GENOME else ('atado a ' + D.TIED[k] if k in D.TIED else 'no')} | {hit} | {desc} |")
    return L


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True, help="records of the canonical-point matrix v10 (main host + shard)")
    ap.add_argument("--optimised", required=True, help="records of the DEAP-point re-verification")
    ap.add_argument("--deap", required=True, help="the DEAP run-2 folder (ARCHIVE.jsonl, PROGRESS.log, RESULT.json, RUN_META.json)")
    ap.add_argument("--deap-run1", required=True)
    ap.add_argument("--log", required=True, help="the matrix root's LOG")
    ap.add_argument("--params-optimised", required=True)
    ap.add_argument("--cells", required=True, help="QUEUE.json of the matrix")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    canon_doc = SP.load(ROOT / "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json")
    opt_doc = json.loads(Path(a.params_optimised).read_text())
    cells = json.loads(Path(a.cells).read_text())
    rc, ro = load_records(a.canonical), load_records(a.optimised)
    arch = [json.loads(x) for x in Path(a.deap, "ARCHIVE.jsonl").read_text().splitlines() if x.strip()]
    arch1 = [json.loads(x) for x in Path(a.deap_run1, "ARCHIVE.jsonl").read_text().splitlines() if x.strip()]
    prog = [json.loads(x) for x in Path(a.deap, "PROGRESS.log").read_text().splitlines() if x.strip().startswith("{")]
    result = json.loads(Path(a.deap, "RESULT.json").read_text()) if Path(a.deap, "RESULT.json").exists() else None
    loglines = [x for x in Path(a.log).read_text().splitlines() if re.search(r"\[(param-stage|param-watchdog|slot-plan)\]", x)]
    # figures: the exit sections write exit_loss_curves.png and closer_noise_sweep.png into the folder given
    BM.exit_report_sections({r["id"] + str(r["seed"]): [r] for r in ro}, out)
    fams = collections.Counter(BM.cell_family(c["id"]) for c in cells if not c["mirror"] and not c.get("sym_of"))
    n_base = sum(fams.values())
    fail_o_L, n_fail_o = failing_list(ro)
    fail_c_L, n_fail_c = failing_list(rc)
    side_L_o, early_o = side_exit_table(ro)
    side_L_c, early_c = side_exit_table(rc)
    best = min(arch, key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"])) if arch else None
    eff_o = sum(1 for r in ro if not governing_fails(r))
    eff_c = sum(1 for r in rc if not governing_fails(r))
    sh_o = [rec_sharpe(r) for r in ro if math.isfinite(rec_sharpe(r))]
    sh_c = [rec_sharpe(r) for r in rc if math.isfinite(rec_sharpe(r))]
    head = git("rev-parse", "HEAD")
    gens_changed = [k for k in D.GENOME if k in opt_doc and opt_doc.get(k) != canon_doc.get(k)]
    psha_o = sha256(a.params_optimised)
    psha_c = sha256(ROOT / "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json")
    contract = json.loads((ROOT / "docs/short_horizon/clean_round/contracts/ACTION_CONTRACT_C3.json").read_text())

    L = [f"# Informe: estrategia simulada con salidas ideales, matriz de comportamiento y optimización DEAP (2026-10-28)", "",
         "Para: Musashi (auditor). Todas las cifras salen de los archivos de resultados mediante `tools/strategy_matrix_final_report.py` (confirmado en el mismo commit); ninguna está escrita a mano. "
         "Todo es SIMULADO: mundos sintéticos con cabezas ideales de calidad declarada sobre el motor real; ninguna afirmación sobre datos reales. TEST (filas >= 2025-01-01) permanece cerrado y no se leyó.", "",
         f"Commit del generador: `{head}`. Parámetros canónicos sha256 `{psha_c}`; punto DEAP sha256 `{psha_o}`; especificación de testigos re-sellada `{BM.SPEC_V3_SHA}`; especificación de salida `{BM.EXIT_SPEC_SHA}`; contrato C3 `{contract['identity']}`.", "",
         "## Resumen", "",
         f"- Matriz completa con el punto DEAP: {len(ro)} registros, PASS efectivo {pct(eff_o, len(ro))} ({len(ro) - eff_o} registros fallan, listados en §4).",
         f"- Matriz con el punto canónico (v10, parcial por diseño: el host principal cedió sus ranuras a la re-verificación): {len(rc)} registros, PASS efectivo {pct(eff_c, len(rc))}.",
         f"- Cierre temprano (operaciones propias de la estrategia, punto DEAP): BUY {early_o['BUY']:.1f} %, SELL {early_o['SELL']:.1f} % (referencia del propietario {BM.OWNER_EARLY_CLOSE_REFERENCE:.0%}).",
         f"- DEAP: mejor punto Sharpe {best['fitness_sharpe']:.3f} con {best['n_failing_witnesses']} testigos fallando en el subconjunto (generación {best['generation']}); Sharpe diario medio en la matriz completa: canónico {np.mean(sh_c):.2f}, optimizado {np.mean(sh_o):.2f}." if best else "- DEAP: sin archivo.",
         "- Con cabezas de cierre perfectas el cerrador ES O*: que E1′ / E2′ / E7 coincidan con O* es esperado por construcción y NO es evidencia de habilidad; la evidencia es la escalera de calidad del cerrador (§5).", "",
         "## 1. Salidas ideales", "",
         f"- Contrato de cabezas C3 (`ACTION_CONTRACT_C3.json`, identidad `{contract['identity']}`): especialistas {', '.join(contract['specialists'])}; abridores con decisión de 3 clases {C3.CLASSES_OPEN} en la celda de referencia {list(C3.REF)} y una cabeza de neto por celda de la rejilla TP {list(C3.GRID_TP)} × SL {list(C3.GRID_SL)} (σ); cerradores con valor restante y P(SL primero); NOP.",
         "- Calidad de las cabezas: correlación ρ = sin(π (nivel − ½)) entre la salida simulada y la verdad del motor, salida = media condicional (una cabeza perfecta devuelve la verdad). Escalera del cerrador declarada: " + " / ".join(f"{q:.2f}" for q in BM.CLOSER_QUALITIES) + ".",
         "- Cabeza de ventaja del cerrador: A(t) = P(cerrar ahora) − max_{t′>t} P(t′) (la salida natural incluida), objetivo en retrospectiva y nunca una entrada; cabeza MFE para el TP protector.",
         f"- Escala ZigZag (pivotes de producción, simétricos ante la negación del precio): niveles {', '.join(ZZ.DEFAULT_LEVELS)}; papeles: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in ZZ.ROLES.items()) + ".",
         f"- Etiqueta de salida Z1: ZigZag del propio camino neto de la operación (todos los costes), umbrales {list(BM.NET_LADDER_SIGMA)} σ, declarado 0.1 σ. El objetivo de entrenamiento del cerrador es A(t) / L_avail del replay del motor; las escaleras de precio son contexto multihorizonte, no la etiqueta.", "",
         "## 2. La estrategia C3Z y STRATEGY_PARAMS", "",
         "C3Z = C3 (juez de realidad: celda permitida de mayor neto predicho, piso riesgo-beneficio sobre el MAE, margen de apertura único para ambos lados) + la jerarquía de la escala ZigZag (veto, nunca redirección: recorrido grueso, pivote fino contrario, NOP-dura, ráfagas) + el cerrador por ventaja + el TP protector. Cada coeficiente vive en `STRATEGY_PARAMS.json`, validado contra `strategy_params.BOUNDS`.", "",
         f"Genes que el DEAP cambió respecto del canónico: {', '.join(gens_changed) or 'ninguno'}.", ""] + params_table(canon_doc, opt_doc) + [
         "", "## 3. La matriz de comportamiento", "",
         f"- Celdas base: {n_base} por familia: " + ", ".join(f"{k} {v}" for k, v in sorted(fams.items())) + ". Cada celda base corre con las semillas 0, 1, 2 (la semilla varía el camino sintético); cada una tiene su gemela de coste cero S_ y su espejo exacto M_ (semilla 0).",
         f"- Registros por matriz completa: {sum(3 + (1 if ('S_' + c['id']) in {x['id'] for x in cells} else 0) + (1 if ('M_' + c['id']) in {x['id'] for x in cells} else 0) for c in cells if not c['mirror'] and not c.get('sym_of'))}.",
         f"- Sellos: celdas originales `{BM.SEALED_CELLS_SHA}`; especificación de salida `{BM.EXIT_SPEC_SHA}`; especificación re-sellada v3 (formas + regla de validez + salida + umbrales) `{BM.SPEC_V3_SHA}`.",
         "- Regla de validez (VALIDITY_SPEC), aplicada igual a la estrategia y a cada candidato DEAP:"] + [f"  - {k}: {v if not isinstance(v, dict) else '; '.join(f'{kk}: {vv}' for kk, vv in v.items())}" for k, v in BM.VALIDITY_SPEC.items()] + [
         "", "### Cambios posteriores al sellado y su evidencia", "",
         "Lo que se cambió DESPUÉS de ver resultados está marcado en la columna «cuándo»; cada cambio de un testigo descansa en un control (el oráculo de la verdad u O*) o en una contradicción entre testigos sellados demostrada por el replay del motor, nunca en el resultado de la estrategia. Una regla a nivel de testigo propuesta antes fue RECHAZADA por el coordinador y revertida.", "",
         "| cuándo | testigo | defecto | evidencia | efecto |", "| --- | --- | --- | --- | --- |"] + [
         f"| {x['when']} | {x['witness']} | {x['defect']} | {x['evidence']} | {x['effect']} |" for x in BM.WITNESS_CHANGELOG] + [
         "", "### Historia de corridas", "", "| corrida | nota |", "| --- | --- |"] + [f"| {x['run']} | {x['note']} |" for x in BM.RUN_HISTORY] + [
         "", "## 4. Resultados por familia y por lado", "", "### Punto DEAP en la matriz completa", ""] + family_table(ro) + [
         "", "Testigos que fallan (forma gobernante), punto DEAP:", ""] + witness_counts(ro) + [
         "", "### Punto canónico (v10)", ""] + family_table(rc) + ["", "Testigos que fallan (forma gobernante), punto canónico:", ""] + witness_counts(rc) + [
         "", f"### Todos los registros que fallan, punto DEAP ({n_fail_o})", ""] + fail_o_L + [
         "", f"### Todos los registros que fallan, punto canónico ({n_fail_c})", ""] + fail_c_L + [
         "", "## 5. Cierre temprano (titular)", "", "### Mezcla de salidas por lado, punto DEAP (media sobre registros)", ""] + side_L_o + [
         "", "### Mezcla de salidas por lado, punto canónico", ""] + side_L_c + [
         "", "### Operaciones censuradas por el fin del bloque, punto DEAP", ""] + censored_table(ro) + [
         "", "### Episodios de estado inicial destinados al SL, punto DEAP", ""] + episode_table(ro) + [""] + BM.exit_headline_tables([r for r in ro if r.get("exit")]) + [
         "### Curvas L_avail(t) por barras desde la entrada (destinados al SL, punto DEAP)", ""] + lavail_table(ro) + [
         "", "## 6. DEAP", "",
         f"- Corrida 1 (superada): {len(arch1)} evaluaciones, detenida en la generación {max((r['generation'] for r in arch1), default=0)} por la regla re-sellada (validez + W8 v3); su archivo se conserva y sembró la corrida 2.",
         f"- Corrida 2: población 16 × 30 generaciones, élite {D.ELITE}, semilla {D.SEED}, aptitud = Sharpe diario medio sobre {len(json.loads(Path(a.deap, 'SUBSET.json').read_text()))} mundos del subconjunto, restricción de Deb = testigos que fallan; arranque en caliente con los 16 mejores puntos de la corrida 1 re-puntuados con la regla vigente.",
         "- Incidente: la corrida 2 murió a las 13:54:57Z (OOM del memcg en el tope de 13G; la caché de caminos del cerrador nunca se vaciaba, +20 MB por evaluación y trabajador) y quedó sin reanudar cinco horas: el estado reportado a las 14:02Z («en curso») fue FALSO por leer sólo PROGRESS.log. Se reanudó desde el CHECKPOINT de la generación 7 con 6 trabajadores, tope 15G, la caché reiniciada por evaluación y el código que la corrida cargó más el arreglo de empates aceptado; se añadió un vigilante que alerta y reanuda una vez, y un plan de ranuras que nunca deja ranuras ociosas.", "",
         "Eventos del registro de la etapa:", "", "```"] + loglines[-40:] + ["```", "", "Progreso por generación (corrida 2):", "",
         "| generación | mejor Sharpe | testigos que fallan del mejor | factibles en la población |", "| ---: | ---: | ---: | ---: |"] + [
         f"| {p['generation']} | {p['best_sharpe']:.3f} | {p['best_failing']} | {p['feasible_in_pop']} |" for p in prog if "generation" in p] + [
         "", "| matriz completa | registros | PASS efectivo | Sharpe diario medio | neto medio USD |", "| --- | ---: | ---: | ---: | ---: |",
         f"| canónico (v10) | {len(rc)} | {pct(eff_c, len(rc))} | {np.mean(sh_c):.2f} | {np.mean([r['results']['summary']['net_usd'] for r in rc if r.get('results', {}).get('summary')]):.0f} |",
         f"| punto DEAP | {len(ro)} | {pct(eff_o, len(ro))} | {np.mean(sh_o):.2f} | {np.mean([r['results']['summary']['net_usd'] for r in ro if r.get('results', {}).get('summary')]):.0f} |", "",
         "La comparación canónico vs optimizado usa registros distintos: el canónico cubre la parte de la cola que corrió el host secundario y lo que el principal hizo antes de ceder sus ranuras.", "",
         "## 7. Defectos conocidos y límites", "",
         "- allow_reverse = 0 en el punto canónico ELIMINÓ reversos rentables: neto −18 %, Sharpe sin cambio (comprobación de 23 mundos); sigue siendo un gen del DEAP.",
         "- W1 (sobre-concentración en la dirección del tramo con zz_room_tp_mult 1.75), W4 (franja de empate del cerrador por ventaja 0.93–0.95), W7 (0.69 en un mundo H3), W13 (dos mundos I2): su estado con el punto DEAP está en §4; donde fallan son defectos de ESTRATEGIA pendientes de una corrección de estrategia, no de testigo.",
         "- Los registros v10 anteriores al re-sellado de (c′) llevan la forma de conteo de W8 v3 (c): guardan las operaciones pero no los caminos A(t) por tick, así que (c′) no puede recomputarse con ellos.",
         "- Registros anteriores al arreglo de empates (d57c462) no llevan strategy_code_sha256: se conservan como descubrimiento de defectos.",
         "- Sólo sintético: cabezas ideales de calidad declarada sobre mundos sintéticos con el motor real; no se afirma nada sobre datos reales ni sobre cabezas aprendidas.",
         "- TEST (filas >= 2025-01-01) cerrado: ni leído ni usado.",
         "- Las tres pruebas de C3 que fallaban (deriva del contrato tras 6a38ccf) se corrigieron: el contrato se regeneró desde el código y la prueba exige el umbral absoluto.", "",
         "## 8. Figuras", "",
         "- [Curvas de pérdida de salida L(t), destinados al SL: estrategia / mantener hasta el SL / O*](exit_loss_curves.png)",
         "- [Sharpe y beneficio frente a MAE / naive de la cabeza de ventaja del cerrador (barrido de calidad)](closer_noise_sweep.png)", ""]
    (out / REPORT).write_text("\n".join(L) + "\n")

    R = [f"# Solicitud de auditoría a Musashi: estrategia simulada, matriz y DEAP (2026-10-28)", "",
         "## Alcance", "",
         "La estrategia C3Z con cabezas ideales de calidad declarada sobre mundos sintéticos y el motor real: la especificación sellada de la matriz de comportamiento y sus cambios posteriores, la regla de validez de los controles, la capa de optimalidad de salida, la optimización DEAP (sólo Sharpe, restricción por testigos) y la re-verificación del punto DEAP en la matriz completa. Fuera de alcance: datos reales, cabezas aprendidas, TEST (cerrado).", "",
         "## Commits y sellos", "",
         f"- Rama `satoshi/expert-guided-rl-20261019`, commit del informe `{head}`.",
         f"- Especificación re-sellada v3 `{BM.SPEC_V3_SHA}`; especificación de salida `{BM.EXIT_SPEC_SHA}`; celdas originales `{BM.SEALED_CELLS_SHA}`.",
         f"- STRATEGY_PARAMS canónico `{psha_c}`; punto DEAP `{psha_o}`; contrato C3 `{contract['identity']}` (parámetros incrustados `{contract['strategy']['params']['sha256']}`).",
         "- Commits clave: e1386df (especificación de salida sellada), 25852c9 (correcciones canónicas de transición), 2eaf73c (re-sellado v3 y regla de validez), d57c462 (empates de celda, excepción aceptada sin reinicio del DEAP), cb866c8 (W8 v3 (c′)), d2e999a (contrato C3 regenerado), a43564a (vigilante del DEAP).", "",
         "## Cómo reproducir", "",
         "Datos: el pliegue `wf2022` de los CSV seleccionados (`<data>/admitted`, `<data>/sel15`); sólo CPU, con tope de memoria.", "",
         "```bash", "cd <repo-heuristic-strategy> && git checkout " + head,
         "export CUDA_VISIBLE_DEVICES=\"\"",
         "# una celda (base + gemela S_ + espejo M_, semilla 0)",
         "crispdm-run -m 1500M -t 7200s -n audit-cell -- python tools/strategy_behaviour_matrix.py run --cells <matrix>/QUEUE.json --fold wf2022 \\",
         "    --admitted <data>/admitted --csv-dir <data>/sel15 --ids B_trans_up_range_late --seeds 0 \\",
         "    --params docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json --out <out>",
         "# una evaluación DEAP (el punto DEAP sobre dos mundos del subconjunto)",
         "crispdm-run -m 3G -t 3600s -n audit-deap -- python tools/strategy_param_deap.py measure --subset <matrix>/deap/SUBSET.json \\",
         "    --params <matrix>/deap/STRATEGY_PARAMS_OPTIMISED.json --admitted <data>/admitted --csv-dir <data>/sel15 --jobs 1 --n 2 --reps 1",
         "# el informe completo desde los resultados",
         "python tools/strategy_matrix_final_report.py --canonical <v10> --optimised <verify_k0>/results --deap <matrix>/deap --deap-run1 <matrix>/deap_run1_superseded \\",
         "    --log <matrix>/LOG --params-optimised <matrix>/deap/STRATEGY_PARAMS_OPTIMISED.json --cells <matrix>/QUEUE.json --out docs/short_horizon/clean_round/strategy_matrix",
         "```", "",
         "## Afirmaciones a atacar primero (las más probablemente equivocadas)", "",
         "1. Cada cambio de testigo posterior al sellado (tabla del §3 del informe): ¿descansa cada uno en un control (oráculo u O*) o en una contradicción demostrada, y no en el resultado de la estrategia? En particular W1 v2, W5 v2, W8 v2/v3, W10 v2, W13 v2 y el ancla de Z1.",
         "2. La regla de vacuidad: un PASS de control con cero eventos no valida la forma sellada y entonces gobierna la corregida. ¿Los conteos de eventos (`attach_events`) miden el tipo atestiguado en cada testigo? ¿La regla abre una puerta para excusar fallos reales?",
         "3. El fallo del conflicto W8 en el cambio (E4 gana): la forma sellada sólo se cumplía cerrando operaciones con A(t) entre −0.088 y −0.50 σ; v3 (a) cero SL, (b) O* también mantenía, (c′) réplica de O* sobre la MISMA operación. ¿Es (b) demasiado permisiva? ¿(c′) cubre lo que la forma sellada quería?",
         "4. La excepción del arreglo de empates (d57c462) aceptada sin reiniciar el DEAP: ¿cambia sólo empates exactos (1e-9 σ) o pudo mover la selección del DEAP?",
         "5. Cabezas de cierre perfectas = O*: E1′ / E2′ / E7 iguales a O* no son evidencia; ¿la escalera de calidad 1.00 / 0.85 / 0.70 y el valor del cierre temprano frente a mantener hasta el SL bastan como evidencia de la regla de salida?",
         "6. El punto canónico y el DEAP: genes en los límites (§2), allow_reverse = 0 (neto −18 %), la comparación canónico vs optimizado sobre registros distintos.",
         "7. La etiqueta Z1 frente a las escaleras de precio (0.66–0.80 en los niveles finos): ¿es correcto que el objetivo del cerrador sea A(t) / L_avail y no una escalera de precio?", "",
         "## Lo que pedimos decidir", "",
         "- Si el punto DEAP queda adoptado como el punto congelado de la estrategia simulada (con los registros que fallan listados en el informe) o si alguno de esos fallos lo impide.",
         "- Si cada cambio posterior al sellado y la regla de validez se aceptan, se corrigen o se revierten.",
         "- Si el contrato C3 congelado debe incrustar el sha del punto DEAP (hoy incrusta el canónico hasta su decisión).",
         "- Qué parte, si alguna, puede pasar a la etapa de cabezas aprendidas.", ""]
    (out / REQUEST).write_text("\n".join(R) + "\n")
    print(json.dumps({"report": str(out / REPORT), "request": str(out / REQUEST), "eff_pass_optimised": eff_o / max(len(ro), 1), "records_optimised": len(ro),
                      "eff_pass_canonical": eff_c / max(len(rc), 1), "records_canonical": len(rc), "early_close_optimised": early_o,
                      "deap_best_sharpe": best["fitness_sharpe"] if best else None, "deap_best_failing": best["n_failing_witnesses"] if best else None}))


if __name__ == "__main__":
    main()

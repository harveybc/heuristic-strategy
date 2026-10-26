#!/bin/bash
# Collect a strategy-simulation work root (runs, interventions, hier, truth meta) into the repo's evidence directory
# within the 20 MB budget, write the manifest (sha256 + size of EVERY file of the root, copied or not), generate the
# report (tables, figures, frozen contract) and splice the generated fragment into the two documents.
# Usage: collect_strategy_sim.sh <mirror of the work root> <checkout>
set -eu
SRC=$1; CODE=$2
PY=${PY:-python}
EV=$CODE/docs/short_horizon/clean_round/evidence/strategy_sim
DOC=$CODE/docs/short_horizon/clean_round/STRATEGY_IDEAL_OUTPUT_SIMULATION_2026_10_26.md
REQ=$CODE/docs/short_horizon/clean_round/MUSASHI_AUDIT_REQUEST_STRATEGY_2026_10_26.md
BUDGET=${BUDGET:-20000000}
mkdir -p $EV/runs $EV/interventions $EV/hier
cd $CODE
# 1. report straight from the mirror (figures, CSV, JSON, fragment, frozen contract)
CUDA_VISIBLE_DEVICES="" $PY tools/strategy_output_simulation.py report --runs $SRC/runs --interventions $SRC/interventions --hier $SRC/hier \
   --contracts docs/short_horizon/clean_round/contracts --docs evidence/strategy_sim --out $EV
# 2. compressed copies within the budget: run records first, then interventions, then the Stage-A documents, then archives
$PY - "$SRC" "$EV" "$BUDGET" <<'EOF'
import gzip, hashlib, json, os, shutil, sys
from pathlib import Path
src, ev, budget = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
used = sum(p.stat().st_size for p in ev.rglob("*") if p.is_file() and p.suffix != ".gz")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
manifest = []
order = [("runs", "RUN_*.json"), ("interventions", "INT_*.json"), ("hier", "STAGE_A_*.json"), ("hier", "CATALOG.json"),
         ("hier", "*_archive.json.gz"), ("runs/ledger", "ARCHIVE_*.json.gz"), ("runs/ledger", "*.csv.gz"), ("truth", "*.json")]
copied = set()
for sub, pat in order:
    for p in sorted((src / sub).glob(pat)) if (src / sub).exists() else []:
        dst = ev / sub / (p.name if p.suffix == ".gz" else p.name + ".gz")
        if p.suffix == ".gz":
            size = p.stat().st_size
            if used + size > budget:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dst)
        else:
            data = gzip.compress(p.read_bytes(), 9)
            if used + len(data) > budget:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data)
        used += dst.stat().st_size
        copied.add(str(p.relative_to(src)))
for p in sorted(src.rglob("*")):
    if p.is_file() and not p.name.startswith(".") and "/code/" not in str(p) and "/data/" not in str(p):
        rel = str(p.relative_to(src))
        manifest.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha(p), "in_git": rel in copied})
(ev / "MANIFEST.json").write_text(json.dumps({"schema": "strategy_sim_manifest.v1", "files": manifest, "budget_bytes": budget, "git_bytes": used,
                                              "note": "every file of the lane's work root, with sha256; in_git marks the compressed copies committed under evidence/strategy_sim"}, indent=1))
print(json.dumps({"files": len(manifest), "in_git": len(copied), "git_bytes": used}))
EOF
# 3. splice the fragment into the document and the artifact table into the audit request
$PY - "$DOC" "$REQ" "$EV" "$CODE" <<'EOF'
import json, re, sys
from pathlib import Path
doc, req, ev, code = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
frag = (ev / "SWEEP_FRAGMENT.md").read_text()
# the per-configuration gene table (one row per fold x variant x gate x level x seed) stays in the evidence fragment
a = frag.find("## Genes congelados por configuración")
b = frag.find("## Sensibilidad de los genes")
if 0 <= a < b:
    frag = frag[:a] + "## Genes congelados por configuración\n\nTabla completa (una fila por fold × escenario × puerta × calidad × semilla) en `evidence/strategy_sim/SWEEP_FRAGMENT.md`; los valores por corrida están en `evidence/strategy_sim/runs/RUN_*.json.gz` (`chosen_genes`, `strategy_effective`).\n\n" + frag[b:]
frag = frag.replace("\n## ", "\n### ")
if frag.startswith("## "):
    frag = "#" + frag
t = doc.read_text()
start, end = t.index("<!-- SWEEP_FRAGMENT -->"), None
m = re.search(r"\n## 8\. ", t)
t = t[:start] + "<!-- SWEEP_FRAGMENT -->\n\n" + frag.strip() + "\n" + t[m.start():]
doc.write_text(t)
man = json.loads((ev / "MANIFEST.json").read_text())
fz = json.loads((code / "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS_FROZEN_v1.json").read_text())
rows = ["## 3. Dónde está cada artefacto", "",
        f"`evidence/strategy_sim/MANIFEST.json` lista **todos** los archivos del directorio de trabajo de la lane ({len(man['files'])} archivos) con su sha256 y tamaño; "
        f"los marcados `in_git` ({sum(1 for f in man['files'] if f['in_git'])}) están comprimidos bajo `evidence/strategy_sim/` ({man['git_bytes'] / 1e6:.1f} MB); el resto se verifica contra el sha en el directorio de trabajo.", "",
        "| artefacto | ruta | contenido |", "| --- | --- | --- |",
        "| contrato congelado | `contracts/STRATEGY_PARAMS_FROZEN_v1.json` | genes por banda de AUC y familia; `sha256_body` = `" + fz["sha256_body"] + "` |",
        "| tablas pooled | `evidence/strategy_sim/SWEEP_POOLED.csv`, `SWEEP_ROWS.csv` | media sobre semillas y folds por variante × puerta × nivel; filas por semilla |",
        "| equilibrio | `evidence/strategy_sim/BREAK_EVEN.json` | calidad de equilibrio por variante (raíz de la expectativa pooled) |",
        "| fitness | `evidence/strategy_sim/FITNESS_ROWS.csv`, `FITNESS_SUMMARY.json` | punto elegido por cada fitness, puntuado una vez en decisión |",
        "| corridas | `evidence/strategy_sim/runs/RUN_*.json.gz` | por corrida: genes elegidos, frente, métricas de selección y decisión, chequeos de corrección, trimestres, sensibilidad, por fitness |",
        "| archivos del optimizador | `runs/ledger/ARCHIVE_*.json.gz` (manifiesto; copiados los que caben) | cada punto evaluado por NSGA-II: genes, fitness, validez |",
        "| libros por trade | `runs/ledger/*.csv.gz` (manifiesto) | una fila por trade: entrada, salida, celda, volumen, neto |",
        "| intervenciones | `evidence/strategy_sim/interventions/INT_*.json.gz` | por intervención × mecanismo: DD con / sin, razón, IC, veredicto |",
        "| jerarquía | `evidence/strategy_sim/hier/CATALOG.json.gz`, `STAGE_A_<familia>.json.gz`, `*_leaves.json.gz`, `*_archive.json.gz` | hojas, testigos, frente de Pareto, región robusta, archivo completo de la Etapa A |",
        "| figuras | `evidence/strategy_sim/*.png` | curvas frente a la calidad por grupo (calidad / ejecución) y frente a MAE/naive |",
        "| fragmento generado | `evidence/strategy_sim/SWEEP_FRAGMENT.md` | el §7 del documento, tal como lo escribió el reporte |", ""]
r = req.read_text()
r = r[:r.index("<!-- ARTIFACTS -->")] + "\n".join(rows)
req.write_text(r)
print("spliced")
EOF

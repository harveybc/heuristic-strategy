#!/bin/bash
# Collect the behaviour-matrix report (JSON + Markdown rows) into the repo evidence and splice the Markdown into the
# document's §6. Usage: collect_behaviour_matrix.sh <mirror of the matrix root (report/ inside)> <checkout>
set -eu
SRC=$1; CODE=$2
EV=$CODE/docs/short_horizon/clean_round/evidence/behaviour_matrix
DOC=$CODE/docs/short_horizon/clean_round/BEHAVIOUR_MATRIX_2026_10_27.md
mkdir -p $EV
cp $SRC/report/BEHAVIOUR_MATRIX.json $SRC/report/BEHAVIOUR_MATRIX.md $EV/
[ -f $SRC/SPEC_SHA.txt ] && cp $SRC/SPEC_SHA.txt $EV/
[ -f $SRC/QUEUE.json ] && gzip -9 -c $SRC/QUEUE.json > $EV/CELLS_sealed.json.gz
python3 - "$DOC" "$EV" <<'PY'
import sys
from pathlib import Path
doc, ev = Path(sys.argv[1]), Path(sys.argv[2])
frag = (ev / "BEHAVIOUR_MATRIX.md").read_text().replace("\n## ", "\n### ")
if frag.startswith("# "):
    frag = frag.split("\n", 1)[1]
t = doc.read_text()
head = t[:t.index("<!-- BEHAVIOUR_MATRIX_FRAGMENT -->")]
doc.write_text(head + "<!-- BEHAVIOUR_MATRIX_FRAGMENT -->\n\n" + frag.strip() + "\n")
print("spliced")
PY

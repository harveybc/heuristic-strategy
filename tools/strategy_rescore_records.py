"""Offline re-score of stored behaviour-matrix records under VALIDITY_SPEC v2 (GM01) against the retired v1 rule: the
record carries the run's witnesses (results), the hold oracle's (oracle) and O*'s (ostar), so both rules apply to the
same measurements without a rerun. The expected witness set comes from the cell spec plus the world-generated witnesses
the record shows (W9 gap, W17 burst: the record does not carry the world meta). Writes RESCORE_RECORDS.json.

  python tools/strategy_rescore_records.py --cells <QUEUE.json> --results <dir> [--results <dir> ...] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import strategy_behaviour_matrix as BM  # noqa: E402


def rescore(rec: dict, cell: dict) -> dict:
    wit, hold, ostar = rec["results"], rec.get("oracle") or {}, rec.get("ostar")
    expected = BM.expected_witness_ids(cell) + [k for k in ("W9_gap_loss_bounded", "W17_no_open_in_burst") if k in wit]
    v2 = BM.validity(wit, hold, ostar, sorted(set(expected)))
    v1 = BM.validity_v1_retired(wit, hold, ostar)
    return {"id": rec["id"], "seed": rec.get("seed"), "v1_failing": sorted(v1["failing"]), "v2_failing": sorted(v2["failing"]), "v2_not_evaluable": v2["not_evaluable"],
            "v2_unvalidated": v2["unvalidated"], "v1_unvalidated": v1["unvalidated"], "v2_defects": v2["defects"], "v1_defects": v1["defects"],
            "v1_PASS": bool(v1["PASS"]), "v2_PASS": bool(v2["PASS"]), "recorded_PASS_effective": rec.get("PASS_effective"),
            "exit_failing": list((rec.get("exit") or {}).get("failing_effective") or [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--results", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = {c["id"]: c for c in json.loads(Path(a.cells).read_text())}
    rows = []
    for d in a.results:
        for p in sorted(Path(d).glob("CELL_*.json")):
            r = json.loads(p.read_text())
            if r.get("error") or r["id"] not in cells or not r.get("results"):
                continue
            x = rescore(r, cells[r["id"]])
            x["source"] = str(Path(d).name)
            rows.append(x)
    n = len(rows)
    summ = {"records": n, "v1_PASS": sum(r["v1_PASS"] for r in rows), "v2_PASS": sum(r["v2_PASS"] for r in rows),
            "v1_failing_witness_total": sum(len(r["v1_failing"]) for r in rows), "v2_failing_witness_total": sum(len(r["v2_failing"]) for r in rows),
            "records_changed": sum(r["v1_failing"] != r["v2_failing"] for r in rows),
            "newly_failing_by_witness": dict(Counter(k for r in rows for k in set(r["v2_failing"]) - set(r["v1_failing"]))),
            "no_longer_failing_by_witness": dict(Counter(k for r in rows for k in set(r["v1_failing"]) - set(r["v2_failing"]))),
            "not_evaluable_by_witness": dict(Counter(k for r in rows for k in r["v2_not_evaluable"])),
            "unvalidated_now_strict_by_witness": dict(Counter(k for r in rows for k in r["v2_unvalidated"]))}
    Path(a.out).write_text(json.dumps({"summary": summ, "spec_v3_sha256": BM.SPEC_V3_SHA, "rows": rows}, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()

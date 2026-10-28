"""Re-score parameter points under the FIXED verifier (VALIDITY_SPEC v2, W4 v3' economic, GM01 / GM02) on given cells, and
report the delta against the retired v1 rule: per cell and point, the failing witnesses under v2 and under v1 (same run,
same witnesses, two rules) plus the exit layer. Worlds are built once per cell (DEAP CachedWorld, perfect heads) and every
point is evaluated on it; CPU, under a crispdm cap.

  python tools/strategy_rescore_validity.py --cells <QUEUE.json> --ids a,b --points name=path.json,... --admitted <d> --csv-dir <d> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402
from tools import strategy_behaviour_matrix as BM  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402
from tools import strategy_param_deap as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--points", required=True, help="name=STRATEGY_PARAMS json, comma separated")
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    AC = S.contract_for_sim(str(S.CONTRACT))
    cx, SCFG = S.load_fold("wf2022", a.admitted, a.csv_dir)
    SCFG = S.with_variant(SCFG, C3.FULL_F3)
    points = {}
    for spec in a.points.split(","):
        name, path = spec.split("=", 1)
        points[name] = SP.validate({**SP.DEFAULTS, **{k: v for k, v in json.loads(Path(path).read_text()).items() if k in SP.BOUNDS or k in SP.LIST_KEYS or k in SP.META_KEYS}})
    cells = {c["id"]: c for c in json.loads(Path(a.cells).read_text())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for cid in a.ids.split(","):
        t0 = time.time()
        c = dict(cells[cid])
        c["quality"], c["subset_key"] = 1.0, cid + "@q1.00"
        w = D.CachedWorld(cx, SCFG, AC, c, 0)
        rec = {"cell": cid, "seed": 0, "spec_v3_sha256": BM.SPEC_V3_SHA, "expected_witnesses": BM.expected_witness_ids(c, w.w), "points": {}}
        for name, doc in points.items():
            r = w.evaluate(doc)
            rec["points"][name] = {"sharpe": r["sharpe"], "net_usd": r["net_usd"], "fails_v2": sorted(r["fails"]), "fails_v1_retired": sorted(r["fails_v1_retired"]),
                                   "validity_v2": r.get("validity_v2"), "params_sha256": SP.sha256_of(next(p for s_ in a.points.split(",") for n, p in [s_.split("=", 1)] if n == name))}
        rec["sec"] = round(time.time() - t0, 1)
        (out / f"RESCORE_{cid}.json").write_text(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"cell": cid, "sec": rec["sec"], **{n: {"v2": len(v["fails_v2"]), "v1": len(v["fails_v1_retired"])} for n, v in rec["points"].items()}}), flush=True)
        del w


if __name__ == "__main__":
    main()

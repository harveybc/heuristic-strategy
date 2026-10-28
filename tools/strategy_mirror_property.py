"""Mirror property test (coordinator order 2026-10-28): for random strategy parameter vectors, every zero-cost twin
(S_) and its exact mirror (M_) world must produce mirrored ledgers -- same entry / exit bars, sides flipped, per-unit
net equal (W14 mirror_check) -- and no per-run W14 verdict. Evaluated through the DEAP's own world evaluation
(CachedWorld), so the test covers exactly what the optimiser scores. Two checks per (pair, genome): W14 as sealed
(mirror world, SAME vector: a behaviour requirement; side-asymmetric legacy close genes fail it legitimately) and the
diagnostic W14_swap (mirror world, SIDE-SWAPPED vector: code symmetry; coordinator ruling 2026-10-28, never a DEAP
constraint). Exit 1 only on a W14_swap failure (a code asymmetry) or a per-run W14 verdict.

  python tools/strategy_mirror_property.py --subset <SUBSET.json> --pairs LHS_46,... --n 50 [--point <json>] --admitted <d> --csv-dir <d> --out <jsonl>
"""
from __future__ import annotations

import argparse
import json
import random
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


def genomes(n: int, seed: int, point: dict | None, points: list | None = None) -> list:
    base = SP.default_doc()
    out = []
    for name, pt in ([("point", point)] if point is not None else []) + [(f"point_{i}", q) for i, q in enumerate(points or [])]:
        p = {**base, **pt}
        for k, src in D.TIED.items():
            p[k] = p[src]
        out.append((name, SP.validate(p)))
    rng = random.Random(seed)
    for i in range(n):
        out.append((f"random_{i}", D.decode([rng.random() for _ in D.GENOME], base)))
    return out


def swap_sides(doc: dict) -> dict:
    """The mirror of a parameter vector: every _l / _s pair exchanged."""
    d = dict(doc)
    for k in doc:
        if k.endswith("_l") and (k[:-2] + "_s") in doc:
            d[k], d[k[:-2] + "_s"] = doc[k[:-2] + "_s"], doc[k]
    return SP.validate(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--point", default=None)
    ap.add_argument("--points-jsonl", default=None, help="parameter points, one {'params': {...}} or {...} per line (e.g. the DEAP warm start)")
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sub = {c["subset_key"]: c for c in json.loads(Path(a.subset).read_text())}
    AC = S.contract_for_sim(str(S.CONTRACT))
    cx, SCFG = S.load_fold("wf2022", a.admitted, a.csv_dir)
    SCFG = S.with_variant(SCFG, C3.FULL_F3)
    pts = [(lambda d: d.get("params", d))(json.loads(x)) for x in Path(a.points_jsonl).read_text().splitlines() if x.strip()] if a.points_jsonl else None
    gs = genomes(a.n, a.seed, json.loads(Path(a.point).read_text()) if a.point else None, pts)
    bad = 0
    with open(a.out, "a") as fh:
        for pid in a.pairs.split(","):
            t0 = time.time()
            ws = D.CachedWorld(cx, SCFG, AC, sub[f"S_{pid}@q1.00"], 0)
            wm = D.CachedWorld(cx, SCFG, AC, sub[f"M_{pid}@q1.00"], 0)
            for name, doc in gs:
                rs, rm = ws.evaluate(doc), wm.evaluate(doc)
                w = BM.mirror_check(rs["_trades"], rm["_trades"])
                per_run_w14 = [f for f in rs["fails"] + rm["fails"] if f.startswith("W14")]
                sw = w if w["pass"] else BM.mirror_check(rs["_trades"], wm.evaluate(swap_sides(doc))["_trades"])   # W14_swap: needed only where W14 fails
                code_ok = bool(sw["pass"] and not per_run_w14)
                bad += int(not code_ok)
                fh.write(json.dumps({"pair": pid, "genome": name, "pass": bool(w["pass"] and not per_run_w14), "W14_swap_pass": bool(sw["pass"]), "code_symmetric": code_ok,
                                     "n": [w["n_a"], w["n_b"]], "bars_match": w["bars_match"], "sides_flipped": w["sides_flipped"],
                                     "max_net_dev_sigma": w["max_net_discrepancy_ex_financing_sigma"], "W14_swap_dev_sigma": sw["max_net_discrepancy_ex_financing_sigma"],
                                     "per_run_w14": per_run_w14}, default=float) + "\n")
                fh.flush()
            print(json.dumps({"pair": pid, "genomes": len(gs), "sec": round(time.time() - t0)}), flush=True)
            del ws, wm
    print(json.dumps({"code_symmetry_failures": bad}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

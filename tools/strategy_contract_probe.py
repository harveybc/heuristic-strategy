"""Business-contract probe (coordinator order 2026-10-29): parameter points re-scored on the DEAP subset worlds under
engine settings passed as OVERRIDES of the sealed config (nothing sealed is changed): (A) the current contract,
(B) leverage 50, (C) leverage 50 and no daily entry cap (the env compares the count with an int, so 'no cap' is a cap
of 10**6; a None would need env.py L167 / L188 changed). Per world, setting and point: Sharpe, net, trades, early
closes, failing witnesses, levels_clipped. Worlds are rebuilt per setting (the controls and the truth run on the engine).

  python tools/strategy_contract_probe.py --subset <json> --points-jsonl <jsonl> --settings A,B,C --admitted <d> --csv-dir <d> --out <dir> [--keys k1,k2]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import execution as X  # noqa: E402
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402
from tools import strategy_param_deap as D  # noqa: E402

NO_CAP = 10 ** 6
SETTINGS = {"A": {"leverage": 100.0, "cap": None}, "B": {"leverage": 50.0, "cap": None}, "C": {"leverage": 50.0, "cap": NO_CAP}}


def with_setting(SCFG, cell: dict, st: dict):
    sc = dataclasses.replace(SCFG, sizing=X.Sizing(rel_volume=SCFG.sizing.rel_volume, leverage=float(st["leverage"])))
    c = dict(cell)
    if st["cap"] is not None:
        sc = dataclasses.replace(sc, max_entries_per_day=int(st["cap"]))
        ex = dict(c.get("exec") or {})
        ex.pop("cap", None)                                           # the cell's own daily cap is lifted too
        c["exec"] = ex
    return sc, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--points-jsonl", required=True)
    ap.add_argument("--settings", default="A,B,C")
    ap.add_argument("--keys", default=None)
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    AC = S.contract_for_sim(str(S.CONTRACT))
    cx, SCFG = S.load_fold("wf2022", a.admitted, a.csv_dir)
    SCFG = S.with_variant(SCFG, C3.FULL_F3)
    base = SP.default_doc()
    pts = []
    for x in Path(a.points_jsonl).read_text().splitlines():
        if not x.strip():
            continue
        d = json.loads(x)
        p = {**base, **d["params"]}
        for k, src in D.TIED.items():
            p[k] = p[src]
        pts.append((d["name"], SP.validate(p)))
    sub = {c["subset_key"]: c for c in json.loads(Path(a.subset).read_text())}
    keys = a.keys.split(",") if a.keys else list(sub)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        t0 = time.time()
        rec = {"world": key, "quality": sub[key]["quality"], "seed": 0, "settings": {}}
        for sname in a.settings.split(","):
            sc, cell = with_setting(SCFG, sub[key], SETTINGS[sname])
            w = D.CachedWorld(cx, sc, AC, cell, 0)
            res = {}
            for name, doc in pts:
                r = w.evaluate(doc)
                res[name] = {k: r.get(k) for k in ("sharpe", "net_usd", "trades", "early_closes", "levels_clipped", "fails", "trade_stats")}
            rec["settings"][sname] = {"leverage": SETTINGS[sname]["leverage"], "max_entries_per_day": SETTINGS[sname]["cap"] or sc.max_entries_per_day, "points": res}
            del w
        rec["sec"] = round(time.time() - t0)
        (out / f"PROBE_{key.replace('@', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"world": key, "sec": rec["sec"], **{s_: {"canonical_sharpe": rec["settings"][s_]["points"]["canonical"]["sharpe"], "canonical_fails": len(rec["settings"][s_]["points"]["canonical"]["fails"])} for s_ in rec["settings"]}}, default=float), flush=True)


if __name__ == "__main__":
    main()

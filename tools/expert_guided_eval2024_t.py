#!/usr/bin/env python3
"""Single 2024 read of a FROZEN ladder trial (no training): E1-T round-3 policies and every E2-T actor of <dir>,
under the lane contract and under Musashi's contract (<= 1 entry/day, 1:50). Writes <dir>/EVAL_2024.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from tools.expert_guided_e0 import jsonable  # noqa: E402
from tools.expert_guided_e1t import MUSASHI_CFG, STUDENT_CFG, Ctx, summary  # noqa: E402
from tools.expert_guided_e2t import actor_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--v4-dir", default="")
    ap.add_argument("--dir", required=True, help="trial dir with e1t/ and e2t/")
    a = ap.parse_args()
    import torch
    from app.short_horizon.expert_guided import temporal as T
    d = Path(a.dir)
    e1 = json.loads((d / "e1t" / "E1T.json").read_text())
    e2 = json.loads((d / "e2t" / "E2T.json").read_text())
    feats = e2["design"]["network"]["features"]
    ds = DG.load(a.admitted, a.csv_dir) if feats == "core15" else DG.load_v4(a.admitted, a.v4_dir)
    cx = Ctx(ds)
    nf, NS = ds.windows.shape[2], len(E.STATE_NAMES)
    arch1 = e1["model"]["arch"]
    ext = e2["design"]["network"]["extractor_arch"]
    head = e2["design"]["network"]["net_arch"]
    out = {"note": "single 2024 read of frozen policies (ladder finalist)", "e1t": {}, "e2t": {}}
    for s in e1["seeds"]:
        net = T.TemporalPolicyNet(nf, DG.WIN, 4, n_reg=2, n_state=NS, **arch1)
        net.load_state_dict(torch.load(d / "e1t" / f"policy_{s}.pt"))
        pol = cx.student(T.TorchPolicy(net))
        out["e1t"][s] = {"lane": summary(cx.rollout("OBS_2024", pol, STUDENT_CFG)[0]),
                         "musashi_contract": summary(cx.rollout("OBS_2024", pol, MUSASHI_CFG)[0])}
    for k in e2["runs"]:
        sd = torch.load(d / "e2t" / f"actor_{k}.pt")
        actor = T.TorchActor(sd, nf, DG.WIN, NS, ext["units"], head, branch=ext["branch"], core=ext["core"], depth=ext["depth"])
        pol = actor_policy(cx, actor)
        out["e2t"][k] = {"lane": summary(cx.rollout("OBS_2024", pol, STUDENT_CFG)[0]),
                         "musashi_contract": summary(cx.rollout("OBS_2024", pol, MUSASHI_CFG)[0])}
    (d / "EVAL_2024.json").write_text(json.dumps(jsonable(out), indent=1))
    print(json.dumps({k: {r: v["lane"]["sharpe_daily_sqrt365"] for r, v in out[k].items()} for k in ("e1t", "e2t")}))


if __name__ == "__main__":
    main()

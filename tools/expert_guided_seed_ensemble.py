#!/usr/bin/env python3
"""Seed ensemble of a finished E3-M arm (owner 2026-10-25): the N seed checkpoints of every module are averaged in
probability space into ONE policy (softmax probabilities averaged; regression heads averaged) and replayed with the
same ``Composed`` routing, env and replay contract as the run. No training. Per period: every seed's own frozen policy
replayed again (a verification: it must reproduce the E3M.json figure), the ensemble, and the uncertainty of each Sharpe
(``stats``: Lo SE, stationary-bootstrap interval).

Arbiter of the ensemble (declared before any result was read): the most frequent frozen arbiter among the seeds (ties
-> "prob"); for "prob" the rate is the median of the seeds' frozen rates and tau is re-derived on the selection block
(INNER_A) from the ensemble's own open scores, exactly as the run derives it per seed. Nothing is chosen on the
validation window (INNER_B of a walk-forward fold, or OBS_2024).

Outputs ENSEMBLE.json in the arm's out dir (or --out).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import stats as ST  # noqa: E402
from tools.expert_guided_e1t import Ctx, summary  # noqa: E402
from tools.expert_guided_e3m import Composed, effective_contracts  # noqa: E402

MODULES = ("open_l", "open_s", "close_l", "close_s")


class ProbEnsemble:
    """Callable with the TorchPolicy contract (x[None, :] -> (logits, reg)): log of the mean softmax across members (so
    the consumer's softmax returns the mean probability exactly) and the mean regression output. N = 1 reproduces the
    single member's probabilities and regressions."""

    def __init__(self, members):
        self.members = list(members)
        if not self.members:
            raise ValueError("ENSEMBLE_EMPTY")

    def __call__(self, x):
        P, R = [], []
        for m in self.members:
            lg, rg = m(x)
            lg = np.asarray(lg, float)
            z = lg - lg.max(axis=1, keepdims=True)
            p = np.exp(z)
            P.append(p / p.sum(axis=1, keepdims=True))
            R.append(None if rg is None else np.asarray(rg, float))
        p = np.mean(P, axis=0)
        lg = np.log(np.clip(p, 1e-300, None))
        rg = None if R[0] is None else np.mean(R, axis=0)
        return lg, rg


def load_module(out_dir: Path, e3m: dict, module: str, seed: str, n_feat: int, win: int):
    import torch
    from app.short_horizon.expert_guided import temporal as T
    arch = e3m["arch"]
    kw = dict(units=arch["units"], branch=arch.get("branch"), core=arch.get("core"), depth=arch["depth"], head=arch["head"])
    if arch.get("groups") is not None:
        kw.update(groups=arch["groups"], group_units=arch["group_units"], group_names=arch["group_names"], group_kinds=arch["group_kinds"])
    sd = torch.load(out_dir / f"FULL_{module}_{seed}.pt", map_location="cpu", weights_only=True)
    nc = int(sd["cls.weight"].shape[0])                # classes and regression heads come from the checkpoint itself
    nr = 0
    if "lo" in sd and "hi" in sd:                      # the regression bounds travel in the checkpoint (persistent buffers)
        kw["reg_bounds"] = tuple(zip(sd["lo"].tolist(), sd["hi"].tolist()))
        nr = len(sd["lo"])
    net = T.TemporalPolicyNet(n_feat, win, nc, n_reg=nr, n_state=len(E.STATE_NAMES), **kw)
    net.load_state_dict(sd, strict=True)
    return T.TorchPolicy(net)


def ensemble_arbiter(trials: dict) -> dict:
    arbs = [trials[s]["arbiter"]["arbiter"] for s in trials]
    cnt = Counter(arbs)
    top = max(cnt.values())
    cands = sorted(a for a, c in cnt.items() if c == top)
    arb = "prob" if "prob" in cands else cands[0]
    rate = None
    if arb == "prob":
        rates = [trials[s]["arbiter"].get("rate") for s in trials if trials[s]["arbiter"]["arbiter"] == "prob" and trials[s]["arbiter"].get("rate") is not None]
        rate = float(np.median(rates)) if rates else 0.06
    return {"arbiter": arb, "rate": rate, "members": arbs, "rule": "most frequent frozen arbiter (ties -> prob); prob rate = median of the seeds' rates; tau re-derived on INNER_A"}


def window_stats(m: dict, cfg) -> dict:
    ret = ST.daily_returns(m["daily_equity"], float(cfg.capital))
    d = ST.describe(ret, point=m["sharpe_daily_sqrt365"])
    s = summary(m)
    return {"metrics": s, "sharpe": d["sharpe"], "engine_sharpe": d["engine_sharpe"], "recomputed_equals_engine": d["recomputed_equals_engine"],
            "se_lo": d["se_lo"], "sharpe_minus_se": d["sharpe_minus_se"], "days": d["days"], "bootstrap": d["bootstrap"]}


def evaluate(out_dir: Path, ds, periods: list, seeds: list | None = None, cfg=None, selection: str = "INNER_A") -> dict:
    e3m = json.loads((out_dir / "E3M.json").read_text())
    trials = e3m["trials"]["FULL"]
    seeds = [str(s) for s in (seeds or list(trials))]
    missing = [s for s in seeds if s not in trials]
    if missing:
        raise SystemExit(f"ENSEMBLE_SEEDS_NOT_IN_RUN:{missing}")
    cx = Ctx(ds)
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    n_feat, win = ds.windows.shape[2], ds.windows.shape[1]
    mask = E.state_mask(e3m["agent_state"]["mode"])
    SCFG, _ = effective_contracts(e3m.get("exec_contract", "sac"), e3m.get("min_orders_per_week"))
    cfg = cfg or SCFG
    pols = {s: {m: load_module(out_dir, e3m, m, s, n_feat, win) for m in MODULES} for s in seeds}
    doc = {"arm_dir": str(out_dir), "recipe": e3m.get("recipe"), "split_contract": ds.meta["split_contract"], "seeds": seeds, "periods": periods,
           "support_hash": ds.meta["support_hash"], "per_seed": {}, "ensemble": {}, "arbiter": ensemble_arbiter({s: trials[s] for s in seeds}),
           "verification": {"per_seed_replay_equals_run": {}}}
    for s in seeds:
        a = trials[s]["arbiter"]
        c = Composed("FULL", pols[s], a["arbiter"], a.get("tau"), mask=mask)
        doc["per_seed"][s] = {}
        for p in periods:
            m = cx.rollout(p, c(cx), cfg)[0]
            doc["per_seed"][s][p] = window_stats(m, cfg)
            run_v = trials[s]["trading"].get(p, {}).get("sharpe_daily_sqrt365")
            if run_v is not None:
                doc["verification"]["per_seed_replay_equals_run"][f"{s}/{p}"] = bool(abs(run_v - m["sharpe_daily_sqrt365"]) < 1e-9)
    ens = {m: ProbEnsemble([pols[s][m] for s in seeds]) for m in MODULES}
    arb = doc["arbiter"]
    tau = None
    if arb["arbiter"] == "prob":
        pm = Composed("FULL", ens, "prob", mask=mask).open_scores(cx, cx.ds.rows(selection))
        tau = float(np.quantile(pm, 1 - arb["rate"]))
    arb["tau"] = tau
    c = Composed("FULL", ens, arb["arbiter"], tau, mask=mask)
    for p in periods:
        m = cx.rollout(p, c(cx), cfg)[0]
        doc["ensemble"][p] = window_stats(m, cfg)
    doc["verification"]["all_per_seed_replays_equal_run"] = all(doc["verification"]["per_seed_replay_equals_run"].values())
    doc["verification"]["recomputed_sharpe_equals_engine"] = all(v["recomputed_equals_engine"] for s in doc["per_seed"].values() for v in s.values()) and \
        all(v["recomputed_equals_engine"] for v in doc["ensemble"].values())
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm-dir", required=True, help="E3-M out dir of the arm (E3M.json + FULL_<module>_<seed>.pt)")
    ap.add_argument("--extra-arm-dir", action="append", default=[], help="seed-extension out dirs of the same arm (pooled members)")
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--recipe", required=True, help="the recipe the arm ran on (fold recipe for WF units)")
    ap.add_argument("--split-contract", required=True)
    ap.add_argument("--periods", default="TRAIN,INNER_A")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rec = json.loads(Path(a.recipe).read_text())
    feats = [c["store_column"] for c in rec["tensor"]["columns"]]
    ds = DG.load_store(a.admitted, a.store, feats, scaling="fixed_train", contract=a.split_contract, recipe=rec)
    out_dir = Path(a.arm_dir)
    if a.extra_arm_dir:
        # pooled seeds: merge the extension dirs' checkpoints and trials into a scratch view of the arm (hard links, read-only)
        import tempfile
        view = Path(tempfile.mkdtemp(prefix="ens_pool_"))
        e = json.loads((out_dir / "E3M.json").read_text())
        for f in out_dir.glob("FULL_*.pt"):
            (view / f.name).hardlink_to(f) if hasattr(Path, "hardlink_to") else (view / f.name).symlink_to(f)
        for d in a.extra_arm_dir:
            d = Path(d)
            ep = json.loads((d / "E3M.json").read_text())
            if ep.get("recipe") != e.get("recipe") or ep["meta"]["support_hash"] != e["meta"]["support_hash"]:
                raise SystemExit(f"POOL_MISMATCH:{d}")
            e["trials"]["FULL"].update(ep["trials"]["FULL"])
            for f in d.glob("FULL_*.pt"):
                (view / f.name).hardlink_to(f) if hasattr(Path, "hardlink_to") else (view / f.name).symlink_to(f)
        (view / "E3M.json").write_text(json.dumps(e))
        out_dir = view
    doc = evaluate(out_dir, ds, [p.strip() for p in a.periods.split(",") if p.strip()])
    doc["arm_dir"] = a.arm_dir
    doc["pooled_dirs"] = a.extra_arm_dir
    out = Path(a.out) if a.out else Path(a.arm_dir) / "ENSEMBLE.json"
    out.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({"out": str(out), "seeds": doc["seeds"], "arbiter": doc["arbiter"],
                      "ensemble": {p: {"sharpe": round(v["sharpe"], 3), "se": round(v["se_lo"], 3), "valid": v["metrics"]["valid"]} for p, v in doc["ensemble"].items()},
                      "per_seed_equal_run": doc["verification"]["all_per_seed_replays_equal_run"]}))


if __name__ == "__main__":
    main()

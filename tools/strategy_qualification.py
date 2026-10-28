"""Strategy qualification panels D3-D6 (coordinator order 2026-10-28, after the architecture contract in the
heuristic-strategy README): every expectation lives in docs/short_horizon/clean_round/contracts/QUALIFICATION_SPEC.json,
committed BEFORE any of these panels ran; this tool only measures and compares against it.

  controls  -- D4 negative controls on matrix cells: permuted head outputs, swapped direction channels, outputs
               delayed by 1 / 3 / 6 bars (the closer's market-only head included); verdict against the sealed spec
  seeded    -- D4 seeded positions (entry ages 0 / 6 / 24 / 72 bars, P&L state loss / break-even / profit, both sides)
               and the reversal-then-recovery episodes; D5 normalized regret (regret / initial cash risk at SL)
  confirm   -- D3 unseen confirmation: paired per-base-cell comparison of two matrix result folders (seeds 3-5),
               bootstrap CI over base cells

CPU only, under crispdm-run; synthetic worlds only; TEST rows are never read.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402
from app.short_horizon.expert_guided import strategy_zz as ZZ  # noqa: E402
from tools import strategy_behaviour_matrix as BM  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402

SPEC_PATH = ROOT / "docs/short_horizon/clean_round/contracts/QUALIFICATION_SPEC.json"
SEED = 20261028
ROW_KEYS_SIDED = ("net", "p3", "mae", "mfe", "zz_room", "zz_counter", "zz_fresh", "zz_fstar")
ROW_KEYS_FLAT = ("pn", "zz_dir", "zz_tau_fine", "zz_nop_hard", "zz_flip_age")


def spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


def spec_sha() -> str:
    import hashlib
    return hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()


def load_base(admitted, csv_dir):
    AC = S.contract_for_sim(str(S.CONTRACT))
    cx, SCFG = S.load_fold("wf2022", admitted, csv_dir)
    return AC, cx, S.with_variant(SCFG, C3.FULL_F3)


def prepare(cx, SCFG, AC, cell: dict, seed: int, params: dict):
    ZZ.configure(params)
    genes = SP.genes_c3(params)
    w = BM.cell_world(cx, SCFG, AC, cell, seed)
    strat, view, cells = BM.strategy_for(w, genes, AC)
    return w, strat, view, cells


# ----------------------------------------------------------------------------- D4 negative controls
class MappedHead:
    """The closer's market-only head read through a mapping (permuted bars, delayed bars, swapped sides)."""

    def __init__(self, bc, bar_map=None, swap=False):
        self.bc, self.bar_map, self.swap = bc, bar_map, swap
        self.S, self.T, self.rows = bc.S, bc.T, {}

    def row(self, b, si):
        bb = int(self.bar_map(b)) if self.bar_map is not None else int(b)
        bb = min(max(bb, 0), self.bc.n - 2)
        return self.bc.row(bb, 1 - si if self.swap else si)


def control_view(view: dict, kind: str, k: int = 0, rng=None) -> dict:
    """A copy of the strategy's head table with the outputs permuted across rows, side channels swapped, or delayed by
    k rows; the closer reads its market-only head through the same transformation."""
    v = dict(view)
    n = len(view["pn"])
    valid = np.flatnonzero(np.isfinite(np.asarray(view["pn"], float)))   # the table's own rows (CAL + BLOCK, observable)
    if kind == "permuted":
        perm = np.arange(n)
        perm[valid] = valid[rng.permutation(len(valid))]
        rmap = lambda a: a[perm]  # noqa: E731
        bar_perm = rng.permutation(view["closer"].bc.n)
        bar_map = lambda b: bar_perm[b]  # noqa: E731
    elif kind == "delayed":
        idx = np.arange(n)
        idx[valid] = valid[np.clip(np.arange(len(valid)) - k, 0, len(valid) - 1)]   # the value of k decision rows earlier
        rmap = lambda a: a[idx]  # noqa: E731
        bar_map = lambda b: b - k  # noqa: E731
    elif kind == "swapped":
        rmap = None
        bar_map = None
    else:
        raise ValueError(kind)
    for key in ROW_KEYS_SIDED + ROW_KEYS_FLAT:
        if key not in view or view[key] is None or np.ndim(view[key]) == 0:
            continue
        a = np.asarray(view[key])
        if kind == "swapped":
            if key in ROW_KEYS_SIDED and a.ndim >= 2 and a.shape[1] == 2:
                v[key] = a[:, ::-1].copy()
            elif key == "zz_dir":
                v[key] = -a
        else:
            v[key] = rmap(a)
    cl = copy.copy(view["closer"])
    cl.__dict__ = dict(view["closer"].__dict__)
    cl.bc = MappedHead(view["closer"].bc, bar_map, swap=(kind == "swapped"))
    v["closer"] = cl
    return v


def run_policy(w, strat, view, cells):
    roll = BM.rollout_cell(w, strat, view, cells)
    m = roll["metrics"]
    costs = sum(float(t.get("spread_usd", 0)) + float(t.get("commission_usd", 0)) + float(t.get("slippage_usd", 0)) - float(t.get("financing_usd", 0)) for t in roll["trades"])
    return {"sharpe": float(m.get("sharpe_daily_sqrt365", float("nan"))), "net_usd": float(m.get("net_profit_usd", float("nan"))), "trades": len(roll["trades"]), "costs_usd": costs}


def cmd_controls(a):
    sp = spec()["D4_negative_controls"]
    AC, cx, SCFG = load_base(a.admitted, a.csv_dir)
    params = SP.load(a.params)
    cells = {c["id"]: c for c in json.loads(Path(a.cells).read_text())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for cid in a.ids.split(","):
        t0 = time.time()
        w, strat, view, cl = prepare(cx, SCFG, AC, cells[cid], 0, params)
        rng = np.random.default_rng(SEED)
        res = {"base": run_policy(w, strat, view, cl)}
        res["permuted"] = run_policy(w, strat, control_view(view, "permuted", rng=rng), cl)
        res["swapped"] = run_policy(w, strat, control_view(view, "swapped"), cl)
        for k in (1, 3, 6):
            res[f"delayed_{k}"] = run_policy(w, strat, control_view(view, "delayed", k=k), cl)
        trend = cells[cid]["regime"] in ("trend_up", "trend_down")
        p = res["permuted"]
        verdict = {
            "permuted": bool(p["costs_usd"] > 0 and -1.5 * p["costs_usd"] <= p["net_usd"] <= -0.5 * p["costs_usd"] and p["sharpe"] <= 0),
            "swapped": (bool(res["swapped"]["net_usd"] < 0 and res["swapped"]["net_usd"] <= -0.5 * res["base"]["net_usd"]) if trend else None),
            "delayed": bool(res["base"]["sharpe"] + 0.05 >= res["delayed_1"]["sharpe"] and res["delayed_1"]["sharpe"] + 0.05 >= res["delayed_3"]["sharpe"]
                            and res["delayed_3"]["sharpe"] + 0.05 >= res["delayed_6"]["sharpe"] and res["delayed_6"]["sharpe"] < res["base"]["sharpe"]),
        }
        rec = {"cell": cid, "seed": 0, "results": res, "verdict": verdict, "expected": sp, "spec_sha256": spec_sha(), "sec": round(time.time() - t0, 1)}
        (out / f"CONTROLS_{cid}.json").write_text(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"cell": cid, "verdict": verdict, "sharpe": {k: round(v["sharpe"], 3) for k, v in res.items()}}), flush=True)


# ----------------------------------------------------------------------------- D4 seeded positions, D5 normalized regret
def seeded_episodes(w, strat, view, cells, ages, every: int = 12) -> list:
    """Orders opened ``age`` bars before a seeding tick with the strategy's envelope on a clean engine (independent
    episodes); the episode starts at the seeding tick, the closer alone decides."""
    cx, SCFG = w["cx"], w["scfg"]
    env = S.OC.make_env(cx, SCFG)
    a, b = BM.H.period_bars(cx.ds, BM.BLOCK)
    eps = []
    from app.short_horizon.expert_guided import env as E
    for age in ages:
        for t in range(a + age, b - 2, every):
            e = t - age
            r = int(cx.rob[e])
            if r < 0 or not cx.obs_ok[r]:
                continue
            for si in (0, 1):
                tp, sl = BM._envelope(ZZ.PolicyC3Z(view, strat, cells, w["truth"].e_time, w["truth"].cost), r, si)
                S.OC._clean(env)
                info = env.step(e, E.LONG if si == 0 else E.SHORT, tp, sl)
                if not info["entry"] or env.pos is None:
                    continue
                eps.append({"pos": copy.copy(env.pos), "side": si, "age": age, "seed_bar": t})
                S.OC._clean(env)
    return eps


def run_seeded(w, eps, decide, closer) -> list:
    cx = w["cx"]
    out = []
    for e in eps:
        pos, si, t0 = e["pos"], e["side"], e["seed_bar"]
        path = closer.path(pos)
        bars, P = path["bars"], path["P"]
        keep = bars >= t0
        if not keep.any():
            continue
        bars, P = bars[keep], P[keep]
        u, sg = path["units"], path["sigma"]
        risk = abs(float(pos.entry_price) - float(pos.sl)) * u
        if not (risk > 0):
            raise ValueError("CONTRACT_ERROR_ZERO_INITIAL_RISK")
        v_star = float(max(np.max(P), path["hold_net"]))                 # V*(t): discretionary closes from t, or N_auto(tau)
        p_t = float(P[0])
        state = "loss" if p_t / u / sg <= -0.25 else ("break_even" if abs(p_t / u / sg) <= 0.10 else ("profit" if p_t / u / sg >= 0.25 else "other"))
        exit_v, early = path["hold_net"], False
        for i, t in enumerate(bars):
            r = int(cx.rob[int(t)])
            if r < 0 or not cx.obs_ok[r]:
                continue
            if decide(r, si, int(t), pos):
                exit_v, early = float(P[i]), True
                break
        dip = float(np.min(P)) / u / sg
        recovered = (path["hold_reason"] == "TP" or float(np.max(P)) / u / sg >= 0.5) and dip <= -0.5 and path["hold_reason"] != "SL"
        out.append({"side": si, "age": e["age"], "state": state, "sl_destined": path["hold_reason"] == "SL", "early": early,
                    "regret_norm": (v_star - exit_v) / risk, "regret_sigma": (v_star - exit_v) / u / sg, "recovery": bool(recovered)})
    return out


def summarise_seeded(rows: list) -> dict:
    res = {}
    for side in (0, 1):
        for age in sorted({r["age"] for r in rows}):
            for state in ("loss", "break_even", "profit"):
                g = [r for r in rows if r["side"] == side and r["age"] == age and r["state"] == state]
                if not g:
                    continue
                sld = [r for r in g if r["sl_destined"]]
                rn = np.asarray([r["regret_norm"] for r in g], float)
                res[f"{'BUY' if side == 0 else 'SELL'}|age{age}|{state}"] = {
                    "n": len(g), "sl_destined": len(sld), "sl_destined_closed_before_sl": float(np.mean([r["early"] for r in sld])) if sld else float("nan"),
                    "normalized_regret_mean": float(rn.mean()), "normalized_regret_p90": float(np.quantile(rn, 0.9))}
    return res


def cmd_seeded(a):
    sp = spec()
    ages = sp["D4_seeded_positions"]["entry_ages_bars"]
    AC, cx, SCFG = load_base(a.admitted, a.csv_dir)
    params = SP.load(a.params)
    cells = {c["id"]: c for c in json.loads(Path(a.cells).read_text())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for cid in a.ids.split(","):
        t0 = time.time()
        w, strat, view, cl = prepare(cx, SCFG, AC, cells[cid], 0, params)
        closer = view["closer"].with_quality(1.0)
        eps = seeded_episodes(w, strat, view, cl, ages)
        pol = ZZ.PolicyC3Z(view, strat, cl, w["truth"].e_time, w["truth"].cost)
        rows = run_seeded(w, eps, pol.close_now, closer)
        star = run_seeded(w, eps, lambda r, si, bar, pos: bool(S.path_advantage(closer.path(pos), bar) >= -BM.EPS_CLOSE), closer)
        summ = summarise_seeded(rows)
        rec_s = [r for r in rows if r["recovery"]]
        rec_o = [r for r in star if r["recovery"]]
        recovery = {"n": len(rec_s), "premature_regret_norm_mean": float(np.mean([r["regret_norm"] for r in rec_s])) if rec_s else float("nan"),
                    "early_share": float(np.mean([r["early"] for r in rec_s])) if rec_s else float("nan"),
                    "Ostar_early_share": float(np.mean([r["early"] for r in rec_o])) if rec_o else float("nan")}
        exp = sp["D4_seeded_positions"]["expected"]
        verdict = {k: bool((v["sl_destined"] < 30 or v["sl_destined_closed_before_sl"] >= 0.95) and v["normalized_regret_mean"] <= 0.10 and v["normalized_regret_p90"] <= 0.30) for k, v in summ.items()}
        verdict["reversal_then_recovery"] = bool(recovery["n"] == 0 or (recovery["premature_regret_norm_mean"] <= 0.10 and recovery["early_share"] <= recovery["Ostar_early_share"] + 0.05))
        rec = {"cell": cid, "panels": summ, "reversal_then_recovery": recovery, "verdict": verdict, "expected": exp, "spec_sha256": spec_sha(), "sec": round(time.time() - t0, 1)}
        (out / f"SEEDED_{cid}.json").write_text(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"cell": cid, "episodes": len(rows), "failing_panels": [k for k, v in verdict.items() if not v], "recovery": recovery}), flush=True)


# ----------------------------------------------------------------------------- D3 unseen confirmation
def cell_means(d: str) -> dict:
    agg = {}
    for p in Path(d).glob("CELL_*.json"):
        r = json.loads(p.read_text())
        if r.get("error") or r["id"].startswith(("S_", "M_")):
            continue
        sh = next((x["sharpe"] for x in (r.get("exit") or {}).get("closer_sweep") or [] if x.get("closer_quality") == 1.0), float("nan"))
        agg.setdefault(r["id"], []).append((sh, float(r["results"]["summary"]["net_usd"])))
    return {k: (float(np.nanmean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) for k, v in agg.items()}


def cmd_confirm(a):
    A, B = cell_means(a.a), cell_means(a.b)
    keys = sorted(set(A) & set(B))
    rng = np.random.default_rng(SEED)
    out = {"cells": len(keys), "spec_sha256": spec_sha()}
    for j, name in ((0, "sharpe"), (1, "net_usd")):
        d = np.asarray([B[k][j] - A[k][j] for k in keys], float)
        d = d[np.isfinite(d)]
        boots = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(10000)] if len(d) else [float("nan")]
        lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        out[name] = {"mean_diff_b_minus_a": float(np.mean(d)) if len(d) else float("nan"), "ci95": [lo, hi], "claim": bool(lo > 0)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("controls", "seeded"):
        r = sub.add_parser(name)
        r.add_argument("--cells", required=True)
        r.add_argument("--ids", required=True)
        r.add_argument("--params", required=True)
        r.add_argument("--admitted", required=True)
        r.add_argument("--csv-dir", required=True)
        r.add_argument("--out", required=True)
    c = sub.add_parser("confirm")
    c.add_argument("--a", required=True, help="the untuned canonical point's results (seeds 3-5)")
    c.add_argument("--b", required=True, help="the DEAP point's results (seeds 3-5)")
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"controls": cmd_controls, "seeded": cmd_seeded, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()

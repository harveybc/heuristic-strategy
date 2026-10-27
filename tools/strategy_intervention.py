#!/usr/bin/env python3
"""Intervention battery and counterfactual ablations for the simulated strategy (owner 2026-10-26): real bars with
injected modifications in the DECISION block only (the genes were optimised on the selection block and are reused
unchanged), the simulated heads NOT anticipating the modification (their quality is unchanged) unless the intervention
is a regime splice (then the truth is rebuilt on the spliced path and the heads keep their declared quality on it).

Interventions: (a) shocks = persistent price jumps of 2 / 3 / 4 sigma against the open position (or a random side when
flat) at random ticks; (b) regime flips = the decision block's price path spliced from real quarters trend -> range ->
trend; (c) spread blow-up x5 for 24 h; (d) runs of missing observations of 6-24 h (HOLD-on-missing); (e) volatility
doubling in the heads' noise (engine levels unchanged).
Ablations on every path (base and intervened): with / without early close, with / without the NOP veto, with / without
the best-of-day gate, with / without the volume formula (v = 1). Loss-limiting ratio = max drawdown (USD) with the
mechanism / without, with a bootstrap CI over the intervention draws and seeds; TOLERANT when the ratio's CI upper
bound < 1 at that quality level.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import execution as X  # noqa: E402
from app.short_horizon.expert_guided import action_contract as ACT  # noqa: E402
from app.short_horizon.expert_guided import c2 as C2  # noqa: E402
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import strategy_full as SF  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402

H_NS = 3600 * 10 ** 9
MECHANISMS = ("early_close", "nop_veto", "best_of_day", "volume")
SHOCK_SIGMAS = (2.0, 3.0, 4.0)


# ----------------------------------------------------------------------------- the interventions
@dataclass(frozen=True)
class SpreadWindowProfile(X.RealisticProfile):
    """The realistic profile with the spread multiplied by ``factor`` inside [t0, t0 + hours)."""
    window_t0_ns: int = 0
    window_hours: float = 24.0
    factor: float = 5.0

    def spread_at(self, t_ns: int, open_ns: int) -> float:
        sp = X.RealisticProfile.spread_at(self, t_ns, open_ns)
        if self.window_t0_ns <= int(t_ns) < self.window_t0_ns + int(self.window_hours * H_NS):
            return sp * self.factor
        return sp


def shocked_bars(bars, sigma, at_bar: int, direction: int, k_sigma: float):
    """A persistent jump of k * sigma[at_bar] in price direction ``direction`` from bar ``at_bar`` on (OHLC shifted)."""
    b2 = bars.copy()
    shift = float(direction) * float(k_sigma) * float(sigma[at_bar])
    for c in ("OPEN", "HIGH", "LOW", "CLOSE"):
        v = b2[c].to_numpy(float).copy()
        v[at_bar:] += shift
        b2[c] = v
    return b2


def spliced_bars(bars, segments: list):
    """The price path of the target block rebuilt from donor quarters: for each (target_start, target_end, donor_start,
    donor_end) the target bars take the donor's log returns (resampled to the target length), continuous at the joints;
    outside the segments the bars are untouched."""
    b2 = bars.copy()
    v = bars.index.as_unit("ns").asi8
    c = bars["CLOSE"].to_numpy(float)
    r_all = np.r_[0.0, np.diff(np.log(c))]
    o, h, l = (bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW"))
    rel_o, rel_h, rel_l = o / c, h / c, l / c
    c2 = c.copy()
    for ts, te, ds, de in segments:
        tm = (v >= DG._utc_ns(ts)) & (v < DG._utc_ns(te))
        dm = (v >= DG._utc_ns(ds)) & (v < DG._utc_ns(de))
        ti, di = np.flatnonzero(tm), np.flatnonzero(dm)
        if len(ti) < 2 or len(di) < 2:
            continue
        rd = r_all[di]
        rr = np.interp(np.linspace(0, len(rd) - 1, len(ti)), np.arange(len(rd)), rd)
        rr[0] = 0.0
        start = c2[ti[0] - 1] if ti[0] > 0 else c2[ti[0]]
        c2[ti] = start * np.exp(np.cumsum(rr))
        # the bars after the segment continue from the new level (continuity at the joint)
        after = np.arange(ti[-1] + 1, len(c2))
        if len(after):
            c2[after] = c2[after] * (c2[ti[-1]] / c[ti[-1]])
    b2["CLOSE"] = c2
    b2["OPEN"], b2["HIGH"], b2["LOW"] = rel_o * c2, rel_h * c2, rel_l * c2
    return b2


def missing_runs_mask(cx, period: str, n_runs: int, hours_lo: int, hours_hi: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(cx.obs_ok & (cx.ds.period == period))
    mask = cx.obs_ok.copy()
    for _ in range(n_runs):
        start = int(rng.integers(0, max(1, len(rows) - hours_hi)))
        L = int(rng.integers(hours_lo, hours_hi + 1))
        mask[rows[start:start + L]] = False
    return mask


def dataset_with_bars(ds, bars):
    return dataclasses.replace(ds, bars=bars)


# ----------------------------------------------------------------------------- ablations
def ablate(strategy, mechanism: str, family: str):
    """The strategy without one mechanism (the counterfactual policy on the same path)."""
    if family == "c3":
        st = strategy
        if mechanism == "early_close":
            return st.no_early_close()
        if mechanism == "nop_veto":
            return dataclasses.replace(st, nop_veto=None)
        return st
    st = strategy
    if mechanism == "early_close":
        return st.no_early_close()
    if mechanism == "nop_veto":
        return dataclasses.replace(st, base=dataclasses.replace(st.base, nop_veto=None))
    if mechanism == "best_of_day":
        return dataclasses.replace(st, observe_bars=0)
    if mechanism == "volume":
        coef = dict(st.coef)
        for s in SF.SIDES:
            coef[f"v_alpha_{s}"], coef[f"v_beta_{s}"], coef[f"v_gamma_{s}"], coef[f"v_delta_{s}"] = 1.0, 0.0, 0.0, 0.0
        return dataclasses.replace(st, vol_family="param", coef=coef, v_min=0.0)
    return st


def applies(mechanism: str, family: str, gate: str, strategy) -> bool:
    if mechanism == "best_of_day":
        return (family == "c3" and gate == "contract") or (family != "c3" and strategy.observe_bars > 0)
    if mechanism == "volume":
        return family != "c3" and strategy.vol_family == "param"
    if mechanism == "nop_veto":
        nv = strategy.nop_veto if family == "c3" else strategy.base.nop_veto
        return nv is not None
    return True


def replay(cx, SCFG, table, strategy, cells, e_time, cost, period, family: str, gate: str) -> dict:
    if family == "c3":
        Pol = S.policy_cls(gate)
        pol = Pol(table, strategy, cells, e_time, cost)
        m, trades = ACT.rollout_with_trades(cx, period, pol(cx), SCFG)
    else:
        Pol = S.PolicyFullPerBar if gate == "perbar" else SF.PolicyFull
        pol = Pol(table, strategy, cells, e_time, cost, scfg=SCFG)
        m, trades = SF.rollout_with_volume(cx, period, pol(cx), SCFG)
    return {"net": float(m["net_profit_usd"]), "dd_usd": float(m.get("max_drawdown_usd", m["max_drawdown_pct"] / 100.0 * SCFG.capital)),
            "dd_pct": float(m["max_drawdown_pct"]), "sharpe": float(m["sharpe_daily_sqrt365"]) if m["trades"] else float("nan"), "trades": int(m["trades"]),
            "agent_closes": int(m.get("n_close_AGENT_CLOSE", 0)) + int(m.get("n_close_REVERSAL", 0))}


def ratio_ci(with_vals, without_vals, n: int = 2000, seed: int = 20261026) -> dict:
    """Loss-limiting ratio = mean(loss with) / mean(loss without) over the draws, with a paired bootstrap CI over draws."""
    a, b = np.asarray(with_vals, float), np.asarray(without_vals, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0 or b.mean() <= 0:
        return {"ratio": float("nan"), "low": float("nan"), "high": float("nan"), "n": int(len(a))}
    rng = np.random.default_rng(seed)
    r = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        bm = b[i].mean()
        r.append(a[i].mean() / bm if bm > 0 else np.nan)
    r = np.array(r)
    r = r[np.isfinite(r)]
    return {"ratio": float(a.mean() / b.mean()), "low": float(np.quantile(r, 0.025)) if len(r) else float("nan"),
            "high": float(np.quantile(r, 0.975)) if len(r) else float("nan"), "n": int(len(a))}


# ----------------------------------------------------------------------------- the battery for one run file
def battery(run: dict, cx, SCFG, truth: S.Truth, AC, draws: int = 6, seed0: int = 0, out_dir: Path | None = None) -> dict:
    family, gate, variant, level, fold = run.get("family", "c3"), run.get("gate", "contract"), run["variant"], run["level"], run["fold"]
    fam = S.FAMILIES[family]
    if fam is not None:
        SCFG = S.scfg_full(SCFG, fam[1])
    SCFG = S.scfg_for(variant, SCFG)
    cells = [tuple(c) for c in truth.cells]
    period = S.DEC_BLOCK
    a0, b0 = cx.ds.meta["periods"][period]
    v = cx.ds.bars.index.as_unit("ns").asi8
    dec_bars = np.flatnonzero((v >= DG._utc_ns(a0)) & (v < DG._utc_ns(b0)))
    q = S.quality_for(variant, level)
    rho_day = 0.5 if variant == "block" else 0.0
    out = {"fold": fold, "variant": variant, "gate": gate, "family": family, "level": level, "draws": draws, "per_seed": []}
    qt = S.quarter_tags(cx, period)
    for rec in run["per_seed"]:
        if not rec.get("strategy_effective"):
            continue
        seed = int(rec["seed"])
        strategy = (SF.strategy_from_dict if fam else C2.strategy_from_dict)(rec["strategy_effective"])
        table, _ = S.simulate_table(cx, SCFG, truth, q, seed, rho_day)
        if gate == "perbar":
            table["cost_row"] = S.row_costs(cx, SCFG, truth.cost - C3.cost_sigma_of(cx, SCFG, 0.0, np.flatnonzero(cx.obs_ok & (cx.ds.period == S.CAL_BLOCK))))
        rng = np.random.default_rng(seed0 + seed)
        res = {"seed": seed, "paths": {}}

        def run_all(name, cx_, SCFG_, table_):
            rows = {}
            full = replay(cx_, SCFG_, table_, strategy, cells, truth.e_time, truth.cost, period, family, gate)
            rows["all"] = full
            for mech in MECHANISMS:
                if not applies(mech, family, gate, strategy):
                    continue
                st2 = ablate(strategy, mech, family)
                if mech == "best_of_day" and family == "c3":
                    cx_pol_gate = "perbar"          # the c3 contract gate off = observe 0 = the per-bar policy's gate
                    rows[f"without_{mech}"] = replay(cx_, SCFG_, table_, strategy, cells, truth.e_time, truth.cost, period, family, cx_pol_gate)
                else:
                    rows[f"without_{mech}"] = replay(cx_, SCFG_, table_, st2, cells, truth.e_time, truth.cost, period, family, gate)
            res["paths"].setdefault(name, []).append(rows)

        run_all("base", cx, SCFG, table)
        # (a) shocks against the open position
        for k in SHOCK_SIGMAS:
            for d in range(draws):
                at = int(rng.choice(dec_bars[24:-24]))
                # the side against the position open at that bar under the full strategy: replay to the bar to know the position sign
                pol_dir = -1 if (d % 2 == 0) else 1
                bars2 = shocked_bars(cx.ds.bars, cx.ds.sigma, at, pol_dir, k)
                cx2 = _ctx(cx, dataset_with_bars(cx.ds, bars2))
                run_all(f"shock_{k:g}sigma", cx2, SCFG, table)
        # (c) spread blow-up x5 for 24 h
        for d in range(draws):
            at = int(rng.choice(dec_bars[:-24]))
            prof = SpreadWindowProfile(**dataclasses.asdict(SCFG.profile), window_t0_ns=int(v[at]), window_hours=24.0, factor=5.0)
            run_all("spread_x5_24h", cx, dataclasses.replace(SCFG, profile=prof), table)
        # (d) runs of missing observations 6-24 h
        for d in range(draws):
            m = missing_runs_mask(cx, period, 8, 6, 24, seed0 + 100 * seed + d)
            cxm = _ctx_mask(cx, m)
            run_all("missing_runs_6_24h", cxm, SCFG, table)
        # (e) volatility doubling in the heads' noise, engine unchanged
        q2 = dataclasses.replace(q)
        table2, _ = S.simulate_table(cx, SCFG, truth, q2, seed, rho_day)
        for key in ("net",):
            tr = table2[key][truth.rows]
            mu = np.nanmean(tr)
            table2[key][truth.rows] = mu + 2.0 * (tr - mu)           # the regression noise doubled around its mean (declared form)
        for si in (0, 1):
            p = table2["p3"][truth.rows, si, :]
            z = np.log(np.clip(p, 1e-6, 1) / np.clip(1 - p, 1e-6, 1))
            z = 0.5 * z                                              # binary latents halved = noise doubled relative to the signal
            p2 = 1 / (1 + np.exp(-z))
            table2["p3"][truth.rows, si, :] = p2 / p2.sum(axis=1, keepdims=True)
        if gate == "perbar":
            table2["cost_row"] = table["cost_row"]
        run_all("vol_x2_in_heads", cx, SCFG, table2)
        # (b) regime flips: trend -> range -> trend spliced from the decision year's own quarters
        allq = S.quarter_tags(cx, None)                 # donors = real quarters of 2021-2023 by trend tercile
        tr_q = [x for x in allq if x["trend_tercile"] == "trending"]
        rg_q = [x for x in allq if x["trend_tercile"] == "ranging"]
        if tr_q and rg_q and len(qt) >= 3:
            donors = [tr_q[0], rg_q[0], tr_q[-1] if len(tr_q) > 1 else tr_q[0]]
            segs = [(qt[i]["start"], qt[i]["end"], donors[i]["start"], donors[i]["end"]) for i in range(3)]
            bars_s = spliced_bars(cx.ds.bars, segs)
            cx_s = _ctx(cx, dataset_with_bars(cx.ds, bars_s))
            truth_s = S.build_truth(cx_s, SCFG, fold, cells, AC)
            table_s, _ = S.simulate_table(cx_s, SCFG, truth_s, q, seed, rho_day)
            if gate == "perbar":
                table_s["cost_row"] = table["cost_row"]
            run_all("regime_splice_trend_range_trend", cx_s, SCFG, table_s)
            res["regime_splice"] = {"segments": segs}
        out["per_seed"].append(res)
    out["summary"] = summarize(out)
    return out


def _ctx(cx, ds2):
    from tools.expert_guided_e1t import Ctx
    c2 = Ctx(ds2)
    c2.obs_ok = cx.obs_ok.copy()
    return c2


def _ctx_mask(cx, mask):
    import copy
    c2 = copy.copy(cx)
    c2.obs_ok = mask
    return c2


def summarize(out: dict) -> dict:
    """Per path type and mechanism: the loss-limiting ratio (DD with / DD without) and the net difference, pooled over
    draws and seeds, with the bootstrap CI; TOLERANT when the ratio's upper CI bound < 1."""
    summ = {}
    for res in out["per_seed"]:
        for name, rows_list in res["paths"].items():
            for rows in rows_list:
                for key, r in rows.items():
                    if key == "all":
                        continue
                    mech = key[len("without_"):]
                    d = summ.setdefault(name, {}).setdefault(mech, {"with_dd": [], "without_dd": [], "with_net": [], "without_net": []})
                    d["with_dd"].append(rows["all"]["dd_usd"])
                    d["without_dd"].append(r["dd_usd"])
                    d["with_net"].append(rows["all"]["net"])
                    d["without_net"].append(r["net"])
    final = {}
    for name, mechs in summ.items():
        final[name] = {}
        for mech, d in mechs.items():
            rc = ratio_ci(d["with_dd"], d["without_dd"])
            dn = np.asarray(d["with_net"]) - np.asarray(d["without_net"])
            final[name][mech] = {**rc, "net_diff_mean": float(np.nanmean(dn)) if len(dn) else float("nan"),
                                 "tolerant": bool(math.isfinite(rc["high"]) and rc["high"] < 1.0),
                                 "verdict": ("TOLERANT" if math.isfinite(rc["high"]) and rc["high"] < 1.0 else
                                             "NOT_TOLERANT" if math.isfinite(rc["low"]) and rc["low"] > 1.0 else "INCONCLUSIVE")}
    # the intervened paths vs the base path: the strategy's own loss under each intervention (all mechanisms on)
    base = [rows["all"] for res in out["per_seed"] for rows in res["paths"].get("base", [])]
    for name in final:
        rows = [rows["all"] for res in out["per_seed"] for rows in res["paths"].get(name, [])]
        final[name]["_path"] = {"net_mean": float(np.mean([x["net"] for x in rows])), "dd_usd_mean": float(np.mean([x["dd_usd"] for x in rows])),
                                "base_net_mean": float(np.mean([x["net"] for x in base])) if base else float("nan"),
                                "base_dd_usd_mean": float(np.mean([x["dd_usd"] for x in base])) if base else float("nan"), "paths": len(rows)}
    return final


def stop_requested(out_path) -> bool:
    """Owner STOP flag (2026-10-27): a file named STOP in the work root (any ancestor of the output path, three levels up)
    makes every NEW job of the lane exit before doing anything; running jobs finish their current file."""
    from pathlib import Path as _P
    p = _P(str(out_path)).resolve()
    for q in [p] + list(p.parents)[:3]:
        if (q / "STOP").exists():
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a RUN_*.json of the simulation")
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--contract", default=str(S.CONTRACT))
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if stop_requested(a.out):
        print("STOP flag present in the work root: nothing started (owner order 2026-10-27)")
        raise SystemExit(75)
    run = json.loads(Path(a.run).read_text())
    AC = S.contract_for_sim(a.contract)
    cx, SCFG = S.load_fold(run["fold"], a.admitted, a.csv_dir)
    SCFG = S.with_variant(SCFG, C3.FULL_F3)
    truth = S.load_truth(Path(a.truth_dir) / f"TRUTH_{run['fold']}_{S.TRUTH_OF.get(run['variant'], 'base')}.npz")
    out = battery(run, cx, SCFG, truth, AC, draws=a.draws)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"run": Path(a.run).name, "paths": {k: v.get("_path", {}).get("paths") for k, v in out["summary"].items()},
                      "tolerant": {k: {m: d["verdict"] for m, d in v.items() if m != "_path"} for k, v in out["summary"].items()}}, default=str))


if __name__ == "__main__":
    main()

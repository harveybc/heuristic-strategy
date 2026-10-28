"""The closer at quality < 1 against the INFORMATION-LIMITED ORACLE (QUALIFICATION_SPEC D7, sealed before running).

For each cell, closer quality q in {1.00, 0.85, 0.70} and head-error autocorrelation phi in a panel: the strategy's own
run (opener and closer) and the same opener with the Bayes closer (a 2-state Kalman filter per position over the same
noisy head stream: A random walk, AR(1) error; close iff E[A | history] >= -EPS_CLOSE). Per side: E3 tick agreement,
E4 destructive-close share, mean regret / initial cash risk at SL, early-close value vs hold-to-SL, plus Sharpe, net,
trades, mean holding bars and the exit mix (the q0.85 Sharpe anomaly).

  python tools/strategy_closer_ladder.py --cells <QUEUE.json> --ids a:0,b:0 --params <json> --admitted <d> --csv-dir <d> \
      --qualities 1.0,0.85,0.70 --phis 0,0.587,0.98 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
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


class PolicyBayes(ZZ.PolicyC3Z):
    """The strategy's opener with the information-limited Bayes closer."""

    def close_now(self, r, si, bar, pos) -> bool:
        self.positions.setdefault((int(pos.entry_bar), int(si)), __import__("dataclasses").replace(pos))
        cl = self.t["closer"]
        a_hat, s_hat, y, h, mu, sd_a, sd_e = cl.advantage_full(r, bar, pos)
        if not np.isfinite(y):
            dec, mean_a = False, float("nan")
        elif not (sd_e > 0):
            mean_a = float(y)                                         # perfect head: the posterior is the truth
            dec = bool(mean_a >= -BM.EPS_CLOSE)
        else:
            key = (int(pos.entry_bar), int(si))
            st = getattr(self, "_kf", {})
            self._kf = st
            phi = float(S.CLOSER_NOISE_PHI)
            q = sd_a * sd_a / S.H_REF_BARS
            if key not in st:
                m = np.array([mu, 0.0])
                P = np.array([[sd_a * sd_a, 0.0], [0.0, sd_e * sd_e]])
            else:
                m0, P0, b0, sde0 = st[key]
                g = max(int(bar) - b0, 1)                             # bars since the last executable tick (the AR stream runs per bar)
                pg = phi ** g
                F = np.array([[1.0, 0.0], [0.0, pg * sd_e / sde0 if sde0 > 0 else 0.0]])   # the error rescales with sd_e(h) as the lead moves
                Qm = np.array([[q * g, 0.0], [0.0, (1.0 - pg * pg) * sd_e * sd_e]])
                m, P = F @ m0, F @ P0 @ F.T + Qm
            Hm = np.array([1.0, 1.0])
            Sv = float(Hm @ P @ Hm)
            K = (P @ Hm) / Sv
            m = m + K * (float(y) - float(Hm @ m))
            P = P - np.outer(K, Hm @ P)
            st[key] = (m, P, int(bar), float(sd_e))
            mean_a = float(m[0])
            dec = bool(mean_a >= -BM.EPS_CLOSE)
        rem, psl = cl.value(r, bar, pos)
        self.closes.append((int(r), int(bar), int(si), float(rem), float(psl), dec, False, False, False, float(mean_a), "bayes"))
        return dec


def measure(w, strat, view, cells, Pol, closer_exact) -> dict:
    cx, SCFG, truth = w["cx"], w["scfg"], w["truth"]
    pol = Pol(view, strat, cells, truth.e_time, truth.cost, observe_bars=int(round(ZZ.PARAMS.get("observe_bars", 0))))
    m, trades = BM.ACT.rollout_with_trades(cx, BM.BLOCK, pol(cx), SCFG)
    roll = {"metrics": m, "trades": trades, "log": pol.log, "closes": pol.closes, "positions": pol.positions}
    st = BM.exit_stats(w, roll, closer_exact)
    blk_end = BM.H.period_bars(cx.ds, BM.BLOCK)[1]
    out = {"sharpe": float(m.get("sharpe_daily_sqrt365", float("nan"))), "net_usd": float(m.get("net_profit_usd", float("nan"))), "trades": len(trades),
           "mean_hold_bars": float(np.mean([int(t["exit_bar_abs"]) - int(t["entry_bar_abs"]) for t in trades])) if trades else float("nan")}
    for si, side in ((0, "BUY"), (1, "SELL")):
        reg, ecv = [], 0.0
        for t in trades:
            if (0 if int(t["direction"]) > 0 else 1) != si:
                continue
            pos = pol.positions.get((int(t["entry_bar_abs"]), si))
            if pos is None:
                continue
            path = closer_exact.path(pos)
            if t["exit_reason"] == "TIME" and int(t["exit_bar_abs"]) >= blk_end - 2 and int(path["exit_bar"]) > int(t["exit_bar_abs"]):
                continue                                              # censored by the block end (as exit_stats)
            risk = abs(float(pos.entry_price) - float(pos.sl)) * max(int(pos.units), 1)
            pstar = max(float(np.max(path["P"])) if len(path["P"]) else -np.inf, path["hold_net"])
            reg.append((pstar - float(t["net_usd"])) / risk)
            if path["hold_reason"] == "SL" and t["exit_reason"] == "EARLY":
                ecv += (float(t["net_usd"]) - path["hold_net"]) / risk
        out[side] = {"E3": st[si]["agreement"], "E4": st[si]["destructive_share"], "closes": st[si]["closes"], "regret_norm": float(np.mean(reg)) if reg else float("nan"),
                     "early_close_value_norm": ecv, "sl_destined_closed_early": st[si]["sl_destined_closed_early"], "R_mean_sigma": st[si]["R_mean"], "mix": st[si]["mix"], "sl_destined": st[si]["sl_destined"]}
    out["_trades_key"] = [(int(t["entry_bar_abs"]), int(t["exit_bar_abs"]), int(t["direction"]), round(float(t["net_usd"]), 6)) for t in trades]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--qualities", default="1.0,0.85,0.70")
    ap.add_argument("--phis", default="0,0.587,0.98")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    AC = S.contract_for_sim(str(S.CONTRACT))
    cx, SCFG = S.load_fold("wf2022", a.admitted, a.csv_dir)
    SCFG = S.with_variant(SCFG, C3.FULL_F3)
    params = SP.load(a.params)
    ZZ.configure(params)
    genes = SP.genes_c3(params)
    cells = {c["id"]: c for c in json.loads(Path(a.cells).read_text())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.ids.split(","):
        cid, seed = spec.rsplit(":", 1)
        t0 = time.time()
        w = BM.cell_world(cx, SCFG, AC, cells[cid], int(seed))
        strat, view0, cl = BM.strategy_for(w, genes, AC)
        exact = view0["closer"].with_quality(1.0)
        rec = {"cell": cid, "seed": int(seed), "params_sha256": SP.sha256_of(a.params), "rows": []}
        for phi in [float(x) for x in a.phis.split(",")]:
            S.CLOSER_NOISE_PHI = phi
            for q in [float(x) for x in a.qualities.split(",")]:
                if q >= 0.999 and phi != float(a.phis.split(",")[0]):
                    continue                                          # quality 1 carries no noise: once
                view = dict(view0)
                view["closer"] = view0["closer"].with_quality(q)
                row = {"quality": q, "phi": phi, "strategy": measure(w, strat, view, cl, ZZ.PolicyC3Z, exact),
                       "bayes": measure(w, strat, view, cl, PolicyBayes, exact)}
                if q >= 0.999:
                    row["q1_bayes_equals_strategy"] = row["strategy"]["_trades_key"] == row["bayes"]["_trades_key"]
                for k in ("strategy", "bayes"):
                    row[k].pop("_trades_key")
                rec["rows"].append(row)
                print(json.dumps({"cell": spec, "q": q, "phi": phi, "strategy": {k: row["strategy"][k] for k in ("sharpe", "net_usd", "trades", "mean_hold_bars")},
                                  "E4": [row["strategy"][s]["E4"] for s in ("BUY", "SELL")], "E4_bayes": [row["bayes"][s]["E4"] for s in ("BUY", "SELL")]}, default=float), flush=True)
        rec["sec"] = round(time.time() - t0, 1)
        (out / f"LADDER_{cid}_s{seed}.json").write_text(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()

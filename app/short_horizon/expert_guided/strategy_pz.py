"""PZ: the PRAGMATIC strategy of the ZigZag-guided lane (docs/short_horizon/clean_round/ESTRATEGIA_PRAGMATICA_PZ_2026_10_30.md).

It runs next to C3Z full_f4 (the formal strategy) on the same contract, engine and worlds, and wins only if it proves
it. Everything here is fixed by the design document; what the document leaves open is DECLARED below (``DECLARED``).

Model outputs (§2, the minimum version: 24 outputs per tick): per MT ZigZag horizon h in {3, 6, 12, 24} (the levels of
strategy_zz.MT_HORIZONS) and per side s in {BUY, SELL}:
  p_up[h]      1 if the next CONFIRMED pivot of level h after the decision bar is a PEAK, 0 if a VALLEY
  fav[h, s]    favourable excursion from the executable price (open of b + 1) to the next pivot of level h in the
               direction of s (the next PEAK for a buy, the next VALLEY for a sell), price units
  adv[h, s]    maximum adverse excursion from that price until that pivot, the pivot's own bar included (MAE)
  tau[h]       bars (hours) until the next confirmed pivot of level h (the pivot p_up names)
A label whose pivot does not exist (beyond the data, or never confirmed) is CENSORED (NaN), never a fabricated NOP.

Opening (§3), per decision row and every (h, s) with an executable order:
  p = p_up[h] (buy) or 1 - p_up[h] (sell);  SL = a_sl x adv_q75 + buffer;  TP = a_tp x fav_q50   (horizon_levels rounding)
  G = p x TP - (1 - p) x SL - round-trip cost - kappa x (fav_q75 - fav_q25);  score = (G / SL) / (tau_q50 + tau0)^alpha
  open the argmax (h, s) when score > theta; both sides are scored before deciding; an exact side tie is a NOP.
Close (§4), per tick of an open position (side s, opening horizon h_p, price P, broker levels SL / TP):
  HOLD = p' (TP - P) - (1 - p') (P - SL) in price, p' = the probability in favour of s at h_p now;
  close when HOLD <= band - cost_close on k consecutive ticks (O*'s tie rule: close unless holding beats closing by more
  than the band), or at once when p'_f < p_urg, p'_f the probability in favour at the next finer level (h = 3 uses 3).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .. import zigzag as Z
from . import c3 as C3
from . import env as E
from . import horizon_levels as HL
from . import menu as M
from . import strategy_zz as ZZ

DESIGN_VERSIONS = ("1.0", "1.1")
DESIGN_VERSION = "1.1"                                       # coordinator ruling 2026-10-30 (after PZ3): a DECLARED DESIGN CORRECTION (document section 7);
                                                             # "1.0" reproduces the pre-registered history byte for byte (the v1.0 tier-1 evidence)
V11_RULES = {"tp": "TP distance = a_tp x fav_q50 - (limit_through + the exit-side spread at the decision hour), rounded inward: a TP at a "
                   "predicted extreme is unexecutable by construction (the engine fills a TP limit only when the price trades THROUGH it by "
                   "limit_through); exit side: a long exits at the BID (the bars are BID: no spread), a short exits at the ASK (the hour's "
                   "spread when the engine triggers shorts on the ask, exec_variants short_trigger_on_ask)",
             "urgent": "close at once only when the fine level turns against the position (p'_f < p_urg) AND its reversal is predicted to "
                       "reach the SL: the fine level's opposite-side fav_q50 >= the remaining distance to the SL; p_urg = 0 switches the "
                       "urgent close off"}
HORIZONS = tuple(int(h) for h in M.MENU_HORIZONS)            # (3, 6, 12, 24): the four MT ZigZag levels
PHI = 0.587                                                  # AR(1) lag-1 of the head error (strategy_output_simulation.CLOSER_NOISE_PHI)
EPS_CLOSE = 0.02                                             # O*'s band, sigma of the trade (strategy_behaviour_matrix.EPS_CLOSE)
QUANTS = (0.05, 0.25, 0.5, 0.75, 0.9, 0.95)
MIN_OUTPUTS = ("p_up", "fav_q50", "adv_q75", "tau_q50")       # per horizon: p_up 1 + fav 2 + adv 2 + tau 1 = 6; x 4 = 24
N_MIN_OUTPUTS = len(HORIZONS) * (1 + 2 + 2 + 1)
GENES = ("a_sl", "a_tp", "buffer_pips", "kappa", "theta", "alpha", "tau0", "rho", "k_persist", "p_urg")
BOUNDS = {"a_sl": [0.5, 2.0], "a_tp": [0.5, 2.0], "buffer_pips": [0.0, 5.0], "kappa": [0.0, 1.0], "theta": [-0.2, 1.0], "alpha": [0.0, 1.0],
          "tau0": [0.5, 6.0], "rho": [0.001, 0.01], "k_persist": [1, 3], "p_urg": [0.0, 0.5]}   # p_urg [0, 0.5], 0 = off (v1.1; v1.0 [0.05, 0.5])
DISCRETE = {"alpha": (0.0, 0.5, 1.0), "k_persist": (1, 2, 3)}
CANONICAL = {"a_sl": 1.0, "a_tp": 1.0, "buffer_pips": 1.0, "rho": 0.005,      # the shared contract's baseline (horizon_levels.GENE_BASELINE)
             "kappa": 0.0, "theta": 0.0, "alpha": 0.0, "tau0": 1.0,        # break-even open, pure profit / risk (the predictions strategy)
             "k_persist": 1, "p_urg": 0.25}                                 # declared before any run (plan to the coordinator, 2026-10-30)
COST_CLOSE_SHARE = 0.5
NOISE_SEED_MULT, NOISE_SEED_ADD = 104729, 31
DECLARED = {
    "labels": "the FINAL pivots of each MT level whose knowable confirmation bar exists (zigzag.confirmation_bars_fast >= 0); "
              "excursions measured from OPEN[b + 1] (the engine's fill price; costs are charged by the engine) on the BID bars",
    "orders": "TP / SL prices from the decision bar's close (the env's reference quote) through horizon_levels.levels with "
              "tp_reference = fav_q50, mae_reference = adv_q75, sigma 0, b_sl = b_tp = 0: SL rounded away, TP toward the entry; "
              "an order the engine would refuse (env._levels_absolute) is not a candidate",
    "cost": "round-trip cost = the row's own cost (strategy_output_simulation.row_costs: spread + 2 slippages + commission + "
            "the declared allowance, sigma at b + 1) x sigma[b + 1] in price; cost_close = 0.5 x the round trip",
    "band": "O*'s band EPS_CLOSE x the trade's sigma (pos.sigma): close when HOLD <= band - cost_close",
    "opening_horizon": "remembered per order at the open (side, TP, SL); a position the policy did not open reads h = 6",
    "reverse": "no reversal (the C3Z canonical allow_reverse = 0): a close is a close, the next flat tick may open",
    "ties": "within a side the finest horizon wins a tie (menu.tie_order 'finest'); an exact tie between the two sides is a NOP",
    "censored": "a slot whose p_up / fav / adv label is censored is not a candidate; tau censored is inert at alpha = 0, "
                "excludes the slot otherwise; a censored p' never triggers the HOLD close",
    "heads_below_1": "each of the 24 outputs through its own AR(1) stream (phi 0.587): p_up as the latent z = rho (2y - 1) + "
                     "sqrt(1 - rho^2) u, Platt-calibrated on the CAL rows; fav / adv / tau as the conditional-mean regression "
                     "mu + rho^2 (y - mu) + rho sqrt(1 - rho^2) sd u; their quantiles = the prediction + the empirical "
                     "quantiles of the CAL residuals; rho = sin(pi (q - 1/2)) as every other head",
}


class PZError(ValueError):
    pass


def validate_genes(g: dict) -> dict:
    out = dict(CANONICAL)
    unknown = set(g) - set(GENES)
    if unknown:
        raise PZError(f"PZ_UNKNOWN_GENES:{sorted(unknown)}")
    out.update({k: float(v) for k, v in g.items()})
    for k in GENES:
        lo, hi = BOUNDS[k]
        v = float(out[k])
        if not (math.isfinite(v) and lo - 1e-12 <= v <= hi + 1e-12):
            raise PZError(f"PZ_GENE_OUT_OF_BOUNDS:{k}={v}:{BOUNDS[k]}")
        if k in DISCRETE and not any(abs(v - x) < 1e-9 for x in DISCRETE[k]):
            raise PZError(f"PZ_GENE_NOT_AN_OPTION:{k}={v}:{DISCRETE[k]}")
    out["k_persist"] = int(round(out["k_persist"]))
    return out


def genes_sha(g: dict) -> str:
    return hashlib.sha256(json.dumps(validate_genes(g), sort_keys=True).encode()).hexdigest()


def load_genes(path) -> dict:
    d = json.loads(Path(path).read_text())
    return validate_genes({k: d[k] for k in GENES if k in d})


def rho_of(quality: float) -> float:
    return float(min(1.0, max(0.0, math.sin(math.pi * (float(quality) - 0.5)))))


# ----------------------------------------------------------------------------- §2 the expert labels
def expert_labels(bars, mt: dict | None = None) -> dict:
    """Per bar b (decision after b): p_up (n, 4), tau (n, 4) in bars, fav / adv (n, 4, 2) in price; NaN = censored."""
    o, hi, lo = (np.asarray(bars[k], float) for k in ("OPEN", "HIGH", "LOW"))
    n = len(o)
    mt = mt if mt is not None else ZZ.mt_levels(bars)
    by_h = {v[2]: k for k, v in ZZ.MT_HORIZONS.items()}
    nh = len(HORIZONS)
    out = {"p_up": np.full((n, nh), np.nan), "tau": np.full((n, nh), np.nan), "fav": np.full((n, nh, 2), np.nan), "adv": np.full((n, nh, 2), np.nan)}
    if n < 2:
        return out
    b = np.arange(n - 1)
    entry = o[b + 1]
    for j, hh in enumerate(HORIZONS):
        m = mt[by_h[hh]]
        piv = [(int(pb), int(k), float(p)) for (pb, k, p), c in zip(m["pivots"], m["conf"]) if int(c) >= 0]
        if not piv:
            continue
        pb = np.array([x[0] for x in piv], np.int64)
        pk = np.array([x[1] for x in piv], np.int64)
        pl = np.array([x[2] for x in piv], float)
        jn = np.searchsorted(pb, b + 1, side="left")
        ok = jn < len(pb)
        out["p_up"][b[ok], j] = (pk[jn[ok]] == Z.PEAK).astype(float)
        out["tau"][b[ok], j] = (pb[jn[ok]] - b[ok]).astype(float)
        for si, kind in ((0, Z.PEAK), (1, Z.VALLEY)):
            sel = pk == kind
            kb, kl = pb[sel], pl[sel]
            if not len(kb):
                continue
            jk = np.searchsorted(kb, b + 1, side="left")
            okk = jk < len(kb)
            bb = b[okk]
            tgt = jk[okk]
            out["fav"][bb, j, si] = (kl[tgt] - entry[okk]) if si == 0 else (entry[okk] - kl[tgt])
            ext = np.full(n, np.nan)                                     # extreme of LOW (buy) / HIGH (sell) over [k, next pivot of the kind]
            prev = -1
            for t in range(len(kb)):
                s0, s1 = prev + 1, int(kb[t])
                if s1 >= s0:
                    seg = lo[s0:s1 + 1] if si == 0 else hi[s0:s1 + 1]
                    ext[s0:s1 + 1] = (np.minimum.accumulate(seg[::-1]) if si == 0 else np.maximum.accumulate(seg[::-1]))[::-1]
                prev = s1
            e1 = ext[bb + 1]
            out["adv"][bb, j, si] = (entry[okk] - e1) if si == 0 else (e1 - entry[okk])
    return out


# ----------------------------------------------------------------------------- the heads at a quality (the ladder)
def _platt(z, y, iters: int = 50) -> tuple:
    """Logistic fit p = 1 / (1 + exp(-(a z + b))) by Newton steps with a tiny ridge; (1, 0) when one class only."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    if len(z) < 2 or y.min() == y.max():
        return 1.0, 0.0
    w = np.zeros(2)
    X = np.stack([z, np.ones_like(z)], axis=1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -40, 40)))
        g = X.T @ (p - y) + 1e-6 * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[0]), float(w[1])


def heads(labels: dict, quality: float, seed: int, cal_bars, phi: float = PHI) -> dict:
    """The PZ heads at ``quality``: p_up (n, 4) calibrated probability; fav / adv / tau as {quantile: array}. At
    quality 1 every quantile is the label (the perfect head). ``cal_bars``: bool (n,) the CAL rows' decision bars."""
    rho = rho_of(quality)
    p_up, tau, fav, adv = (np.asarray(labels[k], float) for k in ("p_up", "tau", "fav", "adv"))
    n = p_up.shape[0]
    if rho >= 1.0 - 1e-12:
        return {"p_up": p_up.copy(), "fav": {q: fav.copy() for q in QUANTS}, "adv": {q: adv.copy() for q in QUANTS},
                "tau": {q: tau.copy() for q in QUANTS}, "rho": 1.0, "mode": "perfect", "pred": {"fav": fav.copy(), "adv": adv.copy(), "tau": tau.copy()}}
    cal = np.asarray(cal_bars, bool)
    rng = np.random.default_rng(NOISE_SEED_MULT * int(seed) + NOISE_SEED_ADD)
    k = math.sqrt(max(1.0 - rho * rho, 0.0))
    out = {"p_up": np.full_like(p_up, np.nan), "fav": {q: np.full_like(fav, np.nan) for q in QUANTS}, "adv": {q: np.full_like(adv, np.nan) for q in QUANTS},
           "tau": {q: np.full_like(tau, np.nan) for q in QUANTS}, "rho": rho, "mode": f"ladder rho={rho:.3f} phi={phi}",
           "pred": {"fav": np.full_like(fav, np.nan), "adv": np.full_like(adv, np.nan), "tau": np.full_like(tau, np.nan)}, "platt": {}}

    def reg(y):
        fin = np.isfinite(y)
        u = ZZ._ar1(rng, n, phi)
        if not fin.any():
            return np.full(n, np.nan), {q: np.full(n, np.nan) for q in QUANTS}
        mu, sd = float(np.mean(y[fin])), float(np.std(y[fin]))
        yh = np.where(fin, mu + rho * rho * (y - mu) + rho * k * sd * u, np.nan)
        m = fin & cal
        res = (y - yh)[m]
        qs = {q: (yh + float(np.quantile(res, q))) if len(res) else np.full(n, np.nan) for q in QUANTS}
        return yh, qs

    for j in range(p_up.shape[1]):                                     # stream order per horizon: p_up, fav B, fav S, adv B, adv S, tau
        y = p_up[:, j]
        fin = np.isfinite(y)
        u = ZZ._ar1(rng, n, phi)
        z = np.where(fin, rho * (2 * y - 1) + k * u, np.nan)
        m = fin & cal
        a, b0 = _platt(z[m], y[m])
        out["platt"][HORIZONS[j]] = (a, b0)
        out["p_up"][:, j] = np.where(fin, 1.0 / (1.0 + np.exp(-np.clip(a * z + b0, -40, 40))), np.nan)
        for key in ("fav", "adv"):
            for si in (0, 1):
                yh, qs = reg(fav[:, j, si] if key == "fav" else adv[:, j, si])
                out["pred"][key][:, j, si] = yh
                for q in QUANTS:
                    out[key][q][:, j, si] = np.maximum(qs[q], 0.0) if key == "adv" else qs[q]
        yh, qs = reg(tau[:, j])
        out["pred"]["tau"][:, j] = yh
        for q in QUANTS:
            out["tau"][q][:, j] = np.maximum(qs[q], 0.0)
    return out


def head_report(labels: dict, hd: dict, eval_bars, cal_bars) -> dict:
    """Per output family on the evaluation bars: interval coverage (nominal 0.90 = [q05, q95]; fav [q25, q75] nominal
    0.50; adv <= q75 nominal 0.75, <= q90 nominal 0.90), MAE of the median against the label, the naive MAE (the CAL
    mean) and MAE / naive; p_up: Brier, naive Brier (CAL base rate), accuracy, ECE (10 bins)."""
    ev, cal = np.asarray(eval_bars, bool), np.asarray(cal_bars, bool)
    rep = {"rho": hd["rho"], "mode": hd["mode"]}
    for key in ("fav", "adv", "tau"):
        y = np.asarray(labels[key], float)
        sel = np.broadcast_to(ev.reshape((-1,) + (1,) * (y.ndim - 1)), y.shape) & np.isfinite(y)
        cs = np.broadcast_to(cal.reshape((-1,) + (1,) * (y.ndim - 1)), y.shape) & np.isfinite(y)
        q = hd[key]
        f = sel & np.isfinite(q[0.05]) & np.isfinite(q[0.95])
        r = {"n": int(sel.sum()), "coverage_90": float(np.mean((y[f] >= q[0.05][f]) & (y[f] <= q[0.95][f]))) if f.any() else float("nan")}
        med = q[0.5]
        fm = sel & np.isfinite(med)
        mae = float(np.mean(np.abs(med[fm] - y[fm]))) if fm.any() else float("nan")
        mu = float(np.mean(y[cs])) if cs.any() else float("nan")
        naive = float(np.mean(np.abs(mu - y[sel]))) if sel.any() and np.isfinite(mu) else float("nan")
        r.update({"mae_q50": mae, "naive_mae": naive, "mae_over_naive": (mae / naive) if naive and np.isfinite(naive) and naive > 0 else float("nan")})
        if key == "fav":
            g = sel & np.isfinite(q[0.25]) & np.isfinite(q[0.75])
            r["coverage_50_q25_q75"] = float(np.mean((y[g] >= q[0.25][g]) & (y[g] <= q[0.75][g]))) if g.any() else float("nan")
        if key == "adv":
            g = sel & np.isfinite(q[0.75])
            r["share_below_q75"] = float(np.mean(y[g] <= q[0.75][g])) if g.any() else float("nan")
            g = sel & np.isfinite(q[0.9])
            r["share_below_q90"] = float(np.mean(y[g] <= q[0.9][g])) if g.any() else float("nan")
        rep[key] = r
    y = np.asarray(labels["p_up"], float)
    p = np.asarray(hd["p_up"], float)
    sel = ev[:, None] & np.isfinite(y) & np.isfinite(p)
    cs = cal[:, None] & np.isfinite(y)
    base = float(np.mean(y[cs])) if cs.any() else float("nan")
    if sel.any():
        yy, pp = y[sel], p[sel]
        bins = np.minimum((pp * 10).astype(int), 9)
        ece = float(sum(abs(pp[bins == i].mean() - yy[bins == i].mean()) * np.mean(bins == i) for i in range(10) if (bins == i).any()))
        rep["p_up"] = {"n": int(sel.sum()), "brier": float(np.mean((pp - yy) ** 2)), "naive_brier": float(np.mean((base - yy) ** 2)) if np.isfinite(base) else float("nan"),
                       "accuracy": float(np.mean((pp >= 0.5) == (yy >= 0.5))), "ece": ece}
        rep["p_up"]["brier_over_naive"] = rep["p_up"]["brier"] / rep["p_up"]["naive_brier"] if rep["p_up"]["naive_brier"] > 0 else float("nan")
    else:
        rep["p_up"] = {"n": 0}
    return rep


# ----------------------------------------------------------------------------- the orders (the PZ menu)
def exit_offsets(env, row_bar) -> np.ndarray:
    """(n_rows, 2) price: what the TP distance gives up so the order is fillable (V11_RULES['tp']): limit_through for
    both sides, plus the hour's spread on the short side when the engine triggers shorts on the ask; NaN without a bar."""
    rb = np.asarray(row_bar, np.int64)
    out = np.full((len(rb), 2), np.nan)
    prof = env.cfg.profile
    through = float(getattr(prof, "limit_through", 0.0))
    ask = bool(getattr(env.cfg.variant, "short_trigger_on_ask", False)) if env.cfg.variant is not None else False
    for r in np.flatnonzero((rb >= 0) & (rb < len(env.v))):
        b = int(rb[r])
        sp = float(prof.spread_at(int(env.v[b]), int(env.mo[int(env.seg_of[b])]))) if ask else 0.0
        out[r] = (through, through + sp)
    return out


def orders(bars, sigma, row_bar, hd: dict, genes: dict, tick: float | None = None, horizons=HORIZONS, exit_offset=None) -> dict:
    """Per row (decision bar b = row_bar[r]), side, slot: the PZ order's TP / SL prices and their distances in sigma of
    the decision; NaN when the inputs are censored or a distance is <= 0. ``mask`` = a finite order. v1.1: the TP
    distance gives up ``exit_offset`` (n_rows, 2) (exit_offsets; required), rounded inward with the TP."""
    g = validate_genes(genes)
    tick = ZZ.LEVEL_TICK if tick is None else float(tick)
    close, sg = np.asarray(bars["CLOSE"], float), np.asarray(sigma, float)
    rb = np.asarray(row_bar, np.int64)
    n = len(rb)
    out = {k: np.full((n, 2, len(HORIZONS)), np.nan) for k in ("tp", "sl", "tp_s", "sl_s")}
    v11 = DESIGN_VERSION != "1.0"
    if v11 and exit_offset is None:
        raise PZError("PZ_V11_NEEDS_EXIT_OFFSET")
    xo = None if exit_offset is None else np.asarray(exit_offset, float)
    lg = {"a_sl": g["a_sl"], "b_sl": 0.0, "buffer_pips": g["buffer_pips"], "a_tp": g["a_tp"], "b_tp": 1.0 if v11 else 0.0}   # v1.1: d_tp = a_tp fav - 1 x offset
    fav50, adv75 = hd["fav"][0.5], hd["adv"][0.75]
    hset = {int(h) for h in horizons}
    for r in np.flatnonzero((rb >= 0) & (rb < len(close))):
        b = int(rb[r])
        quote, s24 = float(close[b]), float(sg[b])
        if not (np.isfinite(quote) and np.isfinite(s24) and s24 > 0):
            continue
        for j, h in enumerate(HORIZONS):
            if h not in hset:
                continue
            for si, d in ((0, 1), (1, -1)):
                f, a = float(fav50[b, j, si]), float(adv75[b, j, si])
                if not (np.isfinite(f) and np.isfinite(a)) or f <= 0 or a < 0:
                    continue
                off = float(xo[r, si]) if (v11 and xo is not None) else 0.0
                if not np.isfinite(off):
                    continue
                try:
                    lv = HL.levels(d, quote, f, a, off, lg, tick)
                except HL.LevelInputError:
                    continue
                if lv["raw_tp_distance"] <= 0 or lv["raw_sl_distance"] <= 0:
                    continue
                out["tp"][r, si, j], out["sl"][r, si, j] = lv["tp"], lv["sl"]
                out["tp_s"][r, si, j] = abs(lv["tp"] - quote) / s24
                out["sl_s"][r, si, j] = abs(quote - lv["sl"]) / s24
    out["mask"] = np.isfinite(out["tp"]) & np.isfinite(out["sl"])
    out["stamp"] = {"strategy": "PZ", "design": DESIGN_VERSION, "genes": {k: g[k] for k in ("a_sl", "a_tp", "buffer_pips")}, "horizons": sorted(hset)}
    return out


def truth_orders_fn(genes: dict):
    """The hook strategy_output_simulation.build_truth_menu calls under ``--strategy pz``: the PZ menu at PERFECT labels
    (the exact replay of the orders the strategy places with ideal outputs; the full_f4 ruling's principle)."""
    def f(bars, sigma, row_bar, params, swap_refs=False, env=None):  # swap_refs: PZ reads no TRAIN reference (inert); env: the truth's engine (v1.1 offsets)
        lab = expert_labels(bars)
        xo = exit_offsets(env, row_bar) if (env is not None and DESIGN_VERSION != "1.0") else None
        return orders(bars, sigma, row_bar, heads(lab, 1.0, 0, np.zeros(len(lab["p_up"]), bool)), genes, exit_offset=xo)
    return f


def view_entries(bars, sigma, row_bar, cost_row, hd: dict, genes: dict, horizons=HORIZONS, exit_offset=None) -> dict:
    """The per-row table entries PolicyPZ reads."""
    rb = np.asarray(row_bar, np.int64)
    n = len(rb)
    close, sg = np.asarray(bars["CLOSE"], float), np.asarray(sigma, float)
    ok = (rb >= 0) & (rb < len(close))
    b = np.where(ok, rb, 0)
    k1 = np.clip(b + 1, 0, len(close) - 1)
    rows = lambda x: np.where(ok.reshape((-1,) + (1,) * (np.ndim(x) - 1)), np.asarray(x)[b], np.nan)  # noqa: E731
    ords = orders(bars, sigma, rb, hd, genes, horizons=horizons, exit_offset=exit_offset)
    cr = np.asarray(cost_row, float)
    return {"pz_genes": validate_genes(genes), "pz_horizons": tuple(int(h) for h in horizons), "pz_p_up": rows(hd["p_up"]),
            "pz_fav25": rows(hd["fav"][0.25]), "pz_fav50": rows(hd["fav"][0.5]), "pz_fav75": rows(hd["fav"][0.75]), "pz_tau50": rows(hd["tau"][0.5]),
            "pz_tp": ords["tp"], "pz_sl": ords["sl"], "pz_tp_s": ords["tp_s"], "pz_sl_s": ords["sl_s"], "pz_stamp": ords["stamp"],
            "pz_quote": np.where(ok, close[b], np.nan), "pz_cost": np.where(ok, cr * sg[k1], np.nan)}


def strategy_object(genes: dict) -> "C3.StrategyC3":
    """The C3 strategy record the shared rollout wrapper reads: early close on, no reversal (DECLARED), no NOP veto."""
    g = validate_genes(genes)
    return C3.StrategyC3(float(g["theta"]), 0.0, 0.0, 0.5, 0.5, False, None, True)


# ----------------------------------------------------------------------------- the policy
class PolicyPZ(ZZ.PolicyC3Z):
    """§3 opening and §4 close on the PZ heads (table entries pz_*); the C3Z decision / close logs in the same shape,
    so every witness of the behaviour matrix reads PZ exactly as it reads C3Z."""

    def __init__(self, table, strategy, cells=C3.CELLS, e_time=(0.0, 0.0), cost: float = 0.0, observe_bars: int = 0):
        super().__init__(table, strategy, cells, e_time, cost, observe_bars)
        self.g = validate_genes(table["pz_genes"])
        self.hset = set(table.get("pz_horizons") or HORIZONS)
        self.open_h = {}
        self.persist = {}

    def _menu(self) -> bool:
        return M.is_menu(self.cells)

    def reverse_ok(self, r) -> bool:
        return True

    def _env(self):
        return (getattr(self, "_ctx", None) or (None, None))

    def _scores(self, r) -> np.ndarray:
        """(2, 4) score of every executable (side, slot) at row r; -inf where not a candidate."""
        g, t = self.g, self.t
        sc = np.full((2, len(HORIZONS)), -np.inf)
        bar, env = self._env()
        quote, cost = float(t["pz_quote"][r]), float(t["pz_cost"][r])
        if not (np.isfinite(quote) and np.isfinite(cost)):
            return sc
        for si, d in ((0, 1), (1, -1)):
            for j, h in enumerate(HORIZONS):
                if h not in self.hset:
                    continue
                tp, sl = float(t["pz_tp"][r, si, j]), float(t["pz_sl"][r, si, j])
                pu = float(t["pz_p_up"][r, j])
                if not (np.isfinite(tp) and np.isfinite(sl) and np.isfinite(pu)):
                    continue
                if env is not None and bar is not None and env._levels_absolute(int(bar), d, tp, sl) is None:
                    continue
                p = pu if d > 0 else 1.0 - pu
                tpd, sld = abs(tp - quote), abs(quote - sl)
                if sld <= 0:
                    continue
                width = float(t["pz_fav75"][r, j, si]) - float(t["pz_fav25"][r, j, si])
                width = width if np.isfinite(width) else 0.0
                G = p * tpd - (1.0 - p) * sld - cost - g["kappa"] * width
                if g["alpha"] > 0:
                    tau = float(t["pz_tau50"][r, j])
                    if not np.isfinite(tau):
                        continue
                    sc[si, j] = (G / sld) / (tau + g["tau0"]) ** g["alpha"]
                else:
                    sc[si, j] = G / sld
        return sc

    def open_decision(self, r, with_score: bool = False):
        sc = self._scores(r)
        best = [M.argmax_slot(sc[si], np.isfinite(sc[si]), 0.0) for si in (0, 1)]
        val = [float(sc[si, best[si]]) if best[si] >= 0 else float("nan") for si in (0, 1)]
        cleared = [bool(best[si] >= 0 and val[si] > self.g["theta"]) for si in (0, 1)]
        chosen = -1
        if cleared[0] and cleared[1]:
            chosen = -1 if abs(val[0] - val[1]) <= C3.TIE_EPS else (0 if val[0] > val[1] else 1)
        elif cleared[0] or cleared[1]:
            chosen = 0 if cleared[0] else 1
        self.log.append((int(r), val[0], val[1], cleared[0], cleared[1], chosen, False, best[0], best[1]))
        self.attr.append({"row": int(r), "blocked": "" if chosen >= 0 or not (cleared[0] or cleared[1]) else "side_tie", "gate_l": "" if cleared[0] else "theta",
                          "gate_s": "" if cleared[1] else "theta", "chosen": chosen, "winner_before_veto": chosen, "zz_dir": 0.0, "fresh": float("nan")})
        if chosen < 0:
            return None
        ci = best[chosen]
        d = 1 if chosen == 0 else -1
        tp, sl = float(self.t["pz_tp"][r, chosen, ci]), float(self.t["pz_sl"][r, chosen, ci])
        self.attr[-1]["h"] = HORIZONS[ci]
        self.open_h[(d, round(tp, 7), round(sl, 7))] = ci
        out = (E.LONG if d > 0 else E.SHORT, tp, sl)
        return out + (val[chosen],) if with_score else out

    def menu_envelope(self, r, si):
        """The PZ order of side si at row r whatever the opener says (forced entry, episodes): the best-scored candidate."""
        sc = self._scores(r)[si]
        ci = M.argmax_slot(sc, np.isfinite(sc), 0.0)
        if ci < 0:
            return None
        tp, sl = float(self.t["pz_tp"][r, si, ci]), float(self.t["pz_sl"][r, si, ci])
        self.open_h[(1 if si == 0 else -1, round(tp, 7), round(sl, 7))] = ci
        return ci, tp, sl

    def close_now(self, r, si, bar, pos) -> bool:
        self.positions.setdefault((int(pos.entry_bar), int(si)), dataclasses.replace(pos))
        if not self.s.early_close:
            return False
        t, g = self.t, self.g
        d = int(pos.direction)
        jp = self.open_h.get((d, round(float(pos.tp), 7), round(float(pos.sl), 7)), HORIZONS.index(6))
        jf = max(jp - 1, 0)
        P, cost = float(t["pz_quote"][r]), float(t["pz_cost"][r])
        pu, pf = float(t["pz_p_up"][r, jp]), float(t["pz_p_up"][r, jf])
        p1 = pu if d > 0 else 1.0 - pu
        pf1 = pf if d > 0 else 1.0 - pf
        up, down = d * (float(pos.tp) - P), d * (P - float(pos.sl))
        cost_close = COST_CLOSE_SHARE * cost if np.isfinite(cost) else 0.0
        band = EPS_CLOSE * float(pos.sigma) if np.isfinite(pos.sigma) else 0.0
        hold = p1 * up - (1.0 - p1) * down if (np.isfinite(p1) and np.isfinite(P)) else float("nan")
        cond = bool(np.isfinite(hold) and hold <= band - cost_close)
        key = (int(pos.entry_bar), int(si))
        self.persist[key] = self.persist.get(key, 0) + 1 if cond else 0
        urgent = bool(np.isfinite(pf1) and pf1 < g["p_urg"])
        if DESIGN_VERSION != "1.0" and urgent:                        # v1.1: only when the fine reversal is predicted to reach the SL
            fo = t.get("pz_fav50")
            reach = float(fo[r, jf, 1 if d > 0 else 0]) if fo is not None else float("nan")
            urgent = bool(np.isfinite(reach) and np.isfinite(down) and reach >= down)
        dec = bool(urgent or (cond and self.persist[key] >= int(g["k_persist"])))
        cl = t.get("closer")
        rem, psl = cl.value(r, bar, pos) if cl is not None else (float("nan"), float("nan"))
        a = (-cost_close - hold) / float(pos.sigma) if (np.isfinite(hold) and pos.sigma > 0) else float("nan")
        self.closes.append((int(r), int(bar), int(si), float(rem), float(psl), dec, cond, urgent, False, float(a), "pz"))
        return dec


class PolicyTrivial(PolicyPZ):
    """The trivial base (§5): always open the side p_up[6 h] names (p = 0.5: NOP) with the h = 6 order of the C3Z menu
    (Musashi's median TRAIN profiles, the contract baseline genes; table entries triv_tp / triv_sl); exits only by SL /
    TP / forced flat."""

    def open_decision(self, r, with_score: bool = False):
        j = HORIZONS.index(6)
        pu = float(self.t["pz_p_up"][r, j])
        si = -1 if not np.isfinite(pu) or pu == 0.5 else (0 if pu > 0.5 else 1)
        bar, env = self._env()
        ok = si >= 0
        if ok:
            tp, sl = float(self.t["triv_tp"][r, si]), float(self.t["triv_sl"][r, si])
            ok = np.isfinite(tp) and np.isfinite(sl) and not (env is not None and bar is not None and env._levels_absolute(int(bar), 1 if si == 0 else -1, tp, sl) is None)
        self.log.append((int(r), float("nan"), float("nan"), ok and si == 0, ok and si == 1, si if ok else -1, False, j if si == 0 else -1, j if si == 1 else -1))
        self.attr.append({"row": int(r), "blocked": "", "chosen": si if ok else -1, "winner_before_veto": si if ok else -1, "zz_dir": 0.0, "fresh": float("nan"), **({"h": 6} if ok else {})})
        if not ok:
            return None
        out = (E.LONG if si == 0 else E.SHORT, tp, sl)
        return out + (0.0,) if with_score else out

    def close_now(self, r, si, bar, pos) -> bool:
        self.positions.setdefault((int(pos.entry_bar), int(si)), dataclasses.replace(pos))
        return False


def trivial_orders(bars, sigma, row_bar, params: dict | None = None) -> dict:
    """The trivial base's orders: the C3Z menu's h = 6 slot at the contract baseline genes and the median references."""
    p = dict(HL.GENE_BASELINE)
    p.pop("rho", None)
    p["ref_p75"] = 0.0
    if params:
        p.update({k: float(params[k]) for k in M.LEVEL_GENES if k in params})
    o = M.orders(np.asarray(bars["CLOSE"], float), sigma, row_bar, p)
    j = HORIZONS.index(6)
    return {"triv_tp": o["tp"][:, :, j], "triv_sl": o["sl"][:, :, j]}


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

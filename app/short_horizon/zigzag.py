"""MetaTrader ZigZag (Depth, Deviation, Backstep) replicated from the public MetaQuotes ZigZag.mq5, the
hindsight expert built on it, its causal-confirmed counterpart, and the pivot-reversal orders.

NON-CAUSAL by construction. The indicator REPAINTS: a bar becomes a candidate extreme only when it is the
Depth-window extreme, a candidate within Backstep bars is erased by a better later one, and the last
extreme of the current leg is replaced until the opposite extreme appears after a Deviation-sized
reversal. A pivot is therefore known only after a Deviation-sized reversal and Depth/Backstep bars.
It must never be an input of a causal model; it defines labels and the HINDSIGHT_NON_CAUSAL_EXPERT.

Replication note: the two-pass structure (candidate maps with the Lowest/Highest Depth search, the
Deviation test on the current bar, the Backstep cleanup of earlier candidates; then the alternating
selection with replacement of the last extreme) follows the MetaQuotes source as published in the
MQL5 code base; it was not diffed bit-for-bit against a specific file version, and point = 1 pip here.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .blocks import AMBIGUOUS, Blocks, OK
from .calendar import HOUR
from .execution import BlockOrder

PEAK, VALLEY = 1, -1
PIP = 1e-4
EXPERT_LABEL = "HINDSIGHT_NON_CAUSAL_EXPERT"


@dataclass(frozen=True)
class Params:
    depth: int = 12
    deviation_pips: float = 5.0
    backstep: int = 3


def _window_extreme(x, depth, fn):
    """fn over the window [i - depth + 1, i] (shorter at the start), like Lowest/Highest in MQL."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.empty(0)
    d = min(depth, n)
    pad = np.r_[np.full(d - 1, x[0]), x]           # short windows at the start reduce to the available bars
    win = np.lib.stride_tricks.sliding_window_view(pad, d)
    out = fn(win, axis=1)
    for i in range(min(d - 1, n)):                 # exact short-window values at the start
        out[i] = fn(x[:i + 1])
    return out


def candidate_maps(high, low, p: Params):
    """First pass: LowMap / HighMap buffers (0 where no candidate)."""
    high, low = np.asarray(high, float), np.asarray(low, float)
    n = len(high)
    dev = p.deviation_pips * PIP
    lmin = _window_extreme(low, p.depth, np.min)
    hmax = _window_extreme(high, p.depth, np.max)
    lowmap, highmap = np.zeros(n), np.zeros(n)
    lastlow = lasthigh = 0.0
    for i in range(n):
        val = lmin[i]
        if val == lastlow:
            val = 0.0
        else:
            lastlow = val
            if low[i] - val > dev:
                val = 0.0
            else:
                for back in range(1, p.backstep + 1):
                    if i - back >= 0 and lowmap[i - back] != 0 and lowmap[i - back] > val:
                        lowmap[i - back] = 0.0
        lowmap[i] = val if low[i] == val else 0.0
        val = hmax[i]
        if val == lasthigh:
            val = 0.0
        else:
            lasthigh = val
            if val - high[i] > dev:
                val = 0.0
            else:
                for back in range(1, p.backstep + 1):
                    if i - back >= 0 and highmap[i - back] != 0 and highmap[i - back] < val:
                        highmap[i - back] = 0.0
        highmap[i] = val if high[i] == val else 0.0
    return lowmap, highmap


def hindsight_pivots(high, low, p: Params) -> list:
    """Second pass: alternating pivots [(bar, PEAK|VALLEY, level)] as the final (repainted) ZigZag buffer."""
    lowmap, highmap = candidate_maps(high, low, p)
    return select_pivots(lowmap, highmap)


def select_pivots(lowmap, highmap) -> list:
    """The alternating selection (second pass) on given candidate maps."""
    n = len(lowmap)
    zz = {}
    look = 0          # 0 unknown, PEAK = looking for a peak (last was a valley), VALLEY = looking for a valley
    last_low = last_high = 0.0
    last_low_pos = last_high_pos = -1
    for i in range(n):
        if look == 0:
            if last_low == 0.0 and last_high == 0.0:
                # SYMMETRIC first pivot (2026-10-27): a bar holding BOTH a high and a low candidate does not start the
                # ZigZag (the legacy rule let the low overwrite the high, so a path and its price mirror started with
                # different first pivots); the first bar with exactly one candidate does
                if highmap[i] != 0 and lowmap[i] == 0:
                    last_high, last_high_pos, look = highmap[i], i, VALLEY
                    zz[i] = (PEAK, last_high)
                elif lowmap[i] != 0 and highmap[i] == 0:
                    last_low, last_low_pos, look = lowmap[i], i, PEAK
                    zz[i] = (VALLEY, last_low)
        elif look == PEAK:
            if lowmap[i] != 0 and lowmap[i] < last_low and highmap[i] == 0:
                zz.pop(last_low_pos, None)
                last_low, last_low_pos = lowmap[i], i
                zz[i] = (VALLEY, last_low)
            if highmap[i] != 0 and lowmap[i] == 0:
                last_high, last_high_pos, look = highmap[i], i, VALLEY
                zz[i] = (PEAK, last_high)
        else:
            if highmap[i] != 0 and highmap[i] > last_high and lowmap[i] == 0:
                zz.pop(last_high_pos, None)
                last_high, last_high_pos = highmap[i], i
                zz[i] = (PEAK, last_high)
            if lowmap[i] != 0 and highmap[i] == 0:
                last_low, last_low_pos, look = lowmap[i], i, PEAK
                zz[i] = (VALLEY, last_low)
    return [(i, kind, lvl) for i, (kind, lvl) in sorted(zz.items())]


def select_pivots_legacy(lowmap, highmap) -> list:
    """The pre-2026-10-27 selection (asymmetric first pivot), kept only to measure the change on real data."""
    lm, hm = np.asarray(lowmap, float).copy(), np.asarray(highmap, float).copy()
    k = next((i for i in range(len(lm)) if lm[i] != 0 or hm[i] != 0), None)
    if k is not None and lm[k] != 0 and hm[k] != 0:
        hm[k] = 0.0                                                # legacy: the low overwrote the high on a bar with both
    return select_pivots(lm, hm)


def hindsight_pivots_legacy(high, low, p: Params) -> list:
    lowmap, highmap = candidate_maps(high, low, p)
    return select_pivots_legacy(lowmap, highmap)


def confirmation_bars(high, low, pivots, p: Params, max_lag: int = 400) -> list:
    """For each final pivot, the first bar t at which the ZigZag computed on bars[:t+1] (replayed from the
    previous final pivot, which fixes the state) contains the pivot and the opposite extreme that locks
    it, with that opposite candidate beyond the Backstep reach. -1 when it never locks inside max_lag.
    """
    high, low = np.asarray(high, float), np.asarray(low, float)
    n = len(high)
    out = []
    for j, (i, kind, lvl) in enumerate(pivots):
        start = pivots[j - 1][0] if j > 0 else 0
        nxt = pivots[j + 1][0] if j + 1 < len(pivots) else None
        conf = -1
        if nxt is not None:
            for t in range(nxt + p.backstep, min(n, i + max_lag + 1)):
                rep = hindsight_pivots(high[start:t + 1], low[start:t + 1], p)
                rep = {(start + b, k) for b, k, _ in rep}
                if (i, kind) in rep and any(b > i and k == -kind for b, k in rep):
                    conf = t
                    break
        out.append(conf)
    return out


def confirmation_bars_fast(high, low, pivots, p: Params) -> list:
    """Knowable confirmation bar of each final pivot (v2 definition).

    A pivot is locked once the selection switches direction after it, which happens at the first later
    bar holding an opposite candidate and no same-side candidate; that candidate can still be erased by
    the next ``backstep`` bars, so the lock is knowable at j* + backstep. Equal, by test, to replaying
    the rule on every prefix with not-yet-final candidates masked. -1 when the lock is beyond the data.
    The v1 function ``confirmation_bars`` waited for the NEXT FINAL pivot (+ backstep) instead, which
    overstates the lag whenever the first opposite candidate is later replaced by a better one.
    """
    lowmap, highmap = candidate_maps(high, low, p)
    n = len(lowmap)
    hi_only = np.flatnonzero((highmap != 0) & (lowmap == 0))
    lo_only = np.flatnonzero((lowmap != 0) & (highmap == 0))
    out = []
    for i, kind, _ in pivots:
        cand = hi_only if kind == VALLEY else lo_only
        j = int(np.searchsorted(cand, i, side="right"))
        if j >= len(cand):
            out.append(-1)
            continue
        c = int(cand[j]) + p.backstep
        out.append(c if c < n else -1)
    return out


def lag_distribution(pivots, conf) -> dict:
    lags = np.array([c - i for (i, _, _), c in zip(pivots, conf) if c >= 0], float)
    if not len(lags):
        return {"n": 0, "mean": math.nan, "median": math.nan, "p90": math.nan, "max": math.nan}
    return {"n": int(len(lags)), "mean": float(lags.mean()), "median": float(np.median(lags)),
            "p90": float(np.quantile(lags, 0.9)), "max": float(lags.max())}


# ------------------------------------------------------------------ signals, blocks, orders
def signals_from_pivots(pivots, delay: int = 0, conf=None) -> list:
    """[(signal bar, direction)]: a valley -> long (+1), a peak -> short (-1). The signal bar is the pivot
    bar + delay (hindsight), or the confirmation bar when ``conf`` is given (causal-confirmed)."""
    out = []
    for j, (i, kind, _) in enumerate(pivots):
        b = (conf[j] if conf is not None else i + delay)
        if b is None or b < 0:
            continue
        out.append((int(b), 1 if kind == VALLEY else -1))
    out.sort()
    return out


def jitter_signals(signals, rng, max_bars: int) -> list:
    if max_bars <= 0:
        return list(signals)
    return sorted((int(b + rng.integers(-max_bars, max_bars + 1)), d) for b, d in signals)


def causal_sigma(bars: pd.DataFrame, window: int = 24, days: int = 20) -> np.ndarray:
    """Expected 24-h volatility at each bar: mean over the last ``days`` completed windows of the realized
    vol (sqrt of summed squared hourly close changes over ``window`` bars), all strictly before the bar."""
    c = bars.CLOSE.to_numpy(float)
    r2 = np.r_[0.0, np.diff(c) ** 2]
    rv = pd.Series(r2).rolling(window).sum().pow(0.5)
    return rv.rolling(days * window, min_periods=5 * window).mean().shift(1).to_numpy()


def causal_range_sigma(bars: pd.DataFrame, window: int = 24, days: int = 20) -> np.ndarray:
    """Mean of the last ``days`` 24-bar ranges strictly before the bar (order-rule effectiveness scale)."""
    h, l = bars.HIGH.rolling(window).max(), bars.LOW.rolling(window).min()
    return (h - l).rolling(days * window, min_periods=5 * window).mean().shift(1).to_numpy()


def signal_blocks(bars: pd.DataFrame, cal, signals, executable: bool = False) -> Blocks:
    """One block per signal: entry at the next bar open, end at the next signal's entry bar or the forced
    flat of the session, whichever comes first. Reference = close of the signal bar.

    ``executable=False`` (default, pinned v1 path, byte-identical): the block starts at b + 1 and the
    realistic profile's entry_delay_bars = 1 then fills at the open of b + 2; entries in the first hours
    after a session reopens are not refused. ``executable=True`` (v2): the decision is taken after the
    completed bar b and must be simulated with ``zigzag_opt.executable_profile()`` (entry_delay_bars = 0),
    so it fills at the open of b + 1; a block starting before the session's operational start (the
    calendar's entry delay after a reopen) is refused with status BEFORE_OPERATIONAL_START.
    """
    if executable:
        return _signal_blocks_executable(bars, cal, signals)
    v = bars.index
    v_ns = v.as_unit("ns").asi8
    seg_of = cal.assign(v)
    close, high, low = (bars[k].to_numpy(float) for k in ("CLOSE", "HIGH", "LOW"))
    n = len(v)
    start, end, seg, blo, bhi, rbar, ref, H, L, ft, status = ([] for _ in range(11))
    for j, (b, d) in enumerate(signals):
        lo = b + 1
        nxt = signals[j + 1][0] + 1 if j + 1 < len(signals) else n
        st = OK
        if lo < 1 or lo >= n or b < 0:
            st, si, s, e, hi = "NO_REFERENCE", 0, v[0], v[0] + HOUR, 1
        else:
            si = int(seg_of[lo])
            s = v[lo]
            e = min(v[min(nxt, n - 1)] if nxt < n else v[n - 1] + HOUR, cal.segments[si].force_flat)
            hi = int(np.searchsorted(v_ns, (e - HOUR).value, side="right"))
            if hi <= lo or s >= cal.segments[si].entry_cutoff or seg_of[b] != si:
                st = "INSUFFICIENT_BARS"
        start.append(s), end.append(e), seg.append(si), blo.append(lo), bhi.append(max(hi, lo + 1))
        rbar.append(b), ref.append(close[b] if st == OK else np.nan)
        H.append(high[lo:hi].max() if st == OK else np.nan), L.append(low[lo:hi].min() if st == OK else np.nan)
        ft.append(AMBIGUOUS), status.append(st)
    return Blocks(0, start, end, np.array(seg, int), np.array(blo, int), np.array(bhi, int), np.array(rbar, int),
                  np.array(ref, float), np.array(H, float), np.array(L, float), np.array(ft, int),
                  np.array(status, object))


def _signal_blocks_executable(bars: pd.DataFrame, cal, signals) -> Blocks:
    v = bars.index
    v_ns = v.as_unit("ns").asi8
    H_NS = HOUR.value
    seg_of = cal.assign(v)
    close, high, low = (bars[k].to_numpy(float) for k in ("CLOSE", "HIGH", "LOW"))
    op = np.array([s.operational_start.value for s in cal.segments], np.int64)
    ff = np.array([s.force_flat.value for s in cal.segments], np.int64)
    ec = np.array([s.entry_cutoff.value for s in cal.segments], np.int64)
    n = len(v)
    m = len(signals)
    start_ns, end_ns = np.zeros(m, np.int64), np.zeros(m, np.int64)
    seg, blo, bhi, rbar = (np.zeros(m, int) for _ in range(4))
    ref, H, L = (np.full(m, np.nan) for _ in range(3))
    status = np.empty(m, object)
    for j, (b, d) in enumerate(signals):
        lo = b + 1
        nxt = signals[j + 1][0] + 1 if j + 1 < m else n
        rbar[j] = b
        if lo < 1 or lo >= n or b < 0:
            status[j], seg[j], start_ns[j], end_ns[j], blo[j], bhi[j] = "NO_REFERENCE", 0, v_ns[0], v_ns[0] + H_NS, lo, lo + 1
            continue
        si = int(seg_of[lo])
        s_ns = v_ns[lo]
        e_ns = min(v_ns[nxt] if nxt < n else v_ns[n - 1] + H_NS, ff[si])
        hi = int(np.searchsorted(v_ns, e_ns - H_NS, side="right"))
        st = "OK"
        if s_ns < op[si]:
            st = "BEFORE_OPERATIONAL_START"
        elif hi <= lo or s_ns >= ec[si] or seg_of[b] != si or e_ns <= s_ns:
            st = "INSUFFICIENT_BARS"
        status[j], seg[j], start_ns[j], end_ns[j], blo[j], bhi[j] = st, si, s_ns, e_ns, lo, max(hi, lo + 1)
        if st == "OK":
            ref[j], H[j], L[j] = close[b], high[lo:hi].max(), low[lo:hi].min()
    start = list(pd.to_datetime(start_ns, utc=True))
    end = list(pd.to_datetime(end_ns, utc=True))
    return Blocks(0, start, end, seg, blo, bhi, rbar, ref, H, L, np.full(m, AMBIGUOUS, int), status)


def reversal_orders(blocks: Blocks, signals, pivots, sigma, mode: str = "hindsight", k_tp: float = 1.0,
                    m_sl: float = 2.0, inset: float = PIP, size: float = 1.0, sigma_rule=None) -> list:
    """Market order per signal block. ``hindsight``: TP = next pivot level -/+ inset (NON-CAUSAL, labelled);
    ``causal``: TP = entry +/- k_tp * sigma. SL = entry -/+ m_sl * sigma in both."""
    piv_bars = np.array([i for i, _, _ in pivots])
    piv_lvl = np.array([lvl for _, _, lvl in pivots])
    out = []
    for k, (b, d) in enumerate(signals):
        if blocks.status[k] != OK or not np.isfinite(sigma[blocks.bar_lo[k]]) or sigma[blocks.bar_lo[k]] <= 0 \
                or (sigma_rule is not None and not np.isfinite(sigma_rule[blocks.bar_lo[k]])):
            out.append(None)
            continue
        r = float(blocks.ref[k])
        s = float(sigma[blocks.bar_lo[k]])
        if mode == "hindsight":
            j = int(np.searchsorted(piv_bars, b, side="right"))
            if j >= len(piv_bars):
                out.append(None)
                continue
            tp = float(piv_lvl[j]) - d * inset
        else:
            tp = r + d * k_tp * s
        sl = r - d * m_sl * s
        if not ((tp > r > sl) if d > 0 else (tp < r < sl)):
            out.append(None)
            continue
        out.append(BlockOrder(d, "MARKET", None, tp, sl, None, size))
    return out


def write_demonstrations(outdir, trades, params: Params, data_sha256: str, costs: dict, extra=None) -> dict:
    """Sealed expert-trade CSV (t_open, side, units, entry, SL, TP, t_close, reason) plus its manifest."""
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [{"t_open": pd.Timestamp(t["entry_time"]).isoformat(), "side": int(t["direction"]), "units": int(t["units"]),
             "entry": float(t["entry_price"]), "sl": float(t["stop_loss"]), "tp": float(t["take_profit"]),
             "t_close": pd.Timestamp(t["exit_time"]).isoformat(), "reason": t["exit_reason"]} for t in trades]
    path = outdir / "expert_trades.csv"
    pd.DataFrame(rows, columns=["t_open", "side", "units", "entry", "sl", "tp", "t_close", "reason"]) \
        .to_csv(path, index=False, float_format="%.6f")
    man = {"label": EXPERT_LABEL, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows),
           "params": asdict(params), "data_sha256": data_sha256, "costs": costs,
           "warning": "Trades use the final repainted ZigZag pivots and the NEXT pivot level as TP: not reproducible "
                      "causally; demonstrations only."}
    if extra:
        man.update(extra)
    (outdir / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str) + "\n")
    return man

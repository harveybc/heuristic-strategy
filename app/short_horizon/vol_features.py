"""Causal per-block volatility features (DEVELOPMENT). Every feature of a block uses only bars whose
label is before the block start (indices < bar_lo) and blocks completed before it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import predictors as PR

WINDOWS = (6, 24, 120, 480)        # last n observed bars (~6 h, 1 day, 1 week, 1 month of trading hours)


def parkinson(high, low) -> float:
    high, low = np.asarray(high, float), np.asarray(low, float)
    return float(np.sqrt(np.mean(np.log(high / low) ** 2) / (4 * np.log(2))))


def efficiency_ratio(close) -> float:
    c = np.asarray(close, float)
    path = np.sum(np.abs(np.diff(c)))
    return float(abs(c[-1] - c[0]) / path) if path > 0 else 0.0


def block_features(bars: pd.DataFrame, blocks, t: pd.DataFrame, B: int) -> pd.DataFrame:
    har = PR.har_features(t, B)[:, 1:]                      # drop the intercept
    cols = {f"har_{i}": har[:, i] for i in range(har.shape[1])}
    H, L, C = (bars[c].to_numpy(float) for c in ("HIGH", "LOW", "CLOSE"))
    out = {f"pk_{n}": np.full(len(t), np.nan) for n in WINDOWS}
    out.update({f"absret_{n}": np.full(len(t), np.nan) for n in (24, 120)})
    out["er_24"] = np.full(len(t), np.nan)
    for i, k in enumerate(t["k"].to_numpy()):
        lo = int(blocks.bar_lo[k])
        for n in WINDOWS:
            if lo - n >= 0:
                out[f"pk_{n}"][i] = parkinson(H[lo - n:lo], L[lo - n:lo])
        for n in (24, 120):
            if lo - 1 - n >= 0 and np.isfinite(out[f"pk_{n}"][i]) and out[f"pk_{n}"][i] > 0:
                out[f"absret_{n}"][i] = abs(np.log(C[lo - 1] / C[lo - 1 - n])) / (out[f"pk_{n}"][i] * np.sqrt(n))
        if lo - 25 >= 0:
            out["er_24"][i] = efficiency_ratio(C[lo - 25:lo])
    cols.update(out)
    cols["prev_U"] = t["U"].shift(1).to_numpy()
    cols["prev_D"] = t["D"].shift(1).to_numpy()
    return pd.DataFrame(cols, index=t.index)

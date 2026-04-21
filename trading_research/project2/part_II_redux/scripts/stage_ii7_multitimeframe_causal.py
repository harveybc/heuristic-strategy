#!/usr/bin/env python3
"""Stage II-7.2: Multi-timeframe causal matrix (14 PCMCI+ runs)."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_BINANCE_DIR = os.path.join(BASE_DIR, "data", "raw", "binance")
RAW_CM_DIR = os.path.join(BASE_DIR, "data", "raw", "coinmetrics")
RAW_BC_DIR = os.path.join(BASE_DIR, "data", "raw", "blockchain_com")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
DELIVERABLES_DIR = os.path.join(BASE_DIR, "deliverables")

IS_START = "2019-01-01"
IS_END = "2019-12-31"

RUN_CONFIGS: Dict[int, Dict[str, str]] = {
    1: {"asset": "btc", "timeframe": "5m", "feature_set": "technical"},
    2: {"asset": "btc", "timeframe": "15m", "feature_set": "technical"},
    3: {"asset": "btc", "timeframe": "1h", "feature_set": "technical"},
    4: {"asset": "btc", "timeframe": "4h", "feature_set": "technical+funding"},
    5: {"asset": "btc", "timeframe": "4h", "feature_set": "technical+onchain"},
    6: {"asset": "btc", "timeframe": "4h", "feature_set": "technical+funding+onchain"},
    7: {"asset": "btc", "timeframe": "1h", "feature_set": "technical+funding"},
    8: {"asset": "btc", "timeframe": "15m", "feature_set": "technical+funding"},
    9: {"asset": "eth", "timeframe": "5m", "feature_set": "technical"},
    10: {"asset": "eth", "timeframe": "15m", "feature_set": "technical"},
    11: {"asset": "eth", "timeframe": "1h", "feature_set": "technical"},
    12: {"asset": "eth", "timeframe": "4h", "feature_set": "technical"},
    13: {"asset": "eth", "timeframe": "4h", "feature_set": "technical+funding"},
    14: {"asset": "eth", "timeframe": "4h", "feature_set": "technical+onchain"},
}

TECH_COLS = [
    "adx",
    "di_spread",
    "atr_pct",
    "atr_ratio",
    "bb_width_pct",
    "bb_position",
    "rsi",
    "roc_12",
    "price_vs_ema50",
    "ema_alignment",
    "stoch_k",
    "macd_hist",
]


@dataclass
class RunResult:
    run_id: int
    label: str
    classification: str
    n_samples: int
    n_lagged_links: int
    strongest_lagged_link: dict | None
    lagged_links: list[dict]
    error: str | None = None


def _tf_to_pandas(tf: str) -> str:
    return {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}[tf]


def _read_ohlcv(asset: str, timeframe: str) -> pd.DataFrame:
    if asset == "btc" and timeframe == "4h":
        path = os.path.join(PROC_DIR, "btcusd_4h.csv")
        df = pd.read_csv(path)
        dt_col = "DateTime" if "DateTime" in df.columns else df.columns[0]
        df["DateTime"] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        col_map = {c.lower(): c for c in df.columns}
        out = pd.DataFrame(
            {
                "DateTime": df["DateTime"],
                "Open": pd.to_numeric(df[col_map.get("open", "Open")], errors="coerce"),
                "High": pd.to_numeric(df[col_map.get("high", "High")], errors="coerce"),
                "Low": pd.to_numeric(df[col_map.get("low", "Low")], errors="coerce"),
                "Close": pd.to_numeric(df[col_map.get("close", "Close")], errors="coerce"),
                "Volume": pd.to_numeric(df[col_map.get("volume", "Volume")], errors="coerce"),
            }
        )
        return out.dropna(subset=["DateTime", "Open", "High", "Low", "Close"]).sort_values("DateTime")

    symbol = f"{asset}usdt"
    path = os.path.join(RAW_BINANCE_DIR, f"{symbol}_{timeframe}_2019_2025.parquet")
    df = pd.read_parquet(path)
    dt_col = "DateTime" if "DateTime" in df.columns else "open_time"
    if dt_col == "open_time":
        dt = pd.to_datetime(df[dt_col], unit="ms", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "DateTime": dt,
            "Open": pd.to_numeric(df["Open"], errors="coerce"),
            "High": pd.to_numeric(df["High"], errors="coerce"),
            "Low": pd.to_numeric(df["Low"], errors="coerce"),
            "Close": pd.to_numeric(df["Close"], errors="coerce"),
            "Volume": pd.to_numeric(df.get("Volume", df.get("volume")), errors="coerce"),
        }
    )
    return out.dropna(subset=["DateTime", "Open", "High", "Low", "Close"]).sort_values("DateTime")


def _compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()

    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-12))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-12))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12) * 100
    adx = dx.rolling(14).mean()
    di_spread = plus_di - minus_di

    atr_pct = atr14 / (close + 1e-12) * 100
    atr_ratio = atr14 / (atr14.rolling(120).mean() + 1e-12)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width_pct = (bb_upper - bb_lower) / (sma20 + 1e-12) * 100
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-12)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)

    roc_12 = close.pct_change(12) * 100

    ema50 = close.ewm(span=50).mean()
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()

    price_vs_ema50 = (close - ema50) / (ema50 + 1e-12) * 100
    ema_alignment = (ema12 - ema26) / (atr14 + 1e-12)

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14 + 1e-12)

    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line

    feat = pd.DataFrame(
        {
            "DateTime": df["DateTime"],
            "adx": adx,
            "di_spread": di_spread,
            "atr_pct": atr_pct,
            "atr_ratio": atr_ratio,
            "bb_width_pct": bb_width_pct,
            "bb_position": bb_position,
            "rsi": rsi,
            "roc_12": roc_12,
            "price_vs_ema50": price_vs_ema50,
            "ema_alignment": ema_alignment,
            "stoch_k": stoch_k,
            "macd_hist": macd_hist,
            "target_fwd_6": np.log(close.shift(-6) / close),
        }
    )
    return feat


def _load_funding(asset: str, timeframe: str) -> pd.DataFrame:
    path = os.path.join(RAW_BINANCE_DIR, f"funding_{asset}usdt_2019_2025.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["DateTime", "funding_rate"])
    df = pd.read_csv(path)
    tcol = "funding_time" if "funding_time" in df.columns else "time"
    df["DateTime"] = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.floor(_tf_to_pandas(timeframe))
    fcol = "funding_rate" if "funding_rate" in df.columns else "fundingRate"
    df["funding_rate"] = pd.to_numeric(df[fcol], errors="coerce")
    df = df[["DateTime", "funding_rate"]].dropna().drop_duplicates(subset=["DateTime"]).sort_values("DateTime")
    return df


def _load_onchain(asset: str, timeframe: str) -> pd.DataFrame:
    path = os.path.join(RAW_CM_DIR, f"{asset}_daily_metrics_2019_2025.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["DateTime"])
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.floor("D")
    keep = [c for c in ["AdrActCnt", "TxCnt", "FeeMeanNtv", "CapRealUSD", "NVTAdj90", "NVTAdj", "DiffMean", "HashRate"] if c in df.columns]
    out = df[["DateTime", *keep]].copy()

    if asset == "btc":
        bpath = os.path.join(RAW_BC_DIR, "btc_metrics_2019_2025.csv")
        if os.path.exists(bpath):
            bdf = pd.read_csv(bpath)
            bdf["DateTime"] = pd.to_datetime(bdf["time"], utc=True, errors="coerce").dt.floor("D")
            bkeep = [c for c in ["mempool_size", "confirmed_tx_per_block", "hash_rate"] if c in bdf.columns]
            if bkeep:
                out = out.merge(bdf[["DateTime", *bkeep]], on="DateTime", how="outer")

    out = out.sort_values("DateTime").drop_duplicates(subset=["DateTime"])
    out["DateTime"] = out["DateTime"].dt.floor(_tf_to_pandas(timeframe))
    return out


def _classify(lagged_links: list[dict]) -> str:
    if not lagged_links:
        return "gamma"
    abs_vals = np.array([abs(x["mci"]) for x in lagged_links])
    pvals = np.array([x["p_value"] for x in lagged_links])
    if np.any((abs_vals > 0.10) & (pvals < 0.01)):
        return "alpha"
    if np.any((abs_vals >= 0.05) & (abs_vals <= 0.10) & (pvals < 0.05)):
        return "beta"
    return "gamma"


def _run_pcmci(df: pd.DataFrame, max_samples: int) -> tuple[list[dict], dict | None, int]:
    import tigramite.data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    feature_cols = [c for c in df.columns if c not in {"DateTime", "target_fwd_6"}]
    data_cols = feature_cols + ["target_fwd_6"]
    m = df[data_cols].dropna()
    m = m[np.all(np.isfinite(m.to_numpy()), axis=1)]

    if len(m) > max_samples:
        idx = np.sort(np.random.RandomState(42).choice(len(m), max_samples, replace=False))
        m = m.iloc[idx]

    var_names = data_cols
    arr = m.to_numpy(dtype=float)

    pcmci = PCMCI(
        dataframe=pp.DataFrame(arr, var_names=[v[:18] for v in var_names]),
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    results = pcmci.run_pcmciplus(tau_max=10, pc_alpha=0.01)

    graph = results["graph"]
    pmat = results["p_matrix"]
    vmat = results["val_matrix"]
    target_idx = len(var_names) - 1

    lagged_links: list[dict] = []
    for i, feat in enumerate(feature_cols):
        for tau in range(1, 11):
            link = str(graph[i, target_idx, tau]).strip()
            if ("-->" in link or "o->" in link) and float(pmat[i, target_idx, tau]) < 0.05:
                lagged_links.append(
                    {
                        "feature": feat,
                        "tau": tau,
                        "mci": float(vmat[i, target_idx, tau]),
                        "p_value": float(pmat[i, target_idx, tau]),
                        "link": link,
                    }
                )

    strongest = None
    if lagged_links:
        strongest = sorted(lagged_links, key=lambda x: abs(x["mci"]), reverse=True)[0]

    return lagged_links, strongest, len(m)


def run_one(run_id: int, max_samples: int) -> RunResult:
    cfg = RUN_CONFIGS[run_id]
    asset = cfg["asset"]
    timeframe = cfg["timeframe"]
    feature_set = cfg["feature_set"]
    label = f"run_{run_id}_{asset}_{timeframe}_{feature_set}"

    try:
        ohlcv = _read_ohlcv(asset, timeframe)
        ohlcv = ohlcv[(ohlcv["DateTime"] >= pd.Timestamp(IS_START, tz="UTC")) & (ohlcv["DateTime"] <= pd.Timestamp(IS_END, tz="UTC"))]

        feat = _compute_technical_features(ohlcv)

        if "funding" in feature_set:
            funding = _load_funding(asset, timeframe)
            feat = feat.merge(funding, on="DateTime", how="left")
            feat["funding_rate"] = feat["funding_rate"].ffill()

        if "onchain" in feature_set:
            onchain = _load_onchain(asset, timeframe)
            feat = feat.merge(onchain, on="DateTime", how="left")
            for col in feat.columns:
                if col not in {"DateTime", "target_fwd_6", *TECH_COLS}:
                    feat[col] = feat[col].ffill()

        lagged_links, strongest, n_samples = _run_pcmci(feat, max_samples=max_samples)
        cls = _classify(lagged_links)
        return RunResult(
            run_id=run_id,
            label=label,
            classification=cls,
            n_samples=n_samples,
            n_lagged_links=len(lagged_links),
            strongest_lagged_link=strongest,
            lagged_links=lagged_links,
        )
    except Exception as exc:
        return RunResult(
            run_id=run_id,
            label=label,
            classification="gamma",
            n_samples=0,
            n_lagged_links=0,
            strongest_lagged_link=None,
            lagged_links=[],
            error=str(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage II-7.2 causal matrix")
    parser.add_argument("--run_ids", default="all", help="comma list like 1,2,3 or 'all'")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--output", default=os.path.join(DELIVERABLES_DIR, "causal_results_II7.json"))
    args = parser.parse_args()

    if args.run_ids == "all":
        run_ids = sorted(RUN_CONFIGS.keys())
    else:
        run_ids = [int(x.strip()) for x in args.run_ids.split(",") if x.strip()]

    results: List[dict] = []
    for rid in run_ids:
        rr = run_one(rid, args.max_samples)
        row = {
            "run_id": rr.run_id,
            "label": rr.label,
            "classification": rr.classification,
            "n_samples": rr.n_samples,
            "n_lagged_links": rr.n_lagged_links,
            "strongest_lagged_link": rr.strongest_lagged_link,
            "lagged_links": rr.lagged_links,
            "error": rr.error,
            "config": RUN_CONFIGS[rid],
        }
        print(f"run {rid}: class={rr.classification} samples={rr.n_samples} links={rr.n_lagged_links} error={rr.error}")
        results.append(row)

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "is_start": IS_START,
        "is_end": IS_END,
        "tau_max": 10,
        "pc_alpha": 0.01,
        "alpha_level": 0.05,
        "max_samples": args.max_samples,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

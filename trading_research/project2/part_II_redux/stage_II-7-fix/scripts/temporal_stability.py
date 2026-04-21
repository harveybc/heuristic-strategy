#!/usr/bin/env python3
"""
Stage II-7-fix.2: Causal Temporal Stability Test.

Tests whether the macd_hist τ=1 causal link found in Stage II-7.2
is temporally stable across 5 IS sub-periods × 2 assets = 10 PCMCI+ runs.

Rule 0.2: STRICT IS-ONLY. No data from 2020-01-01 onward.
Rule 0.3: Writes JSON results to deliverables/.
Rule 0.5: Run via conda activate tensorflow.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Script is at: part_II_redux/stage_II-7-fix/scripts/temporal_stability.py
STAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # stage_II-7-fix/
PART_II_DIR = os.path.dirname(STAGE_DIR)  # part_II_redux/
RAW_BINANCE_DIR = os.path.join(PART_II_DIR, "data", "raw", "binance")
DELIVERABLES_DIR = os.path.join(STAGE_DIR, "deliverables")
LOGS_DIR = os.path.join(STAGE_DIR, "logs")

os.makedirs(DELIVERABLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "stage_II-7-fix_progress.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)


def log_progress(task: str, action: str, status: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log.info(f"[{ts}] [II-7-fix.2] [{task}] [{action}] [{status}]")


# ---------------------------------------------------------------------------
# IS boundary guard (Rule 0.2)
# ---------------------------------------------------------------------------
HO_BOUNDARY = pd.Timestamp("2020-01-01", tz="UTC")


def assert_no_held_out(df: pd.DataFrame, label: str) -> None:
    dt_col = df["DateTime"] if "DateTime" in df.columns else df.index.to_series()
    if pd.to_datetime(dt_col, utc=True).max() >= HO_BOUNDARY:
        raise RuntimeError(
            f"HELD-OUT CONTAMINATION DETECTED in {label}: data extends past {HO_BOUNDARY}"
        )


# ---------------------------------------------------------------------------
# Sub-periods (exact per workplan Section 3.2 STEP 1)
# ---------------------------------------------------------------------------
SUB_PERIODS = [
    {"id": "2017-H2", "start": "2017-08-17", "end": "2017-12-31"},
    {"id": "2018-H1", "start": "2018-01-01", "end": "2018-06-30"},
    {"id": "2018-H2", "start": "2018-07-01", "end": "2018-12-31"},
    {"id": "2019-H1", "start": "2019-01-01", "end": "2019-06-30"},
    {"id": "2019-H2", "start": "2019-07-01", "end": "2019-12-31"},
]

ASSETS = ["BTC", "ETH"]
MIN_BARS = 2000  # per workplan: skip if <2000 bars after dropna

# ---------------------------------------------------------------------------
# PCMCI+ parameters (exact per workplan Section 3.2 STEP 2)
# ---------------------------------------------------------------------------
TAU_MAX = 10
PC_ALPHA = 0.01
ALPHA_LEVEL = 0.05
MAX_SAMPLES = 5000
TARGET = "target_fwd_6"

# Feature names (using Stage II-7.2 actual feature set for comparability)
FEATURE_COLS = [
    "adx", "di_spread", "atr_pct", "atr_ratio", "bb_width_pct",
    "bb_position", "rsi", "roc_12", "price_vs_ema50", "ema_alignment",
    "stoch_k", "macd_hist",
]


# ---------------------------------------------------------------------------
# Binance API historical fetch (for 2017-2018 data not in local parquets)
# ---------------------------------------------------------------------------
def fetch_binance_1h(symbol: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Fetch 1h OHLCV from Binance public API.  No auth required."""
    url = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str, tz="UTC").timestamp() * 1000)

    rows = []
    current = start_ts
    while current < end_ts:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": current,
            "endTime": end_ts,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning(f"  Binance fetch error for {symbol}: {exc}")
            break

        if not data:
            break

        rows.extend(data)
        last_ts = data[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 3_600_000  # +1h in ms
        time.sleep(0.2)  # rate limit

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "QuoteVolume", "Trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["DateTime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["DateTime", "Open", "High", "Low", "Close", "Volume"]].dropna().sort_values("DateTime")


def load_ohlcv_1h(asset: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Load 1h OHLCV for asset, fetching from Binance if local data not available."""
    symbol = f"{asset.lower()}usdt"
    parquet_path = os.path.join(RAW_BINANCE_DIR, f"{symbol}_1h_2019_2025.parquet")

    start_ts = pd.Timestamp(start_str, tz="UTC")
    end_ts = pd.Timestamp(end_str, tz="UTC")

    # Load existing parquet
    df_existing = pd.DataFrame()
    if os.path.exists(parquet_path):
        raw = pd.read_parquet(parquet_path)
        dt = pd.to_datetime(raw["DateTime"] if "DateTime" in raw.columns else raw.index, utc=True)
        df_existing = pd.DataFrame({
            "DateTime": dt,
            "Open": pd.to_numeric(raw["Open"], errors="coerce"),
            "High": pd.to_numeric(raw["High"], errors="coerce"),
            "Low": pd.to_numeric(raw["Low"], errors="coerce"),
            "Close": pd.to_numeric(raw["Close"], errors="coerce"),
            "Volume": pd.to_numeric(raw["Volume"], errors="coerce"),
        }).dropna(subset=["DateTime", "Open", "High", "Low", "Close"])
        df_existing = df_existing.sort_values("DateTime")

    # Check if we need to fetch pre-2019 data
    parquet_start = df_existing["DateTime"].min() if len(df_existing) > 0 else pd.Timestamp("2019-01-01", tz="UTC")

    if start_ts < parquet_start:
        # Always fetch the FULL year so both H1 and H2 sub-periods share one complete cache.
        year = start_ts.year
        year_start_str = f"{year}-01-01"
        year_end_ts = min(
            pd.Timestamp(f"{year}-12-31 23:59", tz="UTC"),
            parquet_start - pd.Timedelta(hours=1),
        )
        cache_path = os.path.join(RAW_BINANCE_DIR, f"{symbol}_1h_prefetch_{year}.parquet")

        df_pre = pd.DataFrame()
        if os.path.exists(cache_path):
            raw_pre = pd.read_parquet(cache_path)
            dt = pd.to_datetime(raw_pre["DateTime"] if "DateTime" in raw_pre.columns else raw_pre.index, utc=True)
            df_pre = pd.DataFrame({
                "DateTime": dt,
                "Open": pd.to_numeric(raw_pre["Open"], errors="coerce"),
                "High": pd.to_numeric(raw_pre["High"], errors="coerce"),
                "Low": pd.to_numeric(raw_pre["Low"], errors="coerce"),
                "Close": pd.to_numeric(raw_pre["Close"], errors="coerce"),
                "Volume": pd.to_numeric(raw_pre["Volume"], errors="coerce"),
            }).dropna()
            log.info(f"  Loaded cached prefetch: {len(df_pre)} rows  (max={df_pre['DateTime'].max().date() if len(df_pre) > 0 else 'n/a'})")

            # If cache is incomplete (only has H1, not H2), re-fetch the full year
            if len(df_pre) == 0 or df_pre["DateTime"].max() < (year_end_ts - pd.Timedelta(days=30)):
                log.info(f"  Cache incomplete — re-fetching full year {year} ({year_start_str} to {year_end_ts.date()})...")
                df_pre = fetch_binance_1h(symbol.upper(), year_start_str, str(year_end_ts.date()))
                if len(df_pre) > 0:
                    df_pre.to_parquet(cache_path, index=False)
                    log.info(f"  Re-cached {len(df_pre)} rows (full year {year}) -> {cache_path}")

        if len(df_pre) == 0:
            log.info(f"  Fetching {symbol} 1h full year {year} ({year_start_str} to {year_end_ts.date()}) via Binance API...")
            df_pre = fetch_binance_1h(symbol.upper(), year_start_str, str(year_end_ts.date()))
            if len(df_pre) > 0:
                df_pre.to_parquet(cache_path, index=False)
                log.info(f"  Fetched and cached {len(df_pre)} rows -> {cache_path}")
            else:
                log.warning(f"  Fetch returned 0 rows for {symbol} year {year}")

        if len(df_pre) > 0:
            df_existing = pd.concat([df_pre, df_existing]).drop_duplicates(subset=["DateTime"]).sort_values("DateTime")

    # Filter to requested period
    mask = (df_existing["DateTime"] >= start_ts) & (df_existing["DateTime"] <= end_ts)
    df_period = df_existing[mask].copy()
    return df_period.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering (identical to Stage II-7.2 stage_ii7_multitimeframe_causal.py)
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ADX / DI
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-12))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-12))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12) * 100
    adx = dx.rolling(14).mean()
    di_spread = plus_di - minus_di

    # ATR features
    atr_pct = atr14 / (close + 1e-12) * 100
    atr_ratio = atr14 / (atr14.rolling(120).mean() + 1e-12)

    # Bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width_pct = (bb_upper - bb_lower) / (sma20 + 1e-12) * 100
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-12)

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)

    # ROC
    roc_12 = close.pct_change(12) * 100

    # EMA features
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    ema50 = close.ewm(span=50).mean()
    price_vs_ema50 = (close - ema50) / (ema50 + 1e-12) * 100
    ema_alignment = (ema12 - ema26) / (atr14 + 1e-12)

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14 + 1e-12)

    # MACD
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line

    # Target
    target_fwd_6 = np.log(close.shift(-6) / close)

    feat = pd.DataFrame({
        "DateTime": df["DateTime"].values,
        "adx": adx.values,
        "di_spread": di_spread.values,
        "atr_pct": atr_pct.values,
        "atr_ratio": atr_ratio.values,
        "bb_width_pct": bb_width_pct.values,
        "bb_position": bb_position.values,
        "rsi": rsi.values,
        "roc_12": roc_12.values,
        "price_vs_ema50": price_vs_ema50.values,
        "ema_alignment": ema_alignment.values,
        "stoch_k": stoch_k.values,
        "macd_hist": macd_hist.values,
        TARGET: target_fwd_6.values,
    })
    return feat


# ---------------------------------------------------------------------------
# PCMCI+ runner
# ---------------------------------------------------------------------------
def run_pcmci(feat: pd.DataFrame, max_samples: int = MAX_SAMPLES) -> tuple[list[dict], int]:
    import tigramite.data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    data_cols = FEATURE_COLS + [TARGET]
    m = feat[data_cols].dropna()
    m = m[np.all(np.isfinite(m.to_numpy()), axis=1)]

    n_available = len(m)
    if n_available > max_samples:
        idx = np.sort(np.random.RandomState(42).choice(len(m), max_samples, replace=False))
        m = m.iloc[idx]

    arr = m.to_numpy(dtype=float)
    var_names = data_cols

    pcmci = PCMCI(
        dataframe=pp.DataFrame(arr, var_names=[v[:18] for v in var_names]),
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    results = pcmci.run_pcmciplus(tau_max=TAU_MAX, pc_alpha=PC_ALPHA)

    graph = results["graph"]
    pmat = results["p_matrix"]
    vmat = results["val_matrix"]
    target_idx = len(var_names) - 1

    lagged_links: list[dict] = []
    for i, feat_name in enumerate(FEATURE_COLS):
        for tau in range(1, TAU_MAX + 1):
            link = str(graph[i, target_idx, tau]).strip()
            p = float(pmat[i, target_idx, tau])
            if ("-->" in link or "o->" in link) and p < ALPHA_LEVEL:
                lagged_links.append({
                    "feature": feat_name,
                    "tau": tau,
                    "MCI": float(vmat[i, target_idx, tau]),
                    "p_value": p,
                    "link": link,
                })

    return lagged_links, n_available


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log_progress("temporal_stability", "script_start", "RUNNING")
    log.info("=" * 70)
    log.info("Stage II-7-fix.2: Causal Temporal Stability Test")
    log.info("=" * 70)

    all_results = []

    for asset in ASSETS:
        for period in SUB_PERIODS:
            pid = period["id"]
            start = period["start"]
            end = period["end"]
            label = f"{asset}_{pid}"
            log_progress("temporal_stability", f"run_{label}", "STARTING")
            log.info(f"\n{'='*50}")
            log.info(f"  Asset: {asset}  Period: {pid}  ({start} -> {end})")
            log.info(f"{'='*50}")

            result = {
                "asset": asset,
                "sub_period": pid,
                "start": start,
                "end": end,
                "samples_used": 0,
                "all_lagged_links": [],
                "macd_hist_tau1_present": False,
                "macd_hist_tau1_MCI": None,
                "macd_hist_tau1_p": None,
                "classification": "skipped",
                "skip_reason": None,
            }

            try:
                # Load OHLCV
                df = load_ohlcv_1h(asset, start, end)
                log.info(f"  Loaded {len(df)} raw OHLCV bars")

                if len(df) == 0:
                    result["skip_reason"] = "no_data_available"
                    result["classification"] = "skipped"
                    log.warning(f"  SKIP: No data available for {label}")
                    all_results.append(result)
                    log_progress("temporal_stability", f"run_{label}", "SKIPPED_NO_DATA")
                    continue

                # HO contamination guard (Rule 0.2)
                assert_no_held_out(df, label)

                # Compute features
                feat = compute_features(df)

                # Check bar count after dropna
                data_cols = FEATURE_COLS + [TARGET]
                valid = feat[data_cols].dropna()
                valid = valid[np.all(np.isfinite(valid.to_numpy()), axis=1)]
                n_valid = len(valid)
                log.info(f"  Valid samples after dropna: {n_valid}")

                if n_valid < MIN_BARS:
                    result["skip_reason"] = f"insufficient_bars_{n_valid}_lt_{MIN_BARS}"
                    result["classification"] = "skipped"
                    log.warning(f"  SKIP: Only {n_valid} valid bars < {MIN_BARS} threshold")
                    all_results.append(result)
                    log_progress("temporal_stability", f"run_{label}", f"SKIPPED_INSUFFICIENT_{n_valid}")
                    continue

                # Run PCMCI+
                log.info(f"  Running PCMCI+ (tau_max={TAU_MAX}, pc_alpha={PC_ALPHA}, max_samples={MAX_SAMPLES})...")
                t0 = time.time()
                lagged_links, samples_used = run_pcmci(feat)
                elapsed = time.time() - t0
                log.info(f"  PCMCI+ complete in {elapsed:.1f}s. Found {len(lagged_links)} lagged links.")

                # Check macd_hist τ=1
                macd_link = next(
                    (lk for lk in lagged_links if lk["feature"] == "macd_hist" and lk["tau"] == 1),
                    None,
                )

                result["samples_used"] = samples_used
                result["all_lagged_links"] = lagged_links
                result["macd_hist_tau1_present"] = macd_link is not None
                if macd_link:
                    result["macd_hist_tau1_MCI"] = macd_link["MCI"]
                    result["macd_hist_tau1_p"] = macd_link["p_value"]
                    log.info(f"  macd_hist τ=1: PRESENT  MCI={macd_link['MCI']:.4f}  p={macd_link['p_value']:.4f}")
                else:
                    log.info(f"  macd_hist τ=1: NOT PRESENT (p>=0.05 or not in graph)")

                # Classify this run
                if not lagged_links:
                    result["classification"] = "γ"
                else:
                    abs_vals = np.array([abs(lk["MCI"]) for lk in lagged_links])
                    pvals = np.array([lk["p_value"] for lk in lagged_links])
                    if np.any((abs_vals > 0.10) & (pvals < 0.01)):
                        result["classification"] = "α"
                    elif np.any((abs_vals >= 0.05) & (abs_vals <= 0.10) & (pvals < 0.05)):
                        result["classification"] = "β"
                    else:
                        result["classification"] = "γ"

                log.info(f"  Classification: {result['classification']}")
                log_progress("temporal_stability", f"run_{label}", f"DONE_class={result['classification']}")

            except RuntimeError as exc:
                # HO contamination or other critical error
                log.error(f"  CRITICAL ERROR for {label}: {exc}")
                result["skip_reason"] = str(exc)
                result["classification"] = "error"
                log_progress("temporal_stability", f"run_{label}", f"ERROR")
                # Write escalation file
                esc_path = os.path.join(
                    STAGE_DIR, "escalations", f"ESCALATION_temporal_{label.replace(' ', '_')}.md"
                )
                os.makedirs(os.path.dirname(esc_path), exist_ok=True)
                with open(esc_path, "w") as f:
                    f.write(f"# ESCALATION: Temporal Stability Run {label}\n\nError: {exc}\n")
                log.error(f"  Escalation written to {esc_path}")

            except Exception as exc:
                log.error(f"  Unexpected error for {label}: {exc}", exc_info=True)
                result["skip_reason"] = str(exc)
                result["classification"] = "error"
                log_progress("temporal_stability", f"run_{label}", "ERROR")

            all_results.append(result)

    # ---------------------------------------------------------------------------
    # Stability classification (per workplan Section 3.2 STEP 5)
    # ---------------------------------------------------------------------------
    stability_summary = {}
    for asset in ASSETS:
        asset_results = [r for r in all_results if r["asset"] == asset and r["classification"] != "skipped" and r["classification"] != "error"]
        n_runs = len(asset_results)
        n_present = sum(1 for r in asset_results if r["macd_hist_tau1_present"])
        signs = [np.sign(r["macd_hist_tau1_MCI"]) for r in asset_results if r["macd_hist_tau1_present"] and r["macd_hist_tau1_MCI"] is not None]
        sign_consistent = (len(set(signs)) <= 1) if signs else False

        if n_runs == 0:
            stability = "CANNOT_ASSESS"
        elif n_present >= 4 and sign_consistent:
            stability = "TEMPORALLY_STABLE"
        elif n_present >= 2:
            stability = "PARTIALLY_STABLE"
        elif n_present == 1:
            stability = "REGIME_SPECIFIC"
        else:
            stability = "NOT_STABLE"

        stability_summary[asset] = {
            "n_runs_completed": n_runs,
            "n_runs_skipped": sum(1 for r in all_results if r["asset"] == asset and r["classification"] == "skipped"),
            "n_macd_hist_tau1_present": n_present,
            "sign_consistent": sign_consistent,
            "stability_classification": stability,
        }

        log.info(f"\n{'='*50}")
        log.info(f"  {asset} STABILITY: {stability}  ({n_present}/{n_runs} sub-periods)")
        log.info(f"{'='*50}")

    # Other stable features (present in 3+ sub-periods across both assets)
    from collections import defaultdict
    feature_period_counts: dict[str, int] = defaultdict(int)
    for r in all_results:
        if r["classification"] not in ("skipped", "error"):
            seen = set()
            for lk in r["all_lagged_links"]:
                key = f"{lk['feature']}_tau{lk['tau']}"
                if key not in seen:
                    feature_period_counts[key] += 1
                    seen.add(key)
    stable_links = sorted(
        [(k, v) for k, v in feature_period_counts.items() if v >= 3],
        key=lambda x: -x[1],
    )

    # ---------------------------------------------------------------------------
    # Write JSON results (Rule 0.3)
    # ---------------------------------------------------------------------------
    output = {
        "task": "II-7-fix.2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "tau_max": TAU_MAX,
            "pc_alpha": PC_ALPHA,
            "alpha_level": ALPHA_LEVEL,
            "max_samples": MAX_SAMPLES,
            "features": FEATURE_COLS,
            "target": TARGET,
            "min_bars_threshold": MIN_BARS,
        },
        "sub_period_results": all_results,
        "stability_summary": stability_summary,
        "stable_links_3plus_periods": [{"key": k, "n_periods": v} for k, v in stable_links],
    }

    json_path = os.path.join(DELIVERABLES_DIR, "temporal_stability_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults written to: {json_path}")

    # ---------------------------------------------------------------------------
    # Write markdown deliverable
    # ---------------------------------------------------------------------------
    _write_markdown(all_results, stability_summary, stable_links, output["parameters"])
    log_progress("temporal_stability", "script_complete", "DONE")
    log.info("\nStage II-7-fix.2 COMPLETE.")


def _write_markdown(
    all_results: list[dict],
    stability_summary: dict,
    stable_links: list[tuple[str, int]],
    params: dict,
) -> None:
    md_path = os.path.join(DELIVERABLES_DIR, "TASK_II-7-fix.2_TEMPORAL_STABILITY.md")
    lines = []

    def add(s: str = "") -> None:
        lines.append(s)

    add("# TASK II-7-fix.2: Causal Temporal Stability Test")
    add()
    add(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    add()

    # Parameters
    add("## Parameters")
    add()
    add("| Parameter | Value |")
    add("|-----------|-------|")
    add(f"| Method | PCMCI+ |")
    add(f"| Independence test | ParCorr |")
    add(f"| tau_max | {params['tau_max']} |")
    add(f"| pc_alpha | {params['pc_alpha']} |")
    add(f"| alpha_level | {params['alpha_level']} |")
    add(f"| max_samples | {params['max_samples']} |")
    add(f"| target_variable | {params['target']} |")
    add(f"| n_features | {len(params['features'])} |")
    add(f"| min_bars_threshold | {params['min_bars_threshold']} |")
    add()

    # Results table
    add("## 1. Sub-Period Results Table")
    add()
    add("| Asset | Sub-Period | Samples | macd_hist τ=1 | MCI | p-value | Class |")
    add("|-------|-----------|---------|--------------|-----|---------|-------|")

    for r in all_results:
        asset = r["asset"]
        pid = r["sub_period"]
        n = r["samples_used"]
        cls = r["classification"]

        if cls in ("skipped", "error"):
            reason = r.get("skip_reason", "unknown")
            add(f"| {asset} | {pid} | — | SKIPPED | — | — | {reason[:30]} |")
        else:
            present = "✓ YES" if r["macd_hist_tau1_present"] else "✗ NO"
            mci = f"{r['macd_hist_tau1_MCI']:.4f}" if r["macd_hist_tau1_MCI"] is not None else "—"
            p = f"{r['macd_hist_tau1_p']:.4f}" if r["macd_hist_tau1_p"] is not None else "—"
            add(f"| {asset} | {pid} | {n} | {present} | {mci} | {p} | {cls} |")
    add()

    # Stability classifications
    add("## 2. macd_hist τ=1 Stability Classification")
    add()
    for asset, info in stability_summary.items():
        stab = info["stability_classification"]
        n_completed = info["n_runs_completed"]
        n_skipped = info["n_runs_skipped"]
        n_present = info["n_macd_hist_tau1_present"]
        sign_ok = info["sign_consistent"]
        add(f"### {asset}")
        add()
        add(f"- Sub-periods completed: **{n_completed}**  (skipped: {n_skipped})")
        add(f"- macd_hist τ=1 present in: **{n_present}/{n_completed}** sub-periods")
        add(f"- MCI sign consistent: **{sign_ok}**")
        add(f"- **CLASSIFICATION: {stab}**")
        add()
        if stab == "TEMPORALLY_STABLE":
            add("> macd_hist τ=1 is robust across IS sub-periods. Evidence supports Part III inclusion.")
        elif stab == "PARTIALLY_STABLE":
            add("> macd_hist τ=1 is present in some but not most sub-periods. Evidence is moderate.")
        elif stab == "REGIME_SPECIFIC":
            add("> macd_hist τ=1 appeared only in 1 sub-period (likely 2019). Original II-7.2 finding weakened.")
        elif stab == "NOT_STABLE":
            add("> macd_hist τ=1 not reproducible. Original II-7.2 finding not supported.")
        elif stab == "CANNOT_ASSESS":
            add("> Insufficient sub-period data to classify stability.")
        add()

    # Other stable features
    add("## 3. Other Stable Lagged Links (≥3 sub-periods)")
    add()
    if stable_links:
        add("| Feature_Tau | N sub-periods present |")
        add("|-------------|----------------------|")
        for key, n in stable_links:
            add(f"| {key} | {n} |")
    else:
        add("No feature-lag pair found in ≥3 sub-periods.")
    add()

    # Conclusion
    add("## 4. Conclusion")
    add()
    btc_stab = stability_summary.get("BTC", {}).get("stability_classification", "UNKNOWN")
    eth_stab = stability_summary.get("ETH", {}).get("stability_classification", "UNKNOWN")
    add(f"- **BTC macd_hist τ=1:** {btc_stab}")
    add(f"- **ETH macd_hist τ=1:** {eth_stab}")
    add()

    # Note about data availability
    add("### Data Availability Note")
    add()
    add("Local 1h parquet files (`btcusdt_1h_2019_2025.parquet`, `ethusdt_1h_2019_2025.parquet`) "
        "start from 2019-01-01. Pre-2019 sub-periods (2017-H2, 2018-H1, 2018-H2) required fetching "
        "from Binance public API. See `skip_reason` in JSON for any sub-periods that could not be loaded.")
    add()

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"Markdown deliverable written to: {md_path}")


if __name__ == "__main__":
    main()

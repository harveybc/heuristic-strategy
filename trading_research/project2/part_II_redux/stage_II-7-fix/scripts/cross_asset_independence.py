#!/usr/bin/env python3
"""
Stage II-7-fix.3: Cross-Asset Independence Test.

Tests whether macd_hist τ=1 in ETH 1h IS is an independent signal
or a common crypto-market-beta artifact shared with BTC.

Procedure:
  1. Compute BTC-ETH return correlation (IS only 2019)
  2. Regress ETH returns on BTC returns (OLS)
  3. Build synthetic residual ETH price from residuals
  4. Compute 12 technical features on residual price
  5. Run PCMCI+ on residual ETH features
  6. Compare vs original ETH PCMCI+ results

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
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Script is at: part_II_redux/stage_II-7-fix/scripts/cross_asset_independence.py
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
    log.info(f"[{ts}] [II-7-fix.3] [{task}] [{action}] [{status}]")


# ---------------------------------------------------------------------------
# IS boundaries (Rule 0.2 — strict IS-only, no 2020+ data)
# ---------------------------------------------------------------------------
IS_START = "2019-01-01"
IS_END = "2019-12-31"
HO_BOUNDARY = pd.Timestamp("2020-01-01", tz="UTC")

# ---------------------------------------------------------------------------
# PCMCI+ parameters (identical to Stage II-7.2)
# ---------------------------------------------------------------------------
TAU_MAX = 10
PC_ALPHA = 0.01
ALPHA_LEVEL = 0.05
MAX_SAMPLES = 5000
TARGET = "target_fwd_6"

FEATURE_COLS = [
    "adx", "di_spread", "atr_pct", "atr_ratio", "bb_width_pct",
    "bb_position", "rsi", "roc_12", "price_vs_ema50", "ema_alignment",
    "stoch_k", "macd_hist",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ohlcv_1h(asset: str) -> pd.DataFrame:
    """Load 1h OHLCV parquet filtered to IS period."""
    symbol = f"{asset.lower()}usdt"
    path = os.path.join(RAW_BINANCE_DIR, f"{symbol}_1h_2019_2025.parquet")
    raw = pd.read_parquet(path)
    dt = pd.to_datetime(raw["DateTime"] if "DateTime" in raw.columns else raw.index, utc=True)
    df = pd.DataFrame({
        "DateTime": dt,
        "Open": pd.to_numeric(raw["Open"], errors="coerce"),
        "High": pd.to_numeric(raw["High"], errors="coerce"),
        "Low": pd.to_numeric(raw["Low"], errors="coerce"),
        "Close": pd.to_numeric(raw["Close"], errors="coerce"),
        "Volume": pd.to_numeric(raw["Volume"], errors="coerce"),
    }).dropna(subset=["DateTime", "Open", "High", "Low", "Close"])
    df = df.sort_values("DateTime").reset_index(drop=True)

    # Filter to IS only
    df = df[(df["DateTime"] >= pd.Timestamp(IS_START, tz="UTC")) &
            (df["DateTime"] <= pd.Timestamp(IS_END, tz="UTC"))].copy()

    # HO guard (Rule 0.2)
    if len(df) > 0 and df["DateTime"].max() >= HO_BOUNDARY:
        raise RuntimeError(f"HELD-OUT CONTAMINATION: {asset} data extends past {HO_BOUNDARY}")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering (identical to Stage II-7.2)
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 12 technical features + forward return target."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

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

    # ATR
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

    # EMA
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

    # Forward return target
    target_fwd_6 = np.log(close.shift(-6) / close)

    feat = pd.DataFrame({
        "DateTime": df["DateTime"].values,
        "Close_raw": close.values,  # keep for reference
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
def run_pcmci(feat: pd.DataFrame, label: str = "") -> tuple[list[dict], int]:
    import tigramite.data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    data_cols = FEATURE_COLS + [TARGET]
    m = feat[data_cols].dropna()
    m = m[np.all(np.isfinite(m.to_numpy()), axis=1)]

    n_available = len(m)
    log.info(f"  [{label}] Valid samples: {n_available}")

    if n_available > MAX_SAMPLES:
        idx = np.sort(np.random.RandomState(42).choice(len(m), MAX_SAMPLES, replace=False))
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


def get_macd_tau1(links: list[dict]) -> dict | None:
    return next((lk for lk in links if lk["feature"] == "macd_hist" and lk["tau"] == 1), None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log_progress("cross_asset_independence", "script_start", "RUNNING")
    log.info("=" * 70)
    log.info("Stage II-7-fix.3: Cross-Asset Independence Test")
    log.info("=" * 70)
    log.info(f"IS period: {IS_START} to {IS_END}")

    # -----------------------------------------------------------------------
    # STEP 1: Load data and compute BTC-ETH return correlation
    # -----------------------------------------------------------------------
    log.info("\n--- STEP 1: Load data and compute BTC-ETH correlation ---")
    btc_df = load_ohlcv_1h("BTC")
    eth_df = load_ohlcv_1h("ETH")
    log.info(f"  BTC rows: {len(btc_df)}  ETH rows: {len(eth_df)}")

    btc_ret = btc_df.set_index("DateTime")["Close"].pct_change().rename("btc_ret")
    eth_ret = eth_df.set_index("DateTime")["Close"].pct_change().rename("eth_ret")

    common = pd.concat([btc_ret, eth_ret], axis=1, join="inner").dropna()
    log.info(f"  Common aligned rows: {len(common)}")

    pearson_corr = float(common["btc_ret"].corr(common["eth_ret"]))
    log.info(f"  BTC-ETH Pearson correlation (1h returns, IS 2019): {pearson_corr:.4f}")

    # -----------------------------------------------------------------------
    # STEP 2: OLS regression: ETH ~ BTC
    # -----------------------------------------------------------------------
    log.info("\n--- STEP 2: OLS regression ETH ~ BTC ---")
    X = sm.add_constant(common["btc_ret"])
    ols_model = sm.OLS(common["eth_ret"], X).fit()
    common["eth_residual"] = ols_model.resid

    r_squared = float(ols_model.rsquared)
    btc_coeff = float(ols_model.params["btc_ret"])
    btc_pval = float(ols_model.pvalues["btc_ret"])
    const_coeff = float(ols_model.params["const"])

    log.info(f"  OLS R²: {r_squared:.4f}")
    log.info(f"  BTC coefficient: {btc_coeff:.4f}  (p={btc_pval:.4e})")
    log.info(f"  Const: {const_coeff:.6f}")

    # -----------------------------------------------------------------------
    # STEP 3: Build synthetic residual ETH price
    # -----------------------------------------------------------------------
    log.info("\n--- STEP 3: Build residual ETH price ---")
    # Synthetic price from cumulative residual returns
    residual_price = (1 + common["eth_residual"]).cumprod() * 100.0

    # Build OHLCV-like DataFrame for residual price
    # For High/Low we approximate: use residual return magnitude as a fraction of residual price
    # This is a simplification — only Close matters for features.
    residual_df = pd.DataFrame({
        "DateTime": common.index,
        "Close": residual_price.values,
        "Open": residual_price.shift(1).fillna(residual_price.iloc[0]).values,
        "High": (residual_price * (1 + common["eth_residual"].abs() * 0.5)).values,
        "Low": (residual_price * (1 - common["eth_residual"].abs() * 0.5)).values,
        "Volume": np.ones(len(common)),  # dummy volume (no volume factor in residual)
    }).dropna()

    # Clip Low to prevent negatives
    residual_df["Low"] = residual_df["Low"].clip(lower=1e-6)
    log.info(f"  Residual ETH price range: {residual_df['Close'].min():.4f} - {residual_df['Close'].max():.4f}")
    log.info(f"  Residual ETH rows: {len(residual_df)}")

    # -----------------------------------------------------------------------
    # STEP 4: Compute features on residual ETH
    # -----------------------------------------------------------------------
    log.info("\n--- STEP 4: Compute features on residual ETH ---")
    residual_feat = compute_features(residual_df)

    # -----------------------------------------------------------------------
    # STEP 4b: Also compute features on original ETH for comparison
    # -----------------------------------------------------------------------
    log.info("\n--- STEP 4b: Compute features on original ETH ---")
    original_eth_feat = compute_features(eth_df)

    # -----------------------------------------------------------------------
    # STEP 5: Run PCMCI+ on ORIGINAL ETH
    # -----------------------------------------------------------------------
    log.info("\n--- STEP 5a: PCMCI+ on ORIGINAL ETH ---")
    t0 = time.time()
    original_links, original_samples = run_pcmci(original_eth_feat, label="ETH_original")
    log.info(f"  Original ETH PCMCI+ complete in {time.time()-t0:.1f}s. {len(original_links)} links found.")

    original_macd = get_macd_tau1(original_links)
    if original_macd:
        log.info(f"  Original ETH macd_hist τ=1: PRESENT  MCI={original_macd['MCI']:.4f}  p={original_macd['p_value']:.4f}")
    else:
        log.info(f"  Original ETH macd_hist τ=1: NOT PRESENT")

    # -----------------------------------------------------------------------
    # STEP 5: Run PCMCI+ on RESIDUAL ETH
    # -----------------------------------------------------------------------
    log.info("\n--- STEP 5b: PCMCI+ on RESIDUAL ETH ---")
    t0 = time.time()
    residual_links, residual_samples = run_pcmci(residual_feat, label="ETH_residual")
    log.info(f"  Residual ETH PCMCI+ complete in {time.time()-t0:.1f}s. {len(residual_links)} links found.")

    residual_macd = get_macd_tau1(residual_links)
    if residual_macd:
        log.info(f"  Residual ETH macd_hist τ=1: PRESENT  MCI={residual_macd['MCI']:.4f}  p={residual_macd['p_value']:.4f}")
    else:
        log.info(f"  Residual ETH macd_hist τ=1: NOT PRESENT")

    # -----------------------------------------------------------------------
    # Determine conclusion
    # -----------------------------------------------------------------------
    if original_macd and residual_macd:
        mci_drop_pct = (original_macd["MCI"] - residual_macd["MCI"]) / (abs(original_macd["MCI"]) + 1e-12) * 100
        if abs(mci_drop_pct) < 25:
            conclusion = "INDEPENDENT_SIGNAL"
            conclusion_text = (
                f"macd_hist τ=1 persists in residual ETH (MCI drop: {mci_drop_pct:.1f}%). "
                "ETH has independent macd_hist signal beyond crypto beta."
            )
        else:
            conclusion = "PARTIALLY_COMMON_FACTOR"
            conclusion_text = (
                f"macd_hist τ=1 weakens substantially in residual ETH (MCI drop: {mci_drop_pct:.1f}%). "
                "Partial common-factor artifact; ETH signal partially dependent on BTC."
            )
    elif original_macd and not residual_macd:
        conclusion = "COMMON_FACTOR_ARTIFACT"
        conclusion_text = (
            "macd_hist τ=1 disappears in residual ETH. "
            "ETH α finding was BTC factor artifact, not independent ETH signal."
        )
    elif not original_macd and not residual_macd:
        conclusion = "NOT_PRESENT_IN_EITHER"
        conclusion_text = (
            "macd_hist τ=1 not found in original ETH 2019 IS (may differ from II-7.2 full IS run). "
            "Cannot assess independence."
        )
    else:
        conclusion = "RESIDUAL_ONLY"
        conclusion_text = (
            "macd_hist τ=1 appears only in residual ETH (not in original). "
            "Unexpected result — residual has signal original did not."
        )

    log.info(f"\n  CONCLUSION: {conclusion}")
    log.info(f"  {conclusion_text}")

    # -----------------------------------------------------------------------
    # Write JSON results (Rule 0.3)
    # -----------------------------------------------------------------------
    output = {
        "task": "II-7-fix.3",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_period": {"start": IS_START, "end": IS_END},
        "parameters": {
            "tau_max": TAU_MAX,
            "pc_alpha": PC_ALPHA,
            "alpha_level": ALPHA_LEVEL,
            "max_samples": MAX_SAMPLES,
            "features": FEATURE_COLS,
            "target": TARGET,
        },
        "step1_btc_eth_correlation": {
            "pearson_correlation": pearson_corr,
            "n_aligned_bars": len(common),
        },
        "step2_ols_regression": {
            "r_squared": r_squared,
            "btc_coefficient": btc_coeff,
            "btc_p_value": btc_pval,
            "const_coefficient": const_coeff,
        },
        "step3_residual_eth": {
            "n_rows": len(residual_df),
            "price_min": float(residual_df["Close"].min()),
            "price_max": float(residual_df["Close"].max()),
        },
        "step5a_original_eth_pcmci": {
            "n_samples": original_samples,
            "n_lagged_links": len(original_links),
            "all_lagged_links": original_links,
            "macd_hist_tau1_present": original_macd is not None,
            "macd_hist_tau1_MCI": original_macd["MCI"] if original_macd else None,
            "macd_hist_tau1_p": original_macd["p_value"] if original_macd else None,
        },
        "step5b_residual_eth_pcmci": {
            "n_samples": residual_samples,
            "n_lagged_links": len(residual_links),
            "all_lagged_links": residual_links,
            "macd_hist_tau1_present": residual_macd is not None,
            "macd_hist_tau1_MCI": residual_macd["MCI"] if residual_macd else None,
            "macd_hist_tau1_p": residual_macd["p_value"] if residual_macd else None,
        },
        "conclusion": conclusion,
        "conclusion_text": conclusion_text,
    }

    json_path = os.path.join(DELIVERABLES_DIR, "cross_asset_independence_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults written to: {json_path}")

    # -----------------------------------------------------------------------
    # Write markdown deliverable
    # -----------------------------------------------------------------------
    _write_markdown(output, original_links, residual_links)
    log_progress("cross_asset_independence", "script_complete", "DONE")
    log.info("\nStage II-7-fix.3 COMPLETE.")


def _write_markdown(output: dict, original_links: list[dict], residual_links: list[dict]) -> None:
    md_path = os.path.join(DELIVERABLES_DIR, "TASK_II-7-fix.3_CROSS_ASSET_INDEPENDENCE.md")
    lines = []

    def add(s: str = "") -> None:
        lines.append(s)

    add("# TASK II-7-fix.3: Cross-Asset Independence Test")
    add()
    add(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    add(f"**IS Period:** {output['is_period']['start']} to {output['is_period']['end']}")
    add()

    # Step 1: BTC-ETH correlation
    add("## 1. BTC-ETH Return Correlation")
    add()
    corr_info = output["step1_btc_eth_correlation"]
    corr = corr_info["pearson_correlation"]
    add(f"| Metric | Value |")
    add(f"|--------|-------|")
    add(f"| Pearson correlation (1h returns) | **{corr:.4f}** |")
    add(f"| N aligned 1h bars | {corr_info['n_aligned_bars']} |")
    add()
    if corr > 0.7:
        add(f"> **High correlation ({corr:.3f} > 0.70):** BTC and ETH share substantial common factor.")
        add("> ETH may not be independent of BTC market beta.")
    elif corr > 0.5:
        add(f"> **Moderate correlation ({corr:.3f}):** Some BTC beta in ETH but not dominant.")
    else:
        add(f"> **Low correlation ({corr:.3f} ≤ 0.50):** Assets are relatively independent.")
    add()

    # Step 2: OLS regression
    add("## 2. OLS Regression: ETH ~ BTC")
    add()
    reg = output["step2_ols_regression"]
    add(f"| Parameter | Value |")
    add(f"|-----------|-------|")
    add(f"| R² | {reg['r_squared']:.4f} |")
    add(f"| BTC coefficient (β) | {reg['btc_coefficient']:.4f} |")
    add(f"| BTC p-value | {reg['btc_p_value']:.2e} |")
    add(f"| Intercept | {reg['const_coefficient']:.6f} |")
    add()
    add(f"**Interpretation:** {reg['r_squared']*100:.1f}% of ETH hourly return variance is explained by BTC returns.")
    add()

    # Step 5a: Original ETH PCMCI+
    add("## 3. PCMCI+ on Original ETH IS 2019")
    add()
    orig = output["step5a_original_eth_pcmci"]
    add(f"- Samples used: {orig['n_samples']}")
    add(f"- Lagged links found (p<0.05): {orig['n_lagged_links']}")
    if orig["macd_hist_tau1_present"]:
        add(f"- macd_hist τ=1: **PRESENT**  MCI={orig['macd_hist_tau1_MCI']:.4f}  p={orig['macd_hist_tau1_p']:.4f}")
    else:
        add(f"- macd_hist τ=1: **NOT PRESENT** (p≥0.05)")
    add()
    if orig["all_lagged_links"]:
        add("| Feature | τ | MCI | p-value |")
        add("|---------|---|-----|---------|")
        for lk in sorted(orig["all_lagged_links"], key=lambda x: abs(x["MCI"]), reverse=True)[:10]:
            add(f"| {lk['feature']} | {lk['tau']} | {lk['MCI']:.4f} | {lk['p_value']:.4f} |")
    else:
        add("No lagged links found.")
    add()

    # Step 5b: Residual ETH PCMCI+
    add("## 4. PCMCI+ on Residual ETH (After BTC Factor Removal)")
    add()
    res = output["step5b_residual_eth_pcmci"]
    add(f"- Samples used: {res['n_samples']}")
    add(f"- Lagged links found (p<0.05): {res['n_lagged_links']}")
    if res["macd_hist_tau1_present"]:
        add(f"- macd_hist τ=1: **PRESENT**  MCI={res['macd_hist_tau1_MCI']:.4f}  p={res['macd_hist_tau1_p']:.4f}")
    else:
        add(f"- macd_hist τ=1: **NOT PRESENT** (p≥0.05)")
    add()
    if res["all_lagged_links"]:
        add("| Feature | τ | MCI | p-value |")
        add("|---------|---|-----|---------|")
        for lk in sorted(res["all_lagged_links"], key=lambda x: abs(x["MCI"]), reverse=True)[:10]:
            add(f"| {lk['feature']} | {lk['tau']} | {lk['MCI']:.4f} | {lk['p_value']:.4f} |")
    else:
        add("No lagged links found.")
    add()

    # Comparison
    add("## 5. Comparison: Original vs Residual ETH")
    add()
    orig_mci = orig.get("macd_hist_tau1_MCI")
    res_mci = res.get("macd_hist_tau1_MCI")
    add(f"| | Original ETH | Residual ETH |")
    add(f"|-|-------------|-------------|")
    add(f"| macd_hist τ=1 present | {orig['macd_hist_tau1_present']} | {res['macd_hist_tau1_present']} |")
    add(f"| macd_hist τ=1 MCI | {f'{orig_mci:.4f}' if orig_mci is not None else '—'} | {f'{res_mci:.4f}' if res_mci is not None else '—'} |")
    add(f"| Total lagged links | {orig['n_lagged_links']} | {res['n_lagged_links']} |")
    add()

    # Conclusion
    add("## 6. Conclusion")
    add()
    add(f"**{output['conclusion']}**")
    add()
    add(output["conclusion_text"])
    add()

    # Implications for Part III
    add("## 7. Implications for Part III")
    add()
    conclusion = output["conclusion"]
    if conclusion == "INDEPENDENT_SIGNAL":
        add("ETH 1h technical α configuration retains Part III candidacy: "
            "its macd_hist signal is not merely a BTC beta artifact.")
    elif conclusion in ("COMMON_FACTOR_ARTIFACT", "PARTIALLY_COMMON_FACTOR"):
        add("ETH 1h technical α configuration should be reviewed: "
            "the causal finding partially or fully reflects BTC market beta rather than independent ETH structure. "
            "User must decide whether ETH 1h remains a Part III candidate.")
    else:
        add("Cannot determine Part III implications — see Note above.")
    add()

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"Markdown deliverable written to: {md_path}")


if __name__ == "__main__":
    main()

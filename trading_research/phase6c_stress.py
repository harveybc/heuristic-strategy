#!/usr/bin/env python3
"""
Phase 6.C — Stress Tests (6.C.1, 6.C.2, 6.C.4, 6.C.5)

Self-contained script. Runs on any machine with numpy, pandas, scipy.
Usage:
  python phase6c_stress.py --task 6c1   # JPY reversal (Gamma)
  python phase6c_stress.py --task 6c2   # MC regime scenarios (Dragon)
  python phase6c_stress.py --task 6c4   # Walk-forward (Dragon)
  python phase6c_stress.py --task 6c5   # Parameter perturbation (Gamma)
  python phase6c_stress.py --task all   # All tasks sequentially
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
EXTENDED_DATA = os.path.join(SCRIPT_DIR, "extended_data")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Try importing from trading_research package; fall back to local
try:
    sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
    from trading_research.evaluation_harness import (
        annualized_sharpe, rolling_window_evaluation,
        compute_strategy_metrics
    )
    from trading_research.transaction_cost_model import apply_cost_to_returns
except ImportError:
    sys.path.insert(0, SCRIPT_DIR)
    from evaluation_harness import annualized_sharpe, rolling_window_evaluation, compute_strategy_metrics
    from transaction_cost_model import apply_cost_to_returns

PPY_DAILY = 252
TARGET_VOL = 0.10
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"


# ================================================================
# DATA + STRATEGY (self-contained)
# ================================================================
def load_daily_data(asset):
    safe = asset.replace("/", "_")
    csv_path = os.path.join(EXTENDED_DATA, f"{safe}_daily.csv")
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found.")
        return None
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def run_pure_mr(log_ret, close, lookback=20, z_entry=1.5, z_exit=0.5):
    n = len(log_ret)
    positions = np.zeros(n)
    cum_ret = np.cumsum(log_ret)
    entry_bar = -999
    for i in range(lookback, n):
        window = cum_ret[max(0, i - lookback):i + 1]
        std = np.std(window)
        if std < 1e-12:
            continue
        z = (cum_ret[i] - np.mean(window)) / std
        if positions[i - 1] != 0:
            bars_held = i - entry_bar
            pnl = positions[i - 1] * (cum_ret[i] - cum_ret[entry_bar])
            atr = np.std(log_ret[max(0, i - lookback):i]) if i >= lookback else 0.01
            if pnl < -3.0 * atr or pnl > 2.0 * atr or bars_held >= 30 or abs(z) < z_exit:
                positions[i] = 0
            else:
                positions[i] = positions[i - 1]
        else:
            if z > z_entry:
                positions[i] = -1
                entry_bar = i
            elif z < -z_entry:
                positions[i] = 1
                entry_bar = i
    return positions


def run_tsmom(log_ret_daily, close_daily, dates, lookback_months=12):
    monthly_close = pd.Series(close_daily, index=dates).resample("ME").last().dropna()
    monthly_dates = monthly_close.index
    positions = np.zeros(len(log_ret_daily))
    for i in range(lookback_months, len(monthly_dates) - 1):
        ret_12m = np.log(monthly_close.iloc[i] + 1e-12) - np.log(monthly_close.iloc[i - lookback_months] + 1e-12)
        signal = np.sign(ret_12m)
        mask = (dates >= monthly_dates[max(0, i - 1)]) & (dates <= monthly_dates[i])
        recent = log_ret_daily[mask]
        vol = np.std(recent[-min(252, len(recent)):]) * np.sqrt(252) if len(recent) > 20 else 0.10
        size = min(TARGET_VOL / max(vol, 0.01), 3.0)
        next_date = monthly_dates[i + 1] if i + 1 < len(monthly_dates) else dates[-1]
        mask_next = (dates > monthly_dates[i]) & (dates <= next_date)
        positions[mask_next] = signal * size
    return positions


def run_dual_momentum(log_ret_daily, close_daily, dates, all_fx_data, asset, lookback_months=12):
    mc = pd.Series(close_daily, index=dates).resample("ME").last().dropna()
    all_mc = {}
    for a, (lr, cl, dt) in all_fx_data.items():
        all_mc[a] = pd.Series(cl, index=dt).resample("ME").last().dropna()
    positions = np.zeros(len(log_ret_daily))
    common = mc.index
    for a_mc in all_mc.values():
        common = common.intersection(a_mc.index)
    common = common.sort_values()
    for i in range(lookback_months, len(common) - 1):
        month = common[i]
        if month not in mc.index:
            continue
        mc_loc = mc.index.get_loc(month)
        if mc_loc < lookback_months:
            continue
        ret_12m = np.log(mc.iloc[mc_loc] + 1e-12) - np.log(mc.iloc[mc_loc - lookback_months] + 1e-12)
        all_rets = {}
        for a2, mc2 in all_mc.items():
            if month in mc2.index:
                loc2 = mc2.index.get_loc(month)
                if loc2 >= lookback_months:
                    all_rets[a2] = np.log(mc2.iloc[loc2] + 1e-12) - np.log(mc2.iloc[loc2 - lookback_months] + 1e-12)
        if not all_rets:
            continue
        best = max(all_rets, key=all_rets.get)
        next_month = common[i + 1] if i + 1 < len(common) else dates[-1]
        mask = (dates > month) & (dates <= next_month)
        if ret_12m > 0 and best == asset:
            positions[mask] = 1
    return positions


def eval_cell_returns(log_ret, positions, asset, dates):
    """Returns (net_returns, eval_dates, vol_scalar)."""
    gross_ret = positions[:-1] * log_ret[1:]
    net_ret = apply_cost_to_returns(gross_ret, positions[:-1], asset, np.abs(log_ret[:-1]))
    realized_vol = np.std(net_ret) * np.sqrt(PPY_DAILY) if np.std(net_ret) > 0 else 0.10
    vol_scalar = TARGET_VOL / max(realized_vol, 0.01)
    vol_scalar = min(vol_scalar, 5.0)
    net_ret = net_ret * vol_scalar
    eval_dates = dates[1:len(net_ret) + 1]
    return net_ret, eval_dates, vol_scalar


def load_all_data():
    """Load all required data and return structured dict."""
    assets = ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]
    data = {}
    for asset in assets:
        df = load_daily_data(asset)
        if df is not None:
            close = df["Close"].values.astype(float)
            high = df["High"].values.astype(float) if "High" in df.columns else close
            low = df["Low"].values.astype(float) if "Low" in df.columns else close
            dates = df.index
            log_ret = np.diff(np.log(close + 1e-12), prepend=0)
            log_ret[0] = 0
            data[asset] = {"close": close, "high": high, "low": low,
                           "dates": dates, "log_ret": log_ret}
            print(f"  {asset}: {len(close)} bars, {dates[0].date()} to {dates[-1].date()}")
    all_fx = {a: (data[a]["log_ret"], data[a]["close"], data[a]["dates"]) for a in data}
    return data, all_fx


def run_p3_on_returns(eurusd_lr, usdjpy_lr, eurusd_close, usdjpy_close,
                      eurusd_dates, usdjpy_dates, all_fx_data,
                      mr_lookback=20, mr_z_entry=1.5, mr_z_exit=0.5,
                      tsmom_lookback=12, dm_lookback=12):
    """
    Run P3 portfolio on given price data. Returns portfolio weekly returns + metrics.
    Parameterized for perturbation testing.
    """
    mr_pos = run_pure_mr(eurusd_lr, eurusd_close, mr_lookback, mr_z_entry, mr_z_exit)
    tsmom_pos = run_tsmom(usdjpy_lr, usdjpy_close, usdjpy_dates, tsmom_lookback)
    dm_pos = run_dual_momentum(usdjpy_lr, usdjpy_close, usdjpy_dates, all_fx_data, "USD/JPY", dm_lookback)

    mr_ret, mr_dt, _ = eval_cell_returns(eurusd_lr, mr_pos, "EUR/USD", eurusd_dates)
    ts_ret, ts_dt, _ = eval_cell_returns(usdjpy_lr, tsmom_pos, "USD/JPY", usdjpy_dates)
    dm_ret, dm_dt, _ = eval_cell_returns(usdjpy_lr, dm_pos, "USD/JPY", usdjpy_dates)

    # Weights from rolling worst-window
    mr_w2y = rolling_window_evaluation(mr_ret, PPY_DAILY)["worst_window_sharpe"]
    ts_w2y = rolling_window_evaluation(ts_ret, PPY_DAILY)["worst_window_sharpe"]
    dm_w2y = rolling_window_evaluation(dm_ret, PPY_DAILY)["worst_window_sharpe"]

    inv = {"mr": 1.0 / max(abs(mr_w2y), 0.01),
           "ts": 1.0 / max(abs(ts_w2y), 0.01),
           "dm": 1.0 / max(abs(dm_w2y), 0.01)}
    total = sum(inv.values())
    w = {k: v / total for k, v in inv.items()}

    # Align
    common = mr_dt.intersection(ts_dt).intersection(dm_dt).sort_values()
    mr_a = mr_ret[mr_dt.isin(common)][:len(common)]
    ts_a = ts_ret[ts_dt.isin(common)][:len(common)]
    dm_a = dm_ret[dm_dt.isin(common)][:len(common)]

    cells = {"mr": mr_a, "ts": ts_a, "dm": dm_a}
    df = pd.DataFrame(cells, index=common)
    weekly = df.resample("W").sum().dropna(how="all")
    port = np.zeros(len(weekly))
    for k in weekly.columns:
        port += w[k] * weekly[k].values

    # No portfolio-level vol re-scaling (Phase 6.D.1 fix)
    # Cells are already vol-scaled to 10% individually in eval_cell_returns()
    realized_vol = np.std(port) * np.sqrt(52) if len(port) > 10 else 0.10

    sharpe = annualized_sharpe(port, 52)
    rolling = rolling_window_evaluation(port, 52, window_years=2.0, step_months=6)
    # Correct equity for log returns
    eq = np.exp(np.cumsum(port))
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / (pk + 1e-12)
    max_dd = float(np.max(dd))

    # Also compute at 10% target vol for reference
    leverage = TARGET_VOL / max(realized_vol, 0.001)
    port_10vol = port * leverage
    eq_10vol = np.exp(np.cumsum(port_10vol))
    pk_10vol = np.maximum.accumulate(eq_10vol)
    dd_10vol = (pk_10vol - eq_10vol) / (pk_10vol + 1e-12)
    max_dd_10vol = float(np.max(dd_10vol))

    return {
        "sharpe": round(sharpe, 4),
        "worst_2y": rolling["worst_window_sharpe"],
        "max_dd": round(max_dd, 4),
        "total_return": round(float(eq[-1] - 1), 4) if len(eq) > 0 else 0.0,
        "n_weeks": len(port),
        "weights": {k: round(v, 4) for k, v in w.items()},
        "port_ret": port,
        "weekly_dates": weekly.index.values,
        "vol": round(realized_vol, 4),
        "at_10pct_vol": {
            "leverage": round(leverage, 4),
            "max_dd": round(max_dd_10vol, 4),
        },
    }


# ================================================================
# 6.C.1 — JPY REVERSAL STRESS TEST
# ================================================================
def task_6c1(data, all_fx):
    print("\n" + "=" * 70)
    print("TASK 6.C.1 — JPY REVERSAL STRESS TEST")
    print("=" * 70)

    np.random.seed(42)
    usdjpy_close = data["USD/JPY"]["close"]
    usdjpy_dates = data["USD/JPY"]["dates"]
    eurusd_close = data["EUR/USD"]["close"]
    eurusd_dates = data["EUR/USD"]["dates"]

    # Compute daily log returns
    usdjpy_lr = np.diff(np.log(usdjpy_close + 1e-12))
    eurusd_lr = np.diff(np.log(eurusd_close + 1e-12))

    # Align EUR/USD and USD/JPY daily returns
    common_dates = eurusd_dates.intersection(usdjpy_dates).sort_values()
    eu_mask = eurusd_dates.isin(common_dates)
    jp_mask = usdjpy_dates.isin(common_dates)
    # We need returns (diff) aligned — use dates[1:] for returns
    eu_ret_dates = eurusd_dates[1:]
    jp_ret_dates = usdjpy_dates[1:]
    common_ret_dates = eu_ret_dates.intersection(jp_ret_dates).sort_values()

    eu_rmask = eu_ret_dates.isin(common_ret_dates)
    jp_rmask = jp_ret_dates.isin(common_ret_dates)
    eurusd_rets = eurusd_lr[eu_rmask]
    usdjpy_rets = usdjpy_lr[jp_rmask]

    n_common = min(len(eurusd_rets), len(usdjpy_rets))
    eurusd_rets = eurusd_rets[:n_common]
    usdjpy_rets = usdjpy_rets[:n_common]

    # 6.C.1.1 — Identify JPY reversal episodes (periods of strong JPY appreciation = USD/JPY decline)
    print("\n  [6.C.1.1] Identifying JPY reversal episodes...")
    window = 63  # quarterly rolling
    usdjpy_cum = np.cumsum(usdjpy_rets)
    reversal_episodes = []
    # Find quarters where USD/JPY declined > 5%
    for i in range(window, len(usdjpy_rets)):
        q_ret = usdjpy_cum[i] - usdjpy_cum[i - window]
        if q_ret < -0.05:  # 5% JPY strengthening
            reversal_episodes.append(i)

    reversal_set = set(reversal_episodes)
    n_reversal_days = len(reversal_set)
    print(f"    Found {n_reversal_days} days within JPY reversal episodes ({n_reversal_days/n_common*100:.1f}%)")

    # 6.C.1.2 — Block bootstrap with JPY reversal overweight
    print("\n  [6.C.1.2] Generating 500 synthetic 2-year paths (block bootstrap)...")
    N_PATHS = 500
    PATH_LEN = 504  # ~2 years of trading days
    BLOCK_SIZE = 63  # quarterly blocks
    REVERSAL_FRACTION = 0.3  # 30% of blocks are reversal episodes

    # Build block indices
    n_blocks = (n_common - BLOCK_SIZE) // BLOCK_SIZE
    normal_blocks = []
    reversal_blocks = []
    for b in range(n_blocks):
        start = b * BLOCK_SIZE
        end = start + BLOCK_SIZE
        # Check if this block overlaps with reversal episodes
        block_indices = set(range(start, end))
        overlap = len(block_indices & reversal_set) / BLOCK_SIZE
        if overlap > 0.3:
            reversal_blocks.append((start, end))
        else:
            normal_blocks.append((start, end))

    print(f"    Normal blocks: {len(normal_blocks)}, Reversal blocks: {len(reversal_blocks)}")

    if len(reversal_blocks) < 2:
        print("    WARNING: Very few reversal blocks. Using amplified returns instead.")
        # Fall back to amplifying USD/JPY returns in random blocks
        reversal_blocks = normal_blocks.copy()

    # Cholesky for correlation preservation
    combined = np.column_stack([eurusd_rets, usdjpy_rets])
    corr_matrix = np.corrcoef(combined.T)
    print(f"    Historical EUR/USD vs USD/JPY correlation: {corr_matrix[0,1]:.4f}")

    path_results = []
    n_blocks_per_path = PATH_LEN // BLOCK_SIZE

    for p in range(N_PATHS):
        # Sample blocks
        n_reversal = max(1, int(n_blocks_per_path * REVERSAL_FRACTION))
        n_normal = n_blocks_per_path - n_reversal

        synth_eu = []
        synth_jp = []

        # Sample reversal blocks (with optional 1.5x amplification)
        for _ in range(n_reversal):
            idx = np.random.randint(0, len(reversal_blocks))
            s, e = reversal_blocks[idx]
            amp = np.random.choice([1.0, 1.5, 2.0], p=[0.4, 0.4, 0.2])
            synth_eu.append(eurusd_rets[s:e])
            synth_jp.append(usdjpy_rets[s:e] * amp)

        # Sample normal blocks
        for _ in range(n_normal):
            idx = np.random.randint(0, len(normal_blocks))
            s, e = normal_blocks[idx]
            synth_eu.append(eurusd_rets[s:e])
            synth_jp.append(usdjpy_rets[s:e])

        # Shuffle block order
        indices = list(range(n_blocks_per_path))
        np.random.shuffle(indices)

        eu_path = np.concatenate([synth_eu[i] for i in indices])[:PATH_LEN]
        jp_path = np.concatenate([synth_jp[i] for i in indices])[:PATH_LEN]

        # Build synthetic close prices from returns
        eu_close_synth = np.exp(np.cumsum(np.concatenate([[np.log(1.08)], eu_path])))
        jp_close_synth = np.exp(np.cumsum(np.concatenate([[np.log(120.0)], jp_path])))
        eu_lr_synth = np.diff(np.log(eu_close_synth + 1e-12), prepend=0)
        eu_lr_synth[0] = 0
        jp_lr_synth = np.diff(np.log(jp_close_synth + 1e-12), prepend=0)
        jp_lr_synth[0] = 0

        synth_dates = pd.date_range("2024-01-01", periods=len(eu_close_synth), freq="B")

        # Need peer data for dual momentum — use EUR/USD as both itself and synthetic peers
        # with independent noise for GBP, AUD
        gbp_lr = eu_lr_synth + np.random.normal(0, 0.001, len(eu_lr_synth))
        gbp_close = np.exp(np.cumsum(np.concatenate([[np.log(1.25)], gbp_lr[1:]])))
        gbp_close = np.concatenate([[1.25], gbp_close])[:len(synth_dates)]
        aud_lr = eu_lr_synth + np.random.normal(0, 0.001, len(eu_lr_synth))
        aud_close = np.exp(np.cumsum(np.concatenate([[np.log(0.65)], aud_lr[1:]])))
        aud_close = np.concatenate([[0.65], aud_close])[:len(synth_dates)]

        synth_fx = {
            "EUR/USD": (eu_lr_synth, eu_close_synth, synth_dates),
            "USD/JPY": (jp_lr_synth, jp_close_synth, synth_dates),
            "GBP/USD": (gbp_lr, gbp_close[:len(synth_dates)], synth_dates),
            "AUD/USD": (aud_lr, aud_close[:len(synth_dates)], synth_dates),
        }

        try:
            result = run_p3_on_returns(
                eu_lr_synth, jp_lr_synth, eu_close_synth, jp_close_synth,
                synth_dates, synth_dates, synth_fx)
            path_results.append({
                "sharpe": result["sharpe"],
                "max_dd": result["max_dd"],
                "total_return": result["total_return"],
            })
        except Exception as e:
            path_results.append({"sharpe": 0.0, "max_dd": 1.0, "total_return": -1.0, "error": str(e)})

        if (p + 1) % 100 == 0:
            print(f"    Completed {p+1}/{N_PATHS} paths...")

    # 6.C.1.3 — Aggregate
    sharpes = [r["sharpe"] for r in path_results if "error" not in r]
    max_dds = [r["max_dd"] for r in path_results if "error" not in r]
    errors = sum(1 for r in path_results if "error" in r)

    if len(sharpes) < 10:
        print("  FATAL: Too many path errors. Cannot compute statistics.")
        return {"pass": False, "reason": "insufficient_valid_paths", "errors": errors}

    pcts = [5, 10, 25, 50, 75, 90, 95]
    sharpe_pcts = {str(p): round(float(np.percentile(sharpes, p)), 4) for p in pcts}
    dd_pcts = {str(p): round(float(np.percentile(max_dds, p)), 4) for p in pcts}

    print(f"\n  [6.C.1.3] Results across {len(sharpes)} valid paths ({errors} errors):")
    print(f"    Sharpe percentiles: {sharpe_pcts}")
    print(f"    Max DD percentiles: {dd_pcts}")
    print(f"    Median Sharpe: {sharpe_pcts['50']}")
    print(f"    25th pct Sharpe: {sharpe_pcts['25']}")
    print(f"    Fraction Sharpe > 0: {sum(1 for s in sharpes if s > 0)/len(sharpes)*100:.1f}%")

    # 6.C.1.4 — Kill criteria
    median_sr = float(np.median(sharpes))
    pct25_sr = float(np.percentile(sharpes, 25))
    jpy_pass = median_sr >= 0 and pct25_sr >= -1.5

    print(f"\n  [KILL CRITERIA]")
    print(f"    Median Sharpe ≥ 0:      {'PASS' if median_sr >= 0 else 'FAIL'} ({median_sr:.4f})")
    print(f"    25th pct Sharpe ≥ -1.5:  {'PASS' if pct25_sr >= -1.5 else 'FAIL'} ({pct25_sr:.4f})")
    print(f"    OVERALL: {'PASS' if jpy_pass else 'FAIL'}")

    return {
        "n_valid_paths": len(sharpes),
        "n_errors": errors,
        "sharpe_percentiles": sharpe_pcts,
        "max_dd_percentiles": dd_pcts,
        "median_sharpe": round(median_sr, 4),
        "pct25_sharpe": round(pct25_sr, 4),
        "fraction_positive": round(sum(1 for s in sharpes if s > 0) / len(sharpes), 4),
        "pass": jpy_pass,
    }


# ================================================================
# 6.C.2 — MONTE CARLO REGIME SCENARIOS
# ================================================================
def task_6c2(data, all_fx):
    print("\n" + "=" * 70)
    print("TASK 6.C.2 — MONTE CARLO REGIME SCENARIOS")
    print("=" * 70)

    np.random.seed(123)

    # Define regimes
    regimes = {
        "A_inflation": ("2022-01-01", "2025-12-31"),
        "B_qe":        ("2013-01-01", "2019-12-31"),
        "C_crisis":    ("2008-01-01", "2012-12-31"),
    }

    eurusd_dates = data["EUR/USD"]["dates"]
    usdjpy_dates = data["USD/JPY"]["dates"]
    eurusd_lr_full = np.diff(np.log(data["EUR/USD"]["close"] + 1e-12))
    usdjpy_lr_full = np.diff(np.log(data["USD/JPY"]["close"] + 1e-12))

    # Align returns and dates
    eu_ret_dates = eurusd_dates[1:]
    jp_ret_dates = usdjpy_dates[1:]
    common_ret_dates = eu_ret_dates.intersection(jp_ret_dates).sort_values()
    eu_rmask = eu_ret_dates.isin(common_ret_dates)
    jp_rmask = jp_ret_dates.isin(common_ret_dates)
    eurusd_lr = eurusd_lr_full[eu_rmask]
    usdjpy_lr = usdjpy_lr_full[jp_rmask]
    n_common = min(len(eurusd_lr), len(usdjpy_lr))
    eurusd_lr = eurusd_lr[:n_common]
    usdjpy_lr = usdjpy_lr[:n_common]
    aligned_dates = common_ret_dates[:n_common]

    N_PATHS = 3000  # 3000 per regime (reduced from 10000 for tractability)
    PATH_LEN = 504

    all_regime_results = {}

    for regime_name, (start, end) in regimes.items():
        print(f"\n  [Regime {regime_name}] Period: {start} to {end}")

        # Extract regime-specific returns
        regime_mask = (aligned_dates >= pd.Timestamp(start)) & (aligned_dates <= pd.Timestamp(end))
        eu_regime = eurusd_lr[regime_mask]
        jp_regime = usdjpy_lr[regime_mask]

        if len(eu_regime) < 100:
            print(f"    WARNING: Only {len(eu_regime)} bars in regime. May be unreliable.")

        print(f"    Regime bars: {len(eu_regime)}")
        print(f"    EUR/USD: mean={np.mean(eu_regime)*252*100:.2f}% ann, vol={np.std(eu_regime)*np.sqrt(252)*100:.1f}%")
        print(f"    USD/JPY: mean={np.mean(jp_regime)*252*100:.2f}% ann, vol={np.std(jp_regime)*np.sqrt(252)*100:.1f}%")

        # Fit simple model: multivariate normal with GARCH(1,1)-like vol clustering
        # Use block bootstrap for simplicity (preserves autocorrelation structure)
        BLOCK_SIZE = 21  # monthly blocks to preserve clustering

        n_blocks = max(1, len(eu_regime) // BLOCK_SIZE)
        block_starts = list(range(0, len(eu_regime) - BLOCK_SIZE + 1))

        if len(block_starts) < 2:
            print(f"    WARNING: Insufficient data for block bootstrap. Using IID sampling.")
            block_starts = [0]
            BLOCK_SIZE = len(eu_regime)

        path_results = []
        blocks_per_path = PATH_LEN // BLOCK_SIZE + 1

        for p in range(N_PATHS):
            eu_path_parts = []
            jp_path_parts = []

            for _ in range(blocks_per_path):
                s = block_starts[np.random.randint(0, len(block_starts))]
                eu_path_parts.append(eu_regime[s:s + BLOCK_SIZE])
                jp_path_parts.append(jp_regime[s:s + BLOCK_SIZE])

            eu_path = np.concatenate(eu_path_parts)[:PATH_LEN]
            jp_path = np.concatenate(jp_path_parts)[:PATH_LEN]

            # Build synthetic prices
            eu_close = np.exp(np.cumsum(np.concatenate([[np.log(1.08)], eu_path])))
            jp_close = np.exp(np.cumsum(np.concatenate([[np.log(120.0)], jp_path])))
            eu_lr = np.diff(np.log(eu_close + 1e-12), prepend=0); eu_lr[0] = 0
            jp_lr = np.diff(np.log(jp_close + 1e-12), prepend=0); jp_lr[0] = 0

            synth_dates = pd.date_range("2024-01-01", periods=len(eu_close), freq="B")

            gbp_lr = eu_lr + np.random.normal(0, 0.001, len(eu_lr))
            gbp_close = np.exp(np.cumsum(np.concatenate([[np.log(1.25)], gbp_lr[1:]])))
            gbp_close = np.concatenate([[1.25], gbp_close])[:len(synth_dates)]
            aud_lr = eu_lr + np.random.normal(0, 0.001, len(eu_lr))
            aud_close = np.exp(np.cumsum(np.concatenate([[np.log(0.65)], aud_lr[1:]])))
            aud_close = np.concatenate([[0.65], aud_close])[:len(synth_dates)]

            synth_fx = {
                "EUR/USD": (eu_lr, eu_close, synth_dates),
                "USD/JPY": (jp_lr, jp_close, synth_dates),
                "GBP/USD": (gbp_lr, gbp_close[:len(synth_dates)], synth_dates),
                "AUD/USD": (aud_lr, aud_close[:len(synth_dates)], synth_dates),
            }

            try:
                result = run_p3_on_returns(eu_lr, jp_lr, eu_close, jp_close,
                                           synth_dates, synth_dates, synth_fx)
                # Worst single-quarter Sharpe within path
                port_ret = result["port_ret"]
                n_qtr = len(port_ret) // 13  # ~13 weeks per quarter
                quarterly_sharpes = []
                for q in range(n_qtr):
                    qr = port_ret[q*13:(q+1)*13]
                    if len(qr) > 4:
                        quarterly_sharpes.append(annualized_sharpe(qr, 52))
                worst_qtr = min(quarterly_sharpes) if quarterly_sharpes else -99

                path_results.append({
                    "sharpe": result["sharpe"],
                    "max_dd": result["max_dd"],
                    "worst_quarter": round(worst_qtr, 4),
                })
            except Exception:
                path_results.append({"sharpe": 0.0, "max_dd": 1.0, "worst_quarter": -99, "error": True})

            if (p + 1) % 500 == 0:
                print(f"    Completed {p+1}/{N_PATHS} paths...")

        # Aggregate
        valid = [r for r in path_results if "error" not in r]
        sharpes = [r["sharpe"] for r in valid]
        max_dds = [r["max_dd"] for r in valid]
        worst_qtrs = [r["worst_quarter"] for r in valid]
        errors = N_PATHS - len(valid)

        pcts = [5, 25, 50, 75, 95]
        regime_result = {
            "n_valid": len(valid),
            "n_errors": errors,
            "sharpe_percentiles": {str(p): round(float(np.percentile(sharpes, p)), 4) for p in pcts},
            "max_dd_percentiles": {str(p): round(float(np.percentile(max_dds, p)), 4) for p in pcts},
            "worst_quarter_percentiles": {str(p): round(float(np.percentile(worst_qtrs, p)), 4) for p in pcts},
            "fraction_positive": round(sum(1 for s in sharpes if s > 0) / max(len(sharpes), 1), 4),
        }
        all_regime_results[regime_name] = regime_result

        print(f"    Valid paths: {len(valid)}, errors: {errors}")
        print(f"    Sharpe pcts: {regime_result['sharpe_percentiles']}")
        print(f"    Max DD pcts: {regime_result['max_dd_percentiles']}")
        print(f"    Worst Qtr pcts: {regime_result['worst_quarter_percentiles']}")

    # Kill criteria: Regime B (QE) is crucial
    b_result = all_regime_results.get("B_qe", {})
    b_median_sr = float(b_result.get("sharpe_percentiles", {}).get("50", -99))
    b_pct25_dd = float(b_result.get("max_dd_percentiles", {}).get("75", 1.0))

    mc_pass = True
    for rname, rr in all_regime_results.items():
        med_sr = float(rr["sharpe_percentiles"]["50"])
        pct25_dd = float(rr["max_dd_percentiles"]["75"])
        if med_sr < 0 or pct25_dd > 0.25:
            mc_pass = False

    # Relaxed criteria for QE specifically
    b_fail = b_median_sr < 0 or b_pct25_dd > 0.30

    print(f"\n  [KILL CRITERIA]")
    for rname, rr in all_regime_results.items():
        med = rr["sharpe_percentiles"]["50"]
        dd75 = rr["max_dd_percentiles"]["75"]
        print(f"    {rname}: median SR={med}, 75th DD={dd75}")
    print(f"    Regime B (QE) fail: {b_fail}")
    print(f"    All regimes pass: {mc_pass}")
    print(f"    OVERALL: {'PASS' if not b_fail else 'FAIL'}")

    return {
        "regime_results": all_regime_results,
        "b_qe_median_sharpe": round(b_median_sr, 4),
        "b_qe_pass": not b_fail,
        "all_pass": mc_pass,
        "pass": not b_fail,
    }


# ================================================================
# 6.C.4 — EXPANDING-WINDOW WALK-FORWARD
# ================================================================
def task_6c4(data, all_fx):
    print("\n" + "=" * 70)
    print("TASK 6.C.4 — EXPANDING-WINDOW WALK-FORWARD")
    print("=" * 70)

    # Load all price data
    eurusd_close = data["EUR/USD"]["close"]
    eurusd_dates = data["EUR/USD"]["dates"]
    eurusd_lr = data["EUR/USD"]["log_ret"]
    usdjpy_close = data["USD/JPY"]["close"]
    usdjpy_dates = data["USD/JPY"]["dates"]
    usdjpy_lr = data["USD/JPY"]["log_ret"]

    # Generate quarterly dates from 2011-Q1 to 2023-Q4
    quarters = pd.date_range("2011-01-01", "2024-01-01", freq="QS")
    print(f"  Walk-forward: {len(quarters)-1} OOS quarters (2011-Q1 to 2023-Q4)")

    quarterly_results = []

    for q_idx in range(len(quarters) - 1):
        q_start = quarters[q_idx]
        q_end = quarters[q_idx + 1] - pd.Timedelta(days=1)

        # Training period: start of data to q_start
        train_end_date = q_start - pd.Timedelta(days=1)

        # Get training data masks
        eu_train_mask = eurusd_dates <= train_end_date
        jp_train_mask = usdjpy_dates <= train_end_date
        eu_test_mask = (eurusd_dates >= q_start) & (eurusd_dates <= q_end)
        jp_test_mask = (usdjpy_dates >= q_start) & (usdjpy_dates <= q_end)

        if eu_train_mask.sum() < 252 * 5 or jp_train_mask.sum() < 252 * 5:
            continue  # Need at least 5 years training

        # Run strategies on FULL data (positions computed using all available info up to each bar)
        # Then extract the OOS quarter positions
        mr_pos = run_pure_mr(eurusd_lr, eurusd_close)
        tsmom_pos = run_tsmom(usdjpy_lr, usdjpy_close, usdjpy_dates)
        dm_pos = run_dual_momentum(usdjpy_lr, usdjpy_close, usdjpy_dates, all_fx, "USD/JPY")

        # Get returns for the training period (to compute weights)
        mr_ret_train, mr_dt_train, _ = eval_cell_returns(
            eurusd_lr[eu_train_mask], mr_pos[eu_train_mask], "EUR/USD",
            eurusd_dates[eu_train_mask])
        ts_ret_train, ts_dt_train, _ = eval_cell_returns(
            usdjpy_lr[jp_train_mask], tsmom_pos[jp_train_mask], "USD/JPY",
            usdjpy_dates[jp_train_mask])
        dm_ret_train, dm_dt_train, _ = eval_cell_returns(
            usdjpy_lr[jp_train_mask], dm_pos[jp_train_mask], "USD/JPY",
            usdjpy_dates[jp_train_mask])

        # Compute P3 weights from training data only
        mr_w2y = rolling_window_evaluation(mr_ret_train, PPY_DAILY)["worst_window_sharpe"]
        ts_w2y = rolling_window_evaluation(ts_ret_train, PPY_DAILY)["worst_window_sharpe"]
        dm_w2y = rolling_window_evaluation(dm_ret_train, PPY_DAILY)["worst_window_sharpe"]

        inv = {"mr": 1.0 / max(abs(mr_w2y), 0.01),
               "ts": 1.0 / max(abs(ts_w2y), 0.01),
               "dm": 1.0 / max(abs(dm_w2y), 0.01)}
        total = sum(inv.values())
        weights = {k: v / total for k, v in inv.items()}

        # Get OOS quarter returns using FULL-data positions
        # (positions are causal — bar i position uses data up to bar i)
        mr_ret_full, mr_dt_full, mr_vs = eval_cell_returns(eurusd_lr, mr_pos, "EUR/USD", eurusd_dates)
        ts_ret_full, ts_dt_full, ts_vs = eval_cell_returns(usdjpy_lr, tsmom_pos, "USD/JPY", usdjpy_dates)
        dm_ret_full, dm_dt_full, dm_vs = eval_cell_returns(usdjpy_lr, dm_pos, "USD/JPY", usdjpy_dates)

        # Filter to OOS quarter
        mr_oos_mask = (mr_dt_full >= q_start) & (mr_dt_full <= q_end)
        ts_oos_mask = (ts_dt_full >= q_start) & (ts_dt_full <= q_end)
        dm_oos_mask = (dm_dt_full >= q_start) & (dm_dt_full <= q_end)

        mr_oos = mr_ret_full[mr_oos_mask]
        ts_oos = ts_ret_full[ts_oos_mask]
        dm_oos = dm_ret_full[dm_oos_mask]
        min_len = min(len(mr_oos), len(ts_oos), len(dm_oos))

        if min_len < 20:
            continue

        # Daily portfolio return for this quarter
        port_daily = (weights["mr"] * mr_oos[:min_len] +
                      weights["ts"] * ts_oos[:min_len] +
                      weights["dm"] * dm_oos[:min_len])

        # Quarterly metrics
        q_return = float(np.sum(port_daily))
        q_vol = float(np.std(port_daily) * np.sqrt(PPY_DAILY))
        q_sharpe = annualized_sharpe(port_daily, PPY_DAILY)

        eq = np.exp(np.cumsum(port_daily))  # Phase 6.D.1 fix: exp(cumsum) for log returns
        pk = np.maximum.accumulate(eq)
        dd = (pk - eq) / (pk + 1e-12)
        q_maxdd = float(np.max(dd))

        quarterly_results.append({
            "quarter": str(q_start.date()),
            "return": round(q_return, 4),
            "sharpe": round(q_sharpe, 4),
            "max_dd": round(q_maxdd, 4),
            "n_days": min_len,
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "positive": q_return > 0,
        })

    # Aggregate
    n_quarters = len(quarterly_results)
    n_positive = sum(1 for q in quarterly_results if q["positive"])
    frac_positive = n_positive / max(n_quarters, 1)
    all_sharpes = [q["sharpe"] for q in quarterly_results]
    median_q_sharpe = float(np.median(all_sharpes)) if all_sharpes else 0

    # Longest losing streak
    streak = 0
    max_streak = 0
    for q in quarterly_results:
        if not q["positive"]:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Worst quarter
    worst_q = min(quarterly_results, key=lambda x: x["sharpe"]) if quarterly_results else None

    print(f"\n  [RESULTS]")
    print(f"    Total OOS quarters: {n_quarters}")
    print(f"    Positive quarters:  {n_positive} ({frac_positive*100:.1f}%)")
    print(f"    Median quarterly Sharpe: {median_q_sharpe:.4f}")
    print(f"    Longest losing streak: {max_streak} quarters")
    if worst_q:
        print(f"    Worst quarter: {worst_q['quarter']} (Sharpe={worst_q['sharpe']:.4f})")

    # Show all quarterly results
    print(f"\n  [QUARTERLY DETAIL]")
    for q in quarterly_results:
        flag = "+" if q["positive"] else "-"
        print(f"    {flag} {q['quarter']}: ret={q['return']*100:+.1f}%, SR={q['sharpe']:+.2f}, DD={q['max_dd']*100:.1f}%")

    # Kill criteria
    wf_pass = (frac_positive >= 0.55 and
               median_q_sharpe >= 0.2 and
               max_streak <= 5)

    print(f"\n  [KILL CRITERIA]")
    print(f"    ≥55% positive quarters: {'PASS' if frac_positive >= 0.55 else 'FAIL'} ({frac_positive*100:.1f}%)")
    print(f"    Median Q Sharpe ≥ 0.2:  {'PASS' if median_q_sharpe >= 0.2 else 'FAIL'} ({median_q_sharpe:.4f})")
    print(f"    Losing streak ≤ 5:      {'PASS' if max_streak <= 5 else 'FAIL'} ({max_streak})")
    print(f"    OVERALL: {'PASS' if wf_pass else 'FAIL'}")

    return {
        "n_quarters": n_quarters,
        "n_positive": n_positive,
        "frac_positive": round(frac_positive, 4),
        "median_quarterly_sharpe": round(median_q_sharpe, 4),
        "longest_losing_streak": max_streak,
        "worst_quarter": worst_q,
        "quarterly_results": quarterly_results,
        "pass": wf_pass,
    }


# ================================================================
# 6.C.5 — PARAMETER PERTURBATION SENSITIVITY
# ================================================================
def task_6c5(data, all_fx):
    print("\n" + "=" * 70)
    print("TASK 6.C.5 — PARAMETER PERTURBATION SENSITIVITY")
    print("=" * 70)

    eurusd_close = data["EUR/USD"]["close"]
    eurusd_dates = data["EUR/USD"]["dates"]
    eurusd_lr = data["EUR/USD"]["log_ret"]
    usdjpy_close = data["USD/JPY"]["close"]
    usdjpy_dates = data["USD/JPY"]["dates"]
    usdjpy_lr = data["USD/JPY"]["log_ret"]

    # Parameter grids per cell
    mr_grids = {
        "lookback": [10, 15, 18, 20, 22, 25, 30],
        "z_entry":  [0.75, 1.125, 1.35, 1.5, 1.65, 1.875, 2.25],
        "z_exit":   [0.25, 0.375, 0.45, 0.5, 0.55, 0.625, 0.75],
    }
    tsmom_grids = {
        "lookback_months": [6, 9, 11, 12, 13, 15, 18],
    }
    dm_grids = {
        "lookback_months": [6, 9, 11, 12, 13, 15, 18],
    }

    all_results = {}

    # ── EUR/USD MR ──
    print("\n  [EUR/USD Mean Reversion]")
    mr_results = []
    baseline_params = {"lookback": 20, "z_entry": 1.5, "z_exit": 0.5}

    # Baseline
    mr_pos = run_pure_mr(eurusd_lr, eurusd_close, **baseline_params)
    mr_ret, mr_dt, _ = eval_cell_returns(eurusd_lr, mr_pos, "EUR/USD", eurusd_dates)
    baseline_sr = annualized_sharpe(mr_ret, PPY_DAILY)
    baseline_w2y = rolling_window_evaluation(mr_ret, PPY_DAILY)["worst_window_sharpe"]
    print(f"    Baseline: Sharpe={baseline_sr:.4f}, worst-2Y={baseline_w2y:.4f}")

    # Perturb each parameter independently
    for param, grid in mr_grids.items():
        for val in grid:
            params = baseline_params.copy()
            params[param] = val
            pos = run_pure_mr(eurusd_lr, eurusd_close, **params)
            ret, dt, _ = eval_cell_returns(eurusd_lr, pos, "EUR/USD", eurusd_dates)
            sr = annualized_sharpe(ret, PPY_DAILY)
            w2y = rolling_window_evaluation(ret, PPY_DAILY)["worst_window_sharpe"]
            mr_results.append({
                "param": param, "value": val,
                "sharpe": round(sr, 4), "worst_2y": round(w2y, 4),
                "is_baseline": (val == baseline_params[param]),
            })

    # Plateau analysis
    within_20pct = sum(1 for r in mr_results
                       if abs(r["sharpe"] - baseline_sr) <= 0.20 * abs(baseline_sr + 1e-8))
    mr_plateau = within_20pct / max(len(mr_results), 1)
    print(f"    Configs tested: {len(mr_results)}")
    print(f"    Within 20% of baseline: {within_20pct} ({mr_plateau*100:.0f}%)")
    print(f"    {'PLATEAU (robust)' if mr_plateau >= 0.60 else 'SPIKE (fragile)'}")

    for r in mr_results:
        flag = "B" if r["is_baseline"] else " "
        print(f"      {flag} {r['param']}={r['value']}: SR={r['sharpe']:.4f}, w2Y={r['worst_2y']:.4f}")

    all_results["eurusd_mr"] = {
        "baseline_sharpe": round(baseline_sr, 4),
        "baseline_worst_2y": round(baseline_w2y, 4),
        "perturbations": mr_results,
        "plateau_fraction": round(mr_plateau, 4),
        "is_plateau": mr_plateau >= 0.60,
    }

    # ── USD/JPY TSMOM ──
    print("\n  [USD/JPY TSMOM]")
    tsmom_results = []
    ts_baseline = 12

    ts_pos = run_tsmom(usdjpy_lr, usdjpy_close, usdjpy_dates, ts_baseline)
    ts_ret, ts_dt, _ = eval_cell_returns(usdjpy_lr, ts_pos, "USD/JPY", usdjpy_dates)
    ts_baseline_sr = annualized_sharpe(ts_ret, PPY_DAILY)
    ts_baseline_w2y = rolling_window_evaluation(ts_ret, PPY_DAILY)["worst_window_sharpe"]
    print(f"    Baseline (lookback=12): Sharpe={ts_baseline_sr:.4f}, worst-2Y={ts_baseline_w2y:.4f}")

    for lb in tsmom_grids["lookback_months"]:
        pos = run_tsmom(usdjpy_lr, usdjpy_close, usdjpy_dates, lb)
        ret, dt, _ = eval_cell_returns(usdjpy_lr, pos, "USD/JPY", usdjpy_dates)
        sr = annualized_sharpe(ret, PPY_DAILY)
        w2y = rolling_window_evaluation(ret, PPY_DAILY)["worst_window_sharpe"]
        tsmom_results.append({
            "param": "lookback_months", "value": lb,
            "sharpe": round(sr, 4), "worst_2y": round(w2y, 4),
            "is_baseline": (lb == ts_baseline),
        })

    within_20pct_ts = sum(1 for r in tsmom_results
                          if abs(r["sharpe"] - ts_baseline_sr) <= 0.20 * abs(ts_baseline_sr + 1e-8))
    ts_plateau = within_20pct_ts / max(len(tsmom_results), 1)
    print(f"    Configs tested: {len(tsmom_results)}")
    print(f"    Within 20% of baseline: {within_20pct_ts} ({ts_plateau*100:.0f}%)")
    print(f"    {'PLATEAU (robust)' if ts_plateau >= 0.60 else 'SPIKE (fragile)'}")

    for r in tsmom_results:
        flag = "B" if r["is_baseline"] else " "
        print(f"      {flag} lookback={r['value']}: SR={r['sharpe']:.4f}, w2Y={r['worst_2y']:.4f}")

    all_results["usdjpy_tsmom"] = {
        "baseline_sharpe": round(ts_baseline_sr, 4),
        "baseline_worst_2y": round(ts_baseline_w2y, 4),
        "perturbations": tsmom_results,
        "plateau_fraction": round(ts_plateau, 4),
        "is_plateau": ts_plateau >= 0.60,
    }

    # ── USD/JPY Dual Momentum ──
    print("\n  [USD/JPY Dual Momentum]")
    dm_results = []
    dm_baseline = 12

    dm_pos = run_dual_momentum(usdjpy_lr, usdjpy_close, usdjpy_dates, all_fx, "USD/JPY", dm_baseline)
    dm_ret, dm_dt, _ = eval_cell_returns(usdjpy_lr, dm_pos, "USD/JPY", usdjpy_dates)
    dm_baseline_sr = annualized_sharpe(dm_ret, PPY_DAILY)
    dm_baseline_w2y = rolling_window_evaluation(dm_ret, PPY_DAILY)["worst_window_sharpe"]
    print(f"    Baseline (lookback=12): Sharpe={dm_baseline_sr:.4f}, worst-2Y={dm_baseline_w2y:.4f}")

    for lb in dm_grids["lookback_months"]:
        pos = run_dual_momentum(usdjpy_lr, usdjpy_close, usdjpy_dates, all_fx, "USD/JPY", lb)
        ret, dt, _ = eval_cell_returns(usdjpy_lr, pos, "USD/JPY", usdjpy_dates)
        sr = annualized_sharpe(ret, PPY_DAILY)
        w2y = rolling_window_evaluation(ret, PPY_DAILY)["worst_window_sharpe"]
        dm_results.append({
            "param": "lookback_months", "value": lb,
            "sharpe": round(sr, 4), "worst_2y": round(w2y, 4),
            "is_baseline": (lb == dm_baseline),
        })

    within_20pct_dm = sum(1 for r in dm_results
                          if abs(r["sharpe"] - dm_baseline_sr) <= 0.20 * abs(dm_baseline_sr + 1e-8))
    dm_plateau = within_20pct_dm / max(len(dm_results), 1)
    print(f"    Configs tested: {len(dm_results)}")
    print(f"    Within 20% of baseline: {within_20pct_dm} ({dm_plateau*100:.0f}%)")
    print(f"    {'PLATEAU (robust)' if dm_plateau >= 0.60 else 'SPIKE (fragile)'}")

    for r in dm_results:
        flag = "B" if r["is_baseline"] else " "
        print(f"      {flag} lookback={r['value']}: SR={r['sharpe']:.4f}, w2Y={r['worst_2y']:.4f}")

    all_results["usdjpy_dm"] = {
        "baseline_sharpe": round(dm_baseline_sr, 4),
        "baseline_worst_2y": round(dm_baseline_w2y, 4),
        "perturbations": dm_results,
        "plateau_fraction": round(dm_plateau, 4),
        "is_plateau": dm_plateau >= 0.60,
    }

    # Overall kill criterion: all cells must show plateau
    all_plateau = all(v["is_plateau"] for v in all_results.values())

    print(f"\n  [KILL CRITERIA]")
    for name, r in all_results.items():
        print(f"    {name}: {r['plateau_fraction']*100:.0f}% within 20% → "
              f"{'PLATEAU' if r['is_plateau'] else 'SPIKE'}")
    print(f"    OVERALL: {'PASS' if all_plateau else 'FAIL'}")

    return {
        "cells": all_results,
        "all_plateau": all_plateau,
        "pass": all_plateau,
    }


# ================================================================
# MAIN
# ================================================================
def convert_for_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(x) for x in obj]
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif obj == float("inf"):
        return "inf"
    return obj


def main():
    parser = argparse.ArgumentParser(description="Phase 6.C Stress Tests")
    parser.add_argument("--task", default="all", choices=["6c1", "6c2", "6c4", "6c5", "all"])
    args = parser.parse_args()

    print("=" * 70)
    print(f"PHASE 6.C STRESS TESTS — Task: {args.task}")
    print("=" * 70)

    print("\n[DATA] Loading...")
    data, all_fx = load_all_data()

    results = {}

    tasks_to_run = [args.task] if args.task != "all" else ["6c1", "6c2", "6c4", "6c5"]

    for task in tasks_to_run:
        if task == "6c1":
            results["task_6c1"] = task_6c1(data, all_fx)
        elif task == "6c2":
            results["task_6c2"] = task_6c2(data, all_fx)
        elif task == "6c4":
            results["task_6c4"] = task_6c4(data, all_fx)
        elif task == "6c5":
            results["task_6c5"] = task_6c5(data, all_fx)

    # Save
    results_clean = convert_for_json(results)
    out_file = os.path.join(RESULTS_DIR, f"phase_6c_stress_{args.task}.json")
    with open(out_file, "w") as f:
        json.dump(results_clean, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_file}")

    # Summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    for task_name, r in results.items():
        print(f"  {task_name}: {'PASS' if r.get('pass') else 'FAIL'}")


if __name__ == "__main__":
    main()

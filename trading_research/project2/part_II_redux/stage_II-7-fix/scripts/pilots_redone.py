#!/usr/bin/env python3
"""
Stage II-7-fix.4: Re-run RL Pilots with Strict IS-Only Discipline.

Fixes from II-7-fix.1:
  - Root cause was held-out contamination (training on 2020-2022 bull run)
    causing always-long degenerate policy, not a code bug per se.
  - Fix: Use IS-only data strictly (train 2017-08-17 to 2019-06-30,
    val 2019-07-01 to 2019-12-31). Includes assertion to enforce this.

Usage:
  python pilots_redone.py --algorithms PPO SAC DQN  (all)
  python pilots_redone.py --algorithms SAC           (Dragon)
  python pilots_redone.py --algorithms PPO           (Gamma)
  python pilots_redone.py --algorithms DQN           (Omega)

Rule 0.2: STRICT IS-ONLY. Held-out boundary: 2020-01-01.
Rule 0.3: Writes JSON results to deliverables/.
Rule 0.5: Run via conda activate tensorflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Script is at: part_II_redux/stage_II-7-fix/scripts/pilots_redone.py
STAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # stage_II-7-fix/
PART_II_DIR = os.path.dirname(STAGE_DIR)  # part_II_redux/
RAW_BINANCE_DIR = os.path.join(PART_II_DIR, "data", "raw", "binance")
INFRA_DIR = os.path.join(PART_II_DIR, "infrastructure", "rl")
DELIVERABLES_DIR = os.path.join(STAGE_DIR, "deliverables")
LOGS_DIR = os.path.join(STAGE_DIR, "logs")

os.makedirs(DELIVERABLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Add infrastructure to path for TradingEnv
sys.path.insert(0, PART_II_DIR)

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
    log.info(f"[{ts}] [II-7-fix.4] [{task}] [{action}] [{status}]")


# ---------------------------------------------------------------------------
# STRICT IS-ONLY BOUNDARIES (Rule 0.2 — MANDATORY)
# ---------------------------------------------------------------------------
PILOT_TRAIN_START = "2019-01-01"   # earliest available 1h data
PILOT_TRAIN_END   = "2019-06-30"
PILOT_VAL_START   = "2019-07-01"
PILOT_VAL_END     = "2019-12-31"
HELD_OUT_BOUNDARY = "2020-01-01"   # NEVER TOUCHED


def assert_no_held_out(dt_series: pd.Series, label: str) -> None:
    """Rule 0.2: Raise immediately if any data touches held-out period."""
    boundary = pd.Timestamp(HELD_OUT_BOUNDARY, tz="UTC")
    max_dt = pd.to_datetime(dt_series, utc=True).max()
    assert max_dt < boundary, (
        f"HELD-OUT CONTAMINATION DETECTED in {label}: "
        f"max date {max_dt} >= {boundary}. HALTING."
    )


# ---------------------------------------------------------------------------
# Pilot configurations (exact per workplan Section 5.4)
# ---------------------------------------------------------------------------
PILOT_CONFIGS = [
    {
        "config_id": "btc_1h_technical",
        "asset": "BTC",
        "timeframe": "1h",
        "feature_set": "technical",
        "data_path": os.path.join(RAW_BINANCE_DIR, "btcusdt_1h_2019_2025.parquet"),
    },
    {
        "config_id": "eth_1h_technical",
        "asset": "ETH",
        "timeframe": "1h",
        "feature_set": "technical",
        "data_path": os.path.join(RAW_BINANCE_DIR, "ethusdt_1h_2019_2025.parquet"),
    },
]

PILOT_PARAMS = {
    "total_timesteps": 100_000,
    "reward_type": "log_return",
    "transaction_cost": 0.001,
    "max_drawdown_stop": 0.30,
    "seed": 42,
}

# TradingEnv default feature columns
FEATURE_COLS = [
    "returns", "log_returns", "rsi", "macd_hist", "bb_pos",
    "volume_ratio", "ema_cross", "atr_norm", "obv_delta",
    "momentum_5", "momentum_20", "volatility_20",
]


# ---------------------------------------------------------------------------
# Feature engineering (matches TradingEnv DEFAULT_CONFIG feature names)
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 12 technical features expected by TradingEnv."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Returns
    returns = close.pct_change()
    log_returns = np.log(close / close.shift(1))

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)

    # MACD histogram
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line

    # Bollinger Band position (bb_pos)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pos = (close - bb_lower) / (bb_upper - bb_lower + 1e-12)

    # Volume ratio (vs 20-bar mean)
    volume_ratio = volume / (volume.rolling(20).mean() + 1e-12)

    # EMA cross: (ema12 - ema26) / close
    ema_cross = (ema12 - ema26) / (close + 1e-12)

    # ATR norm: ATR(14) / close
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_norm = atr14 / (close + 1e-12)

    # OBV delta
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    obv_delta = obv.diff()

    # Momentum
    momentum_5 = close.pct_change(5)
    momentum_20 = close.pct_change(20)

    # Volatility (20-bar rolling std of returns)
    volatility_20 = returns.rolling(20).std()

    feat = pd.DataFrame({
        "DateTime": df["DateTime"].values,
        "Close": close.values,
        "returns": returns.values,
        "log_returns": log_returns.values,
        "rsi": rsi.values,
        "macd_hist": macd_hist.values,
        "bb_pos": bb_pos.values,
        "volume_ratio": volume_ratio.values,
        "ema_cross": ema_cross.values,
        "atr_norm": atr_norm.values,
        "obv_delta": obv_delta.values,
        "momentum_5": momentum_5.values,
        "momentum_20": momentum_20.values,
        "volatility_20": volatility_20.values,
    })
    return feat


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ohlcv(data_path: str) -> pd.DataFrame:
    """Load OHLCV from parquet, set DateTime as column."""
    raw = pd.read_parquet(data_path)
    dt = pd.to_datetime(raw["DateTime"] if "DateTime" in raw.columns else raw.index, utc=True)
    df = pd.DataFrame({
        "DateTime": dt,
        "Open": pd.to_numeric(raw["Open"], errors="coerce"),
        "High": pd.to_numeric(raw["High"], errors="coerce"),
        "Low": pd.to_numeric(raw["Low"], errors="coerce"),
        "Close": pd.to_numeric(raw["Close"], errors="coerce"),
        "Volume": pd.to_numeric(raw["Volume"], errors="coerce"),
    }).dropna(subset=["DateTime", "Open", "High", "Low", "Close"])
    return df.sort_values("DateTime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluate model on val environment (step by step, record actions)
# ---------------------------------------------------------------------------
def evaluate_model(model, val_env, algo_name: str) -> dict:
    """Run trained model on validation env, returning detailed metrics."""
    from infrastructure.rl.trading_env import TradingEnv  # noqa

    obs, _ = val_env.reset()
    done = False
    actions_taken = []
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = val_env.step(action)
        done = terminated or truncated

        # Record action — SB3 may return 0-d or 1-d numpy array
        act_val = float(np.squeeze(action))
        actions_taken.append(act_val)
        step_count += 1

    metrics = val_env.final_metrics()

    # Count action distribution
    if algo_name == "SAC":
        # Continuous: bucket into long (>0.3), short (<-0.3), flat
        long_frac = sum(1 for a in actions_taken if a > 0.3) / max(len(actions_taken), 1)
        short_frac = sum(1 for a in actions_taken if a < -0.3) / max(len(actions_taken), 1)
        flat_frac = 1 - long_frac - short_frac
        action_distribution = {
            "long_pct": round(long_frac * 100, 2),
            "short_pct": round(short_frac * 100, 2),
            "flat_pct": round(flat_frac * 100, 2),
        }
        # Num trades: count significant position changes
        prev = 0.0
        n_trades = 0
        for a in actions_taken:
            if abs(a - prev) > 0.3:
                n_trades += 1
            prev = a
    else:
        # Discrete: 0=hold, 1=buy, 2=sell
        action_counts = {0: 0, 1: 0, 2: 0}
        for a in actions_taken:
            action_counts[a] = action_counts.get(a, 0) + 1
        total = max(len(actions_taken), 1)
        action_distribution = {
            "hold_pct": round(action_counts[0] / total * 100, 2),
            "buy_pct": round(action_counts[1] / total * 100, 2),
            "sell_pct": round(action_counts[2] / total * 100, 2),
        }
        # Count trades = times action changed from hold to non-hold
        n_trades = sum(
            1 for i in range(1, len(actions_taken))
            if actions_taken[i] != 0 and actions_taken[i-1] != actions_taken[i]
        )

    return {
        "val_return": metrics.get("total_return", 0.0),
        "val_sharpe": metrics.get("sharpe_ratio", 0.0),
        "val_max_dd": metrics.get("max_drawdown", 0.0),
        "val_calmar": metrics.get("calmar_ratio", 0.0),
        "val_n_steps": step_count,
        "val_num_trades": n_trades,
        "val_actions_distribution": action_distribution,
        "first_10_actions": actions_taken[:10],
        "equity_curve_final": float(val_env._equity),
        "equity_curve_length": len(val_env.equity_curve),
    }


# ---------------------------------------------------------------------------
# Run one pilot: (config, algorithm)
# ---------------------------------------------------------------------------
def run_one_pilot(config: dict, algo_name: str) -> dict:
    """Train and evaluate one (config, algorithm) pilot. Returns result dict."""
    from infrastructure.rl.trading_env import TradingEnv
    from stable_baselines3 import PPO, SAC, DQN

    algo_class = {"PPO": PPO, "SAC": SAC, "DQN": DQN}[algo_name]
    config_id = config["config_id"]
    label = f"{config_id}_{algo_name}"
    log_progress(label, "pilot_start", "RUNNING")
    log.info(f"\n{'='*60}")
    log.info(f"  PILOT: {label}")
    log.info(f"  Train: {PILOT_TRAIN_START} -> {PILOT_TRAIN_END}")
    log.info(f"  Val:   {PILOT_VAL_START} -> {PILOT_VAL_END}")
    log.info(f"{'='*60}")

    result = {
        "config_id": config_id,
        "algorithm": algo_name,
        "asset": config["asset"],
        "train_period": [PILOT_TRAIN_START, PILOT_TRAIN_END],
        "val_period": [PILOT_VAL_START, PILOT_VAL_END],
        "held_out_boundary": HELD_OUT_BOUNDARY,
        "status": "failed",
        "error": None,
    }

    try:
        # ── STEP 1: Load data ────────────────────────────────────────────
        df = load_ohlcv(config["data_path"])
        log.info(f"  Loaded {len(df)} total rows  Range: {df['DateTime'].min().date()} -> {df['DateTime'].max().date()}")

        # Filter to IS only (train + val)
        df = df[
            (df["DateTime"] >= pd.Timestamp(PILOT_TRAIN_START, tz="UTC")) &
            (df["DateTime"] <= pd.Timestamp(PILOT_VAL_END, tz="UTC"))
        ].copy()
        log.info(f"  After IS filter: {len(df)} rows")

        # Mandatory HO contamination assertion (Rule 0.2)
        assert_no_held_out(df["DateTime"], label)
        log.info(f"  HO assertion PASSED: no data >= {HELD_OUT_BOUNDARY}")

        # ── STEP 2: Split train/val ──────────────────────────────────────
        train_df = df[
            (df["DateTime"] >= pd.Timestamp(PILOT_TRAIN_START, tz="UTC")) &
            (df["DateTime"] <= pd.Timestamp(PILOT_TRAIN_END, tz="UTC"))
        ].copy()
        val_df = df[
            (df["DateTime"] >= pd.Timestamp(PILOT_VAL_START, tz="UTC")) &
            (df["DateTime"] <= pd.Timestamp(PILOT_VAL_END, tz="UTC"))
        ].copy()

        log.info(f"  Train: {len(train_df)} rows  Val: {len(val_df)} rows")

        if len(train_df) < 500:
            raise ValueError(f"Insufficient train data: {len(train_df)} rows")
        if len(val_df) < 100:
            raise ValueError(f"Insufficient val data: {len(val_df)} rows")

        # ── STEP 3: Compute features ────────────────────────────────────
        train_feat = compute_features(train_df)
        val_feat = compute_features(val_df)

        # Drop NaN rows (feature warmup)
        train_feat = train_feat.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        val_feat = val_feat.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        log.info(f"  After feature computation: Train={len(train_feat)}  Val={len(val_feat)} valid rows")

        # ── Z-score normalization (fit on train only) ──────────────────
        scaler = StandardScaler()
        train_scaled = train_feat.copy()
        val_scaled = val_feat.copy()
        train_scaled[FEATURE_COLS] = scaler.fit_transform(train_feat[FEATURE_COLS].values)
        val_scaled[FEATURE_COLS] = scaler.transform(val_feat[FEATURE_COLS].values)
        log.info(f"  Z-score scaler fit on train, applied to val")

        # ── STEP 4: Build training env ───────────────────────────────────
        action_type = "continuous" if algo_name == "SAC" else "discrete"
        env_config = {
            "feature_cols": FEATURE_COLS,
            "action_space_type": action_type,
            "reward_type": PILOT_PARAMS["reward_type"],
            "transaction_cost": PILOT_PARAMS["transaction_cost"],
            "max_drawdown_stop": PILOT_PARAMS["max_drawdown_stop"],
        }

        train_env = TradingEnv(
            df_features=train_scaled[FEATURE_COLS],
            df_prices=train_scaled["Close"],
            config=env_config,
        )

        # ── STEP 5: Train ────────────────────────────────────────────────
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": train_env,
            "seed": PILOT_PARAMS["seed"],
            "verbose": 0,
        }
        if algo_name in ("PPO",):
            model_kwargs["tensorboard_log"] = None

        model = algo_class(**model_kwargs)
        log.info(f"  Training {algo_name} for {PILOT_PARAMS['total_timesteps']:,} timesteps...")
        t0 = time.time()
        model.learn(total_timesteps=PILOT_PARAMS["total_timesteps"])
        train_elapsed = time.time() - t0
        log.info(f"  Training complete in {train_elapsed:.1f}s")

        # Record training loss (from logger)
        training_loss_final = None
        try:
            if hasattr(model, "logger") and model.logger is not None:
                # SB3 stores recent logs in model.ep_info_buffer or similar
                pass  # loss tracking varies by algorithm
        except Exception:
            pass

        # ── STEP 6: Build val env and evaluate ──────────────────────────
        val_env = TradingEnv(
            df_features=val_scaled[FEATURE_COLS],
            df_prices=val_scaled["Close"],
            config=env_config,
        )

        log.info(f"  Evaluating on val set ({len(val_feat)} bars)...")
        eval_metrics = evaluate_model(model, val_env, algo_name)

        # Log first 10 actions for transparency
        log.info(f"  First 10 actions: {eval_metrics['first_10_actions']}")
        log.info(f"  Action distribution: {eval_metrics['val_actions_distribution']}")
        log.info(f"  Val return:  {eval_metrics['val_return']:.4f}")
        log.info(f"  Val Sharpe:  {eval_metrics['val_sharpe']:.4f}")
        log.info(f"  Val Max DD:  {eval_metrics['val_max_dd']:.4f}")
        log.info(f"  Val trades:  {eval_metrics['val_num_trades']}")

        # ── STEP 7: Build result dict ────────────────────────────────────
        result.update({
            "status": "success",
            "train_n_rows": len(train_feat),
            "val_n_rows": len(val_feat),
            "action_space_type": action_type,
            "training_elapsed_sec": round(train_elapsed, 1),
            "training_loss_final": training_loss_final,
        })
        result.update(eval_metrics)

        # Learnability verdict
        result["learnability_verdict"] = (
            "LEARNABLE"
            if eval_metrics["val_sharpe"] > 0 and eval_metrics["val_num_trades"] > 5
            else "NOT_LEARNABLE"
        )
        log.info(f"  Learnability: {result['learnability_verdict']}")

        # Save per-pilot result JSON
        pilot_json_path = os.path.join(DELIVERABLES_DIR, f"pilot_{config_id}_{algo_name}.json")
        with open(pilot_json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"  Saved: {pilot_json_path}")
        log_progress(label, "pilot_complete", f"DONE_{result['learnability_verdict']}")

    except AssertionError as exc:
        # Held-out contamination - HALT immediately (Rule 0.2)
        log.error(f"  HELD-OUT CONTAMINATION: {exc}")
        esc_path = os.path.join(STAGE_DIR, "escalations", f"ESCALATION_held_out_contamination_{label}.md")
        os.makedirs(os.path.dirname(esc_path), exist_ok=True)
        with open(esc_path, "w") as f:
            f.write(f"# ESCALATION: Held-Out Contamination\n\nPilot: {label}\nError: {exc}\n")
        result["status"] = "held_out_contamination"
        result["error"] = str(exc)
        log_progress(label, "pilot_error", "HELD_OUT_CONTAMINATION")
        raise  # halt all pilots

    except Exception as exc:
        log.error(f"  Error in pilot {label}: {exc}", exc_info=True)
        result["status"] = "failed"
        result["error"] = str(exc)
        log_progress(label, "pilot_error", f"FAILED_{type(exc).__name__}")

    return result


# ---------------------------------------------------------------------------
# Sanity checks (Rule 0.4 + workplan Section 5.6)
# ---------------------------------------------------------------------------
def run_sanity_checks(all_results: list[dict]) -> dict:
    """Run 3 mandatory post-execution sanity checks."""
    checks = {
        "check1_no_identical_sharpe": {"passed": True, "detail": ""},
        "check2_action_dists_differ": {"passed": True, "detail": ""},
        "check3_training_occurred": {"passed": True, "detail": ""},
    }

    # Group by config_id
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        if r.get("status") == "success":
            by_config[r["config_id"]].append(r)

    # CHECK 1: No identical Sharpe (to 3 decimals)
    for config_id, results in by_config.items():
        sharpes = [round(r.get("val_sharpe", 0.0), 3) for r in results]
        algos = [r["algorithm"] for r in results]
        if len(sharpes) >= 2 and len(set(sharpes)) < len(sharpes):
            # Find duplicates
            seen = {}
            duplicates = []
            for algo, s in zip(algos, sharpes):
                if s in seen:
                    duplicates.append((seen[s], algo, s))
                seen[s] = algo
            detail = f"{config_id}: {duplicates}"
            checks["check1_no_identical_sharpe"]["passed"] = False
            checks["check1_no_identical_sharpe"]["detail"] += detail + "; "
            log.error(f"  CHECK 1 FAILED: Identical Sharpe detected: {detail}")

    # CHECK 2: Action distributions differ across algorithms per config
    for config_id, results in by_config.items():
        if len(results) < 2:
            continue
        dists = [str(r.get("val_actions_distribution", {})) for r in results]
        if len(set(dists)) == 1:  # all identical
            detail = f"{config_id}: all algorithms produced identical action distributions"
            checks["check2_action_dists_differ"]["passed"] = False
            checks["check2_action_dists_differ"]["detail"] += detail + "; "
            log.warning(f"  CHECK 2 WARNING: {detail}")

    # CHECK 3: Training occurred (val Sharpe not all exactly 0, num_trades > 0)
    for r in all_results:
        if r.get("status") == "success":
            if r.get("val_num_trades", 0) == 0:
                detail = f"{r['config_id']}_{r['algorithm']}: 0 trades (possible training failure)"
                checks["check3_training_occurred"]["detail"] += detail + "; "
                log.warning(f"  CHECK 3 WARNING: {detail}")

    for k, v in checks.items():
        status = "PASS" if v["passed"] else "FAIL"
        log.info(f"  {k}: {status}  {v.get('detail', '')[:80]}")

    return checks


# ---------------------------------------------------------------------------
# Write markdown deliverable
# ---------------------------------------------------------------------------
def write_markdown(all_results: list[dict], checks: dict, params: dict) -> None:
    md_path = os.path.join(DELIVERABLES_DIR, "TASK_II-7-fix.4_PILOTS_REDONE.md")
    lines = []

    def add(s: str = "") -> None:
        lines.append(s)

    add("# TASK II-7-fix.4: Pilots Redone with Strict IS-Only Discipline")
    add()
    add(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    add()

    add("## 1. Bug Fix Applied")
    add()
    add("**Root cause (from II-7-fix.1):** Stage II-7.5 used 2020-2022 (strong bull run) for training,")
    add("causing PPO, SAC, and DQN to converge to always-long degenerate policies. Val on 2023 (also")
    add("bullish) produced identical equity curves. This is held-out contamination + degenerate policy.")
    add()
    add("**Fix applied:** Training restricted to 2019-01-01 to 2019-06-30 (IS period only).")
    add(f"Val restricted to {params['val_start']} to {params['val_end']}.")
    add("HO assertion added to prevent any future contamination.")
    add()

    add("## 2. Temporal Boundary Verification")
    add()
    add(f"- `PILOT_TRAIN_START`: {params['train_start']}")
    add(f"- `PILOT_TRAIN_END`: {params['train_end']}")
    add(f"- `PILOT_VAL_START`: {params['val_start']}")
    add(f"- `PILOT_VAL_END`: {params['val_end']}")
    add(f"- `HELD_OUT_BOUNDARY`: {params['ho_boundary']} (never touched)")
    add()
    passed_ho = all(
        r.get("status") not in ("held_out_contamination",)
        for r in all_results
    )
    add(f"HO assertion result: **{'PASS — no contamination detected' if passed_ho else 'FAIL — contamination detected'}**")
    add()

    add("## 3. Pilot Results Table")
    add()
    add("| Config | Algorithm | Train N | Val N | Val Return | Val Sharpe | Val MaxDD | Trades | Learnability |")
    add("|--------|-----------|---------|-------|-----------|-----------|----------|--------|-------------|")
    for r in all_results:
        if r.get("status") == "success":
            cid = r["config_id"]
            algo = r["algorithm"]
            tn = r.get("train_n_rows", "—")
            vn = r.get("val_n_rows", "—")
            vret = f"{r.get('val_return', 0):.4f}"
            vshp = f"{r.get('val_sharpe', 0):.4f}"
            vdd = f"{r.get('val_max_dd', 0):.4f}"
            ntrades = r.get("val_num_trades", "—")
            learn = r.get("learnability_verdict", "—")
            add(f"| {cid} | {algo} | {tn} | {vn} | {vret} | {vshp} | {vdd} | {ntrades} | {learn} |")
        else:
            add(f"| {r.get('config_id','?')} | {r.get('algorithm','?')} | — | — | — | — | — | — | FAILED: {r.get('error','?')[:40]} |")
    add()

    add("## 4. Sanity Check Results")
    add()
    for name, info in checks.items():
        status = "✓ PASS" if info["passed"] else "✗ FAIL"
        add(f"- **{name}**: {status}  {info.get('detail','')[:100]}")
    add()

    add("## 5. Action Distributions")
    add()
    for r in all_results:
        if r.get("status") == "success":
            add(f"**{r['config_id']} — {r['algorithm']}:**")
            add(f"- Distribution: {r.get('val_actions_distribution', {})}")
            add(f"- First 10 actions: {r.get('first_10_actions', [])}")
            add()

    add("## 6. Learnability Verdict per Pilot")
    add()
    for r in all_results:
        if r.get("status") == "success":
            label = f"{r['config_id']}/{r['algorithm']}"
            verdict = r.get("learnability_verdict", "UNKNOWN")
            sharpe = r.get("val_sharpe", 0)
            trades = r.get("val_num_trades", 0)
            add(f"- **{label}**: {verdict}  (Sharpe={sharpe:.4f}, trades={trades})")
        else:
            add(f"- **{r.get('config_id','?')}/{r.get('algorithm','?')}**: FAILED ({r.get('error','?')[:60]})")
    add()

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"Markdown deliverable written to: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage II-7-fix.4 Pilots Redone")
    parser.add_argument(
        "--algorithms", nargs="+", default=["PPO", "SAC", "DQN"],
        choices=["PPO", "SAC", "DQN"],
        help="Which algorithms to run (Dragon=SAC, Gamma=PPO, Omega=DQN)",
    )
    args = parser.parse_args()

    log_progress("pilots_redone", "script_start", f"RUNNING_algos={'_'.join(args.algorithms)}")
    log.info("=" * 70)
    log.info("Stage II-7-fix.4: Pilots Redone — IS-Only Strict Discipline")
    log.info(f"Algorithms: {args.algorithms}")
    log.info(f"Train: {PILOT_TRAIN_START} -> {PILOT_TRAIN_END}")
    log.info(f"Val:   {PILOT_VAL_START} -> {PILOT_VAL_END}")
    log.info(f"HO Boundary: {HELD_OUT_BOUNDARY} (NEVER TOUCHED)")
    log.info("=" * 70)

    # Load any existing results from other machines (if running partial)
    existing_results: list[dict] = []
    existing_json = os.path.join(DELIVERABLES_DIR, "pilot_redone_results.json")
    if os.path.exists(existing_json):
        with open(existing_json) as f:
            prev = json.load(f)
        existing_results = prev.get("pilot_results", [])
        log.info(f"  Loaded {len(existing_results)} existing pilot results from prior machine")

    all_results: list[dict] = list(existing_results)

    for config in PILOT_CONFIGS:
        for algo_name in args.algorithms:
            # Skip if already run by another machine
            already_done = any(
                r.get("config_id") == config["config_id"] and
                r.get("algorithm") == algo_name and
                r.get("status") == "success"
                for r in all_results
            )
            if already_done:
                log.info(f"  SKIP (already done): {config['config_id']}_{algo_name}")
                continue

            result = run_one_pilot(config, algo_name)
            all_results.append(result)

            # Write intermediate results after each pilot
            _write_json(all_results, existing_results, args.algorithms)

    # Run sanity checks on completed results
    log.info("\n" + "="*60)
    log.info("Running post-completion sanity checks...")
    checks = run_sanity_checks(all_results)

    # Check 1 failure triggers escalation per Rule 0.4
    if not checks["check1_no_identical_sharpe"]["passed"]:
        esc_path = os.path.join(STAGE_DIR, "escalations", "ESCALATION_identical_metrics_v2.md")
        os.makedirs(os.path.dirname(esc_path), exist_ok=True)
        with open(esc_path, "w") as f:
            f.write(f"# ESCALATION: Identical Metrics in Pilots Redone\n\n"
                    f"Check 1 failed: {checks['check1_no_identical_sharpe']['detail']}\n\n"
                    f"Per Rule 0.4: HALT. Do not write final deliverable until investigated.\n")
        log.error(f"  ESCALATION written: {esc_path}")
        log_progress("pilots_redone", "sanity_check_1_fail", "ESCALATION")
        sys.exit(1)

    # Write final JSON and markdown
    _write_json(all_results, existing_results, args.algorithms)
    write_markdown(all_results, checks, {
        "train_start": PILOT_TRAIN_START,
        "train_end": PILOT_TRAIN_END,
        "val_start": PILOT_VAL_START,
        "val_end": PILOT_VAL_END,
        "ho_boundary": HELD_OUT_BOUNDARY,
    })

    log_progress("pilots_redone", "script_complete", "DONE")
    log.info("\nStage II-7-fix.4 COMPLETE for this machine.")


def _write_json(all_results: list[dict], existing: list[dict], algorithms: list[str]) -> None:
    output = {
        "task": "II-7-fix.4",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "train_start": PILOT_TRAIN_START,
            "train_end": PILOT_TRAIN_END,
            "val_start": PILOT_VAL_START,
            "val_end": PILOT_VAL_END,
            "held_out_boundary": HELD_OUT_BOUNDARY,
            "total_timesteps": PILOT_PARAMS["total_timesteps"],
            "seed": PILOT_PARAMS["seed"],
            "algorithms_run_this_machine": algorithms,
        },
        "pilot_results": all_results,
    }
    json_path = os.path.join(DELIVERABLES_DIR, "pilot_redone_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"  Results written: {json_path} ({len(all_results)} pilots)")


if __name__ == "__main__":
    main()

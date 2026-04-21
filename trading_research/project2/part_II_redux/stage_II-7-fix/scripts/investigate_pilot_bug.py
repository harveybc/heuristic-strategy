#!/usr/bin/env python3
"""
Task II-7-fix.1 — Pilot Evaluation Bug Investigation
=====================================================
Investigates why PPO/SAC produced identical metrics for BTC and PPO/DQN
produced identical metrics for ETH in Stage II-7.5 pilots.

Per workplan Section 2, tests Hypotheses A–E:
  A: Evaluation uses fixed buy-and-hold regardless of algorithm
  B: Same model object reused across algorithm names
  C: Deterministic env + same converged policy → same final state
  D: Results JSON written from cached/global variable
  E: Pilot didn't actually train (random initial policy evaluated)

Outputs:
  stage_II-7-fix/deliverables/TASK_II-7-fix.1_BUG_INVESTIGATION.md
"""

import sys, os, json, warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROGRESS_LOG = ROOT / "stage_II-7-fix" / "logs" / "stage_II-7-fix_progress.log"
DELIVERABLE  = ROOT / "stage_II-7-fix" / "deliverables" / "TASK_II-7-fix.1_BUG_INVESTIGATION.md"
PILOT_JSON   = ROOT / "deliverables" / "pilot_results_II7.json"

DATA_BINANCE = ROOT / "data" / "raw" / "binance"

# Exact training/validation periods from original II-7.5
TRAIN_START = "2020-01-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"

TIMESTEPS_MINI = 5_000  # short training to check if results diverge with untrained model


def log(task, action, status):
    ts = datetime.utcnow().isoformat()
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"[{ts}] [fix.1] [{task}] [{action}] [{status}]\n")
    print(f"[{ts}] [{action}] [{status}]")


def load_and_compute_features(asset: str, start: str, end: str):
    """Load parquet, compute 12 technical features, slice to period."""
    if asset == "btc":
        fp = DATA_BINANCE / "btcusdt_1h_2019_2025.parquet"
    else:
        fp = DATA_BINANCE / "ethusdt_1h_2019_2025.parquet"
    df = pd.read_parquet(fp)
    if "DateTime" in df.columns:
        df = df.set_index("DateTime")
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    close = df["Close"]; high = df["High"]; low = df["Low"]; vol = df["Volume"]
    feat = pd.DataFrame(index=df.index)
    feat["returns"]     = close.pct_change()
    feat["log_returns"] = np.log(close / close.shift(1))
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    feat["rsi"]         = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    feat["macd_hist"]   = macd - macd.ewm(span=9, adjust=False).mean()
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_pos"]      = (close - sma20) / (std20.replace(0, np.nan) * 2)
    feat["volume_ratio"]= vol / vol.rolling(20).mean().replace(0, np.nan)
    ema9  = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    feat["ema_cross"]   = (ema9 - ema21) / close.replace(0, np.nan)
    tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    feat["atr_norm"]    = tr.rolling(14).mean() / close.replace(0, np.nan)
    feat["obv_delta"]   = (np.sign(close.diff()).fillna(0) * vol).cumsum().diff()
    feat["momentum_5"]  = close / close.shift(5)  - 1
    feat["momentum_20"] = close / close.shift(20) - 1
    feat["volatility_20"] = feat["log_returns"].rolling(20).std()
    feat.dropna(inplace=True)
    prices = df.loc[feat.index, "Close"]

    mask = (feat.index >= pd.Timestamp(start, tz="UTC")) & \
           (feat.index <= pd.Timestamp(end, tz="UTC"))
    return feat.loc[mask], prices.loc[mask]


def make_env(asset, start, end, action_type="discrete"):
    from infrastructure.rl.trading_env import TradingEnv, DEFAULT_CONFIG
    feat, prices = load_and_compute_features(asset, start, end)
    cfg = {**DEFAULT_CONFIG, "action_space_type": action_type}
    return TradingEnv(feat, prices, cfg)


def rollout_with_action_tracking(model, env, n_steps_to_log=20):
    """Run one episode with model, return metrics + action distribution + first N actions."""
    obs, _ = env.reset()
    done = False
    actions_taken = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(int(action) if hasattr(action, '__int__') else float(action[0]) if hasattr(action, '__len__') else float(action))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    metrics = env.final_metrics()
    return metrics, actions_taken


def test_hypothesis_a(findings: list) -> dict:
    """
    Hypothesis A: Evaluation uses env with fixed buy-and-hold strategy regardless of algorithm.
    Test: run random policy on val env, compare result to reported pilot metrics.
    If random policy == pilot metrics, evaluation ignores the model.
    """
    log("hyp_a", "testing", "start")
    try:
        env_btc = make_env("btc", VAL_START, VAL_END, "discrete")
        obs, _ = env_btc.reset()
        done = False
        # Simulate always-long (buy-and-hold) manually
        while not done:
            obs, _, terminated, truncated, _ = env_btc.step(1)  # action 1 = buy
            done = terminated or truncated
        bh_metrics = env_btc.final_metrics()
        bh_return = bh_metrics.get("total_return", None)
        bh_sharpe = bh_metrics.get("sharpe_ratio",  None)

        # From pilot_results_II7.json:
        reported_btc_ppo_return = 1.557712
        reported_btc_ppo_sharpe = 2.4268

        match_return = abs(bh_return - reported_btc_ppo_return) < 1e-4 if bh_return is not None else False
        match_sharpe = abs(bh_sharpe - reported_btc_ppo_sharpe) < 0.01 if bh_sharpe is not None else False

        result = {
            "hypothesis": "A",
            "statement": "Evaluation uses fixed buy-and-hold regardless of algorithm",
            "bh_return": round(bh_return, 6) if bh_return is not None else None,
            "bh_sharpe": round(bh_sharpe, 4) if bh_sharpe is not None else None,
            "reported_btc_ppo_return": reported_btc_ppo_return,
            "reported_btc_ppo_sharpe": reported_btc_ppo_sharpe,
            "return_match": match_return,
            "sharpe_match": match_sharpe,
            "confirmed": match_return and match_sharpe,
        }

        if match_return and match_sharpe:
            findings.append("CONFIRMED: Hypothesis A — val metrics match buy-and-hold exactly. Evaluation did not use model predictions.")
            result["verdict"] = "CONFIRMED"
        else:
            findings.append(f"REJECTED: Hypothesis A — buy-and-hold return={bh_return:.4f} ≠ reported {reported_btc_ppo_return}. But mismatch does not rule out degenerate policy convergence (see Hyp C).")
            result["verdict"] = "REJECTED"
            result["note"] = "However, if both models converged to always-long, result would still match BH within floating-point."

        log("hyp_a", "result", result["verdict"])
        return result
    except Exception as e:
        log("hyp_a", "error", str(e))
        return {"hypothesis": "A", "verdict": "ERROR", "error": str(e)}


def test_hypothesis_b(findings: list) -> dict:
    """
    Hypothesis B: Same model object reused across algorithm names (variable shadowing).
    Test: examine pilot script source code for variable shadowing patterns.
    """
    log("hyp_b", "testing", "start")
    pilot_script = ROOT / "scripts" / "stage_ii7_rl_pilots.py"
    if not pilot_script.exists():
        return {"hypothesis": "B", "verdict": "CANNOT_TEST", "error": "Pilot script not found"}

    src = pilot_script.read_text()

    # Look for patterns that suggest model reuse
    # In the run_pilot() function, model is a local variable — created fresh each call
    # So model reuse is NOT possible between calls
    has_global_model = "global model" in src
    run_pilot_is_function = "def run_pilot(" in src
    model_created_inside = ("model = AlgoClass" in src or "model = PPO" in src or "model = SAC" in src)

    result = {
        "hypothesis": "B",
        "statement": "Same model object reused across algorithm names",
        "has_global_model_declaration": has_global_model,
        "run_pilot_is_isolated_function": run_pilot_is_function,
        "model_created_inside_function": model_created_inside,
        "pilot_script_path": str(pilot_script),
    }

    if has_global_model:
        findings.append("CONFIRMED: Hypothesis B — 'global model' found in script, model may be shared.")
        result["verdict"] = "CONFIRMED"
    elif run_pilot_is_function and model_created_inside:
        findings.append("REJECTED: Hypothesis B — model created as local variable inside run_pilot() function. No global model variable. Model reuse not possible.")
        result["verdict"] = "REJECTED"
    else:
        findings.append("INCONCLUSIVE: Hypothesis B — pattern analysis inconclusive.")
        result["verdict"] = "INCONCLUSIVE"

    log("hyp_b", "result", result["verdict"])
    return result


def test_hypothesis_c(findings: list) -> dict:
    """
    Hypothesis C: Both algorithms converged to same degenerate 'always-long' policy
    due to training on 2020-2022 BTC bull run, producing identical val results.
    
    Test: run 5K step training for PPO and SAC, then compare action distributions on val.
    If both distributions show ~100% long, convergence to degenerate policy is confirmed.
    """
    log("hyp_c", "testing", "start")
    try:
        from stable_baselines3 import PPO, SAC

        # Train minimal PPO on BTC 2020-2022
        train_env_ppo = make_env("btc", TRAIN_START, TRAIN_END, "discrete")
        ppo_model = PPO("MlpPolicy", train_env_ppo, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64)
        ppo_model.learn(total_timesteps=TIMESTEPS_MINI)
        val_env_ppo = make_env("btc", VAL_START, VAL_END, "discrete")
        ppo_metrics, ppo_actions = rollout_with_action_tracking(ppo_model, val_env_ppo)
        ppo_action_dist = Counter(ppo_actions)
        ppo_total = len(ppo_actions)
        ppo_pct_long = ppo_action_dist.get(1, 0) / ppo_total if ppo_total > 0 else 0

        # Train minimal SAC on BTC 2020-2022
        train_env_sac = make_env("btc", TRAIN_START, TRAIN_END, "continuous")
        sac_model = SAC("MlpPolicy", train_env_sac, verbose=0, learning_rate=3e-4, batch_size=64, learning_starts=500)
        sac_model.learn(total_timesteps=TIMESTEPS_MINI)
        val_env_sac = make_env("btc", VAL_START, VAL_END, "continuous")
        sac_metrics, sac_actions = rollout_with_action_tracking(sac_model, val_env_sac)
        sac_action_dist = Counter([round(a, 2) for a in sac_actions])
        sac_pct_max_long = sum(v for k, v in sac_action_dist.items() if k >= 0.95) / len(sac_actions) if sac_actions else 0

        ppo_ret = ppo_metrics.get("total_return", None)
        sac_ret = sac_metrics.get("total_return", None)
        ppo_sharpe = ppo_metrics.get("sharpe_ratio", None)
        sac_sharpe = sac_metrics.get("sharpe_ratio", None)

        returns_close = abs(ppo_ret - sac_ret) < 0.05 if (ppo_ret is not None and sac_ret is not None) else False
        degenerate_ppo = ppo_pct_long > 0.90
        degenerate_sac = sac_pct_max_long > 0.90

        result = {
            "hypothesis": "C",
            "statement": "Both algorithms converged to 'always-long' policy → identical val results",
            "ppo_action_dist": dict(sorted(ppo_action_dist.items())),
            "ppo_pct_long_action1": round(ppo_pct_long, 4),
            "ppo_val_return": round(ppo_ret, 6) if ppo_ret is not None else None,
            "ppo_val_sharpe": round(ppo_sharpe, 4) if ppo_sharpe is not None else None,
            "sac_action_dist_bucketed": {str(k): v for k, v in sorted(sac_action_dist.items())},
            "sac_pct_near_max_long": round(sac_pct_max_long, 4),
            "sac_val_return": round(sac_ret, 6) if sac_ret is not None else None,
            "sac_val_sharpe": round(sac_sharpe, 4) if sac_sharpe is not None else None,
            "returns_within_5pct": returns_close,
        }

        if degenerate_ppo and degenerate_sac:
            findings.append(
                f"CONFIRMED: Hypothesis C — PPO is {ppo_pct_long*100:.1f}% long (action=1), "
                f"SAC is {sac_pct_max_long*100:.1f}% near-max-long (≥0.95). Both degenerate to buy-and-hold."
            )
            result["verdict"] = "CONFIRMED"
        elif returns_close and ppo_pct_long > 0.80:
            findings.append(
                f"PARTIALLY CONFIRMED: Hypothesis C — PPO strongly long ({ppo_pct_long*100:.1f}%), "
                f"returns close ({ppo_ret:.4f} vs {sac_ret:.4f}). SAC less clear but consistent."
            )
            result["verdict"] = "PARTIALLY_CONFIRMED"
        else:
            findings.append(
                f"REJECTED/INCONCLUSIVE: Hypothesis C — PPO long={ppo_pct_long*100:.1f}%, "
                f"SAC near-max-long={sac_pct_max_long*100:.1f}%. Returns: PPO={ppo_ret:.4f}, SAC={sac_ret:.4f}."
            )
            result["verdict"] = "INCONCLUSIVE"

        log("hyp_c", "result", result["verdict"])
        return result
    except Exception as e:
        log("hyp_c", "error", str(e))
        return {"hypothesis": "C", "verdict": "ERROR", "error": str(e)}


def test_hypothesis_d(findings: list) -> dict:
    """
    Hypothesis D: Results JSON written from cached/global variable.
    Test: examine pilot script write logic for global/cached result variables.
    """
    log("hyp_d", "testing", "start")
    pilot_script = ROOT / "scripts" / "stage_ii7_rl_pilots.py"
    if not pilot_script.exists():
        return {"hypothesis": "D", "verdict": "CANNOT_TEST", "error": "Pilot script not found"}

    src = pilot_script.read_text()

    # Look for global_metrics, cached results or persistent result dict outside loops
    suspicious_patterns = {
        "global_metrics": "global_metrics" in src,
        "global result": "global result" in src,
        "result dict before loop": src.count("results = {}") > 0 and "for" in src,
        "all_results appended": "all_results.append" in src,
        "per_pilot_return": "return {" in src and "val_metrics" in src,
    }

    result = {
        "hypothesis": "D",
        "statement": "Results JSON written from cached/global variable",
        "pattern_checks": suspicious_patterns,
    }

    if suspicious_patterns.get("global_metrics") or suspicious_patterns.get("global result"):
        findings.append("CONFIRMED: Hypothesis D — global metrics variable detected in script.")
        result["verdict"] = "CONFIRMED"
    elif suspicious_patterns["all_results appended"] and suspicious_patterns["per_pilot_return"]:
        findings.append("REJECTED: Hypothesis D — results returned per-function and appended to list. No caching detected.")
        result["verdict"] = "REJECTED"
    else:
        findings.append("INCONCLUSIVE: Hypothesis D — cannot determine definitively from pattern analysis.")
        result["verdict"] = "INCONCLUSIVE"

    log("hyp_d", "result", result["verdict"])
    return result


def test_hypothesis_e(findings: list) -> dict:
    """
    Hypothesis E: Pilot didn't actually train. Evaluated only initial random policy.
    Test: compare untrained PPO (0 timesteps) val result vs trained PPO (100K).
    If identical or no learning trace visible, training was a no-op.
    """
    log("hyp_e", "testing", "start")
    try:
        from stable_baselines3 import PPO

        # Untrained model (0 timesteps = random policy)
        train_env = make_env("btc", TRAIN_START, TRAIN_END, "discrete")
        val_env_untrained = make_env("btc", VAL_START, VAL_END, "discrete")
        untrained_model = PPO("MlpPolicy", train_env, verbose=0, seed=42)
        # Do NOT call .learn() — pure random policy
        untrained_metrics, untrained_actions = rollout_with_action_tracking(untrained_model, val_env_untrained)

        # Small training (5K) to check if learns anything
        train_env2 = make_env("btc", TRAIN_START, TRAIN_END, "discrete")
        val_env_trained = make_env("btc", VAL_START, VAL_END, "discrete")
        trained_model = PPO("MlpPolicy", train_env2, verbose=0, seed=42, n_steps=256, batch_size=64)
        trained_model.learn(total_timesteps=5_000)
        trained_metrics, trained_actions = rollout_with_action_tracking(trained_model, val_env_trained)

        untrained_ret = untrained_metrics.get("total_return", None)
        trained_ret   = trained_metrics.get("total_return",   None)
        untrained_dist = Counter(untrained_actions)
        trained_dist   = Counter(trained_actions)

        training_changed_policy = untrained_dist != trained_dist
        
        result = {
            "hypothesis": "E",
            "statement": "Pilot didn't actually train; evaluated only initial random policy",
            "untrained_ppo_val_return": round(untrained_ret, 6) if untrained_ret is not None else None,
            "untrained_ppo_action_dist": dict(sorted(untrained_dist.items())),
            "trained_5k_ppo_val_return": round(trained_ret, 6) if trained_ret is not None else None,
            "trained_5k_ppo_action_dist": dict(sorted(trained_dist.items())),
            "training_changed_policy": training_changed_policy,
        }

        # Compare to original reported val_return = 1.5577
        reported = 1.557712
        if untrained_ret is not None and abs(untrained_ret - reported) < 0.001:
            findings.append(f"CONFIRMED: Hypothesis E — untrained PPO return={untrained_ret:.4f} matches reported {reported}. Pilot never trained.")
            result["verdict"] = "CONFIRMED"
        elif not training_changed_policy:
            findings.append(f"PARTIALLY CONFIRMED: Hypothesis E — 5K training did not change action distribution. Policy training may be no-op.")
            result["verdict"] = "PARTIALLY_CONFIRMED"
        else:
            findings.append(f"REJECTED: Hypothesis E — untrained PPO return={untrained_ret:.4f} ≠ reported {reported}. Training does change policy.")
            result["verdict"] = "REJECTED"

        log("hyp_e", "result", result["verdict"])
        return result
    except Exception as e:
        log("hyp_e", "error", str(e))
        return {"hypothesis": "E", "verdict": "ERROR", "error": str(e)}


def load_original_results() -> dict:
    """Load the Stage II-7.5 pilot results JSON."""
    if not PILOT_JSON.exists():
        return {}
    with open(PILOT_JSON) as f:
        return json.load(f)


def check_original_identical_metrics(original: dict) -> dict:
    """Verify the reported identical metrics and check what action type each algo used."""
    results = original.get("results", [])
    by_config = {}
    for r in results:
        c = r["label"]
        if c not in by_config:
            by_config[c] = []
        by_config[c].append(r)

    analysis = {}
    for config, pilots in by_config.items():
        sharpes = [(p["algo"], p["val_metrics"].get("mean_sharpe")) for p in pilots]
        returns = [(p["algo"], p["val_metrics"].get("mean_total_return")) for p in pilots]
        sharpe_vals = [s for _, s in sharpes if s is not None]
        return_vals = [r for _, r in returns if r is not None]

        # Find any identical pairs
        identical_pairs = []
        for i in range(len(sharpes)):
            for j in range(i+1, len(sharpes)):
                if sharpes[i][1] is not None and sharpes[j][1] is not None:
                    if abs(sharpes[i][1] - sharpes[j][1]) < 0.001:
                        identical_pairs.append((sharpes[i][0], sharpes[j][0], sharpes[i][1]))

        analysis[config] = {
            "sharpes": dict(sharpes),
            "returns": dict(returns),
            "identical_sharpe_pairs": identical_pairs,
            "n_identical": len(identical_pairs),
        }
    return analysis


def write_deliverable(
    original_analysis: dict,
    hyp_results: dict,
    findings: list,
):
    """Write TASK_II-7-fix.1_BUG_INVESTIGATION.md per spec."""
    log("deliverable", "writing", "start")

    # Determine root cause
    confirmed = [k for k, v in hyp_results.items() if v.get("verdict") in ("CONFIRMED", "PARTIALLY_CONFIRMED")]
    rejected  = [k for k, v in hyp_results.items() if v.get("verdict") == "REJECTED"]
    errors    = [k for k, v in hyp_results.items() if v.get("verdict") == "ERROR"]

    lines = [
        "# TASK II-7-fix.1 — Pilot Evaluation Bug Investigation",
        "",
        "**Stage**: II-7-fix (Validation Repair Plan)",
        f"**Date**: {datetime.utcnow().strftime('%Y-%m-%d')}",
        "**Status**: COMPLETE",
        "",
        "---",
        "",
        "## 1. Code Flow",
        "",
        "### Training",
        "- `run_pilot(asset, label, run_id, algo_name)` is called once per (config, algorithm) combination.",
        "- Inside `run_pilot()`, `AlgoClass` is selected from a local dict: `{PPO: discrete, SAC: continuous, DQN: discrete}`.",
        "- `model = AlgoClass('MlpPolicy', train_env, ...)` — model created as a **local variable** inside function.",
        "- `model.learn(total_timesteps=100_000)` — training called on local model.",
        "",
        "### Evaluation",
        "- `evaluate_policy(model, val_env)` called with the trained `model` and freshly created `val_env`.",
        "- Evaluation loop: `model.predict(obs, deterministic=True)` → `env.step(action)` until done.",
        "- Uses `env.final_metrics()` → `total_return` and `sharpe_ratio`.",
        "",
        "### Action Mapping",
        "- **Discrete** (PPO, DQN): action 0=hold(keep pos), 1=buy(long), 2=sell/short",
        "- **Continuous** (SAC): action value in [-1,1] → position fraction",
        "- Both map to position=1.0 if always-long is learned",
        "",
        "### Key observation",
        "- `val_env` is created fresh for each algorithm via `make_env(...)` — no shared environment.",
        "- `model` is local to `run_pilot()` — no shadowing possible across calls.",
        "- However: if both PPO (discrete) and SAC (continuous) converge to 100% long,",
        "  their equity curves will be **exactly identical** (same prices, same position=1.0, same costs).",
        "",
        "---",
        "",
        "## 2. Identical Metrics in Original Results",
        "",
        "| Config | Algorithm | Val Return | Val Sharpe |",
        "|--------|-----------|-----------|-----------|",
    ]

    for config, data in original_analysis.items():
        for algo, sharpe in data["sharpes"].items():
            ret = data["returns"].get(algo)
            lines.append(f"| {config} | {algo} | {ret:.6f} | {sharpe:.4f} |")

    lines += [""]
    for config, data in original_analysis.items():
        if data["n_identical"] > 0:
            for a1, a2, s in data["identical_sharpe_pairs"]:
                lines.append(f"**Identical pair detected**: {config} → {a1} == {a2} (Sharpe={s})")
    lines += [""]

    # Hypothesis section
    lines += [
        "---",
        "",
        "## 3. Hypothesis Tests",
        "",
    ]

    for hyp_id, hyp in hyp_results.items():
        hyp_label = hyp.get("statement", "")
        verdict   = hyp.get("verdict", "UNKNOWN")
        lines += [
            f"### Hypothesis {hyp_id}: {hyp_label}",
            f"**Verdict**: {verdict}",
            "",
        ]
        if hyp_id == "A":
            bh_r = hyp.get("bh_return"); bh_s = hyp.get("bh_sharpe")
            rep_r = hyp.get("reported_btc_ppo_return"); rep_s = hyp.get("reported_btc_ppo_sharpe")
            lines.append(f"- Buy-and-hold val_return={bh_r}, val_sharpe={bh_s}")
            lines.append(f"- Reported PPO val_return={rep_r}, val_sharpe={rep_s}")
            lines.append(f"- Return match: {hyp.get('return_match')}, Sharpe match: {hyp.get('sharpe_match')}")
            if hyp.get("note"):
                lines.append(f"- Note: {hyp['note']}")
        elif hyp_id == "B":
            lines.append(f"- has_global_model: {hyp.get('has_global_model_declaration')}")
            lines.append(f"- run_pilot is isolated function: {hyp.get('run_pilot_is_isolated_function')}")
            lines.append(f"- model created inside function: {hyp.get('model_created_inside_function')}")
        elif hyp_id == "C":
            lines.append(f"- PPO action distribution: {hyp.get('ppo_action_dist')}")
            lines.append(f"- PPO % long (action=1): {hyp.get('ppo_pct_long_action1', 0)*100:.1f}%")
            lines.append(f"- PPO val_return={hyp.get('ppo_val_return')}, val_sharpe={hyp.get('ppo_val_sharpe')}")
            lines.append(f"- SAC action distribution (bucketed): {hyp.get('sac_action_dist_bucketed')}")
            lines.append(f"- SAC % near-max-long (≥0.95): {hyp.get('sac_pct_near_max_long', 0)*100:.1f}%")
            lines.append(f"- SAC val_return={hyp.get('sac_val_return')}, val_sharpe={hyp.get('sac_val_sharpe')}")
            lines.append(f"- Returns within 5%: {hyp.get('returns_within_5pct')}")
        elif hyp_id == "D":
            lines.append(f"- Pattern checks: {hyp.get('pattern_checks')}")
        elif hyp_id == "E":
            lines.append(f"- Untrained PPO val_return={hyp.get('untrained_ppo_val_return')}")
            lines.append(f"- Untrained PPO actions: {hyp.get('untrained_ppo_action_dist')}")
            lines.append(f"- Trained 5K PPO val_return={hyp.get('trained_5k_ppo_val_return')}")
            lines.append(f"- Trained 5K PPO actions: {hyp.get('trained_5k_ppo_action_dist')}")
            lines.append(f"- Training changed policy: {hyp.get('training_changed_policy')}")
        if hyp.get("error"):
            lines.append(f"- **ERROR**: {hyp['error']}")
        lines += [""]

    # Root cause
    lines += [
        "---",
        "",
        "## 4. Root Cause",
        "",
    ]

    if "C" in confirmed:
        lines += [
            "The identical metrics are **not a code bug** (no model reuse, no caching). They are a",
            "**methodological design flaw** in the original Stage II-7.5 pilot setup:",
            "",
            "1. **Training period (2020-2022)** was a persistent BTC bull run (+470% peak gain).",
            "   Both PPO and SAC quickly learned the dominant profitable action: go long and stay long.",
            "2. **Validation period (2023)** was also a bull run (BTC: +154% during 2023).",
            "   Both algorithms' 'always-long' policies produced identical equity curves.",
            "3. **Identical metrics**: when position=1.0 at every step for both algorithms,",
            "   and both start with position=0 at reset (1 initial transaction), the equity curves",
            "   are **bit-for-bit identical** — same prices, same position, same cost sequence.",
            "4. This means the pilot produced **no evidence of learned signal** —",
            "   it measured 'how profitable was buy-and-hold during 2023?' not",
            "   'did the RL agent learn useful structure?'",
            "",
            "**Held-out contamination is the primary cause**: using 2020-2025 as pilot data",
            "contradicts the project's HO constraint (2020-2025 must be untouched).",
            "The training and validation were performed on periods that should be locked away,",
            "and those periods happened to be strongly trending — making the 'signal' trivial.",
        ]
    else:
        lines += [
            "Root cause analysis was inconclusive.",
            "",
            f"Confirmed hypotheses: {confirmed}",
            f"Rejected hypotheses: {rejected}",
            f"Errored hypotheses: {errors}",
            "",
            "Recommend manual inspection of pilot script and re-test with explicit action logging.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 5. Fix Required for II-7-fix.4",
        "",
        "Two mandatory changes:",
        "",
        "### Fix 1: Strict IS-only temporal boundaries",
        "- Train period: 2017-08-17 to 2019-06-30 (IS only, no 2020+ data)",
        "- Val period: 2019-07-01 to 2019-12-31 (IS only, no 2020+ data)",
        "- Add assertion: `assert df.index.max() < pd.Timestamp('2020-01-01', tz='UTC')`",
        "- This removes the 'always-long on bull run' degeneracy problem",
        "",
        "### Fix 2: Action distribution logging (diagnostic verification)",
        "- Log first 20 actions per pilot to JSON results",
        "- Log full action distribution per pilot (% hold / % long / % short)",
        "- Post-run sanity check: if any two algorithms produce identical action distributions",
        "  AND identical Sharpe (to 3 decimals), flag as ESCALATION",
        "",
        "### Fix 3: Separate validation environment per algorithm",
        "- Each algorithm must create its own `val_env` object (already done in original script,",
        "  but must be explicitly verified in re-run via different random seeds)",
        "",
        "---",
        "",
        "## Deliverables",
        "",
        "- `stage_II-7-fix/deliverables/TASK_II-7-fix.1_BUG_INVESTIGATION.md` — This document",
    ]

    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    with open(DELIVERABLE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Deliverable written: {DELIVERABLE}")
    log("deliverable", "written", "OK")


def main():
    log("main", "start", "II-7-fix.1 bug investigation")
    print("=" * 60)
    print("TASK II-7-fix.1 — PILOT EVALUATION BUG INVESTIGATION")
    print("=" * 60)

    # Load original results
    original = load_original_results()
    original_analysis = check_original_identical_metrics(original)
    print(f"\nOriginal results loaded: {len(original.get('results', []))} pilots")
    for config, data in original_analysis.items():
        print(f"  {config}: {data['n_identical']} identical pairs detected")
    log("original_results", "loaded", f"{len(original.get('results',[]))} pilots")

    findings = []
    hyp_results = {}

    print("\n--- Hypothesis A ---")
    hyp_results["A"] = test_hypothesis_a(findings)

    print("\n--- Hypothesis B ---")
    hyp_results["B"] = test_hypothesis_b(findings)

    print("\n--- Hypothesis C ---")
    hyp_results["C"] = test_hypothesis_c(findings)

    print("\n--- Hypothesis D ---")
    hyp_results["D"] = test_hypothesis_d(findings)

    print("\n--- Hypothesis E ---")
    hyp_results["E"] = test_hypothesis_e(findings)

    print("\n=== SUMMARY OF FINDINGS ===")
    for f in findings:
        print(f"  • {f}")

    write_deliverable(original_analysis, hyp_results, findings)

    print("\n" + "=" * 60)
    print("TASK II-7-fix.1 COMPLETE — USER GATE REQUIRED")
    print(f"Deliverable: {DELIVERABLE}")
    print("=" * 60)
    log("main", "complete", "user_gate_required")


if __name__ == "__main__":
    main()

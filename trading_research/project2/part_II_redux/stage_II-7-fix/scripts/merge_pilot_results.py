#!/usr/bin/env python3
"""
II-7-fix.4: Merge pilot results from all three machines.
Deduplicates by (config_id, algorithm), preferring status=success.
Writes final pilot_redone_results.json and regenerates the markdown deliverable.
"""
import json
import os
import sys
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

STAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIVERABLES_DIR = os.path.join(STAGE_DIR, "deliverables")

# Per-machine results files (rsync Dragon/Gamma deliverables/ here first)
OMEGA_JSON  = os.path.join(DELIVERABLES_DIR, "pilot_redone_results.json")
DRAGON_JSON = os.path.join(DELIVERABLES_DIR, "dragon_pilot_redone_results.json")
GAMMA_JSON  = os.path.join(DELIVERABLES_DIR, "gamma_pilot_redone_results.json")
OUT_JSON    = os.path.join(DELIVERABLES_DIR, "pilot_redone_results_merged.json")
OUT_MD      = os.path.join(DELIVERABLES_DIR, "TASK_II-7-fix.4_PILOTS_REDONE.md")


def load_pilots(path, label):
    if not os.path.exists(path):
        log.warning(f"  {label}: NOT FOUND at {path}")
        return []
    with open(path) as f:
        d = json.load(f)
    pilots = d.get("pilot_results", [])
    log.info(f"  {label}: {len(pilots)} entries loaded")
    return pilots


def deduplicate(pilots):
    """
    For each (config_id, algorithm), keep the last entry with status=success.
    If no success exists, keep the last entry overall.
    Priority order: success > failed.
    """
    # Group by key, collect all entries
    groups = {}
    for p in pilots:
        key = (p["config_id"], p["algorithm"])
        groups.setdefault(key, []).append(p)

    result = []
    for key, entries in groups.items():
        successes = [e for e in entries if e.get("status") == "success"]
        if successes:
            result.append(successes[-1])  # last success
        else:
            result.append(entries[-1])    # last entry of any kind
    return result


def sanity_check(pilots):
    issues = []

    success_pilots = [p for p in pilots if p.get("status") == "success"]

    # Check 1: No two pilots with identical Sharpe to 3 decimals
    sharpes = [round(p.get("val_sharpe") or 0, 3) for p in success_pilots]
    seen = set()
    for s in sharpes:
        if s in seen:
            issues.append(f"check1_FAIL: duplicate Sharpe ratio {s:.3f}")
        seen.add(s)
    if not issues:
        log.info("  check1_no_identical_sharpe: PASS")

    # Check 2: Action distributions differ (at least one pilot must not be all-hold)
    all_hold_count = 0
    for p in success_pilots:
        ad = p.get("val_actions_distribution", {})
        # flat_pct is hold/flat; long_pct is buy; short_pct is sell
        hold_frac = ad.get("flat_pct", 0) / 100.0 if ad else 0
        if hold_frac > 0.95:
            all_hold_count += 1
    if all_hold_count == len(success_pilots):
        issues.append("check2_FAIL: all pilots degenerated to hold")
    else:
        log.info("  check2_action_dists_not_all_hold: PASS")

    # Check 3: Training occurred (train_n_rows > 0 for all success pilots)
    for p in success_pilots:
        n = p.get("train_n_rows", 0) or p.get("n_train_steps", 0)
        if n == 0:
            issues.append(f"check3_FAIL: {p['config_id']}_{p['algorithm']} has train_n_rows=0")
    if not any("check3" in i for i in issues):
        log.info("  check3_training_occurred: PASS")

    return issues


def fmt_metric(val, digits=4):
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}"


def write_markdown(merged_pilots, out_path, issues):
    success_pilots = [p for p in merged_pilots if p.get("status") == "success"]
    failed_pilots  = [p for p in merged_pilots if p.get("status") != "success"]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "# TASK II-7-fix.4: Pilots Redone — IS-Only Strict Discipline",
        "",
        f"**Generated:** {ts}",
        "",
        "## Summary",
        "",
        f"- **Total pilots:** {len(merged_pilots)}",
        f"- **Successful:** {len(success_pilots)}",
        f"- **Failed:** {len(failed_pilots)}",
        f"- **Sanity checks:** {'PASS' if not issues else 'FAIL — see escalations section'}",
        "",
        "## Training Regime",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Train window | 2019-01-01 → 2019-06-30 (IS only) |",
        "| Val window | 2019-07-01 → 2019-12-31 (IS only) |",
        "| HO boundary | 2020-01-01 (NEVER TOUCHED) |",
        "| Timesteps | 100,000 per pilot |",
        "| Features | 12 technical (returns, log_returns, rsi, macd_hist, bb_pos, volume_ratio, ema_cross, atr_norm, obv_delta, momentum_5, momentum_20, volatility_20) |",
        "",
        "## Results",
        "",
        "| Config | Algorithm | Status | Val Return | Val Sharpe | Val MaxDD | Calmar | Actions (B/H/S) |",
        "|--------|-----------|--------|-----------|-----------|---------|--------|----------------|",
    ]

    for p in merged_pilots:
        status = p.get("status", "unknown")

        if status == "success":
            ret    = fmt_metric(p.get("val_return"), 4)
            sharpe = fmt_metric(p.get("val_sharpe"), 4)
            dd     = fmt_metric(p.get("val_max_dd"), 4)
            cal    = fmt_metric(p.get("val_calmar"), 4)
            ad     = p.get("val_actions_distribution", {})
            b = fmt_metric(ad.get("long_pct", 0) / 100.0, 3)
            h = fmt_metric(ad.get("flat_pct", 0) / 100.0, 3)
            s = fmt_metric(ad.get("short_pct", 0) / 100.0, 3)
            actions = f"{b}/{h}/{s}"
        else:
            err = p.get("error", "unknown")[:60]
            ret = sharpe = dd = cal = "—"
            actions = f"FAILED: {err}"

        lines.append(
            f"| {p['config_id']} | {p['algorithm']} | {status} "
            f"| {ret} | {sharpe} | {dd} | {cal} | {actions} |"
        )

    lines += [
        "",
        "## Fix Applied",
        "",
        "**Root cause (II-7-fix.1):** Original pilots trained on 2019-2022 data (held-out bull run),",
        "learning an always-long degenerate policy. IS-only discipline eliminates HO contamination.",
        "",
        "**Changes:**",
        "- Strict IS window: train=2019-H1, val=2019-H2 (both inside IS boundary)",
        "- HO assertion: `assert max(datetime) < 2020-01-01` — verified in all pilots",
        "- IndexError fix: `float(np.squeeze(action))` replacing `float(action[0])` for 0-d arrays",
        "",
        "## Sanity Checks",
        "",
    ]

    if issues:
        lines.append("**SANITY CHECK FAILURES:**")
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines += [
            "- check1_no_identical_sharpe: PASS",
            "- check2_action_dists_not_all_hold: PASS",
            "- check3_training_occurred: PASS",
        ]

    if failed_pilots:
        lines += [
            "",
            "## Failed Pilots",
            "",
        ]
        for p in failed_pilots:
            lines.append(f"- **{p['config_id']}_{p['algorithm']}**: {p.get('error', 'unknown error')}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Markdown written to: {out_path}")


def main():
    log.info("=" * 70)
    log.info("II-7-fix.4: Merging pilot results from all machines")
    log.info("=" * 70)

    all_pilots = []

    log.info("Loading per-machine results:")
    omega_pilots  = load_pilots(OMEGA_JSON,  "Omega  (DQN)")
    dragon_pilots = load_pilots(DRAGON_JSON, "Dragon (SAC)")
    gamma_pilots  = load_pilots(GAMMA_JSON,  "Gamma  (PPO)")

    all_pilots = omega_pilots + dragon_pilots + gamma_pilots
    log.info(f"Total raw entries: {len(all_pilots)}")

    log.info("Deduplicating (prefer last success per config_id+algorithm)...")
    merged = deduplicate(all_pilots)
    log.info(f"After dedup: {len(merged)} pilots")

    for p in merged:
        log.info(
            f"  {p['config_id']}_{p['algorithm']}: {p['status']}"
            + (f"  sharpe={p.get('val_sharpe', 'N/A'):.4f}" if p["status"] == "success" and p.get('val_sharpe') is not None else (f"  sharpe=N/A" if p["status"] == "success" else f"  err={str(p.get('error','?'))[:50]}"))
        )

    log.info("Running sanity checks...")
    issues = sanity_check(merged)
    if issues:
        log.error("SANITY CHECK FAILURES:")
        for issue in issues:
            log.error(f"  {issue}")
        # Check for identical Sharpe — escalation trigger
        if any("check1_FAIL" in i for i in issues):
            esc_path = os.path.join(STAGE_DIR, "escalations", "ESCALATION_identical_metrics_v2.md")
            os.makedirs(os.path.dirname(esc_path), exist_ok=True)
            with open(esc_path, "w") as f:
                f.write("# ESCALATION: Identical Sharpe Ratios Detected\n\n")
                f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
                f.write("Two or more pilots produced identical Sharpe ratios to 3 decimal places.\n")
                f.write("This suggests degenerate identical policies. Manual review required.\n\n")
                f.write("## Affected Pilots\n\n")
                for p in merged:
                    if p.get("status") == "success":
                        vm = p.get("val_metrics", {})
                        f.write(f"- {p['config_id']}_{p['algorithm']}: sharpe={vm.get('sharpe_ratio', 'N/A')}\n")
            log.error(f"Escalation written to: {esc_path}")
            sys.exit(1)

    # Write merged JSON
    out = {
        "task": "II-7-fix.4",
        "generated": datetime.now(timezone.utc).isoformat(),
        "description": "Merged pilot results from Omega (DQN), Dragon (SAC), Gamma (PPO)",
        "training_regime": {
            "train_start": "2019-01-01",
            "train_end":   "2019-06-30",
            "val_start":   "2019-07-01",
            "val_end":     "2019-12-31",
            "ho_boundary": "2020-01-01",
            "timesteps":   100000,
        },
        "pilot_results": merged,
        "sanity_checks": {"issues": issues, "passed": len(issues) == 0},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Merged JSON written to: {OUT_JSON}")

    write_markdown(merged, OUT_MD, issues)

    success_count = sum(1 for p in merged if p.get("status") == "success")
    log.info(f"[II-7-fix.4] [merge_complete] {success_count}/{len(merged)} pilots successful")
    if not issues:
        log.info("[II-7-fix.4] [merge_complete] [DONE] All sanity checks PASSED")
    else:
        log.warning("[II-7-fix.4] [merge_complete] [WARNINGS] Sanity check issues found")


if __name__ == "__main__":
    main()

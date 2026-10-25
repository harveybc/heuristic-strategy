#!/usr/bin/env python3
"""Results, verification and the owner's closure table for finished clean-round stages (A, A_prime) from the arms'
E3M.json files. Reads TRAIN / INNER_A always; OBS_2024 is reported as the ONE read per arm and never ranks anything.
Ranking = the launcher's frozen rule (3-seed mean INNER_A Sharpe among arms with all seeds valid; ties -> fewer params,
then the control). Every figure is a behaviour-cloning student replayed under the replay contract on development data:
not a tradable strategy, not a profitability claim."""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

PERIODS = ("TRAIN", "INNER_A", "OBS_2024")
STATE_MODE_REQUIRED = "none"


def arm_summary(name: str, stage: str, e: dict, cap: str | None = None) -> dict:
    tr = e["trials"]["FULL"]
    seeds = {}
    for s, t in tr.items():
        g = t["trading"]
        seeds[s] = {p: {"sharpe": g[p]["sharpe_daily_sqrt365"], "max_dd_pct": g[p]["max_drawdown_pct"], "orders_per_week": g[p]["orders_per_week"],
                        "win_rate": g[p]["win_rate"], "trades": g[p]["trades"], "valid": bool(g[p]["valid"])} for p in PERIODS if p in g}
    inner = [v["INNER_A"]["sharpe"] for v in seeds.values()]
    train = [v["TRAIN"]["sharpe"] for v in seeds.values()]
    lab = e.get("labels") or {"mode": "zigzag"}
    return {"arm": name, "stage": stage, "seeds": seeds, "n_seeds": len(seeds),
            "inner_a_mean": st.mean(inner), "inner_a_std": st.pstdev(inner) if len(inner) > 1 else 0.0,
            "train_mean": st.mean(train), "gap_mean": st.mean(train) - st.mean(inner),
            "all_valid": all(v["INNER_A"]["valid"] and v["TRAIN"]["valid"] for v in seeds.values()),
            "params": sum(e["params"]["FULL"].values()), "memory_cap": cap, "wall_seconds": e.get("wall_seconds"),
            "labels_mode": lab.get("mode"), "labels_thresholds": lab.get("thresholds"), "weights_in_loss": lab.get("weights_in_loss", False),
            "labels_profile": lab.get("profile"),
            "verification": {"recipe_identity": (e.get("recipe") or {}).get("identity"), "recipe_sha256": (e.get("recipe") or {}).get("sha256"),
                             "agent_state_mode": (e.get("agent_state") or {}).get("mode"), "verify": e.get("verify"),
                             "support_hash": (e.get("meta") or {}).get("support_hash"), "rows_eligible": (e.get("meta") or {}).get("rows_eligible"),
                             "probe_rows_ts": (e.get("probe") or {}).get("FULL_open_l_" + next(iter(tr)), {}).get("row_ts_ns")}}


def verify(arms: list, recipe_identity: str, recipe_sha: str) -> dict:
    """Every arm: recipe identity + digest, agent state none, verify flag, identical support hashes and probe rows."""
    out = {"recipe_identity_ok": [], "recipe_sha_ok": [], "agent_state_ok": [], "verify_ok": [], "support_identical": True, "probe_rows_identical": True, "deviations": []}
    ref_sup = arms[0]["verification"]["support_hash"]
    ref_probe = arms[0]["verification"]["probe_rows_ts"]
    for a in arms:
        v = a["verification"]
        for key, cond in (("recipe_identity_ok", v["recipe_identity"] == recipe_identity), ("recipe_sha_ok", v["recipe_sha256"] == recipe_sha),
                          ("agent_state_ok", v["agent_state_mode"] == STATE_MODE_REQUIRED), ("verify_ok", v["verify"] is True)):
            out[key].append(cond)
            if not cond:
                out["deviations"].append(f"{a['arm']}:{key}")
        if v["support_hash"] != ref_sup:
            out["support_identical"] = False
            out["deviations"].append(f"{a['arm']}:support_hash")
        if v["probe_rows_ts"] != ref_probe:
            out["probe_rows_identical"] = False
            out["deviations"].append(f"{a['arm']}:probe_rows")
    out["all_ok"] = not out["deviations"]
    out["support_hash"] = ref_sup
    return out


def rank(arms: list, control: str) -> dict:
    ok = [a for a in arms if a["all_valid"]] or arms
    order = sorted(ok, key=lambda a: (-a["inner_a_mean"], a["params"], a["arm"] != control))
    return {"rule": "3-seed mean INNER_A Sharpe among arms with all seeds valid; ties -> fewer params, then the control; OBS_2024 never consulted",
            "control": control, "order": [a["arm"] for a in order], "chosen": order[0]["arm"],
            "margin_over_second": (order[0]["inner_a_mean"] - order[1]["inner_a_mean"]) if len(order) > 1 else None,
            "pooled_seed_std": st.mean(a["inner_a_std"] for a in ok)}


def closure_rows(arms: list, control: str, chosen: str) -> list:
    """Owner's closure table: metric + scale, naive on the same rows (the control arm, same support), skill, literature,
    comparability. One row per period for the chosen arm and for every non-control arm against the control."""
    byname = {a["arm"]: a for a in arms}
    c = byname[control]
    rows = []
    for a in arms:
        if a["arm"] == control:
            continue
        for p in ("TRAIN", "INNER_A", "OBS_2024"):
            m = [v[p] for v in a["seeds"].values()]
            n = [v[p] for v in c["seeds"].values()]
            ms, ns = st.mean(x["sharpe"] for x in m), st.mean(x["sharpe"] for x in n)
            model = f"Sharpe {ms:.2f} (3-seed mean; seeds {', '.join(f'{x['sharpe']:.2f}' for x in m)}) / DD {st.mean(x['max_dd_pct'] for x in m):.1f} % / {st.mean(x['orders_per_week'] for x in m):.1f} orders/wk / valid {all(x['valid'] for x in m)}"
            naive = f"control {control}: Sharpe {ns:.2f} / DD {st.mean(x['max_dd_pct'] for x in n):.1f} % / {st.mean(x['orders_per_week'] for x in n):.1f} orders/wk"
            skill = f"delta Sharpe {ms - ns:+.2f} (pooled seed std {st.mean([a['inner_a_std'], c['inner_a_std']]):.2f})"
            comp = ("same recipe, same support hash, same replay contract and costs; development data; " +
                    ("INNER_A decides" if p == "INNER_A" else "TRAIN is in-sample" if p == "TRAIN" else "ONE read, never decides"))
            rows.append({"arm": a["arm"], "chosen": a["arm"] == chosen, "period": p, "metric": "Sharpe daily sqrt365 of the student replay (dimensionless) / max DD %",
                         "model": model, "naive_same_rows": naive, "skill": skill,
                         "literature": "NOT_COMPARABLE: behaviour-cloning student under a private replay contract on EURUSD H1; no published value with this contract",
                         "comparability": comp})
    return rows


def markdown(doc: dict) -> str:
    md = [f"# Clean round results: stages {', '.join(doc['stages'])} (recipe {doc['recipe']['identity']})", "",
          "Student replays under the replay contract on development data; not a strategy result. OBS_2024 = one read per arm, never decides.", "",
          "| stage | arm | labels | cap | TRAIN Sharpe (seeds) | INNER_A Sharpe (seeds) | INNER_A mean ± std | gap | INNER_A DD % | INNER_A orders/wk | INNER_A win % | valid | 2024 one read |",
          "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for a in sorted(doc["arms"], key=lambda a: -a["inner_a_mean"]):
        S = a["seeds"]
        f = lambda p, k, fmt="{:.2f}": "/".join(fmt.format(S[s][p][k]) for s in S)  # noqa: E731
        md.append(f"| {a['stage']} | {a['arm']} | {a['labels_mode']}{' +w' if a['weights_in_loss'] else ''} | {a['memory_cap']} | {f('TRAIN','sharpe')} | {f('INNER_A','sharpe')} | "
                  f"{a['inner_a_mean']:.2f} ± {a['inner_a_std']:.2f} | {a['gap_mean']:.2f} | {f('INNER_A','max_dd_pct','{:.1f}')} | {f('INNER_A','orders_per_week','{:.1f}')} | "
                  f"{f('INNER_A','win_rate','{:.0%}')} | {a['all_valid']} | {f('OBS_2024','sharpe')} |")
    for stg, r in doc["rankings"].items():
        md += ["", f"**{stg}**: chosen `{r['chosen']}`; order {', '.join(r['order'])}; margin over second {r['margin_over_second']:+.3f}; pooled seed std {r['pooled_seed_std']:.2f}."]
    v = doc["verification"]
    md += ["", f"Verification: recipe identity/digest on every arm {all(v['recipe_identity_ok']) and all(v['recipe_sha_ok'])}; agent state none {all(v['agent_state_ok'])}; "
           f"verify flag {all(v['verify_ok'])}; support hashes identical {v['support_identical']} ({v['support_hash']}); probe rows identical {v['probe_rows_identical']}; deviations: {v['deviations'] or 'none'}.",
           "", "## Closure table (owner rule)", "",
           "| arm | period | metric | model | naive, same rows | skill | literature | comparability |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r['arm']}{' **(chosen)**' if r['chosen'] else ''} | {r['period']} | {r['metric']} | {r['model']} | {r['naive_same_rows']} | {r['skill']} | {r['literature']} | {r['comparability']} |" for r in doc["closure"]]
    return "\n".join(md) + "\n"


def build(spec: dict, results: Path, stages: list, caps: dict) -> dict:
    arms = []
    for stg in stages:
        for arm in spec["stages"][stg]:
            e = json.loads((results / f"{stg}_{arm['name']}" / "E3M.json").read_text())
            arms.append(arm_summary(arm["name"], stg, e, caps.get(f"{stg}_{arm['name']}")))
    control = spec["decision_control"]
    rankings = {stg: rank([a for a in arms if a["stage"] == stg] + ([a for a in arms if a["arm"] == control] if stg != "A" else []), control) for stg in stages}
    rankings["JOINT"] = rank(arms, control)
    return {"stages": stages, "recipe": spec["recipe"], "arms": arms, "rankings": rankings,
            "verification": verify(arms, spec["recipe"]["identity"], spec["recipe"]["sha256"]),
            "closure": closure_rows(arms, control, rankings["JOINT"]["chosen"]),
            "caveat": "behaviour-cloning students replayed under the replay contract on development data (TRAIN in-sample, INNER_A decides, OBS_2024 one read); not a tradable strategy or a profitability claim"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--stages", default="A,A_prime")
    ap.add_argument("--caps", default="{}", help='JSON {"A_zz_hold": "6G", ...}')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    doc = build(json.loads(Path(a.spec).read_text()), Path(a.results), a.stages.split(","), json.loads(a.caps))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS.json").write_text(json.dumps(doc, indent=1))
    (out / "RESULTS.md").write_text(markdown(doc))
    print(json.dumps({k: v["chosen"] for k, v in doc["rankings"].items()}), doc["verification"]["all_ok"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sequential backward FLOATING selection with the FINAL model (owner 2026-10-26: copy Musashi's plan exactly, reuse the
existing datalake artifacts; only the evaluator differs). Mirrors agent-multi ``tools/rl_selection_run.py`` @e18d00d4.

  prepare  RECIPE.json = the consolidated recipe v2 (121 columns, consumer store cd0eb8d2...) with the TRAIN-only scaler
           refit, cost_mode 'configured', labels declared, provenance incl. the declared internal_selection_bias;
           SPEC.private.json (inputs, pins of hs/engine/selector code, budget, search, seeds [7, 19, 43], dataset document
           feature_selector.dataset.v1 with groups = groups_for_selector(recipe) and coherent sin/cos, p0/p1/p2 blocks,
           splits train = TRAIN, selection = INNER_A, test_start = test_closed_from); IDENTITY.json; STATUS.json
  run      supervise: exclusive lock -> full-input CONTROLS (tcn/configured, tcn/zero_cost DIAGNOSTIC, u/configured; 3 seeds
           each; every control replayed under both cost modes; only tcn/configured feeds the wrapper) -> FloatingBackward
           over the 121 (fresh E3-M fit per subset x seed; empty group -> branch removed; fusion width recalculated;
           TRAIN-only scaler refit per subset; score = pooled per-trade net expectancy in sigma on INNER_A) -> SEARCH.json,
           CLOSE.json (BUDGET_REACHED_PROVISIONAL or WRAPPER_PHASE_COMPLETE). Tasks: bounded attempts (2), an interrupted
           attempt retained never overwritten, byte-authenticated results, resumable.
  emit     the 4090 queue (one ``run`` under the gate; every E3-M fit capped at 10G inside); NOT launched
  status   STATUS / CONTROLS / CLOSE
Nothing here reads a decision block; nothing is launched."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import fs_evaluator as FSE  # noqa: E402

BUDGET = dict(max_epochs=1000, patience=10, trunk="residual_tcn", fusion_units=8, core_blocks=2, readout="flatten", dropout_wd="recipe", agent_state="none")
SEARCH = dict(min_features=20, max_evaluations=128, tolerance=0.0)
SEEDS = [7, 19, 43]
CONSTRAINTS = dict(agent_state="none", scaling="fixed_train", clipping=False, fit_scope="train_only", test_access=False)
CONTROLS = [("tcn", "configured"), ("tcn", "zero_cost"), ("u", "configured")]
FAMILY_TRUNK = {"tcn": "residual_tcn", "u": "u"}
HS_PINNED = ("app/short_horizon", "tools")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(v, indent=1, sort_keys=True))
    os.replace(tmp, p)


def require(ok, code):
    if not ok:
        raise ValueError(code)


def source_pins(root: Path, subdirs) -> dict:
    pins = {}
    for sub in subdirs:
        for p in sorted((root / sub).rglob("*.py")):
            if "__pycache__" in p.parts or "/tests/" in str(p):
                continue
            pins[str(p.relative_to(root))] = FSE.file_sha(p)
    return pins


def import_selector(selector_root: str | None):
    if selector_root:
        sys.path.insert(0, str(Path(selector_root).resolve()))
    from feature_selector import search, contracts
    return search, contracts


def groups_for_selector(rec: dict) -> list:
    g = rec["representation"]["resolved"]
    cols = rec["list"]["columns"]
    return [{"name": n, "columns": [cols[i] for i in idx], "indices": list(idx), "extractor": k, "units": u}
            for n, idx, u, k in zip(g["names"], g["indices"], g["units"], g["kinds"])]


# ----------------------------------------------------------------------------- prepare (mirrors rl_selection_run.prepare)
def cmd_prepare(a):
    import pandas as pd
    from app.short_horizon.expert_guided import data as DG
    root = Path(a.root)
    require(not root.exists(), "NEW_CAMPAIGN_REQUIRED")
    original = read(a.recipe)
    require(original["tensor"]["window"]["observations"] == 13, "WINDOW_PIN")
    require(original["store"]["sha256"] == FSE.file_sha(a.store), "STORE_PIN")
    end = pd.Timestamp(original["split"]["test_closed_from"], tz="UTC") - pd.Timedelta(hours=1)      # never load TEST, even for constants
    frame = pd.read_parquet(a.store, columns=["timestamp"] + original["list"]["columns"], filters=[("timestamp", "<", end)])
    sc = DG.split_contract(a.contract)
    recipe = FSE.subset_recipe(original, list(original["list"]["columns"]))
    recipe["split"]["arm"] = a.contract
    recipe = FSE.refit_train_scaler(recipe, frame, tuple(sc["periods"]["TRAIN"]))
    recipe["business"]["evaluation_and_policy"]["cost_mode"] = "configured"
    recipe["labels"] = dict(mode=a.labels, expert_used=(a.labels != "none"), inherited_metadata_not_consumed=True)
    recipe["provenance"] = dict(parent_sha256=FSE.file_sha(a.recipe), parent_recipe_identity=original["identity"],
                                scope="new native wrapper; upstream selection retained, not independently revalidated",
                                internal_selection_bias="candidate pool used 2023; INNER_A is development, not independent confirmation")
    FSE.seal_recipe(recipe)
    write(root / "RECIPE.json", recipe)
    selector_root = Path(a.selector_root).resolve()
    spec = dict(schema="expert_guided_selection.v1", contract=a.contract,
                inputs={k: str(Path(getattr(a, k)).resolve()) for k in ("store", "manifest", "admitted", "csv_dir")},
                hs_root=str(ROOT), selector_root=str(selector_root), action_contract=str(Path(a.action_contract).resolve()),
                action_contract_sha256=FSE.file_sha(a.action_contract), budget=BUDGET, search=SEARCH, seeds=SEEDS, python=a.python,
                crispdm_run=a.crispdm_run, memory_cap=a.memory, max_attempts=2, task_timeout_seconds=43200, created_at=now(),
                code=dict(hs=source_pins(ROOT, HS_PINNED), selector=source_pins(selector_root, ["src"])))
    spec["inputs"]["recipe"] = str((root / "RECIPE.json").resolve())
    spec["recipe_sha256"] = FSE.file_sha(root / "RECIPE.json")
    periods = sc["periods"]
    dataset = dict(schema="feature_selector.dataset.v1", scope="development", features=recipe["list"]["columns"], groups=groups_for_selector(recipe),
                   blocks=FSE.coherent_blocks(recipe["list"]["columns"]), support_hash=FSE.digest({"recipe": recipe["identity"], "store": recipe["store"]["sha256"], "contract": a.contract}),
                   scaler_hash=FSE.digest(recipe["scaler"]), representation=recipe["representation"],
                   splits=dict(train=[str(x) for x in periods["TRAIN"]], selection=[str(x) for x in periods[sc["selection"]]], test_start=recipe["split"]["test_closed_from"]),
                   constraints=dict(CONSTRAINTS), files=[dict(path=str(root / "RECIPE.json"), sha256=spec["recipe_sha256"]), dict(path=spec["inputs"]["store"], sha256=recipe["store"]["sha256"])])
    _, contracts = import_selector(spec["selector_root"] + "/src")
    contracts.validate_dataset(dataset)
    spec["dataset"] = dataset
    spec["selection_block"] = sc["selection"]
    write(root / "SPEC.private.json", spec)
    write(root / "IDENTITY.json", dict(spec_sha256=FSE.file_sha(root / "SPEC.private.json")))
    write(root / "STATUS.json", dict(state="PREPARED", features=len(dataset["features"]), branches=len(dataset["groups"]), blocks=len(dataset["blocks"]),
                                     seeds=SEEDS, test_access=False, contract=a.contract))
    print(json.dumps(read(root / "STATUS.json")))


def verify(root: Path) -> dict:
    spec = read(root / "SPEC.private.json")
    require(FSE.file_sha(root / "SPEC.private.json") == read(root / "IDENTITY.json")["spec_sha256"], "SPEC_PIN")
    require(spec["schema"] == "expert_guided_selection.v1" and spec["budget"] == BUDGET and set(spec["search"]) == set(SEARCH), "RUNTIME_PIN")
    for role, base in (("hs", Path(spec["hs_root"])), ("selector", Path(spec["selector_root"]))):
        for name, sha in spec["code"][role].items():
            require(FSE.file_sha(base / name) == sha, "SOURCE_PIN:" + role + ":" + name)
    require(FSE.file_sha(spec["inputs"]["recipe"]) == spec["recipe_sha256"], "RECIPE_PIN")
    require(FSE.file_sha(spec["action_contract"]) == spec["action_contract_sha256"], "ACTION_CONTRACT_PIN")
    return spec


# ----------------------------------------------------------------------------- tasks (mirrors task_for / execute_task)
def params_for(spec: dict, mode: str, family: str) -> dict:
    p = dict(FSE.ExpertGuidedModelEvaluator.plugin_params)
    p.update(python=spec["python"], hs_root=spec["hs_root"], admitted=spec["inputs"]["admitted"], csv_dir=spec["inputs"]["csv_dir"], store=spec["inputs"]["store"],
             parent_recipe=spec["inputs"]["recipe"], action_contract=spec["action_contract"], model_id=f"expert_guided_c3_{family}", trunk=FAMILY_TRUNK[family],
             fusion_units=BUDGET["fusion_units"], core_blocks=BUDGET["core_blocks"], readout=BUDGET["readout"], split_contract=spec["contract"],
             selection_block=spec["selection_block"], max_epochs=BUDGET["max_epochs"], cost_mode=mode, selection_use="cost_aware" if mode == "configured" else "diagnostic",
             crispdm_run=spec.get("crispdm_run") or "", memory_cap=spec.get("memory_cap", "10G"), timeout_seconds=spec["task_timeout_seconds"],
             training_budget=spec["budget"], source_pins=[])
    FSE.ExpertGuidedModelEvaluator().validate(p)
    return p


def task_for(spec: dict, columns: list, seed: int, mode: str, family: str, control: bool) -> dict:
    _, contracts = import_selector(spec["selector_root"] + "/src")
    d = spec["dataset"]
    request = {**contracts.subset_view(d, columns), **{k: d[k] for k in ("support_hash", "scaler_hash", "splits", "constraints")},
               "representation": d["representation"], "seed": seed, "unit": FSE.digest([columns, seed, mode, family])[:20],
               "campaign": FSE.digest(spec), "scope": "development"}
    return dict(request=request, params=params_for(spec, mode, family), control=control)


def authenticate(path: Path, task: dict, evaluator) -> dict:
    return evaluator.verify(read(path), task["request"], task["params"], path.parent)


def execute_task(root: Path, spec: dict, task: dict, evaluator) -> dict:
    """One (subset, seed, mode, family) evaluation with bounded attempts; a finished RESULT.json is authenticated and
    reused; an interrupted attempt's folder is retained, never overwritten."""
    name = FSE.digest(task)
    path = root / "tasks" / (name + ".json")
    if path.exists():
        require(read(path) == task, "TASK_COLLISION")
    else:
        write(path, task)
    for attempt in range(1, spec["max_attempts"] + 1):
        folder = path.parent / (name + f"-attempt-{attempt}")
        result = folder / "RESULT.json"
        if result.exists():
            return authenticate(result, task, evaluator)
        ledger = path.parent / (name + f"-attempt-{attempt}.json")
        if ledger.exists():
            continue                         # an interrupted attempt is retained; never overwrite its artifacts
        write(ledger, dict(state="STARTED", started_at=now(), attempt=attempt, task_sha256=name))
        write(root / "STATUS.json", dict(state="RUNNING", unit=task["request"]["unit"], features=len(task["request"]["features"]), seed=task["request"]["seed"],
                                         cost_mode=task["params"]["cost_mode"], family=task["params"]["model_id"], attempt=attempt, heartbeat_at=now()))
        try:
            dataset = {**spec["dataset"], "scope": "development"}
            r = evaluator.evaluate(task["request"], dataset, task["params"], folder, root)
            write(result, r["native"])
            write(ledger, dict(state="COMPLETE", returncode=0, attempt=attempt, task_sha256=name, finished_at=now()))
            return authenticate(result, task, evaluator)
        except Exception as exc:                                     # noqa: BLE001 — recorded, retried once, then raised
            write(ledger, dict(state="FAILED", error=type(exc).__name__, detail=str(exc), attempt=attempt, task_sha256=name, finished_at=now()))
            time.sleep(min(30 * attempt, 5) if os.environ.get("SBFS_FAST_RETRY") else 30 * attempt)
    raise RuntimeError("TASK_FAILED_AFTER_BOUNDED_RETRIES:" + name)


def aggregate(rows: list) -> dict:
    measured = all(r["score"] is not None and r["gap"] is not None for r in rows)
    return dict(score=statistics.mean(r["score"] for r in rows) if measured else None,
                gap=statistics.mean(r["gap"] for r in rows) if measured else None,
                eligible=measured and all(r["eligible"] for r in rows),
                seeds=[dict(seed=r["native"]["seed"], score=r["score"], train_score=r["train_score"], gap=r["gap"], eligible=r["eligible"],
                            metrics=r["native"]["metrics"], primary=r["native"]["primary"], parameter_count=r["native"].get("parameter_count"),
                            request_sha256=r["native"]["request_sha256"], reason=r["native"]["primary"].get("reason")) for r in rows])


# ----------------------------------------------------------------------------- supervise (mirrors rl_selection_run.supervise)
def supervise(root: Path, evaluator=None, evaluator_path: str | None = None, selector_root: str | None = None):
    spec = verify(root)
    search, _ = import_selector(selector_root or spec["selector_root"] + "/src")
    if evaluator is None:
        evaluator = FSE.load_evaluator(evaluator_path)()
    lock = (root / "RUN.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError("RUN_ALREADY_ACTIVE") from exc
    try:
        if (root / "CLOSE.json").exists():
            require(read(root / "CLOSE.json")["spec_sha256"] == FSE.file_sha(root / "SPEC.private.json"), "CLOSE_PIN")
            return read(root / "CLOSE.json")
        full = spec["dataset"]["features"]
        controls = read(root / "CONTROLS.json") if (root / "CONTROLS.json").exists() else {}
        for family, mode in CONTROLS:
            key = family + "_" + mode
            if key in controls:
                continue
            rows = [execute_task(root, spec, task_for(spec, full, s, mode, family, True), evaluator) for s in spec["seeds"]]
            controls[key] = aggregate(rows)
            write(root / "CONTROLS.json", controls)
        subsets = root / "subsets"

        def evaluate(columns):
            rows = [execute_task(root, spec, task_for(spec, columns, s, "configured", "tcn", columns == full), evaluator) for s in spec["seeds"]]
            result = aggregate(rows)
            write(subsets / (FSE.digest(columns) + ".json"), dict(features=columns, **result))
            return result
        result = search.FloatingBackward().select(spec["dataset"], evaluate, spec["search"])
        write(root / "SEARCH.json", result)
        state = "BUDGET_REACHED_PROVISIONAL" if result["termination"] == "BUDGET" else "WRAPPER_PHASE_COMPLETE"
        close = dict(state=state, termination=result["termination"], evaluations=result["evaluations"], selected=result["selected"], finished_at=now(),
                     spec_sha256=FSE.file_sha(root / "SPEC.private.json"), global_release=False, test_access=False,
                     controls_feeding_wrapper=["tcn_configured"], upstream_methodology="retained; not replaced by wrapper",
                     comparison="winner vs the full candidate set and vs each cardinality's incumbent on the SELECTION block (SEARCH.json best_by_size); decision blocks untouched",
                     limitations=["selection internal to development", "upstream pool used selection-year rows", "cost assumptions not broker calibrated",
                                  "no independent confirmation or TEST read", "zero_cost control is a DIAGNOSTIC replay of the configured-cost policy (E3-M has no zero-cost training)"])
        write(root / "CLOSE.json", close)
        write(root / "STATUS.json", close)
        return close
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)


def cmd_run(a):
    try:
        supervise(Path(a.root), evaluator_path=a.evaluator_path or None, selector_root=a.selector_root or None)
    except Exception as exc:
        write(Path(a.root) / "STATUS.json", dict(state="FAILED", error=type(exc).__name__, detail=str(exc), finished_at=now()))
        raise


def cmd_status(a):
    root = Path(a.root)
    print(json.dumps({k: read(root / (k + ".json")) for k in ("STATUS", "CONTROLS", "CLOSE") if (root / (k + ".json")).exists()}, indent=2))


# ----------------------------------------------------------------------------- queue emission (not launched)
def cmd_emit(a):
    q = a.package_root
    lines = ["#!/bin/bash", "# SBFS with the final model (Musashi's plan, my evaluator); emitted by tools/expert_guided_sbfs.py; NOT launched by the emitter.",
             f"Q={q}; PY={a.python}; CR={a.crispdm_run}; R={a.campaign_root}", "log(){ echo \"$(date -u +%FT%TZ) [sbfs] $*\" >> $Q/LOG; }",
             "test -f $Q/LAUNCH_EXPERT_GUIDED.json || { log 'gate missing: refusing'; exit 3; }", "cd $Q/code || exit 1",
             "if [ ! -f $R/SPEC.private.json ]; then",
             f"  $PY tools/expert_guided_sbfs.py prepare --root $R --recipe $Q/recipe/CONSUMER_RECIPE.json --store $Q/data/store/full_pool_eurusd_h1.parquet "
             f"--manifest $Q/data/store/MANIFEST.json --admitted $Q/data/admitted --csv-dir $Q/data/sel15 --action-contract $Q/contracts/{a.action_contract_name} "
             f"--selector-root $Q/feature_selector --contract {a.contract} --python $PY --crispdm-run $CR --memory {a.memory} || {{ log 'prepare FAILED'; exit 4; }}",
             "fi",
             "# one supervisor; every E3-M fit inside runs under crispdm with the per-fit cap; resumable (tasks/, CONTROLS.json, subsets/, CLOSE.json)",
             f"$PY tools/expert_guided_sbfs.py run --root $R --evaluator-path $Q/code/app/short_horizon/expert_guided/fs_evaluator.py --selector-root $Q/feature_selector/src "
             "> $R/run.log 2>&1 && log 'sbfs run finished' || { log 'sbfs run FAILED'; exit 5; }",
             "log 'chain end (sbfs)'"]
    Path(a.out).write_text("\n".join(lines) + "\n")
    os.chmod(a.out, 0o755)
    print(json.dumps({"queue": a.out, "launched": False, "campaign_root": a.campaign_root}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    for k in ("recipe", "store", "manifest", "admitted", "csv_dir", "action_contract", "selector_root"):
        p.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
    p.add_argument("--contract", default="consolidated")
    p.add_argument("--labels", default="zigzag")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--crispdm-run", default="")
    p.add_argument("--memory", default="10G")
    p.set_defaults(fn=cmd_prepare)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--evaluator-path", default="")
    r.add_argument("--selector-root", default="")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("status")
    s.add_argument("--root", required=True)
    s.set_defaults(fn=cmd_status)
    e = sub.add_parser("emit")
    e.add_argument("--package-root", required=True)
    e.add_argument("--campaign-root", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--contract", default="consolidated")
    e.add_argument("--action-contract-name", default="ACTION_CONTRACT_C3.json")
    e.add_argument("--python", default="~/anaconda3/envs/trading-stack/bin/python")
    e.add_argument("--crispdm-run", default="~/.local/bin/crispdm-run")
    e.add_argument("--memory", default="10G")
    e.set_defaults(fn=cmd_emit)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

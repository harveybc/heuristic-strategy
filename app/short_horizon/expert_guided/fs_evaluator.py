"""Real-model evaluator for Musashi's ``feature_selector`` (owner 2026-10-26: the floating selection must use the FINAL
model; copy his plan, only the evaluator differs).

Entry point ``expert_guided_model`` in the group ``feature_selector.evaluators`` (declared in setup.py). The selector
proposes a column subset; this evaluator (1) derives a SUBSET RECIPE from the parent recipe with Musashi's structural
rules (``rl_subset.subset_recipe`` @ agent-multi e18d00d4: surviving columns in the parent's order, group map restricted,
an EMPTY group removes its branch and the fusion width is recomputed, coherent sin/cos and p0/p1/p2 blocks leave
together, TRAIN-only scaler refit, support hash), sealed under a NEW identity; (2) runs E3-M FRESH as a subprocess with
that recipe, the declared action contract / trunk, one seed; (3) replays the frozen policy on the SELECTION block through
the engine and returns the receipt (``feature_selector.expert_guided_result.v1``): request hash, identities, per-split
trading metrics in Musashi's field set (profit, sharpe, max_drawdown, orders, orders_per_week, win_rate, buy/sell
orders and win rates, early_closes, costs), cost components per split, the PRIMARY score = pooled per-trade net
expectancy in sigma with its day-block bootstrap interval (``stats.day_block_bootstrap_expectancy``), the TRAIN-selection
gap, parameter count, artifact hashes. An undefined score is an explicit ``None`` with a reason, never 0.
``cost_mode='configured'`` is the selection mode; ``cost_mode='zero_cost'`` is DIAGNOSTIC only: the same frozen policy is
REPLAYED under a profile that charges nothing (E3-M has no zero-cost training; declared). A decision block is never
evaluated here (``DECISION_BLOCK_EVALUATION_REFUSED``)."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_RESULT = "feature_selector.expert_guided_result.v1"
IDENTITY_KEYS = ("store", "list", "tensor", "scaler", "representation", "regularization", "split", "business", "labels")
DECISION_BLOCKS = ("INNER_B", "OBS_2024", "TEST")
MIN_TRADES_PER_WEEK = 2.0
COHERENT_SUFFIXES = (("_sin", "_cos"), (".p0", ".p1", ".p2"))
TRADING_METRICS = ("profit", "sharpe", "max_drawdown", "orders", "orders_per_week", "win_rate", "buy_orders", "sell_orders",
                   "buy_win_rate", "sell_win_rate", "early_closes", "costs")
COST_COMPONENTS = ("spread", "slippage", "commission", "financing")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, code: str):
    if not ok:
        raise ValueError(code)


# ----------------------------------------------------------------------------- structural subset recipes
def seal_recipe(rec: dict) -> dict:
    rec["identity_covers"] = list(IDENTITY_KEYS)
    rec["identity"] = "recipe_" + hashlib.sha256(json.dumps({k: rec[k] for k in IDENTITY_KEYS}, sort_keys=True).encode()).hexdigest()[:16]
    return rec


def coherent_blocks(columns: list) -> list:
    """Encoding blocks that leave the subset together: ``x_sin``/``x_cos`` pairs and ``p0/p1/p2`` triplets (Musashi's
    ``coherent_blocks`` extended to the regime probabilities); every other column is its own block."""
    remaining, blocks = set(columns), []
    for c in columns:
        if c not in remaining:
            continue
        block = [c]
        for suffixes in COHERENT_SUFFIXES:
            stem = next((c[: -len(s)] for s in suffixes if c.endswith(s)), None)
            if stem is not None:
                block = [n for n in columns if any(n == stem + s for s in suffixes) and n in remaining]
                break
        blocks.append(block)
        remaining.difference_update(block)
    return blocks


def subset_recipe(parent: dict, columns: list) -> dict:
    """The parent recipe restricted to ``columns`` (same order as the parent); groups restricted, empty groups removed,
    the fusion width (sum of the surviving groups' units) recomputed, scaler projected (``refit_train_scaler`` refits it
    from TRAIN rows), sealed under a new identity with provenance."""
    names = [c["store_column"] for c in parent["tensor"]["columns"]]
    require(columns and len(set(columns)) == len(columns) and set(columns) <= set(names), "SUBSET")
    require(columns == [n for n in names if n in columns], "SUBSET_ORDER")
    take = [names.index(c) for c in columns]
    remap = {old: new for new, old in enumerate(take)}
    rec = copy.deepcopy(parent)
    rec["list"].update(columns=list(columns), n=len(columns), sha256=digest(columns), file=f"structural_subset_{len(columns)}")
    rec["tensor"]["columns"] = [copy.deepcopy(parent["tensor"]["columns"][i]) for i in take]
    sc = rec["scaler"]
    for key in ("features", "mean", "std", "std_raw", "variance", "rule", "fit_rows_per_feature", "nan_share_train"):
        if key in sc and isinstance(sc[key], list):
            sc[key] = [sc[key][i] for i in take]
    sc["n_features"] = len(columns)
    sc.pop("artifact_sha256", None)
    stat = sc.get("stationarity", {})
    if isinstance(stat, dict) and isinstance(stat.get("features"), dict):
        keep = set(c["tensor_name"] for c in rec["tensor"]["columns"])
        stat["features"] = {k: v for k, v in stat["features"].items() if k in keep}
    resolved = parent["representation"]["resolved"]
    out = {k: [] for k in ("names", "indices", "units", "kinds")}
    for name, idx, units, kind in zip(resolved["names"], resolved["indices"], resolved["units"], resolved["kinds"]):
        active = [remap[i] for i in idx if i in remap]
        if active:
            for k, v in zip(out, (name, active, units, kind)):
                out[k].append(v)
    require(out["names"], "EMPTY_SUBSET_REPRESENTATION")
    rec["representation"]["resolved"] = out
    rec["representation"].pop("params_per_module", None)
    rec["representation"].pop("groups_file_sha256", None)
    tensor_names = [c["tensor_name"] for c in rec["tensor"]["columns"]]
    rec["representation"]["groups"] = {"name": "structural_subset", "rule": "parent groups restricted; empty branches removed",
                                       "groups": {n: [tensor_names[i] for i in idx] for n, idx in zip(out["names"], out["indices"])},
                                       "units": dict(zip(out["names"], out["units"])), "kinds": dict(zip(out["names"], out["kinds"]))}
    rec["representation"]["fusion_width"] = int(sum(out["units"]))
    rec["provenance"] = {"parent_recipe_identity": parent["identity"], "derivation": "structural_subset; paired parent support; fresh fit required",
                         "removed_columns": [n for n in names if n not in set(columns)]}
    return seal_recipe(rec)


def refit_train_scaler(rec: dict, frame, train_period: tuple, *, refuse_std_zero: bool = True) -> dict:
    """Fixed TRAIN constants of ``rec``'s columns recomputed on the rows of ``train_period`` (bar starts in
    ``frame.timestamp``; decision rows = bar start + 1 h, as the loader does). Consumer ops first (only ``DIFF`` or none).
    A zero-variance column is refused (audit A1-3). Nothing outside ``train_period`` is read for any statistic."""
    import pandas as pd
    rec = copy.deepcopy(rec)
    ts = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)) + pd.Timedelta(hours=1)
    a, b = (pd.Timestamp(t, tz="UTC") for t in train_period)
    mask = np.asarray((ts >= a) & (ts < b))
    cols = [c["store_column"] for c in rec["tensor"]["columns"]]
    values = frame[cols].to_numpy(float, copy=True)
    for j, c in enumerate(rec["tensor"]["columns"]):
        ops = list(c.get("consumer_ops", []))
        require(ops in ([], ["DIFF"]), "CONSUMER_OP")
        if ops == ["DIFF"]:
            values[:, j] = np.r_[np.nan, np.diff(values[:, j])]
    fit = values[mask]
    require(len(fit) > 1, "SCALER_TRAIN_SUPPORT")
    with np.errstate(all="ignore"):
        mean = np.nanmean(fit, axis=0)
        std_raw = np.nanstd(fit, axis=0)
    require(np.isfinite(mean).all() and np.isfinite(std_raw).all(), "SCALER_FINITE")
    zero = [c for c, s in zip(cols, std_raw) if s == 0]
    require(not (zero and refuse_std_zero), "STD_ZERO_COLUMN:" + ",".join(zero))
    std = np.where(std_raw > 0, std_raw, 1.0)
    n_fit = np.isfinite(fit).sum(axis=0)
    rec["scaler"] = {"policy": "fixed_train", "features": [c["tensor_name"] for c in rec["tensor"]["columns"]], "n_features": len(cols),
                     "fit_period": [str(train_period[0]), str(train_period[1])], "fit_rows": int(mask.sum()), "ddof": 0,
                     "mean": mean.tolist(), "std": std.tolist(), "std_raw": std_raw.tolist(), "variance": (std_raw ** 2).tolist(),
                     "rule": ["z" if s > 0 else "std_zero_to_one" for s in std_raw], "fit_rows_per_feature": [int(x) for x in n_fit],
                     "nan_share_train": [float(1 - x / max(mask.sum(), 1)) for x in n_fit],
                     "nan_policy": "nanmean/nanstd over TRAIN; zero-variance columns refused; NaN windows excluded",
                     "transform": "(x - mean) / std per feature", "stationarity": rec["scaler"].get("stationarity", {"policy": "inherited", "features": {}})}
    return seal_recipe(rec)


# ----------------------------------------------------------------------------- scoring a finished unit
def zero_cost_profile(profile):
    """The same profile with every charge at zero (DIAGNOSTIC replays only)."""
    return dataclasses.replace(profile, spread_schedule=tuple((a, b, 0.0) for a, b, _ in profile.spread_schedule), weekly_open_spread=0.0,
                               slippage_market=0.0, slippage_stop=0.0, commission_per_lot_side=0.0, swap_long_per_lot=0.0, swap_short_per_lot=0.0)


def trades_by_day(trades: list, sigma_price, days: list) -> dict:
    """{day: [net_sigma, ...]} for every day of the block (empty lists for days without trades)."""
    import pandas as pd
    out = {d: [] for d in days}
    for t in trades:
        k, u = int(t["entry_bar_abs"]), int(t["units"])
        s = float(sigma_price[k])
        day = pd.Timestamp(t["exit_time"]).strftime("%Y-%m-%d")
        if u * s > 0 and day in out:
            out[day].append(float(t["net_usd"]) / (u * s))
    return out


def block_days(ds, block: str) -> list:
    import pandas as pd
    a, b = ds.meta["periods"][block]
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(pd.Timestamp(a), pd.Timestamp(b) - pd.Timedelta(days=1), freq="D")]


def trading_summary(m: dict, trades: list) -> dict:
    """Musashi's TRADING_METRICS field set from the engine result and the trade records."""
    n = int(m.get("trades", len(trades)))
    nl = sum(1 for t in trades if int(t["direction"]) > 0)
    ns = sum(1 for t in trades if int(t["direction"]) < 0)
    wins = [t for t in trades if float(t["net_usd"]) > 0]
    wl = sum(1 for t in wins if int(t["direction"]) > 0)
    ws = len(wins) - wl
    comp = {"spread": float(sum(float(t["spread_usd"]) for t in trades)), "slippage": float(sum(float(t["slippage_usd"]) for t in trades)),
            "commission": float(sum(float(t["commission_usd"]) for t in trades)), "financing": float(sum(float(t["financing_usd"]) for t in trades))}
    sharpe = m.get("sharpe_daily_sqrt365")
    return ({"profit": float(m.get("net_profit_usd", 0.0)), "sharpe": (float(sharpe) if sharpe is not None and math.isfinite(float(sharpe)) else None),
             "max_drawdown": float(m.get("max_drawdown_pct", 0.0)) / 100.0, "orders": n, "orders_per_week": float(m.get("orders_per_week", 0.0)),
             "win_rate": (len(wins) / n) if n else None, "buy_orders": nl, "sell_orders": ns, "buy_win_rate": (wl / nl) if nl else None,
             "sell_win_rate": (ws / ns) if ns else None, "early_closes": int(m.get("n_close_AGENT_CLOSE", 0)) + int(m.get("n_close_REVERSAL", 0)),
             "costs": float(sum(comp.values())), "valid": bool(m.get("valid", False))}, comp)


def rebuild_policy(out_dir: Path, ds, seed: str):
    """The frozen policy of ``seed`` saved by E3-M in ``out_dir`` (same rebuild as the early-close audit), plus the
    contexts needed to replay it. Returns (policy_fn, cx, SCFG, trial) or (None, reason) when no strategy was frozen."""
    from tools.expert_guided_e1t import Ctx
    from tools.expert_guided_e3m import Composed, effective_contracts
    from tools import expert_guided_seed_ensemble as SE
    from app.short_horizon.expert_guided import action_contract as ACT, c2 as C2, env as E
    e3m = json.loads((out_dir / "E3M.json").read_text())
    tr = e3m["trials"]["FULL"][str(seed)]
    cx = Ctx(ds)
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    n_feat, win, NS = ds.windows.shape[2], ds.windows.shape[1], len(E.STATE_NAMES)
    SCFG, _ = effective_contracts(e3m.get("exec_contract", "sac"), e3m.get("min_orders_per_week"))
    ac = (e3m.get("action_contract") or {}).get("contract", {})
    c2 = C2.is_c2(ac)
    v1 = bool(e3m.get("action_contract")) and not c2
    if c2:
        pols = {m: SE.load_module(out_dir, e3m, m, seed, n_feat, win) for m in C2.SPECIALISTS}
        t = C2.seed_table(pols, cx, NS, tr["action_contract"]["calibration"], geo=C2.is_c2g(ac))
        st = C2.strategy_of(tr["action_contract"]["strategy"])
        if st is None:
            return None, "NO_FROZEN_STRATEGY"
        pf = C2.PolicyC2(t, st)(cx)
    elif v1:
        cost = ACT.row_cost_sigma(SCFG.profile.round_trip_price(), ds.sigma, ds.row_bar, ac["strategy"]["cost_sigma"])
        pols = {m: SE.load_module(out_dir, e3m, m, seed, n_feat, win) for m in ACT.SPECIALISTS}
        t = ACT.calibrate_table(ACT.raw_table(pols, cx.Xw, np.flatnonzero(cx.obs_ok), NS, cost), tr["action_contract"]["calibration"])
        st = ACT.strategy_of(tr["action_contract"]["strategy"])
        if st is None:
            return None, "NO_FROZEN_STRATEGY"
        pf = ACT.ContractPolicy(t, st)(cx)
    else:
        mask = E.state_mask(e3m["agent_state"]["mode"])
        pols = {m: SE.load_module(out_dir, e3m, m, seed, n_feat, win) for m in SE.MODULES}
        ar = tr["arbiter"]
        pf = Composed("FULL", pols, ar["arbiter"], ar.get("tau"), mask=mask)(cx)
    return (pf, cx, SCFG, tr), None


def score_block(pf, cx, cfg, ds, block: str, ref: dict | None) -> dict:
    """Replay ``pf`` on ``block`` (never a decision block), check it reproduces the run's own metrics when the run scored
    that block, and compute the primary statistic."""
    require(block not in DECISION_BLOCKS, "DECISION_BLOCK_EVALUATION_REFUSED:" + block)
    from app.short_horizon.expert_guided import action_contract as ACT, stats as ST
    m, trades = ACT.rollout_with_trades(cx, block, pf, cfg)
    reproduced = ref is None or (abs(float(m["net_profit_usd"]) - float(ref["net_profit_usd"])) < 1e-6 and int(m["trades"]) == int(ref["trades"]))
    tb = trades_by_day(trades, ds.sigma, block_days(ds, block))
    exp = ST.day_block_bootstrap_expectancy(tb) if trades else None
    summ, comp = trading_summary(m, trades)
    score = float(exp["expectancy"]) if exp is not None and math.isfinite(float(exp["expectancy"])) else None
    return {"score": score, "reason": (None if trades else "ZERO_TRADES"), "expectancy": exp, "metrics": summ, "cost_components": comp,
            "block": block, "reproduced": bool(reproduced), "activity_ok": summ["orders_per_week"] >= MIN_TRADES_PER_WEEK,
            "early_close_ok": summ["early_closes"] > 0, "trades": int(summ["orders"])}


# ----------------------------------------------------------------------------- the plugin
class ExpertGuidedModelEvaluator:
    """feature_selector evaluator: fresh E3-M training of the expert-guided specialists under a declared action contract."""
    api_version = 1
    production_rl = True
    plugin_params = dict(python=sys.executable, hs_root=str(ROOT), admitted="", csv_dir="", store="", parent_recipe="", action_contract="",
                         model_id="expert_guided_c3_tcn", trunk="residual_tcn", fusion_units=8, core_blocks=2, readout="flatten",
                         split_contract="consolidated", train_block="TRAIN", selection_block="INNER_A", max_epochs=1000, exec_contract="sac",
                         min_orders_per_week=MIN_TRADES_PER_WEEK, labels="zigzag", extra_args=[], timeout_seconds=43200,
                         cost_mode="configured", selection_use="cost_aware", crispdm_run="", memory_cap="10G", source_pins=[],
                         e3m_relpath="tools/expert_guided_e3m.py", training_budget={})

    def validate(self, params: dict):
        require(params["parent_recipe"] and params["store"] and params["admitted"] and params["csv_dir"] and params["split_contract"], "EVALUATOR_INPUTS_REQUIRED")
        require(params["selection_block"] not in DECISION_BLOCKS and params["train_block"] not in DECISION_BLOCKS, "DECISION_BLOCK_EVALUATION_REFUSED")
        require(params["trunk"] in ("u", "residual_tcn") and params["readout"] in ("flatten", "last"), "ARCHITECTURE")
        require(float(params["min_orders_per_week"]) >= MIN_TRADES_PER_WEEK, "ACTIVITY_REQUIREMENT")
        require(params["cost_mode"] in ("configured", "zero_cost") and params["selection_use"] in ("cost_aware", "diagnostic"), "COST_MODE")
        require(params["cost_mode"] == "configured" or params["selection_use"] == "diagnostic", "ZERO_COST_DIAGNOSTIC_ONLY")
        require(isinstance(params["training_budget"], dict) and params["training_budget"], "TRAINING_BUDGET_REQUIRED")

    def can_release(self, params: dict) -> bool:
        return params["cost_mode"] == "configured" and params["selection_use"] == "cost_aware"

    def command(self, params: dict, recipe_path: Path, out_dir: Path, seed: int) -> list:
        argv = [params["python"], params["e3m_relpath"], "run", "--admitted", params["admitted"], "--csv-dir", params["csv_dir"], "--store", params["store"],
                "--recipe", str(recipe_path), "--split-contract", params["split_contract"], "--out", str(out_dir), "--seeds", str(seed),
                "--exec-contract", params["exec_contract"], "--min-orders-per-week", str(params["min_orders_per_week"]),
                "--agent-state", "none", "--labels", params["labels"], "--max-epochs", str(params["max_epochs"]),
                "--trunk", params["trunk"], "--fusion-units", str(params["fusion_units"]), "--core-blocks", str(params["core_blocks"]), "--readout", params["readout"]]
        if params["action_contract"]:
            argv += ["--action-contract", params["action_contract"]]
        argv += list(params.get("extra_args") or [])
        if params["crispdm_run"]:
            argv = [params["crispdm_run"], "-q", "-m", params["memory_cap"], "-t", str(int(params["timeout_seconds"])), "-n", f"sbfs-{out_dir.parent.name[:12]}", "--"] + argv
        return argv

    def evaluate(self, request: dict, context: dict, params: dict, work, base) -> dict:
        self.validate(params)
        require(context.get("scope") == "development", "REAL_DATA_REQUIRED")
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        parent = json.loads(Path(params["parent_recipe"]).read_text())
        rec = subset_recipe(parent, list(request["features"]))
        rec["split"]["arm"] = params["split_contract"]
        frame = params.get("_frame")
        if frame is None:
            import pandas as pd
            end = pd.Timestamp(rec["split"].get("test_closed_from", "2025-01-01"), tz="UTC") - pd.Timedelta(hours=1)
            frame = pd.read_parquet(params["store"], columns=["timestamp"] + rec["list"]["columns"], filters=[("timestamp", "<", end)])
        rec = refit_train_scaler(rec, frame, tuple(request["splits"]["train"]))
        recipe_path = work / "RECIPE_subset.json"
        recipe_path.write_text(json.dumps(rec, indent=1))
        out_dir = work / "e3m"
        argv = self.command(params, recipe_path, out_dir, int(request["seed"]))
        (work / "COMMAND.json").write_text(json.dumps(argv, indent=1))
        with (work / "stdout.log").open("wb") as o, (work / "stderr.log").open("wb") as e:
            rc = subprocess.run(argv, cwd=params["hs_root"], stdout=o, stderr=e, timeout=float(params["timeout_seconds"])).returncode
        require(rc == 0, f"E3M_FAILED:{rc}")
        return self.receipt(request, params, work, out_dir, rec)

    def receipt(self, request: dict, params: dict, work: Path, out_dir: Path, rec: dict, ds=None) -> dict:
        from app.short_horizon.expert_guided import data as DG
        e3m = json.loads((out_dir / "E3M.json").read_text())
        require((e3m.get("recipe") or {}).get("identity") == rec["identity"], "RECIPE_IDENTITY_MISMATCH")
        require(e3m.get("split_contract") == params["split_contract"], "SPLIT_CONTRACT_MISMATCH")
        seed = str(request["seed"])
        tr = e3m["trials"]["FULL"][seed]
        if ds is None:
            ds = DG.load_store(params["admitted"], params["store"], rec["list"]["columns"], scaling="fixed_train", contract=params["split_contract"],
                               win=int(rec["tensor"]["window"]["observations"]), stride=int(rec["tensor"]["window"]["stride_rows"]), recipe=rec)
        built, reason = rebuild_policy(out_dir, ds, seed)
        blocks = {}
        if built is None:
            for b in (params["train_block"], params["selection_block"]):
                blocks[b] = {"score": None, "reason": reason, "metrics": None, "cost_components": None, "reproduced": None, "trades": 0}
        else:
            pf, cx, SCFG, _ = built
            cfg = SCFG if params["cost_mode"] == "configured" else dataclasses.replace(SCFG, profile=zero_cost_profile(SCFG.profile))
            for b in (params["train_block"], params["selection_block"]):
                ref = tr["trading"].get(b) if params["cost_mode"] == "configured" else None
                blocks[b] = score_block(pf, cx, cfg, ds, b, ref)
        sel, train = blocks[params["selection_block"]], blocks[params["train_block"]]
        artifacts = [{"path": str(p), "sha256": file_sha(p)} for p in sorted(out_dir.glob("*")) if p.is_file() and p.suffix in (".json", ".pt", ".jsonl")]
        decision_read = sorted(k for k in tr["trading"] if k in DECISION_BLOCKS and tr["trading"][k])
        result = {"schema": SCHEMA_RESULT, "request_sha256": digest(request), "recipe_identity": rec["identity"], "recipe_sha256": file_sha(work / "RECIPE_subset.json"),
                  "features": list(request["features"]), "groups": request.get("groups"), "seed": request["seed"], "support_hash": request["support_hash"],
                  "scaler_hash": request["scaler_hash"], "splits": request["splits"], "model_id": params["model_id"], "training_budget": params["training_budget"],
                  "action_contract": (e3m.get("action_contract") or {}).get("identity"), "cost_mode": params["cost_mode"],
                  "architecture": {"trunk": params["trunk"], "fusion_units": params["fusion_units"], "core_blocks": params["core_blocks"], "readout": params["readout"]},
                  "fresh_training": True, "test_access": False, "observation_contract": request["constraints"],
                  "parameter_count": int(sum((e3m.get("params") or {}).get("FULL", {}).values()) or 0),
                  "metrics": {"train": train["metrics"], "selection": sel["metrics"]}, "cost_components": {"train": train["cost_components"], "selection": sel["cost_components"]},
                  "primary": {"statistic": "pooled per-trade net expectancy (sigma), day-block bootstrap", "block": params["selection_block"], "value": sel["score"],
                              "interval": sel.get("expectancy"), "reason": sel.get("reason"), "reproduced": sel.get("reproduced")},
                  "train_primary": {"value": train["score"], "interval": train.get("expectancy")},
                  "decision_blocks_read_by_e3m": decision_read, "artifacts": artifacts, "fits": tr.get("fits")}
        return self.verify(result, request, params, work)

    def verify(self, result: dict, request: dict, params: dict, base) -> dict:
        self.validate(params)
        require(result.get("schema") == SCHEMA_RESULT and result.get("request_sha256") == digest(request), "RECEIPT_REQUEST_MISMATCH")
        for key in ("features", "seed", "support_hash", "scaler_hash", "splits"):
            require(result.get(key) == request[key], "RECEIPT_MISMATCH:" + key)
        for key in ("model_id", "training_budget", "cost_mode"):
            require(result.get(key) == params[key], "RECEIPT_MODEL_MISMATCH:" + key)
        require(result.get("fresh_training") is True and result.get("test_access") is False, "TRAINING_CONTRACT")
        require(not result.get("decision_blocks_read_by_e3m"), "DECISION_BLOCK_READ_BY_TRAINING")
        require(result["primary"]["block"] not in DECISION_BLOCKS, "DECISION_BLOCK_EVALUATION_REFUSED")
        for item in result.get("artifacts", []):
            p = Path(item["path"])
            require(p.is_file() and file_sha(p) == item["sha256"], "ARTIFACT_CHANGED:" + p.name)
        for split in ("train", "selection"):
            m, c = result["metrics"][split], result["cost_components"][split]
            if m is None:
                continue
            require(all(k in m for k in TRADING_METRICS), "METRICS_MISSING")
            require(set(c) == set(COST_COMPONENTS) and math.isclose(sum(c.values()), m["costs"], rel_tol=1e-9, abs_tol=1e-6), "COST_RECONCILIATION")
            require(m["orders"] == m["buy_orders"] + m["sell_orders"] and m["early_closes"] <= m["orders"], "DIRECTION_COUNT")
            if params["cost_mode"] == "zero_cost":
                require(all(v == 0 for v in c.values()), "ZERO_COST_NONZERO_CHARGE")
        score = result["primary"]["value"]
        tscore = result.get("train_primary", {}).get("value")
        sel = result["metrics"]["selection"] or {}
        gap = (max(0.0, tscore - score) if (score is not None and tscore is not None) else None)
        eligible = bool(score is not None and gap is not None and sel.get("orders_per_week", 0.0) >= MIN_TRADES_PER_WEEK and sel.get("early_closes", 0) > 0
                        and result["primary"].get("reproduced") is not False)
        return {"score": (score if gap is not None else None), "train_score": tscore, "gap": gap, "eligible": eligible, "native": result}


def load_evaluator(path: str | None = None):
    """The plugin class via the installed entry point, or from ``path`` (``--evaluator-path``) when heuristic_strategy is
    not installed in the interpreter that runs feature_selector."""
    if path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hs_fs_evaluator", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ExpertGuidedModelEvaluator
    import importlib.metadata as md
    eps = list(md.entry_points(group="feature_selector.evaluators", name="expert_guided_model"))
    require(len(eps) == 1, "PLUGIN_NOT_INSTALLED: feature_selector.evaluators/expert_guided_model (use --evaluator-path)")
    return eps[0].load()

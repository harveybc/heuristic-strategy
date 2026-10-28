#!/usr/bin/env python3
"""Dependent queue after the fold-scope correction (Musashi's final joint audit A1, 2026-10-23). A process, not an agent:
it waits for the selection v5 aggregate, then runs every DEPENDENT step once, with STATE markers (resumable):

  consumer_store   pool + the consumer scope's fold-local columns (predictor tools/consumer_store.py)
  justification    per-column justification and the family groups for the v5 lists on the consumer store
  profiles_t1      the peer worker's TRAIN-only profiles tool (unchanged) on the consumer store and the v5 lists
  spec             the clean-round spec bound to the consumer store, the v5 lists and a PENDING regularization
  t2_strict        the two decision-source T2 arms of the strict list (agent_state none, base / strong), receipted; the
                   core15 arms are REUSED (their columns are unaffected by the consumer scope; authenticated receipts)
  regularize       verify-t2 over both producers, then the frozen rule (--apply)
  t3_manifest      T3 v2 manifest from the v5 strict list, its groups and the new profiles; typed queues per worker
  t3_satoshi       this lane's half of T3, receipted, on its GPU
The peer worker's half and the per-group decision, the combination and the recipe follow from the session (the
workers cannot reach each other). Same business rules, budgets and frozen decision rules as before."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Q:
    def __init__(self, cfg: dict):
        self.c = cfg
        self.W = Path(cfg["work"])
        self.D = self.W / "data"
        self.st = self.W / "a1q" / "STATE"
        self.st.mkdir(parents=True, exist_ok=True)
        self.log_f = self.W / "LOG"

    def log(self, msg):
        with self.log_f.open("a") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + f" [a1q] {msg}\n")

    def sh(self, argv, name, cwd, mem="4G", wall="14400s", gpu="", child_admits=False):
        cr = [str(Path.home() / ".local/bin/crispdm-run"), "-q", "-m", mem, "-t", wall, "-n", f"a1q-{name}", "--"]
        if child_admits:
            if argv[:3] != [PY, "tools/receipted_run.py", "run"]:
                raise ValueError("CHILD_ADMISSION_REQUIRES_RECEIPTED_RUN")
            # The receipted runner acquires the fit's own lease and memory cap.
            # Reserving again for its waiting parent can prevent that fit from admission.
            cr = []
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="8")
        with (self.W / "a1q" / f"{name}.log").open("a") as fh:
            rc = subprocess.run(cr + argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            self.log(f"{name} FAILED rc={rc}")
            raise SystemExit(rc)

    def step(self, name, fn):
        if (self.st / name).exists():
            return
        self.log(f"{name} start")
        fn()
        (self.st / name).touch()
        self.log(f"{name} DONE")

    # ------------------------------------------------------------------ steps
    def wait_selection(self):
        while "SEL5_AGGREGATE_DONE" not in self.log_f.read_text():
            if "SEL5_AGGREGATE_FAILED" in self.log_f.read_text() or " FAILED (see run logs)" in self.log_f.read_text():
                self.log("selection v5 failed: dependent queue stops")
                raise SystemExit(3)
            time.sleep(300)

    def consumer_store(self):
        so = self.W / "scopes_out"
        self.sh([PY, "tools/consumer_store.py", "--pool", str(self.D / "full_pool_s/full_pool_eurusd_h1.parquet"),
                 "--scope-parquet", str(so / "SCOPE_consumer.parquet"), "--scope-report", str(so / "SCOPE_consumer.json"),
                 "--out", str(self.D / "full_pool_s_a1")], "consumer_store", self.W / "predictor")

    def lists(self):
        """FEATURE_LIST_<name>.json from the aggregate's SHORTLIST.json (the v4 driver did this outside the engine)."""
        fin = self.W / "selection_v5" / "final"
        sl = json.loads((fin / "SHORTLIST.json").read_text())
        for n, cols in sl["lists"].items():
            (fin / f"FEATURE_LIST_{n}.json").write_text(json.dumps({"name": n, "features": cols, "n": len(cols),
                "source": "pool_selection v5 (fold-local producers, audit A1) SHORTLIST.json"}, indent=1))

    def justification(self):
        fin = self.W / "selection_v5" / "final"
        for name in ("core15", "shortlist_strict"):
            self.sh([PY, "tools/expert_guided_input_justification.py", "--feature-list", str(fin / f"FEATURE_LIST_{name}.json"),
                     "--catalog", str(self.D / "eurusd_h1_catalog_s/FEATURE_CATALOG.csv"), "--selection-dir", str(fin),
                     "--store", str(self.D / "full_pool_s_a1/full_pool_eurusd_h1.parquet"), "--out", str(self.W / "just_a1")],
                    f"justification_{name}", self.W / "hs")

    def profiles_t1(self):
        # the peer worker's tool, unchanged, pinned by its commit (read-only use; it imports its own helpers)
        self.sh([PY, "selection_input_profiles.py", "--store", str(self.D / "full_pool_s_a1/full_pool_eurusd_h1.parquet"),
                 "--admitted", str(self.D / "admitted/eurusd.parquet"), "--selection", str(self.W / "selection_v5" / "final"),
                 "--templates", str(self.W / "just_a1"), "--output", str(self.W / "t1_a1")], "profiles_t1", self.W / "musashi_tools")

    def spec(self):
        s = json.loads(Path(self.c["spec"]).read_text())
        store = self.D / "full_pool_s_a1/full_pool_eurusd_h1.parquet"
        s["store"] = {"path": str(store), "sha256": sha(store), "note": "consumer store after the A1 fold-scope correction"}
        fin = self.W / "selection_v5" / "final"
        for st in s["stages"].values():
            for arm in st:
                if arm.get("list"):
                    lp = fin / Path(arm["list"]).name
                    arm["list"], arm["list_sha256"] = str(lp), sha(lp)
                if arm.get("groups"):
                    arm["groups"] = str(self.W / "just_a1" / Path(arm["groups"]).name)
        s["regularization"].update(dropout=None, wd=None, source=None, status="PENDING_T2_A1", chosen=None)
        s["selection"] = {"identity": "selection_v5_fold_local", "ledger_sha256": sha(fin / "LEDGER.jsonl"),
                          "lists": {n: len(json.loads((fin / f"FEATURE_LIST_{n}.json").read_text())["features"]) for n in ("core15", "shortlist_strict", "shortlist")}}
        (self.W / "spec_a1.json").write_text(json.dumps(s, indent=1) + "\n")

    def _t2_units(self):
        old = json.loads(Path(self.c["t2_units_v4"]).read_text())["units"]
        keep = [u for u in old if u["list"] == "core15" and u["agent_state"] == "none"]
        new = []
        for reg, (do, wd) in (("base", ("0.1", "1e-4")), ("strong", ("0.3", "1e-2"))):
            name = f"gap_shortlist_strict_none_{reg}"
            cmd = (f"tools/expert_guided_e3m.py run --admitted <root>/data/admitted --csv-dir <root>/data/sel15 "
                   f"--store <root>/data/full_pool_s_a1/full_pool_eurusd_h1.parquet --feature-list <root>/in/FEATURE_LIST_shortlist_strict.json "
                   f"--out <root>/out/{name} --units 2 --head 16 --variants FULL --max-epochs 1000 --scaling fixed_train --nonstationary diff "
                   f"--agent-state none --dropout {do} --wd {wd} --exec-contract sac --min-orders-per-week 2.0 --split-contract consolidated")
            new.append({"name": name, "list": "shortlist_strict", "agent_state": "none", "regularization": reg, "dropout": float(do), "wd": float(wd),
                        "command": cmd, "identity": hashlib.sha256(cmd.encode()).hexdigest()[:16]})
        return keep, new

    def t2_strict(self):
        R = self.W / "t2_a1"
        if not (R / "code").exists():
            shutil.copytree(self.W / "hs", R / "code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (R / "data").mkdir(parents=True, exist_ok=True)
        (R / "in").mkdir(exist_ok=True)
        for d in ("admitted", "sel15", "full_pool_s_a1"):
            if not (R / "data" / d).exists():
                os.symlink(self.D / d, R / "data" / d)
        shutil.copy(self.W / "selection_v5" / "final" / "FEATURE_LIST_shortlist_strict.json", R / "in")
        keep, new = self._t2_units()
        (R / "T2_UNITS_A1.json").write_text(json.dumps({"units": keep + new, "note": "core15 arms reused (unaffected by the consumer scope)"}, indent=1))
        self.sh([PY, "tools/receipted_run.py", "config", "--root", str(R), "--out", str(R / "CONFIG.json"),
                 "--science-commit", self.c["science_commit"]], "t2_config", R / "code", mem="1G")
        for u in new:
            (R / f"{u['name']}.unit.json").write_text(json.dumps(u))
            if (R / "ops" / f"{u['name']}.RESULT.json").exists():
                continue
            self.sh([PY, "tools/receipted_run.py", "run", "--root", str(R), "--config", str(R / "CONFIG.json"), "--unit", str(R / f"{u['name']}.unit.json"),
                     "--name", u["name"], "--gpu", self.c["gpu"], "--mem", "4G"], f"t2_{u['name']}", R / "code", mem="6G", wall="50000s", gpu=self.c["gpu"], child_admits=True)

    def regularize(self):
        R, M = self.W / "t2_a1", Path(self.c["musashi_t2_replica"])
        J = self.W / "t2_join"
        (J / "out").mkdir(parents=True, exist_ok=True)
        (J / "ops").mkdir(exist_ok=True)
        (J / "lists").mkdir(exist_ok=True)
        keep, new = self._t2_units()
        for u in keep:
            if not (J / "out" / u["name"]).exists():
                os.symlink(M / "out" / u["name"], J / "out" / u["name"])
            shutil.copy(M / "ops" / f"{u['name']}.RESULT.json", J / "ops")
        for u in new:
            if not (J / "out" / u["name"]).exists():
                os.symlink(R / "out" / u["name"], J / "out" / u["name"])
            shutil.copy(R / "ops" / f"{u['name']}.RESULT.json", J / "ops")
        shutil.copy(self.c["core15_list_v4"], J / "lists" / "FEATURE_LIST_core15.json")
        shutil.copy(self.W / "selection_v5" / "final" / "FEATURE_LIST_shortlist_strict.json", J / "lists")
        fin_core = json.loads((self.W / "selection_v5" / "final" / "FEATURE_LIST_core15.json").read_text())["features"]
        if fin_core != json.loads(Path(self.c["core15_list_v4"]).read_text())["features"]:
            self.log("core15 list changed in v5: the core15 arms cannot be reused; stopping for the session")
            raise SystemExit(4)
        self.sh([PY, "tools/expert_guided_clean_round.py", "verify-t2", "--units", str(R / "T2_UNITS_A1.json"), "--results", str(J / "out"),
                 "--receipts", str(J / "ops"), "--config", str(M / "ops" / "CONFIG.private.json"), "--science-commit", self.c["musashi_t2_commit"],
                 "--config", str(R / "CONFIG.json"), "--science-commit", json.loads((R / "CONFIG.json").read_text())["science_commit"],
                 "--lists-dir", str(J / "lists"), "--science-blobs", str(self.W / "science_blobs"),
                 "--out", str(self.W / "T2_VERIFY_A1.json")], "verify_t2", self.W / "hs", mem="2G")
        self.sh([PY, "tools/expert_guided_clean_round.py", "regularize", "--spec", str(self.W / "spec_a1.json"), "--results", str(J / "out"),
                 "--out", str(self.W / "REGULARIZATION_DECISION_A1.json"), "--verified", str(self.W / "T2_VERIFY_A1.json"), "--apply"],
                "regularize", self.W / "hs", mem="2G")

    def t3_manifest(self):
        T = self.W / "t3_a1"
        self.sh([PY, "tools/expert_guided_t3_manifest.py", "emit", "--spec", str(self.W / "spec_a1.json"),
                 "--groups", str(self.W / "just_a1/FEATURE_GROUPS_shortlist_strict.json"),
                 "--list", str(self.W / "selection_v5/final/FEATURE_LIST_shortlist_strict.json"),
                 "--profiles", str(self.W / "t1_a1/PROFILES_shortlist_strict.csv"),
                 "--profiles-manifest-sha256", sha(self.W / "t1_a1/PROFILES_MANIFEST.json"), "--out", str(T)], "t3_emit", self.W / "hs", mem="2G")
        for w in ("satoshi", "musashi"):
            self.sh([PY, "tools/expert_guided_t3_manifest.py", "queue", "--manifest", str(T / "T3_MANIFEST.json"), "--worker", w,
                     "--out", str(T / f"t3_queue_{w}.sh"), "--out-units", str(T / f"T3_UNITS_{w}.json")], f"t3_queue_{w}", self.W / "hs", mem="1G")
        P = self.W / "t3pkg_a1"
        if not (P / "code").exists():
            shutil.copytree(self.W / "hs", P / "code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (P / "data").mkdir(parents=True, exist_ok=True)
        (P / "in").mkdir(exist_ok=True)
        for d in ("admitted", "sel15", "full_pool_s_a1"):
            if not (P / "data" / d).exists():
                os.symlink(self.D / d, P / "data" / d)
        shutil.copy(self.W / "selection_v5/final/FEATURE_LIST_shortlist_strict.json", P / "in")
        if (P / "t3").exists():
            shutil.rmtree(P / "t3")
        shutil.copytree(T, P / "t3")

    def t3_satoshi(self):
        P = self.W / "t3pkg_a1"
        if not (P / "CONFIG.json").exists():
            self.sh([PY, "tools/receipted_run.py", "config", "--root", str(P), "--out", str(P / "CONFIG.json"),
                     "--science-commit", self.c["science_commit"]], "t3_config", P / "code", mem="1G")
        units = json.loads((P / "t3" / "T3_UNITS_satoshi.json").read_text())["units"]
        for u in units:
            if (P / "ops" / f"{u['name']}.RESULT.json").exists():
                continue
            (P / f"{u['name']}.unit.json").write_text(json.dumps(u))
            self.sh([PY, "tools/receipted_run.py", "run", "--root", str(P), "--config", str(P / "CONFIG.json"), "--unit", str(P / f"{u['name']}.unit.json"),
                     "--name", u["name"], "--gpu", self.c["gpu"], "--mem", "4G"], f"t3_{u['name']}", P / "code", mem="6G", wall="50000s", gpu=self.c["gpu"], child_admits=True)
            self.log(f"t3 {u['name']} sealed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    q = Q(json.loads(Path(a.config).read_text()))
    q.wait_selection()
    for name in ("consumer_store", "lists", "justification", "profiles_t1", "spec", "t2_strict", "regularize", "t3_manifest", "t3_satoshi"):
        q.step(name, getattr(q, name))
    q.log("A1Q_SATOSHI_PART_DONE (peer half, decision, combination and recipe follow from the session)")


if __name__ == "__main__":
    main()

"""Mutation witnesses against Musashi's v5 closure tools (his order item 2), on COPIES only: nothing in his directories
is touched, no scp / systemctl path is reached (every mutated run must raise before them). Each case records the
exception code; a mutated input that yields a successful completion claim is a finding."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

W = Path("/var/tmp/a1-fold-local-20261023")
A = W / "closure_attack"
TOOLS = A / "tools"
sys.path.insert(0, str(TOOLS))
import expert_package_worker as w  # noqa: E402
import close_t3_v5 as cl  # noqa: E402
import collect_expert_package as co  # noqa: E402
import prepare_expert_package_t3_v5 as pv  # noqa: E402
import finalize_t3_combination as fc  # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
PKG = A / "pkg"                      # copy of the package I shipped to Musashi (PINS.sha256 c20ddcd7...)
CREP = Path("/var/tmp/musashi-t3-v5-C-replica-20261024")
FAST = Path("/var/tmp/musashi-t3-v5-fast-replica-20261024")
out = {}


def run(name, fn, expect):
    try:
        r = fn()
        out[name] = {"accepted": True, "result": str(r)[:160], "expected": expect, "ok": expect == "ACCEPT"}
    except Exception as e:  # noqa: BLE001
        code = f"{type(e).__name__}:{e}"
        out[name] = {"accepted": False, "error": code[:200], "expected": expect, "ok": expect != "ACCEPT" and expect in code}
    print(name, out[name]["ok"], out[name].get("error", "accepted")[:120], flush=True)


def fresh_pkg(tag):
    d = A / f"pkg_{tag}"
    if d.exists():
        shutil.rmtree(d)
    subprocess.run(["cp", "-al", str(PKG), str(d)], check=True)
    return d


def mutate(path, fn):
    """replace a hard link by a modified private copy"""
    p = Path(path)
    data = p.read_bytes()
    os.unlink(p)
    p.write_bytes(fn(data))


# ---- A. package pins (verify): changed manifest, swapped group
run("baseline_verify_package", lambda: pv.verify(PKG)[0]["identity"], "ACCEPT")
d = fresh_pkg("manifest")
mutate(d / "t3/T3_MANIFEST.json", lambda b: json.dumps(dict(json.loads(b), identity="t3v2_other")).encode())
run("changed_manifest", lambda: pv.verify(d), "V5_PACKAGE_PIN_CHANGED")
d = fresh_pkg("group")
g = json.loads((d / "t3/groups/unit_07.json").read_text()); g["kinds"] = {k: ("conv" if v == "pointwise" else v) for k, v in g["kinds"].items()}
mutate(d / "t3/groups/unit_07.json", lambda b: json.dumps(g, indent=1).encode())
run("swapped_group_kind", lambda: pv.verify(d), "V5_PEER_PIN_CHANGED")
d = fresh_pkg("groupfile")
# the groups file digest in T3_UNITS vs the file: bypass PINS by also rewriting PINS.sha256 for that file (an attacker who
# controls the package); the per-job groups digest must still catch it
mutate(d / "t3/groups/unit_07.json", lambda b: json.dumps(g, indent=1).encode())
pins = (d / "PINS.sha256").read_text().splitlines()
pins = [f"{sha(d / 't3/groups/unit_07.json')}  ./t3/groups/unit_07.json" if l.endswith("./t3/groups/unit_07.json") else l for l in pins]
mutate(d / "PINS.sha256", lambda b: ("\n".join(pins) + "\n").encode())
run("swapped_group_with_rewritten_pins_file", lambda: pv.verify(d), "V5_PACKAGE_PIN_CHANGED")

# ---- B. pairing: changed support of one unit
rows = {f"t3_unit_{i:02d}": {"support_hash": {"TRAIN": "t", "INNER_A": "i"}, "consumed_features": ["a"], "seeds": ["1"]} for i in range(45)}
run("baseline_paired", lambda: cl.paired(rows), "ACCEPT")
bad = json.loads(json.dumps(rows)); bad["t3_unit_31"]["support_hash"]["INNER_A"] = "x"
run("changed_support", lambda: cl.paired(bad), "V5_PAIRING_MISMATCH")
bad = json.loads(json.dumps(rows)); bad.pop("t3_unit_44")
run("missing_unit", lambda: cl.paired(bad), "V5_INCOMPLETE_UNITS")

# ---- C. mirror validation on a hard-link copy of a real replica: altered weights, altered scaler, altered result
def mirror_copy(src, tag):
    d = A / f"mirror_{tag}"
    if d.exists():
        shutil.rmtree(d)
    subprocess.run(["cp", "-al", str(src), str(d)], check=True)
    return d
m = mirror_copy(CREP, "c_base")
run("baseline_validate_mirror_C", lambda: co.validate_mirror(m)["count"], "ACCEPT")
m = mirror_copy(CREP, "c_weights"); mutate(m / "out/t3_unit_C/FULL_open_l_20261020.pt", lambda b: b[:-1] + bytes([b[-1] ^ 1]))
run("altered_weights", lambda: co.validate_mirror(m), "MIRROR_OUTPUT_CHANGED")
m = mirror_copy(CREP, "c_scaler"); sc = json.loads((m / "out/t3_unit_C/SCALER.json").read_text()); sc["mean"][0] += 1e-9
mutate(m / "out/t3_unit_C/SCALER.json", lambda b: json.dumps(sc).encode())
run("altered_scaler", lambda: co.validate_mirror(m), "MIRROR_OUTPUT_CHANGED")
m = mirror_copy(FAST, "fast_result"); e = json.loads((m / "out/t3_unit_07/E3M.json").read_text())
e["trials"]["FULL"]["20261019"]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] += 1.0
mutate(m / "out/t3_unit_07/E3M.json", lambda b: json.dumps(e).encode())
run("altered_result_metrics", lambda: co.validate_mirror(m), "MIRROR_OUTPUT_CHANGED")

# ---- D. combination finalizer: stale replay and replay contract, on a hard-link copy of the C mirror; the raise happens
# before any scp / systemctl call (those appear only after a consistent PASS)
D = Path("/var/tmp/musashi-t3-v5-delivery-20261024")
m = mirror_copy(CREP, "c_tick")
producer = w.read(m / "ops/CONFIG.private.json")
# the replica copy lacks the mirror's REMOTE_REPLICA_STATUS.json (written on Musashi's mirror, not on the replica), so
# tick() returned WAITING_FOR_REPLICA before the attacked checks in the first run (recorded); provide a consistent one
# (the real check of that file is validate_mirror on the replica, exercised above) to reach the stale-replay checks
first = {}
try:
    first = fc.tick({"mirror": str(m), "output": str(A / "tick_out0.json"), "destination": "unused:/x", "timer": "unused.timer",
                     "expected_config_sha256": w.identity(producer), "replay_output": str(A / "none.json"), "replay_plan": str(A / "plan.json"),
                     "source_root": str(m), "python": sys.executable, "replay_script": "/nonexistent"})
except Exception as e:  # noqa: BLE001
    first = {"error": str(e)[:120]}
out["tick_without_replica_status_returns"] = {"result": str(first), "ok": first.get("state") == "WAITING_FOR_REPLICA"}
(m / "REMOTE_REPLICA_STATUS.json").write_text(json.dumps({"verified": True, "closed": True, "count": 1, "note": "synthetic for the witness copy"}))
def tick_cfg(replay_path):
    return {"mirror": str(m), "output": str(A / "tick_out.json"), "destination": "unused:/x", "timer": "unused.timer",
            "expected_config_sha256": w.identity(producer), "replay_output": str(replay_path), "replay_plan": str(A / "plan.json"),
            "source_root": str(m), "python": sys.executable, "replay_script": "/nonexistent"}
rep = json.loads((D / "COMBINATION_REPLAY.json").read_text())
stale = json.loads(json.dumps(rep)); stale["units"]["t3_unit_C"]["result_sha256"] = "0" * 64
(A / "replay_stale.json").write_text(json.dumps(stale))
run("stale_replay", lambda: fc.tick(tick_cfg(A / "replay_stale.json")), "COMBINATION_FINAL_STALE_REPLAY")
short = json.loads(json.dumps(rep)); short["checkpoints"] = 11
(A / "replay_short.json").write_text(json.dumps(short))
run("replay_missing_checkpoint", lambda: fc.tick(tick_cfg(A / "replay_short.json")), "COMBINATION_FINAL_REPLAY_CONTRACT")
inexact = json.loads(json.dumps(rep)); k = next(iter(inexact["units"]["t3_unit_C"]["modules"])); inexact["units"]["t3_unit_C"]["modules"][k]["logits"]["exact"] = False
(A / "replay_inexact.json").write_text(json.dumps(inexact))
run("replay_one_inexact_module", lambda: fc.tick(tick_cfg(A / "replay_inexact.json")), "COMBINATION_FINAL_REPLAY_CONTRACT")
# a replay marked FAIL: tick must NOT raise but must record REPLAY_CORRECTION_REQUIRED and never a success; it would then
# scp/disable timers -> we do not execute that path; we assert the decision function alone instead
failed = json.loads(json.dumps(rep)); failed["state"] = "FAIL"
ctl, comb = w.read(m / "in/T3_CONTROL.json"), w.read(m / "out/t3_unit_C/E3M.json")
dec = fc.decide(ctl, comb)
out["decide_reproduced"] = {"state": dec["state"], "pooled": dec["pooled_seed_std"], "ok": dec["state"] == "COMBINATION_SELECTED"}
mut = json.loads(json.dumps(comb)); [t["trading"]["INNER_A"].__setitem__("sharpe_daily_sqrt365", -5.0) for t in mut["trials"]["FULL"].values()]
out["decide_control_retained_on_bad_C"] = {"state": fc.decide(ctl, mut)["state"], "ok": fc.decide(ctl, mut)["state"] == "CONTROL_RETAINED"}
out["replica_config_roots"] = {"C": producer.get("root"), "pins_point_to_root": all(str(k).startswith(str(producer.get("root"))) for k in producer["pins"])}
(A / "CLOSURE_ATTACK.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"cases": len(out), "all_ok": all(v.get("ok") for v in out.values() if isinstance(v, dict) and "ok" in v)}))

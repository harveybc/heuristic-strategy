#!/usr/bin/env python3
"""Autonomous coordinate capacity ladder for the temporal RL lane (Satoshi), run detached on the GPU worker.

Order (same as Musashi's 52_TEMPORAL_CAPACITY_SWEEP, for comparability): core -> heads -> branches -> depth.
Start incumbent = the registered u=2 design (branch 2, core 2, no E1 hidden head / SAC net_arch [2], depth 1,
dropout 0.1, AdamW wd 1e-4), whose E1-T/E2-T results already exist (--baseline).

  S1 core width      c in {8, 16}
  S2 head width      h in {8, 16}      (E1-T hidden head h; SAC actor/critic net_arch [h])
  S3 branch filters  b in {4, 8}
  S4 depth           one more causal core conv (dilation 8), same width

Every candidate = E1-T (BC+DAgger, 3 seeds) then E2-T (SAC and SAC+demos, 3 seeds each = 6 runs), same data,
window, stride, budget, rules and stopping; 2024 is never read inside the ladder. Both candidates of a stage always
run and are compared with the SAME stage-start incumbent on VAL A (2023 S1) only:

  improve    mean VAL A Sharpe over the 6 E2-T runs > incumbent + 1e-8
  paired     >= 4 of the 6 (arm, seed) pairs beat the incumbent's same (arm, seed)       [Satoshi addition]
  gap        max(mean(TRAIN - VAL A Sharpe), 0) <= max(incumbent gap, 0) + 0.25           [declared noise tolerance]
  risk       mean VAL A max drawdown <= incumbent + 2.0 percentage points                  [declared noise tolerance]
  valid      every VAL A run is frequency-valid (>= 3 orders/week, <= 4/day)
  eligible = all five.  REGULARISATION RESCUE [Satoshi addition]: a candidate with improve AND paired AND valid but
  failing gap or risk is re-run ONCE with dropout 0.2 and weight decay 1e-3 and judged by the same rules.
  Among eligible candidates of a stage: highest VAL A mean Sharpe, then lower VAL A drawdown -> new incumbent.

VAL B (2023 S2) is reported for every trial, never used to choose. After S4 the finalist's frozen policies are read
on 2024 ONCE (tools/expert_guided_eval2024_t.py). Resumable: a trial whose E2T.json exists is not re-run; a failed
step is retried once, then the candidate is recorded as FAILED and the ladder continues.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = {"branch": 2, "core": 2, "head": 0, "depth": 1, "dropout": 0.1, "wd": 1e-4}
STAGES = [("S1_core", "core", [8, 16]), ("S2_head", "head", [8, 16]), ("S3_branch", "branch", [4, 8]),
          ("S4_depth", "depth", [2])]
RESCUE = {"dropout": 0.2, "wd": 1e-3}
TOL_GAP, TOL_DD, PAIRED_MIN = 0.25, 2.0, 4


def key(a):
    return f"b{a['branch']}_c{a['core']}_h{a['head']}_d{a['depth']}_do{a['dropout']}_wd{a['wd']}"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ladder:
    def __init__(self, a):
        self.a = a
        self.root = Path(a.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_p = self.root / "LADDER_STATE.json"
        self.state = json.loads(self.state_p.read_text()) if self.state_p.exists() else {"trials": {}, "stages": []}
        self.log = open(self.root / "LADDER_LOG", "a")

    def say(self, msg):
        self.log.write(f"{now()} {msg}\n")
        self.log.flush()

    def save(self):
        self.state_p.write_text(json.dumps(self.state, indent=1))
        self.report()

    def run(self, name, out_json, cmd, cap, limit):
        if Path(out_json).exists():
            return True
        for attempt in (1, 2):
            self.say(f"start {name} attempt {attempt}")
            with open(self.root / f"{name}.log", "w") as fh:
                rc = subprocess.call([self.a.cr, "-q", "-m", cap, "-t", limit, "-n", name[:40], "--", self.a.py] + cmd,
                                     stdout=fh, stderr=subprocess.STDOUT, cwd=self.a.code)
            self.say(f"end {name} attempt {attempt} rc={rc}")
            if Path(out_json).exists():
                return True
        return False

    def trial(self, arch):
        k = key(arch)
        if k == key(BASE):
            d = Path(self.a.baseline)
        else:
            d = self.root / "trials" / k
            common = ["--admitted", f"{self.a.data}/admitted", "--csv-dir", f"{self.a.data}/sel15", "--v4-dir",
                      f"{self.a.data}/v4", "--features", "core15", "--units", "2", "--branch", str(arch["branch"]),
                      "--core", str(arch["core"]), "--depth", str(arch["depth"]), "--head", str(arch["head"]),
                      "--dropout", str(arch["dropout"]), "--wd", str(arch["wd"]), "--skip-2024"]
            ep = ["--max-epochs", "3"] if self.a.smoke else ["--max-epochs", "1000"]
            bud = ["--budget", "2000", "--eval-every", "1000"] if self.a.smoke else []
            ok = self.run(f"e1t_{k}", d / "e1t" / "E1T.json", ["tools/expert_guided_e1t.py", "run", "--out", str(d / "e1t")]
                          + ep + common, "2G", "21600s")
            ok = ok and self.run(f"e2t_{k}", d / "e2t" / "E2T.json", ["tools/expert_guided_e2t.py", "run", "--out",
                                                                     str(d / "e2t"), "--e1-run", str(d / "e1t")] + bud + common,
                                 "3G", "43200s")
            if not ok:
                self.state["trials"][k] = {"arch": arch, "status": "FAILED", "dir": str(d)}
                self.save()
                return None
        m = self.metrics(d, arch)
        self.state["trials"][k] = m
        self.save()
        return m

    @staticmethod
    def metrics(d, arch):
        e2 = json.loads((d / "e2t" / "E2T.json").read_text())
        e1 = json.loads((d / "e1t" / "E1T.json").read_text()) if (d / "e1t" / "E1T.json").exists() else None
        runs = {}
        for k, r in e2["runs"].items():
            t = r["trading"]
            runs[k] = {p: {x: t[p].get(x) for x in ("sharpe_daily_sqrt365", "net_profit_usd", "max_drawdown_pct",
                                                    "orders_per_week", "trades", "valid")} for p in ("TRAIN", "INNER_A", "INNER_B")}
        sh = lambda p: [runs[k][p]["sharpe_daily_sqrt365"] for k in sorted(runs)]  # noqa: E731
        out = {"arch": arch, "status": "DONE", "dir": str(d), "runs": runs,
               "params_e2_actor_plus_critic": e2.get("params_actor_plus_critic"),
               "params_e1": None if e1 is None else e1["model"].get("params"),
               "val_a_sharpe_mean": float(np.mean(sh("INNER_A"))), "train_sharpe_mean": float(np.mean(sh("TRAIN"))),
               "val_b_sharpe_mean": float(np.mean(sh("INNER_B"))),
               "gap_train_minus_val_a": float(np.mean(np.array(sh("TRAIN")) - np.array(sh("INNER_A")))),
               "val_a_dd_mean": float(np.mean([runs[k]["INNER_A"]["max_drawdown_pct"] for k in runs])),
               "val_a_all_valid": all(runs[k]["INNER_A"]["valid"] for k in runs)}
        if e1:
            fin = [r["rounds"][-1]["trading"] for r in e1["seeds"].values()]
            out["e1"] = {p: float(np.mean([f[p]["sharpe_daily_sqrt365"] for f in fin])) for p in ("TRAIN", "INNER_A", "INNER_B")}
        return out

    @staticmethod
    def judge(c, inc):
        pairs = sum(c["runs"][k]["INNER_A"]["sharpe_daily_sqrt365"] > inc["runs"][k]["INNER_A"]["sharpe_daily_sqrt365"]
                    for k in c["runs"] if k in inc["runs"])
        j = {"improve": c["val_a_sharpe_mean"] > inc["val_a_sharpe_mean"] + 1e-8, "paired": pairs >= PAIRED_MIN,
             "pairs_better": int(pairs),
             "gap": max(c["gap_train_minus_val_a"], 0) <= max(inc["gap_train_minus_val_a"], 0) + TOL_GAP,
             "risk": c["val_a_dd_mean"] <= inc["val_a_dd_mean"] + TOL_DD, "valid": c["val_a_all_valid"]}
        j["eligible"] = all(j[x] for x in ("improve", "paired", "gap", "risk", "valid"))
        j["rescue_candidate"] = j["improve"] and j["paired"] and j["valid"] and not (j["gap"] and j["risk"])
        return j

    def go(self):
        self.say("ladder start")
        inc_arch = dict(BASE)
        inc = self.trial(inc_arch)
        if inc is None:
            self.say("FAIL: baseline unreadable")
            return
        closed = [s for s in self.state["stages"] if not s.get("partial") and "incumbent_after" in s]
        done_stages = {s["stage"] for s in closed}
        for st in closed:                          # resume: replay recorded incumbents
            inc_arch = st["incumbent_after"]
        inc = self.state["trials"][key(inc_arch)]
        stages = [(STAGES[0][0], STAGES[0][1], STAGES[0][2][:1])] if self.a.smoke else STAGES
        for name, field, values in stages:
            if name in done_stages:
                continue
            rec = {"stage": name, "incumbent_before": inc_arch, "candidates": []}
            elig = []
            for v in values:
                arch = dict(inc_arch, **{field: v})
                c = self.trial(arch)
                if c is None:
                    rec["candidates"].append({"arch": arch, "status": "FAILED"})
                    continue
                j = self.judge(c, inc)
                entry = {"arch": arch, "judge": j, "val_a": c["val_a_sharpe_mean"], "train": c["train_sharpe_mean"],
                         "val_b": c["val_b_sharpe_mean"]}
                if j["eligible"]:
                    elig.append((c, arch))
                elif j["rescue_candidate"]:
                    ra = dict(arch, **RESCUE)
                    rc = self.trial(ra)
                    if rc is not None:
                        rj = self.judge(rc, inc)
                        entry["rescue"] = {"arch": ra, "judge": rj, "val_a": rc["val_a_sharpe_mean"],
                                           "train": rc["train_sharpe_mean"], "val_b": rc["val_b_sharpe_mean"]}
                        if rj["eligible"]:
                            elig.append((rc, ra))
                rec["candidates"].append(entry)
                self.state["stages"] = [s for s in self.state["stages"] if s["stage"] != name] + [dict(rec, partial=True)]
                self.save()
            if elig:
                best, best_arch = max(elig, key=lambda e: (e[0]["val_a_sharpe_mean"], -e[0]["val_a_dd_mean"]))
                inc_arch, inc = best_arch, best
            rec["incumbent_after"] = inc_arch
            rec["promoted"] = bool(elig)
            self.state["stages"] = [s for s in self.state["stages"] if s["stage"] != name] + [rec]
            self.say(f"{name} done; incumbent {key(inc_arch)}")
            self.save()
        self.state["finalist"] = inc_arch
        self.save()
        if key(inc_arch) != key(BASE):
            d = Path(inc["dir"])
            self.run("eval2024_finalist", d / "EVAL_2024.json",
                     ["tools/expert_guided_eval2024_t.py", "--admitted", f"{self.a.data}/admitted", "--csv-dir",
                      f"{self.a.data}/sel15", "--v4-dir", f"{self.a.data}/v4", "--dir", str(d)], "3G", "7200s")
        (self.root / "LADDER_DONE").write_text(now())
        self.say("ladder done")

    def report(self):
        tr = [t for t in self.state["trials"].values() if t.get("status") == "DONE"]
        L = ["# Escalera de capacidad por coordenadas (Satoshi), DESARROLLO", "",
             "Orden núcleo → cabezas → ramas → profundidad; 3 semillas por brazo; elección solo en VAL A; VAL B reportado; "
             "2024 no se lee dentro de la escalera.", "",
             "| Arquitectura | Parámetros E2 (actor+crítico) | E1 Sharpe TRAIN / VAL A / VAL B | E2 Sharpe TRAIN | E2 Sharpe VAL A | "
             "E2 Sharpe VAL B | Brecha TRAIN − VAL A | DD % VAL A | Válida |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for t in tr:
            e1 = t.get("e1") or {}
            L.append(f"| {key(t['arch'])} | {t.get('params_e2_actor_plus_critic') or '—'} | "
                     f"{e1.get('TRAIN', float('nan')):.2f} / {e1.get('INNER_A', float('nan')):.2f} / {e1.get('INNER_B', float('nan')):.2f} | "
                     f"{t['train_sharpe_mean']:.2f} | {t['val_a_sharpe_mean']:.2f} | {t['val_b_sharpe_mean']:.2f} | "
                     f"{t['gap_train_minus_val_a']:.2f} | {t['val_a_dd_mean']:.1f} | {'sí' if t['val_a_all_valid'] else 'NO'} |")
        L += ["", "## Etapas", ""]
        for s in self.state["stages"]:
            L.append(f"- **{s['stage']}**{' (en curso)' if s.get('partial') else ''}: titular antes {key(s['incumbent_before'])}"
                     + (f" → después {key(s['incumbent_after'])} (promovido: {s.get('promoted')})" if "incumbent_after" in s else ""))
            for c in s["candidates"]:
                if "judge" in c:
                    L.append(f"  - {key(c['arch'])}: VAL A {c['val_a']:.2f}, TRAIN {c['train']:.2f}, VAL B {c['val_b']:.2f}; "
                             f"criterios {json.dumps({k: v for k, v in c['judge'].items() if k != 'rescue_candidate'})}"
                             + (f"; rescate con regularización → VAL A {c['rescue']['val_a']:.2f}, elegible "
                                f"{c['rescue']['judge']['eligible']}" if "rescue" in c else ""))
                else:
                    L.append(f"  - {key(c['arch'])}: FALLÓ")
        if "finalist" in self.state:
            L += ["", f"Finalista: **{key(self.state['finalist'])}**."]
        (self.root / "LADDER_SUMMARY.md").write_text("\n".join(L) + "\n")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(tr)), 4.4))
            x = np.arange(len(tr))
            for k, lab, c in (("train_sharpe_mean", "TRAIN", "tab:blue"), ("val_a_sharpe_mean", "VAL A", "tab:orange"),
                              ("val_b_sharpe_mean", "VAL B", "tab:green")):
                ax.plot(x, [t[k] for t in tr], "o-", color=c, label=lab)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{key(t['arch'])}\n{t.get('params_e2_actor_plus_critic') or ''} p" for t in tr], fontsize=6,
                               rotation=30, ha="right")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylabel("Sharpe E2-T, media de 6 corridas")
            ax.set_title("Escalera de capacidad: TRAIN contra validación (desarrollo)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(self.root / "escalera_capacidad.png", dpi=150)
            plt.close(fig)
        except Exception as e:                    # the plot never blocks the ladder
            self.say(f"plot skipped: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", required=True, help="dir with admitted/, sel15/, v4/")
    ap.add_argument("--baseline", required=True, help="dir with e1t/ and e2t/ of the registered u=2 design")
    ap.add_argument("--code", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--py", default="python")
    ap.add_argument("--cr", default="crispdm-run")
    ap.add_argument("--smoke", action="store_true", help="plumbing only (3 epochs, 2k transitions, one candidate)")
    Ladder(ap.parse_args()).go()


if __name__ == "__main__":
    main()

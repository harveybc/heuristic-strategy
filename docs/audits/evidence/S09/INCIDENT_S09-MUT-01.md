# INCIDENT_S09-MUT-01: a gate mutant launched real phase-D NEAT training on worker_b

Satoshi, successor technical lead, lane M05. Written 2026-10-01. Hosts are named by
role only.

## Timeline (worker_b local time, 2026-10-01)

| Time | Event |
|---|---|
| about 00:04 | The S09 mutant run reaches mutant `d_entry`. It replaces `if not entry_gate_for_runner("run_phase_d_neat").allowed:` with `if False:` in a scratch copy of this branch, and runs `tests/unit_tests/test_runner_naive_gates.py` under `crispdm-run -m 2G -t 15m`. |
| 00:04–00:19 | With the gate broken, `run_phase_d_neat.main()` continues to `run_neat_optimization()`. That starts a predictor subprocess with `cwd` set to worker_b's `~/Documents/GitHub/predictor` (the campaign's clone, on `master`), which runs a NEAT optimization. |
| 00:09–00:18 | That optimization writes files under `examples/results/phase_1c_direction/phase_d/` in the clone. |
| about 00:19 | `crispdm-run`'s 15-minute wall stops the job, and its scope ends. No process remains (checked afterwards). The remaining eleven mutants run in about 1 s each. |

## Why the test did not stop it

`test_phase_runners_evaluate_zero_times_without_evidence` checked only that the strategy
was never evaluated and that data was never loaded: it patched `app.data_handler.load_csv`.
Phase D runs NEAT optimization before it loads data. Nothing after the gate was
sandboxed, so a broken gate let real external work start. The test still failed in the
end, which means the mutant counted as killed. The real fault is that it failed only
after the side effects had already happened.

## What was touched

All paths are in worker_b's predictor clone, under
`examples/results/phase_1c_direction/phase_d/`:

| Path | Effect |
|---|---|
| `phase_d_ann_direction_long_optimization_candidate_history.csv` | tracked, modified (mtime 00:17) |
| `phase_d_ann_direction_long_optimization_parameters.json` | tracked, modified (mtime 00:09) |
| `phase_d_ann_direction_long_optimization_resume.json` | tracked, deleted |
| `phase_d_ann_direction_long_rss.csv` | gitignored, written at 00:18; whether it existed before is **unknown**. Left in place. |

`git diff --stat` reported 3 files changed, 21 insertions and 1203 deletions. A search for
files changed in the clone in the 40 minutes before inspection found nothing else (only
`__pycache__`).

## Backup and restoration

- **Backup:** the three files as found, plus `git_diff_stat.txt`, are in worker_b
  `~/.local/state/scratch/m05/hs/runtime/d_entry_mutant_damage/`. They are kept until
  the owner rules.
- **Restoration:** `git checkout -- <the three tracked paths>` in the clone. Its scope
  was those three paths only. The clone's status afterwards: one untracked file from
  April that is not mine.
- **Ruling (coordinator):** the restoration stands, but from now on `git checkout --`,
  `reset` and `clean` are never run in a checkout I do not own. The procedure is: back
  up, stop, report, and let the coordinator decide. The modification times placed all
  three changes inside the mutant's window, but I could not prove the tracked files had
  no earlier uncommitted edits. The backup is what makes this recoverable.

## A second hazard: sweep_noise ran HTTP at import time

The first red run of this suite, with the gate absent by design, imported `sweep_noise.py`.
Its module-level code changed directory into the repository's main checkout path, loaded
data, and attempted an HTTP POST `set_noise` to `127.0.0.1:8000` on worker_b. Nothing
was listening, so the connection was refused and had no effect. Nothing was written.

**Fix:** `sweep_noise.py` now does nothing on import. `main()` decides the gate first; then
comes the data, then the service call. The candidate gate now sits before the per-level
POST, and the hard-coded main-checkout path is gone.

## Prevention

Every entry-point test now sandboxes the work that follows the gate. Each of these
raises if reached: `subprocess.Popen`/`run`, NEAT optimization and champion loading,
model loading and prediction generation, data loading, the oracle body, and HTTP.
A broken gate therefore fails fast with no side effects.

`test_sandbox_stops_phase_d_at_its_first_post_gate_side_effect` proves it. It applies
the `d_entry` mutant's effect (a gate that always allows), requires `main()` to raise
at its first post-gate side effect within 1 s, and asserts that
`git status --porcelain` of the predictor clone is identical before and after.
It passes on worker_b.

## Mutant rerun after the fix

Each mutant ran with `crispdm-run -m 2G -t 2m` on worker_b. The predictor clone's
status was unchanged at the end.

| Mutant | Result |
|---|---|
| wfo_entry | 1 failed, 16 passed |
| wfo_library | 1 failed, 16 passed |
| b_entry | 1 failed, 16 passed |
| c_entry | 1 failed, 16 passed |
| d_entry | 1 failed, 16 passed (0.95 s) |
| b_candidate | 1 failed, 16 passed |
| c_candidate | 1 failed, 16 passed |
| d_candidate | 1 failed, 16 passed |
| sweep_entry | 1 failed, 16 passed |
| api_plugin | 1 failed, 16 passed |
| oracle_flag | 1 failed, 16 passed |
| suppressed_candidate_skip | 6 failed, 11 passed |

Every mutant is killed, and none ran longer than about 1.4 s.

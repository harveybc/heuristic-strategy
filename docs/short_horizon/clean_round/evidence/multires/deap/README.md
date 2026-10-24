# Multi-resolution threshold optimization (DEAP NSGA-II) — run record

Tool: `tools/expert_guided_multires_deap.py run` (objectives, budget, convergence rule, choice rule and the hindsight
limitation are in its docstring and copied into `FINAL.json`). Pre-registered budget: population 48, up to 200
generations, convergence = hypervolume change < 1 % over the last 20 generations after >= 80, seeds 1/2/3, SBX eta 15 /
polynomial mutation eta 20, crossover probability 0.9. Replays are label-only (~0.2 s each on CPU).

Run location: the secondary GPU host (CPU only, 3 forked workers under a 4 GiB cap; the GPU is busy with Stage A of
the clean round), directory `<root>/multires-deap-20261024/` with `in/` (read-only hard links of the admitted bars and
sel15), `code/` (this branch at the commit named in `RUN.json`), `out/` (FINAL.json, CHOSEN_THRESHOLDS.json,
READ_LEDGER.jsonl, CLOSURE_TABLE.md, CACHE.json, seed_<s>.json) and `deap.log`.

Collect: copy `out/*` here when `deap.log` ends with the `{"front": …}` line; `CHOSEN_THRESHOLDS.json` is the file an
arm passes as `--multires-thresholds`. Resume after a crash: rerun the same command — finished seeds (`seed_<s>.json`
with a `stop`) are skipped and `CACHE.json` avoids re-simulating known threshold vectors.

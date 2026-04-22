# Part III Gate Template

> Copy this file for each Part III task gate. Fill all sections before
> declaring **Pass**. A gate is **Fail** if any assertion is unchecked or any
> artifact is missing.

---

## Gate: `<task_id>` — `<short title>`

**Scope:** `<1-line objective>`
**Owner:** `<agent / operator>`
**Date (UTC):** `<YYYY-MM-DDTHH:MM:SSZ>`
**Git SHA (agent-multi):** `<sha>`
**Git SHA (gym-fx):** `<sha>`

---

### 1. Context

- **Upstream gates passed:** `<task_ids>`
- **Decision refs:** `<PART_III_EXECUTION_PLAN §x.y>`
- **Inputs consumed:** `<data/config paths>`
- **Exclusions (confirmed out of scope):** `<…>`

### 2. Commands run

```bash
# Exact commands in execution order (copy/paste-able)
<cmd 1>
<cmd 2>
...
```

### 3. Artifacts produced

| Path | Bytes | SHA256 (first 12) | Purpose |
|------|------:|-------------------|---------|
| `logs/partIII/<run_id>/config.json`   |  |  |  |
| `logs/partIII/<run_id>/summary.json`  |  |  |  |
| `logs/partIII/<run_id>/policy.zip`    |  |  |  |
| `logs/partIII/<run_id>/train.log`     |  |  |  |
| `logs/partIII/<run_id>/git_sha.txt`   |  |  |  |

### 4. Assertions checked

- [ ] Train window contains zero rows with `DATE_TIME >= 2020-01-01` (HO hard-boundary).
- [ ] Validation/d5 used at most for `ga_fitness_split=val`; never for backprop.
- [ ] d6 (held-out) untouched by this gate unless explicitly a held-out eval gate.
- [ ] Seeds run match plan (default `{0, 1, 2}`).
- [ ] `logs/partIII/index.csv` updated via `tools/update_registry.py`.
- [ ] All console errors in `train.log` are non-fatal (exit code 0 verified).
- [ ] `<task-specific assertion 1>`
- [ ] `<task-specific assertion 2>`

### 5. Metrics (per seed)

| seed | total_return | sharpe | max_dd | trades | episode_reward | notes |
|-----:|-------------:|-------:|-------:|-------:|---------------:|-------|
|   0  |              |        |        |        |                |       |
|   1  |              |        |        |        |                |       |
|   2  |              |        |        |        |                |       |

Aggregate (mean ± std across seeds): `<…>`

### 6. Pass / Fail

- **Verdict:** `PASS` / `FAIL`
- **Reasoning:** `<why>`
- **Follow-ups created (if any):** `<task_id or N/A>`

### 7. Signoff

- **Operator:** `<name>`
- **Reviewer (optional):** `<name>`
- **Time closed (UTC):** `<YYYY-MM-DDTHH:MM:SSZ>`

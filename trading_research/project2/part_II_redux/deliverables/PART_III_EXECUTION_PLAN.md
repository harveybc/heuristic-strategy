# Project 2 — Part III Execution Plan

**Status:** Active — approved 2026-04-21
**Supersedes:** training-regime section of `PART_III_SCOPE_RECOMMENDATION.md` (which said "Main train 2M on 2020-2022" — incompatible with II-7-fix Rule 0.2)
**Honors:** `Project2_stage_ii_7_fix.md` §Rule 0.2 (held-out = 2020-01-01 → 2025-12-31)

---

## Decisions (locked by user 2026-04-21)

1. **d5 validation window policy:** **B (moderate)** — d5 = 2020-2022 is accessed **only** for GA fitness scoring and algorithm validation; never used to update policy weights. d6 = 2023-2025, touched exactly once per final candidate.
2. **Loader integration:** **A** — `agent-multi/data/project2_manifest.json` references absolute paths inside `heuristic-strategy/trading_research/project2/part_II_redux/data/`. No copies, no symlinks.
3. **Macro features on FX:** **yes** — add SPY + VIX (forward-filled) and FRED daily macro (forward-filled) to the FX feature set.
4. **SAC:** **included** as a tertiary control (BTC 1h only per II-7 evidence showing ETH degeneracy).
5. **NEAT 5-stage:** **deferred** to post-Project-2 (P6 unscheduled).
6. **ATR SL/TP strategy home:** gym-fx `strategy_plugins/direct_{fixed,atr}_sltp.py`. Plugin exposes `apply_action(strategy, action)` consumed by `BTBridgeStrategy`. Config-driven `k_sl`, `k_tp`, `atr_period` are GA-tunable. *(2026-04-21)*
7. **Feature attachment:** static prep presets. `prepare_project2_data.py --features {twelve,twelve_macro,twelve_funding,twelve_onchain}` merges macro/funding/on-chain at prep time; runtime `features.plugins` group not added. *(2026-04-21)*
8. **Normalization:** `prepare_project2_data.py --normalize` fits `StandardScaler` on d4 only, emits `*_norm.csv` + `scaler.json`. CVAE (P3) reads these directly; preprocessor repo bypassed. *(2026-04-21)*

---

## Hard invariants

1. Training data must end ≤ 2019-12-31. `phase_dataset_plugin` refuses any `split="train"` request touching rows ≥ 2020-01-01.
2. d5 (2020-2022) used **only** by the GA optimizer's fitness callback and by the algorithm's validation loop — never backpropagated.
3. d6 (2023-01-01 → 2025-12-31) touched exactly once per final candidate per algorithm in P5.
4. All runs produce a self-contained JSON under `part_II_redux/logs/partIII/<run_id>/`; `logs/partIII/index.csv` is the master registry.
5. Seeds `{0, 1, 2}` minimum per (asset, algo) cell.

---

## Machine allocation

| Machine | GPU | Role |
|---|---|---|
| **Omega** (local) | RTX 4070 12GB | Data/features/eval/reports; FX PPO (lighter GPU load than crypto) |
| **Dragon** 192.168.0.107:62024 | RTX 4090 24GB | Primary PPO BTC+ETH; CVAE training |
| **Gamma** 192.168.0.106:62024 | RTX 5070Ti 16GB | DQN BTC+ETH; parallel GA hparam search; SAC BTC |

Remote execution: `ssh <host> "bash -ic '<cmd>'"` (per `/memories/remote-machines.md` — `.bashrc` auto-activates `tensorflow` env).

---

## Phases

### P0 — Prep & environment verification (Omega)

- P0.1 Save this document to `deliverables/PART_III_EXECUTION_PLAN.md`.
- P0.2 Verify SSH + conda on Dragon, Gamma, Omega.
- P0.3 Pull latest `agent-multi`, `gym-fx`, `feature-extractor`, `preprocessor`, `feature-eng` on each machine via `git pull`.
- P0.4 `pip install -e .` for each repo on each machine (no venv churn — tensorflow env is shared).
- P0.5 Run `agent-multi --load_config examples/config/ppo_gymfx_default.json --total_timesteps 300 --quiet_mode` on each machine; must exit 0.

**Gate:** `P0_environment_verification.md` with three smoke-test logs.

### P1 — II-7-fix execution (Dragon + Gamma + Omega)

Execute `Project2_stage_ii_7_fix.md` tasks II-7-fix.1 through II-7-fix.5 unchanged. Parallelization:
- II-7-fix.1 pilot-bug investigation → Omega (code reading).
- II-7-fix.2 causal temporal stability → Dragon (memory-heavy PCMCI+).
- II-7-fix.3 cross-asset independence → Dragon.
- II-7-fix.4 pilot re-runs with strict IS-only discipline → Dragon=BTC, Gamma=ETH, Omega=FX.
- II-7-fix.5 scope re-write → Omega.

**Gate:** II-7-fix.4 pilots produce non-identical metrics across algorithms (the original bug signature); no `ESCALATION_identical_metrics_*` files remain.

### P2 — agent-multi data layer (Omega) — **P2.1-P2.6 SHIPPED 2026-04-21**

All work inside `/home/harveybc/Documents/GitHub/agent-multi`. Shipped commit: `feat(p2): project2 data layer — manifest, prepare script, 3 pilot configs`.

**Pragmatic pivot vs. original design:** the original plan introduced new entry-point groups (`data.plugins`, `features.plugins`, `strategy.plugins`, `feature_selector.plugins`) inside agent-multi. gym-fx already owns `strategy.plugins`, `data_feed.plugins`, etc., and naming collision would require namespacing. We pivoted to a **static prep-time approach**: `tools/prepare_project2_data.py` materializes per-source / per-split / per-feature CSVs once, and gym-fx's existing `default_data_feed` consumes them. Config switches the feature set by pointing `input_data_file` at a different preset CSV. *(Decision 7, 2026-04-21)*

Shipped artifacts:
- `agent-multi/data/project2_manifest.json` — 4 sources (eurusd/usdjpy/btcusdt/ethusdt 1h), absolute paths into part_II_redux/data/, `train_start` per source, HO boundary 2020-01-01 locked.
- `agent-multi/tools/prepare_project2_data.py` — loads CSV/parquet, normalizes column schema, computes 12-feature II-7.3 set, emits `data/project2/<source>/d4.csv d5.csv d6.csv`. Hard `_split` guard raises `RuntimeError` if d4 intersects 2020-01-01.
- `agent-multi/examples/config/ppo_{btc,eth,eurusd}_1h_*.json` — three pilots using `default_strategy` + 12-feature d4 inputs. Validated end-to-end with 2k-step smoke runs.
- `.gitignore` — `data/project2/*/*.csv` and `examples/results/p2_pilot/` excluded.

**Gate ✅:** d4 max dates for all 4 sources confirmed < 2020-01-01 (BTC 2019-12-31 23:00; EURUSD 2019-12-31 22:00; USDJPY 2019-12-31 21:00; ETH 2019-12-31 23:00). Pilot JSON summaries produced for BTC (8713 bars) and EURUSD (93187 bars).

### P2b — agent & strategy gap-fill (Omega)

The P2 pragmatic pivot left 4 capabilities underdone that P3/P4 require. P2b fills them before P3 begins.

- **P2b.1** `agent-multi/agent_plugins/dqn_agent.py` — SB3 DQN, discrete action space, Dict→flat observation wrapper internal to the plugin (via `gymnasium.wrappers.FlattenObservation`). Same plugin contract as `ppo_agent` (`build / train / predict / save / load / fitness / hparam_schema`). Registered in `setup.py` under `agent.plugins`.
- **P2b.2** `agent-multi/agent_plugins/sac_agent.py` — SB3 SAC, continuous `Box(-1, +1, shape=(1,))` action space; requires env continuous mode.
- **P2b.3** `gym-fx/app/env.py` — add `action_space_mode ∈ {discrete, continuous}` config key (default `discrete`). In continuous mode, action is thresholded at ±0.33 → {-1, 0, +1} inside `BTBridgeStrategy` (Option A, apples-to-apples with discrete baselines per plan decision).
- **P2b.4** `gym-fx/strategy_plugins/direct_fixed_sltp.py` + `direct_atr_sltp.py` — implement new contract method `apply_action(bt_strategy, action, config)` called from `BTBridgeStrategy._apply_action` when plugin provides it. Fixed variant uses `buy_bracket / sell_bracket` at ±`{sl_pips, tp_pips}`. ATR variant maintains a rolling TR buffer and places brackets at `k_sl · ATR / k_tp · ATR`. `atr_period`, `k_sl`, `k_tp` exposed in plugin `plugin_params` and `hparam_schema` for GA tuning.
- **P2b.5** Update three pilot configs to `strategy_plugin=direct_atr_sltp`; re-run 50k-step smoke; assert non-zero trade count in each.
- **P2b.6** Extend `prepare_project2_data.py`:
  - `--features {twelve, twelve_macro, twelve_funding, twelve_onchain}` (default `twelve`). Macro: SPY daily close + VIX close + FRED CPI/UNRATE forward-filled to 1h with 7-day cap. Funding: Binance `funding_*.csv` forward-filled (0 pre-launch). Onchain: Coinmetrics `AdrActCnt, TxCnt, HashRate` daily forward-filled.
  - `--normalize` — fits `sklearn.preprocessing.StandardScaler` on d4 only, applies to d4/d5/d6, emits `d4_norm.csv d5_norm.csv d6_norm.csv` + `scaler.json`.

**Gate P2b:** 3 pilots × 50k steps with `direct_atr_sltp` produce non-zero trade counts and distinct metrics; `--features twelve_macro --normalize` on EURUSD produces non-empty `d4_norm.csv` with SPY/VIX/FRED columns present and d4 max date still < 2020-01-01.

### P2c — run infrastructure (Omega, parallel with P2b.1-4)

- **P2c.1** `agent-multi/tools/seed_sweep.py` — loads a base config, iterates a seed list (default `[0, 1, 2]`), writes each run to `logs/partIII/<run_id>/` where `run_id = <asset>_<algo>_<features>_<strategy>_s<seed>_<utc>`. Each run directory holds `config.json, summary.json, policy.zip, train.log, git_sha.txt`.
- **P2c.2** `agent-multi/tools/update_registry.py` — appends one row per completed run to `logs/partIII/index.csv` with columns `run_id, asset, algo, features, strategy, seed, started_at, finished_at, total_return, sharpe, max_dd, trades, config_hash, git_sha`. Idempotent on `run_id`.
- **P2c.3** `agent-multi/optimizer_plugins/candidate_worker.py` — add d5-fitness hook: after `agent.train()` completes on d4, load the checkpoint into a **weights-frozen** evaluation env configured for the d5 date range, run a single deterministic rollout, return `agent.fitness(d5_summary)` as the GA fitness. d5 is never seen by `model.learn`. Controlled by config key `ga_fitness_split ∈ {train, val}` (default `val` when the d5 path is resolvable from the manifest).

**Gate P2c:** 3-seed sweep of `ppo_btc_1h_twelve_atr.json` produces 3 distinct `logs/partIII/*/` folders and 3 rows in `index.csv`; a 2-generation GA on the same config returns fitness values that differ from the matching training summary `total_return` (proves d5 hook fires).

### P2d — gate deliverable template (Omega, 1 commit)

- **P2d.1** `heuristic-strategy/trading_research/project2/part_II_redux/deliverables/_gate_template.md` — canonical structure (Context → Commands run → Artifacts produced → Assertions checked → Pass/Fail → Signoff). Required format for all P0/P2b/P2c/P3/P4/P5 gate docs.

### P3 — CVAE feature extractor (Dragon)

- P3.1 On Omega: `python tools/prepare_project2_data.py --source btcusdt_1h --features twelve --normalize` → emits `d4_norm.csv / d5_norm.csv / d6_norm.csv` + `scaler.json`. *(Replaces the original preprocessor-run step per Decision 8.)*
- P3.2 `rsync` normalized CSVs + scaler to Dragon; `feature-extractor` CVAE training (latent_dim=32, window=288, trained on `d4_norm.csv` only).
- P3.3 Repeat for ETH 1h.
- P3.4 New agent-multi group `feature_extractor.plugins` + `passthrough_extractor` (default) + `cvae_extractor` (loads `.h5`, inserts latent vector into the observation Dict).
- P3.5 Smoke config `ppo_btc_1h_cvae_atr.json` run on Omega (100k steps).

**Gate P3:** CVAE reconstruction loss on `d5_norm.csv` not exceeding 1.1× loss on held-out fold of `d4_norm.csv`; cvae smoke exits 0.

### P4 — Full-scale RL training (Dragon + Gamma in parallel)

Five training jobs, each 2M steps × 3 seeds. Training window 2005-2019 (FX) or 2019 (crypto, matches II-7 IS).

| Machine | Job | Asset | Algo | Features |
|---|---|---|---|---|
| Dragon | J1 | BTC 1h | PPO | project2_twelve |
| Dragon | J2 | ETH 1h | PPO | project2_twelve |
| Gamma | J3 | BTC 1h | DQN | project2_twelve |
| Gamma | J4 | ETH 1h | DQN | project2_twelve |
| Gamma | J5 | BTC 1h | SAC | project2_twelve (BTC only per II-7 evidence) |
| Omega | J6 | EURUSD 1h | PPO | project2_twelve_plus_macro |

**Parallel GA search** (overlaps P4 training days): Gamma runs `ga_population=8`, `ga_generations=8`, `ga_eval_timesteps=100k` on BTC 1h PPO over {learning_rate, n_steps, batch_size, gamma, gae_lambda, clip_range, ent_coef, obs_window, k_sl, k_tp}. Fitness measured on d5 (2020-2022).

**Gate:** `P4_training_results.md` — 6 jobs × 3 seeds = 18 policies + checkpoints; no identical metrics across seeds (sanity).

### P5 — Held-out evaluation + final report (Omega)

- P5.1 For each of 6 best candidates (GA winner + fixed-hparam control), load policy, run single deterministic rollout on d6 = 2023-01-01 → 2025-12-31.
- P5.2 Metrics: Sharpe, Sortino, Calmar, max DD, win rate, transaction cost ratio, trade count, regime breakdown, equity curve.
- P5.3 Bootstrap CI (1000× trade-level resample).
- P5.4 Compare vs B&H, random, Part II `regime_adaptive` (Sharpe +0.068).
- P5.5 Exit-gate checklist per `PART_III_SCOPE_RECOMMENDATION.md` KPI table (HO Sharpe ≥ 0.5 minimum, ≥ 1.5 target; HO Max DD < 30%; win rate > 48%; TC ratio > 1.5×).
- P5.6 Write `PART_III_FINAL_REPORT.md`.

**Gate:** final report + `logs/partIII/final_HO_results.json`.

---

## Execution schedule (sequential)

P0 serial, P1 serial with internal parallelism, P2 serial, P3 serial, P4 Dragon+Gamma+Omega **parallel**, P5 serial.

Each phase has a user gate — pause for approval before next phase.

## Deliverables index

- `P0_environment_verification.md`
- II-7-fix tasks 1-5 deliverables (existing contract)
- `P2_data_layer_validation.md`
- `P3_cvae_training.md`
- `P4_training_results.md`
- `PART_III_FINAL_REPORT.md` + `logs/partIII/final_HO_results.json`

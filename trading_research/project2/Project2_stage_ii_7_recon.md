# Project 2 — Stage II-7: RL Configuration Reconnaissance Plan

**Purpose:** Before committing Part III compute to specific RL configurations, perform tree pruning across the 6-dimensional design space (data, features, algorithm, reward, action, training) using cheap reconnaissance. Output is evidence-based Part III scope recommendation.

**Status of Project 2 entering this stage:** Part II-Redux closed as null (all 4 IS-passing strategies failed held-out validation per Stage II-6). Path C (RL) was deferred. Stage II-7 corrects the Part III premise: instead of assuming we know the right RL configuration, we screen broadly and cheaply before commitment.

**Why this stage exists:** The user identified the chicken-and-egg problem of RL — features/timeframe/asset can't be determined a priori, but full Part III on every combination is intractable. Stage II-7 is the middle ground: cheap roots, then commit Part III to surviving branches.

---

## 0. Standing Rules (preserved from prior stages)

1. **No synthetic data.** Only real Binance/CoinMetrics/Blockchain.com data.
2. **Pre-registered gates.** Each task has explicit success criteria.
3. **Each task has user gate.** Agent reports and waits before next task.
4. **Escalation protocol active.** If gate fails or unexpected blocker, halt with `ESCALATION_II-7_*.md`.
5. **No held-out contamination.** New data acquired in this stage respects same IS/HO split as Part II-Redux: IS through 2019-12-31, HO 2020-01-01 onward.
6. **Agent contract from Part II-Redux applies.** All credentials handled via env vars, never committed.
7. **Compute distribution principle:** Dragon for heaviest, Gamma for medium-fast, Omega for lightweight/synthesis.

---

## 1. Stage Structure (Three Phases)

| Phase | Purpose | Primary Machines | Output |
|-------|---------|------------------|--------|
| **Phase 1** | Branch pruning via causal/feature analysis (no RL training) | Omega + Dragon + Gamma | Identifies viable data/feature combinations |
| **Phase 2** | Minimum viable RL pilots on surviving branches | Dragon + Gamma | Identifies algorithms that show learning |
| **Phase 3** | Part III scope synthesis | Omega | Part III plan recommendation document |

Phase 1 is data-driven analysis. Phase 2 is small-scale RL learnability check. Phase 3 is decision based on Phase 1+2 evidence.

---

## 2. The Configuration Tree (Reference)

For context — what we're pruning:

```
Branch 1 (DATA × TIMEFRAME):
  1A: BTC/USD 5m   1B: BTC/USD 15m   1C: BTC/USD 1h   1D: BTC/USD 4h
  1E: ETH/USD 5m   1F: ETH/USD 15m   1G: ETH/USD 1h   1H: ETH/USD 4h

Branch 2 (FEATURE SET):
  2A: F-6 technical only (Part II-Redux baseline)
  2B: F-6 + volume/microstructure
  2C: F-6 + funding rate (perpetuals)
  2D: F-6 + on-chain (CoinMetrics Community)
  2E: F-6 + 2B + 2C + 2D combined
  2F: Minimal state (OHLCV + position info only)

Branch 3 (ALGORITHM):
  3A: PPO        3B: SAC        3C: DQN/Rainbow
  3D: Offline RL (CQL/IQL)      3E: NEAT simple-connected (Stage II-7 pilot only)
  3F: NEAT advanced (deferred to Part III capstone — out of Stage II-7 scope)

Branch 4 (REWARD):
  4A: Immediate PnL    4B: Differential Sharpe (Jansen)
  4C: Drawdown-penalized PnL    4D: Cost-aware (heavy trade penalty)
  4E: Causal counterfactual (out of Stage II-7 scope, Part III if relevant)

Branch 5 (ACTION SPACE):
  5A: Discrete {long, flat, short}
  5B: Discrete sized {±large, ±small, flat}
  5C: Continuous position ∈ [-1, +1]
  5D: Hierarchical (regime + sizing) — out of Stage II-7 scope

Branch 6 (TRAINING SCHEMA):
  6A: Fixed single training window
  6B: Rolling retraining (Project 2 style)
  6C: Continual/online learning
  6D: Curriculum learning
```

Stage II-7 prunes branches 1, 2, and validates that branches 3-6 contain at least some viable configurations.

---

## 3. PHASE 1: Branch Pruning

### 3.1 Task II-7.1: Data Acquisition Extension

**Purpose:** Acquire all data needed for Phase 1 analysis. Free sources only (no new credentials).

#### II-7.1.a Binance OHLCV across timeframes

For BTC/USDT and ETH/USDT, fetch via public Binance klines API:

| Asset | Timeframes | Period |
|-------|-----------|--------|
| BTC/USDT | 5m, 15m, 1h | 2019-01-01 to present (held-out boundary at 2020-01-01) |
| ETH/USDT | 5m, 15m, 1h, 4h | 2019-01-01 to present |

(BTC 4h already in Part II-Redux data.)

5m data is large: ~2M bars per asset over 6 years. Agent uses paginated chunks, stores as Parquet for efficiency.

Output: `data/raw/binance/{asset}_{timeframe}_2019_2025.parquet`

#### II-7.1.b Binance funding rates (perpetual futures)

```python
# Endpoint: https://fapi.binance.com/fapi/v1/fundingRate
# No credentials needed
```

For BTCUSDT and ETHUSDT perpetual futures:
- Historical funding rates 2019-09-01 (Binance perps launch) to present
- Update frequency: every 8 hours
- Fields: symbol, funding_rate, funding_time

Output: `data/raw/binance/funding_{asset}_2019_2025.csv`

#### II-7.1.c CoinMetrics Community on-chain metrics

Free tier covers:
- Active addresses (daily)
- Transaction count
- Mean transaction fee
- Realized cap
- NVT ratio
- Difficulty
- Hash rate

```python
# CoinMetrics community API (no key required, rate-limited)
# https://docs.coinmetrics.io/api/v4
# Free metrics list: https://docs.coinmetrics.io/asset-metrics
```

For BTC and ETH, fetch all available free daily metrics 2019-2025.

Output: `data/raw/coinmetrics/{asset}_daily_metrics_2019_2025.csv`

#### II-7.1.d Blockchain.com supplementary

Free metrics not in CoinMetrics:
- Mempool size
- Confirmed transactions per block
- Estimated hash rate (cross-validation with CoinMetrics)

Output: `data/raw/blockchain_com/btc_metrics_2019_2025.csv` (BTC only)

#### II-7.1.e Data validation

Apply same 6-test validation from Stage II-0b to all newly acquired data:
- Bar counts realistic
- No GBM signature
- Fat-tail kurtosis present
- Volatility clustering
- etc.

Output: `data/validation/II-7_data_validation.md`

### 3.2 Task II-7.2: Multi-Timeframe Causal Analysis

**Purpose:** Apply PCMCI+ analysis (same methodology as F-6 and Stage II-0.5) to all new data combinations to identify which timeframes/feature sets show lagged causal structure.

#### II-7.2.a Run matrix

| Run | Asset | Timeframe | Feature set | Priority |
|-----|-------|-----------|-------------|----------|
| 1 | BTC | 5m | 12 technical | HIGH |
| 2 | BTC | 15m | 12 technical | HIGH |
| 3 | BTC | 1h | 12 technical | HIGH |
| 4 | BTC | 4h | 12 technical + funding | HIGH (extends Part II finding) |
| 5 | BTC | 4h | 12 technical + on-chain | HIGH |
| 6 | BTC | 4h | 12 technical + funding + on-chain | HIGH (full feature set) |
| 7 | BTC | 1h | 12 technical + funding | MEDIUM |
| 8 | BTC | 15m | 12 technical + funding | MEDIUM |
| 9 | ETH | 5m | 12 technical | HIGH |
| 10 | ETH | 15m | 12 technical | HIGH |
| 11 | ETH | 1h | 12 technical | HIGH |
| 12 | ETH | 4h | 12 technical | HIGH |
| 13 | ETH | 4h | 12 technical + funding | MEDIUM |
| 14 | ETH | 4h | 12 technical + on-chain | MEDIUM |

14 runs total. Same PCMCI+ parameters as Part II-Redux Stage II-0.5:
- Algorithm: PCMCI+ (tigramite)
- Independence test: ParCorr
- τ_max = 10
- pc_alpha = 0.01
- alpha_level = 0.05
- Target: 6-bar forward log return
- Sample limit: 5000

#### II-7.2.b Feature alignment (critical for cross-timeframe)

- For BTC/USD 4h with funding rate (8h frequency): forward-fill funding to 4h bars, no look-ahead
- For BTC/USD 4h with on-chain (daily frequency): forward-fill daily metrics to 4h bars
- For 1h, 15m, 5m timeframes: same forward-fill principle

Important: Funding rate at time t reflects funding paid AT time t for the period ending t. Use only after confirmation, not at announcement time.

#### II-7.2.c Classification

Per Stage II-0.5 criteria:
- **α:** ≥1 lagged link with |MCI| > 0.10 and p < 0.01
- **β:** ≥1 lagged link with |MCI| ∈ [0.05, 0.10] and p < 0.05
- **γ:** No lagged links with |MCI| > 0.05

#### II-7.2.d Compute distribution

- **Dragon:** Runs 1-7 (BTC variants, larger samples for short timeframes)
- **Gamma:** Runs 9-14 (ETH variants)
- **Omega:** Runs 8 (BTC 15m + funding) plus orchestration

Phase 1 BTC 4h baseline (run 4) comparison: known α with RSI t-6 from Stage II-0.5. Confirm reproducibility.

#### II-7.2.e Deliverable

`TASK_II-7.2_MULTI_TIMEFRAME_CAUSAL.md`:
- Per-run results (links, MCI strengths, p-values)
- Classification α/β/γ per run
- Cross-asset cross-timeframe patterns
- Surviving (asset, timeframe, feature set) configurations for Phase 2

### 3.3 Task II-7.3: Information Coefficient Analysis Across Surviving Branches

**Purpose:** Causal analysis identifies presence of lagged structure but not magnitude of predictive utility. IC analysis adds the second perspective.

#### II-7.3.a Procedure

For each (asset, timeframe, feature set) that survived II-7.2 with α or β classification:

1. Compute Information Coefficient per feature: Spearman rank correlation of feature_t with forward_return_t+h
2. Multiple horizons: 1, 6, 12, 24 bars
3. Rolling 1-year IC: mean, std, IR (IC/std)

#### II-7.3.b Filter criterion

Features with rolling IC IR > 0.3 sustained are "robust predictive features."

#### II-7.3.c Deliverable

`TASK_II-7.3_IC_ANALYSIS.md`: per surviving config, list of features with IR > 0.3 and their IC magnitude. This is feature universe for Phase 2 RL pilots.

### 3.4 Phase 1 Gate

Phase 1 produces a pruned tree:
- Surviving (asset, timeframe) combinations from II-7.2 (probably 2-5)
- Per surviving combination, robust feature subset from II-7.3 (probably 5-10 features)

If Phase 1 finds **all γ** across new data (analogous to Part II-Redux), escalate:
- Strong evidence retail-accessible crypto has no exploitable predictive structure
- Part III RL likely null-equivalent
- User decides: proceed to Phase 2 anyway as final test, or close Project 2

If Phase 1 finds at least one new α or β not in Part II-Redux (e.g., BTC 15m or ETH 4h), proceed to Phase 2 with promising configurations.

---

## 4. PHASE 2: Minimum Viable RL Pilots

### 4.1 Task II-7.4: RL Framework Setup

**Purpose:** Standardize RL infrastructure before pilots. Avoid each pilot reinventing wheel.

#### II-7.4.a Framework selection

Use **Stable-Baselines3** as primary RL framework:
- Mature, well-documented
- Implements PPO, SAC, DQN, A2C, TD3, DDPG
- Active community, frequent updates
- TensorFlow-compatible (your env)

Alternative: **CleanRL** for simpler experiments and easier debugging. Both can be installed alongside.

```bash
# On all machines:
pip install stable-baselines3[extra] gymnasium
```

#### II-7.4.b Trading environment design

Build minimal `gymnasium.Env` subclass for Phase 2 pilots. Located at `infrastructure/rl/trading_env.py`.

Specification:

**Observation space:** Box of shape (lookback × num_features), where:
- lookback configurable (default 24 bars)
- num_features from Phase 1 IC-filtered feature subset
- Plus position state (current position, unrealized PnL, time-in-position, recent action)

**Action space:** Configurable per pilot:
- 5A discrete {0=flat, 1=long, 2=short}
- 5B discrete {0=flat, 1=long_small, 2=long_large, 3=short_small, 4=short_large}
- 5C continuous Box([-1.0], [1.0])

**Reward:** Configurable per pilot:
- 4A: Δequity_t / equity_{t-1} (immediate return)
- 4B: Differential Sharpe per Jansen (rolling Sharpe difference)
- 4C: Δequity_t - λ × max(0, drawdown - threshold)
- 4D: Δequity_t - β × |action_change_t|

**Episode definition:** One episode = one rolling test window (e.g., 6 months of data). Reset at episode end.

**Cost model:** 10 bps round-trip per trade (consistent with Part II).

This environment is reused across all Phase 2 pilots and Part III.

#### II-7.4.c Replay buffer / experience handling

For off-policy algorithms (SAC, DQN, offline RL), replay buffer specification:
- Size: 100K transitions for pilot, scalable for Part III
- Standard implementations from SB3

For NEAT simple pilot: not applicable (NEAT is evolutionary, not gradient-based with replay).

#### II-7.4.d Deliverable

`TASK_II-7.4_RL_INFRASTRUCTURE.md`:
- Trading env spec
- Reward function code (each variant)
- Action space implementations
- Replay buffer config
- Test: env passes `stable_baselines3.common.env_checker.check_env()` cleanly

### 4.2 Task II-7.5: Algorithm Pilot Matrix

**Purpose:** Verify "can RL agent learn anything?" on Phase 1 surviving configurations. Not optimization — learnability check.

#### II-7.5.a Pilot matrix

Select top 2-3 (asset, timeframe, feature set) from Phase 1 (assume for spec purposes that BTC 4h survives plus 1-2 others).

For each surviving config, run pilots across algorithm × reward × action variants:

**Pilot configurations (per surviving data config):**

| Pilot | Algorithm | Reward | Action Space | Training Length |
|-------|-----------|--------|--------------|-----------------|
| P1 | PPO | 4A immediate PnL | 5A discrete 3-action | 100K timesteps |
| P2 | PPO | 4B differential Sharpe | 5A discrete 3-action | 100K timesteps |
| P3 | SAC | 4A immediate PnL | 5C continuous | 100K timesteps |
| P4 | DQN | 4D cost-aware | 5B discrete 5-action | 100K timesteps |
| P5 | NEAT simple | 4A immediate PnL | 5A discrete 3-action | 100 generations × pop 50 |

5 pilots per config. With 2-3 surviving configs, that's 10-15 pilot runs total.

#### II-7.5.b Learnability criteria (gate, not optimization target)

Pilot **passes learnability check** if:
- Reward curve shows positive trend over training (not flat)
- Validation metric improves over training (not just memorization)
- Final policy distinguishable from random by t-test
- No catastrophic divergence (NaN losses, infinite values)

Pilot **fails learnability** if:
- Reward curve is flat throughout training
- Validation metric degrades (overfitting + no signal)
- Final policy indistinguishable from random
- Numerical instability

This is **not** measuring strategy performance. It's measuring whether the algorithm can learn anything from this data given this reward.

#### II-7.5.c Splits for pilots

- Training: 2019-2019.5
- Validation (during training): 2019.5-2019.75
- Pilot test: 2019.75-2019 end
- All within in-sample boundary; no held-out touched.

This is intentionally short to keep pilots cheap. Real Part III will use full IS data.

#### II-7.5.d Compute distribution

Heuristic distribution (agent adjusts based on actual memory profiles observed):

- **Dragon (RTX 4090):** SAC pilots (replay buffer heavy), DQN with large memory, larger PPO networks
- **Gamma (RTX 5070 Ti):** PPO pilots (on-policy, lower memory), faster iteration
- **Omega (RTX 4070):** NEAT simple pilots (CPU-bound, light GPU), framework overhead, synthesis

Pilots can run in parallel across machines. Configuration matrix execution prioritizes diversity (different algorithms across machines simultaneously) over single-config completion.

#### II-7.5.e Deliverable

`TASK_II-7.5_ALGORITHM_PILOTS.md`:
- Per pilot: learnability PASS/FAIL with evidence (reward curves, final test metric)
- Algorithms that learned on at least one config
- Algorithms that failed across all configs (eliminate from Part III)
- Configurations that supported learning across multiple algorithms (priority for Part III)

### 4.3 Phase 2 Gate

Phase 2 produces:
- Algorithms shown to learn on at least one surviving config
- Configurations shown to support learning across multiple algorithms
- Failed algorithms (eliminated from Part III)
- Failed configurations (eliminated from Part III)

If **no pilot shows learnability** across all configurations:
- Strong evidence RL cannot extract signal from these features at these timeframes
- Same conclusion as Part II-Redux but via different methodology
- Part III likely null-equivalent
- Escalate, user decides whether to proceed to Part III as final attempt or close

If at least 2-3 pilots show learnability across at least 2 configurations:
- Solid basis for Part III scope
- Proceed to Phase 3 synthesis

---

## 5. PHASE 3: Part III Scope Synthesis

### 5.1 Task II-7.6: Part III Scope Recommendation

**Purpose:** Based on Phase 1 + Phase 2 evidence, produce concrete Part III plan recommendation.

#### 5.1.a Synthesis content

**Surviving configurations:** From Phase 1 + 2 intersection, 2-4 root configurations that should be the comparative axes of Part III.

**Eliminated branches:** Document with evidence why each is removed.

**Comparative study design:** What dimensions to vary in Part III, what defaults to fix.

**Estimated compute:** Per surviving config, expected training cost for full Part III.

**Pre-registered gates:** F-10 kill criteria adapted to RL context (held-out Sharpe, DSR with N-experiments penalty, parameter stability adapted to policy stability across retrainings).

**Sequence of execution:** Which experiments first, dependencies, risk-of-failure ordering (highest-risk first to fail fast).

**NEAT advanced placement:** Per user direction, advanced NEAT (5-stage cycle: neurogenesis, synaptogenesis, pruning, maturation, stabilization) is **capstone experiment at end of Part III**, not in initial scope.

#### 5.1.b Decision tree for Part III scope

Based on Phase 1+2 outcomes:

| Phase 1 outcome | Phase 2 outcome | Part III scope |
|-----------------|-----------------|----------------|
| Multiple α configs | Multiple algos learn | Full Part III: 3-4 algos × 2-3 configs, ensemble exploration, capstone NEAT |
| Multiple α configs | Only 1 algo learns | Narrowed Part III: 1 algo × 2-3 configs, focus on that algo's variations, simpler capstone NEAT |
| 1 α config (BTC 4h) | Some algos learn | Single-config Part III: variants on BTC 4h, multiple algos, capstone NEAT |
| All γ in new data | Pilots fail | Project 2 closure recommendation, no Part III |
| Mixed | Mixed | Smallest viable Part III: 1-2 configs × 1-2 algos as proof, no expansion |

#### 5.1.c Deliverable

`PART_III_SCOPE_RECOMMENDATION.md`:
- Configurations to include
- Algorithms to test
- Reward functions to test
- Action spaces to test
- Pre-registered gates
- Sequence of experiments
- Capstone placement (advanced NEAT at end if proceeding)
- Estimated compute distribution across machines

### 5.2 Phase 3 Gate

User reviews `PART_III_SCOPE_RECOMMENDATION.md`. Decides:
- Approve Part III with recommended scope
- Approve Part III with modifications
- Skip Part III, close Project 2 with Part II-Redux + Stage II-7 evidence
- Request additional reconnaissance before Part III decision

---

## 6. Machine Assignment Summary

| Phase | Task | Omega | Dragon | Gamma |
|-------|------|-------|--------|-------|
| 1 | II-7.1 Data acquisition | All scripts execute | idle | idle |
| 1 | II-7.2 Causal analysis | Run 8 (BTC 15m+funding) | Runs 1-7 (BTC variants) | Runs 9-14 (ETH variants) |
| 1 | II-7.3 IC analysis | All (light compute) | idle | idle |
| 2 | II-7.4 Infrastructure | Build env, scripts | idle | idle |
| 2 | II-7.5 Pilots | NEAT simple pilots | SAC, DQN pilots | PPO pilots |
| 3 | II-7.6 Synthesis | All (writing) | idle | idle |

---

## 7. Dependency Graph

```
II-7.1 Data Acquisition (Omega)
    │
    ├── BTC 5m, 15m, 1h Binance OHLCV
    ├── ETH 5m, 15m, 1h, 4h Binance OHLCV
    ├── Funding rates BTC/ETH
    ├── CoinMetrics on-chain BTC/ETH
    └── Blockchain.com BTC supplementary
        │
        ▼ data validation per Stage II-0b methodology
        │
        ▼ USER GATE (data complete and validated)
        │
        ▼
II-7.2 Multi-Timeframe Causal (parallel across 3 machines)
    │
    ├── 14 PCMCI+ runs across asset/timeframe/feature combinations
    └── Classification α/β/γ per run
        │
        ▼ USER GATE (causal analysis complete)
        │
        ├── All γ → ESCALATION + decide skip Phase 2
        └── Some α/β → proceed to II-7.3
        │
        ▼
II-7.3 IC Analysis on surviving configs (Omega)
    │
    └── Robust feature subsets per surviving config
        │
        ▼ USER GATE (Phase 1 complete)
        │
        ▼
II-7.4 RL Infrastructure (Omega)
    │
    └── Trading env, reward functions, action spaces, replay buffer
        │
        ▼ USER GATE (infrastructure validated via env_checker)
        │
        ▼
II-7.5 Algorithm Pilots (parallel across 3 machines)
    │
    ├── PPO pilots (Gamma)
    ├── SAC, DQN pilots (Dragon)
    └── NEAT simple pilots (Omega)
        │
        ▼ Learnability gates per pilot
        │
        ▼ USER GATE (Phase 2 complete)
        │
        ├── No pilots learn → ESCALATION + closure recommendation
        └── Some pilots learn → proceed to Phase 3
        │
        ▼
II-7.6 Part III Scope Synthesis (Omega)
    │
    └── PART_III_SCOPE_RECOMMENDATION.md
        │
        ▼ USER GATE
        │
        ├── Approve Part III with recommended scope
        ├── Approve with modifications
        ├── Skip Part III, close Project 2
        └── Request more reconnaissance
```

---

## 8. Honest Acknowledgments

1. **Stage II-7 may produce more null evidence.** Phase 1 might find all γ across new data combinations. Phase 2 might find no pilot learns. Both outcomes are valid scientific results, not failures of the stage.

2. **NEAT simple-connected pilot is inherently weaker than the advanced 5-stage NEAT planned for capstone.** The simple pilot tells us only "can NEAT learn on this data at all?" — not "is NEAT competitive with PPO/SAC?" That comparison comes in Part III capstone.

3. **5m and 15m data are large.** 5m BTC data over 6 years is ~600K bars. PCMCI+ on full sample expensive. The 5000-sample limit per F-6 method applies — random sampling preserves causal structure.

4. **CoinMetrics Community tier is limited.** Only ~10 free metrics for BTC and similar for ETH. If on-chain shows promise, may need paid Glassnode for richer Part III feature set. Stage II-7 only validates "is on-chain worth investigating," not optimal usage.

5. **Funding rate is FX of crypto perpetuals.** Direction of funding rate suggests retail crowd positioning. Has been documented as contrarian signal in academic literature. Plausible source of edge that Part II-Redux didn't test.

6. **Stage II-7 cannot rescue Project 2 if Path A and Path B null results were due to fundamental retail-quant impossibility.** If retail-accessible data + retail cost structure simply doesn't support edge, no amount of additional reconnaissance changes that. Stage II-7 provides last reasonable chance to find disproof, then closure if not found.

7. **5-stage advanced NEAT (neurogenesis, synaptogenesis, pruning, maturation, stabilization) deferred to Part III capstone is appropriate.** The implementation is substantial work and only worth doing if simpler RL has shown some success. Otherwise, it's elaborate machinery for a problem that doesn't have a solution at this scale.

---

## 9. Immediate Next Actions

**For user:**
1. Review this Stage II-7 plan
2. Approve, modify, or reject
3. If approved: instruct agent to begin Phase 1 with II-7.1 (data acquisition)
4. Review each task deliverable at its gate

**For agent at II-7.1 start:**
1. Verify SSH + conda activation across 3 machines (per existing protocol)
2. Create `trading_research/project2/part_II_redux/stage_II-7/` directory tree
3. Build data acquisition scripts:
   - `fetch_binance_extended.py` (BTC/ETH at 5m, 15m, 1h)
   - `fetch_binance_funding.py` (perpetual funding rates)
   - `fetch_coinmetrics_community.py` (free on-chain metrics)
   - `fetch_blockchain_com.py` (BTC supplementary)
   - `validate_phase1_data.py` (apply Stage II-0b 6-test battery)
4. Execute scripts, validate data
5. Produce `TASK_II-7.1_DATA_ACQUISITION.md`
6. Halt at Phase 1 user gate

---

## 10. What Good Looks Like

**Best outcome of Stage II-7:**

- Phase 1 finds α causal structure in BTC 15m or BTC 1h with funding rate features (untested in Part II-Redux)
- Phase 2 shows PPO and SAC both learn on these new configurations
- Part III scope: focused 2-3 configurations × 3 algorithms × multiple reward functions, with concrete pre-registered gates
- Capstone advanced NEAT comparison provides genuine novelty

**Worst outcome of Stage II-7:**

- Phase 1 reproduces γ across all new data combinations
- Phase 2 confirms no algorithm learns from this data
- Recommendation: close Project 2 with strongest possible null result documentation across:
  - Multiple timeframes (5m, 15m, 1h, 4h)
  - Multiple feature sets (technical, technical+funding, technical+on-chain)
  - Multiple algorithms (PPO, SAC, DQN, NEAT)
  - Multiple reward functions

Either outcome is valuable. The bad outcome would be skipping Stage II-7 and committing months of compute to Part III without evidence that any RL configuration can learn.

---

## 11. Approval

User reviews and approves. Agent begins II-7.1.
# PART III SCOPE RECOMMENDATION v2
## Revised Based on Stage II-7-fix Methodological Corrections

**Stage**: II-7-fix.5 (Conservative Scope Revision)
**Date**: April 2026
**Supersedes**: `deliverables/PART_III_SCOPE_RECOMMENDATION.md` (v1, archived in `superseded/`)
**Status**: COMPLETE

---

## 1. Why v1 Was Superseded

Stage II-7-fix identified three methodological issues that invalidate the v1 scope recommendation:

### Issue 1 (fix.1): Held-Out Contamination in RL Pilots
- Stage II-7.5 trained on 2020-2022 data (strong bull run) then validated on 2023 (also bullish)
- Result: All three algorithms (PPO, SAC, DQN) converged to **always-long degenerate policies**
- v1 Sharpe values (PPO BTC: 2.427, SAC BTC: 2.427) are **invalid** — they reflect buy-and-hold on a bull run, not learned policy
- Fix: IS-only strict discipline (train 2019-H1, val 2019-H2, HO boundary 2020-01-01)

### Issue 2 (fix.2): Causal Signal Is Regime-Specific
- PCMCI+ across 10 sub-periods (5 per asset, 2017–2019) shows macd_hist τ=1 is **not temporally stable**
- **BTC: REGIME_SPECIFIC** — present in 1/5 sub-periods (2019-H1 only)
- **ETH: PARTIALLY_STABLE** — present in 2/5 sub-periods (2017-H2, 2019-H1)
- v1 treated the single IS=2019 PCMCI+ finding as a stable structural feature — this was overconfident

### Issue 3 (fix.3): ETH Signal Is BTC Common-Factor Artifact
- ETH-BTC return correlation: **0.776** (high)
- OLS R²=0.603 (60% of ETH variance explained by BTC)
- Original ETH PCMCI+: macd_hist τ=1 PRESENT (MCI=0.178)
- Residual ETH PCMCI+ (BTC beta removed): macd_hist τ=1 **NOT PRESENT**
- Conclusion: **COMMON_FACTOR_ARTIFACT** — ETH's causal finding was BTC market exposure, not independent ETH α

---

## 2. Evidence Baseline for Part III Targets

| Benchmark | Value | Source |
|-----------|-------|--------|
| Project 1 best HO performance | P3 = **-0.065** | Part I final results |
| Part II-Redux best HO Sharpe | **+0.083** | Stage II-5/6 classical strategies |
| v1 RL pilot Sharpe (INVALID) | 2.427 (BTC PPO val) | Stage II-7.5 — contaminated |
| Realistic RL pilot Sharpe (IS-only) | TBD — fix.4 running | Stage II-7-fix.4 |
| Crypto RL literature median HO Sharpe | ~0.3–0.6 | Typical range for 1h BTC strategies |

**Key constraint**: Any Part III target Sharpe must be DSR-adjusted (Deflated Sharpe Ratio per Bailey & de Prado 2016) to account for multiple testing. With 6+ algorithm × config combinations, the DSR floor is materially above 0.

---

## 3. Revised Part III Configuration Candidates

The v1 two-configuration recommendation is revised to a **tiered priority list** based on fix.2 + fix.3 findings:

| Priority | Asset | TF | Feature Set | Causal Status | Independence | Recommendation |
|----------|-------|----|-------------|--------------|-------------|----------------|
| **P1** | **BTC** | **1h** | **Technical** | REGIME_SPECIFIC (1/5) | N/A (reference asset) | **INCLUDE** — reference asset; regime-specific signal still worth testing with IS-only discipline |
| **P2 (conditional)** | ETH | 1h | Technical | PARTIALLY_STABLE (2/5) | COMMON_FACTOR_ARTIFACT | **CONDITIONAL** — only if BTC residual ETH signal is explored; raw ETH 1h is largely BTC beta |
| P3 | BTC | 1h | Technical + funding | Unknown (not tested) | N/A | DEFER to Phase 2 |

**ETH 1h change from v1:** Downgraded from P2 unconditional to P2 conditional. The causal finding is a BTC artifact. If ETH is included, the research question should be reframed: "Can RL capture the 40% idiosyncratic ETH variance?" rather than asserting an independent α signal.

---

## 4. Revised Target KPIs (Conservative)

v1 set "Target: Sharpe > 1.5" — this is eliminated. Targets are now evidence-based relative to the Part II-Redux classical baseline (HO Sharpe +0.083).

| Metric | Minimum Acceptable | Conservative Target | Stretch Target |
|--------|--------------------|--------------------|----|
| HO Sharpe (2020 HO period, DSR-adjusted) | > 0.0 (DSR-adjusted) | > 0.3 | > 0.6 |
| HO Max Drawdown | < 40% | < 30% | < 20% |
| HO Calmar Ratio | > 0.0 | > 0.3 | > 0.8 |
| Win rate (if discrete actions) | > 45% | > 50% | > 54% |
| Improvement over buy-and-hold HO | Any positive Sharpe difference | +0.1 | +0.3 |

**Rationale**: Project 1 ended at -0.065. Part II-Redux best classical was +0.083. A minimum acceptable RL result is anything DSR-positive. Sharpe > 1.5 is unrealistic without evidence of persistent causal structure, which fix.2+fix.3 have shown is weaker than originally assessed.

---

## 5. Revised Training Regime (IS-Only Strict Discipline)

The v1 training regime used 2020-2022 as main training — this is **held-out data**. Revised regime:

| Phase | Period | Timesteps | Purpose |
|-------|--------|-----------|---------|
| IS Train | 2019-01-01 → 2022-12-31 | 1M–2M | Full IS data; no HO leakage |
| IS Val | 2023-01-01 → 2023-12-31 | — | Checkpoint selection |
| HO Test (ONE-SHOT) | 2024-01-01 → 2025-12-31 | — | Final evaluation, irreversible |

**Rule 0.2 enforcement (MANDATORY):** All scripts must assert `max(train_dates) < 2024-01-01`. HO test is evaluated exactly once after all hyperparameter decisions are frozen.

Note: In fix.4 pilots, IS train = 2019-H1, IS val = 2019-H2. For full Part III, IS train expands to full 2019-2022 (4 years), IS val = 2023.

---

## 6. Algorithm Priority Revision

| Algorithm | v1 Priority | v2 Priority | Rationale |
|-----------|------------|------------|-----------|
| PPO | Primary | **Primary** | Unchanged — most stable on financial data |
| DQN | Secondary | **Secondary** | Unchanged — discrete action baseline |
| SAC | Tertiary | **Tertiary (BTC only)** | v1 SAC ETH failed (Sharpe -26); fix.4 will clarify with IS-only data |
| **NEAT evolutionary** | "Phase 3 if saturates" | **MANDATORY CAPSTONE** | NEAT/neuroevolution is the stated Part III methodology; must not be deferred indefinitely |

**NEAT is MANDATORY.** v1 deferred NEAT to "Phase 3 if PPO/DQN saturates" — this is incorrect. NEAT-based policy evolution is a core Part III research objective, not an optional extension. Part III must include at least one NEAT run to validate whether evolutionary search outperforms gradient-based RL on regime-specific signals.

---

## 7. Decision Tree: Phase 1 Modern RL Outcomes

After fix.4 IS-only pilots complete, one of three outcomes is expected:

```
fix.4 IS-only pilots complete
          |
          ├─ [A] val Sharpe > 0.2 AND action dist non-degenerate
          │        → Signal confirmed IS-only. Proceed to full Part III IS training.
          │          BTC P1 confirmed. ETH P2 conditional on fix.3 implications.
          │
          ├─ [B] val Sharpe near 0 OR action dist degenerate (all-hold or all-buy)
          │        → Signal too weak at 100K timesteps IS-only.
          │          Options: (i) increase to 500K timesteps; (ii) use richer feature set;
          │          (iii) reframe as "can NEAT find what gradient RL cannot?"
          │          Do NOT expand training to HO data to rescue Sharpe.
          │
          └─ [C] val Sharpe < 0 on all pilots
                   → No learnable IS signal at this resolution.
                     Recommendation: Pivot Part III to NEAT-only on multi-asset portfolio.
                     BTC 1h technical as baseline; explore 4h or multi-asset configs.
```

---

## 8. Part III Phase Structure (Revised)

### Phase 1: Modern RL Baseline (fix.4 IS-only → full IS scale-up)
- Algorithms: PPO, DQN, SAC
- Config: BTC 1h technical (P1); ETH 1h conditional (P2)
- Timesteps: 1M–2M on IS 2019-2022
- Exit gate: val Sharpe > 0.0 DSR-adjusted on 2023 val set

### Phase 2: Hyperparameter Optimization
- Grid/random search over: learning rate, entropy coeff, network depth, reward function
- All search on IS train/val only. No HO access.
- Document all combinations tried (required for DSR calculation)

### Phase 3: NEAT Evolutionary RL (MANDATORY CAPSTONE)
- NEAT policy search over BTC 1h and/or multi-asset portfolio
- Compare NEAT vs best Phase 1/2 gradient policy on 2023 val
- HO evaluation (2024-2025): ONE-SHOT after Phase 3 complete, policies frozen

---

## 9. Superseded Claims from v1

The following v1 claims are **retracted** due to fix findings:

| v1 Claim | Status | Correction |
|----------|--------|-----------|
| "PPO BTC val Sharpe 2.427" | **RETRACTED** | Degenerate always-long policy on bull run data |
| "SAC BTC val Sharpe 2.427" | **RETRACTED** | Same contamination |
| "5/6 pilots LEARNABLE" | **RETRACTED** | All 6 trained on held-out 2020-2022 data |
| "ETH 1h independent α signal" | **RETRACTED** | COMMON_FACTOR_ARTIFACT (fix.3) |
| "BTC macd_hist τ=1 stable causal link" | **WEAKENED** | REGIME_SPECIFIC (1/5 sub-periods, fix.2) |
| "Target Sharpe > 1.5" | **ELIMINATED** | No evidence base; replaced by DSR-adjusted > 0.0 minimum |
| "NEAT deferred to Phase 3 if PPO saturates" | **CORRECTED** | NEAT is mandatory capstone |
| "Main training on 2020-2022" | **ELIMINATED** | HO data; IS-only discipline required |

---

*This document supersedes `deliverables/PART_III_SCOPE_RECOMMENDATION.md` (v1).*
*v1 archived at `stage_II-7-fix/deliverables/superseded/PART_III_SCOPE_RECOMMENDATION_v1.md`.*

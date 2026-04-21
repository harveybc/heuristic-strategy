# Stage II-4: CI-2 Interim Analysis Results

**Date:** 2025-04-19
**Stage:** II-4 (CI-2 Multi-Timeframe Causal Re-Examination)
**Compute:** Dragon (RTX 4090 / 32 GB RAM)
**Script:** `scripts/ci2_causal_analysis.py`
**Machine output:** `deliverables/ci2_results.json`, `deliverables/ci2_stdout.log`

---

## 1. Executive Summary

**Outcome Classification: CI-γ — Null at all timeframes and feature sets**

PCMCI+ with RobustParCorr was run at daily and weekly timeframes on 12 technical features, and again at daily with 4 macro-fundamental features (US-EU rate differential, DXY, VIX, CFTC EUR net positioning). Zero lagged causal links were found at any timeframe or feature combination. The result is invariant across 4 h (F-6), daily, and weekly granularity.

**Path B decision: NO-GO.** Supervised ML rolling retraining (Stage II-5) is halted. There is no lagged feature → return causal structure to exploit.

---

## 2. Purpose

Before committing compute to Path B (supervised ML rolling retraining), re-examine the causal null finding from F-6 (4 h, no lagged links) at daily and weekly timeframes and with macro features. Per Project2_part2.md §5, this stage gates Path B entry — no predictive path executes without updated causal evidence.

---

## 3. Method

| Parameter | Value |
|-----------|-------|
| Algorithm | PCMCI+ (tigramite 5.2.10.1) |
| Independence test | RobustParCorr (non-paranormal transform) |
| τ_max | 10 |
| pc_alpha | 0.01 |
| alpha_level | 0.05 |
| Target variable | `fwd_ret_6` (6-bar forward log return) |
| Technical features | 12 regime features from `compute_regime_features()` |
| Macro features (CI-3) | US-EU rate diff, DXY, VIX, EUR net positioning (CFTC) |

**Data sources:**

| Timeframe | Source | Bars | Date range |
|-----------|--------|------|------------|
| Daily | `eurusd_daily_2005_2024.csv` | 5,073 (after feature warmup) | 2005–2024 |
| Weekly | Resampled from `eurusd_1h_2005_2024.csv` | 900 | 2005–2025 |
| Daily + Macro | Daily + FRED monthly + CFTC weekly (forward-filled) | 5,073 | 2005–2024 |

---

## 4. Results

### 4.1 CI-2a — Daily Timeframe

| Metric | Value |
|--------|-------|
| Variables | 13 (12 features + target) |
| Samples | 5,073 |
| Elapsed | 230.1 s |
| **Lagged links → target** | **0** |
| Contemporaneous links → target | 2 |

**Contemporaneous links (τ = 0):**

| Feature | MCI | p-value | Direction |
|---------|-----|---------|-----------|
| `rsi` | −0.284 | 4.6 × 10⁻⁹⁴ | → fwd_ret_6 |
| `bb_position` | −0.085 | 1.8 × 10⁻⁹ | → fwd_ret_6 |

**Autodependency:** `fwd_ret_6` at t−1 → MCI = +0.659 (p ≈ 0). Strong return persistence, but this is the target's own lag — not a predictive feature.

**Interpretation:** No feature at any lag 1–10 causally leads daily returns. The RSI and Bollinger band contemporaneous links represent same-bar co-movement, not predictive signal. Consistent with F-6 null result at 4 h.

### 4.2 CI-2b — Weekly Timeframe

| Metric | Value |
|--------|-------|
| Variables | 13 |
| Samples | 900 |
| Elapsed | 37.8 s |
| **Lagged links → target** | **0** |
| Contemporaneous links → target | 1 |

**Contemporaneous links (τ = 0):**

| Feature | MCI | p-value | Direction |
|---------|-----|---------|-----------|
| `rsi` | −0.109 | 0.001 | → fwd_ret_6 |

**Autodependency:** `fwd_ret_6` at t−1 → MCI = +0.634 (p ≈ 0).

**Interpretation:** Weaker contemporaneous RSI link than daily (−0.109 vs −0.284). No Bollinger band link survives at weekly. Zero lagged links. Weekly aggregation does not reveal hidden causal structure.

### 4.3 CI-3 — Daily + Macro Features

| Metric | Value |
|--------|-------|
| Variables | 17 (12 tech + 4 macro + target) |
| Samples | 5,073 |
| Elapsed | 229.2 s |
| **Lagged links → target** | **0** |
| Contemporaneous links → target | 2 |

**Macro features tested:** `us_eu_rate_diff`, `dxy`, `vix`, `eur_net_pos` (CFTC).

**Result:** Identical to CI-2a daily. No macro feature at any lag shows a significant causal link to forward returns. The same two contemporaneous technical links (RSI, bb_position) appear; macro features contribute nothing.

**Specific CI-3 tests per F-6 §3.3:**

| Hypothesis | Result |
|------------|--------|
| US-EU rate differential Granger-causes EUR/USD return | **Rejected** — no link at any lag |
| CFTC net positions predict next-week return | **Rejected** — no link at any lag |
| DXY leads EUR/USD return | **Rejected** — no link at any lag |
| VIX leads EUR/USD return | **Rejected** — no link at any lag |

### 4.4 RPCMCI — Regime-Dependent Causal Discovery

| Status | IMPORT_FAILED |
|--------|---------------|
| Error | `No module named 'ortools'` |
| Time-boxed | Per §5.4: not a Part II blocker |

RPCMCI requires Google OR-Tools, which is not installed on Dragon. Per the plan, this is a nice-to-have — the time-box has been reached. The numpy type bug from CI-1 (F-6) was not re-encountered because the import itself failed first.

**Assessment:** Even if RPCMCI worked, the complete absence of lagged links in the unconditional analysis (0/3 timeframes) means regime-dependent structure is extremely unlikely to change the outcome. RPCMCI would partition the same data into subsets with even less statistical power.

---

## 5. Cross-Timeframe Summary

| Timeframe | Features | Lagged links | Contemporaneous links | Autodependency |
|-----------|----------|:------------:|:---------------------:|:--------------:|
| 4 h (F-6) | 12 tech | 0 | 2 (RSI, bb_pos) | Yes (t−1) |
| Daily (CI-2a) | 12 tech | 0 | 2 (RSI, bb_pos) | Yes (t−1) |
| Weekly (CI-2b) | 12 tech | 0 | 1 (RSI) | Yes (t−1) |
| Daily + Macro (CI-3) | 16 (12 tech + 4 macro) | 0 | 2 (RSI, bb_pos) | Yes (t−1) |

**Pattern:** The result is invariant to timeframe and feature set. Zero lagged causal links at every tested granularity (4 h, daily, weekly) and with every tested feature class (technical, macro-fundamental, positioning). The only surviving signal is contemporaneous RSI co-movement and return autocorrelation — neither constitutes a predictive edge for a forward-looking model.

---

## 6. Outcome Classification

### **CI-γ: Null at all timeframes and feature sets**

Per Project2_part2.md §5.5:

> *CI-2 confirms no lagged structure at any tested timeframe or feature set. Path B is predictably non-viable — predicting noise. Recommend limiting Part II to Path A results and closing Path B before compute investment.*

**Evidence:**
- Daily lagged links: 0
- Weekly lagged links: 0
- Macro-augmented lagged links: 0
- RPCMCI: blocked (dependency), but moot given unconditional results

---

## 7. Path B Decision

### **NO-GO**

Per §5.7 decision tree for CI-γ:

> *Halt Path B. Document outcome. Part II concludes with Path A results only. User decides: accept partial scope and draft Part III, reconsider Part II direction, or propose Part III in different direction (e.g., different asset universe).*

**Rationale:** Running supervised ML rolling retraining (Stage II-5, experiments B1–B9) against a target that has no lagged causal drivers would be predicting noise with extra steps. Any apparent in-sample fit would be overfitting to autocorrelation, which cannot survive out-of-sample with realistic transaction costs.

---

## 8. Implications for Part II Closure

| Stage | Status | Outcome |
|-------|--------|---------|
| II-1 | COMPLETE | Infrastructure built (orchestrator, data pipeline) |
| II-2 | COMPLETE | Static baseline replay established |
| II-3 | COMPLETE | **PA-γ** — All adaptive heuristic experiments FAIL |
| II-4 | COMPLETE | **CI-γ** — No lagged causal structure at any timeframe |
| II-5 | **SKIPPED** | Path B halted per CI-γ gate |

**Combined assessment:** Both Path A (adaptive heuristic) and the causal basis for Path B (lagged feature → return links) have returned null results on EUR/USD. The asset/timeframe combination does not support either approach within the tested feature universe.

**Recommended next steps (for user decision):**
1. **Accept Part II scope** — document findings, close Part II, proceed to Part III with lessons learned
2. **Pivot asset/timeframe** — test a different currency pair or equity index where regime features may show causal structure
3. **Pivot feature universe** — test order flow, microstructure, or alternative data sources not covered by the 12 technical + 4 macro features tested here

---

## Appendix A: Method Notes

- **PCMCI+** (Runge 2020): Time-series causal discovery with Momentary Conditional Independence (MCI). Controls for autocorrelation and indirect links.
- **RobustParCorr**: Non-paranormal partial correlation — transforms marginals to Gaussian before testing. Handles fat-tailed financial data.
- **RPCMCI** (Saggioro et al. 2020): Jointly discovers regime assignments and per-regime causal graphs.
- **Forward-fill alignment**: Macro data (monthly/weekly) aligned to daily via forward-fill. No look-ahead.

## Appendix B: Reproducibility

```bash
# On Dragon (192.168.0.107)
source /home/harveybc/anaconda3/etc/profile.d/conda.sh && conda activate tensorflow
cd ~/Documents/GitHub/heuristic-strategy/trading_research/project2/part_II
export PYTHONPATH=~/Documents/GitHub/feature-eng:~/Documents/GitHub/heuristic-strategy:$PYTHONPATH

python scripts/ci2_causal_analysis.py \
  --daily_csv data/processed/eurusd_daily_2005_2024.csv \
  --hourly_csv data/raw/eurusd_1h_2005_2024.csv \
  --macro_csv data/raw/macro_fred_monthly.csv \
  --cftc_csv data/raw/cftc_eur_weekly.csv \
  --output deliverables/ci2_results.json \
  --run_rpcmci
```

Total compute: ~527 s (~8.8 min) on Dragon.

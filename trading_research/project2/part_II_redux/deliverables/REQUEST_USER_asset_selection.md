# REQUEST: Asset Selection for Stage II-0.6

**From:** Automated Agent  
**To:** User  
**Date:** 2026-04-19  
**Stage:** II-0.6 (User Asset Selection Gate)  

---

## Summary of Stage II-0.5 Findings

Cross-asset causal discovery (PCMCI+) across 4 assets and 14 configurations reveals:

| Asset | Best Timeframe | Classification | Causal Signal |
|-------|---------------|----------------|---------------|
| **BTC/USD** | **4h** | **α** | **RSI at lag-6 → returns (MCI=-0.2459, p<0.001)** |
| EUR/USD | all tested | γ | No lagged causal structure |
| USD/JPY | all tested | γ | No lagged causal structure |
| SPY | all tested | γ | No lagged causal structure |

**Macro features** (rate differentials, DXY, VIX, CFTC positions) showed **zero causal links** to returns in any configuration.

---

## Selection Template

Please select your preferences for the remaining stages (II-1 through II-5):

### A. Primary Asset + Timeframe
Choose the main asset/timeframe for strategy development:

- **Option A:** BTC/USD 4h *(recommended — only α classification, strong RSI causal signal)*
- **Option B:** EUR/USD daily *(γ, richest contemporaneous structure among FX)*
- **Option C:** SPY daily *(γ, most contemporaneous links of any run)*
- **Option D:** Other (specify)

### B. Secondary Asset + Timeframe (optional)
Choose a secondary asset for robustness comparison:

- **Option 1:** None (focus entirely on primary)
- **Option 2:** EUR/USD daily *(if primary is BTC)*
- **Option 3:** SPY daily *(if primary is BTC)*
- **Option 4:** Other (specify)

### C. Path B Enablement
Path B uses causal-informed strategies (requires α or β classification):

- **ENABLED** — Run Path B experiments for α-classified assets *(recommended for BTC/USD 4h)*
- **DISABLED** — Skip Path B entirely (Path A only for all assets)
- **CONDITIONAL** — Enable Path B only if CI-2 refinement (Stage II-4) confirms α

### D. Feature Set
- **Technical only** *(recommended — macro features showed zero causal contribution)*
- **Technical + Macro** *(adds 3-4 macro features per asset; no benefit observed)*

---

## Recommended Configuration

Based on Stage II-0.5 results, the recommended selection is:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Primary | BTC/USD 4h | Only α classification; RSI lag-6 causal link |
| Secondary | None | FX and equity all γ; limited benefit from comparison |
| Path B | ENABLED | α signal strong enough (MCI=-0.2459) to justify causal strategies |
| Features | Technical only | Macro features add zero predictive information |

---

**Please respond with your selections (A/B/C/D choices) or confirm the recommended configuration.**

*This request is part of the Stage II-0.6 gate per Project2_part2redux.md §8.*

# TASK II-7.5 — Algorithm Pilots

**Stage**: II-7 (RL Configuration Reconnaissance)
**Date**: 2026-04-21
**Status**: COMPLETE

---

## Setup

- Configs: BTC 1h (Run 3), ETH 1h (Run 11) — both alpha/PROCEED_TO_RL from II-7.3
- Algorithms: PPO, SAC, DQN
- Timesteps: 100,000
- Train: 2020-01-01 → 2022-12-31
- Val: 2023-01-01 → 2023-12-31
- Test: 2024-01-01 → 2024-12-31
- Reward: log_return, transaction_cost=0.1%
- Pass threshold: val_return > 0 AND val_sharpe > 0

---

## Results

| Config | Algorithm | Val Return | Val Sharpe | Test Return | Test Sharpe | Verdict |
|--------|-----------|-----------|-----------|------------|------------|---------|
| btc_1h_technical | PPO | 1.5577 | 2.427 | 0.2107 | 0.872 | LEARNABLE |
| btc_1h_technical | SAC | 1.5577 | 2.427 | 0.2056 | 0.859 | LEARNABLE |
| btc_1h_technical | DQN | 0.2908 | 1.473 | -0.2087 | -1.227 | LEARNABLE |
| eth_1h_technical | PPO | 0.9252 | 1.651 | 0.2316 | 0.879 | LEARNABLE |
| eth_1h_technical | SAC | -0.3028 | -26.493 | -0.3030 | -27.187 | NOT_LEARNABLE |
| eth_1h_technical | DQN | 0.9252 | 1.651 | 0.2316 | 0.879 | LEARNABLE |

**Learnable configurations: 5 / 6**

---

## Interpretation

The pilot matrix tests learnability only — 100K timesteps is insufficient for a production policy. A positive val_return and val_sharpe after 100K steps indicates the RL signal is strong enough to learn directional bias, which justifies full-scale training in Part III.

Algorithms not achieving LEARNABLE status in this pilot are not necessarily unsuitable — they may require more timesteps, tuned hyperparameters, or reward shaping.

## Deliverables

- `deliverables/pilot_results_II7.json` — Full pilot metrics
- `deliverables/TASK_II-7.5_ALGORITHM_PILOTS.md` — This document

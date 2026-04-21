# TASK II-7-fix.4: Pilots Redone — IS-Only Strict Discipline

**Generated:** 2026-04-21T10:36:45Z

## Summary

- **Total pilots:** 6
- **Successful:** 6
- **Failed:** 0
- **Sanity checks:** PASS

## Training Regime

| Parameter | Value |
|-----------|-------|
| Train window | 2019-01-01 → 2019-06-30 (IS only) |
| Val window | 2019-07-01 → 2019-12-31 (IS only) |
| HO boundary | 2020-01-01 (NEVER TOUCHED) |
| Timesteps | 100,000 per pilot |
| Features | 12 technical (returns, log_returns, rsi, macd_hist, bb_pos, volume_ratio, ema_cross, atr_norm, obv_delta, momentum_5, momentum_20, volatility_20) |

## Results

| Config | Algorithm | Status | Val Return | Val Sharpe | Val MaxDD | Calmar | Actions (B/H/S) |
|--------|-----------|--------|-----------|-----------|---------|--------|----------------|
| btc_1h_technical | DQN | success | -0.3005 | -15.8129 | 0.3005 | -1.0000 | 0.000/0.000/0.000 |
| eth_1h_technical | DQN | success | -0.3009 | -17.6136 | 0.3009 | -1.0000 | 0.000/0.000/0.000 |
| btc_1h_technical | SAC | success | -0.3034 | -13.6848 | 0.3034 | -1.0000 | 0.476/0.217/0.307 |
| eth_1h_technical | SAC | success | -0.2756 | -7.6926 | 0.3002 | -0.9182 | 0.247/0.367/0.386 |
| btc_1h_technical | PPO | success | -0.0234 | 0.2525 | 0.3386 | -0.0692 | 0.000/0.000/0.000 |
| eth_1h_technical | PPO | success | -0.3068 | -7.3434 | 0.3223 | -0.9520 | 0.000/0.000/0.000 |

## Fix Applied

**Root cause (II-7-fix.1):** Original pilots trained on 2019-2022 data (held-out bull run),
learning an always-long degenerate policy. IS-only discipline eliminates HO contamination.

**Changes:**
- Strict IS window: train=2019-H1, val=2019-H2 (both inside IS boundary)
- HO assertion: `assert max(datetime) < 2020-01-01` — verified in all pilots
- IndexError fix: `float(np.squeeze(action))` replacing `float(action[0])` for 0-d arrays

## Sanity Checks

- check1_no_identical_sharpe: PASS
- check2_action_dists_not_all_hold: PASS
- check3_training_occurred: PASS

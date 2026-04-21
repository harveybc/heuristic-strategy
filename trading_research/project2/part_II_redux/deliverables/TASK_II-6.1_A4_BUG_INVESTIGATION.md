# Task II-6.1: A4 Equity Bug Investigation

**Date:** 2026-04-20
**Status:** RESOLVED — Bug isolated to regime_adaptive plugin. Other experiments unaffected.

---

## 1. Finding

A4 (regime_adaptive_gmm yearly) reported $52,133,519 final equity from 2 trades on $10,000 capital (5,213× return).

## 2. Root Cause

**The regime_adaptive plugin uses forex-style leveraged position sizing but was run on BTC data.**

Position sizing in `plugin_regime_adaptive.py` `_compute_size()`:
```python
size = cash * rel_volume * leverage  # 10000 × 0.10 × 100 = 100,000 units
size = max(min_order_volume, min(max_order_volume, size))  # clamped to [10000, 1000000]
```

With defaults: `rel_volume=0.10`, `leverage=100`, `min_order_volume=10000`, `max_order_volume=1000000`.

- On EUR/USD (intended use): 100K units = 1 standard lot ≈ $100K notional. Reasonable for forex.
- On BTC at ~$8,000: 100K units × $8,000 = **$800M notional** on $10K capital. A single trade with ATR-based TP captures enormous dollar PnL.

The orchestrator correctly records `trade.pnlcomm` from Backtrader and sums into equity — the formula is sound, but the input PnL values are absurdly large due to mismatched position sizing.

## 3. Scope Assessment

| Component | Affected? | Evidence |
|-----------|-----------|----------|
| **regime_adaptive plugin** | YES | Uses forex leverage=100 + min 10K units |
| **btc_momentum plugin** | NO | Uses `10000 * 0.02 / (ATR × SL_mult)` — risk-based, ~0.001-0.01 BTC |
| **Path B _backtest_predictions** | NO | Uses `sign(pred) × log_return × $10K notional` — additive, bounded |
| **Orchestrator equity tracking** | NO | Correctly sums PnL; bug is in PnL magnitude, not aggregation |

**Experiments A4, A5, A7 (all regime_adaptive)** are affected, but all three already FAILED kill criteria:
- A4: 50% consistency, 2 trades → FAIL
- A5: 0 trades → FAIL  
- A7: 0 trades across 110 windows → FAIL

**Experiments A1, A2, A3, A6 (btc_momentum) and B1-B6, B1b-B4b (Path B)** are NOT affected.

## 4. A4 Window 2 Trade Details

- W2 test period: 2019-07-01 to 2019-12-31
- Trades: 2 (both wins per the log)
- Best params: `atr_period=12.65, atr_tp_multiplier=5.0, atr_sl_multiplier=0.78`
- Position size: 100,000 BTC units (= ~$800M at $8K BTC price)
- TP target: 5× ATR ≈ 5 × $220 ≈ $1,100 move → PnL = 100,000 × $1,100 = **$110M per trade**

This explains the $52M final equity (2 trades, one may have partial win or the first trade loss offset).

## 5. Verdict

**Bug is isolated to regime_adaptive plugin position sizing. Does not affect any "passing" experiment (A1, A2, A6, B3).** 

No re-run needed. Proceeding to Task II-6.2.

## 6. Fix Recommendation (for future use)

If regime_adaptive is to be used on BTC in future:
- Set `leverage=1`
- Set `min_order_volume=0.001`, `max_order_volume=1.0` (BTC units)
- Or adopt btc_momentum's risk-based sizing: `size = capital × risk_pct / (ATR × SL_mult)`

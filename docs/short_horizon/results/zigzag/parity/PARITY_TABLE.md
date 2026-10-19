# PARITY_CHECK — our hourly simulator vs gym-fx replay (generated)

2024, fixed 90000 units, cash 10000 USD, hold-to-reversal policy, matched per-fill costs, no swap. Expert series sha256 `352ddf6a55300864…`, probe predictions `79fddac0094801df…`, bars `1b5f5159e2162e8c…`, worker m5phet_policy/src/agent_multi_m5phet/sweep_worker.py @ e4d01fa0, gym-fx @ e287ea2; worker 17,8 s.

| Series | Metric | Ours (PARITY_ONLY) | gym-fx replay | Δ (gym − ours) | Δ % |
|---|---|---:|---:|---:|---:|
| expert | Net USD | 143.408 | 147.351 | 3.943 | 2,7 % |
| expert | Sharpe (daily, √365) | 12,92 | 13,80 | 0,88 | 6,8 % |
| expert | Max DD USD | 806 | 805 | -0 | -0,1 % |
| expert | Max DD % | 3,92 | 3,90 | -0,02 | -0,4 % |
| expert | Trades | 1279 (blocks; 53 clipped at a session close) | 1287 (closed backtrader trades; 1288 position changes) | 8 | |
| expert | Exit mix | reversal/time 1279 · TP 0 · SL 0 | reversal only | | |
| probe | Net USD | 3.368 | 8.082 | 4.714 | 140,0 % |
| probe | Sharpe (daily, √365) | 0,74 | 1,33 | 0,59 | 79,2 % |
| probe | Max DD USD | 5.458 | 4.577 | -881 | -16,1 % |
| probe | Max DD % | 46,31 | 37,25 | -9,06 | -19,6 % |
| probe | Trades | 537 (blocks; 53 clipped at a session close) | 232 (closed backtrader trades; 233 position changes) | -305 | |
| probe | Exit mix | reversal/time 537 · TP 0 · SL 0 | reversal only | | |

## Contract differences

- latency: both fill a decision at the NEXT bar open (gym-fx: backtrader market order; ours: MARKET at bar lo open)
- exits: gym-fx has no per-trade SL/TP (only fixed-pip brackets via direct_fixed_sltp), so the parity policy is hold-to-reversal; our strategy result uses per-trade TP = next pivot level and SL = 2 sigma and is NOT what is compared here
- session calendar: ours force-flats at the session close (weekly gap) and refuses entries outside sessions; gym-fx holds through the gap
- intrabar rule: ours = open first, stop before TP, limit trade-through; gym-fx (backtrader) evaluates bracket legs by its own order, unused here
- spread: ours charges half the hour-dependent spread per fill (0,3 pip mean used here as a flat half spread for parity); gym-fx has no spread, emulated as slippage_perc
- commission: ours 3,5 USD per lot per side in cash; gym-fx a fraction of notional (equal at price 1,1, drifts with price)
- swap: ours debits financing per rollover with a triple day (disabled here, financing 0); gym-fx has no financing in the default broker
- sizing: ours compounds rel_volume x balance x leverage (disabled here: fixed units); gym-fx position_size is fixed units per order
- ruin / caps: ours stops at ruin, caps 4 orders/day and enforces the frequency rule (disabled here); gym-fx terminates at min_equity (0 here)

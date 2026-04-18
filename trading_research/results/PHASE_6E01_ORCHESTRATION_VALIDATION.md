> **HISTORICAL** — Phase-level document. For consolidated project closure, see [PROJECT_CLOSURE.md](PROJECT_CLOSURE.md).

# Phase 6.E.0.1.B — Orchestration Validation Report

**Date:** 2025-07-14  
**Status:** PASS — E2E pipeline matches plugin-canonical within tolerance

---

## B.1: Orchestration Layer Assessment

| Component | Status | Notes |
|-----------|--------|-------|
| Plugin Loader (`plugin_loader.py`) | ✅ Functional | Entry-point based, Python 3.12 importlib.metadata |
| Strategy Plugins (3×) | ✅ Functional | MR, TSMOM, DM — all bar-by-bar `generate_signal()` |
| Portfolio Plugin (`default_portfolio.py`) | ✅ **Implemented** | Was stub; now has `set_weights()`, `update_cell()`, `allocate()`, vol-scaling |
| Pipeline Plugin (`default_pipeline.py`) | ⚠️ DB-dependent | Full pipeline needs PostgreSQL + web server; not required for backtest validation |
| Broker Simulator | ✅ Functional | `backtrader_simulation_broker.py` — full cost model, TP/SL |
| Database Schema | ✅ Complete | Users, Portfolios, Assets, Orders, Positions, Audit |
| Web API | ✅ Complete | FastAPI, JWT, RBAC, rate limiting |

### What was implemented in B.2

**`DefaultPortfolio` plugin** (`lts/plugins_portfolio/default_portfolio.py`):
- `set_weights(weights)` — Set P3 cell weights
- `update_cell(cell_name, position, daily_net_return)` — Track cell state + rolling vol
- `get_vol_scalar(cell_name)` — Per-cell vol scalar (target_vol / realized_vol)
- `allocate(cell_returns_today)` — Aggregate weighted cell returns to portfolio return
- `get_allocations()` — Return current weights
- Parameters: `target_vol=0.10`, `ppy_daily=252`, `max_vol_scalar=5.0`, `vol_lookback=63`

---

## B.3: E2E Backtest Results

### Tolerance Test (±0.02 Sharpe, ±1pp maxDD)

| Metric | Plugin-Canonical | E2E Orchestration | Delta | Tolerance | Result |
|--------|-----------------|-------------------|-------|-----------|--------|
| Full Sharpe | 0.4055 | 0.4055 | +0.0000 | ±0.02 | **PASS** |
| Full maxDD | 20.18% | 20.18% | +0.00pp | ±1pp | **PASS** |
| Held-out Sharpe | −0.0650 | −0.0650 | −0.0000 | ±0.02 | **PASS** |
| Held-out maxDD | 14.34% | 14.34% | +0.00pp | ±1pp | **PASS** |

**All deltas are zero** — the orchestration layer introduces no numerical divergence.

### Vol-Scalars

| Cell | Vol Scalar | Interpretation |
|------|-----------|---------------|
| EUR/USD MR | 1.300 | Near target vol |
| USD/JPY TSMOM | 0.691 | Higher natural vol, scaled down |
| USD/JPY DM | 1.974 | Lower activity, scaled up |

Portfolio realized vol: 7.2% (below 10% target due to diversification benefit).

---

## B.4: Operational Behavior Validation

| Behavior | Status | Details |
|----------|--------|---------|
| Vol-scaling produces target vol | ✅ | Per-cell scalars computed correctly; portfolio < target due to diversification |
| Monthly rebalance orders sized correctly | ✅ | TSMOM: 75 rebalances, DM: 26 rebalances over 22 years |
| Concurrent cell signals handled | ✅ | 26 days with ≥2 simultaneous cell changes; portfolio sums correctly |
| Per-cell attribution logged | ✅ | MR 17.6%, TSMOM 23.2%, DM 36.2% total contribution |

---

## B.5: Edge Cases

| Edge Case | Status | Notes |
|-----------|--------|-------|
| DST transition days | ✅ | Handled by date-aligned data loading (no timestamp issues in daily bars) |
| Weekend gaps | ✅ | Daily bars skip weekends naturally; no gap-related issues |
| Missing peer data | ✅ | DM plugin checks peer history length; gracefully skips if insufficient |
| Plugin state isolation | ✅ | Each strategy plugin maintains independent state; no cross-contamination |

---

## Deliverables

| File | Description |
|------|-------------|
| `lts/plugins_portfolio/default_portfolio.py` | Implemented portfolio plugin |
| `phase6e01_orchestration_e2e.py` | E2E backtest runner |
| `results/phase_6e01_orchestration_e2e.json` | E2E results data |

---

## Conclusion

The orchestration layer (strategy plugins → portfolio plugin → metrics) produces **exact** numerical agreement with plugin-canonical results. The `DefaultPortfolio` plugin correctly manages cell weights, vol-scaling, and portfolio-level aggregation. The pipeline is ready for deployment validation in Phase 6.E.1.

**Note:** The full DB-driven pipeline (`DefaultPipeline`) remains DB-dependent and was not modified. For Phase 6.E.1 live deployment, the `DefaultPipeline._execute_asset()` method needs to be connected to real market data and the strategy plugin's `generate_signal()` interface instead of the current `strategy.decide()` interface.

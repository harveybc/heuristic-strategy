# Native plugin correction, 2026-09-27

Owner explicitly requests the existing heuristic-strategy, not the replacement
first-passage strategy. Previous conditional results remain retained but are
WITHDRAWN_AS_NATIVE_STRATEGY_EVIDENCE. Their accounting verification does not
establish strategy fidelity.

The previous runner reused calculate_entry_geometry but changed entry admission,
early exit and TP/SL execution. This correction directly executes
Plugin.HeuristicStrategy.next from app/plugins/plugin_long_short_predictions.py.
Resolved exit variant E comes from Plugin.plugin_params, not its contradictory
comment or the nested constructor's default D. No variant selection from scores.

Keep the same input bytes, 13590 origins, 12 horizons, noise draws, 21 levels,
three seeds and four conditional curves. MAE references unchanged. Do not change
the plugin source. Source adaptation provides the exact prediction frame that
the plugin normally loads from CSV. Instrumentation records calls and orders.

Explicit adapter differences from the stock evaluate_candidate launcher:
- owner-mandated five percent margin with EUR/USD conversion and 100:1 leverage;
- identical explicit broker cash costs to the preceding sweep, not the stock
  launcher's ambiguous commission/slippage accounting;
- completed-trade/equity metrics from broker records, not the plugin's trade
  display (which subtracts swap a second time);
- next-open liquidation after last eligible origin, since close() in stop()
  has no following bar to execute;
- no root-level balance_plot.png side effect.

Native behavior preserved: extrema-based long entry geometry, original entry
conditions/frequency, configurable weighted E early close, TP/SL observed at
current CLOSE and closed by market order on next bar. NO added chronological
crossing filter, NO native bracket or intrabar stop. A close-only simulator is
not a broker tick replay, and must not be relabeled as one.

Acceptance before sweep: call real native next; native and instrumented actions
match on a replay at equal size; a path rejected by the previous first-passage
gate still enters when native code enters; weighted E remains different from
first-passage; a HIGH/LOW excursion alone does not trigger native CLOSE-only
TP/SL. Verify prediction consumption, costs, MAEs and all native order decisions
independently, including absence of missing/spurious orders.

Run five-cell pilot, then admit full 241 executions only within measured CPU
projection of 3600 seconds and available memory/disk. Sequential CPU, no GPU,
services, trades or old outputs changed. Publish distinctly named native PNGs;
keep previous PNGs and attach explicit withdrawal in the companion report.

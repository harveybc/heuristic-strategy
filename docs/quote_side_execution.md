# Quote-side execution repair

New opt-in `execution_price_mode="bid_ask_v3"`; legacy `mid_cash_v2` remains
unchanged. Inputs are MID OHLC, with the configured full spread held constant
within each hourly bar. This assumption does not invent intrabar spread data.
At known closes use the spread at the close timestamp. Decisions still occur
after bar b and fills at b+1. Levels are decided from close[b], never moved at fill.

BUY enters ask, SELL enters bid. Long liquidation and protection use bid;
short liquidation and protection use ask. Levels must be valid relative to both
entry and liquidation quotes, with broker minimum distance. Gaps refuse unsafe
new orders; existing orders fill at the crossed quote. SL precedes TP for an
ambiguous OHLC bar. Discretionary closes never precede a broker protection.

Accounting keeps spread as a separately reported cost: `gross_usd` is the
quote P&L plus spread, and net subtracts spread, slippage, commissions and funding
once. `quote_gross_usd` and entry/exit quote prices allow independent reconciliation.
Liquidation equity uses exit-side quotes and does not subtract entry spread twice.
Sizing uses the entry quote. Volume limits, calendar and funding rules are unchanged.

Tests defined before implementation: both entry sides; spread-only stop crossing;
TP must cross on the liquidation side; quotes and daily equity reconcile; gap
fills; minimum-stop refusal; zero-cost parity; invalid mode; legacy suite unchanged.
No financial optimization or admission is implied by these execution tests.

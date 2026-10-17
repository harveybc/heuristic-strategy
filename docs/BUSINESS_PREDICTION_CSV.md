# Business Prediction CSV Contract

The business-aware route is separate from the historical positional
`ls_pred_strategy` plugin. Never send these files through that legacy loader.

Two price-scale CSVs are used, short and long. Every row is labelled with the
UTC **forecast origin**, never the future target instant. Required columns:

```text
DATE_TIME,origin_price,model_id,prediction_h2,available_h2,...
```

Short horizons are 2/3/4/5/6/12 physical hours. Long horizons are
18/24/36/48/60/72 physical hours. Each explicitly named prediction has a
corresponding `available_hN` equal to `1` or `0`. Unavailable predictions must
be blank, not zero and not the current price. `model_id` identifies the actual
weekly checkpoint. Metadata must retain the selected recipe, scales, code,
seed, training period, calendar and artifact hashes.

`app/policies/business_prediction_csv.py::load_pair` checks both CSVs against
the close price at exactly the same observed origin. It refuses naive times,
duplicates, unsorted rows, missing columns and off-grid timestamps. It never
shifts rows, fills gaps or creates weekend bars. Missing forecast origins are
abstentions, not a reason to discard base OHLC bars.

The executor MUST process all observed base bars and apply the versioned
calendar before reading predictions. The operational start is market open
plus 12 hours; the forced-flat deadline is market close minus 12 hours. The
entry cutoff can be earlier (for example Thursday). Forced exit is independent
of CSV availability, thresholds and signals. A backtest without an executable
fill before the deadline is a failed execution, not a synthetic successful
close. These offsets remain explicit JSON parameters.

The existing `business_pair_policy.py` opens from the long family only and
manages an existing single position using both families in horizon order.
SL-first causes early close; TP-first does not. The predictor business executor
loads that policy from its pinned git blob and checks the calendar/masks again.

The CSV contains forecasts and availability, not realised future labels.
Availability is derived from the calendar and finite input-driven predictions,
never from whether the future target happened to exist. Research MAE masks
are retained separately.

# AGENTS.md — heuristic-strategy

Instructions for AI coding agents working in this repository.
Human-facing documentation is in [`README.md`](README.md).

## Project overview

heuristic-strategy backtests rule-based trading strategies with
[backtrader](https://www.backtrader.com/) and tunes their parameters with a
[DEAP](https://deap.readthedocs.io/) genetic algorithm. Strategies are plugins
selected by name; their entry/exit signals come either from prediction CSV files
or, per tick, from a running prediction_provider HTTP service. On top of the
single-run optimizer it carries an anchored walk-forward optimization (WFO)
harness that trains on a rolling window of years and reports out-of-sample
results per fold.

It does not train ML models, serve predictions, or place orders on any venue —
it reads historical OHLC CSVs and simulates. No repository in the stack imports
it as a dependency; it is a leaf tool.

## Agent quickstart (install → run → show the user results)

Verified end to end on Python 3.12.13 with backtrader release 1.9.78 build 123,
deap 1.4.4, pandas, numpy, requests.

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install backtrader deap pandas numpy requests matplotlib
pip install -e ../trading-contracts     # or: pip install trading-contracts
pip install -e .
```

`setup.py` declares only `trading-contracts>=0.1.0`; the real runtime
dependencies are in `requirements.txt`, which contains a corrupted line
(`pytestpip install graph`) — install packages individually as above instead of
using `-r requirements.txt`.

`pip install -e .` is **required**, not optional: strategies are resolved
through the `heuristic_strategy.plugins` entry-point group, so without the
install every run fails with `Plugin ... not found in group
heuristic_strategy.plugins`.

Use a dedicated virtualenv. This package installs a generic top-level `app`
package that sibling repositories also use; always run from the repository root
with `PYTHONPATH=./` (as `heuristic.sh` does) so the local `app` wins.

### 2. Smoke test

The full suite does not pass — parts of `tests/` predate the current codebase.
Run the healthy subset:

```bash
python -m pytest tests/unit_tests/test_prediction_entry_exit_policy.py \
                 tests/unit_tests/test_prediction_source_substitution.py \
                 tests/unit_tests/test_prediction_entry_exit_backtrader_replay.py -q
```

Expected: **17 passed** in about 1 second — the trade lifecycle policy, its
prediction sources and a backtrader replay of the policy.

For the full picture: `python -m pytest tests -q --continue-on-collection-errors`
gives **26 passed, 8 failed, 8 collection errors**. The errors are stale modules
importing an autoencoder-era API (`app.autoencoder_manager`, encoder plugins)
that no longer exists; the failures are in
`tests/unit_tests/test_prediction_client.py` and
`tests/integration_tests/test_configuration_handling.py`. Always pass the
`tests` path — collecting from the repository root also picks up the vendored
`timeseries-gan/` working-tree directory and fails.

### 3. Representative run — one short backtest

The bundled root configs (`config_direction_atr_cnn.json`,
`config_regime_adaptive.json`, ...) all set `"prediction_source": "API"` and
need a prediction_provider service on `http://127.0.0.1:8000`. The
CSV path needs no service, but the defaults in `app/config.py` use Windows path
separators (`tests\\data\\...`) and do not resolve on Linux, so pass the bundled
files explicitly:

```bash
mkdir -p run_out
PYTHONPATH=./ python app/main.py \
  --plugin ls_pred_strategy \
  --base_dataset_file tests/data/phase_2_3_base_d3.csv \
  --hourly_predictions_file tests/data/phase_2_3_cnn_1h_prediction_d3.csv \
  --daily_predictions_file tests/data/phase_2_3_cnn_1d_prediction_d3.csv \
  --population_size 2 --num_generations 1 \
  --trades_csv_file run_out/trades.csv \
  --summary_csv_file run_out/summary.csv \
  --balance_plot_file run_out/balance_plot.png \
  --save_parameters run_out/parameters.json \
  --save_config run_out/config_out.json \
  --save_log run_out/debug_log.json
```

Observed: exits 0 in ~6 seconds, evaluates 2 candidates for 1 generation
(a minimal GA, i.e. essentially two backtests) and writes `run_out/trades.csv`
(~100 trades with `open_dt,close_dt,volume,pnl,pips,duration,max_dd`),
`run_out/summary.csv` (best parameters + profit), `run_out/parameters.json`,
`run_out/config_out.json` and `run_out/balance_plot.png`.

Two things to know before running:

- **Create the output directory first.** The app does not create it and silently
  fails to save otherwise.
- **The balance plot is written to `./balance_plot.png` at the repository root
  and then *moved*** to `--balance_plot_file`. That overwrites (and, on a move,
  deletes) the committed root `balance_plot.png`. Restore it with
  `git checkout -- balance_plot.png`. The move also fails across filesystems, so
  keep the output directory inside the repository.
- Without `--population_size` / `--num_generations` the defaults are 20 × 30 —
  a full optimization sweep, not a quick run.

### 4. Analytics — walk-forward, profit vs risk

`run_wfo.py` runs anchored walk-forward optimization with a purely reactive
regime strategy (`regime_wfo` needs no predictions and no service). A two-fold
version finishes in seconds:

```bash
python run_wfo.py --train_years 1 --first_test_year 2009 --last_test_year 2010 \
  --population_size 4 --num_generations 1 --min_trades 1 \
  --save_results run_out/wfo_results.json --save_trades run_out/wfo_trades.csv
```

Observed: **17 seconds**, 2 folds over `tests/data/eurusd_hour_2005_2020.csv`,
printing a per-fold table (OOS profit, trades, win %, Sharpe, best params) and
an equity-curve line. Always pass `--save_results` / `--save_trades`: the
defaults (`wfo_results.json`, `wfo_oos_trades.csv`) overwrite committed
experiment records at the repository root.

`run_wfo.py --population_size 30 --num_generations 20` over 11 folds is the real
campaign and takes hours — do not launch it as a demo.

### 5. Final message to the user

> The backtest finished. Results are in `run_out/`: `trades.csv` (one row per
> closed trade with pnl, pips, duration and max drawdown), `summary.csv` (best
> parameter set and total profit), `parameters.json`, the merged effective
> config in `config_out.json`, and the equity plot `balance_plot.png`. The
> walk-forward run adds `run_out/wfo_results.json` (per-fold out-of-sample
> profit, trades, win rate, Sharpe) and `run_out/wfo_trades.csv`. There is no UI.
>
> Suggested first analysis: **profit against risk.** From `wfo_trades.csv`,
> group by `wfo_fold` and plot each fold's net profit (sum of `pnl`) on the
> y-axis against that fold's maximum equity drawdown (peak-to-trough of the
> running `balance` column) on the x-axis. Points in the upper-left are folds
> where the strategy earned more per unit of risk; a cloud that drifts
> down-and-right across folds is the usual sign that in-sample parameters are
> not surviving out of sample. The same plot works per trade on `trades.csv`
> using `pnl` against `max_dd`.

## Build, test and lint commands

```bash
pip install -e .                                    # required: registers strategy plugins
PYTHONPATH=./ python app/main.py --help             # CLI reference (exits 0)
python run_wfo.py --help                            # WFO options (exits 0)
sh heuristic.sh --help                              # same as app/main.py, sets PYTHONPATH for you
python -m pytest tests -q --continue-on-collection-errors
```

No linter, formatter or CI configuration exists in this repository. Do not add
one without asking.

## Layout

| Path | Contents |
|---|---|
| `app/main.py` | CLI entry point: merge config → load plugin → run pipeline |
| `app/config.py`, `app/cli.py`, `app/config_merger.py`, `app/config_handler.py` | Configuration defaults, flags, merge order, local/remote load-save |
| `app/optimizer.py` | DEAP genetic optimization of strategy parameters |
| `app/walk_forward_optimizer.py` | Rolling-window walk-forward loop, per-fold OOS metrics |
| `app/data_processor.py`, `app/data_handler.py` | CSV loading, alignment, prediction generation |
| `app/heuristic_strategy.py` | backtrader strategy wrapper |
| `app/plugins/` | Strategy plugins (`plugin_long_short_predictions`, `plugin_direction_atr`, `plugin_regime_adaptive`, `plugin_regime_wfo`, `plugin_btc_momentum`, `plugin_api_predictions`) |
| `app/policies/` | Reusable trade lifecycle policy `prediction_entry_exit_v1` and its prediction sources, typed with trading-contracts |
| `app/prediction_client.py` | HTTP client for prediction_provider |
| `run_wfo.py`, `merge_wfo_results.py`, `deploy_wfo.sh`, `monitor_wfo.sh` | Walk-forward harness |
| `run_phase_b_cnn.py`, `run_phase_c_ensemble.py`, `run_phase_d_neat.py`, `run_oracle_ceiling.py` | Historical phase experiment runners |
| `tests/data/` | Committed OHLC and prediction CSV fixtures |
| `tests/` | pytest suite (partly stale, see Smoke test) |

## Conventions and constraints

- **Config precedence**: defaults in `app/config.py` → `--load_config` JSON →
  CLI flags → unknown `--flags` (merged too, so any config key is settable from
  the CLI). Merge happens twice: once before the plugin loads and once after, so
  plugin `plugin_params` defaults participate.
- **Plugin architecture**: strategies live in the `heuristic_strategy.plugins`
  entry-point group and lifecycle policies in `trade_lifecycle_policy.plugins`,
  both declared in `setup.py`. A new plugin requires a `setup.py` entry and a
  reinstall.
- **Prediction sources**: `"prediction_source": "CSV"` (default; hourly + daily
  prediction CSVs) or `"API"` (per-tick `POST /api/v1/predict/entry` and
  `/exit` against `pp_api_url`). `regime_*` strategies are purely reactive and
  need neither.
- **Data contract**: base dataset is OHLC with a `DATE_TIME` column and
  `headers: true`; prediction CSVs are aligned to the base dataset's date range
  by `app/data_processor.py`, which trims all inputs to the common range.
- **Reproducibility**: GA runs are stochastic and unseeded. The committed
  `phase_*_results.json`, `wfo_results*.json` and `*_trades.csv` files are
  records of specific historical runs; they are not regenerated automatically
  and must not be "refreshed" casually.
- Costs are explicit in every strategy plugin (`spread_pips`,
  `commission_per_lot`, `slippage_pips`, `swap_per_lot_per_day`). Keep them when
  adding variants — results without costs are not comparable to the committed
  ones.

## Do not touch

- Committed experiment records at the repository root:
  `phase_b_cnn_results.json`, `phase_c_ensemble_results.json`,
  `phase_d_neat_results.json`, `oracle_ceiling_results.json`, `wfo_results*.json`,
  every `*_trades.csv`, `sweep_*_results.csv`, `balance_plot.png`,
  `parameters.json`, `config_*_out.json`. Runs overwrite these by default — send
  output to `run_out/` and revert accidental changes.
- `tests/data/` — committed fixtures; `eurusd_hour_2005_2020.csv` is ~4.8 MB and
  is the WFO input.
- `timeseries-gan/` — a gitignored vendored copy of a legacy project, not part
  of this package. It breaks root-level pytest collection; leave it alone.
- `.venv/`, `*.egg-info/`, `__pycache__/`, `.pytest_cache/` — generated.
- Sibling repositories (prediction_provider, predictor, lts, doin-*).

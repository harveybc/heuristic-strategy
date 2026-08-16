# Heuristic Strategy Optimizer

A Backtrader + DEAP toolkit for backtesting and genetic-algorithm
optimization of trading strategies. Strategies are pluggable via Python
entry points; prediction inputs come either from CSV files or, per tick,
from a running [prediction_provider](https://github.com/harveybc/prediction_provider)
HTTP service. On top of the single-run optimizer the repository carries a
walk-forward optimization (WFO) harness with committed phase results, and it
exports a reusable, contract-typed trade lifecycle policy consumed through
the `trade_lifecycle_policy.plugins` entry-point group.

## Status

**ACTIVE — component repository.** Package `heuristic_strategy` version
**0.1.0** ([`setup.py`](setup.py)). It is a leaf tool: no sibling repository
imports it as a dependency (siblings that show similar code use their own
vendored copies).

**Trading status: backtesting and simulation only.** This repository places
no orders on any venue; the wider stack it belongs to runs in paper/demo
venues only, with real capital not enabled. Nothing here is financial
advice.

## Run this with an AI agent

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or any coding agent with shell access:

> Read `AGENTS.md` in this repository and follow the **Agent quickstart** section end to end: set up the environment, run the smoke test, execute the example backtest, then tell me the exact file paths or URL where I can see the results and one analysis I should try first.

`AGENTS.md` is the [agents.md](https://agents.md) convention, read natively by most coding agents.

## Role and non-responsibilities

**Owns**

- Strategy backtests over historical OHLC data (Backtrader) and GA
  optimization of strategy parameters (DEAP), including anchored
  walk-forward optimization ([`run_wfo.py`](run_wfo.py)).
- The trade lifecycle policy `prediction_entry_exit_v1`
  ([`app/policies/prediction_entry_exit.py`](app/policies/prediction_entry_exit.py),
  [`app/policies/prediction_sources.py`](app/policies/prediction_sources.py)),
  typed with [trading-contracts](https://github.com/harveybc/trading-contracts)
  models and registered under `trade_lifecycle_policy.plugins`.

**Does not own**

- Prediction serving (prediction_provider), live/paper execution
  ([lts](https://github.com/harveybc/lts)), model training
  ([predictor](https://github.com/harveybc/predictor)) or distributed
  optimization ([doin-node](https://github.com/harveybc/doin-node)); this
  repository is not a DOIN network participant.

## Architecture and plugin reference

The CLI ([`app/main.py`](app/main.py)) merges configuration (defaults →
`--load_config` JSON → CLI flags), loads one strategy plugin from the
`heuristic_strategy.plugins` group and runs the backtest/GA pipeline
([`app/optimizer.py`](app/optimizer.py),
[`app/walk_forward_optimizer.py`](app/walk_forward_optimizer.py)).

### Strategy plugins — `heuristic_strategy.plugins` (from `setup.py`)

| Entry point | Module | Description |
|---|---|---|
| `default` | [`app/plugins/plugin_long_short_predictions.py`](app/plugins/plugin_long_short_predictions.py) | CSV-based long/short prediction strategy (hourly + daily prediction files) |
| `ls_pred_strategy` | same module as `default` | Alias registration of the CSV long/short strategy |
| `api_predictions` | [`app/plugins/plugin_api_predictions.py`](app/plugins/plugin_api_predictions.py) | Queries prediction_provider `POST /api/v1/predict/entry` / `POST /api/v1/predict/exit` per tick; `GET /api/v1/model/info` for metadata |
| `direction_atr` | [`app/plugins/plugin_direction_atr.py`](app/plugins/plugin_direction_atr.py) | Direction predictions with ATR-scaled TP/SL geometry |
| `regime_adaptive` | [`app/plugins/plugin_regime_adaptive.py`](app/plugins/plugin_regime_adaptive.py) | Regime-conditional parameter switching |
| `regime_wfo` | [`app/plugins/plugin_regime_wfo.py`](app/plugins/plugin_regime_wfo.py) | Regime strategy variant used by the WFO harness |
| `btc_momentum` | [`app/plugins/plugin_btc_momentum.py`](app/plugins/plugin_btc_momentum.py) | BTC momentum strategy |

### Lifecycle policy plugins — `trade_lifecycle_policy.plugins`

| Entry point | Module | Description |
|---|---|---|
| `prediction_entry_exit_v1` | [`app/policies/prediction_entry_exit.py`](app/policies/prediction_entry_exit.py) | Reusable entry/exit decision policy over prediction sources, emitting trading-contracts-typed decisions |

### Walk-forward optimization harness

[`run_wfo.py`](run_wfo.py) (anchored expanding-window WFO over the 15-year
EURUSD dataset), [`merge_wfo_results.py`](merge_wfo_results.py),
[`deploy_wfo.sh`](deploy_wfo.sh), [`monitor_wfo.sh`](monitor_wfo.sh), and
the phase runners [`run_phase_b_cnn.py`](run_phase_b_cnn.py),
[`run_phase_c_ensemble.py`](run_phase_c_ensemble.py),
[`run_phase_d_neat.py`](run_phase_d_neat.py),
[`run_oracle_ceiling.py`](run_oracle_ceiling.py). Their committed outputs
(`phase_b_cnn_results.json`, `phase_c_ensemble_results.json`,
`phase_d_neat_results.json`, `oracle_ceiling_results.json`, matching
`*_trades.csv`, [`phase_d_neat_configs/`](phase_d_neat_configs/)) are
historical experiment records.

## Requirements

- Python: no `python_requires` declared in [`setup.py`](setup.py); verified
  with **Python 3.12.13** (backtrader 1.9.78.123, deap 1.4).
- [`setup.py`](setup.py) declares only `trading-contracts>=0.1.0`; the
  working dependency set (backtrader, deap, pandas, numpy, requests, …) is
  in [`requirements.txt`](requirements.txt) — which currently contains one
  malformed line (see [Limitations](#limitations)).

## Installation

```bash
git clone https://github.com/harveybc/trading-contracts.git
git clone https://github.com/harveybc/heuristic-strategy.git
pip install -e ./trading-contracts
cd heuristic-strategy
pip install backtrader deap pandas numpy requests matplotlib
pip install -e .
```

Not re-executed in a clean environment for this document (unverified).
The package installs a generic top-level package named `app`, which can be
shadowed by sibling repositories in a shared environment — run from the
repository root with `PYTHONPATH=./` (as [`heuristic.sh`](heuristic.sh)
does) or use a dedicated virtual environment.

## Smallest working example

From the repository root:

```bash
PYTHONPATH=./ python app/main.py --help    # verified: exits 0, prints CLI reference
python run_wfo.py --help                   # verified: exits 0, prints WFO options
```

A short CSV-only backtest (no service required, verified: exits 0 in ~6 s,
writes `run_out/trades.csv`, `run_out/summary.csv` and the balance plot):

```bash
mkdir -p run_out
PYTHONPATH=./ python app/main.py --plugin ls_pred_strategy \
  --base_dataset_file tests/data/phase_2_3_base_d3.csv \
  --hourly_predictions_file tests/data/phase_2_3_cnn_1h_prediction_d3.csv \
  --daily_predictions_file tests/data/phase_2_3_cnn_1d_prediction_d3.csv \
  --population_size 2 --num_generations 1 \
  --trades_csv_file run_out/trades.csv --summary_csv_file run_out/summary.csv \
  --balance_plot_file run_out/balance_plot.png \
  --save_parameters run_out/parameters.json --save_config run_out/config_out.json
```

The dataset paths must be given explicitly: the defaults in
[`app/config.py`](app/config.py) use Windows separators (`tests\data\...`) and
do not resolve on Linux. Create the output directory first — the app does not.

A two-fold walk-forward run needs no predictions at all (verified: 17 s):

```bash
python run_wfo.py --train_years 1 --first_test_year 2009 --last_test_year 2010 \
  --population_size 4 --num_generations 1 --min_trades 1 \
  --save_results run_out/wfo_results.json --save_trades run_out/wfo_trades.csv
```

A repository-owned end-to-end config is
[`config_direction_atr_cnn.json`](config_direction_atr_cnn.json): it runs the
`direction_atr` plugin over the committed dataset
[`tests/data/phase_1c_direction_test_ohlc.csv`](tests/data/phase_1c_direction_test_ohlc.csv)
with `prediction_source: "API"`, so it additionally requires a local
prediction_provider instance listening on port 8000 (execution unverified
for this document):

```bash
PYTHONPATH=./ python app/main.py --load_config config_direction_atr_cnn.json
```

## Tests and validation

```bash
python -m pytest tests -q --continue-on-collection-errors
```

Observed result: `26 passed, 8 failed, 8 errors`. The collection errors come
from stale test modules that still import an autoencoder-era API
(`app.autoencoder_manager`, encoder plugins) which no longer exists; the
failures are in `tests/unit_tests/test_prediction_client.py` and
`tests/integration_tests/test_configuration_handling.py`. The healthy subset —
the trade lifecycle policy, its prediction sources and the backtrader replay —
passes cleanly:

```bash
python -m pytest tests/unit_tests/test_prediction_entry_exit_policy.py \
                 tests/unit_tests/test_prediction_source_substitution.py \
                 tests/unit_tests/test_prediction_entry_exit_backtrader_replay.py -q
# observed: 17 passed
```

Collecting from the repository root instead of `tests/` additionally pulls in
the vendored `timeseries-gan/tests` and fails — always pass the `tests` path.

## Artifacts, data and outputs

- Inputs: OHLC and prediction CSVs under [`tests/data/`](tests/data/).
- Outputs: optimized parameters (`config_*_out.json`,
  [`parameters.json`](parameters.json)), balance plots, debug logs and the
  committed WFO/phase result JSON + trade CSV files listed above.
- Reproducibility: GA runs are stochastic; committed result files record
  the outcomes of specific historical runs and are not regenerated
  automatically.

## Safety, security and credentials

No credentials are used or stored; the only network access is to a
locally-run prediction_provider URL supplied via `pp_api_url`. Backtesting
only — no venue connectivity, no capital at risk, not financial advice.

## Limitations

- **Vendored `timeseries-gan/` subtree.** A full copy of the legacy
  [timeseries-gan](https://github.com/harveybc/timeseries-gan) project
  (predecessor of
  [synthetic-datagen](https://github.com/harveybc/synthetic-datagen)) may
  be present locally at `timeseries-gan/` — a gitignored working-tree
  directory, not committed to this repository's history. When present it
  is not part of the installed package, breaks root-level pytest
  collection, and its `LICENSE.txt` applies to that subtree only.
- **Stale test modules.** Part of `tests/` predates the current codebase
  and fails collection (8 errors, see above).
- **Malformed `requirements.txt`.** It contains the line
  `pytestpip install graph` (a corrupted merge of `pytest` and a pip
  command); install dependencies individually as shown above. It also omits
  `matplotlib`, which `app/heuristic_strategy.py` imports at module level.
- **Declared vs. actual dependencies.** `setup.py` declares only
  `trading-contracts`; Backtrader/DEAP and the rest must be installed
  separately.
- **The balance plot is written to `./balance_plot.png` and then moved** to
  `--balance_plot_file`, so a run overwrites (or, on a successful move,
  removes) the committed root `balance_plot.png`; the move also fails across
  filesystems. Keep the output directory inside the repository and restore the
  file with `git checkout -- balance_plot.png`.
- Committed experiment residue at the repository root (result JSONs, trade
  CSVs, `balance_plot.png`, `debug_log_*` outputs, prediction CSV exports)
  is historical record, not documentation.
- No top-level LICENSE file currently exists in this repository.

#!/usr/bin/env python3
"""
Phase 5 — Question 3: EUR/USD Daily MR Operational Deployment via OANDA

Tasks 3.1-3.6: Build MR strategy, position sizing, OANDA demo deployment,
logging/monitoring, auto-pause protocol, 90-day observation.

This script:
1. Generates the LTS strategy plugin for EUR/USD daily MR
2. Runs a comprehensive backtest validation
3. Creates deployment configuration
4. Sets up monitoring infrastructure
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.evaluation_harness import annualized_sharpe, rolling_window_evaluation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PPY_DAILY = 252

# ============================================================
# TASK 3.1 — EUR/USD Daily MR Strategy Parameters
# ============================================================
# From Phase 3.5 "plateau center" and Phase 4 Track A refinement
MR_PARAMS = {
    "lookback": 20,          # rolling window for z-score
    "z_entry": 1.5,          # z-score threshold to enter
    "z_exit": 0.5,           # z-score threshold to exit
    "max_holding_bars": 30,  # maximum bars to hold position
    "sl_atr_multiple": 3.0,  # stop-loss: 3x ATR
    "tp_z_exit": True,       # take-profit at z_exit level
    "atr_lookback": 20,      # ATR computation window
}

# ============================================================
# TASK 3.2 — Position Sizing Parameters
# ============================================================
POSITION_SIZING = {
    "target_sharpe_range": (0.2, 0.4),
    "max_drawdown_pct": 0.15,        # 15% max allowed DD
    "risk_per_trade_pct": 0.005,     # 0.5% of equity per trade
    "max_leverage": 2.0,             # max 2x leverage
    "initial_capital": 10000,        # OANDA demo account
    "currency": "USD",
    "instrument": "EUR_USD",
    "min_units": 100,                # minimum trade size
}

# ============================================================
# TASK 3.5 — Auto-Pause Protocol
# ============================================================
AUTO_PAUSE = {
    "max_drawdown_pct": 0.15,        # 15% DD → pause
    "max_consecutive_losses": 10,     # 10 consecutive losses → pause
    "max_slippage_multiple": 2.0,     # 2× expected slippage → pause
    "max_divergence_sigma": 2.0,      # 2σ divergence from backtest → pause
    "review_period_days": 7,          # weekly review
    "min_observation_days": 90,       # 90-day minimum before conclusions
}


def run_mr_backtest(close, dates, params=None):
    """
    Run EUR/USD daily MR strategy backtest.
    Returns per-bar positions, entry/exit logs.
    """
    if params is None:
        params = MR_PARAMS

    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    n = len(close)
    lookback = params["lookback"]
    z_entry = params["z_entry"]
    z_exit = params["z_exit"]
    max_hold = params["max_holding_bars"]
    sl_mult = params["sl_atr_multiple"]

    positions = np.zeros(n)
    trade_log = []
    entry_bar = -999
    entry_price = 0
    entry_z = 0

    # Precompute ATR
    atr = np.zeros(n)
    for i in range(1, n):
        window = max(1, min(params["atr_lookback"], i))
        atr[i] = np.std(log_ret[max(0, i - window):i]) * close[i]

    for i in range(lookback, n):
        window_close = close[max(0, i - lookback):i + 1]
        mean_price = np.mean(window_close)
        std_price = np.std(window_close)
        if std_price < 1e-12:
            continue
        z = (close[i] - mean_price) / std_price

        if positions[i - 1] != 0:
            bars_held = i - entry_bar
            current_pnl = positions[i - 1] * (close[i] - entry_price) / entry_price
            sl_level = sl_mult * atr[entry_bar] / close[entry_bar] if atr[entry_bar] > 0 else 0.03

            # Exit conditions
            exit_reason = None
            if abs(z) < z_exit:
                exit_reason = "z_exit"
            elif current_pnl < -sl_level:
                exit_reason = "stop_loss"
            elif bars_held >= max_hold:
                exit_reason = "max_hold"

            if exit_reason:
                trade_log.append({
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "entry_date": str(dates[entry_bar]),
                    "exit_date": str(dates[i]),
                    "direction": "long" if positions[i - 1] > 0 else "short",
                    "entry_price": float(entry_price),
                    "exit_price": float(close[i]),
                    "pnl_pct": round(float(current_pnl) * 100, 4),
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                    "entry_z": round(float(entry_z), 3),
                    "exit_z": round(float(z), 3),
                })
                positions[i] = 0
            else:
                positions[i] = positions[i - 1]
        else:
            # Entry conditions
            if z > z_entry:
                positions[i] = -1  # MR: sell when overextended
                entry_bar = i
                entry_price = close[i]
                entry_z = z
            elif z < -z_entry:
                positions[i] = 1  # MR: buy when oversold
                entry_bar = i
                entry_price = close[i]
                entry_z = z

    return positions, trade_log


def compute_position_size(equity, atr_pct, params=None):
    """
    Compute position size in units for a trade.
    Risk 0.5% of equity, capped by max leverage.
    """
    if params is None:
        params = POSITION_SIZING

    risk_amount = equity * params["risk_per_trade_pct"]
    # Units such that SL (3*ATR) = risk_amount
    sl_pct = MR_PARAMS["sl_atr_multiple"] * atr_pct
    if sl_pct < 1e-6:
        sl_pct = 0.01
    units = risk_amount / (sl_pct * equity) * equity
    # Cap by leverage
    max_units = equity * params["max_leverage"] / 1.0  # EUR/USD ~1:1
    units = min(units, max_units)
    units = max(units, params["min_units"])
    return int(units)


# ============================================================
# TASK 3.1 — Generate LTS Strategy Plugin
# ============================================================
LTS_PLUGIN_CODE = '''#!/usr/bin/env python3
"""
EUR/USD Daily Mean-Reversion Strategy Plugin for LTS.

Phase 5 Q3: Operational deployment via OANDA demo account.
Strategy: z-score mean reversion on EUR/USD daily close.
"""
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from app.plugin_base import PluginBase
except ImportError:
    class PluginBase:
        """Fallback if LTS not installed."""
        plugin_params = {}
        def __init__(self, config=None):
            self.params = dict(self.plugin_params)
            if config:
                self.params.update(config)


class EurUsdMrStrategy(PluginBase):
    """EUR/USD daily mean-reversion strategy."""

    plugin_params = {
        "lookback": 20,
        "z_entry": 1.5,
        "z_exit": 0.5,
        "max_holding_bars": 30,
        "sl_atr_multiple": 3.0,
        "atr_lookback": 20,
        "risk_per_trade_pct": 0.005,
        "max_leverage": 2.0,
        "instrument": "EUR_USD",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._price_history = []
        self._position = 0  # 0=flat, 1=long, -1=short
        self._entry_bar = 0
        self._entry_price = 0.0
        self._bars_since_entry = 0

    def _compute_z_score(self, prices):
        """Compute z-score of latest price vs rolling window."""
        if len(prices) < self.params["lookback"]:
            return 0.0
        window = prices[-self.params["lookback"]:]
        mean_p = np.mean(window)
        std_p = np.std(window)
        if std_p < 1e-12:
            return 0.0
        return (prices[-1] - mean_p) / std_p

    def _compute_atr_pct(self, prices):
        """ATR as percentage of price."""
        if len(prices) < self.params["atr_lookback"] + 1:
            return 0.01
        returns = np.diff(np.log(prices[-self.params["atr_lookback"]-1:] + 1e-12))
        return float(np.std(returns))

    def generate_signal(self, asset, market_data=None, predictions=None):
        """
        Generate trading signal for EUR/USD MR strategy.

        Returns:
            dict: {action, parameters}
        """
        if market_data is None:
            return {"action": "none", "parameters": {}}

        # Extract current price
        current_price = None
        if isinstance(market_data, dict):
            current_price = market_data.get("close") or market_data.get("price")
        elif isinstance(market_data, (list, np.ndarray)):
            current_price = float(market_data[-1])

        if current_price is None:
            return {"action": "none", "parameters": {}}

        self._price_history.append(float(current_price))

        # Need enough history
        if len(self._price_history) < self.params["lookback"]:
            return {"action": "none", "parameters": {}}

        z = self._compute_z_score(self._price_history)
        atr_pct = self._compute_atr_pct(self._price_history)

        # If in position, check exit
        if self._position != 0:
            self._bars_since_entry += 1
            current_pnl = self._position * (current_price - self._entry_price) / self._entry_price
            sl_level = self.params["sl_atr_multiple"] * atr_pct

            should_exit = False
            exit_reason = ""

            if abs(z) < self.params["z_exit"]:
                should_exit = True
                exit_reason = "z_exit_reached"
            elif current_pnl < -sl_level:
                should_exit = True
                exit_reason = "stop_loss"
            elif self._bars_since_entry >= self.params["max_holding_bars"]:
                should_exit = True
                exit_reason = "max_hold_reached"

            if should_exit:
                logger.info(f"EXIT {asset}: reason={exit_reason}, pnl={current_pnl:.4f}, "
                           f"bars={self._bars_since_entry}, z={z:.2f}")
                self._position = 0
                self._bars_since_entry = 0
                return {
                    "action": "close",
                    "parameters": {
                        "reason": exit_reason,
                        "pnl_pct": round(current_pnl * 100, 4),
                    }
                }
            return {"action": "none", "parameters": {}}

        # Not in position — check entry
        if z > self.params["z_entry"]:
            # Overextended upward → sell (mean reversion)
            direction = "sell"
            self._position = -1
        elif z < -self.params["z_entry"]:
            # Overextended downward → buy
            direction = "buy"
            self._position = 1
        else:
            return {"action": "none", "parameters": {}}

        self._entry_price = current_price
        self._entry_bar = len(self._price_history)
        self._bars_since_entry = 0

        # Position sizing
        sl_distance = self.params["sl_atr_multiple"] * atr_pct * current_price
        tp_price = current_price - self._position * abs(z - np.sign(z) * self.params["z_exit"]) * np.std(self._price_history[-self.params["lookback"]:])
        sl_price = current_price + self._position * sl_distance

        # Ensure SL/TP are on correct side
        if direction == "buy":
            sl_price = current_price - abs(sl_distance)
            tp_price = max(tp_price, current_price + abs(sl_distance) * 0.5)
        else:
            sl_price = current_price + abs(sl_distance)
            tp_price = min(tp_price, current_price - abs(sl_distance) * 0.5)

        logger.info(f"ENTRY {asset}: {direction}, z={z:.2f}, price={current_price:.5f}, "
                   f"sl={sl_price:.5f}, tp={tp_price:.5f}")

        return {
            "action": "open",
            "parameters": {
                "side": direction,
                "instrument": self.params["instrument"],
                "stop_loss": round(sl_price, 5),
                "take_profit": round(tp_price, 5),
                "order_type": "MARKET",
                "z_score": round(z, 3),
                "atr_pct": round(atr_pct, 6),
            }
        }
'''


# ============================================================
# MAIN — Backtest validation + deployment config generation
# ============================================================
def main():
    print("=" * 70)
    print("PHASE 5 — Q3: EUR/USD MR OPERATIONAL DEPLOYMENT")
    print("=" * 70)

    # Load EUR/USD data
    import yfinance as yf
    print("\n--- Loading EUR/USD daily data ---")
    cache_path = os.path.join(RESULTS_DIR, "eurusd_daily.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = yf.download("EURUSD=X", start="2003-01-01", end="2026-12-31",
                          interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(cache_path)

    print(f"  {len(df)} daily bars from {df.index[0].date()} to {df.index[-1].date()}")

    close = df["Close"].values.astype(float)
    dates = df.index

    # Task 3.1: Run backtest
    print("\n--- Task 3.1: MR Backtest Validation ---")
    positions, trade_log = run_mr_backtest(close, dates)

    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])

    # Strategy returns
    strat_ret = positions[:-1] * log_ret[1:]
    sr = annualized_sharpe(strat_ret, PPY_DAILY)
    rolling = rolling_window_evaluation(strat_ret, PPY_DAILY)

    # Trade statistics
    n_trades = len(trade_log)
    if n_trades > 0:
        wins = [t for t in trade_log if t["pnl_pct"] > 0]
        losses = [t for t in trade_log if t["pnl_pct"] <= 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
        avg_bars = np.mean([t["bars_held"] for t in trade_log])

        # Exit reason breakdown
        exit_reasons = {}
        for t in trade_log:
            r = t["exit_reason"]
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        avg_bars = 0
        exit_reasons = {}

    print(f"  Sharpe: {sr:+.3f}")
    print(f"  Worst 2Y: {rolling['worst_window_sharpe']:+.3f}")
    print(f"  Trades: {n_trades}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Avg win: {avg_win:+.3f}%, Avg loss: {avg_loss:+.3f}%")
    print(f"  Avg bars held: {avg_bars:.1f}")
    print(f"  Exit reasons: {exit_reasons}")

    # Equity curve and drawdown
    equity = np.cumsum(strat_ret)
    eq_curve = np.exp(equity)
    peak = np.maximum.accumulate(eq_curve)
    dd = (peak - eq_curve) / (peak + 1e-12)
    max_dd = float(np.max(dd))
    print(f"  Max drawdown: {max_dd:.1%}")
    print(f"  Total return: {float(np.exp(np.sum(strat_ret)) - 1):.1%}")

    # Task 3.2: Position sizing validation
    print("\n--- Task 3.2: Position Sizing ---")
    initial_capital = POSITION_SIZING["initial_capital"]
    typical_atr_pct = np.mean(np.abs(log_ret[20:])) * MR_PARAMS["sl_atr_multiple"]
    typical_units = compute_position_size(initial_capital, typical_atr_pct / MR_PARAMS["sl_atr_multiple"])
    leverage_used = typical_units / initial_capital  # rough estimate for EUR/USD
    print(f"  Initial capital: ${initial_capital:,.0f}")
    print(f"  Typical position: {typical_units:,} units")
    print(f"  Leverage used: {leverage_used:.1f}x")
    print(f"  Risk per trade: {POSITION_SIZING['risk_per_trade_pct']:.1%} = ${initial_capital * POSITION_SIZING['risk_per_trade_pct']:,.0f}")

    # Task 3.3: Write LTS plugin file
    print("\n--- Task 3.3: LTS Plugin Generation ---")
    lts_plugin_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                   "lts", "plugins_strategy")
    plugin_path = os.path.join(lts_plugin_dir, "eurusd_mr_strategy.py")

    if os.path.isdir(lts_plugin_dir):
        with open(plugin_path, "w") as f:
            f.write(LTS_PLUGIN_CODE)
        print(f"  Written: {plugin_path}")
    else:
        # Write alongside this script as fallback
        fallback_path = os.path.join(os.path.dirname(__file__), "eurusd_mr_strategy.py")
        with open(fallback_path, "w") as f:
            f.write(LTS_PLUGIN_CODE)
        print(f"  LTS dir not found, written to: {fallback_path}")
        plugin_path = fallback_path

    # Task 3.4: Deployment config
    print("\n--- Task 3.4: Deployment Configuration ---")
    deployment_config = {
        "strategy": {
            "name": "EurUsdMrStrategy",
            "plugin_path": plugin_path,
            "params": MR_PARAMS,
        },
        "broker": {
            "plugin": "OandaBroker",
            "environment": "practice",  # DEMO account only
            "instrument": "EUR_USD",
            "note": "Set OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN environment variables",
        },
        "position_sizing": POSITION_SIZING,
        "auto_pause": AUTO_PAUSE,
        "schedule": {
            "frequency": "daily",
            "check_time_utc": "21:00",  # After NY close
            "note": "Signal generated once per day at market close",
        },
        "monitoring": {
            "log_dir": os.path.join(RESULTS_DIR, "q3_deployment_logs"),
            "trade_log_file": "trades.jsonl",
            "weekly_report_file": "weekly_reports.jsonl",
            "metrics_tracked": [
                "cumulative_pnl", "current_drawdown", "consecutive_losses",
                "sharpe_rolling_30d", "avg_slippage", "trade_count",
            ],
        },
        "observation_period": {
            "start_date": datetime.date.today().isoformat(),
            "min_days": 90,
            "end_date": (datetime.date.today() + datetime.timedelta(days=90)).isoformat(),
        },
    }

    config_path = os.path.join(RESULTS_DIR, "phase5_q3_deployment_config.json")
    with open(config_path, "w") as f:
        json.dump(deployment_config, f, indent=2, default=str)
    print(f"  Config: {config_path}")

    # Create log directory
    log_dir = deployment_config["monitoring"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    print(f"  Log dir: {log_dir}")

    # Task 3.5: Auto-pause protocol summary
    print("\n--- Task 3.5: Auto-Pause Protocol ---")
    for rule, value in AUTO_PAUSE.items():
        print(f"  {rule}: {value}")

    # Task 3.6: Observation schedule
    print("\n--- Task 3.6: 90-Day Observation ---")
    print(f"  Start: {deployment_config['observation_period']['start_date']}")
    print(f"  End: {deployment_config['observation_period']['end_date']}")
    print(f"  Weekly reviews on Fridays")

    # Assemble full results
    print("\n" + "=" * 70)
    print("Q3 — DEPLOYMENT READINESS SUMMARY")
    print("=" * 70)

    # Backtest validation verdict
    ready = True
    issues = []
    if sr < 0:
        ready = False
        issues.append(f"Negative Sharpe ({sr:+.3f})")
    if rolling["worst_window_sharpe"] < -2.0:
        ready = False
        issues.append(f"Worst 2Y too extreme ({rolling['worst_window_sharpe']:+.3f})")
    if n_trades < 20:
        ready = False
        issues.append(f"Too few trades ({n_trades})")
    if win_rate < 0.30:
        ready = False
        issues.append(f"Win rate too low ({win_rate:.1%})")

    if ready:
        print(f"\n  ✅ READY FOR DEMO DEPLOYMENT")
    else:
        print(f"\n  ⚠ DEPLOYMENT CONCERNS:")
        for issue in issues:
            print(f"    - {issue}")
        print(f"\n  Deploying to DEMO account for observation regardless (Phase 5 rule: Q3 runs regardless)")

    output = {
        "question": "Q3",
        "deployment_ready": ready,
        "deployment_issues": issues,
        "backtest": {
            "sharpe": round(sr, 4),
            "worst_2y_sharpe": rolling["worst_window_sharpe"],
            "regime_robustness": rolling["regime_robustness"],
            "n_trades": n_trades,
            "win_rate": round(win_rate, 4),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "avg_bars_held": round(avg_bars, 1) if avg_bars else 0,
            "max_drawdown": round(max_dd, 4),
            "total_return": round(float(np.exp(np.sum(strat_ret)) - 1), 4),
            "exit_reasons": exit_reasons,
        },
        "strategy_params": MR_PARAMS,
        "position_sizing": POSITION_SIZING,
        "auto_pause": AUTO_PAUSE,
        "plugin_path": plugin_path,
        "config_path": config_path,
        "n_sample_trades": trade_log[:10],  # first 10 trades for review
    }

    output_path = os.path.join(RESULTS_DIR, "phase5_q3_deployment_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()

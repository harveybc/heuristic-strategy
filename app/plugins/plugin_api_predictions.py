"""
Plugin for API-based Heuristic Trading Strategy.

Uses the Prediction Provider API (entry/exit endpoints) to obtain binary
directional predictions on every tick, instead of loading CSV prediction files.

Flow:
  - When NO position is open: calls POST /api/v1/predict/entry to get
    buy_entry_binary and sell_entry_binary signals.
  - When a position IS open: calls POST /api/v1/predict/exit to get
    exit_binary signal (keep open or close early).

Configuration:
    plugin: api_predictions
    prediction_source: API
    pp_api_url: http://127.0.0.1:8000
    pp_timeout: 5.0
"""

import datetime
import os
import backtrader as bt
import pandas as pd
import numpy as np
import os as _os
_QUIET = _os.environ.get("STRATEGY_QUIET", "0") == "1"


class Plugin:
    """
    API Prediction Plugin for Heuristic Trading Strategy.

    - Embeds an ApiHeuristicStrategy that fetches binary predictions per tick
      from the Prediction Provider API (entry + exit endpoints).
    - Exposes plugin_params with default values and the required methods for
      optimization.
    """

    plugin_params = {
        'pip_cost': 0.00001,
        'rel_volume': 0.02,
        'min_order_volume': 10000,
        'max_order_volume': 1000000,
        'leverage': 100,
        'profit_threshold': 5,
        'min_drawdown_pips': 10,
        'tp_multiplier': 0.9,
        'sl_multiplier': 2.0,
        'lower_rr_threshold': 0.5,
        'upper_rr_threshold': 2.0,
        'max_trades_per_5days': 3,
        'exit_variant': 'E',
        # Trading costs
        'spread_pips': 2.0,
        'commission_per_lot': 7.0,
        'slippage_pips': 1.0,
        'swap_per_lot_per_day': 10.0,
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.trades = []

    def set_params(self, **kwargs):
        """Update plugin parameters dynamically."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        """Return debugging information for the plugin."""
        return {key: self.params[key] for key in self.params}

    # -------------------------------------------------------------------------
    # Required optimization interface
    # -------------------------------------------------------------------------
    def get_optimizable_params(self):
        """Return parameters that can be optimized along with their bounds."""
        return [
            ("profit_threshold", 0.5, 20),
            ("tp_multiplier", 0.5, 1.5),
            ("sl_multiplier", 1.5, 6.0),
            ("lower_rr_threshold", 0.2, 1.0),
            ("upper_rr_threshold", 1.3, 6.0),
        ]

    def evaluate_candidate(self, individual, base_data, hourly_predictions, daily_predictions, config):
        """
        Evaluate a candidate parameter set via the Prediction Provider API.

        Predictions are fetched per-tick inside the strategy (no CSV files
        needed).  hourly_predictions and daily_predictions are ignored.
        """
        import backtrader as bt

        profit_threshold, tp_multiplier, sl_multiplier, lower_rr, upper_rr = individual

        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            self.ApiHeuristicStrategy,
            pip_cost=self.params['pip_cost'],
            rel_volume=self.params['rel_volume'],
            min_order_volume=self.params['min_order_volume'],
            max_order_volume=self.params['max_order_volume'],
            leverage=self.params['leverage'],
            profit_threshold=profit_threshold,
            min_drawdown_pips=self.params['min_drawdown_pips'],
            tp_multiplier=tp_multiplier,
            sl_multiplier=sl_multiplier,
            lower_rr_threshold=lower_rr,
            upper_rr_threshold=upper_rr,
            max_trades_per_5days=self.params['max_trades_per_5days'],
            exit_variant=self.params['exit_variant'],
            swap_per_lot_per_day=self.params['swap_per_lot_per_day'],
            pp_api_url=config.get("pp_api_url", "http://127.0.0.1:8000"),
            pp_timeout=float(config.get("pp_timeout", 5.0)),
        )

        data_feed = bt.feeds.PandasData(dataname=base_data)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(10000.0)

        # Apply realistic trading costs
        spread_cost = self.params['spread_pips'] * self.params['pip_cost']
        slippage_cost = self.params['slippage_pips'] * self.params['pip_cost']
        total_spread = spread_cost + slippage_cost
        commission_per_unit = self.params['commission_per_lot'] / 100000.0
        cerebro.broker.setcommission(commission=commission_per_unit, margin=None, mult=1.0)
        cerebro.broker.set_slippage_fixed(total_spread / 2.0, slip_open=True, slip_limit=True)

        try:
            runresult = cerebro.run()
        except Exception as e:
            if not _QUIET:
                print("Error during backtest (API mode):", e)
            return (-1e6, {"num_trades": 0, "win_pct": 0, "max_dd": 0, "sharpe": 0})

        final_value = cerebro.broker.getvalue()
        profit = final_value - 10000.0
        if not _QUIET:
            print(f"Evaluated candidate {individual} -> Profit: {profit:.2f}")

        strat_instance = runresult[0]
        trades_list = getattr(strat_instance, "trades", [])
        self.trades = trades_list

        num_trades = len(trades_list)
        stats = {"num_trades": num_trades, "win_pct": 0, "max_dd": 0, "sharpe": 0}
        if num_trades > 0:
            wins = sum(1 for tr in trades_list if tr['pnl'] > 0)
            win_pct = (wins / num_trades) * 100
            max_dd = max(tr['max_dd'] for tr in trades_list)
            profits = [tr['pnl'] for tr in trades_list]
            std_profit = np.std(profits) if num_trades > 1 else 0
            sharpe = (profit / std_profit) if std_profit > 0 else 0
            stats.update({"win_pct": win_pct, "max_dd": max_dd, "sharpe": sharpe})

        if not _QUIET:
            print(f"[EVALUATE] Candidate result => Profit: {profit:.2f}, "
                  f"Trades: {stats.get('num_trades', 0)}, "
                  f"Win%: {stats.get('win_pct', 0):.1f}, "
                  f"MaxDD: {stats.get('max_dd', 0):.2f}, "
                  f"Sharpe: {stats.get('sharpe', 0):.2f}")

        return (profit, stats)

    # =========================================================================
    # Inner Backtrader Strategy — API mode (entry/exit endpoints)
    # =========================================================================
    class ApiHeuristicStrategy(bt.Strategy):
        """
        Forex Dynamic Volume Strategy using Prediction Provider API.

        - When NO position: calls /predict/entry → buy_entry_binary,
          sell_entry_binary → decides to buy, sell, or skip.
        - When IN position: first checks hard TP/SL levels, then calls
          /predict/exit → exit_binary → decides to keep or close early.
        """

        params = dict(
            pip_cost=0.00001,
            rel_volume=0.02,
            min_order_volume=10000,
            max_order_volume=1000000,
            leverage=100,
            profit_threshold=5,
            min_drawdown_pips=10,
            tp_multiplier=0.9,
            sl_multiplier=2.0,
            lower_rr_threshold=0.5,
            upper_rr_threshold=2.0,
            max_trades_per_5days=3,
            exit_variant='E',
            swap_per_lot_per_day=10.0,
            pp_api_url='http://127.0.0.1:8000',
            pp_timeout=5.0,
        )

        def __init__(self):
            super().__init__()
            from app.prediction_client import ApiPredictionSource
            self._pred_source = ApiPredictionSource(
                self.p.pp_api_url, self.p.pp_timeout
            )

            self.data0 = self.datas[0]
            self.initial_balance = self.broker.getvalue()
            self.trade_entry_dates = []
            self.balance_history = []
            self.date_history = []
            self.trade_low = None
            self.trade_high = None
            self.trades = []
            self.current_tp = None
            self.current_sl = None
            self.current_direction = None
            self.order_direction = None
            self.order_entry_price = None
            self.trade_entry_bar = None
            self.current_volume = None

        def next(self):
            dt = self.data0.datetime.datetime(0)
            dt_hour = dt.replace(minute=0, second=0, microsecond=0)
            current_price = self.data0.close[0]

            # Record balance and time for plotting.
            balance = self.broker.getvalue()
            self.balance_history.append(balance)
            self.date_history.append(dt)

            # ── IN POSITION: check TP/SL then ask PP for early exit ──
            if self.position:
                if self.current_direction == 'buy':
                    if self.trade_low is None or current_price < self.trade_low:
                        self.trade_low = current_price
                    # Hard TP hit
                    if current_price >= self.current_tp:
                        self.close()
                        return
                    # Hard SL hit
                    if current_price <= self.current_sl:
                        self.close()
                        return
                elif self.current_direction == 'sell':
                    if self.trade_high is None or current_price > self.trade_high:
                        self.trade_high = current_price
                    # Hard TP hit
                    if current_price <= self.current_tp:
                        self.close()
                        return
                    # Hard SL hit
                    if current_price >= self.current_sl:
                        self.close()
                        return

                # Ask PP: should I close early?
                exit_pred = self._pred_source.get_exit_prediction(
                    dt_hour,
                    direction=self.current_direction,
                    tp_price=self.current_tp,
                    sl_price=self.current_sl,
                )
                if exit_pred["available"]:
                    exit_bin = exit_pred.get("exit_binary", 1)
                    if exit_bin == 0:
                        # PP says TP unlikely → close early
                        self.close()
                return

            # ── NO POSITION: initialise tracking ──
            self.trade_low = current_price
            self.trade_high = current_price

            # Enforce trade frequency.
            recent_trades = [d for d in self.trade_entry_dates if (dt - d).days < 5]
            if len(recent_trades) >= self.p.max_trades_per_5days:
                return

            # Compute TP/SL distances in pips for the entry request
            default_pip_margin = self.p.profit_threshold * self.p.pip_cost
            tp_pips = self.p.tp_multiplier * self.p.profit_threshold
            sl_pips = self.p.sl_multiplier * self.p.profit_threshold

            # Ask PP: should I open a buy or sell?
            entry_pred = self._pred_source.get_entry_prediction(
                dt_hour, tp_pips=tp_pips, sl_pips=sl_pips,
            )
            if not entry_pred["available"]:
                return

            buy_bin = entry_pred.get("buy_entry_binary", 0)
            sell_bin = entry_pred.get("sell_entry_binary", 0)

            # Determine signal
            if buy_bin == 1 and sell_bin == 0:
                signal = 'buy'
            elif sell_bin == 1 and buy_bin == 0:
                signal = 'sell'
            elif buy_bin == 1 and sell_bin == 1:
                signal = 'buy'  # both viable → default to buy
            else:
                return  # both 0 → no action

            # Compute absolute TP/SL price levels
            if signal == 'buy':
                chosen_tp = current_price + self.p.tp_multiplier * default_pip_margin
                chosen_sl = current_price - self.p.sl_multiplier * default_pip_margin
            else:
                chosen_tp = current_price - self.p.tp_multiplier * default_pip_margin
                chosen_sl = current_price + self.p.sl_multiplier * default_pip_margin

            chosen_rr = 1.0  # neutral RR for API mode
            order_size = self._compute_size(chosen_rr)
            if order_size <= 0:
                return

            self.trade_entry_dates.append(dt)
            self.trade_entry_bar = len(self)
            self.current_volume = order_size

            if signal == 'buy':
                self.buy(size=order_size)
                self.current_direction = 'buy'
            else:
                self.sell(size=order_size)
                self.current_direction = 'sell'

            self.current_tp = chosen_tp
            self.current_sl = chosen_sl

        # -----------------------------------------------------------------
        def _compute_size(self, rr):
            min_vol = self.p.min_order_volume
            max_vol = self.p.max_order_volume
            if rr >= self.p.upper_rr_threshold:
                size = max_vol
            elif rr <= self.p.lower_rr_threshold:
                size = min_vol
            else:
                size = min_vol + ((rr - self.p.lower_rr_threshold) /
                                  (self.p.upper_rr_threshold - self.p.lower_rr_threshold)) * (max_vol - min_vol)
            cash = self.broker.getcash()
            max_from_cash = cash * self.p.rel_volume * self.p.leverage
            return min(size, max_from_cash)

        def notify_order(self, order):
            if order.status in [order.Completed]:
                self.order_entry_price = order.executed.price
                self.order_direction = 'buy' if order.isbuy() else 'sell'

        def notify_trade(self, trade):
            if trade.isclosed:
                duration = len(self) - (self.trade_entry_bar if self.trade_entry_bar is not None else 0)
                dt = self.data0.datetime.datetime(0)
                entry_price = self.order_entry_price if self.order_entry_price is not None else 0
                exit_price = trade.price
                profit_usd = trade.pnlcomm
                # Deduct swap/overnight costs
                overnight_days = max(0, duration / 24.0)
                volume = self.current_volume if self.current_volume is not None else 0
                lots = volume / 100000.0
                swap_cost = overnight_days * lots * self.p.swap_per_lot_per_day
                profit_usd -= swap_cost
                direction = self.order_direction
                if direction == 'buy':
                    profit_pips = (exit_price - entry_price) / self.p.pip_cost
                    intra_dd = (entry_price - self.trade_low) / self.p.pip_cost if self.trade_low is not None else 0
                elif direction == 'sell':
                    profit_pips = (entry_price - exit_price) / self.p.pip_cost
                    intra_dd = (self.trade_high - entry_price) / self.p.pip_cost if self.trade_high is not None else 0
                else:
                    profit_pips = 0
                    intra_dd = 0
                current_balance = self.broker.getvalue()
                open_dt = self.trade_entry_dates[-1] if self.trade_entry_dates else "N/A"
                trade_record = {
                    'open_dt': open_dt,
                    'close_dt': dt,
                    'volume': volume,
                    'pnl': profit_usd,
                    'pips': profit_pips,
                    'duration': duration,
                    'max_dd': intra_dd,
                }
                self.trades.append(trade_record)
                if not _QUIET:
                    print(f"[DEBUG]   TRADE CLOSED ({direction}): Date={dt}, "
                          f"Entry={entry_price:.5f}, Exit={exit_price:.5f}, "
                          f"Volume={volume}, PnL={profit_usd:.2f}, "
                          f"Pips={profit_pips:.2f}, Duration={duration} bars, "
                          f"MaxDD={intra_dd:.2f}, Balance={current_balance:.2f}")
                self.order_entry_price = None
                self.current_tp = None
                self.current_sl = None
                self.current_direction = None
                self.current_volume = None

        def stop(self):
            if self.position:
                self.close()
            # Clean up API session
            if self._pred_source is not None:
                self._pred_source.close()
            min_balance = min(self.balance_history) if self.balance_history else 0
            n_trades = len(self.trades)
            if n_trades > 0:
                avg_profit_usd = sum(t['pnl'] for t in self.trades) / n_trades
                avg_profit_pips = sum(t['pips'] for t in self.trades) / n_trades
                avg_duration = sum(t['duration'] for t in self.trades) / n_trades
                avg_max_dd = sum(t['max_dd'] for t in self.trades) / n_trades
            else:
                avg_profit_usd = avg_profit_pips = avg_duration = avg_max_dd = 0
            final_balance = self.broker.getvalue()
            if not _QUIET:
                print("\n==== Summary ====")
                print(f"Initial Balance (USD): {self.initial_balance:.2f}")
                print(f"Final Balance (USD):   {final_balance:.2f}")
                print(f"Minimum Balance (USD): {min_balance:.2f}")
                print(f"Number of Trades: {n_trades}")
                print(f"Average Profit (USD): {avg_profit_usd:.2f}")
                print(f"Average Profit (pips): {avg_profit_pips:.2f}")
                print(f"Average Max Drawdown (pips): {avg_max_dd:.2f}")
                print(f"Average Trade Duration (bars): {avg_duration:.2f}")
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(self.date_history, self.balance_history, label="Balance")
            plt.xlabel("Date")
            plt.ylabel("Balance (USD)")
            plt.title("Balance vs Date (API Predictions)")
            plt.legend()
            plt.savefig("balance_plot.png")
            plt.close()

    # -------------------------------------------------------------------------
    # Dummy methods for interface compatibility
    # -------------------------------------------------------------------------
    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def build_model(self, input_shape):
        if not _QUIET: print("build_model() not applicable for trading strategy plugin.")

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        if not _QUIET: print("train() not applicable for trading strategy plugin.")

    def predict(self, data):
        if not _QUIET: print("predict() not applicable for trading strategy plugin.")
        return None

    def calculate_mse(self, y_true, y_pred):
        if not _QUIET: print("calculate_mse() not applicable for trading strategy plugin.")
        return None

    def calculate_mae(self, y_true, y_pred):
        if not _QUIET: print("calculate_mae() not applicable for trading strategy plugin.")
        return None

    def save(self, file_path):
        if not _QUIET: print("save() not applicable for trading strategy plugin.")

    def load(self, file_path):
        if not _QUIET: print("load() not applicable for trading strategy plugin.")

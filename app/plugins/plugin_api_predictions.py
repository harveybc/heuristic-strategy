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
        'rel_volume': 0.10,
        'min_order_volume': 10000,
        'max_order_volume': 1000000,
        'leverage': 100,
        # DEAP-optimized defaults (worst-case costs, pop=10 gen=10)
        'profit_threshold': 25.50,
        'min_drawdown_pips': 10,
        'tp_multiplier': 5.15,
        'sl_multiplier': 3.66,
        'lower_rr_threshold': 2.67,
        'upper_rr_threshold': 4.46,
        'max_trades_per_5days': 5,
        'exit_variant': 'E',
        # Early-stopping exit predictions (Model B)
        'exit_enabled': True,
        # Trading costs (in pipettes, 1 real pip = 10 pipettes)
        # WORST-CASE scenario — "train on hard"
        'spread_pips': 30.0,        # 3.0 real pips — covers news spikes / illiquid sessions
        'commission_per_lot': 10.0,  # $10 per standard lot (100K) — high-end retail ECN
        'slippage_pips': 10.0,       # 1.0 real pips — adverse execution
        'swap_per_lot_per_day': 15.0, # $15/lot/day — punishes overnight holds
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
            ("profit_threshold", 5, 30),
            ("tp_multiplier", 0.5, 2.0),
            ("sl_multiplier", 2.0, 6.0),
            ("lower_rr_threshold", 0.2, 2.0),
            ("upper_rr_threshold", 2.0, 8.0),
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
            exit_enabled=self.params['exit_enabled'],
            swap_per_lot_per_day=self.params['swap_per_lot_per_day'],
            pp_api_url=config.get("pp_api_url", "http://127.0.0.1:8000"),
            pp_timeout=float(config.get("pp_timeout", 5.0)),
            spread_pips=self.params['spread_pips'],
            commission_per_lot=self.params['commission_per_lot'],
            slippage_pips=self.params['slippage_pips'],
        )

        data_feed = bt.feeds.PandasData(dataname=base_data)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(10000.0)
        # Cheat-on-close: fill market orders at the current bar's close
        # so entry price matches what the oracle sees (close[0]).
        cerebro.broker.set_coc(True)

        # Apply realistic trading costs
        spread_cost = self.params['spread_pips'] * self.params['pip_cost']
        slippage_cost = self.params['slippage_pips'] * self.params['pip_cost']
        total_spread = spread_cost + slippage_cost
        commission_per_unit = self.params['commission_per_lot'] / 100000.0
        # Forex margin: 1/leverage per unit (e.g. 100:1 → $0.01 per unit)
        forex_margin = 1.0 / self.params['leverage']
        cerebro.broker.setcommission(commission=commission_per_unit, margin=forex_margin, mult=1.0)
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
        - When IN position: checks TP/SL using intra-bar high/low
          (matching the oracle's scanning logic). No exit predictions —
          the entry oracle already verified TP will be hit.
        - Uses cheat-on-close so entry price = close[0], matching oracle.
        """

        params = dict(
            pip_cost=0.00001,
            rel_volume=0.10,
            min_order_volume=10000,
            max_order_volume=1000000,
            leverage=100,
            profit_threshold=25.50,
            min_drawdown_pips=10,
            tp_multiplier=5.15,
            sl_multiplier=3.66,
            lower_rr_threshold=2.67,
            upper_rr_threshold=4.46,
            max_trades_per_5days=5,
            exit_variant='E',
            exit_enabled=True,
            swap_per_lot_per_day=15.0,
            pp_api_url='http://127.0.0.1:8000',
            pp_timeout=5.0,
            # Trading costs — sent to PP so oracle can account for them
            spread_pips=30.0,
            commission_per_lot=10.0,
            slippage_pips=10.0,
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

            # ── IN POSITION: check TP/SL ──
            # TP uses CLOSE (matching oracle's close-based scan) so that
            # self.close() with CoC fills at close >= TP, guaranteeing profit.
            # SL uses intra-bar LOW/HIGH to catch worst-case stop-outs.
            if self.position:
                # Force-close before weekend (Friday 20:00) to avoid gap risk
                if dt.weekday() == 4 and dt.hour >= 20:
                    self.close()
                    return

                bar_high = self.data0.high[0]
                bar_low = self.data0.low[0]
                bar_close = self.data0.close[0]
                if self.current_direction == 'buy':
                    if self.trade_low is None or bar_low < self.trade_low:
                        self.trade_low = bar_low
                    if bar_low <= self.current_sl:
                        self.close()
                        return
                    if bar_close >= self.current_tp:
                        self.close()
                        return
                elif self.current_direction == 'sell':
                    if self.trade_high is None or bar_high > self.trade_high:
                        self.trade_high = bar_high
                    if bar_high >= self.current_sl:
                        self.close()
                        return
                    if bar_close <= self.current_tp:
                        self.close()
                        return

                # ── EARLY-STOPPING via exit prediction (Model B) ──
                # Ask PP: "is SL predicted to hit before TP in the short term?"
                # exit_binary=1 → keep open (TP still likely)
                # exit_binary=0 → close early (SL danger detected)
                if self.p.exit_enabled:
                    exit_pred = self._pred_source.get_exit_prediction(
                        dt_hour,
                        direction=self.current_direction,
                        tp_price=self.current_tp,
                        sl_price=self.current_sl,
                    )
                    if exit_pred["available"] and exit_pred.get("exit_binary", 1) == 0:
                        if not _QUIET:
                            print(f"[EARLY_STOP] {dt} — exit model predicts SL danger, closing {self.current_direction}")
                        self.close()
                        return

                return

            # ── NO POSITION: initialise tracking ──
            self.trade_low = current_price
            self.trade_high = current_price

            # No new entries on Friday (oracle horizon is nearly zero)
            if dt.weekday() == 4:
                return

            # Enforce trade frequency.
            recent_trades = [d for d in self.trade_entry_dates if (dt - d).days < 5]
            if len(recent_trades) >= self.p.max_trades_per_5days:
                return

            # Compute TP/SL distances in pips for the entry request
            default_pip_margin = self.p.profit_threshold * self.p.pip_cost
            tp_pips = self.p.tp_multiplier * self.p.profit_threshold
            sl_pips = self.p.sl_multiplier * self.p.profit_threshold

            # Ask PP: should I open a buy or sell?
            # Send trading costs so oracle can make a fully-informed decision
            entry_pred = self._pred_source.get_entry_prediction(
                dt_hour, tp_pips=tp_pips, sl_pips=sl_pips,
                spread_pips=self.p.spread_pips,
                commission_per_lot=self.p.commission_per_lot,
                slippage_pips=self.p.slippage_pips,
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

            # Use bars_remaining from oracle for position sizing
            # Confidence from Bayesian models modulates size (oracle=1.0 → no change)
            bars_remaining = entry_pred.get("bars_remaining", 0)
            confidence = entry_pred.get(
                "buy_confidence" if signal == 'buy' else "sell_confidence", 1.0
            )
            order_size = self._compute_size(bars_remaining, confidence)
            if order_size <= 0:
                return

            self.trade_entry_dates.append(dt)
            self.trade_entry_bar = len(self)
            self.current_volume = order_size
            self.current_tp = chosen_tp
            self.current_sl = chosen_sl

            if signal == 'buy':
                self.buy(size=order_size)
                self.current_direction = 'buy'
            else:
                self.sell(size=order_size)
                self.current_direction = 'sell'

        # -----------------------------------------------------------------
        def _compute_size(self, bars_remaining, confidence=1.0):
            """Position sizing based on bars remaining and model confidence.

            Fewer bars remaining means the oracle expects TP to be hit
            quickly → higher confidence → larger position.

            bars_remaining ~ 1-10  → near max_order_volume  (high confidence)
            bars_remaining ~ 100+  → near min_order_volume  (low confidence)

            The upper/lower RR thresholds are repurposed as bar-count
            boundaries:
              - <= lower_rr_threshold * 24  → max size
              - >= upper_rr_threshold * 24  → min size
            Linear interpolation between those boundaries.

            Bayesian models supply confidence in [0, 1] to further modulate
            size.  Oracle always sends 1.0 (no effect).
            """
            min_vol = self.p.min_order_volume
            max_vol = self.p.max_order_volume
            # Convert RR thresholds to bar-count boundaries (hours in a day)
            fast_bars = self.p.lower_rr_threshold * 24   # e.g. 0.5 * 24 = 12
            slow_bars = self.p.upper_rr_threshold * 24   # e.g. 2.0 * 24 = 48

            if bars_remaining <= fast_bars:
                size = max_vol
            elif bars_remaining >= slow_bars:
                size = min_vol
            else:
                # Linear: fewer bars → bigger size
                frac = (bars_remaining - fast_bars) / (slow_bars - fast_bars)
                size = max_vol - frac * (max_vol - min_vol)

            cash = self.broker.getcash()
            max_from_cash = cash * self.p.rel_volume * self.p.leverage
            size = min(size, max_from_cash)

            # Modulate by model confidence (oracle=1.0 → no change)
            size *= max(0.0, min(1.0, confidence))
            return size

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
                direction = self.current_direction
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

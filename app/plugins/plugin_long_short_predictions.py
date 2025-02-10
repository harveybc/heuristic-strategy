import datetime
import os
import backtrader as bt
import pandas as pd


class Plugin:
    """
    Plugin for Heuristic Trading Strategy.

    - Embeds a HeuristicStrategy that replicates your original strategy exactly.
    - Exposes plugin_params with default values and the required methods for optimization.
    """

    # Default plugin parameters (must be present for optimizer integration)
    plugin_params = {
        'price_file': '../trading-signal/output.csv',
        'pred_file': '../trading-signal/output.csv',
        'date_start': datetime.datetime(2010, 1, 1),
        'date_end': datetime.datetime(2015, 1, 1),
        'pip_cost': 0.00001,
        'rel_volume': 0.05,
        'min_order_volume': 10000,
        'max_order_volume': 1000000,
        'leverage': 1000,
        'profit_threshold': 5,
        'min_drawdown_pips': 10,
        'tp_multiplier': 0.9,
        'sl_multiplier': 2.0,
        'lower_rr_threshold': 0.5,
        'upper_rr_threshold': 2.0,
        'max_trades_per_5days': 3,
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

    # -----------------------------------------------------------------------------
    # Required optimization interface
    # -----------------------------------------------------------------------------
    def get_optimizable_params(self):
        """Return parameters that can be optimized along with their bounds."""
        return [
            ("profit_threshold", 1, 20),
            ("tp_multiplier", 0.8, 1.2),
            ("sl_multiplier", 1.5, 3.0),
            ("rel_volume", 0.01, 0.1),
            ("lower_rr_threshold", 0.3, 1.0),
            ("upper_rr_threshold", 1.5, 3.0),
        ]

    def evaluate_candidate(self, individual, base_data, hourly_predictions, daily_predictions, config):
        """
        Evaluates a candidate strategy parameter set by merging the provided predictions (hourly and daily)
        into one DataFrame (saved as CSV for the strategy to read) and running a backtest.
        If config["show_trades"] is True, prints the individual trades with all details.
        Returns a tuple (profit, stats) where stats include trade count, etc.
        """
        import os
        import pandas as pd
        import backtrader as bt

        # Unpack candidate parameters.
        profit_threshold, tp_multiplier, sl_multiplier, rel_volume, lower_rr, upper_rr = individual

        # Merge predictions from hourly and daily data (if provided).
        merged_df = pd.DataFrame()
        if hourly_predictions is not None and not hourly_predictions.empty:
            renamed_h = {col: f"Prediction_h_{i+1}" for i, col in enumerate(hourly_predictions.columns)}
            hr = hourly_predictions.rename(columns=renamed_h)
            merged_df = hr.copy() if merged_df.empty else merged_df.join(hr, how="outer")
        if daily_predictions is not None and not daily_predictions.empty:
            renamed_d = {col: f"Prediction_d_{i+1}" for i, col in enumerate(daily_predictions.columns)}
            dr = daily_predictions.rename(columns=renamed_d)
            merged_df = dr.copy() if merged_df.empty else merged_df.join(dr, how="outer")

        if merged_df.empty:
            print(f"[evaluate_candidate] => Merged predictions are empty for candidate {individual}. Returning profit=0.0.")
            return (0.0, {"num_trades": 0, "win_pct": 0, "max_dd": 0, "sharpe": 0})

        if merged_df.index.name is None or merged_df.index.name != "DATE_TIME":
            merged_df = merged_df.copy()
            merged_df.index.name = "DATE_TIME"

        temp_pred_file = "temp_predictions.csv"
        merged_df.reset_index().to_csv(temp_pred_file, index=False)

        # Determine date range.
        if hasattr(base_data.index, 'min') and hasattr(base_data.index, 'max'):
            date_start = base_data.index.min().to_pydatetime()
            date_end   = base_data.index.max().to_pydatetime()
        else:
            date_start = self.params['date_start']
            date_end   = self.params['date_end']

        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            self.HeuristicStrategy,
            pred_file=temp_pred_file,
            pip_cost=self.params['pip_cost'],
            rel_volume=rel_volume,
            min_order_volume=self.params['min_order_volume'],
            max_order_volume=self.params['max_order_volume'],
            leverage=self.params['leverage'],
            profit_threshold=profit_threshold,
            date_start=date_start,
            date_end=date_end,
            min_drawdown_pips=self.params['min_drawdown_pips'],
            tp_multiplier=tp_multiplier,
            sl_multiplier=sl_multiplier,
            lower_rr_threshold=lower_rr,
            upper_rr_threshold=upper_rr,
            max_trades_per_5days=self.params['max_trades_per_5days']
        )
        data_feed = bt.feeds.PandasData(dataname=base_data)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(10000.0)

        try:
            runresult = cerebro.run()
        except Exception as e:
            print("Error during backtest:", e)
            if os.path.exists(temp_pred_file):
                os.remove(temp_pred_file)
            return (-1e6, {"num_trades": 0, "win_pct": 0, "max_dd": 0, "sharpe": 0})

        final_value = cerebro.broker.getvalue()
        profit = final_value - 10000.0
        print(f"Evaluated candidate {individual} -> Profit: {profit:.2f}")

        if config.get("show_trades", True):
            strat_instance = runresult[0]
            trades_list = getattr(strat_instance, "trades", [])
            if trades_list:
                print(f"Trades for candidate {individual}:")
                for i, tr in enumerate(trades_list, 1):
                    open_dt = tr.get('open_dt', 'N/A')
                    close_dt = tr.get('close_dt', 'N/A')
                    volume = tr.get('volume', 0)
                    pnl = tr.get('pnl', 0)
                    pips = tr.get('pips', 0)
                    max_dd = tr.get('max_dd', 0)
                    print(f"  Trade #{i}: OpenDT={open_dt}, ExitDT={close_dt}, Volume={volume}, "
                          f"PnL={pnl:.2f}, Pips={pips:.2f}, MaxDD={max_dd:.2f}")
            else:
                print("  No trades were made for this candidate.")

        if os.path.exists(temp_pred_file):
            os.remove(temp_pred_file)

        # For demonstration, also compute basic stats from trades.
        num_trades = len(getattr(strat_instance, "trades", []))
        stats = {
            "num_trades": num_trades,
            "win_pct": 0,
            "max_dd": 0,
            "sharpe": 0
        }
        if num_trades > 0:
            wins = [1 for tr in strat_instance.trades if tr['pnl'] > 0]
            win_pct = (sum(wins) / num_trades) * 100
            max_dd = max(tr['max_dd'] for tr in strat_instance.trades)
            # Dummy sharpe ratio calculation: profit / std (if std > 0)
            profits = [tr['pnl'] for tr in strat_instance.trades]
            std_profit = (sum((p - (sum(profits)/num_trades))**2 for p in profits)/num_trades)**0.5 if num_trades > 1 else 0
            sharpe = (profit / std_profit) if std_profit > 0 else 0
            stats.update({"win_pct": win_pct, "max_dd": max_dd, "sharpe": sharpe})

        print(f"[EVALUATE] Candidate result => Profit: {profit:.2f}, "
              f"Trades: {stats.get('num_trades', 0)}, "
              f"Win%: {stats.get('win_pct', 0):.1f}, "
              f"MaxDD: {stats.get('max_dd', 0):.2f}, "
              f"Sharpe: {stats.get('sharpe', 0):.2f}")

        return (profit, stats)
    # -----------------------------------------------------------------------------
    # Embedded Heuristic Strategy
    # -----------------------------------------------------------------------------
    class HeuristicStrategy(bt.Strategy):
        """
        Forex Dynamic Volume Strategy using perfect future predictions.

        This replicates the original HeuristicStrategy exactly, with all printed messages
        and the same logic for trade entries, sizing, frequency, and final summary.
        """

        def __init__(self, pred_file, pip_cost, rel_volume, min_order_volume, max_order_volume,
                     leverage, profit_threshold, date_start, date_end, min_drawdown_pips,
                     tp_multiplier, sl_multiplier, lower_rr_threshold, upper_rr_threshold,
                     max_trades_per_5days, *args, **kwargs):
            super().__init__()
            # Store parameters in self.params for convenience
            self.params.pred_file = pred_file
            self.params.pip_cost = pip_cost
            self.params.rel_volume = rel_volume
            self.params.min_order_volume = min_order_volume
            self.params.max_order_volume = max_order_volume
            self.params.leverage = leverage
            self.params.profit_threshold = profit_threshold
            self.params.date_start = date_start
            self.params.date_end = date_end
            self.params.min_drawdown_pips = min_drawdown_pips
            self.params.tp_multiplier = tp_multiplier
            self.params.sl_multiplier = sl_multiplier
            self.params.lower_rr_threshold = lower_rr_threshold
            self.params.upper_rr_threshold = upper_rr_threshold
            self.params.max_trades_per_5days = max_trades_per_5days

            # Load the CSV with predictions
            pred_df = pd.read_csv(self.params.pred_file, parse_dates=['DATE_TIME'])
            pred_df = pred_df[
                (pred_df['DATE_TIME'] >= self.params.date_start) & 
                (pred_df['DATE_TIME'] <= self.params.date_end)
            ]
            pred_df['DATE_TIME'] = pred_df['DATE_TIME'].apply(
                lambda dt: dt.replace(minute=0, second=0, microsecond=0)
            )
            pred_df.set_index('DATE_TIME', inplace=True)

            # Track how many prediction columns exist
            self.num_hourly_preds = len([c for c in pred_df.columns if c.startswith('Prediction_h_')])
            self.num_daily_preds = len([c for c in pred_df.columns if c.startswith('Prediction_d_')])
            self.pred_df = pred_df

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
            self.trade_entry_bar = None

        def next(self):
            dt = self.data0.datetime.datetime(0)
            dt_hour = dt.replace(minute=0, second=0, microsecond=0)
            current_price = self.data0.close[0]

            # Record current balance and date for plotting.
            self.balance_history.append(self.broker.getvalue())
            self.date_history.append(dt)

            # --- If in position, handle exit logic ---
            if self.position:
                if self.current_direction == 'long':
                    # Update intra‐trade low
                    if self.trade_low is None or current_price < self.trade_low:
                        self.trade_low = current_price
                    # If we have predictions for dt_hour, check them
                    if dt_hour in self.pred_df.index:
                        preds_hourly = [
                            self.pred_df.loc[dt_hour].get(f'Prediction_h_{i}', current_price)
                            for i in range(1, self.num_hourly_preds + 1)
                        ]
                        preds_daily = [
                            self.pred_df.loc[dt_hour].get(f'Prediction_d_{i}', current_price)
                            for i in range(1, self.num_daily_preds + 1)
                        ]
                        predicted_min = min(preds_hourly + preds_daily)
                    else:
                        predicted_min = current_price
                    # Exit conditions: if current_price >= TP or predicted_min < SL
                    if current_price >= self.current_tp or predicted_min < self.current_sl:
                        self.close()
                        return

                elif self.current_direction == 'short':
                    # Update intra‐trade high
                    if self.trade_high is None or current_price > self.trade_high:
                        self.trade_high = current_price
                    # If we have predictions for dt_hour, check them
                    if dt_hour in self.pred_df.index:
                        preds_hourly = [
                            self.pred_df.loc[dt_hour].get(f'Prediction_h_{i}', current_price)
                            for i in range(1, self.num_hourly_preds + 1)
                        ]
                        preds_daily = [
                            self.pred_df.loc[dt_hour].get(f'Prediction_d_{i}', current_price)
                            for i in range(1, self.num_daily_preds + 1)
                        ]
                        predicted_max = max(preds_hourly + preds_daily)
                    else:
                        predicted_max = current_price
                    # Exit conditions: if current_price <= TP or predicted_max > SL
                    if current_price <= self.current_tp or predicted_max > self.current_sl:
                        self.close()
                        return

                # If we handled exit logic, do not attempt new entries
                return

            # --- If not in position, handle entry logic ---

            # Reset intra‐trade extremes
            self.trade_low = current_price
            self.trade_high = current_price

            # Enforce trade frequency: no more than max_trades_per_5days in last 5 days
            recent_trades = [d for d in self.trade_entry_dates if (dt - d).days < 5]
            if len(recent_trades) >= self.p.max_trades_per_5days:
                return

            # Check if we have predictions for dt_hour
            if dt_hour not in self.pred_df.index:
                return
            row = self.pred_df.loc[dt_hour]

            # Attempt to load daily predictions
            try:
                daily_preds = [row[f'Prediction_d_{i}'] for i in range(1, self.num_daily_preds + 1)]
            except KeyError:
                return

            # If daily_preds is empty or all NaN => skip
            if not daily_preds or all(pd.isna(daily_preds)):
                return

            # --- Long (Buy) Calculations ---
            ideal_profit_pips_buy = (max(daily_preds) - current_price) / self.p.pip_cost
            ideal_drawdown_pips_buy = max(
                (current_price - min(daily_preds)) / self.p.pip_cost,
                self.p.min_drawdown_pips
            )
            rr_buy = ideal_profit_pips_buy / ideal_drawdown_pips_buy if ideal_drawdown_pips_buy > 0 else 0
            tp_buy = current_price + self.p.tp_multiplier * ideal_profit_pips_buy * self.p.pip_cost
            sl_buy = current_price - self.p.sl_multiplier * ideal_drawdown_pips_buy * self.p.pip_cost

            # --- Short (Sell) Calculations ---
            ideal_profit_pips_sell = (current_price - min(daily_preds)) / self.p.pip_cost
            ideal_drawdown_pips_sell = max(
                (max(daily_preds) - current_price) / self.p.pip_cost,
                self.p.min_drawdown_pips
            )
            rr_sell = ideal_profit_pips_sell / ideal_drawdown_pips_sell if ideal_drawdown_pips_sell > 0 else 0
            tp_sell = current_price - self.p.tp_multiplier * ideal_profit_pips_sell * self.p.pip_cost
            sl_sell = current_price + self.p.sl_multiplier * ideal_drawdown_pips_sell * self.p.pip_cost

            # Determine signals
            long_signal = (ideal_profit_pips_buy >= self.p.profit_threshold)
            short_signal = (ideal_profit_pips_sell >= self.p.profit_threshold)

            # Choose the signal with higher RR
            if long_signal and (rr_buy >= rr_sell):
                signal = 'long'
                chosen_tp = tp_buy
                chosen_sl = sl_buy
                chosen_rr = rr_buy
            elif short_signal and (rr_sell > rr_buy):
                signal = 'short'
                chosen_tp = tp_sell
                chosen_sl = sl_sell
                chosen_rr = rr_sell
            else:
                signal = None

            if signal is None:
                return

            # Compute order size
            order_size = self.compute_size(chosen_rr)
            if order_size <= 0:
                return

            # Record trade entry
            self.trade_entry_dates.append(dt)
            self.trade_entry_bar = len(self)

            # Place order
            if signal == 'long':
                self.buy(size=order_size)
                self.current_direction = 'long'
            elif signal == 'short':
                self.sell(size=order_size)
                self.current_direction = 'short'

            # Store chosen TP and SL
            self.current_tp = chosen_tp
            self.current_sl = chosen_sl

        def compute_size(self, rr):
            """Compute order size by linear interpolation between min and max volumes based on RR."""
            min_vol = self.params.min_order_volume
            max_vol = self.params.max_order_volume
            if rr >= self.params.upper_rr_threshold:
                size = max_vol
            elif rr <= self.params.lower_rr_threshold:
                size = min_vol
            else:
                size = min_vol + ((rr - self.params.lower_rr_threshold) /
                                  (self.params.upper_rr_threshold - self.params.lower_rr_threshold)) * (max_vol - min_vol)

            cash = self.broker.getcash()
            max_from_cash = cash * self.params.rel_volume * self.params.leverage
            return min(size, max_from_cash)

        def notify_order(self, order):
            """
            When an order is completed, record the execution price
            and capture its direction, just like in the original strategy.
            """
            if order.status in [order.Completed]:
                self.order_entry_price = order.executed.price
                # Set the direction based on whether it's a buy or sell
                self.order_direction = 'long' if order.isbuy() else 'short'


        def notify_trade(self, trade):
            """
            When a trade closes, record its details and print a summary.
            This method appends each trade’s information—including open and close datetimes,
            entry and exit prices, profit in USD and pips, duration, maximum drawdown, and the trade direction—
            to the strategy’s trades list.
            """
            if trade.isclosed:
                # Calculate trade duration in bars.
                duration = len(self) - (self.trade_entry_bar if self.trade_entry_bar is not None else 0)
                # Capture the close datetime from the current bar.
                close_dt = self.data0.datetime.datetime(0)
                # Retrieve the entry price (or default to the current price).
                entry_price = self.order_entry_price if self.order_entry_price is not None else self.data0.close[0]
                exit_price = trade.price
                profit_usd = trade.pnlcomm
                direction = self.order_direction
                if direction == 'long':
                    profit_pips = (exit_price - entry_price) / self.p.pip_cost
                    intra_dd = (entry_price - self.trade_low) / self.p.pip_cost if self.trade_low is not None else 0
                elif direction == 'short':
                    profit_pips = (entry_price - exit_price) / self.p.pip_cost
                    intra_dd = (self.trade_high - entry_price) / self.p.pip_cost if self.trade_high is not None else 0
                else:
                    profit_pips = 0
                    intra_dd = 0
                current_balance = self.broker.getvalue()
                # Use the last recorded trade entry datetime as the open time.
                open_dt = self.trade_entry_dates[-1] if self.trade_entry_dates else "N/A"
                trade_record = {
                    'open_dt': open_dt,
                    'close_dt': close_dt,
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl': profit_usd,
                    'pips': profit_pips,
                    'duration': duration,
                    'max_dd': intra_dd,
                    'direction': direction
                }
                self.trades.append(trade_record)
                # Immediately print trade details if desired.
                print(f"TRADE CLOSED ({direction}): OpenDT={open_dt}, CloseDT={close_dt}, "
                    f"Entry={entry_price:.5f}, Exit={exit_price:.5f}, "
                    f"PnL={profit_usd:.2f}, Pips={profit_pips:.2f}, "
                    f"Duration={duration} bars, MaxDD={intra_dd:.2f}, Balance={current_balance:.2f}")
                # Reset order-related variables.
                self.order_entry_price = None
                self.current_tp = None
                self.current_sl = None
                self.current_direction = None

        def stop(self):
            """
            At the end of the simulation, force-close any open position so that all trades are recorded.
            Then compute and print summary statistics (including number of trades, average profit in USD and pips,
            maximum drawdown, and average trade duration), and finally save the balance plot.
            """
            # Force-close any open position.
            if self.position:
                self.close()
            # (After a forced close, notify_trade() should be called and record the final trade.)
            
            # Compute summary statistics.
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

            print("\n==== Summary ====")
            print(f"Initial Balance (USD): {self.initial_balance:.2f}")
            print(f"Final Balance (USD):   {final_balance:.2f}")
            print(f"Minimum Balance (USD): {min_balance:.2f}")
            print(f"Number of Trades: {n_trades}")
            print(f"Average Profit (USD): {avg_profit_usd:.2f}")
            print(f"Average Profit (pips): {avg_profit_pips:.2f}")
            print(f"Average Max Drawdown (pips): {avg_max_dd:.2f}")
            print(f"Average Trade Duration (bars): {avg_duration:.2f}")

            # Save the balance plot.
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(self.date_history, self.balance_history, label="Balance")
            plt.xlabel("Date")
            plt.ylabel("Balance (USD)")
            plt.title("Balance vs Date")
            plt.legend()
            plt.savefig("balance_plot.png")
            plt.close()



    # -------------------------------------------------------------------------
    # Dummy methods for model building/training (to maintain interface compatibility)
    # -------------------------------------------------------------------------
    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def build_model(self, input_shape):
        print("build_model() not applicable for trading strategy plugin.")

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        print("train() not applicable for trading strategy plugin.")

    def predict(self, data):
        print("predict() not applicable for trading strategy plugin.")
        return None

    def calculate_mse(self, y_true, y_pred):
        print("calculate_mse() not applicable for trading strategy plugin.")
        return None

    def calculate_mae(self, y_true, y_pred):
        print("calculate_mae() not applicable for trading strategy plugin.")
        return None

    def save(self, file_path):
        print("save() not applicable for trading strategy plugin.")

    def load(self, file_path):
        print("load() not applicable for trading strategy plugin.")

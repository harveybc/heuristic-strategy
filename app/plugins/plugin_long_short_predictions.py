import datetime
import os
import backtrader as bt
import pandas as pd
import numpy as np

class Plugin:
    """
    Plugin for Heuristic Trading Strategy.

    - Embeds a HeuristicStrategy that replicates your original strategy exactly.
    - Exposes plugin_params with default values and the required methods for optimization.
    """

    # Default plugin parameters (must be present for optimizer integration)
    plugin_params = {
        'pip_cost': 0.00001,
        'rel_volume': 0.02, # uses max 2% of balance for each order (default) 
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
            ("profit_threshold", 0.5, 20),
            ("tp_multiplier", 0.5, 1.5),
            ("sl_multiplier", 1.5, 6.0),
            #("rel_volume", 0.01, 0.1),
            ("lower_rr_threshold", 0.2, 1.0),
            ("upper_rr_threshold", 1.3, 6.0),
            # size of the time horizon
            ("time_horizon", 2, 24)

        ]

    def evaluate_candidate(self, individual, base_data, hourly_predictions, daily_predictions, config):
        """
        Evaluates a candidate strategy parameter set using the provided datasets.
        Supports both external prediction files and auto-generated predictions.
        """
        import os
        import pandas as pd
        import backtrader as bt

        # Unpack candidate parameters:
        # Now the candidate tuple is:
        # (profit_threshold, tp_multiplier, sl_multiplier, lower_rr_threshold, upper_rr_threshold, time_horizon)
        profit_threshold, tp_multiplier, sl_multiplier, lower_rr, upper_rr, time_horizon = individual

        # If both predictions are missing or empty, auto-generate predictions using the candidate's time_horizon.

        if (config['hourly_predictions_file'] is None) and (config['daily_predictions_file'] is None):
            print(f"[evaluate_candidate] Auto-generating predictions using time_horizon={int(time_horizon)} for candidate {individual}.")
            config["time_horizon"] = int(time_horizon)
            from data_processor import process_data
            processed = process_data(config)
            hourly_predictions = processed["hourly"]
            daily_predictions = processed["daily"]

        # Use provided predictions without modifying them.
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

        # Ensure predictions have a datetime index.
        if merged_df.index.name is None or merged_df.index.name != "DATE_TIME":
            merged_df.index.name = "DATE_TIME"

        # Save merged predictions to a temporary CSV file.
        temp_pred_file = "temp_predictions.csv"
        merged_df.reset_index().to_csv(temp_pred_file, index=False)

        # Build the Cerebro backtest.
        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            self.HeuristicStrategy,
            pred_file=temp_pred_file,
            pip_cost=self.params['pip_cost'],
            rel_volume=self.params['rel_volume'],  # use plugin default value
            min_order_volume=self.params['min_order_volume'],
            max_order_volume=self.params['max_order_volume'],
            leverage=self.params['leverage'],
            profit_threshold=profit_threshold,
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

        # Run the backtest.
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

        # Retrieve trades from the strategy instance.
        strat_instance = runresult[0]
        trades_list = getattr(strat_instance, "trades", [])
        if config.get("show_trades", True):
            if trades_list:
                print(f"Trades for candidate {individual}:")
                for i, tr in enumerate(trades_list, 1):
                    print(f"  Trade #{i}: OpenDT={tr.get('open_dt', 'N/A')}, ExitDT={tr.get('close_dt', 'N/A')}, "
                        f"Volume={tr.get('volume', 0)}, PnL={tr.get('pnl', 0):.2f}, "
                        f"Pips={tr.get('pips', 0):.2f}, MaxDD={tr.get('max_dd', 0):.2f}")
            else:
                print("  No trades were made for this candidate.")

        if os.path.exists(temp_pred_file):
            os.remove(temp_pred_file)

        # Update plugin trades with those from this evaluation.
        self.trades = trades_list

        # Compute summary statistics.
        num_trades = len(trades_list)
        stats = {"num_trades": num_trades, "win_pct": 0, "max_dd": 0, "sharpe": 0}
        if num_trades > 0:
            wins = sum(1 for tr in trades_list if tr['pnl'] > 0)
            win_pct = (wins / num_trades) * 100
            max_dd = max(tr['max_dd'] for tr in trades_list)
            profits = [tr['pnl'] for tr in trades_list]
            avg_profit = sum(profits) / num_trades
            std_profit = np.std(profits) if num_trades > 1 else 0
            sharpe = (profit / std_profit) if std_profit > 0 else 0
            stats.update({"win_pct": win_pct, "max_dd": max_dd, "sharpe": sharpe})

        print(f"[EVALUATE] Candidate result => Profit: {profit:.2f}, "
            f"Trades: {stats.get('num_trades', 0)}, "
            f"Win%: {stats.get('win_pct', 0):.1f}, "
            f"MaxDD: {stats.get('max_dd', 0):.2f}, "
            f"Sharpe: {stats.get('sharpe', 0):.2f}")

        return (profit, stats)



    class HeuristicStrategy(bt.Strategy):
        """
        Forex Dynamic Volume Strategy using perfect future predictions.

        This replicates the original HeuristicStrategy exactly, with all printed messages
        and the same logic for trade entries, sizing, frequency, and final summary.
        """
        def __init__(self, pred_file, pip_cost, rel_volume, min_order_volume, max_order_volume,
                    leverage, profit_threshold, min_drawdown_pips,
                    tp_multiplier, sl_multiplier, lower_rr_threshold, upper_rr_threshold,
                    max_trades_per_5days, *args, **kwargs):
            super().__init__()
            self.params.pred_file = pred_file
            self.params.pip_cost = pip_cost
            self.params.rel_volume = rel_volume
            self.params.min_order_volume = min_order_volume
            self.params.max_order_volume = max_order_volume
            self.params.leverage = leverage
            self.params.profit_threshold = profit_threshold
            self.params.min_drawdown_pips = min_drawdown_pips
            self.params.tp_multiplier = tp_multiplier
            self.params.sl_multiplier = sl_multiplier
            self.params.lower_rr_threshold = lower_rr_threshold
            self.params.upper_rr_threshold = upper_rr_threshold
            self.params.max_trades_per_5days = max_trades_per_5days

            # Load predictions from CSV.
            pred_df = pd.read_csv(self.params.pred_file, parse_dates=['DATE_TIME'])
            pred_df.set_index('DATE_TIME', inplace=True)
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
            #print(f"[DEBUG]   next() called at {dt} (dt_hour: {dt_hour}), current_price: {current_price:.5f}")
            
            # Record balance and time for plotting.
            balance = self.broker.getvalue()
            self.balance_history.append(balance)
            self.date_history.append(dt)
            #print(f"[DEBUG]   Recorded balance: {balance:.2f}")

            # --- If in position, handle exit logic ---
            if self.position:
                #print(f"[DEBUG]   In position: current_direction={self.current_direction}")
                if self.current_direction == 'long':
                    if self.trade_low is None or current_price < self.trade_low:
                        self.trade_low = current_price
                        #print(f"[DEBUG]   (Long) Updated trade_low to: {self.trade_low:.5f}")
                    if dt_hour in self.pred_df.index:
                        preds_hourly = [self.pred_df.loc[dt_hour].get(f'Prediction_h_{i}', current_price)
                                        for i in range(1, self.num_hourly_preds + 1)]
                        preds_daily = [self.pred_df.loc[dt_hour].get(f'Prediction_d_{i}', current_price)
                                       for i in range(1, self.num_daily_preds + 1)]
                        predicted_min = min(preds_hourly + preds_daily)
                        #print(f"[DEBUG]   (Long) Predicted_min from hourly: {preds_hourly}, daily: {preds_daily} => {predicted_min:.5f}")
                    else:
                        predicted_min = current_price
                        #print(f"[DEBUG]   (Long) dt_hour {dt_hour} not in prediction index")
                    # Condition 1: current price reaches TP.
                    if current_price >= self.current_tp:
                        #print(f"[DEBUG]   (Long) Exit condition 1 met: current_price {current_price:.5f} >= TP {self.current_tp:.5f}")
                        self.close()
                        return
                    # Condition 2: predicted_min below SL.
                    if predicted_min < self.current_sl:
                        #print(f"[DEBUG]   (Long) Exit condition 2 met: predicted_min {predicted_min:.5f} < SL {self.current_sl:.5f}")
                        self.close()
                        return
                elif self.current_direction == 'short':
                    if self.trade_high is None or current_price > self.trade_high:
                        self.trade_high = current_price
                        #print(f"[DEBUG]   (Short) Updated trade_high to: {self.trade_high:.5f}")
                    if dt_hour in self.pred_df.index:
                        preds_hourly = [self.pred_df.loc[dt_hour].get(f'Prediction_h_{i}', current_price)
                                        for i in range(1, self.num_hourly_preds + 1)]
                        preds_daily = [self.pred_df.loc[dt_hour].get(f'Prediction_d_{i}', current_price)
                                       for i in range(1, self.num_daily_preds + 1)]
                        predicted_max = max(preds_hourly + preds_daily)
                        #print(f"[DEBUG]   (Short) Predicted_max from hourly: {preds_hourly}, daily: {preds_daily} => {predicted_max:.5f}")
                    else:
                        predicted_max = current_price
                        #print(f"[DEBUG]   (Short) dt_hour {dt_hour} not in prediction index")
                    # Condition 1: current price reaches TP.
                    if current_price <= self.current_tp:
                        #print(f"[DEBUG]   (Short) Exit condition 1 met: current_price {current_price:.5f} <= TP {self.current_tp:.5f}")
                        self.close()
                        return
                    # Condition 2: predicted_max above SL.
                    if predicted_max > self.current_sl:
                        #print(f"[DEBUG]   (Short) Exit condition 2 met: predicted_max {predicted_max:.5f} > SL {self.current_sl:.5f}")
                        self.close()
                        return
                return  # Do not attempt new entries if still in a position.
            else:
                # Not in position: reset trade extremes.
                self.trade_low = current_price
                self.trade_high = current_price
                #print(f"[DEBUG]   Not in position: Reset trade_low and trade_high to {current_price:.5f}")

            # Enforce trade frequency.
            recent_trades = [d for d in self.trade_entry_dates if (dt - d).days < 5]
            if len(recent_trades) >= self.p.max_trades_per_5days:
                #print(f"[DEBUG]   Trade frequency limit reached: {len(recent_trades)} trades in last 5 days")
                return

            if dt_hour not in self.pred_df.index:
                #print(f"[DEBUG]   No prediction data for dt_hour {dt_hour}")
                return
            row = self.pred_df.loc[dt_hour]
            try:
                daily_preds = [row[f'Prediction_d_{i}'] for i in range(1, self.num_daily_preds + 1)]
            except KeyError:
                #print(f"[DEBUG]   Daily prediction keys not found at dt_hour {dt_hour}")
                return
            if not daily_preds or all(pd.isna(daily_preds)):
                #print(f"[DEBUG]   Daily predictions at {dt_hour} are empty or NaN")
                return

            # --- Compute entry conditions for long ---
            ideal_profit_pips_buy = (max(daily_preds) - current_price) / self.p.pip_cost
            ideal_drawdown_pips_buy = max((current_price - min(daily_preds)) / self.p.pip_cost,
                                          self.p.min_drawdown_pips)
            rr_buy = ideal_profit_pips_buy / ideal_drawdown_pips_buy if ideal_drawdown_pips_buy > 0 else 0
            tp_buy = current_price + self.p.tp_multiplier * ideal_profit_pips_buy * self.p.pip_cost
            sl_buy = current_price - self.p.sl_multiplier * ideal_drawdown_pips_buy * self.p.pip_cost

            # --- Compute entry conditions for short ---
            ideal_profit_pips_sell = (current_price - min(daily_preds)) / self.p.pip_cost
            ideal_drawdown_pips_sell = max((max(daily_preds) - current_price) / self.p.pip_cost,
                                           self.p.min_drawdown_pips)
            rr_sell = ideal_profit_pips_sell / ideal_drawdown_pips_sell if ideal_drawdown_pips_sell > 0 else 0
            tp_sell = current_price - self.p.tp_multiplier * ideal_profit_pips_sell * self.p.pip_cost
            sl_sell = current_price + self.p.sl_multiplier * ideal_drawdown_pips_sell * self.p.pip_cost

            #print(f"[DEBUG]   Entry calculations at {dt}:")
            #print(f"        current_price: {current_price:.5f}")
            #print(f"        Daily predictions: {daily_preds}")
            #print(f"        Long -> ideal_profit_pips: {ideal_profit_pips_buy:.2f}, ideal_drawdown: {ideal_drawdown_pips_buy:.2f}, RR: {rr_buy:.2f}, TP: {tp_buy:.5f}, SL: {sl_buy:.5f}")
            #print(f"        Short -> ideal_profit_pips: {ideal_profit_pips_sell:.2f}, ideal_drawdown: {ideal_drawdown_pips_sell:.2f}, RR: {rr_sell:.2f}, TP: {tp_sell:.5f}, SL: {sl_sell:.5f}")

            long_signal = (ideal_profit_pips_buy >= self.p.profit_threshold)
            short_signal = (ideal_profit_pips_sell >= self.p.profit_threshold)

            if long_signal and (rr_buy >= rr_sell):
                signal = 'long'
                chosen_tp = tp_buy
                chosen_sl = sl_buy
                chosen_rr = rr_buy
                #print(f"[DEBUG]   Long signal triggered")
            elif short_signal and (rr_sell > rr_buy):
                signal = 'short'
                chosen_tp = tp_sell
                chosen_sl = sl_sell
                chosen_rr = rr_sell
                #print(f"[DEBUG]   Short signal triggered")
            else:
                #print(f"[DEBUG]   No entry signal triggered")
                return

            order_size = self.compute_size(chosen_rr)
            #print(f"[DEBUG]   Computed order size: {order_size:.2f}")
            if order_size <= 0:
                print("[DEBUG] Order size <= 0, skipping trade")
                return

            self.trade_entry_dates.append(dt)
            self.trade_entry_bar = len(self)
            self.current_volume = order_size
            #print(f"[DEBUG]   Placing {signal} order at {current_price:.5f} with volume {order_size:.2f}")

            if signal == 'long':
                self.buy(size=order_size)
                self.current_direction = 'long'
            elif signal == 'short':
                self.sell(size=order_size)
                self.current_direction = 'short'

            self.current_tp = chosen_tp
            self.current_sl = chosen_sl
            #print(f"[DEBUG]   Set TP: {self.current_tp:.5f}, SL: {self.current_sl:.5f}")

        def compute_size(self, rr):
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
            if order.status in [order.Completed]:
                self.order_entry_price = order.executed.price
                self.order_direction = 'long' if order.isbuy() else 'short'

        def notify_trade(self, trade):
            if trade.isclosed:
                duration = len(self) - (self.trade_entry_bar if self.trade_entry_bar is not None else 0)
                dt = self.data0.datetime.datetime(0)
                entry_price = self.order_entry_price if self.order_entry_price is not None else 0
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
                open_dt = self.trade_entry_dates[-1] if self.trade_entry_dates else "N/A"
                trade_record = {
                    'open_dt': open_dt,
                    'close_dt': dt,
                    'volume': self.current_volume if hasattr(self, "current_volume") and self.current_volume is not None else 0,
                    'pnl': profit_usd,
                    'pips': profit_pips,
                    'duration': duration,
                    'max_dd': intra_dd
                }
                self.trades.append(trade_record)
                print(f"[DEBUG]   TRADE CLOSED ({direction}): Date={dt}, Entry={entry_price:.5f}, Exit={exit_price:.5f}, "
                      f"Volume={trade_record['volume']}, PnL={profit_usd:.2f}, Pips={profit_pips:.2f}, "
                      f"Duration={duration} bars, MaxDD={intra_dd:.2f}, Balance={current_balance:.2f}")
                self.order_entry_price = None
                self.current_tp = None
                self.current_sl = None
                self.current_direction = None
                self.current_volume = None
        
        
        def stop(self):
            if self.position:
                self.close()
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
    # Dummy methods for interface compatibility
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

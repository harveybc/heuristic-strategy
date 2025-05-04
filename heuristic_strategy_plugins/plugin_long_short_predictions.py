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
        'rel_volume': 0.03,  # uses max0.5% of balance for each order (default)
        'min_order_volume': 10000,
        'max_order_volume': 1000000,
        'leverage': 100,
        'profit_threshold': 300,
        'min_drawdown_pips': 10,
        'tp_multiplier': 0.9,
        'sl_multiplier': 2.0,
        'lower_rr_threshold': -1000.0,
        'upper_rr_threshold': 1000.0,

        # Default uncertainty values used if none are provided:
        #"default_uncertainty_short_term": 0.0005,
        "default_uncertainty_short_term": 0.000001,
        #"default_uncertainty_long_term": 0.002,
        "default_uncertainty_long_term": 0.000001,
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.trades = []
        self.stats = {}

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
            ("profit_threshold", 0, 1000),
            ("tp_multiplier", 0.3, 3.0),
            ("sl_multiplier", 0.5, 6.0),
            ("lower_rr_threshold", -1000.0, 0.0),
            ("upper_rr_threshold", 0.1, 1000.0),
            #("time_horizon", 1, 48)
        ]

    # --- Updated evaluate_candidate method ---
    def evaluate_candidate(self, individual, base_data, hourly_predictions, daily_predictions, config):
        import os
        import pandas as pd
        import backtrader as bt

        # Unpack candidate parameters:
        # (profit_threshold, tp_multiplier, sl_multiplier, lower_rr_threshold, upper_rr_threshold, time_horizon)
        profit_threshold, tp_multiplier, sl_multiplier, lower_rr, upper_rr = individual
        #profit_threshold = self.params['profit_threshold']

        # Check that predictions are available.
        if (hourly_predictions is None or hourly_predictions.empty or 
            daily_predictions is None or daily_predictions.empty):
            print(f"[evaluate_candidate] Predictions are missing or empty for candidate {individual}. Returning profit=0.0.")
            return (0.0, {"num_trades": 0, "win_pct": 0, "max_dd": 0, "sharpe": 0})

        # Merge predictions (rename columns for clarity).
        merged_df = pd.DataFrame()
        renamed_h = {col: f"Prediction_h_{i+1}" for i, col in enumerate(hourly_predictions.columns)}
        hr = hourly_predictions.rename(columns=renamed_h)
        merged_df = hr.copy()
        renamed_d = {col: f"Prediction_d_{i+1}" for i, col in enumerate(daily_predictions.columns)}
        dr = daily_predictions.rename(columns=renamed_d)
        merged_df = merged_df.join(dr, how="outer")

        # --- Merge uncertainties from config (which were preprocessed by the data processor) ---
        # If uncertainties are not provided, create DataFrames using default constant values.
        uncertainty_hourly = config.get("uncertainty_hourly")
        if uncertainty_hourly is None:
            default_val = self.params.get('default_uncertainty_short_term', 0.0)
            uncertainty_hourly = pd.DataFrame(default_val, index=hourly_predictions.index,
                                               columns=[f"Uncertainty_h_{i+1}" for i in range(hourly_predictions.shape[1])])
        else:
            # If the file was loaded from file, rename its columns.
            uncertainty_hourly = uncertainty_hourly.rename(
                columns=lambda x: f"Uncertainty_h_{x.split('_')[-1]}" if x.startswith("Uncertainty_") and not x.startswith("Uncertainty_h_") else x
            )
        uncertainty_daily = config.get("uncertainty_daily")
        if uncertainty_daily is None:
            default_val = self.params.get('default_uncertainty_long_term', 0.0)
            uncertainty_daily = pd.DataFrame(default_val, index=daily_predictions.index,
                                              columns=[f"Uncertainty_d_{i+1}" for i in range(daily_predictions.shape[1])])
        else:
            uncertainty_daily = uncertainty_daily.rename(
                columns=lambda x: f"Uncertainty_d_{x.split('_')[-1]}" if x.startswith("Uncertainty_") and not x.startswith("Uncertainty_d_") else x
            )

        merged_df = merged_df.join(uncertainty_hourly, how="inner")
        merged_df = merged_df.join(uncertainty_daily, how="inner")

        if merged_df.empty:
            print(f"[evaluate_candidate] => Merged predictions and uncertainties are empty for candidate {individual}. Returning profit=0.0.")
            return (0.0, {"num_trades": 0, "win_pct": 0, "max_dd": 0, "sharpe": 0})

        if merged_df.index.name is None or merged_df.index.name != "DATE_TIME":
            merged_df.index.name = "DATE_TIME"

        # Pass the merged DataFrame directly to the strategy.
        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            self.HeuristicStrategy,
            pred_df=merged_df,
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
            max_trades_per_5days=config['max_trades_per_5days']
        )
        data_feed = bt.feeds.PandasData(dataname=base_data)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(10000.0)

        try:
            runresult = cerebro.run()
        except Exception as e:
            print("Error during backtest:", e)
            return (-1e6, {"num_trades": 0, "win_pct": 0, "max_dd": 0, "sharpe": 0})

        final_value = cerebro.broker.getvalue()
        profit = final_value - 10000.0
        print(f"Evaluated candidate {individual} -> Profit: {profit:.2f}")

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

        self.trades = trades_list

        num_trades = len(trades_list)
        stats = {"num_trades": num_trades, "win_pct": 0, "max_dd": 0, "sharpe": 0}
        if num_trades > 0:
            wins = sum(1 for tr in trades_list if tr['pnl'] > 0)
            win_pct = (wins / num_trades) * 100
            max_dd = max(tr['max_dd'] for tr in trades_list)
            profits = [tr['pnl'] for tr in trades_list]
            std_profit = np.std(profits) if num_trades > 1 else 0
            risk_free_rate = 0.1
            initial_capital = 10000.0
            risk_free_return = initial_capital * risk_free_rate
            excess_profit = profit - risk_free_return
            sharpe = excess_profit / std_profit if std_profit > 0 else 0
            stats.update({"win_pct": win_pct, "max_dd": max_dd, "sharpe": sharpe})

        # --- New functionality: Calculate risk as maximum drawdown ratio over balance history ---
        balance_history = strat_instance.balance_history
        running_max = -float('inf')
        max_drawdown_ratio = 0.0
        for b in balance_history:
            if b > running_max:
                running_max = b
            else:
                drawdown_ratio = (running_max - b) / running_max if running_max != 0 else 0
                if drawdown_ratio > max_drawdown_ratio:
                    max_drawdown_ratio = drawdown_ratio
        stats["risk"] = max_drawdown_ratio

        print(f"[EVALUATE] Candidate result => Profit: {profit:.2f}, "
              f"Trades: {stats.get('num_trades', 0)}, "
              f"Win%: {stats.get('win_pct', 0):.1f}, "
              f"MaxDD: {stats.get('max_dd', 0):.2f}, "
              f"Sharpe: {stats.get('sharpe', 0):.2f}",
              f"Risk: {stats.get('risk', 0):.2f}")
        self.stats = stats
        return (profit, stats)



    class HeuristicStrategy(bt.Strategy):
        """
        Forex Dynamic Volume Strategy using perfect future predictions.
        This replicates the original HeuristicStrategy exactly.
        """
        # --- Updated __init__ method of HeuristicStrategy ---
        def __init__(self, pred_df, pip_cost, rel_volume, min_order_volume, max_order_volume,
                     leverage, profit_threshold, min_drawdown_pips,
                     tp_multiplier, sl_multiplier, lower_rr_threshold, upper_rr_threshold,
                     max_trades_per_5days, *args, **kwargs):
            super().__init__()
            self.pred_df = pred_df
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

            self.num_hourly_preds = len([c for c in self.pred_df.columns if c.startswith('Prediction_h_')])
            self.num_daily_preds = len([c for c in self.pred_df.columns if c.startswith('Prediction_d_')])
            self.num_hourly_uncs = len([c for c in self.pred_df.columns if c.startswith('Uncertainty_h_')])
            self.num_daily_uncs = len([c for c in self.pred_df.columns if c.startswith('Uncertainty_d_')])

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

        # --- Corrected next() method ---
        def next(self):
            dt = self.data0.datetime.datetime(0)
            dt_hour = dt.replace(minute=0, second=0, microsecond=0)
            current_price = self.data0.close[0]
            balance = self.broker.getvalue()
            self.balance_history.append(balance)
            self.date_history.append(dt)

            # --- START: Signal Generation Logic (Moved to the beginning) ---
            signal = None
            chosen_rr = 0
            # Initialize TP/SL variables used for potential entry
            tp_buy_entry, sl_buy_entry, tp_sell_entry, sl_sell_entry = None, None, None, None

            if dt_hour in self.pred_df.index:
                row = self.pred_df.loc[dt_hour]
                try:
                    # --- Daily Predictions for Entry Signal & TP/SL Calculation ---
                    daily_preds = [row[f'Prediction_d_{i}'] for i in range(1, self.num_daily_preds + 1)]
                    if not daily_preds or all(pd.isna(p) for p in daily_preds): # Check for NaNs
                         daily_preds = []
                except KeyError:
                    daily_preds = []

                if daily_preds: # Only proceed if we have valid daily predictions
                    # Prepare adjusted daily predictions for TP/SL calculation (using global extremes)
                    if self.num_daily_uncs > 0:
                        daily_uncs = [row.get(f'Uncertainty_d_{i}', 0) for i in range(1, self.num_daily_uncs + 1)]
                        # Ensure uncertainty list matches prediction list length if needed
                        if len(daily_uncs) < len(daily_preds): daily_uncs.extend([0]*(len(daily_preds)-len(daily_uncs)))
                        elif len(daily_uncs) > len(daily_preds): daily_uncs = daily_uncs[:len(daily_preds)]
                        adjusted_preds_buy = [pred - unc for pred, unc in zip(daily_preds, daily_uncs)]
                        adjusted_preds_sell = [pred + unc for pred, unc in zip(daily_preds, daily_uncs)]
                    else:
                        adjusted_preds_buy = list(daily_preds) # Ensure list copy
                        adjusted_preds_sell = list(daily_preds)

                    # Calculate potential TP/SL for Buy based on adjusted daily extremes
                    if adjusted_preds_buy:
                        profit_pred_buy = max(adjusted_preds_buy)
                        max_idx_buy = adjusted_preds_buy.index(profit_pred_buy)
                        min_before_max_buy = min(adjusted_preds_buy[:max_idx_buy+1])
                        ideal_profit_pips_buy = max(0, (profit_pred_buy - current_price) / self.p.pip_cost) # Ensure non-negative
                        ideal_drawdown_pips_buy = max(0, (profit_pred_buy - min_before_max_buy) / self.p.pip_cost) # Ensure non-negative
                        if ideal_drawdown_pips_buy >= self.p.min_drawdown_pips: # Avoid division by zero or tiny drawdown
                            tp_buy_entry = current_price + self.p.tp_multiplier * ideal_profit_pips_buy * self.p.pip_cost
                            sl_buy_entry = current_price - self.p.sl_multiplier * ideal_drawdown_pips_buy * self.p.pip_cost

                    # Calculate potential TP/SL for Sell based on adjusted daily extremes
                    if adjusted_preds_sell:
                        min_pred_sell = min(adjusted_preds_sell)
                        min_idx_sell = adjusted_preds_sell.index(min_pred_sell)
                        max_before_min_sell = max(adjusted_preds_sell[: min_idx_sell + 1])
                        ideal_profit_pips_sell = max(0, (current_price - min_pred_sell) / self.p.pip_cost) # Ensure non-negative
                        ideal_drawdown_pips_sell = max(0, (max_before_min_sell - min_pred_sell) / self.p.pip_cost) # Ensure non-negative
                        if ideal_drawdown_pips_sell >= self.p.min_drawdown_pips: # Avoid division by zero or tiny drawdown
                            tp_sell_entry = current_price - self.p.tp_multiplier * ideal_profit_pips_sell * self.p.pip_cost
                            sl_sell_entry = current_price + self.p.sl_multiplier * ideal_drawdown_pips_sell * self.p.pip_cost

                    # --- Signal Generation based on UNADJUSTED Daily Threshold Crossing ---
                    future = daily_preds # Use raw daily preds for signal
                    future_moves = [(p - current_price)/self.p.pip_cost for p in future]

                    buy_idx  = next((i for i, m in enumerate(future_moves) if m >= self.p.profit_threshold), None)
                    sell_idx = next((i for i, m in enumerate(future_moves) if m <= -self.p.profit_threshold), None)

                    #print(f"[{self.data.datetime.datetime(0)}] Daily Pred Scan: buy_idx={buy_idx}, sell_idx={sell_idx}", flush=True) # Log potential signal

                    if buy_idx is None and sell_idx is None:
                        pass # No signal based on threshold crossing
                    elif sell_idx is not None and (buy_idx is None or sell_idx < buy_idx):
                        # Check if TP/SL for sell were successfully calculated
                        if tp_sell_entry is not None and sl_sell_entry is not None:
                            signal = 'short'
                            chosen_rr = ( -future_moves[sell_idx] ) # Use pip move for RR
                    elif buy_idx is not None and (sell_idx is None or buy_idx < sell_idx):
                         # Check if TP/SL for buy were successfully calculated
                        if tp_buy_entry is not None and sl_buy_entry is not None:
                            signal = 'long'
                            chosen_rr = future_moves[buy_idx] # Use pip move for RR
                    # else: signal remains None if buy_idx == sell_idx or TP/SL failed

            # --- END: Signal Generation Logic ---


            # --- Position Management (Check for Early Close using Hourly Preds) ---
            if self.position:
                # Update trade high/low
                if self.trade_low is None or current_price < self.trade_low: self.trade_low = current_price
                if self.trade_high is None or current_price > self.trade_high: self.trade_high = current_price

                should_close = False
                reason = "N/A"

                # Check Hard TP/SL first
                if self.current_direction == 'long':
                    if current_price >= self.current_tp: should_close, reason = True, "Hard TP"
                    elif current_price <= self.current_sl: should_close, reason = True, "Hard SL"
                elif self.current_direction == 'short':
                    if current_price <= self.current_tp: should_close, reason = True, "Hard TP"
                    elif current_price >= self.current_sl: should_close, reason = True, "Hard SL"

                # Check Early Close based on Hourly Predictions if not already closing
                if not should_close and dt_hour in self.pred_df.index and self.num_hourly_preds > 0:
                    row = self.pred_df.loc[dt_hour]
                    try:
                        hourly_preds = [row[f'Prediction_h_{i}'] for i in range(1, self.num_hourly_preds + 1)]
                        if not hourly_preds or all(pd.isna(p) for p in hourly_preds):
                            hourly_preds = []
                    except KeyError:
                        hourly_preds = []

                    if hourly_preds: # Only check if hourly preds are valid
                        # Apply uncertainty for early close check
                        if self.num_hourly_uncs > 0:
                            hourly_uncs = [row.get(f'Uncertainty_h_{i}', 0) for i in range(1, self.num_hourly_uncs + 1)]
                            if len(hourly_uncs) < len(hourly_preds): hourly_uncs.extend([0]*(len(hourly_preds)-len(hourly_uncs)))
                            elif len(hourly_uncs) > len(hourly_preds): hourly_uncs = hourly_uncs[:len(hourly_preds)]
                            adjusted_hourly_preds_long = [pred + unc for pred, unc in zip(hourly_preds, hourly_uncs)] # Worse case for long
                            adjusted_hourly_preds_short = [pred - unc for pred, unc in zip(hourly_preds, hourly_uncs)] # Worse case for short
                        else:
                            adjusted_hourly_preds_long = list(hourly_preds)
                            adjusted_hourly_preds_short = list(hourly_preds)

                        # Check if any hourly prediction crosses the SL
                        if self.current_direction == 'long' and adjusted_hourly_preds_long:
                            predicted_hourly_min = min(adjusted_hourly_preds_long)
                            if predicted_hourly_min < self.current_sl:
                                should_close, reason = True, f"Hourly Pred Min ({predicted_hourly_min:.5f}) < SL ({self.current_sl:.5f})"
                        elif self.current_direction == 'short' and adjusted_hourly_preds_short:
                            predicted_hourly_max = max(adjusted_hourly_preds_short)
                            if predicted_hourly_max > self.current_sl:
                                should_close, reason = True, f"Hourly Pred Max ({predicted_hourly_max:.5f}) > SL ({self.current_sl:.5f})"

                if should_close:
                    print(f"[{dt}] Closing position ({self.current_direction}) due to: {reason}")
                    self.close() # Close position
                    return # IMPORTANT: Exit next() after closing or managing position

                # If not closing, just return as we are managing the position
                return

            # --- Entry Logic (Only reached if NOT in position) ---
            else:
                # Reset trade high/low when not in position
                self.trade_low = None
                self.trade_high = None

            # Check trade frequency limit (only if not in position)
            recent_trades = [d for d in self.trade_entry_dates if (dt - d).days < 5]
            if len(recent_trades) >= self.p.max_trades_per_5days:
                # print(f"[{dt}] Trade limit reached, skipping entry.") # Optional log
                return

            # --- Place order based on the signal determined earlier ---
            if signal is not None:
                # Assign the TP/SL calculated during signal generation
                if signal == 'long':
                    chosen_tp, chosen_sl = tp_buy_entry, sl_buy_entry
                else: # signal == 'short'
                    chosen_tp, chosen_sl = tp_sell_entry, sl_sell_entry

                # Final check if TP/SL are valid before proceeding
                if chosen_tp is None or chosen_sl is None:
                     print(f"[{dt}] Signal '{signal}' generated, but TP/SL calculation failed, skipping trade.")
                     return

                order_size = self.compute_size(chosen_rr)
                if order_size <= 0:
                    print(f"[{dt}] Signal '{signal}' generated, but order size <= 0 ({order_size:.2f}), skipping trade")
                    return

                # Record entry details BEFORE placing order
                self.trade_entry_dates.append(dt)
                self.trade_entry_bar = len(self) # Bar index of entry
                self.current_volume = order_size
                self.current_tp = chosen_tp # Store TP/SL for management
                self.current_sl = chosen_sl
                self.current_direction = signal # Store direction

                # Reset trade high/low for the new trade
                self.trade_low = current_price
                self.trade_high = current_price

                # DEBUG before placing order
               # print(f"[{dt}] Attempting to place order: Signal={signal}, Size={order_size:.2f}, TP={chosen_tp:.5f}, SL={chosen_sl:.5f}, RR={chosen_rr:.2f}")

                # Place the actual order
                if signal == 'long':
                    self.buy(size=order_size)
                elif signal == 'short':
                    self.sell(size=order_size)

            # --- End of Entry Logic ---

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

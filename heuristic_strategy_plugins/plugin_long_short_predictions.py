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
        'rel_volume': 0.03,  # uses max 3% of balance for each order (default)
        'min_order_volume': 1000, # minimum micro lot size of 0.01 lots
        'max_order_volume': 1000000, # typpical max lot size of 10 lots
        'leverage': 100,
        'profit_threshold': 300,
        'min_drawdown_pips': 10,
        'tp_multiplier': 0.9,
        'sl_multiplier': 2.0,
        'lower_rr_threshold': 0.1,
        'upper_rr_threshold': 1.0,
        
        # Default uncertainty values used if none are provided:
        "default_uncertainty_short_term": 0.0005,
        #"default_uncertainty_short_term": 0.000001,
        "default_uncertainty_long_term": 0.009,
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
            ("profit_threshold", 100, 1000), # Still in pips
            ("tp_multiplier", 0.3, 3.0),
            ("sl_multiplier", 0.5, 6.0),
            # --- MODIFIED Bounds for Ratio Thresholds ---
            # Define sensible bounds for the profit/risk ratio
            ("lower_rr_threshold", 0.1, 1.5), # e.g., Lower threshold between 0.1 and 1.5 RR
            ("upper_rr_threshold", 1.0, 5.0), # e.g., Upper threshold between 1.0 and 5.0 RR
            # --- End Modification ---
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
            max_trades_per_5days=config['max_trades_per_5days'],
            # --- ADDED: Pass use_first_match from config ---
            use_first_match=config.get("use_first_match", True) # Default to True
            # --- END ADDED ---
        )
        data_feed = bt.feeds.PandasData(dataname=base_data)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(10000.0)

        # --- ADDED: Explicitly set broker leverage ---
        # Use the leverage defined in the plugin's parameters
        cerebro.broker.setcommission(
            commission=0.0, # Assuming no commission for now, adjust if needed
            margin=None,    # Use default margin calculation based on leverage
            leverage=self.params['leverage'] # Set the leverage explicitly
        )
        # --- END ADDED ---

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
                     max_trades_per_5days, use_first_match, *args, **kwargs): # Added use_first_match
            super().__init__()
            # --- ADDED: Store use_first_match directly ---
            self.use_first_match = use_first_match
            # --- END ADDED ---

            self.pred_df = pred_df
            # --- Assuming manual parameter storage based on visible code ---
            # Create a namespace or similar if self.params doesn't exist
            if not hasattr(self, 'params'):
                 from types import SimpleNamespace
                 self.params = SimpleNamespace()
            # --- Store other parameters ---
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
            # --- End parameter storage ---

            # ... (rest of __init__: num_preds, num_uncs, initial_balance, etc.) ...
            # ... existing code ...
            self.num_hourly_preds = len([c for c in self.pred_df.columns if c.startswith('Prediction_h_')])
            self.num_daily_preds = len([c for c in self.pred_df.columns if c.startswith('Prediction_d_')])
            self.num_hourly_uncs = len([c for c in self.pred_df.columns if c.startswith('Uncertainty_h_')])
            self.num_daily_uncs = len([c for c in self.pred_df.columns if c.startswith('Uncertainty_d_')])

            self.data0 = self.datas[0]
            # --- Corrected initial_balance assignment ---
            # self.initial_balance = self.broker.getvalue() # Cannot call this here, broker not ready
            self.initial_balance = None # Will be set in start()
            # --- End Correction ---
            self.trade_entry_dates = []
            self.balance_history = []
            self.date_history = []
            self.trade_low = None
            self.trade_high = None
            self.trades = []
            self.current_tp = None
            self.current_sl = None
            self.current_direction = None
            self.trade_entry_bar = None
            self.order_entry_price = None
            self.entry_order_direction = None
            # ... existing code ...
            print(f"Strategy Initialized: use_first_match={self.use_first_match}", flush=True) # Use self.use_first_match

        # --- Add start() method if missing, or ensure initial_balance is set ---
        def start(self):
            self.initial_balance = self.broker.getvalue()
            print(f"Strategy Start: Initial Balance={self.initial_balance:.2f}")
            # Initialize balance history with starting balance if needed
            dt = self.data0.datetime.datetime(0) if len(self.data0) else None # Handle empty data case
            if dt:
                self.balance_history.append(self.initial_balance)
                self.date_history.append(dt)

        # --- Corrected next() method ---
        def next(self):
            dt = self.data0.datetime.datetime(0)
            dt_hour = dt.replace(minute=0, second=0, microsecond=0)
            current_price = self.data0.close[0]
            balance = self.broker.getvalue()
            self.balance_history.append(balance)
            self.date_history.append(dt)

            # --- START: Signal Generation Logic (Revised TP/SL Calculation) ---
            signal = None
            chosen_rr = 0
            tp_entry, sl_entry = None, None # Generic TP/SL for entry
            ideal_profit_pips = 0 # Initialize
            effective_drawdown_pips = 0 # Initialize

            if dt_hour in self.pred_df.index:
                row = self.pred_df.loc[dt_hour]
                try:
                    daily_preds = [row[f'Prediction_d_{i}'] for i in range(1, self.num_daily_preds + 1)]
                    if not daily_preds or all(pd.isna(p) for p in daily_preds):
                         daily_preds = []
                except KeyError:
                    daily_preds = []

                # --- ADDED: Get Daily Uncertainties ---
                daily_uncs = []
                if self.num_daily_uncs > 0:
                    try:
                        daily_uncs = [row.get(f'Uncertainty_d_{i}', 0) for i in range(1, self.num_daily_uncs + 1)]
                        # Ensure daily_uncs has same length as daily_preds, padding with 0 if needed
                        if len(daily_uncs) < len(daily_preds):
                            daily_uncs.extend([0]*(len(daily_preds)-len(daily_uncs)))
                        elif len(daily_uncs) > len(daily_preds):
                            daily_uncs = daily_uncs[:len(daily_preds)]
                    except KeyError:
                        daily_uncs = [0] * len(daily_preds) # Fallback to zeros
                else:
                    daily_uncs = [0] * len(daily_preds) # Fallback to zeros if no uncertainty columns
                # --- END ADDED ---


                if daily_preds: # Only proceed if we have valid daily predictions
                    # --- Signal Generation based on UNADJUSTED Daily Threshold Crossing ---
                    future = daily_preds # Use raw daily preds for signal
                    future_moves = [(p - current_price)/self.p.pip_cost for p in future]

                    if self.use_first_match:
                        # --- EXACTLY THE CURRENT WORKING LOGIC (First Match) ---
                        buy_idx  = next((i for i, m in enumerate(future_moves) if m >= self.params.profit_threshold), None) # Use self.params
                        sell_idx = next((i for i, m in enumerate(future_moves) if m <= -self.params.profit_threshold), None) # Use self.params

                        # --- Determine Signal and Calculate TP/SL based on Signal Event ---
                        if sell_idx is not None and (buy_idx is None or sell_idx < buy_idx):
                            # --- Potential Short Signal ---
                            idx = sell_idx
                            ideal_profit_pips = -future_moves[idx] # Profit is the negative move (positive value)
                            max_before_signal = current_price
                            if idx > 0: max_before_signal = max([current_price] + future[:idx])
                            price_at_signal = future[idx]
                            ideal_drawdown_pips = max(0, (max_before_signal - price_at_signal) / self.params.pip_cost) # Use self.params

                            signal = 'short'
                            effective_drawdown_pips = max(ideal_drawdown_pips, self.params.min_drawdown_pips) # Use self.params

                            # --- Apply Uncertainty to Profit and Drawdown ---
                            uncertainty_at_idx = daily_uncs[idx] if idx < len(daily_uncs) else 0
                            uncertainty_pips = uncertainty_at_idx / self.params.pip_cost # Use self.params
                            adjusted_profit_pips = max(0, ideal_profit_pips - uncertainty_pips)
                            adjusted_drawdown_pips = effective_drawdown_pips + uncertainty_pips
                            # --- End Uncertainty Application ---

                            chosen_rr = adjusted_profit_pips / adjusted_drawdown_pips if adjusted_drawdown_pips > 0 else 0

                            # --- Use Adjusted Pips for TP/SL ---
                            tp_entry = current_price - self.params.tp_multiplier * adjusted_profit_pips * self.params.pip_cost # Use self.params
                            sl_entry = current_price + self.params.sl_multiplier * adjusted_drawdown_pips * self.params.pip_cost # Use self.params
                            # --- End TP/SL Adjustment ---

                        elif buy_idx is not None and (sell_idx is None or buy_idx < sell_idx):
                            # --- Potential Long Signal ---
                            idx = buy_idx
                            ideal_profit_pips = future_moves[idx] # Profit is the positive move
                            min_before_signal = current_price
                            if idx > 0: min_before_signal = min([current_price] + future[:idx])
                            # --- Corrected Drawdown Calculation for Long ---
                            ideal_drawdown_pips = max(0, (current_price - min_before_signal) / self.params.pip_cost) # Use self.params
                            # --- End Correction ---

                            signal = 'long'
                            effective_drawdown_pips = max(ideal_drawdown_pips, self.params.min_drawdown_pips) # Use self.params

                            # --- Apply Uncertainty to Profit and Drawdown ---
                            uncertainty_at_idx = daily_uncs[idx] if idx < len(daily_uncs) else 0
                            uncertainty_pips = uncertainty_at_idx / self.params.pip_cost # Use self.params
                            adjusted_profit_pips = max(0, ideal_profit_pips - uncertainty_pips)
                            adjusted_drawdown_pips = effective_drawdown_pips + uncertainty_pips
                            # --- End Uncertainty Application ---

                            chosen_rr = adjusted_profit_pips / adjusted_drawdown_pips if adjusted_drawdown_pips > 0 else 0

                            # --- Use Adjusted Pips for TP/SL ---
                            tp_entry = current_price + self.params.tp_multiplier * adjusted_profit_pips * self.params.pip_cost # Use self.params
                            sl_entry = current_price - self.params.sl_multiplier * adjusted_drawdown_pips * self.params.pip_cost # Use self.params
                            # --- End TP/SL Adjustment ---
                        # --- End Existing Logic ---

                    else:
                        # --- NEW LOGIC (Best RR Match) ---
                        best_buy_rr = -1.0
                        best_buy_idx = None
                        best_buy_tp = None
                        best_buy_sl = None

                        best_sell_rr = -1.0
                        best_sell_idx = None
                        best_sell_tp = None
                        best_sell_sl = None

                        for i, move in enumerate(future_moves):
                            # Check for potential Buy signal at index i
                            if move >= self.params.profit_threshold: # Use self.params
                                temp_ideal_profit_pips = move
                                temp_min_before_signal = current_price
                                if i > 0: temp_min_before_signal = min([current_price] + future[:i])
                                temp_ideal_drawdown_pips = max(0, (current_price - temp_min_before_signal) / self.params.pip_cost) # Use self.params
                                temp_effective_drawdown_pips = max(temp_ideal_drawdown_pips, self.params.min_drawdown_pips) # Use self.params

                                temp_uncertainty_at_idx = daily_uncs[i] if i < len(daily_uncs) else 0
                                temp_uncertainty_pips = temp_uncertainty_at_idx / self.params.pip_cost # Use self.params
                                temp_adj_profit = max(0, temp_ideal_profit_pips - temp_uncertainty_pips)
                                temp_adj_drawdown = temp_effective_drawdown_pips + temp_uncertainty_pips
                                current_rr = temp_adj_profit / temp_adj_drawdown if temp_adj_drawdown > 0 else 0

                                if current_rr > best_buy_rr:
                                    best_buy_rr = current_rr
                                    best_buy_idx = i
                                    best_buy_tp = current_price + self.params.tp_multiplier * temp_adj_profit * self.params.pip_cost # Use self.params
                                    best_buy_sl = current_price - self.params.sl_multiplier * temp_adj_drawdown * self.params.pip_cost # Use self.params

                            # Check for potential Sell signal at index i
                            elif move <= -self.params.profit_threshold: # Use self.params
                                temp_ideal_profit_pips = -move
                                temp_max_before_signal = current_price
                                if i > 0: temp_max_before_signal = max([current_price] + future[:i])
                                temp_price_at_signal = future[i]
                                temp_ideal_drawdown_pips = max(0, (temp_max_before_signal - temp_price_at_signal) / self.params.pip_cost) # Use self.params
                                temp_effective_drawdown_pips = max(temp_ideal_drawdown_pips, self.params.min_drawdown_pips) # Use self.params

                                temp_uncertainty_at_idx = daily_uncs[i] if i < len(daily_uncs) else 0
                                temp_uncertainty_pips = temp_uncertainty_at_idx / self.params.pip_cost # Use self.params
                                temp_adj_profit = max(0, temp_ideal_profit_pips - temp_uncertainty_pips)
                                temp_adj_drawdown = temp_effective_drawdown_pips + temp_uncertainty_pips
                                current_rr = temp_adj_profit / temp_adj_drawdown if temp_adj_drawdown > 0 else 0

                                if current_rr > best_sell_rr:
                                    best_sell_rr = current_rr
                                    best_sell_idx = i
                                    best_sell_tp = current_price - self.params.tp_multiplier * temp_adj_profit * self.params.pip_cost # Use self.params
                                    best_sell_sl = current_price + self.params.sl_multiplier * temp_adj_drawdown * self.params.pip_cost # Use self.params

                        # Compare best buy and sell signals found
                        if best_buy_rr > best_sell_rr and best_buy_idx is not None:
                            signal = 'long'
                            chosen_rr = best_buy_rr
                            tp_entry = best_buy_tp
                            sl_entry = best_buy_sl
                        elif best_sell_rr >= best_buy_rr and best_sell_idx is not None: # Prefer short on tie
                            signal = 'short'
                            chosen_rr = best_sell_rr
                            tp_entry = best_sell_tp
                            sl_entry = best_sell_sl
                        # else: signal remains None
                        # --- End New Logic ---
                    # --- END: Conditional Signal Logic ---

            # --- Position Management ---
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
                    #print(f"[{dt}] Closing position ({self.current_direction}) due to: {reason}")
                    self.close() # Close position
                    # CRITICAL: Return AFTER closing to prevent immediate re-entry on the same bar
                    return # Exit next() after closing position

                # --- REMOVED THE PREMATURE RETURN ---
                # If not closing, execution will now CONTINUE to the entry logic below,
                # even if a position is currently held.
                # return # <<<< REMOVED THIS LINE (Previously around line 380)

            # --- Entry Logic (Now reachable even if self.position was true, unless closed above) ---
            else:
                # Reset trade high/low when not in position (This part is fine)
                self.trade_low = None
                self.trade_high = None

            # Check trade frequency limit (only if not in position - This might need adjustment if allowing reversals)
            # Consider if this check should happen earlier or be modified if reversing positions is intended.
            recent_trades = [d for d in self.trade_entry_dates if (dt - d).days < 5]
            if len(recent_trades) >= self.params.max_trades_per_5days and not self.position: # Apply only if flat?
                 # print(f"[{dt}] Trade limit reached, skipping entry.") # Optional log
                 return

            # --- Place order based on the signal determined earlier ---
            if signal is not None:
                if self.position:
                    return # Already in position

                chosen_tp, chosen_sl = tp_entry, sl_entry
                if chosen_tp is None or chosen_sl is None:
                     print(f"[{dt}] Signal '{signal}' generated, but TP/SL calculation failed, skipping trade.")
                     return

                # --- Step 1: Calculate size based on RR (using existing compute_size) ---
                order_size_rr_based = self.compute_size(chosen_rr, current_price, self.initial_balance)
                # --- End Step 1 ---

                # --- Step 2: Calculate Max Size based on Notional Value Cap ---
                current_balance = self.broker.getvalue() # Use current equity/balance
                # --- CORRECTED: Calculate max NOTIONAL value allowed ---
                max_notional_value_allowed = current_balance * self.params.rel_volume * self.params.leverage
                # --- End Correction ---
                max_size_from_notional_cap = 0
                if current_price > 0:
                    # Calculate max size whose notional value is <= max_notional_value_allowed
                    max_size_from_notional_cap = max_notional_value_allowed / current_price
                else:
                    print(f"[{dt}] WARNING: Current price is zero. Cannot calculate notional cap size.")
                    max_size_from_notional_cap = 0

                # Ensure the notional cap size is non-negative
                max_size_from_notional_cap = max(0, max_size_from_notional_cap)

                # --- End Step 2 ---

                # --- Step 3: Determine Final Order Size ---
                # Final size is the minimum of the RR-based size and the notional-cap size
                order_size = min(order_size_rr_based, max_size_from_notional_cap)

                # --- Step 4: Final Clamp within Absolute Min/Max Volume ---
                # Ensure the final size is within the absolute min/max bounds,
                # respecting the min_vol unless the notional cap forced it lower/zero.
                if order_size > 0:
                    # If size is positive, ensure it's at least min_vol and at most max_vol
                    order_size = max(self.params.min_order_volume, min(order_size, self.params.max_order_volume))
                else:
                    # If size was capped to zero or less, ensure it's exactly zero
                    order_size = 0

                # --- UPDATED Debug Print Variable Name ---
                #print(f"[{dt}] DEBUG: Size Calc ({signal}): RR={chosen_rr:.2f} -> RR_Size={order_size_rr_based:.2f}. NotionalCapSize={max_size_from_notional_cap:.2f}. FinalSize={order_size:.2f}", flush=True)
                # --- End Step 4 ---


                # Check Final Order Size
                if order_size <= 0:
                    print(f"[{dt}] Signal '{signal}' generated, but final order size <= 0 ({order_size:.2f}) after value cap/clamp, skipping trade")
                    return

                # Record entry details BEFORE placing order
                self.trade_entry_dates.append(dt)
                self.trade_entry_bar = len(self) # Bar index of entry
                self.current_volume = order_size # Use the final capped size
                self.current_tp = chosen_tp # Store TP/SL for management
                self.current_sl = chosen_sl
                self.current_direction = signal # Store direction

                # Reset trade high/low for the new trade
                self.trade_low = current_price
                self.trade_high = current_price

                # Place the actual order
                if signal == 'long':
                    #print(f"[{dt}] DEBUG: PLACING LONG ORDER: Size={order_size:.2f}", flush=True)
                    self.buy(size=order_size)
                elif signal == 'short':
                    #print(f"[{dt}] DEBUG: PLACING SHORT ORDER: Size={order_size:.2f}", flush=True)
                    # --- ADDED MISSING CALL ---
                    self.sell(size=order_size)
                    # --- END ADDED ---

            # --- End of Entry Logic ---

        # --- Corrected compute_size method (Removed Value Limit Constraint) ---
        # Arguments current_price, initial_balance are kept for signature consistency but not used here
        def compute_size(self, rr, current_price, initial_balance):
            min_vol = self.params.min_order_volume
            max_vol = self.params.max_order_volume
            lower_rr = self.params.lower_rr_threshold # Ratio-based
            upper_rr = self.params.upper_rr_threshold # Ratio-based

            # 1. Calculate size based on RR thresholds (linear interpolation)
            size_from_rr = 0
            if upper_rr <= lower_rr: # Handle invalid range
                size_from_rr = min_vol if rr <= lower_rr else max_vol
            elif rr >= upper_rr:
                size_from_rr = max_vol
            elif rr <= lower_rr:
                size_from_rr = min_vol
            else:
                # Linear interpolation between min_vol and max_vol
                size_from_rr = min_vol + ((rr - lower_rr) / (upper_rr - lower_rr)) * (max_vol - min_vol)

            # --- REMOVED Value Limit Calculation and Constraint ---
            # max_value_allowed = initial_balance * self.params.rel_volume * self.params.leverage
            # max_size_from_value = float('inf')
            # if current_price > 0:
            #     max_size_from_value = max_value_allowed / current_price
            # else:
            #     print(f"[{self.data0.datetime.datetime(0)}] WARNING: Current price is zero in compute_size. Cannot calculate size.")
            #     return 0
            # --- END REMOVED ---

            # 3. Determine final size: Use RR-based size, capped only by absolute max_vol param
            #    Ensure it doesn't go below min_vol
            # --- MODIFIED: Removed max_size_from_value from min() ---
            final_size = max(min_vol, min(size_from_rr, max_vol))
            # --- END MODIFIED ---

            # Ensure final size is non-negative
            final_size = max(0, final_size)

            # --- Optional Debug Print ---
            # print(f"[{self.data0.datetime.datetime(0)}] Compute Size (No Value Limit): rr={rr:.2f}, size_rr={size_from_rr:.2f}, final_size={final_size:.2f}")
            # --- End Debug Print ---

            return final_size

        def notify_order(self, order):
            dt = self.data0.datetime.datetime(0)
            if order.status in [order.Submitted, order.Accepted]:
                #print(f"[{dt}] NOTIFY ORDER ACCEPTED/SUBMITTED: Ref={order.ref}")
                return # Do nothing for these states

            if order.status in [order.Completed]:
                # Check if it's an entry order completion we were expecting
                is_entry_attempt = (self.entry_order_direction is not None)

                if is_entry_attempt:
                    # Check if the completed order matches the expected direction
                    order_matches_direction = False
                    if self.entry_order_direction == 'long' and order.isbuy():
                        order_matches_direction = True
                    elif self.entry_order_direction == 'short' and order.issell():
                        order_matches_direction = True

                    if order_matches_direction:
                         # --- REMOVED FAULTY CHECK ---
                         # If order_matches_direction is true, this IS the entry confirmation.
                         # self.position is already updated by Backtrader when this runs.
                         # if not self.position: # <<< REMOVED THIS LINE
                         # --- END REMOVED ---

                         # Record the entry price from the first fill notification for this attempt
                         if self.order_entry_price is None:
                             self.order_entry_price = order.executed.price
                             #print(f"[{dt}] NOTIFY ORDER COMPLETED (Entry Detected): Ref={order.ref}, Entry Direction Set To='{self.current_direction}', Entry Price Set To={self.order_entry_price:.5f}", flush=True)
                         #else:
                             # Log subsequent fills if needed (e.g., for partial fills)
                             #print(f"[{dt}] NOTIFY ORDER COMPLETED (Subsequent Fill): Ref={order.ref}, Price={order.executed.price:.5f}, Size={order.executed.size}", flush=True)

                         # Reset the flag indicating we are waiting for an entry confirmation.
                         self.entry_order_direction = None # Reset flag after processing entry

                    else:
                         # Order completed, but it didn't match the expected entry direction.
                         #print(f"[{dt}] NOTIFY ORDER COMPLETED (Mismatch/Non-Entry): Ref={order.ref}, Expected='{self.entry_order_direction}', Got={'BUY' if order.isbuy() else 'SELL'}, Exec Size={order.executed.size}", flush=True)
                         # Reset flag if it was somehow still set
                         self.entry_order_direction = None

                #else:
                     # Order completed, but we weren't expecting an entry (likely a close order)
                     #print(f"[{dt}] NOTIFY ORDER COMPLETED (Close Order?): Ref={order.ref}, Type={'BUY' if order.isbuy() else 'SELL'}, Exec Size={order.executed.size}", flush=True)


            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                print(f"[{dt}] NOTIFY ORDER FAILED/CANCELLED: Ref={order.ref}, Status={order.getstatusname()}")
                # Reset entry flag if the failed order was intended as an entry
                is_entry_attempt = (self.entry_order_direction is not None)
                if is_entry_attempt:
                     # Check if the failed order direction matches the expected entry direction
                     failed_matches_direction = False
                     if self.entry_order_direction == 'long' and order.isbuy():
                         failed_matches_direction = True
                     elif self.entry_order_direction == 'short' and order.issell():
                         failed_matches_direction = True

                     if failed_matches_direction:
                         print(f"[{dt}] DEBUG: Resetting entry_order_direction flag due to failed {self.entry_order_direction} order.")
                         self.entry_order_direction = None
                         # Do not reset self.current_direction/volume here, wait for next bar or notify_trade

        def notify_trade(self, trade):
            dt = self.data0.datetime.datetime(0)

            if trade.isclosed:
                # --- MODIFIED: Use state variables set during entry ---
                # Use the direction and volume stored when the order was placed in next()
                direction = self.current_direction # Should be 'long' or 'short'
                entry_price = self.order_entry_price if self.order_entry_price is not None else trade.price # Use stored entry price if available, fallback to trade.price
                trade_volume = self.current_volume if self.current_volume is not None else 0 # Use stored volume

                if direction is None or trade_volume is None:
                     # Fallback if state wasn't set correctly (shouldn't happen ideally)
                     print(f"[{dt}] WARNING: State variables (current_direction/current_volume) missing in notify_trade. Falling back.")
                     direction = 'long' if trade.pnlcomm > 0 else 'short' # Guess based on PnL (less reliable)
                     trade_volume = abs(trade.size) # Fallback to potentially unreliable trade.size
                     entry_price = trade.price

                #print(f"[{dt}] NOTIFY TRADE CLOSED - Using State: Direction='{direction}', Volume={trade_volume:.2f}, Stored EntryPrice={self.order_entry_price if self.order_entry_price is not None else 'N/A'}, TradePrice={trade.price:.5f}", flush=True)
                # --- End Modification ---

                duration = len(self) - (self.trade_entry_bar if self.trade_entry_bar is not None else 0)
                profit_usd = trade.pnlcomm
                profit_pips = 0
                intra_dd = 0
                exit_price_implied = entry_price # Use the determined entry_price

                pip_cost = self.p.pip_cost

                # Ensure calculations are safe
                if entry_price != 0 and trade_volume != 0 and pip_cost != 0:
                    if direction == 'long':
                        exit_price_implied = entry_price + (profit_usd / trade_volume)
                        profit_pips = (exit_price_implied - entry_price) / pip_cost
                        if self.trade_low is not None:
                             intra_dd = max(0, (entry_price - self.trade_low) / pip_cost)
                    elif direction == 'short':
                        exit_price_implied = entry_price - (profit_usd / trade_volume)
                        profit_pips = (entry_price - exit_price_implied) / pip_cost
                        if self.trade_high is not None:
                             intra_dd = max(0, (self.trade_high - entry_price) / pip_cost)
                else:
                     # This warning might still appear if trade_volume was None and fallback failed
                     print(f"[{dt}] WARNING: Cannot calculate pips/exit price accurately (Entry={entry_price:.5f}, Volume={trade_volume:.2f}, PipCost={pip_cost})")


                current_balance = self.broker.getvalue()
                open_dt = self.trade_entry_dates[-1] if self.trade_entry_dates else "N/A"

                trade_record = {
                    'open_dt': open_dt,
                    'close_dt': dt,
                    'volume': trade_volume, # Use volume from state
                    'pnl': profit_usd,
                    'pips': profit_pips,
                    'duration': duration,
                    'max_dd': intra_dd,
                    'direction': direction # Use direction from state
                }
                self.trades.append(trade_record)
                print(f"[DEBUG]   TRADE CLOSED ({direction}): Date={dt}, Entry={entry_price:.5f}, Exit=(Implied){exit_price_implied:.5f}, "
                      f"Volume={trade_record['volume']:.2f}, PnL={profit_usd:.2f}, Pips={profit_pips:.2f}, "
                      f"Duration={duration} bars, MaxDD={intra_dd:.2f}, Balance={current_balance:.2f}")

                # --- CRITICAL: Reset state variables AFTER logging the closed trade ---
                self.order_entry_price = None
                self.entry_order_direction = None # Reset this too, although less critical now
                self.current_tp = None
                self.current_sl = None
                self.current_direction = None # Reset direction state
                self.current_volume = None  # Reset volume state
                self.trade_entry_bar = None
                # Reset high/low in next() when confirmed flat

        def stop(self):
            if self.position:
                self.close()
            min_balance = min(self.balance_history) if self.balance_history else 0
            n_trades = len(self.trades)
            n_long_trades = 0
            n_short_trades = 0
            long_trade_percentage = 0.0
            short_trade_percentage = 0.0

            if n_trades > 0:
                avg_profit_usd = sum(t['pnl'] for t in self.trades) / n_trades
                avg_profit_pips = sum(t['pips'] for t in self.trades) / n_trades
                avg_duration = sum(t['duration'] for t in self.trades) / n_trades
                avg_max_dd = sum(t['max_dd'] for t in self.trades) / n_trades
                n_long_trades = sum(1 for t in self.trades if t.get('direction') == 'long')
                n_short_trades = n_trades - n_long_trades # Calculate shorts
                long_trade_percentage = (n_long_trades / n_trades) * 100
                short_trade_percentage = (n_short_trades / n_trades) * 100
            else:
                avg_profit_usd = avg_profit_pips = avg_duration = avg_max_dd = 0

            final_balance = self.broker.getvalue()
            print("\n==== Summary ====")
            print(f"Initial Balance (USD): {self.initial_balance:.2f}")
            print(f"Final Balance (USD):   {final_balance:.2f}")
            print(f"Minimum Balance (USD): {min_balance:.2f}")
            print(f"Number of Trades: {n_trades}")
            if n_trades > 0:
                print(f"  Long Trades:  {n_long_trades} ({long_trade_percentage:.1f}%)")
                print(f"  Short Trades: {n_short_trades} ({short_trade_percentage:.1f}%)")
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

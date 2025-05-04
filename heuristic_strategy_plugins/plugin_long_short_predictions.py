import datetime
import os
import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        params = (
            ('pip_cost', 0.0001),
            ('rel_volume', 0.01),
            ('min_order_volume', 1000),
            ('max_order_volume', 1000000),
            ('leverage', 30),
            ('profit_threshold', 50), # Pips
            ('min_drawdown_pips', 10), # Minimum acceptable drawdown for SL calculation
            ('tp_multiplier', 1.0),
            ('sl_multiplier', 1.0),
            ('lower_rr_threshold', 100), # RR in pips for min size
            ('upper_rr_threshold', 1000), # RR in pips for max size
            ('max_trades_per_5days', 3),
        )

        def __init__(self, pred_df, pip_cost, rel_volume, min_order_volume, max_order_volume,
                     leverage, profit_threshold, min_drawdown_pips,
                     tp_multiplier, sl_multiplier, lower_rr_threshold, upper_rr_threshold,
                     max_trades_per_5days, *args, **kwargs):

            self.pred_df = pred_df
            # Override params from constructor if provided
            self.p.pip_cost = pip_cost
            self.p.rel_volume = rel_volume
            self.p.min_order_volume = min_order_volume
            self.p.max_order_volume = max_order_volume
            self.p.leverage = leverage
            self.p.profit_threshold = profit_threshold
            self.p.min_drawdown_pips = min_drawdown_pips
            self.p.tp_multiplier = tp_multiplier
            self.p.sl_multiplier = sl_multiplier
            self.p.lower_rr_threshold = lower_rr_threshold
            self.p.upper_rr_threshold = upper_rr_threshold
            self.p.max_trades_per_5days = max_trades_per_5days

            self.trades = []
            self.balance_history = []
            self.date_history = []
            self.trade_entry_dates = []
            self.initial_balance = 0
            self.num_daily_preds = kwargs.get('num_daily_preds', 6)
            self.num_hourly_preds = kwargs.get('num_hourly_preds', 6)

            # --- State Variables ---
            self.current_tp = None
            self.current_sl = None
            self.current_direction = None # 'long' or 'short' when in position
            self.trade_entry_bar = None # Bar number of entry
            self.current_volume = None # Store volume of current trade
            self.trade_low = None # Track low price during trade
            self.trade_high = None # Track high price during trade
            # REMOVED state variables previously used for passing info

        def start(self):
            self.initial_balance = self.broker.getvalue()
            self.balance_history.append(self.initial_balance)
            # Use data start date if available
            start_date = self.data0.datetime.date(0) if len(self.data0) > 0 else "N/A"
            self.date_history.append(start_date)

        def next(self):
            dt = self.data0.datetime.datetime(0)
            dt_hour = dt.replace(minute=0, second=0, microsecond=0)
            current_price = self.data0.close[0]
            balance = self.broker.getvalue()
            self.balance_history.append(balance)
            self.date_history.append(dt)

            # --- START: Signal Generation Logic ---
            signal = None
            chosen_rr = 0
            tp_entry, sl_entry = None, None

            if dt_hour in self.pred_df.index:
                row = self.pred_df.loc[dt_hour]
                try:
                    daily_preds = [row[f'Prediction_d_{i}'] for i in range(1, self.num_daily_preds + 1)]
                    if not daily_preds or all(pd.isna(p) for p in daily_preds): daily_preds = []
                except KeyError: daily_preds = []

                if daily_preds:
                    future = daily_preds
                    future_moves = [(p - current_price)/self.p.pip_cost for p in future]
                    buy_idx  = next((i for i, m in enumerate(future_moves) if m >= self.p.profit_threshold), None)
                    sell_idx = next((i for i, m in enumerate(future_moves) if m <= -self.p.profit_threshold), None)

                    if sell_idx is not None and (buy_idx is None or sell_idx < buy_idx):
                        idx = sell_idx
                        ideal_profit_pips = -future_moves[idx]
                        max_before_signal = current_price
                        if idx > 0: max_before_signal = max([current_price] + future[:idx])
                        price_at_signal = future[idx]
                        ideal_drawdown_pips = max(0, (max_before_signal - price_at_signal) / self.p.pip_cost)
                        effective_drawdown_pips = max(ideal_drawdown_pips, self.p.min_drawdown_pips)

                        signal = 'short'
                        chosen_rr = ideal_profit_pips
                        tp_entry = current_price - self.p.tp_multiplier * ideal_profit_pips * self.p.pip_cost
                        sl_entry = current_price + self.p.sl_multiplier * effective_drawdown_pips * self.p.pip_cost

                    elif buy_idx is not None and (sell_idx is None or buy_idx < sell_idx):
                        idx = buy_idx
                        ideal_profit_pips = future_moves[idx]
                        min_before_signal = current_price
                        if idx > 0: min_before_signal = min([current_price] + future[:idx])
                        price_at_signal = future[idx]
                        ideal_drawdown_pips = max(0, (price_at_signal - min_before_signal) / self.p.pip_cost)
                        effective_drawdown_pips = max(ideal_drawdown_pips, self.p.min_drawdown_pips)

                        signal = 'long'
                        chosen_rr = ideal_profit_pips
                        tp_entry = current_price + self.p.tp_multiplier * ideal_profit_pips * self.p.pip_cost
                        sl_entry = current_price - self.p.sl_multiplier * effective_drawdown_pips * self.p.pip_cost
            # --- END: Signal Generation Logic ---

            # --- Position Management ---
            if self.position:
                # Update trade high/low only if state is consistent
                if self.current_direction is not None:
                    if self.trade_low is None or current_price < self.trade_low: self.trade_low = current_price
                    if self.trade_high is None or current_price > self.trade_high: self.trade_high = current_price

                should_close = False
                reason = "N/A"
                # Check Hard TP/SL (use state set at entry)
                if self.current_direction == 'long':
                    if self.current_tp is not None and current_price >= self.current_tp: should_close, reason = True, "Hard TP"
                    elif self.current_sl is not None and current_price <= self.current_sl: should_close, reason = True, "Hard SL"
                elif self.current_direction == 'short':
                    if self.current_tp is not None and current_price <= self.current_tp: should_close, reason = True, "Hard TP"
                    elif self.current_sl is not None and current_price >= self.current_sl: should_close, reason = True, "Hard SL"

                # Check Early Close (Hourly Preds)
                if not should_close and dt_hour in self.pred_df.index and self.num_hourly_preds > 0:
                    # ... (Insert your existing hourly early close logic here) ...
                    # Example placeholder:
                    # hourly_preds = ...
                    # if some_condition(hourly_preds, self.current_direction):
                    #    should_close, reason = True, "Hourly Early Close"
                    pass

                if should_close:
                    print(f"[{dt}] Closing position ({self.current_direction or 'Unknown'}) due to: {reason}")
                    self.close() # Close position
                    # Reset direction/TP/SL immediately after initiating close
                    # self.current_direction = None # Let notify_trade handle final reset
                    # self.current_tp = None
                    # self.current_sl = None
                    return # Exit next() after closing position

                # If not closing, just wait for next bar.
                return # Return if still managing position

            # --- Entry Logic (Only reached if NOT in position) ---
            else: # Reset high/low when confirmed flat
                self.trade_low = None
                self.trade_high = None

            # Trade frequency limit
            recent_trades = [d for d in self.trade_entry_dates if (dt - d).days < 5]
            if len(recent_trades) >= self.p.max_trades_per_5days:
                 # print(f"[{dt}] Trade limit reached, skipping entry.") # Optional log
                 return

            # Place order based on signal
            if signal is not None:
                # Ensure we are definitely flat before entering
                if self.position:
                    print(f"[{dt}] Signal '{signal}' generated, but position object still exists. Skipping entry.")
                    return

                chosen_tp, chosen_sl = tp_entry, sl_entry
                if chosen_tp is None or chosen_sl is None:
                     print(f"[{dt}] Signal '{signal}' generated, but TP/SL is None, skipping trade.")
                     return

                order_size = self.compute_size(chosen_rr)
                if order_size <= 0:
                    print(f"[{dt}] Signal '{signal}' generated, but order size <= 0 ({order_size:.2f}), skipping trade")
                    return

                # --- Set state JUST BEFORE placing order ---
                self.current_tp = chosen_tp
                self.current_sl = chosen_sl
                self.current_direction = signal # Set intended direction
                self.current_volume = order_size
                self.trade_entry_bar = len(self) # Bar number index
                self.trade_entry_dates.append(dt)
                self.trade_low = current_price # Initialize trade high/low for this trade
                self.trade_high = current_price

                # Place the actual order
                if signal == 'long':
                    print(f"[{dt}] PLACING LONG ORDER: Size={order_size:.2f}, TP={chosen_tp:.5f}, SL={chosen_sl:.5f}", flush=True)
                    self.buy(size=order_size)
                elif signal == 'short':
                    print(f"[{dt}] PLACING SHORT ORDER: Size={order_size:.2f}, TP={chosen_tp:.5f}, SL={chosen_sl:.5f}", flush=True)
                    self.sell(size=order_size)

        def compute_size(self, rr):
            min_vol = self.params.min_order_volume
            max_vol = self.params.max_order_volume
            # Ensure thresholds are valid
            lower_rr = self.params.lower_rr_threshold
            upper_rr = self.params.upper_rr_threshold
            if upper_rr <= lower_rr: # Avoid division by zero / invalid range
                size = min_vol if rr <= lower_rr else max_vol
            elif rr >= upper_rr:
                size = max_vol
            elif rr <= lower_rr:
                size = min_vol
            else:
                size = min_vol + ((rr - lower_rr) / (upper_rr - lower_rr)) * (max_vol - min_vol)

            cash = self.broker.getcash()
            # Ensure leverage and rel_volume are positive
            leverage = max(1, self.params.leverage) # Use at least 1x leverage
            rel_volume = max(0, self.params.rel_volume)
            max_from_cash = cash * rel_volume * leverage

            calculated_size = min(size, max_from_cash)
            # Ensure final size is not negative
            return max(0, calculated_size)

        def notify_order(self, order):
            dt = self.data0.datetime.datetime(0)
            if order.status in [order.Completed]:
                print(f"[{dt}] NOTIFY ORDER COMPLETED: Ref={order.ref}, Type={'BUY' if order.isbuy() else 'SELL'}, Exec Size={order.executed.size}, Exec Price={order.executed.price:.5f}", flush=True)
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                print(f"[{dt}] NOTIFY ORDER FAILED/CANCELLED: Ref={order.ref}, Status={order.getstatusname()}", flush=True)
                # If an entry order failed, reset the state that was set in next()
                # Check if the failed order matches the intended entry state
                is_entry_order_failed = False
                if self.current_direction == 'long' and order.isbuy() and not self.position:
                    is_entry_order_failed = True
                elif self.current_direction == 'short' and order.issell() and not self.position:
                    is_entry_order_failed = True

                if is_entry_order_failed:
                     print(f"[{dt}] Entry order ({self.current_direction}) failed, resetting state.")
                     self.current_tp = None
                     self.current_sl = None
                     self.current_direction = None
                     self.current_volume = None
                     self.trade_entry_bar = None
                     if self.trade_entry_dates: self.trade_entry_dates.pop() # Remove tentative entry date
                     self.trade_low = None
                     self.trade_high = None

        def notify_trade(self, trade):
            dt = self.data0.datetime.datetime(0)
            if trade.isclosed:
                # --- Determine Direction and Entry Price from Trade Object ---
                # trade.size > 0 for closed longs (closed by sell), < 0 for closed shorts (closed by buy)
                direction = 'long' if trade.size > 0 else 'short'
                entry_price = trade.price # Use average entry price from trade object

                print(f"[{dt}] NOTIFY TRADE CLOSED - Inferred Direction='{direction}', Avg EntryPrice={entry_price:.5f}", flush=True)

                # Use state tracked during the trade if available, otherwise default
                entry_bar = self.trade_entry_bar if self.trade_entry_bar is not None else (len(self) - 1) # Approx if state missing
                duration = len(self) - entry_bar
                trade_low_during = self.trade_low
                trade_high_during = self.trade_high

                profit_usd = trade.pnlcomm
                profit_pips = 0
                intra_dd = 0
                exit_price_implied = entry_price # Default

                # Ensure volume is non-zero before calculating pips/exit
                trade_volume = abs(trade.size)
                if entry_price != 0 and trade_volume != 0 and self.p.pip_cost != 0:
                    # Calculate implied exit price based on PnL
                    if direction == 'long':
                        # PnL = (Exit - Entry) * Volume * PipValueFactor (usually 1/pip_cost)
                        # Exit = Entry + PnL / (Volume * PipValueFactor)
                        # Assuming PipValueFactor = 1 / pip_cost (needs verification based on broker/asset)
                        exit_price_implied = entry_price + (profit_usd / trade_volume) # Simpler if price is direct quote currency
                        # Let's recalculate pips directly from prices if possible
                        profit_pips = (exit_price_implied - entry_price) / self.p.pip_cost # Check this calculation carefully
                        if trade_low_during is not None:
                             intra_dd = max(0, (entry_price - trade_low_during) / self.p.pip_cost)
                    elif direction == 'short':
                        # PnL = (Entry - Exit) * Volume * PipValueFactor
                        # Exit = Entry - PnL / (Volume * PipValueFactor)
                        exit_price_implied = entry_price - (profit_usd / trade_volume) # Simpler if price is direct quote currency
                        profit_pips = (entry_price - exit_price_implied) / self.p.pip_cost # Check this calculation carefully
                        if trade_high_during is not None:
                             intra_dd = max(0, (trade_high_during - entry_price) / self.p.pip_cost)
                else:
                     print(f"[{dt}] WARNING: Cannot calculate pips/exit price accurately (Entry={entry_price}, Size={trade_volume}, PipCost={self.p.pip_cost})")


                current_balance = self.broker.getvalue()
                # Use trade object datetime if available and reliable
                open_dt_approx = self.trade_entry_dates[-1] if self.trade_entry_dates else "N/A"

                trade_record = {
                    'open_dt': open_dt_approx, # Approximate open time
                    'close_dt': dt,
                    'volume': trade_volume,
                    'pnl': profit_usd,
                    'pips': profit_pips,
                    'duration': duration,
                    'max_dd': intra_dd,
                    'direction': direction # Use inferred direction
                }
                self.trades.append(trade_record)
                print(f"[DEBUG]   TRADE CLOSED ({direction}): Date={dt}, Entry={entry_price:.5f}, Exit=(Implied){exit_price_implied:.5f}, "
                      f"Volume={trade_record['volume']:.2f}, PnL={profit_usd:.2f}, Pips={profit_pips:.2f}, "
                      f"Duration={duration} bars, MaxDD={intra_dd:.2f}, Balance={current_balance:.2f}")

                # --- Reset state AFTER processing the closed trade ---
                # Reset state variables associated with being in a position
                self.current_tp = None
                self.current_sl = None
                self.current_direction = None # Position is now closed
                self.current_volume = None
                self.trade_entry_bar = None
                # High/low are reset in next() when flat is confirmed

        def stop(self):
            # --- Final Summary Calculation ---
            if self.position:
                print("Warning: Position still open at the end of backtest. Closing.")
                self.close() # Ensure position is closed before final summary

            min_balance = min(self.balance_history) if self.balance_history else self.initial_balance
            final_balance = self.broker.getvalue()
            n_trades = len(self.trades)
            n_long_trades = 0
            n_short_trades = 0
            long_trade_percentage = 0.0
            short_trade_percentage = 0.0
            avg_profit_usd = 0.0
            avg_profit_pips = 0.0
            avg_duration = 0.0
            avg_max_dd = 0.0

            if n_trades > 0:
                try:
                    avg_profit_usd = sum(t['pnl'] for t in self.trades) / n_trades
                except ZeroDivisionError: avg_profit_usd = 0.0
                try:
                    avg_profit_pips = sum(t['pips'] for t in self.trades) / n_trades
                except ZeroDivisionError: avg_profit_pips = 0.0
                try:
                    avg_duration = sum(t['duration'] for t in self.trades) / n_trades
                except ZeroDivisionError: avg_duration = 0.0
                try:
                    avg_max_dd = sum(t['max_dd'] for t in self.trades) / n_trades
                except ZeroDivisionError: avg_max_dd = 0.0

                n_long_trades = sum(1 for t in self.trades if t.get('direction') == 'long')
                n_short_trades = n_trades - n_long_trades
                try:
                    long_trade_percentage = (n_long_trades / n_trades) * 100
                    short_trade_percentage = (n_short_trades / n_trades) * 100
                except ZeroDivisionError:
                    long_trade_percentage = 0.0
                    short_trade_percentage = 0.0

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

            # --- Plotting ---
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(self.date_history, self.balance_history)
                plt.title('Balance Over Time')
                plt.xlabel('Date')
                plt.ylabel('Balance (USD)')
                plt.grid(True)
                plt.savefig('balance_plot.png') # Save the plot
                # plt.show() # Optionally display plot
                plt.close() # Close plot to free memory
            except Exception as e:
                print(f"Error generating plot: {e}")

            # --- Save Trades and Summary ---
            try:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv('trades.csv', index=False)
            except Exception as e:
                print(f"Error saving trades CSV: {e}")

            summary_data = {
                "Initial Balance": self.initial_balance,
                "Final Balance": final_balance,
                "Minimum Balance": min_balance,
                "Number of Trades": n_trades,
                "Number Long Trades": n_long_trades,
                "Number Short Trades": n_short_trades,
                "Long Trade Percentage": long_trade_percentage,
                "Short Trade Percentage": short_trade_percentage,
                "Average Profit USD": avg_profit_usd,
                "Average Profit Pips": avg_profit_pips,
                "Average Duration Bars": avg_duration,
                "Average Max Drawdown Pips": avg_max_dd
            }
            try:
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_csv('summary.csv', index=False)
            except Exception as e:
                print(f"Error saving summary CSV: {e}")

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

# Example usage within a Cerebro setup (replace with your actual setup)
if __name__ == '__main__':
    # This is a placeholder for how you might run Cerebro
    # You need to provide your data feed and prediction dataframe loading
    cerebro = bt.Cerebro()

    # --- Dummy Data Feed ---
    # Replace with your actual data loading
    import datetime
    dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
    data = pd.DataFrame({
        'datetime': dates,
        'open': np.random.rand(500) * 10 + 1.1,
        'high': lambda x: x['open'] + np.random.rand(500) * 0.5,
        'low': lambda x: x['open'] - np.random.rand(500) * 0.5,
        'close': lambda x: x['open'] + (np.random.rand(500) - 0.5) * 0.8,
        'volume': np.random.randint(100, 1000, 500)
    })
    data['high'] = data.apply(lambda row: row['open'] + np.random.rand() * 0.001, axis=1)
    data['low'] = data.apply(lambda row: row['open'] - np.random.rand() * 0.001, axis=1)
    data['close'] = data.apply(lambda row: row['open'] + (np.random.rand() - 0.5) * 0.0008, axis=1)
    data = data.set_index('datetime')
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    # --- Dummy Prediction Data ---
    # Replace with your actual prediction loading
    pred_dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
    pred_data = {}
    for i in range(1, 7):
        pred_data[f'Prediction_d_{i}'] = data['close'].shift(-i*24) # Dummy future prices
    pred_df = pd.DataFrame(pred_data, index=pred_dates)
    pred_df = pred_df.fillna(method='ffill').fillna(method='bfill') # Fill NaNs

    # --- Strategy Parameters ---
    # Use defaults or load from config
    strategy_params = {
        'pred_df': pred_df,
        'pip_cost': 0.0001,
        'rel_volume': 0.01,
        'min_order_volume': 1000,
        'max_order_volume': 1000000,
        'leverage': 30,
        'profit_threshold': 50,
        'min_drawdown_pips': 10,
        'tp_multiplier': 1.0,
        'sl_multiplier': 1.0,
        'lower_rr_threshold': 100,
        'upper_rr_threshold': 1000,
        'max_trades_per_5days': 10, # Increased for dummy data
        'num_daily_preds': 6,
        'num_hourly_preds': 0 # Disable hourly for this example
    }

    cerebro.addstrategy(Plugin.HeuristicStrategy, **strategy_params)
    cerebro.broker.set_cash(10000.0)
    cerebro.broker.setcommission(commission=0.0002) # Example commission

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

import datetime
import os
import backtrader as bt
import pandas as pd
import numpy as np

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
        """Evaluates a candidate strategy parameter set using the provided datasets."""
        import os
        import pandas as pd
        import backtrader as bt

        # Unpack candidate parameters
        profit_threshold, tp_multiplier, sl_multiplier, rel_volume, lower_rr, upper_rr = individual

        # Merge predictions
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

        # Ensure predictions have a datetime index
        if merged_df.index.name is None or merged_df.index.name != "DATE_TIME":
            merged_df.index.name = "DATE_TIME"

        # Save merged predictions to a temporary CSV file
        temp_pred_file = "temp_predictions.csv"
        merged_df.reset_index().to_csv(temp_pred_file, index=False)

        # Build the Cerebro backtest
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

        self.trades = trades_list

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
        """

        def __init__(self, *args, **kwargs):
            super().__init__()
            for key, value in kwargs.items():
                setattr(self.params, key, value)

            self.pred_df = pd.read_csv(self.params.pred_file, parse_dates=['DATE_TIME']).set_index('DATE_TIME')

            self.data0 = self.datas[0]
            self.initial_balance = self.broker.getvalue()
            self.trade_entry_dates = []
            self.balance_history = []
            self.date_history = []
            self.trades = []
            self.current_tp = None
            self.current_sl = None
            self.current_direction = None

        def next(self):
            dt = self.data0.datetime.datetime(0)
            print(f"[DEBUG]   next() called at {dt}")

        def compute_size(self, rr):
            return max(self.params.min_order_volume, min(self.params.max_order_volume, self.params.rel_volume * self.broker.getcash()))





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

import datetime
import os
import backtrader as bt
import pandas as pd

class Plugin:
    """
    Heuristic Trading Strategy Plugin for Backtrader.
    
    This plugin wraps a trading strategy that uses pre-computed future predictions
    (both hourly and daily) to decide on trade entries and exits.
    
    It provides the necessary interface for optimization, including:
        - `get_optimizable_params()`
        - `evaluate_candidate()`
    
    It also supports configurable parameters via `plugin_params`, which allows
    the optimizer to tune the strategy dynamically.
    """
    
    # Default parameters for the trading strategy.
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
        'max_trades_per_5days': 3
    }
    
    # Debug variables.
    plugin_debug_vars = [
        'price_file', 'pred_file', 'date_start', 'date_end',
        'pip_cost', 'rel_volume', 'profit_threshold', 'tp_multiplier', 'sl_multiplier'
    ]
    
    def __init__(self):
        self.params = self.plugin_params.copy()
        self.trades = []

    def set_params(self, **kwargs):
        """Updates plugin parameters dynamically."""
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """Returns key parameters for debugging."""
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """Adds plugin debug info to an external dictionary."""
        debug_info.update(self.get_debug_info())
    
    # -------------------------------------------------------------------------
    # Optimization Interface
    # -------------------------------------------------------------------------
    def get_optimizable_params(self):
        """
        Returns a list of tuples for each optimizable parameter: (name, lower_bound, upper_bound).
        These parameters are used by the optimizer to search for the best strategy configuration.
        """
        return [
            ("profit_threshold", 1, 20),
            ("tp_multiplier", 0.8, 1.2),
            ("sl_multiplier", 1.5, 3.0),
            ("rel_volume", 0.01, 0.1),
            ("lower_rr_threshold", 0.3, 1.0),
            ("upper_rr_threshold", 1.5, 3.0)
        ]
    
    def evaluate_candidate(self, individual, base_data, hourly_predictions, daily_predictions, config):
        """
        Evaluates a candidate parameter set by running a backtest using Backtrader.
        
        Args:
            individual (list): Candidate parameter values.
            base_data (pd.DataFrame): Historical price data.
            hourly_predictions (pd.DataFrame): Hourly predictions data.
            daily_predictions (pd.DataFrame): Daily predictions data.
            config (dict): Additional configuration.
        
        Returns:
            tuple: (profit) representing the fitness score.
        """
        # Unpack candidate parameters.
        profit_threshold, tp_multiplier, sl_multiplier, rel_volume, lower_rr, upper_rr = individual
        
        # Ensure daily_predictions has a proper DATE_TIME index before saving.
        temp_pred_file = "temp_predictions.csv"
        if daily_predictions.index.name is None:
            daily_predictions = daily_predictions.copy()
            daily_predictions.index.name = "DATE_TIME"
        daily_predictions.reset_index().to_csv(temp_pred_file, index=False)
        
        # Determine start and end dates from the base_data index.
        if hasattr(base_data.index, 'min') and hasattr(base_data.index, 'max'):
            date_start = base_data.index.min().to_pydatetime()
            date_end = base_data.index.max().to_pydatetime()
        else:
            date_start = self.params['date_start']
            date_end = self.params['date_end']
        
        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            HeuristicStrategy,
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
            cerebro.run()
        except Exception as e:
            print("Error during backtest:", e)
            if os.path.exists(temp_pred_file):
                os.remove(temp_pred_file)
            return (-1e6,)

        final_value = cerebro.broker.getvalue()
        profit = final_value - 10000.0
        print(f"Evaluated candidate {individual} -> Profit: {profit:.2f}")
        
        if os.path.exists(temp_pred_file):
            os.remove(temp_pred_file)

        return (profit,)
    
    # -------------------------------------------------------------------------
    # Dummy methods for compatibility (not applicable for trading strategy)
    # -------------------------------------------------------------------------
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

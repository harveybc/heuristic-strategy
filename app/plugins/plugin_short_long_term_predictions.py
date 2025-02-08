import datetime
import os
import backtrader as bt
import pandas as pd
import random
import logging

# =============================================================================
# Plugin for Heuristic Trading Strategy (Backtrader)
#
# This plugin wraps your HeuristicStrategy (which uses both hourly and daily predictions)
# and exposes the original interface (set_params, get_debug_info, add_debug_info, plugin_params,
# plugin_debug_vars, etc.) as well as the optimization interface required by the optimizer module.
#
# The optimization interface includes:
#    - get_optimizable_params(): Returns a list of tuples (name, lower_bound, upper_bound).
#    - evaluate_candidate(individual, base_data, hourly_predictions, daily_predictions, config):
#         Runs a backtest with the candidate parameter set and returns its profit.
#
# The trading strategy (HeuristicStrategy) is expected to remain exactly the same.
# =============================================================================

class Plugin:
    """
    Heuristic Trading Strategy Plugin for Backtrader.
    
    This plugin wraps a trading strategy (HeuristicStrategy) that uses pre-computed future predictions
    (both hourly and daily) to decide on trade entries and exits. It exposes the standard interface
    (set_params, get_debug_info, add_debug_info, plugin_params, plugin_debug_vars) as well as the
    optimization interface (get_optimizable_params, evaluate_candidate) required by our optimizer module.
    
    Note: Methods related to model building and training (build_model, train, predict, calculate_mse,
    calculate_mae, save, load) are retained for interface compatibility but are not applicable for a trading
    strategy. They simply print a message indicating so.
    """
    
    # Default parameters for the trading strategy.
    plugin_params = {
        'price_file': '../trading-signal/output.csv',     # File path for price data (if needed)
        'pred_file': '../trading-signal/output.csv',        # File path for predictions data
        'date_start': datetime.datetime(2010, 1, 1),
        'date_end': datetime.datetime(2015, 1, 1),
        'pip_cost': 0.00001,         # For EURUSD: 1 pip = 0.00001
        'rel_volume': 0.05,          # Fraction of cash to risk
        'min_order_volume': 10000,   # Minimum order volume
        'max_order_volume': 1000000, # Maximum order volume
        'leverage': 1000,
        'profit_threshold': 5,       # Minimum ideal profit (in pips) required for entry
        'min_drawdown_pips': 10,     # Minimum drawdown (in pips)
        'tp_multiplier': 0.9,        # Multiplier for take-profit level
        'sl_multiplier': 2.0,        # Multiplier for stop-loss level
        'lower_rr_threshold': 0.5,   # RR threshold below which use minimum volume
        'upper_rr_threshold': 2.0,   # RR threshold above which use maximum volume
        'max_trades_per_5days': 3
    }
    
    # Variables for debugging—these keys will be exported if needed.
    plugin_debug_vars = [
        'price_file', 'pred_file', 'date_start', 'date_end', 
        'pip_cost', 'rel_volume', 'profit_threshold', 'tp_multiplier', 'sl_multiplier'
    ]
    
    def __init__(self):
        # Copy the default parameters.
        self.params = self.plugin_params.copy()
        # There is no neural network model in this strategy.
        self.model = None

    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided keyword arguments.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Return a dictionary of debug info from the plugin parameters.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Add the plugin's debug info to an external dictionary.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)
    
    # -------------------------------------------------------------------------
    # Optimization Interface
    # -------------------------------------------------------------------------
    def get_optimizable_params(self):
        """
        Returns a list of tuples for each optimizable parameter: (name, lower_bound, upper_bound).
        These parameters are used by the optimizer to search for the best strategy configuration.
        
        The parameters below can be tuned:
            - profit_threshold: [1, 20]
            - tp_multiplier: [0.8, 1.2]
            - sl_multiplier: [1.5, 3.0]
            - rel_volume: [0.01, 0.1]
            - lower_rr_threshold: [0.3, 1.0]
            - upper_rr_threshold: [1.5, 3.0]
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
        Evaluate a candidate parameter set by running a backtest using Backtrader.
        
        Args:
            individual (list): List of parameter values (in the order defined by get_optimizable_params).
            base_data (pd.DataFrame): Actual price data with a datetime index.
            hourly_predictions (pd.DataFrame): Hourly predictions data (not used in this evaluation).
            daily_predictions (pd.DataFrame): Daily predictions data used by the strategy.
            config (dict): Additional configuration.
            
        Returns:
            tuple: A tuple containing the profit (final broker value minus initial cash).
        """
        # Unpack candidate parameters.
        profit_threshold, tp_multiplier, sl_multiplier, rel_volume, lower_rr, upper_rr = individual
        
        # Write daily_predictions to a temporary CSV file (the strategy expects a pred_file).
        temp_pred_file = "temp_predictions.csv"
        # Ensure the index is named "DATE_TIME"
        if daily_predictions.index.name is None:
            daily_predictions = daily_predictions.copy()
            daily_predictions.index.name = "DATE_TIME"
        daily_predictions.reset_index().to_csv(temp_pred_file, index=False)
        
        # Determine the backtest date range from base_data's datetime index.
        if hasattr(base_data.index, 'min') and hasattr(base_data.index, 'max'):
            date_start = base_data.index.min().to_pydatetime()
            date_end = base_data.index.max().to_pydatetime()
        else:
            date_start = self.params['date_start']
            date_end = self.params['date_end']
        
        # Set up Cerebro and add the strategy.
        cerebro = bt.Cerebro()
        # Import your HeuristicStrategy (unchanged) here.
        from heuristic_strategy import HeuristicStrategy
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
        
        # Create a data feed from base_data.
        data_feed = bt.feeds.PandasData(dataname=base_data)
        cerebro.adddata(data_feed)
        
        # Set initial cash.
        cerebro.broker.setcash(10000.0)
        # Do NOT add a sizer so that the strategy uses its computed order sizing.
        try:
            cerebro.run()
        except Exception as e:
            print("Error during backtest:", e)
            # Return a very poor fitness if backtest fails.
            if os.path.exists(temp_pred_file):
                os.remove(temp_pred_file)
            return (-1e6,)
        
        final_value = cerebro.broker.getvalue()
        profit = final_value - 10000.0
        print(f"Evaluated candidate {individual} -> Profit: {profit:.2f}")
        
        # Clean up the temporary prediction file.
        if os.path.exists(temp_pred_file):
            os.remove(temp_pred_file)
        
        return (profit,)
    
    # -------------------------------------------------------------------------
    # Dummy methods for interface compatibility (not applicable for trading strategy)
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

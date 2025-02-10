import datetime
import os
import backtrader as bt
import pandas as pd


class Plugin:
    """
    Plugin for Heuristic Trading Strategy.

    - Implements a Backtrader strategy using precomputed future predictions.
    - Provides the required plugin interface (`plugin_params`, `set_params`, `get_optimizable_params`, etc.).
    - Integrates seamlessly with the optimizer and pipeline.

    This plugin embeds `HeuristicStrategy`, which follows the same logic as the provided strategy.
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
        """Run the strategy with candidate parameters and return performance metrics."""
        profit_threshold, tp_multiplier, sl_multiplier, rel_volume, lower_rr, upper_rr = individual

        temp_pred_file = "temp_predictions.csv"
        if daily_predictions.index.name is None:
            daily_predictions = daily_predictions.copy()
            daily_predictions.index.name = "DATE_TIME"
        daily_predictions.reset_index().to_csv(temp_pred_file, index=False)

        date_start = base_data.index.min().to_pydatetime() if hasattr(base_data.index, 'min') else self.params['date_start']
        date_end = base_data.index.max().to_pydatetime() if hasattr(base_data.index, 'max') else self.params['date_end']

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
            cerebro.run()
        except Exception as e:
            print("Error during backtest:", e)
            os.remove(temp_pred_file)
            return (-1e6,)

        final_value = cerebro.broker.getvalue()
        profit = final_value - 10000.0
        print(f"Evaluated candidate {individual} -> Profit: {profit:.2f}")

        os.remove(temp_pred_file)
        return (profit,)

    class HeuristicStrategy(bt.Strategy):
        """
        Forex trading strategy using precomputed predictions to optimize risk-reward ratio.
        """

        params = Plugin.plugin_params.copy()

        def __init__(self):
            self.pred_df = pd.read_csv(self.p.pred_file, parse_dates=['DATE_TIME'])
            self.pred_df = self.pred_df[
                (self.pred_df['DATE_TIME'] >= self.p.date_start) & (self.pred_df['DATE_TIME'] <= self.p.date_end)
            ]
            self.pred_df['DATE_TIME'] = self.pred_df['DATE_TIME'].apply(lambda dt: dt.replace(minute=0, second=0, microsecond=0))
            self.pred_df.set_index('DATE_TIME', inplace=True)

            self.num_hourly_preds = len([col for col in self.pred_df.columns if col.startswith('Prediction_h_')])
            self.num_daily_preds = len([col for col in self.pred_df.columns if col.startswith('Prediction_d_')])

            self.data0 = self.datas[0]
            self.initial_balance = self.broker.getvalue()
            self.trade_entry_dates = []
            self.trades = []

        def next(self):
            dt = self.data0.datetime.datetime(0)
            dt_hour = dt.replace(minute=0, second=0, microsecond=0)
            current_price = self.data0.close[0]

            if dt_hour not in self.pred_df.index:
                return

            row = self.pred_df.loc[dt_hour]
            try:
                daily_preds = [row[f'Prediction_d_{i}'] for i in range(1, self.num_daily_preds+1)]
            except KeyError:
                return

            ideal_profit_pips_buy = (max(daily_preds) - current_price) / self.p.pip_cost
            ideal_drawdown_pips_buy = max((current_price - min(daily_preds)) / self.p.pip_cost, self.p.min_drawdown_pips)
            rr_buy = ideal_profit_pips_buy / ideal_drawdown_pips_buy if ideal_drawdown_pips_buy > 0 else 0
            tp_buy = current_price + self.p.tp_multiplier * ideal_profit_pips_buy * self.p.pip_cost
            sl_buy = current_price - self.p.sl_multiplier * ideal_drawdown_pips_buy * self.p.pip_cost

            ideal_profit_pips_sell = (current_price - min(daily_preds)) / self.p.pip_cost
            ideal_drawdown_pips_sell = max((max(daily_preds) - current_price) / self.p.pip_cost, self.p.min_drawdown_pips)
            rr_sell = ideal_profit_pips_sell / ideal_drawdown_pips_sell if ideal_drawdown_pips_sell > 0 else 0
            tp_sell = current_price - self.p.tp_multiplier * ideal_profit_pips_sell * self.p.pip_cost
            sl_sell = current_price + self.p.sl_multiplier * ideal_drawdown_pips_sell * self.p.pip_cost

            long_signal = ideal_profit_pips_buy >= self.p.profit_threshold
            short_signal = ideal_profit_pips_sell >= self.p.profit_threshold

            if long_signal and (rr_buy >= rr_sell):
                self.buy(size=self.compute_size(rr_buy))
            elif short_signal and (rr_sell > rr_buy):
                self.sell(size=self.compute_size(rr_sell))

        def compute_size(self, rr):
            """Compute order size by linear interpolation between min and max volumes based on RR."""
            size = self.p.min_order_volume + ((rr - self.p.lower_rr_threshold) /
                (self.p.upper_rr_threshold - self.p.lower_rr_threshold)) * (
                self.p.max_order_volume - self.p.min_order_volume)
            return min(size, self.broker.getcash() * self.p.rel_volume * self.p.leverage)


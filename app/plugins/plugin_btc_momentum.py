"""
Plugin for BTC Momentum Trading Strategy.

Simple trend-following strategy for BTC/USD using EMA crossover + RSI filter.
Designed as the primary strategy (strategy_1) for BTC in Path A experiments.

RSI is the only feature with a lagged causal link to BTC 4h returns
(PCMCI+ Stage II-0.5: MCI=-0.2459 at lag-6, p<0.001).

Strategy logic:
  - Long when: fast EMA > slow EMA AND RSI < overbought threshold
  - Short when: fast EMA < slow EMA AND RSI > oversold threshold
  - Exit: ATR-based TP/SL or EMA cross reversal

Optimizable params: ema_fast, ema_slow, rsi_period, rsi_overbought,
    rsi_oversold, atr_period, atr_tp_multiplier, atr_sl_multiplier
"""

import os
import numpy as np
import pandas as pd
import backtrader as bt

_QUIET = os.environ.get("STRATEGY_QUIET", "0") == "1"


class Plugin:
    """BTC Momentum Trading Strategy Plugin."""

    plugin_params = {
        'pip_cost': 1.0,           # BTC price unit (1 USD for BTC)
        'rel_volume': 0.10,
        'min_order_volume': 0.001,  # Min BTC order size
        'max_order_volume': 10.0,   # Max BTC order size
        'leverage': 1,              # No leverage for BTC by default
        # EMA crossover
        'ema_fast': 12,
        'ema_slow': 26,
        # RSI filter (causal feature from II-0.5)
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        # ATR-based TP/SL
        'atr_period': 14,
        'atr_tp_multiplier': 2.5,
        'atr_sl_multiplier': 1.5,
        # Position management
        'cooldown_bars': 6,         # Min bars between trades
        'allow_short': True,        # Allow short positions
        # Trading costs (BTC exchange typical)
        'spread_pct': 0.001,        # 0.1% spread
        'commission_pct': 0.001,    # 0.1% taker fee
        'slippage_pct': 0.0005,     # 0.05% slippage
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.trades = []

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {key: self.params[key] for key in self.params}

    def get_optimizable_params(self):
        return [
            ("ema_fast", 5, 20),
            ("ema_slow", 20, 60),
            ("rsi_period", 7, 28),
            ("rsi_overbought", 60, 85),
            ("rsi_oversold", 15, 40),
            ("atr_period", 7, 28),
            ("atr_tp_multiplier", 1.0, 5.0),
            ("atr_sl_multiplier", 0.5, 3.0),
        ]

    def evaluate_candidate(self, individual, base_data, hourly_predictions,
                           daily_predictions, config):
        (ema_fast, ema_slow, rsi_period, rsi_overbought, rsi_oversold,
         atr_period, atr_tp_mult, atr_sl_mult) = individual

        # Ensure ema_fast < ema_slow
        if ema_fast >= ema_slow:
            ema_fast, ema_slow = ema_slow - 1, ema_slow

        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            self.BTCMomentumStrategy,
            ema_fast=int(ema_fast),
            ema_slow=int(ema_slow),
            rsi_period=int(rsi_period),
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            atr_period=int(atr_period),
            atr_tp_multiplier=atr_tp_mult,
            atr_sl_multiplier=atr_sl_mult,
            cooldown_bars=self.params['cooldown_bars'],
            allow_short=self.params['allow_short'],
        )

        data_feed = bt.feeds.PandasData(dataname=base_data)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(10000.0)
        cerebro.broker.set_coc(True)

        # BTC commission as percentage
        total_cost_pct = (self.params['commission_pct'] +
                          self.params['spread_pct'] / 2 +
                          self.params['slippage_pct'])
        cerebro.broker.setcommission(commission=total_cost_pct, mult=1.0)

        try:
            runresult = cerebro.run()
        except Exception as e:
            if not _QUIET:
                print(f"Error during backtest (btc_momentum): {e}")
            return (-1e6, {"num_trades": 0, "win_pct": 0,
                           "max_dd": 0, "sharpe": 0})

        final_value = cerebro.broker.getvalue()
        profit = final_value - 10000.0
        if not _QUIET:
            print(f"Evaluated {[round(x,2) for x in individual]} -> Profit: {profit:.2f}")

        strat_instance = runresult[0]
        trades_list = getattr(strat_instance, "trades", [])
        self.trades = trades_list

        num_trades = len(trades_list)
        stats = {"num_trades": num_trades, "win_pct": 0,
                 "max_dd": 0, "sharpe": 0}
        if num_trades > 0:
            wins = sum(1 for tr in trades_list if tr['pnl'] > 0)
            stats["win_pct"] = (wins / num_trades) * 100
            pnl_list = [tr['pnl'] for tr in trades_list]
            equity = [10000.0]
            for p in pnl_list:
                equity.append(equity[-1] + p)
            equity = np.array(equity)
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
            stats["max_dd"] = float(np.max(dd))
            std_pnl = np.std(pnl_list) if num_trades > 1 else 0
            stats["sharpe"] = float(np.mean(pnl_list) / std_pnl) if std_pnl > 0 else 0

        return (profit, stats)

    # =================================================================
    # Inner Backtrader Strategy — BTC Momentum
    # =================================================================
    class BTCMomentumStrategy(bt.Strategy):
        """
        BTC trend-following strategy using EMA crossover + RSI filter.
        """

        params = dict(
            ema_fast=12,
            ema_slow=26,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            atr_period=14,
            atr_tp_multiplier=2.5,
            atr_sl_multiplier=1.5,
            cooldown_bars=6,
            allow_short=True,
        )

        def __init__(self):
            self.ema_fast = bt.indicators.EMA(self.data.close,
                                               period=self.p.ema_fast)
            self.ema_slow = bt.indicators.EMA(self.data.close,
                                               period=self.p.ema_slow)
            self.rsi = bt.indicators.RSI(self.data.close,
                                          period=self.p.rsi_period)
            self.atr = bt.indicators.ATR(self.data,
                                          period=self.p.atr_period)
            self.crossover = bt.indicators.CrossOver(self.ema_fast,
                                                      self.ema_slow)
            self.trades = []
            self._entry_price = None
            self._entry_bar = None
            self._tp_price = None
            self._sl_price = None
            self._direction = 0  # 1=long, -1=short
            self._last_trade_bar = -999

        def next(self):
            bars_since_last = len(self) - self._last_trade_bar

            if self.position:
                # Check TP/SL
                if self._direction == 1:
                    if self.data.high[0] >= self._tp_price:
                        self._close_trade(self._tp_price, 'TP')
                        return
                    if self.data.low[0] <= self._sl_price:
                        self._close_trade(self._sl_price, 'SL')
                        return
                    # Exit on EMA cross reversal
                    if self.ema_fast[0] < self.ema_slow[0]:
                        self._close_trade(self.data.close[0], 'SIGNAL')
                        return
                elif self._direction == -1:
                    if self.data.low[0] <= self._tp_price:
                        self._close_trade(self._tp_price, 'TP')
                        return
                    if self.data.high[0] >= self._sl_price:
                        self._close_trade(self._sl_price, 'SL')
                        return
                    if self.ema_fast[0] > self.ema_slow[0]:
                        self._close_trade(self.data.close[0], 'SIGNAL')
                        return
            else:
                # No position — check entry
                if bars_since_last < self.p.cooldown_bars:
                    return

                atr_val = self.atr[0]
                if atr_val <= 0:
                    return

                # Long entry: EMA fast > slow AND RSI not overbought
                if (self.ema_fast[0] > self.ema_slow[0] and
                        self.rsi[0] < self.p.rsi_overbought):
                    self._entry_price = self.data.close[0]
                    self._tp_price = self._entry_price + atr_val * self.p.atr_tp_multiplier
                    self._sl_price = self._entry_price - atr_val * self.p.atr_sl_multiplier
                    self._direction = 1
                    self._entry_bar = len(self)
                    size = max(0.001, 10000 * 0.02 / (atr_val * self.p.atr_sl_multiplier))
                    self.buy(size=round(size, 4))

                # Short entry: EMA fast < slow AND RSI not oversold
                elif (self.p.allow_short and
                      self.ema_fast[0] < self.ema_slow[0] and
                      self.rsi[0] > self.p.rsi_oversold):
                    self._entry_price = self.data.close[0]
                    self._tp_price = self._entry_price - atr_val * self.p.atr_tp_multiplier
                    self._sl_price = self._entry_price + atr_val * self.p.atr_sl_multiplier
                    self._direction = -1
                    self._entry_bar = len(self)
                    size = max(0.001, 10000 * 0.02 / (atr_val * self.p.atr_sl_multiplier))
                    self.sell(size=round(size, 4))

        def _close_trade(self, exit_price, reason):
            pnl = (exit_price - self._entry_price) * self._direction
            # Scale PnL by approximate position value
            if self.position.size != 0:
                pnl = pnl * abs(self.position.size)

            peak_equity = 10000 + max(0, pnl)
            max_dd = abs(min(0, pnl)) / peak_equity if peak_equity > 0 else 0

            self.trades.append({
                'entry_bar': self._entry_bar,
                'exit_bar': len(self),
                'entry_price': self._entry_price,
                'exit_price': exit_price,
                'direction': 'long' if self._direction == 1 else 'short',
                'pnl': pnl,
                'reason': reason,
                'max_dd': max_dd,
                'bars_held': len(self) - self._entry_bar,
            })

            self._last_trade_bar = len(self)
            self.close()
            self._direction = 0
            self._entry_price = None
            self._tp_price = None
            self._sl_price = None

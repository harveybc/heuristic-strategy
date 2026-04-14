"""
Plugin for Direction + ATR-based Heuristic Trading Strategy.

Uses the Prediction Provider API with direction classification models:
  - Entry: direction_long model → P(up) > threshold → buy, P(up) < (1-threshold) → sell
  - Exit:  direction_short model → opposing direction → close position
  - TP/SL: ATR-based (adaptive to volatility) instead of fixed pip thresholds

This is the "Path A" strategy: simple directional prediction with
ATR-based dynamic stop levels.

Configuration:
    plugin: direction_atr
    prediction_source: API
    pp_api_url: http://127.0.0.1:8000
"""

import datetime
import os
import backtrader as bt
import pandas as pd
import numpy as np
import os as _os
_QUIET = _os.environ.get("STRATEGY_QUIET", "0") == "1"


class Plugin:
    """
    Direction + ATR Plugin for Heuristic Trading Strategy.

    Uses ATR (Average True Range) for adaptive TP/SL sizing and
    direction classification for entry/exit signals.
    """

    plugin_params = {
        'pip_cost': 0.00001,
        'rel_volume': 0.10,
        'min_order_volume': 10000,
        'max_order_volume': 1000000,
        'leverage': 100,
        # ATR-based TP/SL multipliers
        'atr_period': 14,
        'atr_tp_multiplier': 2.0,
        'atr_sl_multiplier': 1.5,
        # Trade management
        'max_trades_per_5days': 5,
        'exit_enabled': True,
        # Trading costs
        'spread_pips': 30.0,
        'commission_per_lot': 10.0,
        'slippage_pips': 10.0,
        'swap_per_lot_per_day': 15.0,
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
            ("atr_period", 7, 28),
            ("atr_tp_multiplier", 1.0, 4.0),
            ("atr_sl_multiplier", 0.5, 3.0),
        ]

    def evaluate_candidate(self, individual, base_data, hourly_predictions,
                           daily_predictions, config):
        atr_period, atr_tp_mult, atr_sl_mult = individual

        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            self.DirectionATRStrategy,
            pip_cost=self.params['pip_cost'],
            rel_volume=self.params['rel_volume'],
            min_order_volume=self.params['min_order_volume'],
            max_order_volume=self.params['max_order_volume'],
            leverage=self.params['leverage'],
            atr_period=int(atr_period),
            atr_tp_multiplier=atr_tp_mult,
            atr_sl_multiplier=atr_sl_mult,
            max_trades_per_5days=self.params['max_trades_per_5days'],
            exit_enabled=self.params['exit_enabled'],
            swap_per_lot_per_day=self.params['swap_per_lot_per_day'],
            pp_api_url=config.get("pp_api_url", "http://127.0.0.1:8000"),
            pp_timeout=float(config.get("pp_timeout", 5.0)),
            spread_pips=self.params['spread_pips'],
            commission_per_lot=self.params['commission_per_lot'],
            slippage_pips=self.params['slippage_pips'],
        )

        data_feed = bt.feeds.PandasData(dataname=base_data)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(10000.0)
        cerebro.broker.set_coc(True)

        spread_cost = self.params['spread_pips'] * self.params['pip_cost']
        slippage_cost = self.params['slippage_pips'] * self.params['pip_cost']
        total_spread = spread_cost + slippage_cost
        commission_per_unit = self.params['commission_per_lot'] / 100000.0
        # Forex margin: 1/leverage per unit (e.g. 100:1 → $0.01 per unit)
        forex_margin = 1.0 / self.params['leverage']
        cerebro.broker.setcommission(
            commission=commission_per_unit, margin=forex_margin, mult=1.0
        )
        cerebro.broker.set_slippage_fixed(
            total_spread / 2.0, slip_open=True, slip_limit=True
        )

        try:
            runresult = cerebro.run()
        except Exception as e:
            if not _QUIET:
                print("Error during backtest (direction ATR):", e)
            return (-1e6, {"num_trades": 0, "win_pct": 0,
                           "max_dd": 0, "sharpe": 0})

        final_value = cerebro.broker.getvalue()
        profit = final_value - 10000.0
        if not _QUIET:
            print(f"Evaluated candidate {individual} -> Profit: {profit:.2f}")

        strat_instance = runresult[0]
        trades_list = getattr(strat_instance, "trades", [])
        self.trades = trades_list

        num_trades = len(trades_list)
        stats = {"num_trades": num_trades, "win_pct": 0,
                 "max_dd": 0, "sharpe": 0}
        if num_trades > 0:
            wins = sum(1 for tr in trades_list if tr['pnl'] > 0)
            stats["win_pct"] = (wins / num_trades) * 100
            stats["max_dd"] = max(tr['max_dd'] for tr in trades_list)
            profits = [tr['pnl'] for tr in trades_list]
            std_profit = np.std(profits) if num_trades > 1 else 0
            stats["sharpe"] = (profit / std_profit) if std_profit > 0 else 0

        if not _QUIET:
            print(f"[EVALUATE] Profit: {profit:.2f}, "
                  f"Trades: {num_trades}, Win%: {stats['win_pct']:.1f}")

        return (profit, stats)

    # =====================================================================
    # Inner Backtrader Strategy — Direction + ATR
    # =====================================================================
    class DirectionATRStrategy(bt.Strategy):
        """
        Forex strategy using direction classification + ATR stops.

        Entry:  direction_long model → P(up) > threshold → buy
                                       P(up) < (1-threshold) → sell
        Exit:   TP/SL from ATR, early exit from direction_short model
        """

        params = dict(
            pip_cost=0.00001,
            rel_volume=0.10,
            min_order_volume=10000,
            max_order_volume=1000000,
            leverage=100,
            atr_period=14,
            atr_tp_multiplier=2.0,
            atr_sl_multiplier=1.5,
            max_trades_per_5days=5,
            exit_enabled=True,
            swap_per_lot_per_day=15.0,
            pp_api_url='http://127.0.0.1:8000',
            pp_timeout=5.0,
            spread_pips=30.0,
            commission_per_lot=10.0,
            slippage_pips=10.0,
        )

        def __init__(self):
            super().__init__()
            from app.prediction_client import ApiPredictionSource
            self._pred_source = ApiPredictionSource(
                self.p.pp_api_url, self.p.pp_timeout
            )

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
            self.order_entry_price = None
            self.trade_entry_bar = None
            self.current_volume = None

            # ATR indicator
            self.atr = bt.indicators.ATR(
                self.data0, period=self.p.atr_period
            )

        def next(self):
            dt = self.data0.datetime.datetime(0)
            dt_hour = dt.replace(minute=0, second=0, microsecond=0)
            current_price = self.data0.close[0]

            self.balance_history.append(self.broker.getvalue())
            self.date_history.append(dt)

            # ── IN POSITION: check TP/SL ──
            if self.position:
                # Friday close
                if dt.weekday() == 4 and dt.hour >= 20:
                    self.close()
                    return

                bar_high = self.data0.high[0]
                bar_low = self.data0.low[0]
                bar_close = self.data0.close[0]

                if self.current_direction == 'buy':
                    if self.trade_low is None or bar_low < self.trade_low:
                        self.trade_low = bar_low
                    if bar_low <= self.current_sl:
                        self.close()
                        return
                    if bar_close >= self.current_tp:
                        self.close()
                        return
                elif self.current_direction == 'sell':
                    if self.trade_high is None or bar_high > self.trade_high:
                        self.trade_high = bar_high
                    if bar_high >= self.current_sl:
                        self.close()
                        return
                    if bar_close <= self.current_tp:
                        self.close()
                        return

                # Early exit via direction_short model
                if self.p.exit_enabled:
                    exit_pred = self._pred_source.get_exit_prediction(
                        dt_hour,
                        direction=self.current_direction,
                        tp_price=self.current_tp,
                        sl_price=self.current_sl,
                    )
                    if (exit_pred["available"]
                            and exit_pred.get("exit_binary", 1) == 0):
                        if not _QUIET:
                            print(f"[EARLY_STOP] {dt} — direction_short "
                                  f"opposes {self.current_direction}")
                        self.close()
                        return

                return

            # ── NO POSITION ──
            self.trade_low = current_price
            self.trade_high = current_price

            # No new entries on Friday
            if dt.weekday() == 4:
                return

            # Wait for ATR warmup
            if len(self) <= self.p.atr_period:
                return

            # Enforce trade frequency
            recent = [d for d in self.trade_entry_dates
                      if (dt - d).days < 5]
            if len(recent) >= self.p.max_trades_per_5days:
                return

            # ATR-based TP/SL distances
            current_atr = self.atr[0]
            if current_atr <= 0:
                return

            tp_distance = current_atr * self.p.atr_tp_multiplier
            sl_distance = current_atr * self.p.atr_sl_multiplier

            tp_pips = tp_distance / self.p.pip_cost
            sl_pips = sl_distance / self.p.pip_cost

            # Ask PP for direction prediction
            entry_pred = self._pred_source.get_entry_prediction(
                dt_hour, tp_pips=tp_pips, sl_pips=sl_pips,
                spread_pips=self.p.spread_pips,
                commission_per_lot=self.p.commission_per_lot,
                slippage_pips=self.p.slippage_pips,
            )
            if not entry_pred["available"]:
                return

            buy_bin = entry_pred.get("buy_entry_binary", 0)
            sell_bin = entry_pred.get("sell_entry_binary", 0)

            if buy_bin == 1 and sell_bin == 0:
                signal = 'buy'
            elif sell_bin == 1 and buy_bin == 0:
                signal = 'sell'
            elif buy_bin == 1 and sell_bin == 1:
                signal = 'buy'
            else:
                return

            # Compute absolute TP/SL from ATR
            if signal == 'buy':
                chosen_tp = current_price + tp_distance
                chosen_sl = current_price - sl_distance
            else:
                chosen_tp = current_price - tp_distance
                chosen_sl = current_price + sl_distance

            # Position sizing
            bars_remaining = entry_pred.get("bars_remaining", 120)
            confidence = entry_pred.get(
                "buy_confidence" if signal == 'buy' else "sell_confidence",
                1.0
            )
            order_size = self._compute_size(bars_remaining, confidence)
            if order_size <= 0:
                return

            self.trade_entry_dates.append(dt)
            self.trade_entry_bar = len(self)
            self.current_volume = order_size
            self.current_tp = chosen_tp
            self.current_sl = chosen_sl

            if signal == 'buy':
                self.buy(size=order_size)
                self.current_direction = 'buy'
            else:
                self.sell(size=order_size)
                self.current_direction = 'sell'

        def _compute_size(self, bars_remaining, confidence=1.0):
            min_vol = self.p.min_order_volume
            max_vol = self.p.max_order_volume
            fast_bars = 12
            slow_bars = 48

            if bars_remaining <= fast_bars:
                size = max_vol
            elif bars_remaining >= slow_bars:
                size = min_vol
            else:
                frac = ((bars_remaining - fast_bars)
                        / (slow_bars - fast_bars))
                size = max_vol - frac * (max_vol - min_vol)

            cash = self.broker.getcash()
            max_from_cash = cash * self.p.rel_volume * self.p.leverage
            size = min(size, max_from_cash)
            size *= max(0.0, min(1.0, confidence))
            return size

        def notify_order(self, order):
            if order.status in [order.Completed]:
                self.order_entry_price = order.executed.price
                self.order_direction = (
                    'buy' if order.isbuy() else 'sell'
                )

        def notify_trade(self, trade):
            if trade.isclosed:
                duration = len(self) - (
                    self.trade_entry_bar
                    if self.trade_entry_bar is not None else 0
                )
                dt = self.data0.datetime.datetime(0)
                entry_price = (self.order_entry_price
                               if self.order_entry_price else 0)
                exit_price = trade.price
                profit_usd = trade.pnlcomm
                overnight_days = max(0, duration / 24.0)
                volume = (self.current_volume
                          if self.current_volume else 0)
                lots = volume / 100000.0
                swap_cost = (overnight_days * lots
                             * self.p.swap_per_lot_per_day)
                profit_usd -= swap_cost
                direction = self.current_direction

                if direction == 'buy':
                    profit_pips = ((exit_price - entry_price)
                                   / self.p.pip_cost)
                    intra_dd = ((entry_price - self.trade_low)
                                / self.p.pip_cost
                                if self.trade_low else 0)
                elif direction == 'sell':
                    profit_pips = ((entry_price - exit_price)
                                   / self.p.pip_cost)
                    intra_dd = ((self.trade_high - entry_price)
                                / self.p.pip_cost
                                if self.trade_high else 0)
                else:
                    profit_pips = 0
                    intra_dd = 0

                current_balance = self.broker.getvalue()
                open_dt = (self.trade_entry_dates[-1]
                           if self.trade_entry_dates else "N/A")
                trade_record = {
                    'open_dt': open_dt,
                    'close_dt': dt,
                    'volume': volume,
                    'pnl': profit_usd,
                    'pnl_pips': profit_pips,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'max_dd': intra_dd,
                    'duration_bars': duration,
                    'balance': current_balance,
                }
                self.trades.append(trade_record)

                if not _QUIET:
                    result = "WIN" if profit_usd > 0 else "LOSS"
                    print(f"[{result}] {direction} "
                          f"{entry_price:.5f}→{exit_price:.5f} "
                          f"PnL: {profit_pips:.0f} pips "
                          f"({profit_usd:.2f} USD) "
                          f"Duration: {duration} bars "
                          f"Balance: {current_balance:.2f}")

                self.current_direction = None
                self.current_tp = None
                self.current_sl = None
                self.trade_low = None
                self.trade_high = None
                self.order_entry_price = None
                self.trade_entry_bar = None
                self.current_volume = None

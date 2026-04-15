"""
Plugin for Regime-Adaptive Heuristic Trading Strategy.

Classifies each bar into a market regime using technical indicators
(ADX, DI, ATR percentile, BB width, RSI, EMA alignment) then applies
the appropriate strategy for that regime:

  Regime 1 (HIGH_VOL_BEARISH_FADING) → buy_reversal  (oversold bounce)
  Regime 2 (STRONG_DOWNTREND)        → sell_trend     (follow the trend)
  Regime 3 (STRONG_UPTREND)          → sell_exhaustion (take profit / short)
  Regime 4 (MILD_RANGE)              → flat           (no trade)
  Regime 5 (LOW_VOL_BEARISH_PULLBACK)→ buy_meanrevert (dip buy in uptrend)
  Regime 6 (LOW_VOL_BULLISH_DRIFT)   → buy_trend      (ride the drift)

All thresholds are optimizable via NEAT or genetic algorithm.
No ML predictions needed — purely reactive to current market state.
"""

import datetime
import os
import backtrader as bt
import pandas as pd
import numpy as np
import os as _os
_QUIET = _os.environ.get("STRATEGY_QUIET", "0") == "1"


REGIME_NAMES = {
    1: "HIGH_VOL_BEARISH_FADING",
    2: "STRONG_DOWNTREND",
    3: "STRONG_UPTREND",
    4: "MILD_RANGE",
    5: "LOW_VOL_BEARISH_PULLBACK",
    6: "LOW_VOL_BULLISH_DRIFT",
}


class Plugin:
    """Regime-Adaptive Trading Strategy Plugin."""

    plugin_params = {
        'pip_cost': 0.00001,
        'rel_volume': 0.10,
        'min_order_volume': 10000,
        'max_order_volume': 1000000,
        'leverage': 100,
        # ATR-based TP/SL
        'atr_period': 14,
        'atr_tp_multiplier': 2.5,
        'atr_sl_multiplier': 1.5,
        # Regime detection thresholds
        'adx_strong': 35.0,
        'adx_mild': 25.0,
        'di_strong': 15.0,
        'di_mild': 5.0,
        'atr_pct_high': 0.65,
        'atr_pct_low': 0.35,
        'rsi_overbought': 65.0,
        'rsi_oversold': 40.0,
        # Per-regime action toggles (1=active, 0=skip)
        'regime_1_active': 1,   # HIGH_VOL_BEARISH_FADING → buy
        'regime_2_active': 0,   # STRONG_DOWNTREND → sell (disabled: losing regime)
        'regime_3_active': 0,   # STRONG_UPTREND → sell (disabled by default: weak edge)
        'regime_4_active': 0,   # MILD_RANGE → flat
        'regime_5_active': 1,   # LOW_VOL_BEARISH_PULLBACK → buy
        'regime_6_active': 1,   # LOW_VOL_BULLISH_DRIFT → buy
        # Entry filters
        'entry_on_transition_only': True,   # Only enter when regime CHANGES
        'transition_window': 8,             # hourly bars after regime change to allow entry
        'cooldown_bars': 12,                # min hourly bars between trades
        'rsi_confirm': True,                # require RSI confirmation
        'bb_confirm': True,                 # require BB position confirmation
        # Trade management
        'max_trades_per_5days': 3,
        'exit_on_regime_change': True,
        # Trading costs
        'spread_pips': 15.0,
        'commission_per_lot': 7.0,
        'slippage_pips': 5.0,
        'swap_per_lot_per_day': 10.0,
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
            ("atr_tp_multiplier", 1.5, 5.0),
            ("atr_sl_multiplier", 0.8, 3.0),
            ("adx_strong", 25.0, 50.0),
            ("di_strong", 8.0, 25.0),
            ("di_mild", 2.0, 12.0),
            ("atr_pct_high", 0.5, 0.85),
            ("atr_pct_low", 0.15, 0.45),
        ]

    def evaluate_candidate(self, individual, base_data, hourly_predictions,
                           daily_predictions, config):
        (atr_period, atr_tp_mult, atr_sl_mult,
         adx_strong, di_strong, di_mild,
         atr_pct_high, atr_pct_low) = individual

        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            self.RegimeAdaptiveStrategy,
            pip_cost=self.params['pip_cost'],
            rel_volume=self.params['rel_volume'],
            min_order_volume=self.params['min_order_volume'],
            max_order_volume=self.params['max_order_volume'],
            leverage=self.params['leverage'],
            atr_period=int(atr_period),
            atr_tp_multiplier=atr_tp_mult,
            atr_sl_multiplier=atr_sl_mult,
            adx_strong=adx_strong,
            adx_mild=self.params['adx_mild'],
            di_strong=di_strong,
            di_mild=di_mild,
            atr_pct_high=atr_pct_high,
            atr_pct_low=atr_pct_low,
            rsi_overbought=self.params['rsi_overbought'],
            rsi_oversold=self.params['rsi_oversold'],
            regime_1_active=self.params['regime_1_active'],
            regime_2_active=self.params['regime_2_active'],
            regime_3_active=self.params['regime_3_active'],
            regime_4_active=self.params['regime_4_active'],
            regime_5_active=self.params['regime_5_active'],
            regime_6_active=self.params['regime_6_active'],
            entry_on_transition_only=self.params['entry_on_transition_only'],
            transition_window=self.params['transition_window'],
            cooldown_bars=self.params['cooldown_bars'],
            rsi_confirm=self.params['rsi_confirm'],
            bb_confirm=self.params['bb_confirm'],
            max_trades_per_5days=self.params['max_trades_per_5days'],
            exit_on_regime_change=self.params['exit_on_regime_change'],
            swap_per_lot_per_day=self.params['swap_per_lot_per_day'],
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
                print("Error during backtest (regime_adaptive):", e)
            return (-1e6, {"num_trades": 0, "win_pct": 0,
                           "max_dd": 0, "sharpe": 0})

        final_value = cerebro.broker.getvalue()
        profit = final_value - 10000.0
        if not _QUIET:
            print(f"Evaluated candidate {[round(x,3) for x in individual]} -> Profit: {profit:.2f}")

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
    # Inner Backtrader Strategy — Regime Adaptive
    # =====================================================================
    class RegimeAdaptiveStrategy(bt.Strategy):
        """
        Forex strategy that adapts to detected market regime.
        
        No predictions needed — purely reactive to current indicator state.
        Each regime maps to a specific trading action:
          - buy_reversal, sell_trend, sell_exhaustion, flat,
            buy_meanrevert, buy_trend
        """

        params = dict(
            pip_cost=0.00001,
            rel_volume=0.10,
            min_order_volume=10000,
            max_order_volume=1000000,
            leverage=100,
            atr_period=14,
            atr_tp_multiplier=2.5,
            atr_sl_multiplier=1.5,
            # Regime thresholds
            adx_strong=35.0,
            adx_mild=25.0,
            di_strong=15.0,
            di_mild=5.0,
            atr_pct_high=0.65,
            atr_pct_low=0.35,
            rsi_overbought=65.0,
            rsi_oversold=40.0,
            # Per-regime toggles
            regime_1_active=1,
            regime_2_active=1,
            regime_3_active=0,
            regime_4_active=0,
            regime_5_active=1,
            regime_6_active=1,
            # Entry filters
            entry_on_transition_only=True,
            transition_window=8,
            cooldown_bars=12,
            rsi_confirm=True,
            bb_confirm=True,
            # Trade management
            max_trades_per_5days=3,
            exit_on_regime_change=True,
            swap_per_lot_per_day=10.0,
            spread_pips=15.0,
            commission_per_lot=7.0,
            slippage_pips=5.0,
        )

        def __init__(self):
            super().__init__()
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
            self.current_regime_at_entry = None
            self.order_direction = None
            self.order_entry_price = None
            self.true_entry_price = None  # Track actual entry price separately
            self.trade_entry_bar = None
            self.current_volume = None
            self.last_trade_bar = -999  # For cooldown

            # Regime transition tracking
            self.prev_regime = 4  # Start as MILD_RANGE
            self.regime_changed_bar = -999  # Bar when last regime change happened

            # Backtrader ATR indicator
            self.atr = bt.indicators.ATR(self.data0, period=self.p.atr_period)

            # Accumulate hourly bars for 4h resampling
            self._hourly_close = []
            self._hourly_high = []
            self._hourly_low = []
            self._hourly_open = []
            # 4h resampled history for regime computation
            self._close_history = []
            self._high_history = []
            self._low_history = []
            self._bar_count = 0
            # Cache for RSI/BB used in confirmation
            self._current_rsi = 50.0
            self._current_bb_position = 0.5

        def _append_4h_bar(self):
            """Accumulate hourly bars and create 4h bars."""
            self._hourly_close.append(self.data0.close[0])
            self._hourly_high.append(self.data0.high[0])
            self._hourly_low.append(self.data0.low[0])
            self._hourly_open.append(self.data0.open[0])
            self._bar_count += 1

            if self._bar_count % 4 == 0 and self._bar_count >= 4:
                # Create 4h bar from last 4 hourly bars
                idx = len(self._hourly_close)
                c4 = self._hourly_close[idx-4:idx]
                h4 = self._hourly_high[idx-4:idx]
                l4 = self._hourly_low[idx-4:idx]
                self._close_history.append(c4[-1])
                self._high_history.append(max(h4))
                self._low_history.append(min(l4))
                return True  # new 4h bar created
            return False

        # --- Regime detection (computed from accumulating history) ---

        def _compute_regime(self):
            """Classify current bar into a regime. Also updates RSI/BB cache."""
            if len(self._close_history) < 200:
                return 4  # default: MILD_RANGE until warmup

            c = np.array(self._close_history)
            h = np.array(self._high_history)
            l = np.array(self._low_history)

            # ADX + DI (EMA-based, 14-period)
            span = 14
            tr = np.maximum(h[1:] - l[1:],
                            np.maximum(np.abs(h[1:] - c[:-1]),
                                       np.abs(l[1:] - c[:-1])))
            
            plus_dm = np.maximum(h[1:] - h[:-1], 0.0)
            minus_dm = np.maximum(l[:-1] - l[1:], 0.0)
            mask = plus_dm > minus_dm
            plus_dm = np.where(mask, plus_dm, 0.0)
            minus_dm = np.where(~mask, minus_dm, 0.0)

            def ema(arr, sp):
                a = 2.0 / (sp + 1)
                out = np.zeros(len(arr))
                out[0] = arr[0]
                for i in range(1, len(arr)):
                    out[i] = a * arr[i] + (1 - a) * out[i - 1]
                return out

            atr_s = ema(tr, span)
            plus_di = 100 * ema(plus_dm, span) / (atr_s + 1e-10)
            minus_di = 100 * ema(minus_dm, span) / (atr_s + 1e-10)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = ema(dx, span)

            current_adx = adx[-1]
            di_spread = plus_di[-1] - minus_di[-1]

            # ATR percentile (last 120 bars)
            atr_14_window = 14
            recent_tr = tr[-120:]
            atr_vals = pd.Series(recent_tr).rolling(atr_14_window).mean().dropna().values
            if len(atr_vals) > 1:
                current_atr_val = atr_vals[-1]
                atr_pct = np.sum(atr_vals <= current_atr_val) / len(atr_vals)
            else:
                atr_pct = 0.5

            # RSI (14-period, EMA) — cache for confirmation
            delta = np.diff(c)
            gain = np.maximum(delta, 0)
            loss_arr = np.maximum(-delta, 0)
            avg_gain = ema(gain[-span*3:], span)
            avg_loss = ema(loss_arr[-span*3:], span)
            if avg_loss[-1] > 1e-10:
                self._current_rsi = 100 - 100 / (1 + avg_gain[-1] / avg_loss[-1])
            else:
                self._current_rsi = 100.0

            # BB position — cache for confirmation
            close_s = pd.Series(c[-30:])
            bb_mid = close_s.rolling(20).mean()
            bb_std = close_s.rolling(20).std()
            if bb_std.iloc[-1] > 1e-10:
                bb_upper = bb_mid.iloc[-1] + 2 * bb_std.iloc[-1]
                bb_lower = bb_mid.iloc[-1] - 2 * bb_std.iloc[-1]
                self._current_bb_position = (c[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)
            else:
                self._current_bb_position = 0.5

            # Classify using thresholds
            regime = 4  # default MILD_RANGE

            if current_adx >= self.p.adx_strong and di_spread > self.p.di_strong:
                regime = 3  # STRONG_UPTREND
            elif current_adx >= self.p.adx_strong and di_spread < -self.p.di_strong:
                regime = 2  # STRONG_DOWNTREND
            elif atr_pct >= self.p.atr_pct_high and di_spread < -self.p.di_mild:
                regime = 1  # HIGH_VOL_BEARISH_FADING
            elif atr_pct <= self.p.atr_pct_low:
                if di_spread < -self.p.di_mild:
                    regime = 5  # LOW_VOL_BEARISH_PULLBACK
                elif di_spread > self.p.di_mild:
                    regime = 6  # LOW_VOL_BULLISH_DRIFT

            return regime

        def _regime_signal(self, regime):
            """Map regime to trade signal."""
            actions = {
                1: 'buy' if self.p.regime_1_active else None,
                2: 'sell' if self.p.regime_2_active else None,
                3: 'sell' if self.p.regime_3_active else None,
                4: None,
                5: 'buy' if self.p.regime_5_active else None,
                6: 'buy' if self.p.regime_6_active else None,
            }
            return actions.get(regime, None)

        def _confirm_entry(self, regime, signal):
            """Apply RSI and BB confirmation filters."""
            rsi = self._current_rsi
            bb_pos = self._current_bb_position

            if self.p.rsi_confirm:
                if signal == 'buy' and rsi > self.p.rsi_overbought:
                    return False  # Don't buy when overbought
                if signal == 'sell' and rsi < self.p.rsi_oversold:
                    return False  # Don't sell when oversold

            if self.p.bb_confirm:
                # Mean-revert regimes (1, 5): price should be near lower BB
                if regime in (1, 5) and bb_pos > 0.5:
                    return False  # Not low enough for mean-revert buy
                # Trend-follow sell (2): price should be near upper BB or mid
                if regime == 2 and bb_pos < 0.3:
                    return False  # Already at bottom, don't sell more
                # Bullish drift (6): price shouldn't be at extreme top
                if regime == 6 and bb_pos > 0.9:
                    return False  # Too stretched

            return True

        def next(self):
            dt = self.data0.datetime.datetime(0)
            current_price = self.data0.close[0]

            # Accumulate hourly bars into 4h bars for regime analysis
            new_4h = self._append_4h_bar()

            self.balance_history.append(self.broker.getvalue())
            self.date_history.append(dt)

            # Recompute regime only on new 4h bars (saves CPU + reduces noise)
            regime = self.prev_regime
            if new_4h:
                regime = self._compute_regime()

                # Track regime transitions
                current_bar = len(self)
                if regime != self.prev_regime:
                    self.regime_changed_bar = current_bar
                    if not _QUIET:
                        print(f"[REGIME_CHANGE] {dt} "
                              f"{REGIME_NAMES.get(self.prev_regime, '?')} → "
                              f"{REGIME_NAMES.get(regime, '?')} "
                              f"RSI={self._current_rsi:.1f} BB={self._current_bb_position:.2f}")
                    self.prev_regime = regime

            current_bar = len(self)

            # ── IN POSITION: check TP/SL and regime exit ──
            if self.position:
                # Friday close
                if dt.weekday() == 4 and dt.hour >= 20:
                    if not _QUIET:
                        print(f"[FRIDAY_CLOSE] {dt}")
                    self.close()
                    return

                bar_high = self.data0.high[0]
                bar_low = self.data0.low[0]

                if self.current_direction == 'buy':
                    if self.trade_low is None or bar_low < self.trade_low:
                        self.trade_low = bar_low
                    if bar_low <= self.current_sl:
                        self.close()
                        return
                    if current_price >= self.current_tp:
                        self.close()
                        return
                elif self.current_direction == 'sell':
                    if self.trade_high is None or bar_high > self.trade_high:
                        self.trade_high = bar_high
                    if bar_high >= self.current_sl:
                        self.close()
                        return
                    if current_price <= self.current_tp:
                        self.close()
                        return

                # Exit on regime change (opposing signal)
                if self.p.exit_on_regime_change:
                    new_signal = self._regime_signal(regime)
                    if new_signal is not None and new_signal != self.current_direction:
                        if not _QUIET:
                            print(f"[REGIME_EXIT] {dt} regime={regime}"
                                  f" ({REGIME_NAMES.get(regime,'?')})"
                                  f" opposes {self.current_direction}")
                        self.close()
                        return

                return

            # ── NO POSITION ──
            self.trade_low = current_price
            self.trade_high = current_price

            # No new entries on Friday
            if dt.weekday() == 4:
                return

            # Wait for warmup
            if current_bar <= max(self.p.atr_period, 200):
                return

            # Cooldown between trades
            if (current_bar - self.last_trade_bar) < self.p.cooldown_bars:
                return

            # Trade frequency limit
            recent = [d for d in self.trade_entry_dates if (dt - d).days < 5]
            if len(recent) >= self.p.max_trades_per_5days:
                return

            # ATR for TP/SL
            current_atr = self.atr[0]
            if current_atr <= 0:
                return

            # Get regime signal
            signal = self._regime_signal(regime)
            if signal is None:
                return

            # Transition-only filter: only trade within N bars of regime change
            if self.p.entry_on_transition_only:
                bars_since_change = current_bar - self.regime_changed_bar
                if bars_since_change > self.p.transition_window:
                    return

            # Confirmation filters (RSI + BB)
            if not self._confirm_entry(regime, signal):
                return

            # Compute TP/SL
            tp_distance = current_atr * self.p.atr_tp_multiplier
            sl_distance = current_atr * self.p.atr_sl_multiplier

            if signal == 'buy':
                chosen_tp = current_price + tp_distance
                chosen_sl = current_price - sl_distance
            else:
                chosen_tp = current_price - tp_distance
                chosen_sl = current_price + sl_distance

            # Position sizing
            order_size = self._compute_size()
            if order_size <= 0:
                return

            self.trade_entry_dates.append(dt)
            self.trade_entry_bar = current_bar
            self.last_trade_bar = current_bar
            self.current_volume = order_size
            self.current_tp = chosen_tp
            self.current_sl = chosen_sl
            self.current_regime_at_entry = regime
            self.true_entry_price = current_price

            if signal == 'buy':
                self.buy(size=order_size)
                self.current_direction = 'buy'
            else:
                self.sell(size=order_size)
                self.current_direction = 'sell'

            if not _QUIET:
                print(f"[ENTRY] {dt} {signal.upper()} @ {current_price:.5f}"
                      f" regime={regime} ({REGIME_NAMES.get(regime, '?')})"
                      f" TP={chosen_tp:.5f} SL={chosen_sl:.5f}"
                      f" ATR={current_atr:.5f}"
                      f" RSI={self._current_rsi:.1f} BB={self._current_bb_position:.2f}")

        def _compute_size(self):
            cash = self.broker.getcash()
            size = cash * self.p.rel_volume * self.p.leverage
            size = max(self.p.min_order_volume, min(self.p.max_order_volume, size))
            return size

        def notify_order(self, order):
            if order.status in [order.Completed]:
                # Only record entry price on the opening side
                if not self.position or self.order_entry_price is None:
                    self.order_entry_price = order.executed.price

        def notify_trade(self, trade):
            if trade.isclosed:
                duration = len(self) - (
                    self.trade_entry_bar if self.trade_entry_bar else 0
                )
                dt = self.data0.datetime.datetime(0)
                entry_price = self.true_entry_price or 0
                exit_price = self.data0.close[0]
                profit_usd = trade.pnlcomm
                overnight_days = max(0, duration / 24.0)
                volume = self.current_volume or 0
                lots = volume / 100000.0
                swap_cost = overnight_days * lots * self.p.swap_per_lot_per_day
                profit_usd -= swap_cost
                direction = self.current_direction

                if direction == 'buy':
                    profit_pips = (exit_price - entry_price) / self.p.pip_cost
                    intra_dd = (entry_price - (self.trade_low or entry_price)) / self.p.pip_cost
                else:
                    profit_pips = (entry_price - exit_price) / self.p.pip_cost
                    intra_dd = ((self.trade_high or entry_price) - entry_price) / self.p.pip_cost

                label = "WIN" if profit_usd > 0 else "LOSS"
                if not _QUIET:
                    print(f"[{label}] {direction} {entry_price:.5f}→{exit_price:.5f}"
                          f" PnL: {profit_pips:.0f} pips ({profit_usd:.2f} USD)"
                          f" Duration: {duration} bars"
                          f" Balance: {self.broker.getvalue():.2f}"
                          f" Regime: {REGIME_NAMES.get(self.current_regime_at_entry, '?')}")

                trade_record = {
                    'open_dt': (dt - datetime.timedelta(hours=duration))
                               .strftime('%Y-%m-%d %H:%M:%S'),
                    'close_dt': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'volume': volume,
                    'pnl': profit_usd,
                    'pnl_pips': profit_pips,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'max_dd': intra_dd,
                    'duration_bars': duration,
                    'balance': self.broker.getvalue(),
                    'regime': self.current_regime_at_entry,
                }
                self.trades.append(trade_record)

                # Reset
                self.current_direction = None
                self.current_tp = None
                self.current_sl = None
                self.current_regime_at_entry = None
                self.order_entry_price = None
                self.true_entry_price = None
                self.trade_entry_bar = None
                self.current_volume = None
                self.trade_low = None
                self.trade_high = None

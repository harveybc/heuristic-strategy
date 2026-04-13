import pandas as pd
import numpy as np

trades = pd.read_csv('trades.csv')
print('='*60)
print('   HEURISTIC-STRATEGY BACKTEST — FULL RESULTS')
print('='*60)

# Data periods
print()
print('--- DATA PERIODS ---')
print('HS OHLC data:  tests/data/phase_2_3_base_d3_last_year.csv')
ohlc = pd.read_csv('tests/data/phase_2_3_base_d3_last_year.csv')
print(f'  Start: {ohlc.DATE_TIME.iloc[0]}')
print(f'  End:   {ohlc.DATE_TIME.iloc[-1]}')
print(f'  Bars:  {len(ohlc)}')

pp_data = pd.read_csv('/home/harveybc/Documents/GitHub/predictor/examples/data_downsampled/phase_1_b/normalized_d6.csv')
print('PP Normalized:  normalized_d6.csv')
print(f'  Start: {pp_data.DATE_TIME.iloc[0]}')
print(f'  End:   {pp_data.DATE_TIME.iloc[-1]}')
print(f'  Bars:  {len(pp_data)}')

print()
print('--- TRADE SUMMARY ---')
n = len(trades)
wins = (trades.pnl > 0).sum()
losses = (trades.pnl <= 0).sum()
win_pct = wins/n*100 if n else 0
avg_win = trades.loc[trades.pnl>0, 'pnl'].mean() if wins else 0
avg_loss = trades.loc[trades.pnl<=0, 'pnl'].mean() if losses else 0
print(f'Total Trades:       {n}')
print(f'Winning Trades:     {wins} ({win_pct:.1f}%)')
print(f'Losing Trades:      {losses} ({100-win_pct:.1f}%)')
print(f'Avg Win (USD):      {avg_win:.2f}')
print(f'Avg Loss (USD):     {avg_loss:.2f}')
if avg_loss != 0:
    print(f'Win/Loss Ratio:     {abs(avg_win/avg_loss):.2f}')

print()
print('--- P&L METRICS ---')
initial = 10000.0
cumbal = initial + trades.pnl.cumsum()
final = cumbal.iloc[-1]
total_pnl = final - initial
print(f'Initial Balance:    ${initial:,.2f}')
print(f'Final Balance:      ${final:,.2f}')
print(f'Total P/L:          ${total_pnl:,.2f}')
print(f'Return:             {total_pnl/initial*100:.2f}%')
print(f'Avg Trade P/L:      ${trades.pnl.mean():.2f}')
print(f'Median Trade P/L:   ${trades.pnl.median():.2f}')
print(f'Std Dev P/L:        ${trades.pnl.std():.2f}')
print(f'Best Trade:         ${trades.pnl.max():.2f}')
print(f'Worst Trade:        ${trades.pnl.min():.2f}')

print()
print('--- PIP METRICS ---')
print(f'Total Pips:         {trades.pips.sum():.0f}')
print(f'Avg Pips/Trade:     {trades.pips.mean():.1f}')
print(f'Median Pips/Trade:  {trades.pips.median():.1f}')
if wins:
    print(f'Avg Win (pips):     {trades.loc[trades.pnl>0, "pips"].mean():.1f}')
if losses:
    print(f'Avg Loss (pips):    {trades.loc[trades.pnl<=0, "pips"].mean():.1f}')

print()
print('--- DRAWDOWN ---')
peak = cumbal.cummax()
dd = cumbal - peak
max_dd_usd = dd.min()
max_dd_pct = (dd / peak).min() * 100
print(f'Max Drawdown (USD): ${max_dd_usd:,.2f}')
print(f'Max Drawdown (%):   {max_dd_pct:.2f}%')
print(f'Min Balance:        ${cumbal.min():,.2f}')
print(f'Avg Trade MaxDD:    {trades.max_dd.mean():.1f} pips')
print(f'Worst Trade MaxDD:  {trades.max_dd.max():.0f} pips')

print()
print('--- RISK METRICS ---')
if trades.pnl.std() > 0:
    days = (pd.to_datetime(ohlc.DATE_TIME.iloc[-1]) - pd.to_datetime(ohlc.DATE_TIME.iloc[0])).days
    trades_per_year = n / (days / 365.25)
    sharpe = (trades.pnl.mean() / trades.pnl.std()) * np.sqrt(trades_per_year)
    print(f'Sharpe Ratio:       {sharpe:.2f}')

gross_profit = trades.loc[trades.pnl>0, 'pnl'].sum()
gross_loss = abs(trades.loc[trades.pnl<=0, 'pnl'].sum())
pf = gross_profit / gross_loss if gross_loss else float('inf')
print(f'Profit Factor:      {pf:.3f}')
print(f'Gross Profit:       ${gross_profit:,.2f}')
print(f'Gross Loss:         ${gross_loss:,.2f}')

expectancy = trades.pnl.mean()
print(f'Expectancy/Trade:   ${expectancy:.2f}')

if max_dd_usd < 0:
    days = (pd.to_datetime(ohlc.DATE_TIME.iloc[-1]) - pd.to_datetime(ohlc.DATE_TIME.iloc[0])).days
    ann_ret = total_pnl * (365.25 / days)
    calmar = ann_ret / abs(max_dd_usd)
    print(f'Calmar Ratio:       {calmar:.2f}')

if abs(max_dd_usd) > 0:
    recovery = total_pnl / abs(max_dd_usd)
    print(f'Recovery Factor:    {recovery:.2f}')

print()
print('--- DURATION ---')
print(f'Avg Duration (bars):{trades.duration.mean():.1f}')
print(f'Min Duration:       {trades.duration.min()} bars')
print(f'Max Duration:       {trades.duration.max()} bars')
print(f'Median Duration:    {trades.duration.median():.0f} bars')

print()
print('--- DIRECTION ---')
print('All trades: BUY (buy_entry model only)')

print()
print('--- TRADE PERIOD ---')
print(f'First Trade:  {trades.open_dt.iloc[0]}')
print(f'Last Trade:   {trades.open_dt.iloc[-1]}')

print()
print('--- COST ASSUMPTIONS (worst-case) ---')
print('Spread:       3.0 real pips (30 pipettes)')
print('Commission:   $10/lot')
print('Slippage:     1.0 real pips (10 pipettes)')
print('Swap:         $15/lot/day')

print()
print('--- OPTIMIZATION ---')
print('Population: 1, Generations: 1 (single random candidate)')
print('Parameters used:')
import json
with open('parameters.json') as f:
    params = json.load(f)
for k, v in params.items():
    print(f'  {k}: {v}')
print('='*60)

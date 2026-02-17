#!/usr/bin/env python3
"""
Long-term vs Short-term Prediction Importance Analysis
Generates publication-quality plots from cross-sensitivity data.

Key question: How much does hourly (short-term) vs daily (long-term)
prediction quality matter for trading profitability?
"""
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "results/cross_sensitivity/cross_sensitivity_olap.db"
OUT_DIR = Path(__file__).parent.parent / "results/cross_sensitivity/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM cross_sensitivity ORDER BY hourly_noise, daily_noise", conn)
daily_m = pd.read_sql("SELECT * FROM v_daily_marginal ORDER BY daily_noise", conn)
hourly_m = pd.read_sql("SELECT * FROM v_hourly_marginal ORDER BY hourly_noise", conn)
conn.close()

noise_levels = sorted(df['hourly_noise'].unique())

# ============================================================
# PLOT 1: Profit Heatmap (hourly × daily noise)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pivot_profit = df.pivot(index='hourly_noise', columns='daily_noise', values='profit')
pivot_win = df.pivot(index='hourly_noise', columns='daily_noise', values='win_pct')

# Profit heatmap
im1 = axes[0].imshow(pivot_profit.values, cmap='RdYlGn', aspect='auto',
                      vmin=-500, vmax=2800)
axes[0].set_xticks(range(len(noise_levels)))
axes[0].set_xticklabels([f"{x:.3f}" for x in noise_levels], rotation=45, fontsize=8)
axes[0].set_yticks(range(len(noise_levels)))
axes[0].set_yticklabels([f"{x:.3f}" for x in noise_levels], fontsize=8)
axes[0].set_xlabel("Daily (Long-term) Noise σ", fontsize=11)
axes[0].set_ylabel("Hourly (Short-term) Noise σ", fontsize=11)
axes[0].set_title("Trading Profit ($)", fontsize=13, fontweight='bold')
for i in range(len(noise_levels)):
    for j in range(len(noise_levels)):
        val = pivot_profit.values[i, j]
        color = 'white' if abs(val) > 1500 or val < 0 else 'black'
        axes[0].text(j, i, f"{val:.0f}", ha='center', va='center', fontsize=7, color=color)
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Win rate heatmap
im2 = axes[1].imshow(pivot_win.values, cmap='RdYlGn', aspect='auto',
                      vmin=30, vmax=100)
axes[1].set_xticks(range(len(noise_levels)))
axes[1].set_xticklabels([f"{x:.3f}" for x in noise_levels], rotation=45, fontsize=8)
axes[1].set_yticks(range(len(noise_levels)))
axes[1].set_yticklabels([f"{x:.3f}" for x in noise_levels], fontsize=8)
axes[1].set_xlabel("Daily (Long-term) Noise σ", fontsize=11)
axes[1].set_ylabel("Hourly (Short-term) Noise σ", fontsize=11)
axes[1].set_title("Win Rate (%)", fontsize=13, fontweight='bold')
for i in range(len(noise_levels)):
    for j in range(len(noise_levels)):
        val = pivot_win.values[i, j]
        color = 'white' if val < 50 else 'black'
        axes[1].text(j, i, f"{val:.0f}%", ha='center', va='center', fontsize=7, color=color)
plt.colorbar(im2, ax=axes[1], shrink=0.8)

fig.suptitle("Cross-Sensitivity: Hourly (Short-term) vs Daily (Long-term) Prediction Quality",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "heatmaps_profit_winrate.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'heatmaps_profit_winrate.png'}")

# ============================================================
# PLOT 2: Marginal Importance Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Profit marginal
axes[0].plot(daily_m['daily_noise'], daily_m['avg_profit'], 'bo-', linewidth=2,
             markersize=8, label='Daily (Long-term) noise effect')
axes[0].plot(hourly_m['hourly_noise'], hourly_m['avg_profit'], 'rs-', linewidth=2,
             markersize=8, label='Hourly (Short-term) noise effect')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel("Gaussian Noise σ", fontsize=11)
axes[0].set_ylabel("Average Profit ($)", fontsize=11)
axes[0].set_title("Profit Sensitivity to Prediction Noise", fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Win rate marginal
axes[1].plot(daily_m['daily_noise'], daily_m['avg_win'], 'bo-', linewidth=2,
             markersize=8, label='Daily (Long-term) noise effect')
axes[1].plot(hourly_m['hourly_noise'], hourly_m['avg_win'], 'rs-', linewidth=2,
             markersize=8, label='Hourly (Short-term) noise effect')
axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
axes[1].set_xlabel("Gaussian Noise σ", fontsize=11)
axes[1].set_ylabel("Average Win Rate (%)", fontsize=11)
axes[1].set_title("Win Rate Sensitivity to Prediction Noise", fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle("Which Predictions Matter More? Long-term vs Short-term",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "marginal_importance.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'marginal_importance.png'}")

# ============================================================
# PLOT 3: Degradation curves — normalized from perfect (0 noise)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Normalize: % change from 0-noise baseline
daily_base_profit = daily_m['avg_profit'].iloc[0]
hourly_base_profit = hourly_m['avg_profit'].iloc[0]
daily_base_win = daily_m['avg_win'].iloc[0]
hourly_base_win = hourly_m['avg_win'].iloc[0]

daily_profit_pct = (daily_m['avg_profit'] - daily_base_profit) / abs(daily_base_profit) * 100
hourly_profit_pct = (hourly_m['avg_profit'] - hourly_base_profit) / abs(hourly_base_profit) * 100
daily_win_pct = (daily_m['avg_win'] - daily_base_win) / abs(daily_base_win) * 100
hourly_win_pct = (hourly_m['avg_win'] - hourly_base_win) / abs(hourly_base_win) * 100

axes[0].plot(daily_m['daily_noise'], daily_profit_pct, 'bo-', linewidth=2,
             markersize=8, label='Daily (Long-term)')
axes[0].plot(hourly_m['hourly_noise'], hourly_profit_pct, 'rs-', linewidth=2,
             markersize=8, label='Hourly (Short-term)')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel("Gaussian Noise σ", fontsize=11)
axes[0].set_ylabel("Profit Change from Baseline (%)", fontsize=11)
axes[0].set_title("Profit Degradation Rate", fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(daily_m['daily_noise'], daily_win_pct, 'bo-', linewidth=2,
             markersize=8, label='Daily (Long-term)')
axes[1].plot(hourly_m['hourly_noise'], hourly_win_pct, 'rs-', linewidth=2,
             markersize=8, label='Hourly (Short-term)')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel("Gaussian Noise σ", fontsize=11)
axes[1].set_ylabel("Win Rate Change from Baseline (%)", fontsize=11)
axes[1].set_title("Win Rate Degradation Rate", fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle("Degradation: How Fast Does Performance Drop?",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "degradation_curves.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'degradation_curves.png'}")

# ============================================================
# PLOT 4: Max Drawdown heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
pivot_dd = df.pivot(index='hourly_noise', columns='daily_noise', values='max_dd')
im = ax.imshow(pivot_dd.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(noise_levels)))
ax.set_xticklabels([f"{x:.3f}" for x in noise_levels], rotation=45, fontsize=8)
ax.set_yticks(range(len(noise_levels)))
ax.set_yticklabels([f"{x:.3f}" for x in noise_levels], fontsize=8)
ax.set_xlabel("Daily (Long-term) Noise σ", fontsize=11)
ax.set_ylabel("Hourly (Short-term) Noise σ", fontsize=11)
ax.set_title("Maximum Drawdown ($) — Risk Sensitivity", fontsize=13, fontweight='bold')
for i in range(len(noise_levels)):
    for j in range(len(noise_levels)):
        val = pivot_dd.values[i, j]
        color = 'white' if val > 3000 else 'black'
        ax.text(j, i, f"{val:.0f}", ha='center', va='center', fontsize=7, color=color)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(OUT_DIR / "max_drawdown_heatmap.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'max_drawdown_heatmap.png'}")

# ============================================================
# Summary statistics
# ============================================================
print("\n=== IMPORTANCE ANALYSIS ===")
print(f"Daily noise profit range: ${daily_m['avg_profit'].min():.0f} to ${daily_m['avg_profit'].max():.0f} (spread: ${daily_m['avg_profit'].max()-daily_m['avg_profit'].min():.0f})")
print(f"Hourly noise profit range: ${hourly_m['avg_profit'].min():.0f} to ${hourly_m['avg_profit'].max():.0f} (spread: ${hourly_m['avg_profit'].max()-hourly_m['avg_profit'].min():.0f})")
print(f"\nDaily noise win rate range: {daily_m['avg_win'].min():.1f}% to {daily_m['avg_win'].max():.1f}% (spread: {daily_m['avg_win'].max()-daily_m['avg_win'].min():.1f}pp)")
print(f"Hourly noise win rate range: {hourly_m['avg_win'].min():.1f}% to {hourly_m['avg_win'].max():.1f}% (spread: {hourly_m['avg_win'].max()-hourly_m['avg_win'].min():.1f}pp)")

# Profit stays positive threshold
for _, row in daily_m.iterrows():
    if row['avg_profit'] < 0:
        print(f"\nDaily: profit goes negative at σ={row['daily_noise']}")
        break
else:
    print(f"\nDaily: profit stays positive through σ={daily_m['daily_noise'].max()}")

for _, row in hourly_m.iterrows():
    if row['avg_profit'] < 0:
        print(f"Hourly: profit goes negative at σ={row['hourly_noise']}")
        break
else:
    print(f"Hourly: profit stays positive through σ={hourly_m['hourly_noise'].max()}")

print("\n=== CONCLUSION ===")
daily_spread = daily_m['avg_win'].max() - daily_m['avg_win'].min()
hourly_spread = hourly_m['avg_win'].max() - hourly_m['avg_win'].min()
if daily_spread > hourly_spread:
    print(f"Long-term (daily) predictions have GREATER impact on win rate ({daily_spread:.1f}pp vs {hourly_spread:.1f}pp)")
else:
    print(f"Short-term (hourly) predictions have GREATER impact on win rate ({hourly_spread:.1f}pp vs {daily_spread:.1f}pp)")

print("\nAll 4 plots saved to:", OUT_DIR)

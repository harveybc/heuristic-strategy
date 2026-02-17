#!/usr/bin/env python3
"""Plot noise sensitivity results for high-freq vs low-freq trading with 4h predictions."""
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DB = Path(__file__).parent.parent / "results/noise_sensitivity_4h/noise_sensitivity_4h_olap.db"
OUT = Path(__file__).parent.parent / "results/noise_sensitivity_4h/plots"
OUT.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB)

# ============================================================
# PLOT 1: Profit heatmaps side by side (high vs low freq)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (mode, title) in enumerate([("high_freq", "High Frequency (20 trades/5d)"),
                                       ("low_freq", "Low Frequency (3 trades/5d - PDT)")]):
    df = pd.read_sql(f"SELECT * FROM experiments WHERE freq_mode='{mode}'", conn)
    pivot = df.pivot(index='hourly_noise', columns='daily_noise', values='profit')
    noise = sorted(df['hourly_noise'].unique())
    
    # Log scale for better visualization of large range
    log_vals = np.log10(pivot.values.clip(min=1))
    im = axes[idx].imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    axes[idx].set_xticks(range(len(noise)))
    axes[idx].set_xticklabels([f"{x:.3f}" for x in noise], rotation=45, fontsize=7)
    axes[idx].set_yticks(range(len(noise)))
    axes[idx].set_yticklabels([f"{x:.3f}" for x in noise], fontsize=7)
    axes[idx].set_xlabel("Daily Noise σ", fontsize=10)
    axes[idx].set_ylabel("Hourly Noise σ", fontsize=10)
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    for i in range(len(noise)):
        for j in range(len(noise)):
            val = pivot.values[i, j]
            txt = f"${val/1000:.0f}K" if val >= 1000 else f"${val:.0f}"
            color = 'white' if val < pivot.values.mean()*0.5 else 'black'
            axes[idx].text(j, i, txt, ha='center', va='center', fontsize=6, color=color)
    plt.colorbar(im, ax=axes[idx], shrink=0.8)

fig.suptitle("Trading Profit: High vs Low Frequency (4h predictions on 1h bars)", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / "profit_heatmaps_dual_freq.png", dpi=150, bbox_inches='tight')
print(f"Saved: profit_heatmaps_dual_freq.png")

# ============================================================
# PLOT 2: Marginal comparison (high vs low)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, (mode, label, color) in enumerate([("high_freq", "High Freq", "blue"), ("low_freq", "Low Freq (PDT)", "red")]):
    dm = pd.read_sql(f"SELECT * FROM v_{mode}_daily_marginal", conn)
    hm = pd.read_sql(f"SELECT * FROM v_{mode}_hourly_marginal", conn)
    
    # Profit - daily marginal
    axes[0,0].plot(dm['daily_noise'], dm['avg_profit']/1000, f'{color[0]}o-', linewidth=2, markersize=6, label=label)
    # Profit - hourly marginal
    axes[0,1].plot(hm['hourly_noise'], hm['avg_profit']/1000, f'{color[0]}s-', linewidth=2, markersize=6, label=label)
    # Win rate - daily
    axes[1,0].plot(dm['daily_noise'], dm['avg_win'], f'{color[0]}o-', linewidth=2, markersize=6, label=label)
    # Win rate - hourly
    axes[1,1].plot(hm['hourly_noise'], hm['avg_win'], f'{color[0]}s-', linewidth=2, markersize=6, label=label)

axes[0,0].set_title("Daily Noise → Profit", fontweight='bold')
axes[0,0].set_ylabel("Avg Profit ($K)")
axes[0,1].set_title("Hourly Noise → Profit", fontweight='bold')
axes[1,0].set_title("Daily Noise → Win Rate", fontweight='bold')
axes[1,0].set_ylabel("Win Rate (%)")
axes[1,1].set_title("Hourly Noise → Win Rate", fontweight='bold')

for ax in axes.flat:
    ax.set_xlabel("Gaussian Noise σ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[1,0].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
axes[1,1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)

fig.suptitle("High vs Low Frequency: Noise Sensitivity (4h predictions, 1h bars)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / "marginal_dual_freq.png", dpi=150, bbox_inches='tight')
print(f"Saved: marginal_dual_freq.png")

# ============================================================
# PLOT 3: Win rate heatmaps
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (mode, title) in enumerate([("high_freq", "High Frequency"), ("low_freq", "Low Frequency (PDT)")]):
    df = pd.read_sql(f"SELECT * FROM experiments WHERE freq_mode='{mode}'", conn)
    pivot = df.pivot(index='hourly_noise', columns='daily_noise', values='win_pct')
    noise = sorted(df['hourly_noise'].unique())
    
    im = axes[idx].imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=40, vmax=80)
    axes[idx].set_xticks(range(len(noise)))
    axes[idx].set_xticklabels([f"{x:.3f}" for x in noise], rotation=45, fontsize=7)
    axes[idx].set_yticks(range(len(noise)))
    axes[idx].set_yticklabels([f"{x:.3f}" for x in noise], fontsize=7)
    axes[idx].set_xlabel("Daily Noise σ", fontsize=10)
    axes[idx].set_ylabel("Hourly Noise σ", fontsize=10)
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    for i in range(len(noise)):
        for j in range(len(noise)):
            val = pivot.values[i, j]
            color = 'white' if val < 50 else 'black'
            axes[idx].text(j, i, f"{val:.0f}%", ha='center', va='center', fontsize=6, color=color)
    plt.colorbar(im, ax=axes[idx], shrink=0.8)

fig.suptitle("Win Rate: High vs Low Frequency (4h predictions)", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / "winrate_heatmaps_dual_freq.png", dpi=150, bbox_inches='tight')
print(f"Saved: winrate_heatmaps_dual_freq.png")

conn.close()
print(f"\nAll plots saved to: {OUT}")

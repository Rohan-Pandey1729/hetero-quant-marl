"""
Generate publication-quality figures for the paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style for publication
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})

# Data from experiments
data = {
    'FP32': -63.93,
    'INT8': -64.72,
    'INT4': -63.09,
    'INT2': -136.61,
}

rescue_data = {
    'All INT2\n(0 rescuers)': -136.61,
    '1 INT8 +\n2 INT2': -74.80,
    '2 INT8 +\n1 INT2': -63.70,
    'All INT8': -64.72,
}

position_data = {
    'agent_0\nINT8': -76.07,
    'agent_1\nINT8': -92.35,
    'agent_2\nINT8': -138.55,
}

position_std = {
    'agent_0\nINT8': 2.74,
    'agent_1\nINT8': 1.98,
    'agent_2\nINT8': 0.51,
}

# Figure 1: Uniform Quantization
fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(data))
colors = ['#2ecc71', '#3498db', '#3498db', '#e74c3c']
bars = ax.bar(x, list(data.values()), color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(list(data.keys()))
ax.set_ylabel('Mean Episode Reward')
ax.set_xlabel('Quantization Precision')
ax.set_title('(a) Uniform Quantization: All Agents Same Precision')
ax.axhline(y=-90, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Random Policy')
ax.axhline(y=-63.93, color='green', linestyle=':', alpha=0.7, linewidth=1.5, label='FP32 Baseline')
ax.set_ylim(-160, -40)
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 8, f'{val:.1f}', 
            ha='center', va='top', fontsize=11, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('experiments/fig1_uniform_quantization.png', dpi=300, bbox_inches='tight')
plt.savefig('experiments/fig1_uniform_quantization.pdf', bbox_inches='tight')
plt.close()

# Figure 2: Rescue Experiment
fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(rescue_data))
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#2ecc71']
bars = ax.bar(x, list(rescue_data.values()), color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(list(rescue_data.keys()))
ax.set_ylabel('Mean Episode Reward')
ax.set_xlabel('Team Composition')
ax.set_title('(b) Rescuing INT2 Teams with High-Precision Agents')
ax.axhline(y=-90, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Random Policy')
ax.set_ylim(-160, -40)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='y')

# Add annotations
ax.annotate('', xy=(1, -74.80), xytext=(0, -136.61),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(0.5, -105, '+60%\nrecovery', ha='center', fontsize=10, color='green', fontweight='bold')

for bar, val in zip(bars, rescue_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 8, f'{val:.1f}', 
            ha='center', va='top', fontsize=11, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('experiments/fig2_rescue_experiment.png', dpi=300, bbox_inches='tight')
plt.savefig('experiments/fig2_rescue_experiment.pdf', bbox_inches='tight')
plt.close()

# Figure 3: Position Sensitivity (KEY FINDING)
fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(position_data))
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax.bar(x, list(position_data.values()), yerr=list(position_std.values()),
              color=colors, alpha=0.8, edgecolor='black', linewidth=1.2, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(list(position_data.keys()))
ax.set_ylabel('Mean Episode Reward')
ax.set_xlabel('Which Agent Kept at INT8 (Others at INT2)')
ax.set_title('(c) Agent Position Sensitivity: Which Agent to Prioritize?')
ax.axhline(y=-136.61, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='All INT2')
ax.axhline(y=-64.72, color='green', linestyle=':', alpha=0.7, linewidth=1.5, label='All INT8')
ax.set_ylim(-160, -40)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, position_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 8, f'{val:.1f}', 
            ha='center', va='top', fontsize=11, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('experiments/fig3_position_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig('experiments/fig3_position_sensitivity.pdf', bbox_inches='tight')
plt.close()

# Figure 4: Combined figure for paper
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A
ax = axes[0]
x = range(len(data))
colors = ['#2ecc71', '#3498db', '#3498db', '#e74c3c']
bars = ax.bar(x, list(data.values()), color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(list(data.keys()))
ax.set_ylabel('Mean Episode Reward')
ax.set_title('(a) Uniform Quantization')
ax.axhline(y=-90, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylim(-155, -45)
ax.grid(True, alpha=0.3, axis='y')

# Panel B
ax = axes[1]
x = range(len(rescue_data))
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#2ecc71']
bars = ax.bar(x, list(rescue_data.values()), color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(list(rescue_data.keys()), fontsize=9)
ax.set_title('(b) Rescuing INT2 Teams')
ax.axhline(y=-90, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylim(-155, -45)
ax.grid(True, alpha=0.3, axis='y')

# Panel C
ax = axes[2]
x = range(len(position_data))
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax.bar(x, list(position_data.values()), yerr=list(position_std.values()),
              color=colors, alpha=0.8, edgecolor='black', linewidth=1, capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(list(position_data.keys()))
ax.set_title('(c) Position Sensitivity')
ax.axhline(y=-136.61, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylim(-155, -45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('experiments/fig_combined.png', dpi=300, bbox_inches='tight')
plt.savefig('experiments/fig_combined.pdf', bbox_inches='tight')
plt.close()

print("Figures saved to experiments/")
print("  - fig1_uniform_quantization.png/pdf")
print("  - fig2_rescue_experiment.png/pdf")
print("  - fig3_position_sensitivity.png/pdf")
print("  - fig_combined.png/pdf")
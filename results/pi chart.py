import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# 1. Load your Composite Data
df = pd.read_csv("optimizer_composite_ranking.csv")

# 2. Select Optimizers
optimizers = ["sgd", "adam", "adabelief", "lamb", "stable-spam", "adafactor"]
optimizer_labels = {
    "sgd": "SGD (Baseline)",
    "adam": "Adam (industry Standard)",
    "adabelief": "AdaBelief (Best Final Loss)",
    "lamb": "LAMB (Best Stability)",
    "stable-spam": "Stable-SPAM (Best Sparsity and Convergence)",
    "adafactor": "Adafactor (Over all Winner)"
}
df_subset = df[df["Optimizer"].isin(optimizers)]

# 3. Define the Metrics to Display
categories = ['Loss Score', 'Sparsity Score', 'Stability Score', 'Convergence Score']
labels = ['Final Loss\n(35%)', 'Sparsity\n(35%)', 'Update Stability\n(20%)', 'Convergence\n(10%)']
N = len(categories)

# 4. Setup the Radar Chart
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += [angles[0]]  # Close the loop

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': 'polar'})

# 5. Define colors and styles for each optimizer
colors = {
    "sgd": "#95a5a6",              # Grey
    "adam": "#3498db",             # Blue
    "adabelief": "#e74c3c",        # Red
    "lamb": "#9b59b6",             # Purple
    "stable-spam": "#f1c40f",      # Yellow
    "adafactor": "#2ecc71"         # Green
}

styles = {
    "sgd": "--",
    "adam": "-",
    "adabelief": ":",
    "lamb": "-",
    "stable-spam": "-.",
    "adafactor": "-"
}

# 6. Plot each optimizer
for optimizer in optimizers:
    df_opt = df_subset[df_subset["Optimizer"] == optimizer]
    if not df_opt.empty:
        values = df_opt[categories].values.flatten().tolist()
        values += [values[0]]  # Close the loop
        
        ax.plot(angles, values, linewidth=2.5, linestyle=styles[optimizer], label=optimizer_labels[optimizer], color=colors[optimizer])

# 7. Formatting
plt.xticks(angles[:-1], labels, color='black', size=12)
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
plt.ylim(0, 100)

plt.title("Optimizer Performance Comparison", size=16, color='black', y=1.08, weight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig("optimizers_radar_chart.png", dpi=300, bbox_inches='tight')
print("✅ Optimizers radar chart saved as 'optimizers_radar_chart.png'")
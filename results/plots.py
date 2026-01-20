import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# 1. Load your Composite Data
df = pd.read_csv("optimizer_composite_ranking.csv")

# 2. Select Top 5 Optimizers for the Plot
top_5_names = ["stable-spam", "adafactor", "lion", "adam-mini", "adabelief"]
df_subset = df[df["Optimizer"].isin(top_5_names)].set_index("Optimizer")

# Re-order them to match the ranking
df_subset = df_subset.reindex(top_5_names)

# 3. Define the Metrics to Display (and pretty labels)
categories = ['Loss Score', 'Sparsity Score', 'Stability Score', 'Convergence Score']
labels = ['Final Loss\n(Weighted 35%)', 'Sparsity\n(Weighted 35%)', 'Update Stability\n(Weighted 20%)', 'Convergence\n(Weighted 10%)']
N = len(categories)

# 4. Setup the Radar Chart
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += [angles[0]]  # Close the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

# Helper function to plot one optimizer
def plot_radar(row_name, color):
    values = df_subset.loc[row_name, categories].values.flatten().tolist()
    values += [values[0]]  # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=row_name, color=color)
    ax.fill(angles, values, color=color, alpha=0.1)

# 5. Plot the Contenders
# Colors: Winner (Red/Orange), Runner-up (Green), Others (Blue/Purple)
plot_radar("stable-spam", "#e74c3c")  # Red for Winner
plot_radar("adafactor", "#2ecc71")    # Green for 2nd
plot_radar("lion", "#9b59b6")         # Purple for 3rd
plot_radar("adabelief", "#95a5a6")    # Grey for reference

# 6. Formatting
plt.xticks(angles[:-1], labels, color='black', size=12)
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
plt.ylim(0, 100)

plt.title("Why Stable-SPAM Won: The Balance of Metrics", size=16, color='black', y=1.1, weight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

plt.tight_layout()
plt.savefig("poster_radar_chart.png", dpi=300)
print("✅ Radar chart saved as 'poster_radar_chart.png'")
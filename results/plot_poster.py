import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

# --- SETUP ---
# Set the visual style for a scientific poster
sns.set_theme(style="whitegrid", context="talk") # 'talk' context makes fonts bigger for posters
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})

# --- PART 1: THE BAR CHART (Final Rankings) ---
# Data from your Word document
data = {
    "Optimizer": ["Adafactor", "Stable-SPAM", "Lion", "AdaBelief", "Adam-Mini", 
                  "SGD", "LAMB", "AdamW", "RMSProp"], # Selected subset to avoid clutter
    "Final Loss": [0.078, 0.078, 0.083, 0.076, 0.094, 
                   0.109, 0.101, 0.216, 0.229],
    "Family": ["Efficient", "Experimental", "Sign-Based", "Adaptive", "Efficient", 
               "Baseline", "Large-Batch", "Adaptive", "Baseline"]
}
df_bar = pd.DataFrame(data).sort_values("Final Loss")

plt.figure(figsize=(10, 6))
# Colors: Green for winners, Grey for baselines, Blue/Purple for others
palette = {"Efficient": "#2ecc71", "Experimental": "#f1c40f", "Sign-Based": "#9b59b6", 
           "Adaptive": "#3498db", "Baseline": "#95a5a6", "Large-Batch": "#34495e"}

ax = sns.barplot(data=df_bar, x="Optimizer", y="Final Loss", hue="Family", dodge=False, palette=palette)

# Add value labels on top of bars (Crucial for posters)
for i in ax.containers:
    ax.bar_label(i, fmt='%.3f', padding=3, fontsize=12)

plt.title("Final Loss Comparison (Lower is Better)", fontsize=18, weight='bold')
plt.xlabel("")
plt.ylabel("Cross Entropy Loss")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.savefig("poster_barchart.png", dpi=300)
print("✅ Generated: poster_barchart.png")


# --- PART 2: THE HERO TRAJECTORIES (Line Chart) ---
# The optimizers we want to highlight
heroes = {
    "sgd": ("SGD (Baseline)", "#95a5a6", "--"),
    "adamw": ("AdamW (Standard)", "#3498db", "-"),
    "adafactor": ("Adafactor (Winner)", "#2ecc71", "-"),
    "lion": ("Lion (Sparse King)", "#9b59b6", "-"),
    "stable-spam": ("Stable-SPAM (New)", "#f1c40f", "-.")
}

plt.figure(figsize=(10, 6))

for file_key, (label, color, style) in heroes.items():
    filename = f"{file_key}_seed42.csv"
    # Logic to find the file
    path = os.path.join("logs", filename) if os.path.exists(os.path.join("logs", filename)) else filename
    
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Smoothing the line slightly makes it look better on posters
            sns.lineplot(x=df["step"], y=df["loss"], label=label, color=color, linestyle=style, linewidth=3)
        except Exception as e:
            print(f"⚠️ Could not read {filename}: {e}")
    else:
        print(f"⚠️ Missing file: {filename} (Skipping line)")

plt.title("Training Stability & Convergence", fontsize=18, weight='bold')
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("poster_trajectories.png", dpi=300)
print("✅ Generated: poster_trajectories.png")


# --- PART 3: THE SPARSITY STORY (3-subplot comparison) ---
# Focus on the 5 main optimizers: SGD, AdamW, AdaBelief, Lion, Stable-SPAM
heroes_main = {
    "sgd": ("SGD\n(Baseline)", "#95a5a6"),
    "adamw": ("AdamW\n(Industry Standard)", "#3498db"),
    "adabelief": ("AdaBelief\n(Efficient Winner)", "#2ecc71"),
    "lion": ("Lion\n(Sparse King)", "#9b59b6"),
    "stable-spam": ("Stable-SPAM\n(Research Novelty)", "#f1c40f")
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Load data for all 5 heroes
data_loaded = {}
for file_key in heroes_main.keys():
    filename = f"{file_key}_seed42.csv"
    path = os.path.join("logs", filename) if os.path.exists(os.path.join("logs", filename)) else filename
    
    if os.path.exists(path):
        try:
            data_loaded[file_key] = pd.read_csv(path)
        except Exception as e:
            print(f"⚠️ Could not read {filename}: {e}")

# --- SUBPLOT 1: Loss Convergence ---
ax = axes[0]
for file_key, (label, color) in heroes_main.items():
    if file_key in data_loaded:
        df = data_loaded[file_key]
        ax.plot(df["step"], df["loss"], label=label, color=color, linewidth=2.5, marker='o', markersize=4)

ax.set_title("Loss Convergence", fontsize=14, weight='bold')
ax.set_xlabel("Training Steps", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# --- SUBPLOT 2: Sparsity Evolution ---
ax = axes[1]
for file_key, (label, color) in heroes_main.items():
    if file_key in data_loaded:
        df = data_loaded[file_key]
        ax.plot(df["step"], df["sparsity"], label=label, color=color, linewidth=2.5, marker='s', markersize=4)

ax.set_title("Sparsity Evolution (%)", fontsize=14, weight='bold')
ax.set_xlabel("Training Steps", fontsize=12)
ax.set_ylabel("Sparsity (%)", fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

# --- SUBPLOT 3: Update Magnitude (Drift) ---
ax = axes[2]
for file_key, (label, color) in heroes_main.items():
    if file_key in data_loaded:
        df = data_loaded[file_key]
        # Take absolute value of drift to show magnitude
        ax.plot(df["step"], np.abs(df["drift"]), label=label, color=color, linewidth=2.5, marker='^', markersize=4)

ax.set_title("Update Magnitude (|Drift|)", fontsize=14, weight='bold')
ax.set_xlabel("Training Steps", fontsize=12)
ax.set_ylabel("Absolute Drift", fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("poster_sparsity_convergence.png", dpi=300, bbox_inches='tight')
print("✅ Generated: poster_sparsity_convergence.png")
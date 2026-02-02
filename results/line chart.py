import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

# --- PART 2: THE HERO TRAJECTORIES (Line Chart) ---
# The optimizers we want to highlight
heroes = {
    "sgd": ("SGD (Baseline)", "#95a5a6", "--"),
    "adam": ("Adam (industry Standard)", "#3498db", "-"),
    "adabelief": ("AdaBelief (Best Final Loss)", "#e74c3c", ":"),
    "lamb": ("LAMB (Best Stability)", "#9b59b6", "-"),
    "stable-spam": ("Stable-SPAM (Best Sparsity and Convergence)", "#f1c40f", "-."),
    "adafactor": ("Adafactor (Over all Winner)", "#2ecc71", "-")
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


# --- THREE SIDE-BY-SIDE CHARTS: Loss, Sparsity, Drift ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Load data for all heroes
data_loaded = {}
for file_key in heroes.keys():
    filename = f"{file_key}_seed42.csv"
    path = os.path.join("logs", filename) if os.path.exists(os.path.join("logs", filename)) else filename
    
    if os.path.exists(path):
        try:
            data_loaded[file_key] = pd.read_csv(path)
        except Exception as e:
            print(f"⚠️ Could not read {filename}: {e}")

# --- SUBPLOT 1: Loss ---
ax = axes[0]
for file_key, (label, color, style) in heroes.items():
    if file_key in data_loaded:
        df = data_loaded[file_key]
        ax.plot(df["step"], df["loss"], label=label, color=color, linestyle=style, linewidth=2.5)

ax.set_title("Loss Convergence", fontsize=14, weight='bold')
ax.set_xlabel("Training Steps", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# --- SUBPLOT 2: Sparsity ---
ax = axes[1]
for file_key, (label, color, style) in heroes.items():
    if file_key in data_loaded:
        df = data_loaded[file_key]
        ax.plot(df["step"], df["sparsity"], label=label, color=color, linestyle=style, linewidth=2.5)

ax.set_title("Sparsity Evolution (%)", fontsize=14, weight='bold')
ax.set_xlabel("Training Steps", fontsize=12)
ax.set_ylabel("Sparsity (%)", fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# --- SUBPLOT 3: Drift (Update Magnitude) ---
ax = axes[2]
for file_key, (label, color, style) in heroes.items():
    if file_key in data_loaded:
        df = data_loaded[file_key]
        ax.plot(df["step"], np.abs(df["drift"]), label=label, color=color, linestyle=style, linewidth=2.5)

ax.set_title("Update Magnitude (|Drift|)", fontsize=14, weight='bold')
ax.set_xlabel("Training Steps", fontsize=12)
ax.set_ylabel("Absolute Drift", fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

fig.suptitle("Training Stability & Convergence", fontsize=16, weight='bold', y=1.00)
plt.tight_layout()
plt.savefig("hero_trajectories_3charts.png", dpi=300, bbox_inches='tight')
print("✅ Generated: hero_trajectories_3charts.png")
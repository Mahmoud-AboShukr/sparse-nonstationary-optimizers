import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style for research paper quality
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Load Data
optimizers = {
    "SGD": ("sgd_seed42.csv", "orange", "--"),
    "AdamW": ("adamw_seed42.csv", "blue", "-"),
    "Adafactor": ("adafactor_seed42.csv", "magenta", "-."),
    "Lion": ("lion_seed42.csv", "purple", "-")
}

data = {}
for name, (file, color, style) in optimizers.items():
    try:
        data[name] = pd.read_csv(f"logs/{file}") # Adjust path if needed
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping.")

# 2. Create Plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot A: Loss (Log Scale usually looks better for wide gaps, but linear is fine here)
for name, df in data.items():
    color, style = optimizers[name][1], optimizers[name][2]
    axes[0].plot(df["step"], df["loss"], label=name, color=color, linestyle=style, linewidth=2)
axes[0].set_title("Training Loss Stability")
axes[0].set_xlabel("Training Steps")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Plot B: Sparsity (The Key Finding)
for name, df in data.items():
    color, style = optimizers[name][1], optimizers[name][2]
    axes[1].plot(df["step"], df["sparsity"], label=name, color=color, linestyle=style, linewidth=2)
axes[1].set_title("Gradient Sparsity Evolution")
axes[1].set_xlabel("Training Steps")
axes[1].set_ylabel("Sparsity (%)")
axes[1].legend(loc='lower right')

# Plot C: Gradient Drift
for name, df in data.items():
    color, style = optimizers[name][1], optimizers[name][2]
    # Smoothing drift for readability (optional)
    axes[2].plot(df["step"], df["drift"], label=name, color=color, linestyle=style, linewidth=2, alpha=0.8)
axes[2].set_title("Gradient Directional Drift")
axes[2].set_xlabel("Training Steps")
axes[2].set_ylabel("Cosine Similarity (1.0 = Stable)")
axes[2].axhline(0, color='black', linewidth=1, linestyle=':') # Zero line

plt.tight_layout()
plt.savefig("plots/final_benchmark_results.png", dpi=300)
print("Plot saved to plots/final_benchmark_results.png")
plt.show()
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Change to results directory for file operations
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- LOAD ALL OPTIMIZER LOGS ---
logs_dir = "../logs"
optimizers = {}

for filename in os.listdir(logs_dir):
    if filename.endswith("_seed42.csv"):
        opt_name = filename.replace("_seed42.csv", "")
        filepath = os.path.join(logs_dir, filename)
        try:
            optimizers[opt_name] = pd.read_csv(filepath)
            print(f"[OK] Loaded: {opt_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {opt_name}: {e}")

print(f"\nTotal optimizers loaded: {len(optimizers)}\n")

# --- COMPUTE METRICS FOR EACH OPTIMIZER ---
results = []

for opt_name, df in optimizers.items():
    metrics = {
        "Optimizer": opt_name,
        # === LOSS METRICS ===
        "Final Loss": df["loss"].iloc[-1],
        "Avg Loss (excl. init)": df["loss"].iloc[1:].mean(),
        "Min Loss": df["loss"].min(),
        "Loss Variance": df["loss"].iloc[1:].var(),
        "Loss Stability (lower=better)": df["loss"].iloc[1:].std(),
        
        # === SPARSITY METRICS ===
        "Final Sparsity (%)": df["sparsity"].iloc[-1],
        "Avg Sparsity (%)": df["sparsity"].mean(),
        "Max Sparsity (%)": df["sparsity"].max(),
        "Sparsity at step 5": df["sparsity"].iloc[1] if len(df) > 1 else np.nan,
        
        # === UPDATE/DRIFT METRICS ===
        "Final Drift": df["drift"].iloc[-1],
        "Avg Absolute Drift": np.abs(df["drift"]).mean(),
        "Max Absolute Drift": np.abs(df["drift"]).max(),
        "Update Stability": np.abs(df["drift"]).std(),
        
        # === CONVERGENCE METRICS ===
        "Steps to convergence (<0.5 loss)": len(df[df["loss"] < 0.5]) if any(df["loss"] < 0.5) else np.nan,
        "Early convergence (at step 5)": df["loss"].iloc[1] if len(df) > 1 else np.nan,
    }
    results.append(metrics)

results_df = pd.DataFrame(results)

# --- RANKING FUNCTION ---
def rank_optimizers(df, metric_col, ascending=True, top_n=5):
    """Rank optimizers by a specific metric"""
    ranked = df[["Optimizer", metric_col]].dropna().sort_values(metric_col, ascending=ascending).head(top_n)
    return ranked.reset_index(drop=True)

# --- DISPLAY FULL ANALYSIS TABLE ---
print("=" * 150)
print("COMPREHENSIVE OPTIMIZER ANALYSIS - Full Metrics Table")
print("=" * 150)
print(results_df.to_string(index=False))
print("\n")

# --- KEY RANKINGS ---
print("=" * 150)
print("TOP PERFORMERS BY CATEGORY")
print("=" * 150)

rankings = {
    "[Best Final Loss]": rank_optimizers(results_df, "Final Loss", ascending=True),
    "[Best Final Sparsity]": rank_optimizers(results_df, "Final Sparsity (%)", ascending=False),
    "[Fastest Early Convergence]": rank_optimizers(results_df, "Early convergence (at step 5)", ascending=True),
    "[Most Stable Loss]": rank_optimizers(results_df, "Loss Stability (lower=better)", ascending=True),
    "[Most Stable Updates]": rank_optimizers(results_df, "Update Stability", ascending=True),
    "[Best Average Sparsity]": rank_optimizers(results_df, "Avg Sparsity (%)", ascending=False),
    "[Highest Peak Sparsity]": rank_optimizers(results_df, "Max Sparsity (%)", ascending=False),
    "[Best Overall Loss Variance]": rank_optimizers(results_df, "Loss Variance", ascending=True),
}

for category, ranking_df in rankings.items():
    print(f"\n{category}")
    print("-" * 100)
    print(ranking_df.to_string(index=False))

# --- COMPOSITE SCORING ---
print("\n" + "=" * 150)
print("COMPOSITE SCORES (Multi-Metric Ranking)")
print("=" * 150)

# Robust normalization function using percentile-based scaling
def robust_normalize(series, invert=False):
    """
    Normalize using percentile-based robust scaling to handle outliers and clustering.
    Returns values in [0, 100] range. Avoids extreme 0s and 100s.
    """
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    median = series.median()
    iqr = q75 - q25
    
    # If no IQR variance, fall back to min-max with better handling
    if iqr == 0:
        min_val = series.min()
        max_val = series.max()
        if min_val == max_val:
            return np.full(len(series), 50.0)  # All identical -> neutral
        # If small range, spread out more gracefully
        normalized = 50 + 40 * (series - median) / (max_val - min_val)
    else:
        # Robust z-score: (x - median) / IQR
        robust_z = (series - median) / iqr
        
        # Invert for "lower is better" metrics
        if invert:
            robust_z = -robust_z
        
        # Use sigmoid-like mapping to avoid extreme values
        # Tanh maps (-inf, inf) to (-1, 1), then scale to (5, 95)
        normalized = 50 + 45 * np.tanh(robust_z / 2) / np.tanh(1)
    
    normalized = np.clip(normalized, 1, 99)  # Avoid pure 0 and 100
    
    return normalized

# Normalize metrics to 0-100 scale for composite score
composite_df = results_df[["Optimizer"]].copy()

# Final Loss (lower is better, so invert)
composite_df["Loss Score"] = robust_normalize(results_df["Final Loss"], invert=True)

# Final Sparsity (higher is better)
composite_df["Sparsity Score"] = robust_normalize(results_df["Final Sparsity (%)"], invert=False)

# Update Stability (lower drift is better, so invert)
composite_df["Stability Score"] = robust_normalize(results_df["Update Stability"], invert=True)

# Loss Stability (lower variance is better)
composite_df["Convergence Score"] = robust_normalize(results_df["Loss Stability (lower=better)"], invert=True)

# Compute composite
composite_df["OVERALL SCORE"] = (
    0.35 * composite_df["Loss Score"] +  # Loss is most important
    0.35 * composite_df["Sparsity Score"] +  # Sparsity is core to your research
    0.20 * composite_df["Stability Score"] +  # Stability matters
    0.10 * composite_df["Convergence Score"]  # Convergence matters
)

composite_sorted = composite_df.sort_values("OVERALL SCORE", ascending=False).reset_index(drop=True)
print("\nWeighting: Loss (35%) | Sparsity (35%) | Stability (20%) | Convergence (10%)")
print(composite_sorted.to_string(index=False))

# --- EXPORT TO CSV ---
results_df.to_csv("optimizer_analysis.csv", index=False)
composite_sorted.to_csv("optimizer_composite_ranking.csv", index=False)

print("\n" + "=" * 150)
print("[EXPORT] optimizer_analysis.csv (full metrics)")
print("[EXPORT] optimizer_composite_ranking.csv (composite scores)")
print("=" * 150)

# --- INSIGHTS ---
print("\n" + "=" * 150)
print("KEY INSIGHTS")
print("=" * 150)

best_loss = results_df.loc[results_df["Final Loss"].idxmin()]
best_sparsity = results_df.loc[results_df["Final Sparsity (%)"].idxmax()]
best_stability = results_df.loc[results_df["Loss Stability (lower=better)"].idxmin()]
best_composite = composite_sorted.iloc[0]

print(f"\n[Best Loss] {best_loss['Optimizer']} ({best_loss['Final Loss']:.4f})")
print(f"[Best Sparsity] {best_sparsity['Optimizer']} ({best_sparsity['Final Sparsity (%)']:.1f}%)")
print(f"[Most Stable] {best_stability['Optimizer']} (std: {best_stability['Loss Stability (lower=better)']:.4f})")
print(f"[Overall Winner] {best_composite['Optimizer']} (Score: {best_composite['OVERALL SCORE']:.1f}/100)")

# Top 3 overall
print(f"\n[Top 3 Overall Performers]")
for idx, row in composite_sorted.head(3).iterrows():
    print(f"   {idx+1}. {row['Optimizer']:20s} - Score: {row['OVERALL SCORE']:6.1f} | Loss: {row['Loss Score']:5.1f} | Sparsity: {row['Sparsity Score']:5.1f}")

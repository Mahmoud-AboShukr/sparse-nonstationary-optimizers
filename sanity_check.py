import torch
import sys

# Ensure Python can find our 'src' folder
sys.path.append(".") 

from src.models.gpt import get_gpt_model
from src.metrics.diagnostics import GradientDiagnostics

def run_check():
    print("--- 1. Setting up Environment ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n--- 2. Loading Model ---")
    # Using the research config
    model = get_gpt_model(model_type="research-sparse").to(device)
    print("Model loaded successfully.")

    print("\n--- 3. initializing Diagnostics ---")
    diagnostics = GradientDiagnostics(model)
    
    print("\n--- 4. Simulating Training Step ---")
    # Create fake text data (Batch size 4, sequence length 32)
    dummy_input = torch.randint(0, 50257, (4, 32)).to(device)
    
    # Forward pass
    outputs = model(dummy_input, labels=dummy_input)
    loss = outputs.loss
    print(f"Forward pass complete. Initial Loss: {loss.item():.4f}")
    
    # Backward pass (Calculates gradients)
    loss.backward()
    print("Backward pass complete (Gradients computed).")

    print("\n--- 5. Testing Metrics Code ---")
    # Get the flat gradient vector
    flat_grads = diagnostics.get_gradient_stats()
    
    # Measure Sparsity
    sparsity = diagnostics.measure_sparsity(flat_grads)
    print(f"Gradient Sparsity: {sparsity:.2f}% (Should be low for random data, < 100%)")
    
    # Measure Drift (Step 0)
    drift_0 = diagnostics.measure_drift(flat_grads, step=0)
    print(f"Drift Score (Step 0): {drift_0:.4f} (Should be exactly 1.0)")

    # Simulate Step 1 (change data slightly to create drift)
    model.zero_grad()
    dummy_input2 = torch.randint(0, 50257, (4, 32)).to(device)
    loss2 = model(dummy_input2, labels=dummy_input2).loss
    loss2.backward()
    flat_grads_2 = diagnostics.get_gradient_stats()
    
    # Measure Drift (Step 1)
    drift_1 = diagnostics.measure_drift(flat_grads_2, step=1)
    print(f"Drift Score (Step 1): {drift_1:.4f} (Should be < 1.0)")

    print("\n✅ SUCCESS: All systems go.")

if __name__ == "__main__":
    run_check()
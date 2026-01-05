import argparse
import torch
import csv
import os
import time
from tqdm import tqdm
from torch.amp import autocast, GradScaler # Updated for PyTorch 2.x

from src.models.gpt import get_gpt_model
from src.data.loader import get_dataloader
from src.metrics.diagnostics import GradientDiagnostics
from src.optimizers.factory import get_optimizer
from transformers import AutoTokenizer

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Experiment: {args.optimizer} (Seed: {args.seed}) ---")
    print(f"--- Strategy: Batch {args.batch_size} | Accum {args.accum_steps} | Mixed Precision ON ---")

    # 1. Prepare Data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataloader = get_dataloader(tokenizer, batch_size=args.batch_size)

    # 2. Model & Optimizer
    model = get_gpt_model("research-sparse").to(device)
    optimizer = get_optimizer(model, args)
    
    # 3. Diagnostics & Logging
    diagnostics = GradientDiagnostics(model)
    log_file = os.path.join("logs", f"{args.optimizer}_seed{args.seed}.csv")
    
    # Check if logs folder exists
    os.makedirs("logs", exist_ok=True)
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "sparsity", "drift", "time_sec"])

    # 4. Mixed Precision Scaler
    scaler = GradScaler() # Replaces standard backward pass

    model.train()
    step = 0
    global_step = 0 # Tracks actual optimizer updates
    start_time = time.time()
    
    print("Training started...")
    progress_bar = tqdm(total=args.steps)
    
    optimizer.zero_grad() # Initialize gradients
    data_iter = iter(dataloader)
    
    while global_step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # --- A. Forward Pass (with Autocast for Memory Saving) ---
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs, labels=labels)
            # Normalize loss because we are summing up gradients over multiple steps
            loss = outputs.loss / args.accum_steps

        # --- B. Backward Pass (Scaled) ---
        scaler.scale(loss).backward()

        # --- C. Optimizer Step (Only every N steps) ---
        step += 1
        if step % args.accum_steps == 0:
            
            # 1. Record Metrics (Optional: doing it here is cleaner)
            if global_step % args.log_interval == 0:
                # We must unscale gradients briefly to measure them correctly
                scaler.unscale_(optimizer) 
                flat_grads = diagnostics.get_gradient_stats()
                
                if flat_grads is not None:
                    sparsity = diagnostics.measure_sparsity(flat_grads)
                    drift = diagnostics.measure_drift(flat_grads, global_step)
                    
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([global_step, loss.item() * args.accum_steps, sparsity, drift, time.time()-start_time])
                    
                    progress_bar.set_description(f"Loss: {loss.item() * args.accum_steps:.4f} | Sparsity: {sparsity:.1f}%")

            # 2. Actual Update
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            global_step += 1
            progress_bar.update(1)

    print(f"\nExperiment Complete. Results saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, required=True, help="sgd, adamw, adafactor, lion")
    parser.add_argument("--steps", type=int, default=50, help="Number of VALID optimization steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Keep this at 1 for 4GB VRAM")
    parser.add_argument("--accum_steps", type=int, default=8, help="Simulate larger batches (e.g. 8*1 = 8)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=5)
    
    args = parser.parse_args()
    train(args)
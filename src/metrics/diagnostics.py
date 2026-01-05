import torch
import numpy as np

class GradientDiagnostics:
    def __init__(self, model):
        """
        Initializes the diagnostic tool.
        We keep a 'shadow' copy of gradients to measure drift over time.
        """
        self.model = model
        self.history = {} # To store running averages if needed
        
    def get_gradient_stats(self):
        """
        Collects all gradients from the model into a single flat vector
        for easy statistical analysis.
        """
        grads = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Flatten and detach so we don't mess up the training graph
                grads.append(param.grad.view(-1).detach())
        
        if not grads:
            return None
            
        # Concatenate into one giant vector (e.g., 100M params -> 100M length vector)
        return torch.cat(grads)

    def measure_sparsity(self, flat_grad, threshold=1e-5):
        """
        Calculates what % of gradients are effectively zero.
        Ref: 'Sparse gradients — Only a small subset of parameters are updated' [cite: 1]
        """
        total_params = flat_grad.numel()
        # count how many are smaller than threshold
        zeros = (torch.abs(flat_grad) < threshold).sum().item()
        
        return (zeros / total_params) * 100.0

    def measure_drift(self, flat_grad, step):
        """
        Measures 'Non-stationary gradients' by comparing current gradient
        direction to the historical average.
        Ref: 'Gradient statistics drift over time' [cite: 1]
        """
        # If it's the first step, we have no history to compare to
        if step == 0:
            self.running_avg = flat_grad.clone()
            return 1.0 # Perfect alignment with itself

        # 1. Calculate Cosine Similarity between Current Grad and History
        # Sim = (A . B) / (|A| * |B|)
        dot_product = torch.dot(flat_grad, self.running_avg)
        norm_curr = torch.norm(flat_grad)
        norm_hist = torch.norm(self.running_avg)
        
        similarity = dot_product / (norm_curr * norm_hist + 1e-8)
        
        # 2. Update the running average (Simulating a momentum buffer)
        # Using a simple exponential moving average (EMA)
        beta = 0.9
        self.running_avg = beta * self.running_avg + (1 - beta) * flat_grad
        
        return similarity.item()
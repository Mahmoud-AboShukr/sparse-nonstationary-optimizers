import torch
from transformers.optimization import Adafactor

def get_optimizer(model, args):
    """
    Factory function to initialize the requested optimizer.
    Ref: 'Compare optimizer behaviour under identical training conditions'
    """
    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    opt_name = args.optimizer.lower()
    lr = args.lr

    print(f"Initializing Optimizer: {opt_name.upper()}")

    if opt_name == "sgd":
        # Baseline: known to fail under sparse gradients
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    
    elif opt_name == "adamw":
        # Industry standard
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    elif opt_name == "adafactor":
        # Memory efficient, sparse-friendly
        # We disable 'scale_parameter' to keep the LR fixed for fair comparison
        return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False)
    
    elif opt_name == "lion":
        # Optional: Sign-based update
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=lr, weight_decay=0.01)
        except ImportError:
            raise ImportError("Lion optimizer not found. Run: pip install lion-pytorch")
    
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
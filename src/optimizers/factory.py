import torch
import torch.optim as optim
from transformers.optimization import Adafactor

def get_optimizer(model, args):
    """
    Factory function to initialize the requested optimizer.
    """
    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    opt_name = args.optimizer.lower()
    lr = args.lr
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 0.01

    print(f"Initializing Optimizer: {opt_name.upper()}")

    # =========================================
    # 1. FRIEND'S LIST (Classics & Scaling)
    # =========================================
    if opt_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    elif opt_name == "adam":
        return optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        return optim.RMSprop(params, lr=lr, alpha=0.99, weight_decay=weight_decay)
    elif opt_name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adadelta":
        return optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    
    # Large Batch / Scaling
    elif opt_name == "lamb":
        try:
            import torch_optimizer as topt
            return topt.Lamb(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("Torch-Optimizer not found. Run: pip install torch-optimizer")
            
    elif opt_name == "adabelief":
        try:
            from adabelief_pytorch import AdaBelief
            return AdaBelief(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("AdaBelief not found. Run: pip install adabelief-pytorch")

    elif opt_name == "came":
        try:
            from came_pytorch import CAME
            return CAME(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("CAME not found. Run: pip install git+https://github.com/yangluo7/CAME.git")

    # =========================================
    # 2. RAMZY'S LIST (Modern & Efficiency)
    # =========================================
    elif opt_name == "adafactor":
        return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, weight_decay=weight_decay)
    
    elif opt_name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("Lion not found. Run: pip install lion-pytorch")

    elif opt_name == "galore":
        try:
            from galore_torch import GaLoreAdamW
            return GaLoreAdamW(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("GaLore not found. Run: pip install galore-torch")

    # Quantized Optimizers (8-bit)
    elif "8bit" in opt_name:
        try:
            import bitsandbytes as bnb
            if "adamw" in opt_name:
                return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
            else:
                return bnb.optim.Adam8bit(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("BitsAndBytes not found. Run: pip install bitsandbytes")

    # Adam-Mini and Q-Adam-Mini
    elif "adam-mini" in opt_name:
        try:
            from adam_mini import Adam_mini
            # Infer config from model (GPT-specific hack)
            n_embd = getattr(model.config, "n_embd", 2048)
            n_head = getattr(model.config, "n_head", 16)
            n_query_groups = getattr(model.config, "n_head", 16)
            
            # Check for Quantization flag
            use_quant = True if "q-" in opt_name else False
            if use_quant: print("-> Enabling Quantization for Adam-mini")

            return Adam_mini(
                named_parameters=model.named_parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                model_sharding=False,
                dim=n_embd, n_heads=n_head, n_kv_heads=n_query_groups,
                use_quant=use_quant
            )
        except ImportError:
            raise ImportError("Adam-mini not found. Run: pip install git+https://github.com/zyushun/Adam-mini.git")

    # =========================================
    # 3. MANUAL DOWNLOADS (Experimental)
    # =========================================
    # These typically don't have pip packages yet.
    # You must download the .py file and put it in src/optimizers/
    
    elif opt_name == "spam":
        try:
            from .spam import SPAM
            return SPAM(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("SPAM code missing. Download 'spam.py' to src/optimizers/")

    elif opt_name == "stable-spam":
        try:
            from .stable_spam import StableSPAM
            return StableSPAM(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("Stable-SPAM code missing. Download 'stable_spam.py' to src/optimizers/")

    elif opt_name == "adamem":
        try:
            from .adamem import AdaMem
            return AdaMem(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("AdaMem code missing. Download 'adamem.py' to src/optimizers/")
            
    elif opt_name == "slamb":
        try:
            from .slamb import SLAMB
            return SLAMB(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("SLAMB code missing. Download 'slamb.py' to src/optimizers/")

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
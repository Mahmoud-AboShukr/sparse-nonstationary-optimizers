try:
    import torch
except ImportError as e:
    raise ImportError(
        "Missing dependency: install 'torch'. For CUDA-supported wheels use the PyTorch install guide: https://pytorch.org/get-started/locally/\n"
        "Example (CUDA 12.4, pip): python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n"
        "Example (CPU-only, pip): python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    ) from e

try:
    from transformers import GPT2Config, GPT2LMHeadModel
except ImportError as e:
    raise ImportError(
        "Missing dependency: install 'transformers'. Run: python -m pip install transformers"
    ) from e

# Device selection: prefer CUDA if available
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_gpt_model(model_type="small"):
    """
    Creates a GPT model based on the research project specs.
    
    Ref: 'Embedding Size 1024-2048, intentionally large to promote sparsity' 
    """
    if model_type == "research-sparse":
        # Custom config to match the paper's 'sparse gradient' requirement
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1024,        # Large embedding size (Paper objective)
            n_layer=12,         # 12-24 layers [cite: 145]
            n_head=16,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
    else:
        # Standard GPT-2 Small (124M params) for sanity checking
        config = GPT2Config.from_pretrained("gpt2")
        # We initialize from scratch, not pretrained weights, as this is a training study
    
    model = GPT2LMHeadModel(config)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Quick test if run directly
    print(f"Using device: {_device}")
    if _device.type == "cuda":
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "<no device>"
        print(
            f"CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}, "
            f"devices: {device_count}, name: {device_name}"
        )

    model = get_gpt_model("research-sparse").to(_device)
    print(f"Model created with {count_parameters(model):,} parameters and moved to {_device}.")
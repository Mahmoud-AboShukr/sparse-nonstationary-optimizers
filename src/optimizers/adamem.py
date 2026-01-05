"""AdaMem placeholder implementation

This file provides a minimal AdaMem optimizer implementation (wrapper around
torch.optim.AdamW) so the repository can import `src.optimizers.adamem.AdaMem`.

If you want the full AdaMem algorithm, replace this file with the official
implementation or extend this class with the AdaMem logic.
"""
from typing import Iterable, Tuple
import torch
from torch.optim import Optimizer


class AdaMem(torch.optim.AdamW):
    """Minimal AdaMem-compatible optimizer implemented as a thin wrapper
    around :class:`torch.optim.AdamW` to satisfy imports and tests.

    This acts as a drop-in replacement for AdamW with the same constructor
    signature. It intentionally does not implement the AdaMem-specific
    optimizations; replace this with a real AdaMem implementation if needed.
    """

    def __init__(self, params: Iterable[torch.nn.parameter.Parameter], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    # The base AdamW implementation already provides step(), zero_grad(),
    # state_dict(), load_state_dict(), etc. Keep the interface minimal here.


__all__ = ["AdaMem"]

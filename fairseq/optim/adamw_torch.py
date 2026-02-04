# my_user_dir/optim/adamw_torch.py
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch

from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer


def _parse_betas(betas: Any) -> Tuple[float, float]:
    # fairseq CLI often passes betas as a string like "(0.9, 0.999)"
    if isinstance(betas, tuple):
        return betas
    if isinstance(betas, str):
        betas = betas.strip()
        return tuple(map(float, betas.strip("()").split(",")))  # type: ignore
    raise TypeError(f"Unsupported betas type: {type(betas)}")


@dataclass
class AdamWTorchConfig(FairseqDataclass):
    lr: float = field(default=1e-3, metadata={"help": "learning rate"})
    betas: str = field(default="(0.9, 0.999)", metadata={"help": "betas for AdamW"})
    eps: float = field(default=1e-8, metadata={"help": "epsilon"})
    weight_decay: float = field(default=0.01, metadata={"help": "weight decay (AdamW style)"})
    amsgrad: bool = field(default=False, metadata={"help": "use AMSGrad variant"})


@register_optimizer("adamw_torch", dataclass=AdamWTorchConfig)
class FairseqAdamWTorch(FairseqOptimizer):
    """
    Fairseq optimizer wrapper around torch.optim.AdamW.
    Use with: --optimizer adamw_torch
    """
    def __init__(self, cfg: AdamWTorchConfig, params):
        super().__init__(cfg)
        betas = _parse_betas(cfg.betas)

        self._optimizer = torch.optim.AdamW(
            params,
            lr=cfg.lr,
            betas=betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            amsgrad=cfg.amsgrad,
            fused=True
        )

    @property
    def optimizer(self):
        return self._optimizer

    def step(self, closure=None, **kwargs):
        return self._optimizer.step(closure=closure)

    def zero_grad(self):
        self._optimizer.zero_grad()

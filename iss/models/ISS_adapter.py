"""Shared and domain-specific low-rank updates used by the formal R1 model."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SaRALinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, r: int = 4,
                 r_shared: int = 1, lora_alpha: float = 1.0,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        if not 0 <= r_shared <= r:
            raise ValueError((r, r_shared))
        self.r = int(r)
        self.r_shared = int(r_shared)
        self.r_specific = int(r - r_shared)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / self.r if self.r else 1.0
        self.lora_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        if self.r_shared:
            self.sara_A_shared = nn.Parameter(torch.zeros(in_features, self.r_shared))
            self.sara_B_shared = nn.Parameter(torch.empty(self.r_shared, out_features))
            nn.init.kaiming_uniform_(self.sara_B_shared, a=math.sqrt(5))
        else:
            self.register_parameter("sara_A_shared", None)
            self.register_parameter("sara_B_shared", None)
        if self.r_specific:
            self.sara_A_spec = nn.Parameter(torch.zeros(in_features, self.r_specific))
            self.sara_B_spec = nn.Parameter(torch.empty(self.r_specific, out_features))
            nn.init.kaiming_uniform_(self.sara_B_spec, a=math.sqrt(5))
        else:
            self.register_parameter("sara_A_spec", None)
            self.register_parameter("sara_B_spec", None)

    def forward(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.weight, self.bias)
        dropped = self.lora_dropout(x)
        delta = torch.zeros_like(y)
        if self.sara_A_shared is not None:
            delta = delta + (dropped @ self.sara_A_shared) @ self.sara_B_shared
        if self.sara_A_spec is not None:
            delta = delta + (dropped @ self.sara_A_spec) @ self.sara_B_spec
        return y + self.scaling * delta

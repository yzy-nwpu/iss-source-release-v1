"""Predict a domain index from current utterance encoder features."""
from __future__ import annotations

from pathlib import Path

import torch

from iss.models.network import make_sepformer
from iss.routing.dpan import load_selector
from iss.training.checkpoint import initialize_from_checkpoint


@torch.inference_mode()
def predict_domain(feature: torch.Tensor, selector_checkpoint: str | Path,
                   device: str = "cpu") -> tuple[int, str, torch.Tensor]:
    selector, payload = load_selector(selector_checkpoint, device)
    value = feature.to(device=device, dtype=torch.float32).reshape(1, -1)
    logits = selector(value)
    index = int(logits.argmax(dim=-1).item())
    return index, payload["class_map"][index], logits[0]


def load_frozen_encoder(task: str, checkpoint: str | Path, device: str = "cpu"):
    model = make_sepformer(task, 0, 0)
    initialize_from_checkpoint(model, checkpoint)
    model.eval().requires_grad_(False).to(device)
    return model["encoder"]

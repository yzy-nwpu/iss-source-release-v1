"""Construct the full waveform SepFormer used by the ISS experiments."""
from __future__ import annotations

from iss.models.iss_model import build_model, convert_base, forward
from iss.models.iss_model import adapter_state, load_adapters, reset_specific


def make_sepformer(task: str, shared_rank: int, specific_rank: int, architecture=None):
    if task not in {"separation", "denoising"}:
        raise ValueError(f"Unknown task: {task}")
    speakers = 2 if task == "separation" else 1
    if architecture and "activation_checkpoint_layers" in architecture:
        from iss.models.iss_batch_model import build_model as build_checkpointed
        return build_checkpointed(speakers, shared_rank, specific_rank,
                                  int(architecture["activation_checkpoint_layers"]), architecture)
    return build_model(speakers, shared_rank, specific_rank, architecture)


__all__ = ["make_sepformer", "build_model", "convert_base", "forward",
           "adapter_state", "load_adapters", "reset_specific"]

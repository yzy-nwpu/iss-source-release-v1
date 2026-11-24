"""Evaluation-time model construction and stage/route state composition."""
from __future__ import annotations

from pathlib import Path

import torch

from iss.models.network import make_sepformer
from iss.training.checkpoint import initialize_from_checkpoint
from iss.training.state import adapter_state, compose_adapters, load_adapters, validate_stage_bank


def load_trajectory(config: dict, device: str = "cpu"):
    task = config["task"]
    domains = list(config["domains"])
    domain_order = list(config["domain_order"])
    if len(domain_order) != len(domains) + 1 or domain_order[1:] != domains:
        raise ValueError("domain_order must be [source domain, *incremental domains]")
    if domain_order[0] != config["source"]["domain"]:
        raise ValueError("domain_order source domain does not match source.domain")
    model = make_sepformer(task, *config["model"]["ranks"], config["model"])
    initialize_from_checkpoint(model, config["base_checkpoint"])
    for module in model.modules():
        if hasattr(module, "scaling") and hasattr(module, "sara_A_spec"):
            module.scaling = float(config["model"].get("adapter_scaling", 0.25))
    template = adapter_state(model)
    checkpoint_dir = Path(config["trajectory_dir"]).expanduser()
    states = {}
    for stage, domain in enumerate(domains, 1):
        path = checkpoint_dir / f"stage{stage}_{domain}" / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing stage checkpoint: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["stage"] != stage or state["domain"] != domain:
            raise ValueError(f"Incorrect stage mapping in {path}")
        if state.get("config") != config:
            raise ValueError(f"Stage checkpoint was trained with a different config: {path}")
        states[stage] = state
    base_hash = states[1]["base_fingerprint"]
    validate_stage_bank(states, domain_order, base_hash)
    model.eval().requires_grad_(False).to(device)
    return model, template, states


def activate(model, template, states, final_stage: int, expert_stage: int) -> None:
    load_adapters(model, compose_adapters(template, states, final_stage, expert_stage))

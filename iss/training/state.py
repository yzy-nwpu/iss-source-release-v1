"""Stage state creation, freezing, adapter composition, and atomic persistence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def training_fingerprint(config: dict, input_paths: list[str | Path]) -> str:
    """Bind resumable state to the exact config and manifest contents."""
    inputs = []
    for value in sorted((Path(path).expanduser() for path in input_paths), key=str):
        if not value.is_file():
            raise FileNotFoundError(f"Training input manifest does not exist: {value}")
        digest = hashlib.sha256(value.read_bytes()).hexdigest()
        inputs.append({"path": str(value.resolve()), "sha256": digest})
    payload = {"config": config, "inputs": inputs}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def adapter_state(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if "sara_" in key}


def set_trainable(model) -> list[torch.nn.Parameter]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("sara_" in name)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("The model has no trainable CSA/DSA parameters")
    return trainable


def freeze_backbone(model) -> str:
    state = model.state_dict()
    digest = hashlib.sha256()
    for key in sorted(state):
        if "sara_" in key:
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def compose_adapters(template: dict[str, torch.Tensor], states: dict[int, dict],
                     stage: int, expert: int) -> dict[str, torch.Tensor]:
    """Use CSA from the requested model stage and DSA from its owning domain."""
    if not 0 <= expert <= stage:
        raise ValueError(f"expert {expert} must be in [0, {stage}]")
    composed = {}
    for key, initial in template.items():
        if not key.endswith(("_shared", "_spec")):
            raise ValueError(f"Non-adapter state supplied: {key}")
        if stage == 0 or (key.endswith("_spec") and expert == 0):
            value = torch.zeros_like(initial)
        else:
            owner = stage if key.endswith("_shared") else expert
            if owner not in states:
                raise KeyError(f"Missing adapter state for stage {owner}")
            value = states[owner]["adapters"][key]
        if value.shape != initial.shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid adapter tensor {key}")
        composed[key] = value
    return composed


def load_adapters(model, adapters: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    expected = {key for key in current if "sara_" in key}
    if set(adapters) != expected:
        raise ValueError(f"Adapter key mismatch: missing={sorted(expected-set(adapters))[:5]}, "
                         f"unexpected={sorted(set(adapters)-expected)[:5]}")
    current.update(adapters)
    model.load_state_dict(current, strict=True)


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    os.replace(temporary, path)


def create_stage_state(stage: int, domain: str, adapters: dict[str, torch.Tensor],
                       domain_order: list[str], config: dict, base_fingerprint: str,
                       validation_score: float, epoch: int) -> dict:
    if stage < 1 or stage >= len(domain_order):
        raise ValueError("Stage is outside the configured domain sequence")
    if domain_order[stage] != domain:
        raise ValueError("Stage/domain does not follow the configured arrival order")
    return {"format_version": 1, "stage": stage, "domain": domain,
            "domain_order": list(domain_order), "class_map": list(domain_order[1:stage + 1]),
            "adapters": {key: value.detach().cpu().clone() for key, value in adapters.items()},
            "config": dict(config), "base_fingerprint": base_fingerprint,
            "validation_score": float(validation_score), "epoch": int(epoch)}


def validate_stage_bank(states: dict[int, dict], domain_order: list[str], base_fingerprint: str) -> None:
    if sorted(states) != list(range(1, max(states, default=0) + 1)):
        raise ValueError("Stage bank must be contiguous and one-based")
    for stage, state in states.items():
        if stage >= len(domain_order):
            raise ValueError("Stage bank exceeds the configured domain sequence")
        if state["stage"] != stage or state["domain"] != domain_order[stage]:
            raise ValueError(f"Stage {stage} has a mismatched domain mapping")
        if state["domain_order"] != domain_order:
            raise ValueError(f"Stage {stage} has a mismatched full domain order")
        if state["base_fingerprint"] != base_fingerprint:
            raise ValueError(f"Stage {stage} was built from a different frozen base")
        if state["class_map"] != domain_order[1:stage + 1]:
            raise ValueError(f"Stage {stage} has a mismatched route index map")

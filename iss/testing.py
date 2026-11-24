"""Short synthetic smoke coverage for the actual model, training, and DPAN code."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from iss.models.network import make_sepformer
from iss.models.iss_model import forward
from iss.routing.dpan import (fit_from_features, load_memory, load_selector,
                              sample_multivariate_laplace)
from iss.routing.inference import predict_domain
from iss.training.checkpoint import initialize_from_checkpoint
from iss.training.common import fit_epoch, read_rows
from iss.training.state import (adapter_state, atomic_torch_save, compose_adapters,
                                freeze_backbone, load_adapters, set_trainable,
                                validate_stage_bank)


def _manifests(root: Path, task: str, domain: str, offset: int, sample_rate: int = 8000):
    records = []
    length = 4096
    t = np.arange(length, dtype=np.float32) / sample_rate
    for index in range(8):
        frequency = 180 + 17 * index + 350 * offset
        target = (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        speech = root / f"{task}_{domain}_{index}_speech.wav"
        sf.write(speech, target, sample_rate)
        targets = {"s1_wav": str(speech)}
        mixture = target.copy()
        if task == "separation":
            second = (0.15 * np.sin(2 * np.pi * (frequency + 250) * t)).astype(np.float32)
            second_path = root / f"{task}_{domain}_{index}_s2.wav"
            sf.write(second_path, second, sample_rate)
            targets["s2_wav"] = str(second_path)
            mixture += second
        else:
            noise = (0.025 * np.sin(2 * np.pi * (frequency + 1250) * t)).astype(np.float32)
            noise_path = root / f"{task}_{domain}_{index}_noise.wav"
            sf.write(noise_path, noise, sample_rate)
            targets["noise_wav"] = str(noise_path)
            mixture += noise
        mixture_path = root / f"{task}_{domain}_{index}_mix.wav"
        sf.write(mixture_path, mixture, sample_rate)
        records.append({"utterance_id": f"{task}:{domain}:{index}", "task": task,
                        "domain": domain, "split": "tr", "sample_rate": sample_rate,
                        "mix_wav": str(mixture_path), "targets": targets})
    manifest = root / f"{task}_{domain}.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    return manifest


def _save_base(task: str, architecture: dict, path: Path):
    base = make_sepformer(task, 0, 0, architecture)
    torch.save({"state_dict": base.state_dict()}, path)
    return path


def _smoke_task(task: str, root: Path, architecture: dict, route_results: list) -> dict:
    base_path = _save_base(task, architecture, root / f"{task}_base.pt")
    model = make_sepformer(task, 1, 1, architecture)
    initialize_from_checkpoint(model, base_path)
    trainable = set_trainable(model)
    backbone_before = freeze_backbone(model)
    domain1 = _manifests(root, task, "domain1", 0)
    domain2 = _manifests(root, task, "domain2", 1)
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    train_rows = read_rows(domain1)
    loss1 = fit_epoch(model, train_rows, task, optimizer, "cpu", 2,
                      shared_reference=None, seed=10)
    first_state = adapter_state(model)
    groups = {"shared": False, "specific": False}
    for name, parameter in model.named_parameters():
        if "sara_" in name and parameter.grad is not None:
            groups["shared" if "_shared" in name else "specific"] |= bool(parameter.grad.abs().sum() > 0)
    if not all(groups.values()):
        raise AssertionError(f"CSA/DSA gradients missing: {groups}")

    inherited = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                 if "sara_" in name and name.endswith("_shared")}
    current = adapter_state(model)
    for key in current:
        if key.endswith("_shared"):
            current[key] = first_state[key].clone()
    load_adapters(model, current)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "sara_" in name and "_spec" in name:
                if "sara_A_" in name:
                    parameter.zero_()
                elif "sara_B_" in name:
                    torch.nn.init.kaiming_uniform_(parameter, a=np.sqrt(5))
    train_rows2 = read_rows(domain2)
    loss2 = fit_epoch(model, train_rows2, task, optimizer, "cpu", 2,
                      shared_reference=inherited, seed=20)
    second_state = adapter_state(model)
    if freeze_backbone(model) != backbone_before:
        raise AssertionError("Frozen SepFormer backbone changed during adapter updates")
    domain_order = ["source", "domain1", "domain2"]
    states = {
        1: {"stage": 1, "domain": "domain1", "domain_order": domain_order,
            "class_map": domain_order[1:2], "adapters": first_state,
            "base_fingerprint": backbone_before},
        2: {"stage": 2, "domain": "domain2", "domain_order": domain_order,
            "class_map": domain_order[1:3], "adapters": second_state,
            "base_fingerprint": backbone_before},
    }
    validate_stage_bank(states, domain_order, backbone_before)
    template = adapter_state(model)
    for predicted_index, predicted_domain in route_results:
        expert = predicted_index + 1
        if predicted_domain != domain_order[expert]:
            raise AssertionError("DPAN class map did not resolve to the configured domain")
        composed = compose_adapters(template, states, 2, expert)
        key = next(key for key in composed if key.endswith("_spec"))
        if not torch.equal(composed[key], states[expert]["adapters"][key]):
            raise AssertionError("Predicted domain did not select its corresponding DSA")

    checkpoint = root / f"{task}_stage2.pt"
    atomic_torch_save({"stage": 2, "domain": "domain2", "domain_order": domain_order,
                       "class_map": ["domain1", "domain2"], "adapters": second_state,
                       "base_fingerprint": backbone_before}, checkpoint)
    mutated = {key: torch.zeros_like(value) for key, value in second_state.items()}
    load_adapters(model, mutated)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_adapters(model, restored["adapters"])
    if any(not torch.equal(model.state_dict()[key].cpu(), value) for key, value in second_state.items()):
        raise AssertionError("Stage checkpoint did not restore adapter state")

    sample_mix = torch.randn(1, 1024)
    prediction = forward(model, sample_mix)
    if prediction.shape[:2] != sample_mix.shape:
        raise AssertionError(f"Output length mismatch: {prediction.shape}, {sample_mix.shape}")
    return {"task": task, "forward_shape": list(prediction.shape),
            "stage1_loss": loss1, "stage2_loss": loss2,
            "csa_gradient": groups["shared"], "dsa_gradient": groups["specific"],
            "frozen_backbone_unchanged": True, "new_domain_created": True,
            "checkpoint_restored": True, "predicted_dsa_selection": True}


def _smoke_dpan(root: Path) -> tuple[dict, list]:
    rng = np.random.RandomState(7)
    first = rng.normal(loc=-2.0, scale=0.2, size=(48, 16)).astype(np.float32)
    second = rng.normal(loc=2.0, scale=0.2, size=(48, 16)).astype(np.float32)
    first_path, second_path = root / "domain1.npy", root / "domain2.npy"
    np.save(first_path, first)
    np.save(second_path, second)
    config = {"epochs": 30, "batch_size": 16, "lr": 1.0, "momentum": 0.9,
              "distribution": "laplace"}
    output = root / "dpan"
    fit_from_features([("domain1", str(first_path)), ("domain2", str(second_path))],
                      output, config, "separation", ["domain1", "domain2"], seed=11)
    memory1 = load_memory(output / "separation/stage1_domain1/compressed_memory.npz")
    memory2 = load_memory(output / "separation/stage2_domain2/compressed_memory.npz")
    replay = sample_multivariate_laplace(memory1, 48, seed=9)
    if replay.shape != first.shape or not np.isfinite(replay).all():
        raise AssertionError("Multivariate Laplace pseudo-feature sampling failed")
    selector_path = output / "separation/stage2_domain2/selector_seed11.pt"
    selector, payload = load_selector(selector_path)
    if payload["class_map"] != ["domain1", "domain2"]:
        raise AssertionError("Saved DPAN class map did not survive reload")
    route_results = []
    for feature in (first.mean(0), second.mean(0)):
        index, domain, _logits = predict_domain(torch.from_numpy(feature), selector_path)
        route_results.append((index, domain))
    predictions = [index for index, _domain in route_results]
    if predictions != [0, 1]:
        raise AssertionError(f"DPAN failed to distinguish domains: {predictions}")
    return {"distribution_fit": True, "pseudo_features": int(replay.shape[0]),
            "history_and_current_joint_training": True, "predicted_indices": predictions,
            "class_map_restored": payload["class_map"],
            "training_updates": 30 * 3}, route_results


def run_smoke() -> dict:
    torch.manual_seed(13)
    architecture = {"channels": 16, "kernel_size": 8, "stride": 4,
                    "intra_layers": 1, "inter_layers": 1, "heads": 4,
                    "d_ffn": 32, "dual_path_layers": 1, "chunk_size": 8,
                    "activation_checkpoint_layers": 1}
    with tempfile.TemporaryDirectory(prefix="iss-source-smoke-") as name:
        root = Path(name)
        dpan_report, route_results = _smoke_dpan(root)
        reports = [_smoke_task(task, root, architecture, route_results)
                   for task in ("separation", "denoising")]
        reports.append(dpan_report)
        return {"scope": "synthetic short-waveform check; not paper-scale training",
                "temporary_artifacts": "outside candidate repository", "checks": reports}


def main() -> None:
    print(json.dumps(run_smoke(), indent=2))


if __name__ == "__main__":
    main()

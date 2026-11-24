"""Train one task's sequential CSA/DSA trajectory from a supplied source base."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from iss.models.network import make_sepformer
from iss.training.checkpoint import initialize_from_checkpoint
from iss.training.common import fit_epoch, read_rows, validate
from iss.training.state import (adapter_state, atomic_json_save, atomic_torch_save,
                                freeze_backbone, load_adapters, set_trainable,
                                training_fingerprint)


def train_incremental(config: dict) -> Path:
    task = config["task"]
    domains = list(config["domains"])
    domain_order = list(config["domain_order"])
    if len(domain_order) != len(domains) + 1 or domain_order[1:] != domains:
        raise ValueError("domain_order must be [source domain, *incremental domains]")
    if domain_order[0] != config["source"]["domain"]:
        raise ValueError("domain_order source domain does not match source.domain")
    manifests = config["manifests"]
    model_config = config["model"]
    training = config["training"]
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = config.get("base_checkpoint")
    if not checkpoint:
        raise ValueError("base_checkpoint is required; train or provide a domain-1 model")
    model = make_sepformer(task, *model_config["ranks"], model_config)
    initialize_from_checkpoint(model, checkpoint)
    for module in model.modules():
        if hasattr(module, "scaling") and hasattr(module, "sara_A_spec"):
            module.scaling = float(model_config.get("adapter_scaling", 0.25))
    trainable = set_trainable(model)
    base_fingerprint = freeze_backbone(model)
    model.to(device)

    manifest_paths = [path for domain in domains
                      for path in (manifests[domain]["train"], manifests[domain]["validation"])]
    run_fingerprint = training_fingerprint(config, manifest_paths)

    from speechbrain.augment.time_domain import SpeedPerturb
    speed = [95, 100, 105] if training.get("speed_perturb", True) else [100]
    perturb = SpeedPerturb(orig_freq=int(config.get("sample_rate", 8000)), speeds=speed).to(device)
    output = Path(config["output_dir"]).expanduser() / task / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    class_map = []
    stage_states: dict[int, dict] = {}
    resume_path = output / "last.pt"

    for stage, domain in enumerate(domains, 1):
        if domain not in manifests:
            raise KeyError(f"No manifests configured for domain {domain}")
        train_rows = read_rows(manifests[domain]["train"])
        valid_rows = read_rows(manifests[domain]["validation"])
        class_map.append(domain)
        stage_dir = output / f"stage{stage}_{domain}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        best_path = stage_dir / "best.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location="cpu", weights_only=False)
            if (state["base_fingerprint"] != base_fingerprint or state["class_map"] != class_map
                    or state.get("domain_order") != domain_order
                    or state.get("run_fingerprint") != run_fingerprint):
                raise ValueError(f"Existing stage checkpoint does not match this run: {best_path}")
            stage_states[stage] = state
            load_adapters(model, state["adapters"])
            continue

        resume = None
        if resume_path.exists():
            candidate = torch.load(resume_path, map_location="cpu", weights_only=False)
            if candidate["stage"] == stage:
                resume = candidate
                if (candidate["base_fingerprint"] != base_fingerprint
                        or candidate["class_map"] != class_map
                        or candidate.get("domain_order") != domain_order
                        or candidate.get("run_fingerprint") != run_fingerprint):
                    raise ValueError("Resume state belongs to a different base or domain map")
                load_adapters(model, candidate["adapters"])
        if resume is None:
            if stage > 1:
                previous = stage_states[stage - 1]["adapters"]
                current = adapter_state(model)
                for key in current:
                    if key.endswith("_shared"):
                        current[key] = previous[key].clone()
                load_adapters(model, current)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "sara_" in name and "_spec" in name:
                        if "sara_A_" in name:
                            parameter.zero_()
                        elif "sara_B_" in name:
                            torch.nn.init.kaiming_uniform_(parameter, a=np.sqrt(5))
            inherited = {name: parameter.detach().cpu().clone()
                         for name, parameter in model.named_parameters()
                         if "sara_" in name and name.endswith("_shared")}
            epoch_start = 0
            best_score = -float("inf")
            best_epoch = None
            optimizer = torch.optim.Adam(trainable, lr=float(training["lr"]), weight_decay=0.0)
            scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
        else:
            inherited = resume["inherited_shared"]
            epoch_start = int(resume["epoch"])
            best_score = float(resume["best_score"])
            best_epoch = resume["best_epoch"]
            optimizer = torch.optim.Adam(trainable, lr=float(training["lr"]), weight_decay=0.0)
            optimizer.load_state_dict(resume["optimizer"])
            scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
            scaler.load_state_dict(resume["scaler"])
            torch.set_rng_state(resume["rng_torch"])
            random.setstate(resume["rng_python"])
            np.random.set_state(resume["rng_numpy"])

        inherited = {name: value.to(device) for name, value in inherited.items()}
        atomic_json_save({"task": task, "stage": stage, "domain": domain, "seed": seed,
                          "class_map": class_map, "base_fingerprint": base_fingerprint,
                          "train_rows": len(train_rows), "validation_rows": len(valid_rows),
                          "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                          "optimizer": "Adam", "lr": float(training["lr"]),
                          "batch_size": int(training["batch_size"]),
                          "epochs": int(training["epochs"]), "loss": "negative PIT-SI-SNR",
                          "shared_drift_weight": float(training["shared_drift_weight"]),
                          "shared_drift_xi": float(training["shared_drift_xi"])},
                         stage_dir / "run.json")

        for epoch in range(epoch_start, int(training["epochs"])):
            loss = fit_epoch(model, train_rows, task, optimizer, device,
                             int(training["batch_size"]), perturb if speed != [100] else None,
                             scaler, inherited if stage > 1 else None,
                             float(training["shared_drift_weight"]),
                             float(training["shared_drift_xi"]),
                             float(training["gradient_clip"]), seed + stage * 100003 + epoch)
            score = validate(model, valid_rows, task, device, training["selection"])
            record = {"stage": stage, "domain": domain, "epoch": epoch + 1,
                      "training_negative_si_snr": loss, "validation_score": score,
                      "selection": training["selection"]}
            print(json.dumps(record), flush=True)
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                state = {"format_version": 1, "task": task, "stage": stage,
                         "domain": domain, "class_map": list(class_map),
                         "domain_order": domain_order, "run_fingerprint": run_fingerprint,
                         "adapters": adapter_state(model),
                         "base_fingerprint": base_fingerprint, "validation_score": score,
                         "epoch": epoch + 1, "config": config}
                atomic_torch_save(state, best_path)
            atomic_torch_save({"stage": stage, "domain": domain, "epoch": epoch + 1,
                               "best_score": best_score, "best_epoch": best_epoch,
                               "adapters": adapter_state(model),
                               "inherited_shared": inherited,
                               "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                               "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
                               "rng_python": random.getstate(), "base_fingerprint": base_fingerprint,
                               "class_map": list(class_map), "domain_order": domain_order,
                               "run_fingerprint": run_fingerprint}, resume_path)
            atomic_json_save(record, stage_dir / f"epoch{epoch + 1:04d}.json")

        if best_epoch is None:
            raise RuntimeError(f"No validation checkpoint selected for stage {stage}")
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        load_adapters(model, state["adapters"])
        stage_states[stage] = state
        atomic_json_save({"status": "training_and_validation_complete",
                          "stage": stage, "domain": domain, "best_epoch": best_epoch,
                          "best_score": best_score, "checkpoint": str(best_path),
                          "base_fingerprint": base_fingerprint,
                          "frozen_parameters_unchanged": freeze_backbone(model) == base_fingerprint},
                         stage_dir / "complete.json")

    atomic_json_save({"task": task, "domain_order": domain_order, "class_map": domains,
                      "stages": [str(output / f"stage{i}_{domain}" / "best.pt")
                                 for i, domain in enumerate(domains, 1)],
                      "base_checkpoint": str(Path(checkpoint).expanduser()),
                      "base_fingerprint": base_fingerprint}, output / "trajectory.json")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    import yaml
    print(train_incremental(yaml.safe_load(args.config.read_text())))


if __name__ == "__main__":
    main()

"""Train the domain-1 SepFormer source model used for initialization."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from iss.models.network import make_sepformer
from iss.training.common import fit_epoch, read_rows, validate
from iss.training.state import atomic_json_save, atomic_torch_save, training_fingerprint


def train_source(config: dict, config_path: Path) -> Path:
    task = config["task"]
    source = config["source"]
    domain_order = list(config["domain_order"])
    if not domain_order or domain_order[0] != source["domain"]:
        raise ValueError("domain_order must begin with the configured source domain")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = make_sepformer(task, 0, 0, config.get("model")).to(device)
    train_rows = read_rows(source["train_manifest"])
    valid_rows = read_rows(source["validation_manifest"])
    run_fingerprint = training_fingerprint(
        config, [source["train_manifest"], source["validation_manifest"]])
    training = config["training"]
    from speechbrain.augment.time_domain import SpeedPerturb
    perturb = SpeedPerturb(orig_freq=8000, speeds=[95, 100, 105]).to(device) \
        if training.get("speed_perturb", True) else None
    epochs = int(training.get("source_epochs", training["epochs"]))
    batch_size = int(training.get("source_batch_size", training["batch_size"]))
    source_lr = float(training.get("source_lr", training["lr"]))
    selection = training.get("source_selection", training.get("selection", "si_snr"))
    optimizer = torch.optim.Adam(model.parameters(), lr=source_lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2)
    output = Path(config.get("source_output_dir", config["output_dir"])).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    history = []
    resume_path = output / "resume.pt"
    epoch_start = 0
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Source resume state belongs to different config or manifests")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        torch.set_rng_state(resume["rng_torch"])
        random.setstate(resume["rng_python"])
        np.random.set_state(resume["rng_numpy"])
        epoch_start = int(resume["epoch"])
        best_score = float(resume["best_score"])
        history = resume["history"]
    for epoch in range(epoch_start, epochs):
        train_loss = fit_epoch(model, train_rows, task, optimizer, device,
                               batch_size, perturb, scaler,
                               gradient_clip=float(training["gradient_clip"]),
                               seed=seed + epoch)
        score = validate(model, valid_rows, task, device, selection)
        scheduler.step(-score)
        history.append({"epoch": epoch + 1, "training_negative_si_snr": train_loss,
                        "validation_score": score, "selection": selection,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if score > best_score:
            best_score = score
            atomic_torch_save({"format_version": 1, "task": task, "stage": 0,
                               "domain": source["domain"], "domain_order": domain_order,
                               "run_fingerprint": run_fingerprint,
                               "state_dict": model.state_dict(), "epoch": epoch + 1,
                               "validation_score": score, "config": config},
                              output / "base_model.pt")
        atomic_torch_save({"epoch": epoch + 1, "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                           "scaler": scaler.state_dict(), "rng_torch": torch.get_rng_state(),
                           "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
                           "best_score": best_score, "history": history,
                           "run_fingerprint": run_fingerprint}, output / "resume.pt")
        atomic_json_save({"task": task, "domain": source["domain"], "epoch": epoch + 1,
                          "best_validation_score": best_score, "history": history,
                          "checkpoint": str(output / "base_model.pt")}, output / "training.json")
        print(json.dumps(history[-1]), flush=True)
    return output / "base_model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text())
    print(train_source(config, args.config))


if __name__ == "__main__":
    main()

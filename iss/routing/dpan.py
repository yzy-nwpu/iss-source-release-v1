"""Domain Prediction and Adapter Navigation (DPAN).

The selector is a linear classifier over 256-dimensional, temporally pooled
SepFormer encoder features. At each domain arrival, old real feature rows are
replaced by samples from saved multivariate-Laplace memory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


def estimate_memory(features: np.ndarray, diagonal_loading: float | None = None) -> dict:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("DPAN memory needs at least two finite feature rows")
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = (covariance + covariance.T) * 0.5
    if diagonal_loading is None:
        loading = 0.0
        for trial in range(9):
            try:
                chol = np.linalg.cholesky(covariance + loading * np.eye(values.shape[1]))
                break
            except np.linalg.LinAlgError:
                loading = max(np.trace(covariance) / values.shape[1] * 1e-10, 1e-16) * 10**trial
        else:
            raise np.linalg.LinAlgError("DPAN feature covariance is not positive definite")
    else:
        loading = float(diagonal_loading)
        chol = np.linalg.cholesky(covariance + loading * np.eye(values.shape[1]))
    return {"mean": mean, "cov": covariance, "chol": chol,
            "n": np.asarray(values.shape[0], dtype=np.int64),
            "loading": np.asarray(loading, dtype=np.float64)}


def sample_multivariate_laplace(memory: dict, n_samples: int, seed: int) -> np.ndarray:
    """Sample mu + sqrt(E) L z; E~Exp(scale=1), z~N(0,I), L L^T=Sigma."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    rng = np.random.RandomState(seed)
    dimension = int(np.asarray(memory["mean"]).shape[0])
    exponential = rng.exponential(scale=1.0, size=n_samples)
    gaussian = rng.randn(n_samples, dimension) @ np.asarray(memory["chol"]).T
    samples = np.asarray(memory["mean"]) + np.sqrt(exponential)[:, None] * gaussian
    return samples.astype(np.float32)


def train_selector(current_features: np.ndarray, memories: list[dict], previous: nn.Linear | None,
                   stage: int, config: dict, seed: int, device: str = "cpu") -> nn.Linear:
    """Train one cumulative selector using current rows and one replay pool/domain."""
    if stage != len(memories) or stage < 1:
        raise ValueError("stage must match the number of saved memories")
    current = np.asarray(current_features, dtype=np.float32)
    dimension = int(current.shape[1])
    torch.manual_seed(seed + stage * 1000)
    selector = nn.Linear(dimension, stage).to(device)
    if previous is not None:
        if previous.out_features != stage - 1 or previous.in_features != dimension:
            raise ValueError("Previous DPAN classifier has incompatible shape")
        with torch.no_grad():
            selector.weight[:stage - 1].copy_(previous.weight)
            selector.bias[:stage - 1].copy_(previous.bias)

    replay_arrays = [sample_multivariate_laplace(memory, int(memory["n"]), seed + 10000 * stage + index)
                     for index, memory in enumerate(memories[:-1])]
    replay_x = (torch.as_tensor(np.concatenate(replay_arrays), device=device)
                if replay_arrays else None)
    replay_y = (torch.as_tensor(np.concatenate([np.full(len(x), i, dtype=np.int64)
                                                 for i, x in enumerate(replay_arrays)]), device=device)
                if replay_arrays else None)
    current_x = torch.as_tensor(current, device=device)
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    optimizer = torch.optim.SGD(selector.parameters(), lr=float(config["lr"]),
                                momentum=float(config["momentum"]), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    updates_per_epoch = math.ceil(len(current_x) / batch_size)
    for _epoch in range(epochs):
        order = torch.randperm(len(current_x), device=device)
        selector.train()
        for start in range(0, len(order), batch_size):
            current_indices = order[start:start + batch_size]
            if replay_x is None:
                x = current_x[current_indices]
                y = torch.full((len(x),), stage - 1, dtype=torch.long, device=device)
            else:
                n_new = batch_size // stage
                new_indices = current_indices[torch.randperm(len(current_indices), device=device)[:n_new]]
                n_new = len(new_indices)
                n_old = batch_size - n_new
                old_indices = torch.randperm(len(replay_x), device=device)[:n_old]
                x = torch.cat((current_x[new_indices], replay_x[old_indices]), dim=0)
                y = torch.cat((torch.full((n_new,), stage - 1, dtype=torch.long, device=device),
                               replay_y[old_indices]))
            loss = torch.nn.functional.cross_entropy(selector(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
    selector.eval()
    selector.training_updates = updates_per_epoch * epochs
    return selector


def save_memory(path: str | Path, memory: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **memory)


def load_memory(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_selector(path: str | Path, selector: nn.Linear, *, task: str, stage: int,
                  class_map: list[str], config: dict, seed: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": selector.state_dict(), "task": task, "stage": stage,
                "class_map": list(class_map), "config": dict(config), "seed": seed}, path)


def load_selector(path: str | Path, device: str = "cpu") -> tuple[nn.Linear, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    selector = nn.Linear(payload["state_dict"]["weight"].shape[1],
                         payload["state_dict"]["weight"].shape[0]).to(device)
    selector.load_state_dict(payload["state_dict"], strict=True)
    selector.eval().requires_grad_(False)
    return selector, payload


def fit_from_features(feature_manifests: list[tuple[str, str]], output_dir: str | Path,
                      config: dict, task: str, domains: list[str], seed: int = 2026,
                      device: str = "cpu") -> None:
    """Fit DPAN stagewise. Each pair is (domain, current-domain train features .npy)."""
    if len(feature_manifests) != len(domains):
        raise ValueError("Provide one current-domain feature array per domain")
    root = Path(output_dir) / task
    root.mkdir(parents=True, exist_ok=True)
    memories, previous = [], None
    for stage, (domain, path) in enumerate(feature_manifests, 1):
        if domain != domains[stage - 1]:
            raise ValueError(f"Expected domain {domains[stage - 1]}, got {domain}")
        current = np.load(path, allow_pickle=False).astype(np.float32)
        memory = estimate_memory(current)
        stage_dir = root / f"stage{stage}_{domain}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        save_memory(stage_dir / "compressed_memory.npz", memory)
        memories.append(memory)
        previous = train_selector(current, memories, previous, stage, config, seed, device)
        save_selector(stage_dir / f"selector_seed{seed}.pt", previous, task=task,
                      stage=stage, class_map=domains[:stage], config=config, seed=seed)
        metadata = {"task": task, "stage": stage, "domain": domain,
                    "class_map": domains[:stage], "old_real_features_read": False,
                    "source_feature_file": Path(path).name,
                    "feature_count": int(len(current)), "feature_dim": int(current.shape[1]),
                    "laplace_diagonal_loading": float(memory["loading"]),
                    "replay_pool_per_old_domain": [int(item["n"]) for item in memories[:-1]],
                    "optimizer": "SGD", "learning_rate": float(config["lr"]),
                    "momentum": float(config["momentum"]), "weight_decay": 0.0,
                    "epochs": int(config["epochs"]), "batch_size": int(config["batch_size"]),
                    "loss": "cross_entropy", "training_updates": int(previous.training_updates)}
        (stage_dir / "fit.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit stagewise DPAN from pooled encoder features")
    parser.add_argument("--task", choices=("separation", "denoising"), required=True)
    parser.add_argument("--features", nargs="+", required=True,
                        help="DOMAIN=/path/to/current_domain_train_features.npy, in arrival order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=10.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    pairs = [item.split("=", 1) for item in args.features]
    cfg = {"epochs": args.epochs, "batch_size": args.batch_size,
           "lr": args.lr, "momentum": args.momentum, "distribution": "laplace"}
    fit_from_features(pairs, args.output_dir, cfg, args.task,
                      [domain for domain, _path in pairs], args.seed, args.device)


if __name__ == "__main__":
    main()

"""Extract current-domain features and fit all cumulative DPAN stages."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from iss.routing.dpan import fit_from_features
from iss.routing.features import extract_pooled_features


def fit_config(config: dict) -> Path:
    task = config["task"]
    domains = list(config["domains"])
    features = []
    root = Path(config.get("dpan_features_dir", Path(config["output_dir"]).expanduser() /
                          task / "dpan_features")).expanduser()
    for domain in domains:
        manifest = config["manifests"][domain]["train"]
        output = root / f"{domain}_train.npy"
        extract_pooled_features(manifest, task, config["base_checkpoint"], output,
                                config.get("device", "cpu"))
        features.append((domain, str(output)))
    fit_from_features(features, Path(config["dpan_dir"]).expanduser(),
                      config["dpan"], task, domains,
                      int(config["dpan"].get("seed", 2026)), config.get("device", "cpu"))
    return Path(config["dpan_dir"]).expanduser() / task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(fit_config(yaml.safe_load(args.config.read_text())))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check task YAML contracts and public model entry paths."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED = {
    "separation": ["English", "Yoruba", "Tamil", "Catalan", "Mandarin"],
    "denoising": ["base", "alarm", "cough", "dd", "mg"],
}


def main() -> None:
    for task in ("separation", "denoising"):
        path = ROOT / "configs" / f"{task}.yaml"
        config = yaml.safe_load(path.read_text())
        if config["task"] != task or config["domain_order"] != EXPECTED[task]:
            raise SystemExit(f"Invalid {task} task/domain order in {path}")
        if config["domain_order"][1:] != list(config["domains"]):
            raise SystemExit(f"Incremental domains do not match domain_order in {path}")
        if config["source"]["domain"] != config["domain_order"][0]:
            raise SystemExit(f"Source domain does not match domain_order in {path}")
        if not config.get("base_checkpoint"):
            raise SystemExit(f"Missing base_checkpoint in {path}")
    for module, attribute in (("iss.models.network", "make_sepformer"),
                              ("iss.models.iss_model", "forward"),
                              ("iss.routing.dpan", "fit_from_features")):
        if not callable(getattr(importlib.import_module(module), attribute)):
            raise SystemExit(f"Could not resolve {module}.{attribute}")
    print("Both task configs and public model/DPAN object paths resolve.")


if __name__ == "__main__":
    main()

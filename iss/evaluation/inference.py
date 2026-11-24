"""DPAN-routed single-file inference for separation or denoising."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from iss.data.audio import read_audio
from iss.evaluation.protocol import activate, load_trajectory
from iss.models.iss_model import forward
from iss.routing.dpan import load_selector


@torch.inference_mode()
def infer(config: dict, input_path: str | Path, output_path: str | Path) -> dict:
    task, domains = config["task"], list(config["domains"])
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model, template, states = load_trajectory(config, device)
    final_stage = len(domains)
    dpan = Path(config["dpan_dir"]).expanduser() / task / \
        f"stage{final_stage}_{domains[-1]}" / \
        f"selector_seed{config.get('dpan_seed', 2026)}.pt"
    selector, selector_state = load_selector(dpan, device)
    if (selector_state["task"] != task or selector_state["stage"] != final_stage
            or selector_state["class_map"] != domains
            or selector_state["config"] != config["dpan"]
            or selector_state["seed"] != config.get("dpan_seed", 2026)):
        raise ValueError("DPAN task, stage, config, seed, or class map does not match")
    mixture = read_audio(str(input_path), int(config.get("sample_rate", 8000)))
    mix = mixture[None].to(device)
    feature = model["encoder"](mix).mean(dim=-1)
    predicted_index = int(selector(feature.float()).argmax(-1).item())
    expert_stage = predicted_index + 1
    activate(model, template, states, final_stage, expert_stage)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        estimate = forward(model, mix)[0].float().cpu().numpy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, estimate, int(config.get("sample_rate", 8000)))
    return {"input": str(Path(input_path).resolve()), "output": str(output_path.resolve()),
            "task": task, "predicted_domain": domains[predicted_index],
            "predicted_domain_index": predicted_index, "activated_dsa_stage": expert_stage,
            "output_samples": int(estimate.shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = infer(yaml.safe_load(args.config.read_text()), args.input, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

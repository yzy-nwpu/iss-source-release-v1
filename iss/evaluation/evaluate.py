"""Evaluate DPAN-routed and oracle-routed experts on the configured test domains."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from iss.evaluation.metrics import measure
from iss.evaluation.protocol import activate, load_trajectory
from iss.models.iss_model import forward
from iss.routing.dpan import load_selector
from iss.training.common import read_rows


@torch.inference_mode()
def evaluate(config: dict) -> Path:
    task, domains = config["task"], list(config["domains"])
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model, template, states = load_trajectory(config, device)
    final_stage = len(domains)
    dpan_root = Path(config["dpan_dir"]).expanduser() / task
    selector_path = dpan_root / f"stage{final_stage}_{domains[-1]}" / \
        f"selector_seed{config.get('dpan_seed', 2026)}.pt"
    selector, payload = load_selector(selector_path, device)
    if (payload["task"] != task or payload["stage"] != final_stage
            or payload["class_map"] != domains
            or payload["config"] != config["dpan"]
            or payload["seed"] != config.get("dpan_seed", 2026)):
        raise ValueError("DPAN class map does not match the configured arrival order")
    output = Path(config["output_dir"]).expanduser() / task / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for true_stage, domain in enumerate(domains, 1):
        rows = read_rows(config["manifests"][domain]["test"])
        for row in rows:
            mixture, rate = sf.read(row["mix_wav"], dtype="float32")
            if rate != int(config.get("sample_rate", 8000)) or mixture.ndim != 1:
                raise ValueError(f"Invalid test mixture {row['mix_wav']}")
            target_keys = ("s1_wav", "s2_wav") if task == "separation" else ("s1_wav",)
            targets = [sf.read(row["targets"][key], dtype="float32")[0] for key in target_keys]
            length = min(len(mixture), *(len(x) for x in targets))
            reference = np.stack([x[:length] for x in targets], axis=-1)
            mix = torch.from_numpy(mixture[:length].copy())[None].to(device)
            feature = model["encoder"](mix).mean(dim=-1)
            predicted_index = int(selector(feature.float()).argmax(-1).item())
            predicted_expert = predicted_index + 1
            record = {"utterance_id": row["utterance_id"], "true_domain": domain,
                      "true_expert_stage": true_stage,
                      "predicted_domain": domains[predicted_index],
                      "predicted_expert_stage": predicted_expert,
                      "routing_correct": predicted_expert == true_stage}
            for router, expert in (("dpan", predicted_expert), ("oracle", true_stage)):
                activate(model, template, states, final_stage, expert)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    prediction = forward(model, mix)[0, :length].float().cpu().numpy()
                try:
                    scores = measure(reference, prediction, int(config.get("sample_rate", 8000)))
                    record[router] = scores
                except Exception as error:
                    record[router] = {"error": f"{type(error).__name__}: {error}"}
            rows_out.append(record)
    table_path = output / "utterance_metrics.jsonl"
    table_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows_out))
    summary = {"task": task, "domains": domains, "samples": len(rows_out),
               "dpan_routing_accuracy": float(np.mean([row["routing_correct"] for row in rows_out])),
               "routers": {}}
    for router in ("dpan", "oracle"):
        summary["routers"][router] = {}
        for metric in ("si_snr", "sdr", "pesq"):
            values = [row[router].get(metric) for row in rows_out
                      if isinstance(row.get(router), dict) and row[router].get(metric) is not None]
            summary["routers"][router][metric] = {
                "valid": len(values), "expected": len(rows_out),
                "mean": float(np.mean(values)) if len(values) == len(rows_out) else None,
                "available_mean": float(np.mean(values)) if values else None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate(yaml.safe_load(args.config.read_text())))


if __name__ == "__main__":
    main()

"""Extract the exact frozen encoder mean-pooling features consumed by DPAN."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from iss.models.network import make_sepformer
from iss.data.audio import read_audio
from iss.training.checkpoint import initialize_from_checkpoint
from iss.training.common import read_rows


@torch.inference_mode()
def extract_pooled_features(manifest: str | Path, task: str, base_checkpoint: str | Path,
                            output: str | Path, device: str = "cpu") -> np.ndarray:
    model = make_sepformer(task, 0, 0)
    initialize_from_checkpoint(model, base_checkpoint)
    encoder = model["encoder"].to(device).eval()
    rows = read_rows(manifest)
    result = np.empty((len(rows), encoder.conv1d.out_channels), dtype=np.float32)
    for index, row in enumerate(rows):
        mixture = read_audio(row["mix_wav"], int(row.get("sample_rate", 8000)))
        embedding = encoder(mixture[None].to(device)).mean(dim=-1)[0]
        result[index] = embedding.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Encoder features contain NaN or Inf")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, result)
    return result

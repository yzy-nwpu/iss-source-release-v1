"""Variable-length waveform loading, padding, and task-aware mixing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def read_audio(path: str, sample_rate: int = 8000) -> torch.Tensor:
    waveform, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != sample_rate:
        raise ValueError(f"{path}: expected {sample_rate} Hz, got {rate}")
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError(f"{path}: expected finite mono audio")
    return torch.from_numpy(waveform.copy())


class ManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, sample_rate: int = 8000):
        self.path = Path(manifest)
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        self.sample_rate = sample_rate
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")
        ids = [row["utterance_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate utterance_id in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        mixture = read_audio(row["mix_wav"], self.sample_rate)
        target_paths = row["targets"]
        targets = [read_audio(target_paths[key], self.sample_rate)
                   for key in ("s1_wav", "s2_wav") if key in target_paths]
        length = min([mixture.numel(), *(target.numel() for target in targets)])
        mixture = mixture[:length]
        targets = [target[:length] for target in targets]
        return row, mixture, torch.stack(targets, dim=-1)


def collate_waveforms(items):
    rows, mixtures, targets = zip(*items)
    lengths = torch.tensor([x.numel() for x in mixtures], dtype=torch.long)
    longest = int(lengths.max())
    mixture = torch.stack([torch.nn.functional.pad(x, (0, longest - x.numel())) for x in mixtures])
    target = torch.stack([torch.nn.functional.pad(x, (0, 0, 0, longest - x.shape[0])) for x in targets])
    return rows, mixture, target, lengths

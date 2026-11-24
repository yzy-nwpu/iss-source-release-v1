"""Shared task losses, audio loading, augmentation, and validation."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from iss.models.iss_model import forward


def read_rows(manifest: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {manifest}")
    return rows


def audio(path: str, sample_rate: int = 8000) -> torch.Tensor:
    wav, rate = sf.read(path, dtype="float32")
    if rate != sample_rate or wav.ndim != 1 or not np.isfinite(wav).all():
        raise ValueError(f"Invalid mono {sample_rate} Hz audio: {path}")
    return torch.from_numpy(wav.copy())


def load_example(row: dict, task: str, augment: bool, speed_perturb=None,
                 sample_rate: int = 8000, device: str = "cpu"):
    target_keys = ("s1_wav", "s2_wav") if task == "separation" else ("s1_wav",)
    targets = [audio(row["targets"][key], sample_rate) for key in target_keys]
    if augment and speed_perturb is not None:
        transformed = []
        for target in targets:
            transformed.append(speed_perturb(target[None].to(device))[0].cpu())
        targets = transformed
    length = min(target.numel() for target in targets)
    if task == "denoising" and row["targets"].get("noise_wav"):
        noise = audio(row["targets"]["noise_wav"], sample_rate)
        length = min(length, noise.numel())
        mixture = targets[0][:length] + noise[:length]
        targets = [targets[0][:length]]
    elif augment:
        targets = [target[:length] for target in targets]
        mixture = torch.stack(targets).sum(dim=0)
    else:
        mixture = audio(row["mix_wav"], sample_rate)[:length]
        targets = [target[:length] for target in targets]
    return mixture[None].to(device), torch.stack(targets, dim=-1)[None].to(device)


def pit_si_snr_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    from speechbrain.nnet.losses import get_si_snr_with_pitwrapper
    # SpeechBrain's recipe loss uses [batch,time,speakers] and is permutation invariant.
    return get_si_snr_with_pitwrapper(target.float(), prediction.float()).mean()


def fit_epoch(model, rows, task: str, optimizer, device: str, batch_size: int,
              speed_perturb=None, grad_scaler=None, shared_reference=None,
              drift_weight: float = 1e-3, drift_xi: float = 5e-4,
              gradient_clip: float = 5.0, seed: int = 0):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    model.train()
    losses = []
    for start in range(0, len(order), batch_size):
        selected = [rows[i] for i in order[start:start + batch_size]]
        items = [load_example(row, task, True, speed_perturb, device=device) for row in selected]
        lengths = [mix.shape[1] for mix, _target in items]
        longest = max(lengths)
        mixtures = torch.cat([torch.nn.functional.pad(mix, (0, longest - mix.shape[1]))
                              for mix, _target in items])
        targets = torch.cat([torch.nn.functional.pad(target, (0, 0, 0, longest - target.shape[1]))
                             for _mix, target in items])
        optimizer.zero_grad(set_to_none=True)
        use_amp = device.startswith("cuda") and grad_scaler is not None
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            prediction = forward(model, mixtures)
            sample_losses = torch.stack([pit_si_snr_loss(prediction[i:i+1, :length],
                                                         targets[i:i+1, :length])
                                         for i, length in enumerate(lengths)])
            task_loss = sample_losses.mean()
            regularizer = task_loss.new_zeros(())
            if shared_reference:
                current = dict(model.named_parameters())
                numerator = task_loss.new_zeros((), dtype=torch.float64)
                denominator = task_loss.new_zeros((), dtype=torch.float64)
                for name, old in shared_reference.items():
                    value = current[name]
                    numerator += (value.double() - old.to(device).double()).square().sum()
                    denominator += old.to(device).double().square().sum()
                regularizer = (numerator / (denominator + drift_xi)).to(task_loss.dtype)
            loss = task_loss + drift_weight * regularizer
        if use_amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], gradient_clip)
        if use_amp:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()
        losses.extend(float(value) for value in sample_losses.detach().cpu())
    return float(np.mean(losses))


@torch.inference_mode()
def validate(model, rows, task: str, device: str, selection: str = "si_snr") -> float:
    model.eval()
    scores = []
    pesq_scores = []
    for row in rows:
        mixture, target = load_example(row, task, False, device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            prediction = forward(model, mixture)
        if selection == "pesq" and task == "denoising":
            from pesq import pesq
            pesq_scores.append(float(pesq(8000, target[0, :, 0].cpu().numpy(),
                                          prediction[0, :, 0].float().cpu().numpy(), "nb")))
        else:
            scores.append(-float(pit_si_snr_loss(prediction, target)))
    values = pesq_scores if pesq_scores else scores
    if not values or not np.isfinite(values).all():
        raise RuntimeError("Validation returned no finite scores")
    return float(np.mean(values))

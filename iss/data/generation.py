"""Materialize fixed two-speaker and speech-plus-noise recipe rows."""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample


SAMPLE_RATE = 8000


def _read_mono(path: str | Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float64", always_2d=False)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite mono waveform: {path}")
    if len(audio) == 0:
        raise ValueError(f"Empty waveform: {path}")
    return audio, int(rate)


def _active_level(audio: np.ndarray) -> np.ndarray:
    window = min(400, len(audio))
    envelope = np.convolve(np.abs(audio), np.ones(window, dtype=np.float64) / window,
                           mode="same")
    active = envelope > 0.1 * float(envelope.max())
    if not active.any():
        raise ValueError("Cannot normalize a silent source")
    level = float(np.sqrt(np.mean(np.square(audio[active]))))
    return audio / (level + 1e-6)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:100] or "utterance"


def _rows(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    ids = [row.get("ID", str(index)) for index, row in enumerate(rows)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"IDs must be unique in {path}")
    return rows


def generate_separation(pairs_csv: str | Path, output_root: str | Path, split: str,
                        length_policy: str = "min", overwrite: bool = False) -> int:
    """Create 8-kHz min/max two-speaker mixtures from fixed pair rows.

    CSV columns: ID,s1_wav,s2_wav,s1_gain_db,s2_gain_db. Split membership and
    speaker pairing are deliberately supplied by the input file.
    """
    if split not in {"tr", "cv", "tt"} or length_policy not in {"min", "max"}:
        raise ValueError("split must be tr/cv/tt and length_policy must be min/max")
    rows = _rows(pairs_csv, {"ID", "s1_wav", "s2_wav", "s1_gain_db", "s2_gain_db"})
    root = Path(output_root) / "wav8k" / length_policy / split
    destinations = {name: root / name for name in ("mix", "s1", "s2")}
    for directory in destinations.values():
        directory.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        rate1_audio, rate1 = _read_mono(row["s1_wav"])
        rate2_audio, rate2 = _read_mono(row["s2_wav"])
        if rate1 != rate2:
            raise ValueError(f"Pair {row['ID']} has different source sample rates")
        target_length = int(len(rate1_audio) * SAMPLE_RATE / rate1)
        first = resample(rate1_audio, target_length)
        second = resample(rate2_audio, int(len(rate2_audio) * SAMPLE_RATE / rate2))
        first = _active_level(first) * 10.0 ** (float(row["s1_gain_db"]) / 20.0)
        second = _active_level(second) * 10.0 ** (float(row["s2_gain_db"]) / 20.0)
        length = min(len(first), len(second)) if length_policy == "min" else max(len(first), len(second))
        first = np.pad(first[:length], (0, max(length - len(first), 0)))
        second = np.pad(second[:length], (0, max(length - len(second), 0)))
        mixture = first + second
        peak = float(np.max(np.abs(np.concatenate((mixture, first, second)))))
        if peak <= 0 or not math.isfinite(peak):
            raise ValueError(f"Pair {row['ID']} produced an invalid mixture")
        scale = 0.9 / peak
        first *= scale
        second *= scale
        mixture *= scale
        name = f"{index:07d}_{_slug(row['ID'])}.wav"
        paths = {key: directory / name for key, directory in destinations.items()}
        if not overwrite and any(path.exists() for path in paths.values()):
            raise FileExistsError(f"Output exists for pair {row['ID']}: {paths['mix']}")
        for key, audio in (("s1", first), ("s2", second), ("mix", mixture)):
            sf.write(paths[key], audio, SAMPLE_RATE)
    return len(rows)


def _add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(clean).all() or not np.isfinite(noise).all() or len(clean) == 0 or len(noise) == 0:
        raise ValueError("Clean and noise signals must be non-empty and finite")
    if len(noise) <= len(clean):
        repeats = max(1, math.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    if len(noise) == len(clean):
        start = 0
    else:
        start = int(rng.randint(0, len(noise) - len(clean)))
    segment = noise[start:start + len(clean)]
    clean_energy = float(np.dot(clean, clean))
    noise_energy = float(np.dot(segment, segment))
    if clean_energy <= 0 or noise_energy <= 0:
        raise ValueError("Clean and noise signals must have non-zero energy")
    ratio = 10.0 ** (float(snr_db) / 10.0)
    scaled_noise = segment * np.sqrt(clean_energy / (ratio * noise_energy))
    return clean + scaled_noise, scaled_noise


def generate_denoising(noisy_csv: str | Path, output_root: str | Path, split: str,
                       seed: int = 1234, overwrite: bool = False) -> int:
    """Create 8-kHz noisy mixtures from fixed speech/noise/SNR rows.

    CSV columns: ID,speech_wav,noise_wav,snr_db. Split membership, noise
    identity, and SNR are supplied by the input file; only noise crop offsets
    are sampled, with an explicit seed.
    """
    if split not in {"tr", "cv", "tt"}:
        raise ValueError("split must be tr/cv/tt")
    rows = _rows(noisy_csv, {"ID", "speech_wav", "noise_wav", "snr_db"})
    root = Path(output_root) / "wav8k" / "min" / split
    destinations = {name: root / name for name in ("mix_single", "s1", "noise")}
    for directory in destinations.values():
        directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    for index, row in enumerate(rows):
        clean, _ = librosa.load(row["speech_wav"], sr=SAMPLE_RATE, mono=True,
                                 dtype=np.float64)
        noise, _ = librosa.load(row["noise_wav"], sr=SAMPLE_RATE, mono=True,
                                dtype=np.float64)
        mixture, scaled_noise = _add_noise(clean, noise, float(row["snr_db"]), rng)
        name = f"{index:07d}_{_slug(row['ID'])}.wav"
        paths = {key: directory / name for key, directory in destinations.items()}
        if not overwrite and any(path.exists() for path in paths.values()):
            raise FileExistsError(f"Output exists for item {row['ID']}: {paths['mix_single']}")
        sf.write(paths["s1"], clean, SAMPLE_RATE)
        sf.write(paths["noise"], scaled_noise, SAMPLE_RATE)
        sf.write(paths["mix_single"], mixture, SAMPLE_RATE)
    return len(rows)

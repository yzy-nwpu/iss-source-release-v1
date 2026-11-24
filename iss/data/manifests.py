"""Create audit-friendly JSONL manifests from already prepared recipe CSVs.

Dataset identity and speaker splits come from the upstream corpus recipe. This
converter keeps CSV row order and never invents a random train/validation/test
split.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _path(value: str, root: Path) -> str:
    path = Path(value)
    return str((root / path).resolve() if not path.is_absolute() else path.resolve())


def convert_csv(csv_path: Path, *, task: str, domain: str, split: str,
                audio_root: Path, output_path: Path, sample_rate: int = 8000) -> int:
    if task not in {"separation", "denoising"}:
        raise ValueError(f"Unknown task {task!r}")
    if split not in {"tr", "cv", "tt"}:
        raise ValueError(f"Unknown split {split!r}; expected tr/cv/tt")
    n_speakers = 2 if task == "separation" else 1
    required = {"ID", "mix_wav", "s1_wav"}
    if n_speakers == 2:
        required.add("s2_wav")
    rows = []
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for index, raw in enumerate(reader):
            item_id = raw["ID"] or index
            target = {"s1_wav": _path(raw["s1_wav"], audio_root)}
            if n_speakers == 2:
                target["s2_wav"] = _path(raw["s2_wav"], audio_root)
            if task == "denoising" and raw.get("noise_wav"):
                target["noise_wav"] = _path(raw["noise_wav"], audio_root)
            rows.append({
                "utterance_id": f"{domain}:{split}:{item_id}",
                "task": task, "domain": domain, "split": split,
                "sample_rate": sample_rate,
                "mix_wav": _path(raw["mix_wav"], audio_root),
                "targets": target,
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("separation", "denoising"), required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split", choices=("tr", "cv", "tt"), required=True)
    parser.add_argument("--csv", type=Path, required=True, help="Prepared recipe CSV")
    parser.add_argument("--audio-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=8000)
    args = parser.parse_args()
    count = convert_csv(args.csv, task=args.task, domain=args.domain, split=args.split,
                        audio_root=args.audio_root, output_path=args.output,
                        sample_rate=args.sample_rate)
    print(json.dumps({"output": str(args.output), "rows": count}))


if __name__ == "__main__":
    main()

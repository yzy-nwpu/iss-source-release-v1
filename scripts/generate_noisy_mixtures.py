#!/usr/bin/env python3
"""Generate fixed speech/noise/SNR mixtures from an explicit CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iss.data.generation import generate_denoising


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-csv", type=Path, required=True)
    parser.add_argument("--split", choices=("tr", "cv", "tt"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = generate_denoising(args.items_csv, args.output_root, args.split,
                               args.seed, args.overwrite)
    print(f"Generated {count} noisy mixtures under {args.output_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate fixed-pair two-speaker mixtures from an explicit CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iss.data.generation import generate_separation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--split", choices=("tr", "cv", "tt"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--length-policy", choices=("min", "max"), default="min")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = generate_separation(args.pairs_csv, args.output_root, args.split,
                                args.length_policy, args.overwrite)
    print(f"Generated {count} mixtures under {args.output_root}")


if __name__ == "__main__":
    main()

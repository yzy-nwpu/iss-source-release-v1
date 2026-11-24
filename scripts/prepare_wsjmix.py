#!/usr/bin/env python3
"""Index an already prepared WSJ0Mix-style tree into recipe CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iss.data.prepare_wsjmix import prepare_wsjmix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Prepared root with wav8k/min/{tr,cv,tt}/{mix,s1,s2}")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_wsjmix(str(args.data_root), str(args.output_dir), n_spks=2, fs=8000)
    print(" ".join(str(args.output_dir / f"wsj_{split}.csv")
                    for split in ("tr", "cv", "tt")))


if __name__ == "__main__":
    main()

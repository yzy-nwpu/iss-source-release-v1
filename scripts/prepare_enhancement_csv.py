#!/usr/bin/env python3
"""Index an already prepared 8-kHz single-speech/noise tree into recipe CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iss.data.prepare_enhancement import prepare_wham_whamr_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Prepared root with wav8k/min/{tr,cv,tt}/{mix_single,s1,noise}")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_wham_whamr_csv(str(args.data_root), str(args.output_dir), fs=8000,
                           task="enhancement")
    print(" ".join(str(args.output_dir / f"wsj0_{split}.csv")
                    for split in ("tr", "cv", "tt")))


if __name__ == "__main__":
    main()

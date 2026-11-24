"""Load user-supplied source-domain initialization with strict key accounting."""
from __future__ import annotations

from pathlib import Path

import torch

from iss.models.iss_model import convert_base


def initialize_from_checkpoint(model, checkpoint: str | Path) -> None:
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Required source checkpoint does not exist: {path}")
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad_missing = [key for key in missing if "sara_" not in key]
        if bad_missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={bad_missing[:8]}, "
                               f"unexpected={unexpected[:8]}")
        return
    module_files = {name: path / f"{name}.ckpt" for name in ("encoder", "masknet", "decoder")}
    absent = [str(item) for item in module_files.values() if not item.is_file()]
    if absent:
        raise FileNotFoundError("Expected base_model.pt or encoder.ckpt, masknet.ckpt, "
                                f"decoder.ckpt; missing {absent}")
    for name, source_path in module_files.items():
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        converted, _mapping = convert_base(source, model[name].state_dict())
        model[name].load_state_dict(converted, strict=True)

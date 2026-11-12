"""Portable checkpoint and RNG helpers; load only checkpoints you trust."""
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

CHECKPOINT_FORMAT = "nctd-portable-v1"


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_new(path, payload, *, binary=False):
    """Create a complete new file without replacing an existing destination."""
    path = Path(path)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        if binary:
            with temporary.open("xb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        # The temporary and destination are on the same filesystem. Linking is
        # atomic and raises FileExistsError instead of overwriting old output.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    # Full optimizer/Python/NumPy RNG state requires pickle. A format tag is not
    # a security boundary: never load an untrusted downloaded .pt file.
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"format", "config", "step", "model_state", "ema_state",
                "optimizer_state", "teacher_state", "rng", "data_fingerprint"}
    if not isinstance(value, dict) or not required.issubset(value) or value["format"] != CHECKPOINT_FORMAT:
        raise ValueError("Expected a complete nctd-portable-v1 checkpoint; old harness checkpoints are not accepted")
    if isinstance(value["step"], bool) or not isinstance(value["step"], int) or value["step"] < 1:
        raise ValueError("Checkpoint must contain at least one completed training step")
    return value


def make_generators(seed):
    return {name: torch.Generator().manual_seed(seed+offset) for name, offset in
            (("batch", 2201), ("time", 2202), ("noise", 2204), ("trajectory", 2303))}


def seed_global(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def capture_rng(generators):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            "generators": {name: generator.get_state() for name, generator in generators.items()}}


def restore_rng(state, generators, device):
    if set(state["generators"]) != set(generators):
        raise ValueError("Checkpoint generator set mismatch")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.device(device).type == "cuda":
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; cannot restore all saved CUDA RNG states")
        torch.cuda.set_rng_state_all(state["cuda"])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name])


def resolve_device(name):
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This entry supports cpu or an explicitly selected cuda device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device

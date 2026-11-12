"""Local, checksum-verified CIFAR-10 loading; no download or test-set access."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

CIFAR_BATCH_MD5 = (
    "c99cafc152244af753f735de768cd75f", "d4bba439e000b95fd0a9bffe97cbabec",
    "54ebc095f3ab1f0389bbae665268c751", "634d18415352ddfa80567beed471001a",
    "482c414d41f54cd18b22e5b47cb7c3cb",
)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass
class TrainingData:
    pixels: torch.Tensor
    fingerprint: dict

    def __len__(self):
        return len(self.pixels)

    def batch(self, indices, device="cpu"):
        return self.pixels[indices].to(device=device, dtype=torch.float32)/127.5-1


def load_training_data(data_root=None, *, synthetic=False, seed=20, synthetic_images=64):
    """Return uint8 RGB training pixels and a path-independent data fingerprint.

    Synthetic pixels are independent uniform noise for software tests only.
    Official CIFAR Python batches are unpickled only after matching known MD5s.
    The fixed seed-12345 permutation reserves its final 5,000 images; this
    training-only entry does not load CIFAR's separate official test batch.
    """
    if synthetic:
        if isinstance(synthetic_images, bool) or not isinstance(synthetic_images, int) or synthetic_images < 1:
            raise ValueError("synthetic_images must be a positive integer")
        generator = torch.Generator().manual_seed(seed+10)
        pixels = torch.randint(0, 256, (synthetic_images, 3, 32, 32),
                               generator=generator, dtype=torch.uint8)
        info = {"kind": "independent_uniform_pixels_smoke", "synthetic": True,
                "not_formal_result": True, "seed": seed+10, "count": len(pixels),
                "pixels_sha256": tensor_digest(pixels)}
    else:
        if data_root is None:
            raise ValueError("Real training requires --data-root containing official CIFAR-10 Python batches")
        root = Path(data_root)
        folder = root if (root/"data_batch_1").is_file() else root/"cifar-10-batches-py"
        arrays, receipts = [], {}
        for index, expected in enumerate(CIFAR_BATCH_MD5, 1):
            name = f"data_batch_{index}"
            raw = (folder/name).read_bytes()
            actual = hashlib.md5(raw).hexdigest()
            if actual != expected:
                raise ValueError(f"Official CIFAR-10 checksum mismatch: {name}")
            payload = pickle.loads(raw, encoding="latin1")
            array = payload["data"]
            if not isinstance(array, np.ndarray) or array.shape != (10000, 3072) or array.dtype != np.uint8:
                raise ValueError(f"Unexpected official CIFAR-10 array: {name}")
            arrays.append(torch.from_numpy(array.copy()).reshape(-1, 3, 32, 32))
            receipts[name] = {"md5": actual, "sha256": hashlib.sha256(raw).hexdigest()}
        permutation = torch.randperm(50000, generator=torch.Generator().manual_seed(12345))
        selected = permutation[:45000]
        pixels = torch.cat(arrays)[selected]
        info = {"kind": "cifar10_official_training_split", "synthetic": False,
                "not_formal_result": False, "count": 45000, "split_seed": 12345,
                "reserved_validation_count": 5000, "test_set_accessed": False,
                "files": receipts, "train_indices_sha256": tensor_digest(selected),
                "pixels_sha256": tensor_digest(pixels)}
    info["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(info, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return TrainingData(pixels, info)

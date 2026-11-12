"""Compute pixel diagnostics and optional KID from explicitly supplied features.

No pretrained feature extractor or implicit download is included. KID requires
features extracted with the same network, weights and preprocessing on both
sets. This command cannot validate that external feature-extraction protocol.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .metrics import image_diagnostics, kid


def load_array(path):
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Expected a floating-point .npy array")
    return torch.from_numpy(array.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True, help="Raw float NCHW RGB .npy")
    parser.add_argument("--reference", type=Path, required=True, help="Float NCHW RGB .npy in [-1,1]")
    parser.add_argument("--generated-features", type=Path)
    parser.add_argument("--reference-features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if bool(args.generated_features) != bool(args.reference_features):
        parser.error("Supply both feature arrays, or neither")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    generated, reference = load_array(args.generated), load_array(args.reference)
    if (generated.ndim != 4 or reference.ndim != 4 or generated.shape[1] != 3
            or reference.shape[1] != 3 or reference.min() < -1 or reference.max() > 1):
        raise ValueError("Expected RGB NCHW; reference images must lie in [-1,1]")
    report = {"raw": image_diagnostics(generated, reference),
              "clamped": image_diagnostics(generated.clamp(-1, 1), reference.clamp(-1, 1)),
              "kid": None,
              "note": "Pixel distribution diagnostics are not semantic/perceptual quality scores."}
    if args.generated_features:
        features, ref_features = load_array(args.generated_features), load_array(args.reference_features)
        if len(features) != len(generated) or len(ref_features) != len(reference):
            raise ValueError("Features must correspond to the supplied image counts")
        report["kid"] = kid(features, ref_features)
        report["feature_protocol"] = "Externally supplied; extractor/preprocessing must be documented by the caller"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

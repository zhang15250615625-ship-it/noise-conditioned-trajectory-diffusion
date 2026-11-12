"""Sample a portable checkpoint's EMA with deterministic VP64 DDIM."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .core import BandDiffusion, haar_inverse
from .model import make_model, model_hash, warm_embedding
from .runtime import file_digest, load_checkpoint, resolve_device, save_new


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--steps", type=int, default=31, help="DDIM network evaluations; 1..64")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    return p


def run(args):
    if args.output.exists():
        raise FileExistsError("Sampling output exists; choose a new --output")
    if args.count < 1 or not 1 <= args.steps <= 64 or args.threads < 1 or args.seed < 0:
        raise ValueError("Expected positive count/threads, nonnegative seed, and DDIM steps in 1..64")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    model = make_model(checkpoint["config"]["variant"], checkpoint["config"]["seed"]).to(device)
    model.load_state_dict(checkpoint["ema_state"], strict=True)
    model.eval().requires_grad_(False)
    warm_embedding(device=device, batch=args.count)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        bands = BandDiffusion().sample(model, (args.count, 4, 3, 16, 16),
            sample_steps=args.steps, generator=generator, device=device,
            prediction_type="v", sampler="ddim")
        pixels = haar_inverse(bands).cpu()
    if not torch.isfinite(pixels).all():
        raise FloatingPointError("Generated nonfinite pixels")
    rendered = ((pixels.clamp(-1, 1)+1)*127.5).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/"raw_samples.npy").open("xb") as handle:
        np.save(handle, pixels.numpy(), allow_pickle=False)
    for index, array in enumerate(rendered):
        with (args.output/f"sample_{index:04d}.png").open("xb") as handle:
            Image.fromarray(array).save(handle, format="PNG")
    columns = math.ceil(math.sqrt(args.count))
    grid = Image.new("RGB", (columns*32, math.ceil(args.count/columns)*32), (255, 255, 255))
    for index, array in enumerate(rendered):
        grid.paste(Image.fromarray(array), ((index % columns)*32, (index//columns)*32))
    with (args.output/"grid.png").open("xb") as handle:
        grid.save(handle, format="PNG")
    synthetic = checkpoint["data_fingerprint"].get("synthetic", True)
    result = {"sampler": "DDIM", "prediction_type": "v", "diffusion_grid": 64,
              "nfe": args.steps, "count": args.count, "seed": args.seed,
              "checkpoint_sha256": file_digest(args.checkpoint), "ema_sha256": model_hash(model),
              "training_step": checkpoint["step"], "variant": checkpoint["config"]["variant"],
              "synthetic_training": synthetic, "not_formal_result": synthetic,
              "formal_quality_evaluation": False, "display": "clamp[-1,1], round to uint8",
              "raw_samples_file": "raw_samples.npy", "raw_samples_layout": "NCHW float32, unclipped",
              "raw_min": pixels.min().item(), "raw_max": pixels.max().item(),
              "raw_out_of_range_fraction": (pixels.abs() > 1).float().mean().item(),
              "note": "PNGs are actual EMA samples; smoke checkpoints do not demonstrate learned natural-image generation."}
    save_new(args.output/"sampling.json", result)
    return result


def main():
    print(run(parser().parse_args()))


if __name__ == "__main__":
    main()

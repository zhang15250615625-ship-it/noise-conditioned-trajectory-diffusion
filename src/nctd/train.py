"""Portable training reconstruction; not a replay of unpublished run history.

CPU smoke: python -m nctd.train --synthetic --steps 2 --batch 2 --output runs/smoke
Real training requires local official CIFAR-10 files. Trajectory fine-tuning
requires a trained portable parent EMA; initialization resets Adam and EMA
history. Resume instead restores the full state into a NEW output directory.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import platform
import time

import torch

from .core import BandDiffusion, haar_forward
from .data import load_training_data
from .model import make_model, make_optimizer, model_hash, parameter_count, warm_embedding
from .objective import trajectory_loss
from .runtime import (CHECKPOINT_FORMAT, capture_rng, cpu_tree, file_digest,
                      load_checkpoint, make_generators, resolve_device, restore_rng,
                      save_new, seed_global)

DEFAULTS = {
    "variant": "dense", "seed": 20, "steps": 2, "batch": 2,
    "lr": 2e-4, "delta_lr": 1e-4, "ema_decay": .999,
    "trajectory_weight": 0., "trajectory_span": 4, "trajectory_start_min": 4,
    "trajectory_normalize": True, "aux_batch": 1,
    "checkpoint_every": 1000, "synthetic_images": 64, "synthetic": False,
}


def validate_config(config):
    if set(config) != set(DEFAULTS):
        raise ValueError("Unknown or missing config keys: "+str(sorted(set(config)^set(DEFAULTS))))
    if config["variant"] not in {"dense", "noise", "static"}:
        raise ValueError("variant must be dense, noise, or static")
    for key in ("seed", "steps", "batch", "aux_batch", "checkpoint_every", "synthetic_images"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < (0 if key == "seed" else 1):
            raise ValueError(key+" must be a positive integer (seed may be zero)")
    for key in ("lr", "delta_lr", "ema_decay", "trajectory_weight"):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]):
            raise ValueError(key+" must be a finite number")
    if min(config["lr"], config["delta_lr"]) <= 0 or not 0 <= config["ema_decay"] < 1 or config["trajectory_weight"] < 0:
        raise ValueError("Invalid learning rate, EMA decay, or trajectory weight")
    if type(config["trajectory_span"]) is not int or config["trajectory_span"] not in (4, 8):
        raise ValueError("trajectory_span must be 4 or 8")
    if type(config["trajectory_start_min"]) is not int or not config["trajectory_span"] <= config["trajectory_start_min"] <= 64:
        raise ValueError("trajectory_start_min must lie between the span and 64")
    if type(config["trajectory_normalize"]) is not bool or type(config["synthetic"]) is not bool:
        raise ValueError("synthetic and trajectory_normalize must be booleans")
    if config["aux_batch"] > config["batch"]:
        raise ValueError("aux_batch cannot exceed batch")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--synthetic", action="store_true", default=None)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output", type=Path, default=Path("runs/smoke"))
    p.add_argument("--steps", type=int, help="Total target step, including the steps already completed by --resume")
    p.add_argument("--batch", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--init-from", type=Path, help="Initialize model/EMA/teacher from a trained portable parent EMA; reset Adam/RNG/step")
    source.add_argument("--resume", type=Path, help="Restore a complete portable checkpoint into a new output directory")
    return p


def read_config(args, resumed):
    config = copy.deepcopy(resumed["config"] if resumed is not None else DEFAULTS)
    if args.config:
        update = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(update, dict) or set(update)-set(DEFAULTS):
            raise ValueError("Config must be an object containing documented keys only")
        config.update(update)
    for key in ("steps", "batch", "synthetic"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    # A reduced CLI batch also bounds its auxiliary subset, useful for smoke
    # tests of a real-training config. The effective value is always recorded.
    if args.batch is not None:
        config["aux_batch"] = min(config["aux_batch"], config["batch"])
    validate_config(config)
    if resumed is not None:
        changed = [key for key in config if key != "steps" and config[key] != resumed["config"][key]]
        if changed:
            raise ValueError("Resume cannot change training/data configuration: "+", ".join(changed))
        if config["steps"] <= resumed["step"]:
            raise ValueError("--steps must exceed the checkpoint's completed step")
    return config


def initialize_from_parent(model, config, parent):
    source_variant = parent["config"]["variant"]
    if source_variant == config["variant"]:
        model.load_state_dict(parent["ema_state"], strict=True)
    elif source_variant == "dense" and config["variant"] in {"noise", "static"}:
        result = model.backbone.load_state_dict(parent["ema_state"], strict=False)
        expected = {name for name in model.backbone.state_dict()
                    if name.rsplit(".", 1)[-1] in {"delta_weight", "delta_bias"}}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError("Dense parent does not match the conditional backbone")
        if any(torch.count_nonzero(model.backbone.state_dict()[name]) for name in expected):
            raise ValueError("New conditional branches must be zero initialized")
    else:
        raise ValueError("Parent variant must match, or be dense when initializing noise/static")


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.mul_(decay).add_(source, alpha=1-decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def run(args):
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new --output, including for resume")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    resumed = load_checkpoint(args.resume) if args.resume else None
    parent = load_checkpoint(args.init_from) if args.init_from else None
    config = read_config(args, resumed)
    if config["trajectory_weight"] > 0 and resumed is None and not config["synthetic"]:
        if parent is None or parent["data_fingerprint"].get("synthetic", True):
            raise ValueError("Real trajectory training requires --init-from a trained, non-synthetic portable parent EMA")
    data = load_training_data(args.data_root, synthetic=config["synthetic"],
                              seed=config["seed"], synthetic_images=config["synthetic_images"])
    if resumed is not None and data.fingerprint != resumed["data_fingerprint"]:
        raise ValueError("Resume data fingerprint does not match the checkpoint")
    if parent is not None and not config["synthetic"] and parent["data_fingerprint"] != data.fingerprint:
        raise ValueError("A real fine-tuning parent must use the same verified training data")
    seed_global(config["seed"])
    generators = make_generators(config["seed"])
    model = make_model(config["variant"], config["seed"]).to(device)
    if parent is not None:
        initialize_from_parent(model, config, parent)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    teacher = copy.deepcopy(model).eval().requires_grad_(False) if config["trajectory_weight"] > 0 else None
    optimizer = make_optimizer(model, lr=config["lr"], delta_lr=config["delta_lr"])
    start = 0
    if resumed is not None:
        model.load_state_dict(resumed["model_state"], strict=True)
        ema.load_state_dict(resumed["ema_state"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state"])
        if (teacher is None) != (resumed["teacher_state"] is None):
            raise ValueError("Checkpoint teacher state conflicts with the configured objective")
        if teacher is not None:
            teacher.load_state_dict(resumed["teacher_state"], strict=True)
        start = resumed["step"]
    diffusion = BandDiffusion()
    warm_embedding(device=device, batch=config["batch"])
    if teacher is not None and config["aux_batch"] != config["batch"]:
        warm_embedding(device=device, batch=config["aux_batch"])
    if resumed is not None:
        # Last initialization operation: nothing above may consume resumed RNG.
        restore_rng(resumed["rng"], generators, device)
    initial_hash = model_hash(model)
    teacher_hash = model_hash(teacher) if teacher is not None else None
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "device": str(device), "threads": args.threads}
    provenance = {"mode": "resume" if resumed else "parent_ema_initialization" if parent else "from_scratch",
                  "parent_sha256": file_digest(args.resume or args.init_from) if (resumed or parent) else None,
                  "start_step": start, "initial_model_sha256": initial_hash,
                  "teacher_sha256": teacher_hash}
    save_new(args.output/"config.json", config)
    save_new(args.output/"data_fingerprint.json", data.fingerprint)
    save_new(args.output/"run_info.json", {"format": CHECKPOINT_FORMAT, "environment": environment,
             "provenance": provenance, "parameters": parameter_count(model),
             "portable_reconstruction": True, "historical_harness_replay": False,
             "synthetic": config["synthetic"], "not_formal_result": config["synthetic"],
             "note": "Uniform synthetic pixels test software only; this runner does not reproduce unpublished historical checkpoints."})
    started = time.perf_counter()
    last = None
    with (args.output/"training.jsonl").open("x", encoding="utf-8") as log:
        for step in range(start+1, config["steps"]+1):
            indices = torch.randint(len(data), (config["batch"],), generator=generators["batch"])
            clean = haar_forward(data.batch(indices, device))
            time_index = torch.randint(1, 65, (len(clean),), generator=generators["time"]).to(device)
            times = time_index.float()/64
            # CPU generators make training streams explicit and portable. GPU
            # kernels may still differ numerically across devices/frameworks.
            noise = torch.randn(clean.shape, generator=generators["noise"]).to(device)
            noisy = diffusion.corrupt(clean, times, noise)
            target = diffusion.training_target(clean, noise, times, prediction_type="v")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            anchor = (model(noisy, diffusion.log_snr(times))-target).square().mean()
            auxiliary, endpoint_mse = anchor.new_zeros(()), None
            if teacher is not None:
                size = config["aux_batch"]
                starts = torch.randint(config["trajectory_start_min"], 65, (size,),
                                       generator=generators["trajectory"]).to(device)
                auxiliary_noisy = diffusion.corrupt(clean[:size], starts.float()/64, noise[:size])
                auxiliary, receipt = trajectory_loss(
                    diffusion, model, teacher, auxiliary_noisy, starts,
                    span=config["trajectory_span"], start_min=config["trajectory_start_min"],
                    normalize=config["trajectory_normalize"])
                endpoint_mse = receipt["endpoint_mse"]
            loss = anchor+config["trajectory_weight"]*auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not torch.isfinite(grad):
                raise FloatingPointError("Nonfinite training gradient")
            optimizer.step()
            update_ema(ema, model, config["ema_decay"])
            last = {"step": step, "v_mse": anchor.item(), "trajectory_loss": auxiliary.item(),
                    "endpoint_mse": endpoint_mse, "total_loss": loss.item(),
                    "gradient_norm": grad.item(), "elapsed_seconds": time.perf_counter()-started,
                    "synthetic": config["synthetic"], "not_formal_result": config["synthetic"]}
            log.write(json.dumps(last, allow_nan=False)+"\n")
            log.flush()
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                checkpoint = {"format": CHECKPOINT_FORMAT, "config": config, "step": step,
                    "model_state": cpu_tree(model.state_dict()), "ema_state": cpu_tree(ema.state_dict()),
                    "optimizer_state": cpu_tree(optimizer.state_dict()),
                    "teacher_state": cpu_tree(teacher.state_dict()) if teacher is not None else None,
                    "rng": cpu_tree(capture_rng(generators)), "data_fingerprint": data.fingerprint,
                    "environment": environment, "provenance": provenance,
                    "ema_updates": step, "portable_reconstruction": True}
                save_new(args.output/f"checkpoint_{step:06d}.pt", checkpoint, binary=True)
            if step <= start+2 or step % 100 == 0 or step == config["steps"]:
                print(json.dumps(last, allow_nan=False), flush=True)
    if teacher is not None and model_hash(teacher) != teacher_hash:
        raise RuntimeError("Frozen teacher changed")
    summary = {"completed_step": config["steps"], "updates_this_invocation": config["steps"]-start,
               "checkpoint": f"checkpoint_{config['steps']:06d}.pt", "last_training_metrics": last,
               "model_sha256": model_hash(model), "ema_sha256": model_hash(ema),
               "teacher_unchanged": True if teacher is not None else None,
               "synthetic": config["synthetic"], "not_formal_result": config["synthetic"],
               "evaluation_performed": False}
    save_new(args.output/"summary.json", summary)
    return summary


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()

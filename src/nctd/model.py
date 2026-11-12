"""Two linear kernel branches; exact algebraic fusion at a common timestep.

No normalization or activation is moved/fused. Branch training supports distinct
times within a batch. Prepared inference explicitly refuses mixed timesteps,
stale weights, another model's cache, and gradient-enabled use.
"""
from contextvars import ContextVar
import copy
import hashlib
import math
import threading
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .core import BandDiffusion, check_bands, haar_forward, haar_inverse
from .vendor.ddpm_torch.models.unet import UNet

NAMES = (
    "net.upsamples.level_1.0.conv1",
    "net.upsamples.level_1.0.conv2",
    "net.upsamples.level_1.1.conv1",
    "net.upsamples.level_1.1.conv2",
)
_CONTEXT = ContextVar("nctd_time_kernel_context", default=None)
VARIANTS = ("dense", "static", "noise")
INIT_LOCK = threading.RLock()


class UNetAdapter(nn.Module):
    """Expose the pixel U-Net through the orthonormal Haar-band interface.

    The published default is RGB, width 32, with synchronous band clocks.
    Haar coefficients are an interface representation, not a learned block.
    """
    def __init__(self, representation="pixel", width=32, terminal_alpha=1e-5, grid=64):
        super().__init__()
        if representation not in {"pixel", "haar"} or width < 32 or width % 32:
            raise ValueError("Vendor GroupNorm needs width >=32 divisible by 32")
        self.representation = representation
        self.terminal_alpha, self.grid = terminal_alpha, grid
        channels = 3 if representation == "pixel" else 12
        self.net = UNet(channels, width, channels, (1, 2, 2), 1,
                        (False, False, False), drop_rate=0.)

    def forward(self, bands, log_snr):
        check_bands(bands)
        if bands.shape[2] != 3 or log_snr.shape != bands.shape[:2]:
            raise ValueError("Expected RGB bands and [B,4] schedule state")
        if not torch.isfinite(log_snr).all() or not torch.allclose(log_snr, log_snr[:, :1].expand_as(log_snr)):
            raise ValueError("This control supports only synchronous band clocks")
        alpha = log_snr[:, 0].double().sigmoid()
        cosine2 = ((alpha-self.terminal_alpha)/(1-self.terminal_alpha)).clamp(0, 1)
        timesteps = (cosine2.sqrt().acos()*(2*self.grid/math.pi)).float()
        if self.representation == "pixel":
            return haar_forward(self.net(haar_inverse(bands), timesteps))
        return self.net(bands.flatten(1, 2), timesteps).reshape_as(bands)


def make_model(variant="dense", seed=20):
    """Build the recorded width-32 model with the original initialization.

    A local RNG scope preserves the caller's CPU random state. Both kernel
    variants clone the same dense initialization and start with zero deltas.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown model variant")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    with INIT_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+200)
        source = UNetAdapter("pixel", 32)
        return source if variant == "dense" else TimeKernelUNet(source, variant)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def model_hash(model):
    """Hash all state tensors, including their names, in sorted order."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_optimizer(model, lr=2e-4, delta_lr=1e-4):
    """Adam with separate base/delta groups from the final joint-training arm."""
    if not all(math.isfinite(value) and value > 0 for value in (lr, delta_lr)):
        raise ValueError("Learning rates must be finite and positive")
    ordinary, delta = [], []
    for name, parameter in model.named_parameters():
        destination = delta if name.rsplit(".", 1)[-1] in {"delta_weight", "delta_bias"} else ordinary
        destination.append(parameter)
    groups = [{"params": ordinary, "lr": lr, "group_name": "base"}]
    if delta:
        groups.append({"params": delta, "lr": delta_lr, "group_name": "delta"})
    return torch.optim.Adam(groups, lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0.)


def warm_embedding(device="cpu", batch=2):
    """Warm the scripted embedding without consuming random numbers."""
    from .vendor.ddpm_torch.functions import get_timestep_embedding
    times = torch.full((batch,), 32., device=device)
    for _ in range(8):
        get_timestep_embedding(times, 32)


def signature(module):
    return tuple((id(p), p._version, str(p.device), str(p.dtype)) for p in module.parameters())


def tensor_bytes(tensors):
    unique = {(t.device, t.data_ptr()): t for t in tensors if t is not None}
    return sum(t.numel()*t.element_size() for t in unique.values())


class TimeKernelConv(nn.Module):
    def __init__(self, source, *, owner=None, name=""):
        super().__init__()
        if (source.weight.ndim != 4 or source.groups != 1
                or getattr(source, "padding_mode", "zeros") != "zeros"):
            raise ValueError("Only ungrouped zero-padded convolution is supported")
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.delta_weight = nn.Parameter(torch.zeros_like(self.weight))
        self.bias = None if source.bias is None else nn.Parameter(source.bias.detach().clone())
        self.delta_bias = None if source.bias is None else nn.Parameter(torch.zeros_like(self.bias))
        self.stride, self.padding, self.dilation = source.stride, source.padding, source.dilation
        self.owner, self.name = owner, name

    def conv(self, x, weight, bias):
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation)

    def branches(self, x, coefficient):
        if (coefficient.shape != (len(x),) or coefficient.device != x.device
                or coefficient.dtype != x.dtype or not torch.isfinite(coefficient).all()):
            raise ValueError("Expected one finite same-device/dtype coefficient per image")
        return self.conv(x, self.weight, self.bias)+coefficient[:, None, None, None]*self.conv(x, self.delta_weight, self.delta_bias)

    def fused_parameters(self, coefficient):
        return self.weight+coefficient*self.delta_weight, (
            None if self.bias is None else self.bias+coefficient*self.delta_bias)

    def fused_scalar(self, x, coefficient):
        """Differentiable algebraic oracle; not the cached inference path."""
        weight, bias = self.fused_parameters(coefficient)
        return self.conv(x, weight, bias)

    def forward(self, x):
        context = _CONTEXT.get()
        if context is None or context["owner"] is not self.owner:
            raise RuntimeError("Time kernel called outside its owning wrapper")
        if context["mode"] == "branches":
            return self.branches(x, context["coefficient"])
        weight, bias = context["kernels"][self.name]
        return self.conv(x, weight, bias)


@dataclass
class KernelBank:
    owner: object
    parameter_signature: tuple
    variant: str
    kernels: dict
    levels: dict
    versions: dict
    bytes: int


class TimeKernelUNet(nn.Module):
    def __init__(self, source, variant):
        super().__init__()
        if variant not in {"static", "noise"}:
            raise ValueError("Unknown prototype variant")
        self.variant = variant
        self.owner = object()
        self.backbone = copy.deepcopy(source).requires_grad_(True)
        modules = dict(self.backbone.named_modules())
        for name in NAMES:
            parent, leaf = name.rsplit(".", 1)
            setattr(modules[parent], leaf, TimeKernelConv(modules[name], owner=self.owner, name=name))

    def kernel_modules(self):
        return {n: m for n, m in self.backbone.named_modules() if isinstance(m, TimeKernelConv)}

    def coefficient(self, levels):
        if levels.ndim != 2 or levels.shape[1] != 4 or not torch.isfinite(levels).all():
            raise ValueError("Expected finite synchronous Bx4 log-SNR")
        if not torch.allclose(levels, levels[:, :1].expand_as(levels), atol=1e-6, rtol=0):
            raise ValueError("Band clocks differ")
        if self.variant == "static":
            return torch.ones_like(levels[:, 0])
        # g=2 sigma^2-1 in [-1,1]. Fixed, parameter-free, not learned routing.
        return (2*torch.sigmoid(-levels[:, 0].double())-1).to(levels.dtype)

    def forward(self, bands, levels):
        coefficient = self.coefficient(levels)
        token = _CONTEXT.set({"owner": self.owner, "mode": "branches", "coefficient": coefficient})
        try:
            return self.backbone(bands, levels)
        finally:
            _CONTEXT.reset(token)

    @torch.no_grad()
    def prepare(self, indices):
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError("Prepared inference requires an eval, frozen model")
        indices = tuple(indices)
        if (not indices or len(set(indices)) != len(indices)
                or any(isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= 64 for i in indices)):
            raise ValueError("Expected distinct positive VP64 indices")
        parameter = next(self.parameters())
        modules = self.kernel_modules(); kernels, levels = {}, {}
        shared = None
        for index in indices:
            level = BandDiffusion().log_snr(torch.tensor([index/64], device=parameter.device, dtype=parameter.dtype))
            coefficient = self.coefficient(level)[0]
            if self.variant == "static" and shared is not None:
                packed = shared
            else:
                packed = {name: tuple(t.detach() if t is not None else None for t in module.fused_parameters(coefficient))
                          for name, module in modules.items()}
                if self.variant == "static": shared = packed
            kernels[index], levels[index] = packed, level
        tensors = [t for pack in kernels.values() for pair in pack.values() for t in pair if t is not None]
        tensors.extend(levels.values())
        versions = {index: tuple((id(t), t._version) for pair in pack.values() for t in pair if t is not None)+((id(levels[index]), levels[index]._version),)
                    for index, pack in kernels.items()}
        return KernelBank(self.owner, signature(self), self.variant, kernels, levels, versions, tensor_bytes(tensors))

    def merged_forward(self, bands, levels, *, index, bank):
        if torch.is_grad_enabled() or self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError("Cached merge is inference-only")
        if bank.owner is not self.owner or bank.variant != self.variant or bank.parameter_signature != signature(self):
            raise ValueError("Foreign or stale parameter cache")
        if index not in bank.kernels:
            raise ValueError("Timestep absent from prepared cache")
        actual = tuple((id(t), t._version) for pair in bank.kernels[index].values() for t in pair if t is not None)+((id(bank.levels[index]), bank.levels[index]._version),)
        if actual != bank.versions[index]:
            raise ValueError("Prepared kernel cache modified")
        expected = bank.levels[index]
        if (levels.shape != (len(bands), 4) or levels.device != expected.device or levels.dtype != expected.dtype
                or not torch.allclose(levels, expected.expand_as(levels), atol=2e-6, rtol=0)):
            raise ValueError("Mixed/wrong timestep for prepared kernel")
        token = _CONTEXT.set({"owner": self.owner, "mode": "merged", "kernels": bank.kernels[index]})
        try:
            return self.backbone(bands, levels)
        finally:
            _CONTEXT.reset(token)


@torch.no_grad()
def set_nonzero_probe(model, *, seed=20261026, relative_scale=.01):
    """Synthetic perturbation to test nontrivial fusion; NOT learning/training."""
    generator = torch.Generator().manual_seed(seed)
    for module in model.kernel_modules().values():
        scale = module.weight.detach().cpu().double().square().mean().sqrt().item()*relative_scale
        probe = torch.randn(module.delta_weight.shape, generator=generator, dtype=torch.float64)*scale
        module.delta_weight.copy_(probe.to(module.delta_weight))
        if module.delta_bias is not None:
            module.delta_bias.copy_((torch.randn(module.delta_bias.shape, generator=generator, dtype=torch.float64)*scale).to(module.delta_bias))


def original_tensors_unchanged(model, source):
    state = model.backbone.state_dict()
    return all(name in state and torch.equal(value, state[name]) for name, value in source.state_dict().items())


def architecture(model, source):
    return {"variant": model.variant, "locations": list(NAMES), "baseline_parameters": sum(p.numel() for p in source.parameters()),
        "prototype_parameters": sum(p.numel() for p in model.parameters()),
        "prototype_parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
        "added_parameters": sum(p.numel() for n, p in model.named_parameters() if ".delta_" in n),
        "original_tensors_unchanged": original_tensors_unchanged(model, source),
        "weights_sha256": model_hash(model)}


def sampler_indices(nfe):
    if isinstance(nfe, bool) or not isinstance(nfe, int) or not 1 <= nfe <= 64:
        raise ValueError("Invalid DDIM NFE")
    return tuple(torch.linspace(64, 0, nfe+1, dtype=torch.float64).round().long().tolist()[:-1])


class PreparedSampler:
    """Map the unchanged sampler's call order to explicit cached positive times."""
    def __init__(self, model, bank, nfe):
        self.model, self.bank, self.indices = model, bank, sampler_indices(nfe)
        self.calls = 0

    @property
    def training(self): return self.model.training

    def eval(self): self.model.eval()

    def train(self, mode=True): self.model.train(mode)

    def __call__(self, z, levels):
        index = self.indices[self.calls % len(self.indices)]
        result = self.model.merged_forward(z, levels, index=index, bank=self.bank)
        self.calls += 1
        return result

"""Small CPU distribution diagnostics; none is FID or semantic image quality."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def sliced_wasserstein(x, y, *, projections=64, quantiles=128, seed=31415):
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(x.shape+y.shape) < 1:
        raise ValueError("Expected nonempty ND feature matrices with matching dimensions")
    if projections < 1 or quantiles < 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Invalid projections, quantiles, or features")
    x, y = x.detach().cpu().double(), y.detach().cpu().double()
    directions = torch.randn(x.shape[1], projections, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    directions = directions/directions.norm(dim=0).clamp_min(1e-12)
    q = (torch.arange(quantiles, dtype=torch.float64)+.5)/quantiles
    # Quantile approximation to sliced W1, supporting unequal sample counts.
    a, b = torch.quantile(x@directions, q, dim=0), torch.quantile(y@directions, q, dim=0)
    return (a-b).abs().mean().item()


@torch.no_grad()
def image_diagnostics(generated, reference):
    if (generated.ndim != 4 or reference.ndim != 4 or generated.shape[1:] != reference.shape[1:] or
            min(generated.shape+reference.shape) < 1 or min(generated.shape[-2:]) < 4):
        raise ValueError("Expected matching nonempty BCHW images with H,W >= 4")
    generated, reference = generated.detach().cpu().float(), reference.detach().cpu().float()
    if not torch.isfinite(generated).all() or not torch.isfinite(reference).all():
        raise ValueError("Non-finite images")
    lowpass = [F.adaptive_avg_pool2d(x, (8, 8)).flatten(1) for x in (generated, reference)]
    patches = []
    for x in (generated, reference):
        p = F.unfold(x, kernel_size=4, stride=4).transpose(1, 2).flatten(0, 1)
        if len(p) > 4096:
            p = p[torch.randperm(len(p), generator=torch.Generator().manual_seed(27182))[:4096]]
        patches.append(p)
    return {"lowpass_swd": sliced_wasserstein(*lowpass), "patch_swd": sliced_wasserstein(*patches),
            "generated_mean": generated.mean().item(), "reference_mean": reference.mean().item(),
            "generated_variance": generated.double().var(unbiased=False).item(),
            "reference_variance": reference.double().var(unbiased=False).item(),
            "generated_fraction_outside_unit_range": (generated.abs() > 1).float().mean().item(),
            "generated_images": len(generated), "reference_images": len(reference),
            "note": "Fixed 64 random projections/128 quantiles, pooled 8x8 RGB and nonoverlapping 4x4 patches; no clipping. Diagnostic proxies, NOT FID, semantics or perceptual quality."}


def polynomial_mmd(x, y):
    """Unbiased MMD², k(x,y)=(x.y/d+1)^3; negative values are valid."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(len(x), len(y)) < 2:
        raise ValueError("Need matching feature dimension and >=2 samples per group")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Nonfinite features")
    x, y = x.cpu().double(), y.cpu().double()
    xx, yy, xy = [(a@b.T/x.shape[1]+1).pow(3) for a, b in ((x,x), (y,y), (x,y))]
    return ((xx.sum()-xx.diag().sum())/(len(x)*(len(x)-1)) +
            (yy.sum()-yy.diag().sum())/(len(y)*(len(y)-1)) - 2*xy.mean()).item()

def kid(x, y, *, seed=9317, subsets=50, subset_size=256):
    if min(len(x), len(y)) < subset_size or subset_size < 2 or subsets < 2:
        raise ValueError("Invalid KID subset settings")
    g = torch.Generator().manual_seed(seed)
    values = [polynomial_mmd(x[torch.randperm(len(x), generator=g)[:subset_size]],
                             y[torch.randperm(len(y), generator=g)[:subset_size]])
              for _ in range(subsets)]
    v = torch.tensor(values, dtype=torch.float64)
    return {"mean": v.mean().item(), "subset_std": v.std(unbiased=False).item(),
            "subsets": subsets, "subset_size": subset_size, "seed": seed,
            "full_unbiased_mmd2": polynomial_mmd(x, y),
            "note": "Raw KID, not x1000. Subset std is NOT training-seed uncertainty or a confidence interval."}

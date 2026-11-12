"""Differentiable two-step DDIM student and frozen four-step reference.

The auxiliary target is a model-generated endpoint, not a ground-truth image
or a ground-truth trajectory. The diffusion grid is fixed at 64 steps.
"""
import torch


class CallCounter:
    """Count real forwards/sample-forwards, including gradient-enabled ones."""
    def __init__(self, model):
        self.model = model
        self.calls = self.samples = self.grad_calls = self.grad_samples = 0

    def __call__(self, z, levels):
        self.calls += 1
        self.samples += len(z)
        if torch.is_grad_enabled():
            self.grad_calls += 1
            self.grad_samples += len(z)
        return self.model(z, levels)

    def receipt(self):
        return {k: getattr(self, k) for k in ("calls", "samples", "grad_calls", "grad_samples")}


def ddim_step(diffusion, z, prediction, current_index, next_index):
    """Standard VP/v DDIM with gradients through the complete student step."""
    if diffusion.config.steps != 64:
        raise ValueError("The local trajectory objective requires the VP64 grid")
    if (current_index.shape != (len(z),) or next_index.shape != current_index.shape
            or current_index.device != z.device or next_index.device != z.device
            or current_index.dtype != torch.long or next_index.dtype != torch.long
            or (next_index < 0).any() or (current_index > 64).any()
            or (current_index <= next_index).any()):
        raise ValueError("Expected ordered integer grid times in [0,64]")
    later, earlier = current_index.to(z.dtype)/64, next_index.to(z.dtype)/64
    clean, eps = diffusion.decode_prediction(z, prediction, later, prediction_type="v")
    abar = diffusion.alpha_bar(earlier)
    return diffusion.broadcast(abar.sqrt(), z)*clean + diffusion.broadcast((1-abar).sqrt(), z)*eps


def rollout(diffusion, model, z, start_index, offsets):
    for current, previous in zip(offsets[:-1], offsets[1:]):
        t, s = start_index-current, start_index-previous
        prediction = model(z, diffusion.log_snr(t.to(z.dtype)/64))
        z = ddim_step(diffusion, z, prediction, t, s)
    return z


def trajectory_loss(diffusion, student, teacher, noisy, start_index, *,
                    span=4, start_min=None, normalize=True):
    """Compare two student steps to four teacher steps over span 4 or 8.

    ``start_min`` validates a chosen training range; callers must sample starts
    from that range. ``normalize=False`` returns raw endpoint MSE without
    changing the external loss weight. Neither student's state is detached.
    """
    if isinstance(span, bool) or not isinstance(span, int) or span not in {4, 8}:
        raise ValueError("span must be 4 or 8")
    if start_min is None:
        start_min = span
    if isinstance(start_min, bool) or not isinstance(start_min, int) or not span <= start_min <= 64:
        raise ValueError("start_min must be an integer between span and 64")
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a boolean")
    if (start_index.shape != (len(noisy),) or start_index.dtype != torch.long
            or start_index.device != noisy.device or (start_index < start_min).any()
            or (start_index > 64).any()):
        raise ValueError("Short trajectory requires integer starts in the declared range")
    prediction = rollout(diffusion, student, noisy, start_index, (0, span//2, span))
    with torch.no_grad():
        reference = rollout(diffusion, teacher, noisy.detach(), start_index,
                            (0, span//4, span//2, 3*span//4, span))
        t, s = start_index.to(noisy.dtype)/64, (start_index-span).to(noisy.dtype)/64
        angle = diffusion.alpha_bar(t).sqrt().acos()-diffusion.alpha_bar(s).sqrt().acos()
        scale = diffusion.broadcast(angle.sin(), noisy)
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise FloatingPointError("Invalid trajectory normalization")
    residual = prediction-reference.detach()
    if normalize:
        residual = residual/scale
    loss = residual.square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite trajectory objective")
    return loss, {
        "endpoint_mse": (prediction.detach()-reference).square().mean().item(),
        "span": span,
        "start_min": start_min,
        "normalized": normalize,
        "normalization": (
            f"sin(theta(start)-theta(start-{span})); not a ground-truth trajectory error"
            if normalize else "none; raw model-endpoint MSE"
        ),
    }

"""Orthogonal Haar representation and covariance-consistent Gaussian diffusion.

Time 0 is clean; time 1 has alpha_bar=terminal_alpha (approximately noise).
Every sampling trajectory has ONE fixed path parameter delta. Different bands
can have different clocks, but reverse transitions use their actual coefficients.
This is a Gaussian diffusion prototype, not a new diffusion theory.
"""
from dataclasses import dataclass
import math

import torch


PREDICTION_TYPES = ("epsilon", "v")
SAMPLERS = ("ancestral", "ddim", "flow_euler", "flow_heun")


def check_prediction_type(prediction_type):
    if prediction_type not in PREDICTION_TYPES:
        raise ValueError("prediction_type must be epsilon or v")


def sampler_nfe(sampler, steps):
    if sampler not in SAMPLERS or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Invalid sampler or number of steps")
    # At t=0, E[v|x0]=E[epsilon|x0]=0 exactly; Heun uses that endpoint,
    # not an untrained extrapolation of the model to log-SNR infinity.
    return 2*steps-1 if sampler == "flow_heun" else steps


def _floating(x, name):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise ValueError(name + " must be a floating-point tensor")
    if not torch.isfinite(x).all():
        raise ValueError(name + " must be finite")


def check_bands(x):
    _floating(x, "bands")
    if x.ndim != 5 or x.shape[1] != 4 or min(x.shape) < 1:
        raise ValueError("Expected nonempty [batch, 4, channels, height, width]")


def haar_forward(x):
    """One orthonormal level, order LL, LH, HL, HH; no padding or resizing."""
    _floating(x, "images")
    if (x.ndim != 4 or min(x.shape) < 1 or
            x.shape[-2] % 2 or x.shape[-1] % 2):
        raise ValueError("Expected nonempty BCHW images with even H and W")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return torch.stack(((a+b+c+d)/2, (a-b+c-d)/2,
                        (a+b-c-d)/2, (a-b-c+d)/2), dim=1)


def haar_inverse(z):
    check_bands(z)
    ll, lh, hl, hh = z.unbind(1)
    a, b = (ll+lh+hl+hh)/2, (ll-lh+hl-hh)/2
    c, d = (ll+lh-hl-hh)/2, (ll-lh-hl+hh)/2
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    return torch.stack((top, bottom), dim=-2).flatten(-3, -2)


@dataclass(frozen=True)
class DiffusionConfig:
    steps: int = 64
    terminal_alpha: float = 1e-5

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 2:
            raise ValueError("steps must be an integer >= 2")
        if not math.isfinite(self.terminal_alpha) or not 0 < self.terminal_alpha < 1e-2:
            raise ValueError("terminal_alpha must be finite and in (0, .01)")


class BandDiffusion:
    def __init__(self, config=None):
        self.config = config or DiffusionConfig()

    def alpha_bar(self, t, delta=0.0):
        _floating(t, "time")
        if t.ndim != 1 or t.numel() == 0 or ((t < 0) | (t > 1)).any():
            raise ValueError("time must be a nonempty vector in [0,1]")
        td = t.to(torch.float64)
        delta = torch.as_tensor(delta, device=t.device, dtype=torch.float64)
        if delta.ndim == 0:
            delta = delta.expand(t.shape[0])
        if delta.shape != t.shape or not torch.isfinite(delta).all() or (delta.abs() >= 1).any():
            raise ValueError("delta must be scalar or batch-vector, with abs(delta) < 1")
        signs = td.new_tensor([-1, 1, 1, 1])
        tau = td[:, None] + delta[:, None]*signs*torch.sin(math.pi*td[:, None])/math.pi
        # Cosine alpha_bar, with identical clean/noise endpoints for every band.
        abar = self.config.terminal_alpha + (1-self.config.terminal_alpha)*torch.cos(math.pi*tau/2).square()
        return abar

    @staticmethod
    def broadcast(coeff, reference):
        return coeff.to(device=reference.device, dtype=reference.dtype)[:, :, None, None, None]

    def log_snr(self, t, delta=0.0):
        abar = self.alpha_bar(t, delta).clamp(1e-12, 1-1e-12)
        # Schedule coefficient ratio, NOT true signal-to-noise ratio of content.
        return (abar.log()-torch.log1p(-abar)).to(t.dtype)

    def angle_derivative(self, t, delta=0.0):
        """d theta_k/dt for theta=acos(sqrt(abar)); finite at both endpoints.

        For the diagonal VP diffusion, the probability-flow ODE is dz_k/dt =
        theta'_k * E[v_k|z_t]. This is an existing diffusion ODE, not MeanFlow.
        """
        abar = self.alpha_bar(t, delta)  # validates times and paths
        td = t.double()
        d = torch.as_tensor(delta, device=t.device, dtype=torch.float64)
        if d.ndim == 0:
            d = d.expand_as(td)
        signed = d[:, None]*td.new_tensor([-1., 1., 1., 1.])
        tau = td[:, None]+signed*torch.sin(math.pi*td[:, None])/math.pi
        rate = 1+signed*torch.cos(math.pi*td[:, None])
        return (math.pi/2)*math.sqrt(1-self.config.terminal_alpha)*torch.cos(math.pi*tau/2)*rate/abar.sqrt()

    def corrupt(self, clean, t, noise, delta=0.0):
        check_bands(clean)
        check_bands(noise)
        if noise.shape != clean.shape or noise.device != clean.device or noise.dtype != clean.dtype:
            raise ValueError("noise and clean bands must have identical shape/device/dtype")
        if t.shape != (clean.shape[0],) or t.device != clean.device:
            raise ValueError("time must match batch and device")
        abar = self.alpha_bar(t, delta)
        a, s = self.broadcast(abar.sqrt(), clean), self.broadcast((1-abar).sqrt(), clean)
        return a*clean + s*noise

    def decode_clean(self, noisy, predicted_noise, t, delta=0.0):
        check_bands(noisy)
        if predicted_noise.shape != noisy.shape:
            raise ValueError("prediction shape mismatch")
        abar = self.alpha_bar(t, delta)
        return (noisy-self.broadcast((1-abar).sqrt(), noisy)*predicted_noise)/self.broadcast(abar.sqrt(), noisy)

    def _prediction_coefficients(self, reference, other, t, delta):
        check_bands(reference)
        check_bands(other)
        if other.shape != reference.shape or other.device != reference.device or other.dtype != reference.dtype:
            raise ValueError("prediction/target tensors must match shape/device/dtype")
        if t.shape != (reference.shape[0],) or t.device != reference.device:
            raise ValueError("time must match batch and device")
        abar = self.alpha_bar(t, delta)
        return self.broadcast(abar.sqrt(), reference), self.broadcast((1-abar).sqrt(), reference)

    def training_target(self, clean, noise, t, delta=0.0, *, prediction_type="epsilon"):
        """Standard parameterization control, NOT a new diffusion objective.

        v = sqrt(abar)*epsilon - sqrt(1-abar)*x0, independently per band.
        Unweighted v MSE and epsilon MSE imply different timestep weighting.
        """
        check_prediction_type(prediction_type)
        a, s = self._prediction_coefficients(clean, noise, t, delta)
        return noise if prediction_type == "epsilon" else a*noise-s*clean

    def decode_prediction(self, noisy, prediction, t, delta=0.0, *, prediction_type="epsilon"):
        """Return (clean, epsilon); v decoding never divides by sqrt(abar).

        Do not convert v to epsilon and then decode clean: that would reintroduce
        cancellation and division at the high-noise endpoint.
        """
        check_prediction_type(prediction_type)
        a, s = self._prediction_coefficients(noisy, prediction, t, delta)
        if prediction_type == "epsilon":
            return self.decode_clean(noisy, prediction, t, delta), prediction
        return a*noisy-s*prediction, s*noisy+a*prediction

    def posterior(self, noisy, clean, earlier, later, delta=0.0):
        """Exact q(z_earlier | z_later, x0) for the aggregated Markov transition.

        Supports skipped grid points, not arbitrary replacement of beta by time.
        Variance is per band; a model substitutes its clean prediction for x0.
        """
        check_bands(noisy)
        check_bands(clean)
        batch = noisy.shape[0]
        if clean.shape != noisy.shape or clean.device != noisy.device or clean.dtype != noisy.dtype:
            raise ValueError("clean and noisy bands must match")
        if earlier.shape != (batch,) or later.shape != (batch,):
            raise ValueError("transition times must match the batch")
        if earlier.device != noisy.device or later.device != noisy.device:
            raise ValueError("transition times must match the device")
        ap, at = self.alpha_bar(earlier, delta), self.alpha_bar(later, delta)
        if (earlier >= later).any():
            raise ValueError("earlier must be strictly less than later")
        ratio = at/ap
        beta = 1-ratio
        c0 = ap.sqrt()*beta/(1-at)
        ct = ratio.sqrt()*(1-ap)/(1-at)
        variance = (beta*(1-ap)/(1-at)).clamp_min(0)
        mean = self.broadcast(c0, noisy)*clean+self.broadcast(ct, noisy)*noisy
        return mean, self.broadcast(variance, noisy)

    def reverse_step(self, noisy, predicted_noise, earlier, later, noise, delta=0.0, *, prediction_type="epsilon"):
        if noise.shape != noisy.shape or noise.device != noisy.device or noise.dtype != noisy.dtype:
            raise ValueError("reverse noise must match the current state")
        clean, _ = self.decode_prediction(noisy, predicted_noise, later, delta, prediction_type=prediction_type)
        mean, variance = self.posterior(noisy, clean, earlier, later, delta)
        return mean+variance.sqrt()*noise

    @torch.no_grad()
    def sample(self, model, shape, *, delta=0.0, sample_steps=None, generator=None,
               device="cpu", dtype=torch.float32, prediction_type="epsilon", sampler="ancestral"):
        check_prediction_type(prediction_type)
        if len(shape) != 5 or shape[1] != 4 or min(shape) < 1:
            raise ValueError("sample shape must be nonempty B4CHW")
        count = self.config.steps if sample_steps is None else sample_steps
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= self.config.steps:
            raise ValueError("sample_steps must be in [1, diffusion steps]")
        sampler_nfe(sampler, count)
        if sampler.startswith("flow_") and prediction_type != "v":
            raise ValueError("Probability-flow controls require v prediction")
        grid = torch.linspace(self.config.steps, 0, count+1, dtype=torch.float64).round().to(torch.long)
        if not (grid[:-1] > grid[1:]).all():
            raise ValueError("sampling grid must be strictly decreasing")
        # Validate path before consuming RNG or altering model mode.
        self.alpha_bar(torch.ones(shape[0], device=device, dtype=dtype), delta)
        training = model.training
        model.eval()
        try:
            z = torch.randn(shape, device=device, dtype=dtype, generator=generator)
            for current, previous in zip(grid[:-1].tolist(), grid[1:].tolist()):
                later = torch.full((shape[0],), current/self.config.steps, device=device, dtype=dtype)
                earlier = torch.full_like(later, previous/self.config.steps)
                prediction = model(z, self.log_snr(later, delta))
                if sampler == "ancestral":
                    noise = torch.zeros_like(z) if previous == 0 else torch.randn(shape, device=device, dtype=dtype, generator=generator)
                    z = self.reverse_step(z, prediction, earlier, later, noise, delta, prediction_type=prediction_type)
                elif sampler == "ddim":
                    clean, eps = self.decode_prediction(z, prediction, later, delta, prediction_type=prediction_type)
                    ap = self.alpha_bar(earlier, delta)
                    z = self.broadcast(ap.sqrt(), z)*clean+self.broadcast((1-ap).sqrt(), z)*eps
                else:
                    dt = self.broadcast((earlier-later)[:, None].expand(-1, 4), z)
                    drift = self.broadcast(self.angle_derivative(later, delta), z)*prediction
                    proposal = z+dt*drift
                    if sampler == "flow_heun":
                        end_drift = torch.zeros_like(z) if previous == 0 else (
                            self.broadcast(self.angle_derivative(earlier, delta), z)*
                            model(proposal, self.log_snr(earlier, delta)))
                        z = z+dt*(drift+end_drift)/2
                    else:
                        z = proposal
                if not torch.isfinite(z).all():
                    raise FloatingPointError("non-finite generated state")
            return z
        finally:
            model.train(training)


def synthetic_bands(count, channels, image_size, rho, generator=None):
    """Unit-variance Gaussian bands with off-diagonal covariance rho.

    Gaussian synthetic images have unbounded range; never clip or call them CIFAR.
    """
    if min(count, channels, image_size) < 1 or image_size % 2 or not 0 <= rho < 1:
        raise ValueError("Invalid synthetic dimensions or rho (requires [0,1))")
    shape = (count, 4, channels, image_size//2, image_size//2)
    common = torch.randn((count, 1)+shape[2:], generator=generator)
    independent = torch.randn(shape, generator=generator)
    return math.sqrt(rho)*common+math.sqrt(1-rho)*independent


def gaussian_oracle(noisy, abar, rho):
    """Exact conditional clean/noise means and clean MSE for synthetic_bands.

    Oracle is evaluation-only, not an input or teacher for the learned model.
    """
    check_bands(noisy)
    if abar.shape != noisy.shape[:2] or not 0 <= rho < 1:
        raise ValueError("Invalid oracle parameters")
    if not torch.isfinite(abar).all() or ((abar < 0) | (abar > 1)).any():
        raise ValueError("alpha_bar must lie in [0,1]")
    abar = abar.to(device=noisy.device, dtype=torch.float64)
    cov = (1-rho)*torch.eye(4, device=noisy.device, dtype=torch.float64)+rho
    a, s = abar.sqrt(), (1-abar).sqrt()
    obs_cov = a[:, :, None]*cov*a[:, None, :]+torch.diag_embed(1-abar)
    flat = noisy.to(torch.float64).flatten(2)
    solved = torch.linalg.solve(obs_cov, flat)
    cross = cov[None, :, :]*a[:, None, :]
    clean = torch.bmm(cross, solved).reshape_as(noisy)
    eps = (s[:, :, None]*solved).reshape_as(noisy)
    posterior_cov = cov[None, :, :]-torch.bmm(cross, torch.linalg.solve(obs_cov, cross.transpose(1, 2)))
    return clean.to(noisy.dtype), eps.to(noisy.dtype), posterior_cov.diagonal(dim1=1, dim2=2).clamp_min(0)

"""Public, CPU-only numerical tests without datasets or historical checkpoints."""
import copy
import unittest

import torch

from nctd.core import BandDiffusion, DiffusionConfig, haar_forward, haar_inverse
from nctd.model import make_model, make_optimizer, parameter_count, set_nonzero_probe
from nctd.objective import CallCounter, trajectory_loss


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_haar_roundtrip_and_energy(self):
        pixels = torch.randn(2, 3, 32, 32, dtype=torch.float64)
        bands = haar_forward(pixels)
        torch.testing.assert_close(haar_inverse(bands), pixels, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(bands.square().sum(), pixels.square().sum())

    def test_v_prediction_inverts_corruption_at_endpoints(self):
        diffusion = BandDiffusion()
        clean, noise = torch.randn(2, 4, 3, 16, 16), torch.randn(2, 4, 3, 16, 16)
        times = torch.tensor([0., 1.])
        noisy = diffusion.corrupt(clean, times, noise)
        target = diffusion.training_target(clean, noise, times, prediction_type="v")
        recovered, epsilon = diffusion.decode_prediction(noisy, target, times, prediction_type="v")
        torch.testing.assert_close(recovered, clean)
        torch.testing.assert_close(epsilon, noise)

    def test_zero_delta_matches_dense_and_preserves_rng(self):
        rng = torch.get_rng_state().clone()
        dense = make_model("dense", seed=20)
        noise = make_model("noise", seed=20)
        static = make_model("static", seed=20)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(parameter_count(dense), 1063651)
        self.assertEqual(parameter_count(noise), 1266659)
        self.assertEqual(parameter_count(static), 1266659)
        bands = torch.randn(2, 4, 3, 16, 16)
        levels = BandDiffusion().log_snr(torch.tensor([.125, .875]))
        with torch.no_grad():
            # TorchScript embedding warmup can change float32 rounding on the
            # first call; tolerate only the resulting tiny numerical error.
            torch.testing.assert_close(dense(bands, levels), noise(bands, levels), rtol=1e-5, atol=2e-10)
            torch.testing.assert_close(dense(bands, levels), static(bands, levels), rtol=1e-5, atol=2e-10)

    def test_gate_and_fused_convolution_with_nonzero_delta(self):
        model = make_model("noise")
        set_nonzero_probe(model)
        levels = BandDiffusion().log_snr(torch.tensor([.125, .875]))
        coefficient = model.coefficient(levels)
        self.assertLess(coefficient[0].item(), 0)
        self.assertGreater(coefficient[1].item(), 0)
        module = next(iter(model.kernel_modules().values()))
        x = torch.randn(2, module.weight.shape[1], 16, 16)
        branches = module.branches(x, coefficient)
        separate = torch.cat([module.fused_scalar(x[i:i+1], coefficient[i]) for i in range(2)])
        torch.testing.assert_close(branches, separate, atol=3e-6, rtol=3e-5)

    def test_local_trajectory_gradient_and_teacher_calls(self):
        diffusion = BandDiffusion()
        for span, start_min in ((4, 4), (4, 8), (8, 8)):
            with self.subTest(span=span, start_min=start_min):
                student = make_model("noise")
                teacher = copy.deepcopy(student).eval()
                # Keep teacher parameters requiring gradients to also verify no_grad.
                counted_student, counted_teacher = CallCounter(student), CallCounter(teacher)
                z = torch.randn(1, 4, 3, 16, 16, requires_grad=True)
                loss, _ = trajectory_loss(diffusion, counted_student, counted_teacher, z,
                                          torch.tensor([32]), span=span, start_min=start_min)
                loss.backward()
                self.assertEqual(counted_student.calls, 2)
                self.assertEqual(counted_student.grad_calls, 2)
                self.assertEqual(counted_teacher.calls, 4)
                self.assertEqual(counted_teacher.grad_calls, 0)
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                self.assertGreater(sum(p.grad.abs().sum().item() for p in student.parameters()
                                       if p.grad is not None), 0)
                self.assertIsNotNone(z.grad)
                self.assertTrue(torch.isfinite(z.grad).all())

    def test_invalid_trajectory_ranges_rejected(self):
        z = torch.randn(1, 4, 3, 4, 4)
        zero = lambda noisy, levels: torch.zeros_like(noisy)
        for span, minimum, start in ((8, 8, 7), (4, 8, 7), (4, 4, 65), (3, 3, 32)):
            with self.subTest(span=span, minimum=minimum, start=start):
                with self.assertRaises(ValueError):
                    trajectory_loss(BandDiffusion(), zero, zero, z, torch.tensor([start]),
                                    span=span, start_min=minimum)
        with self.assertRaises(ValueError):
            trajectory_loss(BandDiffusion(DiffusionConfig(32)), zero, zero, z, torch.tensor([16]))

    def test_final_optimizer_groups(self):
        model = make_model("noise")
        optimizer = make_optimizer(model)
        groups = optimizer.param_groups
        self.assertEqual([g["lr"] for g in groups], [2e-4, 1e-4])
        delta_ids = {id(p) for name, p in model.named_parameters()
                     if name.rsplit(".", 1)[-1] in {"delta_weight", "delta_bias"}}
        self.assertEqual(delta_ids, {id(p) for p in groups[1]["params"]})

    def test_ddim_finite_and_restores_training_mode(self):
        model = make_model("noise")
        model.train()
        images = BandDiffusion().sample(model, (1, 4, 3, 16, 16), sample_steps=4,
                                       prediction_type="v", sampler="ddim",
                                       generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.isfinite(images).all())
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()

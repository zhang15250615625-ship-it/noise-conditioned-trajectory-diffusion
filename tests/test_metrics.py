import unittest

import torch

from nctd.metrics import image_diagnostics, kid, polynomial_mmd


class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identical_images_have_zero_distribution_distances(self):
        images = torch.randn(8, 3, 32, 32)
        result = image_diagnostics(images, images.clone())
        self.assertEqual(result["lowpass_swd"], 0.)
        self.assertEqual(result["patch_swd"], 0.)

    def test_kid_uses_unbiased_off_diagonal_kernel(self):
        # For this scalar example: kxx=[[1,1],[1,8]], kyy=[[125,343],[343,1000]],
        # kxy=[[1,1],[27,64]], so 1+343-2*(93/4)=297.5.
        x, y = torch.tensor([[0.], [1.]]), torch.tensor([[2.], [3.]])
        self.assertAlmostEqual(polynomial_mmd(x, y), 297.5)
        self.assertAlmostEqual(polynomial_mmd(y, x), 297.5)

    def test_subset_sampling_reproducible_without_global_rng_changes(self):
        gen = torch.Generator().manual_seed(10)
        x, y = torch.randn(20, 8, generator=gen), torch.randn(24, 8, generator=gen)
        state = torch.get_rng_state().clone()
        first = kid(x, y, subsets=3, subset_size=16)
        self.assertEqual(first, kid(x, y, subsets=3, subset_size=16))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_nonfinite_features_rejected(self):
        with self.assertRaises(ValueError):
            polynomial_mmd(torch.full((4, 8), float("nan")), torch.zeros(4, 8))


if __name__ == "__main__":
    unittest.main()

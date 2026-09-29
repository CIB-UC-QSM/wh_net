import tempfile
import unittest
from pathlib import Path

import torch

from whnet.learned_regularizers import (
    FixedResidualProximal,
    extract_fixed_regularizer_model,
    set_regularizer_weights,
)
from whnet.model import ADMMUnrolledNet, ProximalNetwork


class LearnedRegularizerTests(unittest.TestCase):
    def test_extraction_removes_mlps_and_uses_fixed_weights(self):
        source = ADMMUnrolledNet(
            ProximalNetwork(features=[2], spectral_norm=False),
            ProximalNetwork(features=[2], spectral_norm=False),
            num_iters=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save(source.state_dict(), checkpoint)
            model = extract_fixed_regularizer_model(
                checkpoint,
                0.1,
                0.2,
                num_iters=2,
                features=[2],
                spectral_norm=False,
            )
        self.assertIsInstance(model.chinet, FixedResidualProximal)
        self.assertIsInstance(model.phinet, FixedResidualProximal)
        self.assertFalse(any("mlp" in name for name, _ in model.named_modules()))
        self.assertAlmostEqual(float(model.chinet.weight), 0.1)
        self.assertAlmostEqual(float(model.phinet.weight), 0.2)
        set_regularizer_weights(model, 0.3, 0.4)
        self.assertAlmostEqual(float(model.chinet.weight), 0.3)
        self.assertAlmostEqual(float(model.phinet.weight), 0.4)

        phase = torch.randn(1, 1, 8, 8, 8)
        mask = torch.ones_like(phase)
        dipole = torch.zeros_like(phase)
        with torch.inference_mode():
            chi, background = model(phase, mask, dipole, mask)
        self.assertEqual(chi.shape, phase.shape)
        self.assertEqual(background.shape, phase.shape)
        self.assertTrue(bool(torch.isfinite(chi).all()))
        self.assertTrue(bool(torch.isfinite(background).all()))

    def test_rejects_invalid_weights(self):
        source = ProximalNetwork(features=[2], spectral_norm=False)
        with self.assertRaises(ValueError):
            FixedResidualProximal(source, -1.0)


if __name__ == "__main__":
    unittest.main()

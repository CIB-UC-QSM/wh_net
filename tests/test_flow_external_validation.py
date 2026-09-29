import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import torch
from torch import nn

from whnet.adaptive_wh_tv_validation import ExternalValidationCase
from whnet.flow_external_validation import evaluate_flow_external_cases
from whnet.flow_admm import FLOW_DATA_TERM, FLOW_SOLVER


class IdentityEnsemble(nn.Module):
    def forward(self, phase, mask, dipole, weights=None, **kwargs):
        return phase * mask, phase * mask / 2, [{"iteration": 1, "data_rms": 0.0}]


class ExternalFlowEvaluationTests(unittest.TestCase):
    def test_writes_comparisons_physical_volumes_and_available_metrics(self):
        phase = torch.ones((1, 1, 8, 8, 8))
        mask = torch.ones_like(phase)
        dipole = torch.zeros_like(phase)
        cases = (
            ExternalValidationCase(
                "cosmos", phase, mask, dipole, phase, phase / 2, 0.2, 1
            ),
            ExternalValidationCase("sim2", phase, mask, dipole, phase, None, 0.3),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            results = evaluate_flow_external_cases(
                IdentityEnsemble(), cases, device="cpu", output_dir=output
            )
            self.assertEqual(results["cosmos"]["chi_nrmse"], 0.0)
            self.assertEqual(results["cosmos"]["background_nrmse"], 0.0)
            self.assertEqual(results["cosmos"]["data_term"], FLOW_DATA_TERM)
            self.assertEqual(results["cosmos"]["solver_type"], FLOW_SOLVER)
            self.assertEqual(
                results["cosmos"]["fidelity_weight_convention"], "residual_multiplier"
            )
            self.assertNotIn("background_nrmse", results["sim2"])
            self.assertTrue((output / "cosmos" / "chi_comparison.png").is_file())
            self.assertTrue((output / "cosmos" / "background_comparison.png").is_file())
            self.assertTrue(
                (output / "sim2" / "background_reconstruction.png").is_file()
            )
            volume = nib.load(output / "cosmos" / "chi_reconstruction.nii.gz")
            self.assertEqual(volume.shape, (6, 6, 6))
            self.assertAlmostEqual(float(volume.get_fdata().mean()), 0.2, places=6)
            self.assertEqual(
                json.loads((output / "metrics.json").read_text())["sim2"]["chi_nrmse"],
                0.0,
            )
            with self.assertRaises(FileExistsError):
                evaluate_flow_external_cases(
                    IdentityEnsemble(), cases, device="cpu", output_dir=output
                )


if __name__ == "__main__":
    unittest.main()

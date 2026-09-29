import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from whnet.adaptive_wh_tv import AdaptiveWHTV, WHTVParameterNet
from whnet.adaptive_wh_tv_validation import (
    ExternalValidationCase,
    _center_pad,
    evaluate_external_cases,
)
from whnet.adaptive_wh_tv_training import (
    AdaptiveWHTVTrainConfig,
    RunLogger,
    _gradient_norm,
)
from whnet.physics import continuous_dipole_kernel
from whnet.wh_tv import differentiable_wh_tv, wh_tv


class AdaptiveWHTVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(13)

    def inputs(self, shape=(8, 8, 8)):
        phase = torch.randn(1, 1, *shape) * 0.02
        mask = torch.ones_like(phase)
        dipole = torch.from_numpy(continuous_dipole_kernel(shape)).float()[None, None]
        return phase, mask, dipole

    def test_differentiable_forward_matches_reference_solver(self):
        phase, mask, dipole = self.inputs()
        weight = 0.25 + torch.rand_like(phase)
        parameters = {
            "alpha1": 1e-3,
            "beta": 1e6,
            "mu1": 0.1,
            "mu2": 1.0,
            "muh": 5.0,
        }
        expected = wh_tv(
            phase,
            dipole,
            mask,
            weight=weight,
            max_iterations=7,
            check_every=7,
            **parameters,
        )
        with torch.no_grad():
            actual = differentiable_wh_tv(
                phase,
                dipole,
                mask,
                fidelity_precision=weight.square(),
                max_iterations=7,
                gradient_tail=0,
                checkpoint_block=3,
                **parameters,
            )
        torch.testing.assert_close(actual.chi, expected.chi, rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(
            actual.background, expected.background, rtol=2e-5, atol=1e-7
        )

    def test_last_k_solver_has_finite_parameter_and_precision_gradients(self):
        phase, mask, dipole = self.inputs()
        precision = torch.ones_like(phase, requires_grad=True)
        values = [
            torch.tensor([value], requires_grad=True)
            for value in (1e-3, 1e6, 0.1, 1.0, 5.0)
        ]
        output = differentiable_wh_tv(
            phase,
            dipole,
            mask,
            fidelity_precision=precision,
            alpha1=values[0],
            beta=values[1],
            mu1=values[2],
            mu2=values[3],
            muh=values[4],
            max_iterations=6,
            gradient_tail=3,
            checkpoint_block=2,
        )
        (output.chi.square().mean() + output.background.square().mean()).backward()
        for gradient in [value.grad for value in values] + [precision.grad]:
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_predictor_bounds_variance_normalization_and_gradients(self):
        phase, mask, dipole = self.inputs()
        mask[:, :, :2] = 0
        model = AdaptiveWHTV(
            WHTVParameterNet((2, 4)),
            solver_iterations=5,
            gradient_tail=3,
            checkpoint_block=2,
        )
        output = model(phase * mask, mask, dipole)
        selected = mask.bool()
        torch.testing.assert_close(
            output.variance[selected], torch.ones_like(output.variance[selected])
        )
        torch.testing.assert_close(
            output.precision[selected], torch.ones_like(output.precision[selected])
        )
        self.assertAlmostEqual(
            float(output.variance[selected].log().mean().detach()), 0.0, places=5
        )
        self.assertTrue(bool((output.variance[~selected] == 0).all()))
        self.assertTrue(bool((output.precision[~selected] == 0).all()))
        parameters = output.parameters
        self.assertAlmostEqual(float(parameters.alpha1.detach()), 1e-3)
        self.assertAlmostEqual(float(parameters.beta.detach()), 1e8)
        self.assertAlmostEqual(float(parameters.mu1.detach()), 10**-0.5)
        self.assertAlmostEqual(float(parameters.mu2.detach()), 1.0)
        self.assertAlmostEqual(float(parameters.muh.detach()), 100.0)
        self.assertTrue(bool((parameters.alpha1 >= 1e-7).all()))
        self.assertTrue(bool((parameters.alpha1 <= 1e1).all()))
        self.assertTrue(bool((parameters.beta >= 1e6).all()))
        self.assertTrue(bool((parameters.beta <= 1e10).all()))
        ratio = parameters.mu1 / parameters.alpha1
        self.assertTrue(bool((ratio >= 1).all()))
        self.assertTrue(bool((ratio <= 1e5).all()))
        self.assertTrue(bool((parameters.mu2 >= 1e-3).all()))
        self.assertTrue(bool((parameters.mu2 <= 1e3).all()))
        self.assertTrue(bool((parameters.muh >= 1e-1).all()))
        self.assertTrue(bool((parameters.muh <= 1e5).all()))

        (output.chi.square().mean() + output.background.square().mean()).backward()
        variance_gradient = model.predictor.variance_head.weight.grad
        parameter_gradient = model.predictor.parameter_head[-1].weight.grad
        self.assertGreater(float(variance_gradient.abs().sum()), 0)
        self.assertGreater(float(parameter_gradient.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(variance_gradient).all()))
        self.assertTrue(bool(torch.isfinite(parameter_gradient).all()))

    def test_predictor_heads_remain_fp32_under_autocast(self):
        phase, mask, _ = self.inputs()
        predictor = WHTVParameterNet((2, 4))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            variance, precision, parameters = predictor(phase, mask)
        self.assertEqual(variance.dtype, torch.float32)
        self.assertEqual(precision.dtype, torch.float32)
        for value in parameters.as_dict().values():
            self.assertEqual(value.dtype, torch.float32)

    def test_external_evaluator_records_nrmse_and_comparison_images(self):
        phase, mask, dipole = self.inputs()
        model = AdaptiveWHTV(
            WHTVParameterNet((2,)),
            solver_iterations=3,
            gradient_tail=0,
            checkpoint_block=1,
        )
        case = ExternalValidationCase(
            name="tiny",
            phase=phase,
            mask=mask,
            dipole=dipole,
            chi=torch.randn_like(phase) * 0.01,
            background=torch.randn_like(phase) * 0.002,
            factor=0.2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = evaluate_external_cases(
                model,
                (case,),
                "cpu",
                bf16=False,
                output_dir=temporary,
            )
            self.assertIn("external_tiny_chi_nrmse", result.metrics)
            self.assertIn("external_tiny_background_nrmse", result.metrics)
            self.assertTrue((Path(temporary) / "tiny/chi_comparison.png").is_file())
            self.assertTrue(
                (Path(temporary) / "tiny/background_comparison.png").is_file()
            )
            import matplotlib.pyplot as plt

            for figure in result.figures.values():
                plt.close(figure)

    def test_sim2_center_padding_matches_legacy_offsets(self):
        volume = np.ones((2, 4, 3), dtype=np.float32)
        padded = _center_pad(volume, 2)
        self.assertEqual(padded.shape, (8, 8, 8))
        self.assertEqual(float(padded.sum()), float(volume.sum()))

    def test_gradient_norm_does_not_overflow_for_finite_fp32_gradients(self):
        parameter = torch.nn.Parameter(torch.zeros(4))
        parameter.grad = torch.full_like(parameter, 1e30)
        norm = _gradient_norm([parameter])
        self.assertTrue(bool(torch.isfinite(norm)))
        self.assertAlmostEqual(float(norm / 2e30), 1.0, places=6)

    def test_short_solver_requires_diagnostic_override(self):
        with self.assertRaises(ValueError):
            AdaptiveWHTVTrainConfig(solver_iterations=10).validate()
        AdaptiveWHTVTrainConfig(
            solver_iterations=10,
            gradient_tail=5,
            allow_short_solver=True,
        ).validate()

    def test_comet_logger_records_metrics_and_figure(self):
        events = {}

        class FakeExperiment:
            def __init__(self, **kwargs):
                events["init"] = kwargs

            def add_tag(self, tag):
                events["tag"] = tag

            def log_parameters(self, parameters):
                events["parameters"] = parameters

            def log_metrics(self, metrics, step=None):
                events["metrics"] = (metrics, step)

            def log_figure(self, **kwargs):
                events["figure"] = kwargs

            def end(self):
                events["ended"] = True

        original = sys.modules.get("comet_ml")
        sys.modules["comet_ml"] = types.SimpleNamespace(Experiment=FakeExperiment)
        try:
            import matplotlib.pyplot as plt

            with tempfile.TemporaryDirectory() as temporary:
                logger = RunLogger(
                    Path(temporary),
                    AdaptiveWHTVTrainConfig(comet=True),
                )
                logger.log({"event": "train", "step": 3, "loss": 0.5})
                figure = plt.figure()
                logger.log_figure(figure, epoch=1, step=3)
                logger.close()
                self.assertTrue(
                    (Path(temporary) / "images/validation_epoch_0001.png").is_file()
                )
        finally:
            if original is None:
                sys.modules.pop("comet_ml", None)
            else:
                sys.modules["comet_ml"] = original
        self.assertEqual(events["metrics"][1], 3)
        self.assertEqual(events["figure"]["step"], 3)
        self.assertTrue(events["ended"])

    def test_training_resume_matches_uninterrupted_run(self):
        from whnet.adaptive_wh_tv_training import (
            load_adaptive_wh_tv,
            run_adaptive_wh_tv_training,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            identities = {}
            generator = torch.Generator().manual_seed(21)
            for index in range(8):
                chi = torch.randn((1, 8, 8, 8), generator=generator) * 0.03
                mask = torch.zeros_like(chi, dtype=torch.bool)
                mask[:, 1:-1, 1:-1, 1:-1] = True
                torch.save(chi * mask, data / f"image_{index:04d}.pt")
                torch.save(mask, data / f"mask_{index:04d}.pt")
                identities[f"image_{index:04d}.pt"] = f"cohort:{index // 2}"
            manifest = root / "subjects.json"
            manifest.write_text(__import__("json").dumps(identities))
            config = AdaptiveWHTVTrainConfig(
                data_dir=str(data),
                cache_dir=str(root / "cache"),
                source_manifest=str(manifest),
                checkpoint_dir=str(root / "resumed"),
                epochs=1,
                batch_size=1,
                accumulation_steps=2,
                num_workers=0,
                solver_iterations=3,
                gradient_tail=2,
                checkpoint_block=1,
                features=(2,),
                device="cpu",
                allow_short_solver=True,
                max_train_volumes=4,
                max_val_subjects=1,
                val_views=1,
                max_steps=1,
                log_every=10,
                external_validation=False,
                save_every=10,
                channels_last_3d=False,
                wh_warmup_epochs=0,
                warmup_epochs=0,
            )
            run_adaptive_wh_tv_training(config)
            checkpoint = root / "resumed/training_last.pt"
            from dataclasses import replace

            run_adaptive_wh_tv_training(
                replace(config, resume=str(checkpoint), max_steps=0)
            )
            run_adaptive_wh_tv_training(
                replace(
                    config,
                    checkpoint_dir=str(root / "control"),
                    max_steps=0,
                )
            )
            resumed = torch.load(root / "resumed/model_last.pth", weights_only=True)
            control = torch.load(root / "control/model_last.pth", weights_only=True)
            for key in resumed:
                torch.testing.assert_close(
                    resumed[key], control[key], rtol=2e-5, atol=1e-7
                )
            loaded = load_adaptive_wh_tv(root / "resumed/model_best.pth")
            self.assertFalse(loaded.training)
            self.assertEqual(loaded.solver_iterations, 3)
            self.assertTrue(
                (root / "resumed/images/validation_epoch_0001.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()

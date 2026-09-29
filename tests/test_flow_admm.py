import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from whnet.flow_admm import (
    FLOW_DATA_TERM,
    FLOW_ENSEMBLE_FORMAT,
    FLOW_SOLVER,
    FlowADMMConfig,
    FlowADMMNet,
    FlowVelocity3D,
    flow_denoise,
    load_flow_admm,
    load_flow_run,
    mean_flow_prior,
    phase_scale,
    whnet_closed_form_physics,
)
from whnet.flow_training import (
    FlowRunLogger,
    FlowTrainConfig,
    evaluate_prior,
    flow_matching_loss,
    run_flow_training,
)
from whnet.flow_evaluation import evaluate_held_out, evaluate_whnet
from whnet.flow_tuning import evaluate_solver
from whnet.data import FieldSynthesizer
from whnet.model import EfficientADMMUnrolledNet
from whnet.physics import continuous_dipole_kernel


class ConstantVelocity(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, t, mask, condition=None):
        return self.value * mask


class FlowADMMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_flow_network_gradients_and_denoiser_endpoint(self):
        network = FlowVelocity3D(4)
        x = torch.randn(1, 1, 9, 10, 11)
        mask = torch.ones_like(x)
        target = network(x, torch.tensor([0.4]), mask)
        target.square().mean().backward()
        self.assertIsNotNone(network.out.weight.grad)
        torch.testing.assert_close(flow_denoise(network, x, 1.0, mask), x)
        mask[..., :2] = 0
        self.assertEqual(
            float(network(x, 0.4, mask)[..., :2].detach().abs().sum()), 0.0
        )
        conditional = FlowVelocity3D(4, condition_on_field=True)
        with self.assertRaisesRegex(ValueError, "condition"):
            conditional(x, 0.4, mask)
        self.assertEqual(conditional(x, 0.4, mask, condition=x).shape, x.shape)

    def test_background_flow_uses_total_field_source(self):
        mask = torch.ones(1, 1, 8, 8, 8)
        background = torch.zeros_like(mask)
        measured = torch.ones_like(mask)
        loss = flow_matching_loss(
            ConstantVelocity(0),
            background,
            mask,
            torch.tensor([0.25]),
            measured,
            measured,
        )
        self.assertAlmostEqual(float(loss), 1.0)

    def test_flow_logger_mirrors_metrics_to_comet(self):
        import matplotlib.pyplot as plt

        events = {"metrics": [], "images": [], "init": [], "ended": 0}

        class FakeExperiment:
            def __init__(self, **kwargs):
                events["init"].append(kwargs)

            def set_name(self, name):
                events["name"] = name

            def get_key(self):
                return "saved-experiment-key"

            def add_tag(self, tag):
                events["tag"] = tag

            def log_parameters(self, parameters):
                events["parameters"] = parameters

            def log_metrics(self, metrics, prefix=None, step=None, epoch=None):
                events["metrics"].append((metrics, prefix, step, epoch))

            def log_image(self, **kwargs):
                self.assert_image_exists(kwargs["image_data"])
                events["images"].append(kwargs)

            @staticmethod
            def assert_image_exists(path):
                if not Path(path).is_file():
                    raise AssertionError("image must exist before upload")

            def end(self):
                events["ended"] += 1

        fake_module = types.SimpleNamespace(
            Experiment=FakeExperiment, ExistingExperiment=FakeExperiment
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("sys.modules", {"comet_ml": fake_module}),
        ):
            path = Path(directory)
            with FlowRunLogger(
                path, FlowTrainConfig(branch="background", comet=True)
            ) as logger:
                logger.log({"event": "train", "epoch": 1, "step": 3, "loss": 0.5})
                logger.log(
                    {"event": "validation", "epoch": 1, "step": 3, "denoise_mse": 0.2}
                )
                logger.log_image(
                    plt.figure(),
                    name="validation_background_example00_t050",
                    epoch=1,
                    step=3,
                    metadata={"flow_time": 0.5},
                )
            with FlowRunLogger(
                path,
                FlowTrainConfig(
                    branch="background", comet=True, resume="training_last.pt"
                ),
            ) as logger:
                logger.log_image(
                    plt.figure(),
                    name="validation_background_example00_t050",
                    epoch=2,
                    step=6,
                    metadata={"flow_time": 0.5},
                )
            self.assertEqual(
                json.loads((path / "comet.json").read_text()),
                {"experiment_key": "saved-experiment-key"},
            )
            for epoch in (1, 2):
                folder = path / "images" / "validation_background_example00_t050"
                self.assertTrue((folder / f"epoch_{epoch:04d}.png").is_file())
                details = json.loads((folder / f"epoch_{epoch:04d}.json").read_text())
                self.assertEqual(details["epoch"], epoch)
                self.assertEqual(details["optimizer_step"], 3 * epoch)
            lines = [
                json.loads(line)
                for line in (path / "metrics.jsonl").read_text().splitlines()
            ]
        self.assertEqual(
            events["init"],
            [
                {"project_name": "wh-net-qsm", "auto_metric_logging": False},
                {
                    "experiment_key": "saved-experiment-key",
                    "auto_metric_logging": False,
                },
            ],
        )
        self.assertEqual(events["tag"], "flow-background")
        self.assertEqual(events["parameters"]["branch"], "background")
        self.assertEqual(
            events["metrics"],
            [
                ({"loss": 0.5}, "train", 3, 1),
                ({"denoise_mse": 0.2}, "validation", 3, 1),
            ],
        )
        self.assertEqual(
            [line["event"] for line in lines], ["train", "validation", "image", "image"]
        )
        self.assertEqual(events["ended"], 2)
        self.assertEqual([row["step"] for row in events["images"]], [1, 2])
        self.assertEqual(events["images"][0]["name"], events["images"][1]["name"])
        self.assertTrue(all(not row["overwrite"] for row in events["images"]))

    def test_validation_images_are_fixed_and_background_preview_uses_total_field(self):
        import matplotlib.pyplot as plt

        phase = torch.linspace(-0.1, 0.2, 512).reshape(1, 1, 8, 8, 8)
        mask = torch.ones_like(phase)
        chi, background = phase * 0.1, phase * 0.7
        batch = {"subject": ["qsm:fixed"], "seed": torch.tensor([12])}

        def synthesize(_):
            return phase, mask, mask, mask, chi, background

        inputs, images = [], []

        class Logger:
            def log_image(self, figure, **kwargs):
                images.append(kwargs)
                plt.close(figure)

        def figure(*values, title):
            inputs.append(tuple(value.clone() for value in values))
            return plt.figure()

        rng_before = torch.get_rng_state().clone()
        for branch in ("chi", "background"):
            with patch("whnet.flow_training._prior_figure", side_effect=figure):
                first = evaluate_prior(
                    ConstantVelocity(0),
                    [batch],
                    synthesize,
                    branch,
                    torch.device("cpu"),
                    logger=Logger(),
                    epoch=1,
                    step=2,
                    image_examples=1,
                )
                second = evaluate_prior(
                    ConstantVelocity(0.1),
                    [batch],
                    synthesize,
                    branch,
                    torch.device("cpu"),
                    logger=Logger(),
                    epoch=2,
                    step=4,
                    image_examples=1,
                )
            count = 4 if branch == "chi" else 5
            for index in range(count):
                earlier, later = inputs[index], inputs[index + count]
                for value in (0, 1, 2, 4):
                    torch.testing.assert_close(
                        earlier[value], later[value], rtol=0, atol=0
                    )
                self.assertGreater(float((earlier[3] - later[3]).abs().max()), 0)
                self.assertEqual(images[index]["name"], images[index + count]["name"])
                self.assertEqual(images[index]["epoch"], 1)
                self.assertEqual(images[index + count]["epoch"], 2)
            self.assertIn("denoise_nrmse_t020", first)
            self.assertIn("denoise_nrmse", second)
            if branch == "background":
                self.assertIn("total_field_separation_nrmse", first)
                self.assertAlmostEqual(
                    first["total_field_separation_nrmse"], 3 / 7, places=6
                )
                torch.testing.assert_close(inputs[4][0], phase[0])
                torch.testing.assert_close(inputs[4][1], phase[0])
                torch.testing.assert_close(inputs[4][3], phase[0])
                self.assertEqual(
                    images[4]["metadata"]["kind"], "total_field_separation"
                )
            inputs.clear()
            images.clear()
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)

    def test_monte_carlo_average_is_reproducible_and_uses_sample_count(self):
        x = torch.randn(1, 1, 8, 8, 8)
        mask = torch.ones_like(x)
        velocity = ConstantVelocity(0.25)
        first = mean_flow_prior(
            velocity, x, mask, 0.5, 3, torch.Generator().manual_seed(9)
        )
        second = mean_flow_prior(
            velocity, x, mask, 0.5, 3, torch.Generator().manual_seed(9)
        )
        different = mean_flow_prior(
            velocity, x, mask, 0.5, 1, torch.Generator().manual_seed(9)
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertGreater(float((first - different).abs().mean()), 0.01)

    def test_closed_form_physics_matches_dense_subproblems(self):
        shape = (4, 4, 5)
        count = 80
        kernel = torch.from_numpy(continuous_dipole_kernel(shape)).double()
        basis = torch.eye(count, dtype=torch.float64).reshape(count, *shape)
        operator = (
            torch.fft.ifftn(
                kernel * torch.fft.fftn(basis, dim=(-3, -2, -1)), dim=(-3, -2, -1)
            )
            .real.flatten(1)
            .T
        )
        values = [torch.randn(2, 1, *shape) for _ in range(7)]
        split, background, prior_chi, prior_b, dual_field, dual_chi, dual_b = values
        result, result_b, field = whnet_closed_form_physics(
            *values, torch.ones_like(split), kernel[None, None, ..., :3].float()
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            autocast_result = whnet_closed_form_physics(
                *values, torch.ones_like(split), kernel[None, None, ..., :3].float()
            )
        for expected_fp32, actual in zip((result, result_b, field), autocast_result):
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected_fp32, rtol=0, atol=0)
        normal = operator.T @ operator + torch.eye(count, dtype=torch.float64)
        rhs = operator.T @ (split - dual_field - background).flatten(1).double().T
        rhs += (prior_chi - dual_chi).flatten(1).double().T
        expected = torch.linalg.solve(normal, rhs).T
        expected_field = expected @ operator.T
        expected_b = 0.5 * (
            (split - dual_field + prior_b - dual_b).flatten(1).double() - expected_field
        )
        torch.testing.assert_close(
            result.flatten(1).double(), expected, atol=5e-7, rtol=2e-5
        )
        torch.testing.assert_close(
            field.flatten(1).double(), expected_field, atol=5e-7, rtol=2e-5
        )
        torch.testing.assert_close(
            result_b.flatten(1).double(), expected_b, atol=5e-7, rtol=2e-5
        )

    def test_two_priors_and_solver_are_seeded_and_supported(self):
        config = FlowADMMConfig(
            iterations=2,
            t_min=0.8,
            t_max=0.9,
            samples_early=2,
            samples_middle=2,
            samples_late=2,
        )
        solver = FlowADMMNet(ConstantVelocity(0.0), ConstantVelocity(0.2), config)
        phase = torch.randn(1, 1, 8, 8, 8) * 0.1
        mask = torch.zeros_like(phase)
        mask[..., 1:-1, 1:-1, 1:-1] = 1
        dipole = torch.zeros(1, 1, 8, 8, 5)
        first = solver(phase * mask, mask, dipole, seed=12, return_diagnostics=True)
        second = solver(phase * mask, mask, dipole, seed=12, return_diagnostics=True)
        for a, b in zip(first[:2], second[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(float((a * (1 - mask)).abs().sum()), 0.0)
            self.assertTrue(bool(torch.isfinite(a).all()))
        self.assertEqual(len(first[2]), 2)
        self.assertEqual(first[2][-1]["chi_samples"], 2)
        self.assertEqual(first[2][-1]["background_evaluations"], 1)
        self.assertGreater(float((first[0] - first[1]).abs().mean()), 1e-4)

    def test_one_step_data_prox_and_prior_draws(self):
        config = FlowADMMConfig(
            iterations=1,
            tau=2.0,
            t_min=0.5,
            t_max=0.5,
            samples_early=1,
            samples_middle=1,
            samples_late=1,
        )
        solver = FlowADMMNet(ConstantVelocity(0), ConstantVelocity(0), config)
        phase = torch.randn(1, 1, 8, 8, 8) * 0.1
        mask = torch.ones_like(phase)
        dipole = torch.zeros(1, 1, 8, 8, 5)
        chi, background = solver(phase, mask, dipole, seed=4)
        generator = torch.Generator().manual_seed(4)
        first_noise = torch.randn(phase.shape, generator=generator)
        scale = phase_scale(phase, mask)
        torch.testing.assert_close(chi, 0.5 * first_noise * scale, rtol=2e-5, atol=2e-6)
        # s = 2/3 y, prior_b = 1/2 y, and the physical b = (s + prior_b)/2.
        torch.testing.assert_close(background, (7 / 12) * phase, rtol=2e-5, atol=2e-6)

    def test_split_solver_matches_original_whnet_with_squared_weights(self):
        # WH-Net accepts precision weights. Flow accepts residual multipliers,
        # so the reference must receive tau * W^2, not W or tau * W.
        shape = (4, 5, 5)
        mask = torch.ones(1, 1, *shape)
        mask[..., 0, :2, :] = 0
        phase = torch.linspace(-0.1, 0.2, mask.numel()).reshape_as(mask) * mask
        weights = torch.linspace(0.0, 2.0, mask.numel()).reshape_as(mask) * mask
        dipole = torch.from_numpy(continuous_dipole_kernel(shape))[None, None].float()
        tau = 0.8

        class ProjectedIdentity(nn.Module):
            def forward(self, value, *args):
                return value * mask

        original = EfficientADMMUnrolledNet(
            ProjectedIdentity(),
            ProjectedIdentity(),
            num_iters=5,
            eps=0,
            normalize_input=False,
            checkpoint_steps=False,
        ).eval()
        with torch.no_grad():
            expected_chi, expected_b = original(
                phase, mask, dipole, tau * weights.square()
            )
        solver = FlowADMMNet(
            ConstantVelocity(0),
            ConstantVelocity(0),
            FlowADMMConfig(iterations=5, tau=tau),
        )
        with (
            patch("whnet.flow_admm.mean_flow_prior", side_effect=lambda _, x, *args: x),
            patch(
                "whnet.flow_admm.conditional_field_prior",
                side_effect=lambda _, x, *args: x,
            ),
        ):
            chi, background, trace = solver(
                phase,
                mask,
                dipole,
                weights,
                seed=0,
                return_diagnostics=True,
            )
        torch.testing.assert_close(chi, expected_chi * mask, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(background, expected_b * mask, atol=2e-7, rtol=2e-5)
        residual = (
            torch.fft.ifftn(
                dipole * torch.fft.fftn(chi, dim=(-3, -2, -1)), dim=(-3, -2, -1)
            ).real
            + background
            - phase
        ) * mask
        scale = float(phase_scale(phase, mask))
        self.assertAlmostEqual(
            trace[-1]["data_rms"],
            float((residual.square().sum() / mask.sum()).sqrt()) / scale,
            places=6,
        )
        self.assertAlmostEqual(
            trace[-1]["weighted_data_rms"],
            float(((weights * residual).square().sum() / mask.sum()).sqrt()) / scale,
            places=6,
        )
        self.assertEqual(float((chi * (1 - mask)).abs().sum()), 0.0)
        self.assertEqual(float((background * (1 - mask)).abs().sum()), 0.0)
        self.assertFalse(any(key.startswith("cg_") for row in trace for key in row))
        self.assertIn("data_split_rms", trace[-1])

    def test_data_step_accepts_masks_without_a_harmonic_stencil(self):
        mask = torch.zeros(1, 1, 8, 8, 8)
        mask[..., 3, 3, 3] = 1
        solver = FlowADMMNet(
            ConstantVelocity(0), ConstantVelocity(0), FlowADMMConfig(iterations=1)
        )
        chi, background, trace = solver(
            mask * 0.1,
            mask,
            torch.zeros(1, 1, 8, 8, 5),
            seed=4,
            return_diagnostics=True,
        )
        self.assertTrue(
            bool(torch.isfinite(chi).all() and torch.isfinite(background).all())
        )
        self.assertEqual(trace[-1]["harmonic_rms"], 0.0)
        with self.assertRaisesRegex(ValueError, "at least one voxel"):
            solver(mask, torch.zeros_like(mask), torch.zeros(1, 1, 8, 8, 5))
        with self.assertRaisesRegex(ValueError, "squared fidelity weights"):
            solver(mask, mask, torch.zeros(1, 1, 8, 8, 5), weights=mask * 1e30)

    def test_baseline_comparison_reports_weighted_residual_rms(self):
        phase = torch.ones(1, 1, 8, 8, 8)
        mask = torch.ones_like(phase)
        weights = torch.linspace(0, 2, phase.numel()).reshape_as(phase)
        dipole = torch.zeros(1, 1, 8, 8, 5)
        scores = evaluate_whnet(
            lambda *args: (torch.zeros_like(phase), torch.zeros_like(phase)),
            [{"subject": ["subject"]}],
            lambda _: (phase, mask, weights, dipole, phase, phase),
        )
        self.assertEqual(scores["data_rms"], 1.0)
        self.assertAlmostEqual(
            scores["weighted_data_rms"],
            float(weights.square().mean().sqrt()),
            places=6,
        )

    def test_checkpoint_branch_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            chi = Path(directory) / "chi.pt"
            background = Path(directory) / "background.pt"
            for branch, path in (("chi", chi), ("background", background)):
                network = FlowVelocity3D(4, condition_on_field=branch == "background")
                torch.save(
                    {
                        "format": "whnet-flow-v2",
                        "branch": branch,
                        "source": "gaussian"
                        if branch == "chi"
                        else "measured_total_field",
                        "base_channels": 4,
                        "model": network.state_dict(),
                        "normalization": "masked_phase_rms_clamp_1e-3",
                    },
                    path,
                )
            self.assertIsInstance(load_flow_admm(chi, background), FlowADMMNet)
            with self.assertRaises(ValueError):
                load_flow_admm(background, chi)
            manifest = Path(directory) / "ensemble.json"
            metadata = {
                "format": FLOW_ENSEMBLE_FORMAT,
                "data_term": FLOW_DATA_TERM,
                "solver_type": FLOW_SOLVER,
                "chi_checkpoint": str(chi),
                "background_checkpoint": str(background),
                "solver": {"iterations": 2, "tau": 0.5},
            }
            manifest.write_text(json.dumps(metadata))
            restored = load_flow_run(manifest)
            self.assertEqual(restored.config, FlowADMMConfig(iterations=2, tau=0.5))
            metadata["data_term"] = "precision_weighted_residual"
            manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "invalid FlowADMM ensemble"):
                load_flow_run(manifest)
            metadata["format"] = "whnet-flow-ensemble-v2"
            manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "rerun whnet-flow-tune"):
                load_flow_run(manifest)
            metadata["format"] = "whnet-flow-ensemble-v3"
            manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "closed-form solver"):
                load_flow_run(manifest)

    def test_bounded_training_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            mapping = {}
            for index in range(3):
                image = f"image_{index:04d}.pt"
                torch.save(torch.randn(1, 8, 8, 8) * 0.01, data / image)
                torch.save(torch.ones(1, 8, 8, 8), data / f"mask_{index:04d}.pt")
                mapping[image] = f"qsm:{index:04d}"
            manifest = root / "subjects.json"
            manifest.write_text(json.dumps(mapping))
            run = root / "run"
            settings = FlowTrainConfig(
                branch="chi",
                data_dir=str(data),
                cache_dir=str(root / "cache"),
                source_manifest=str(manifest),
                run_dir=str(run),
                epochs=2,
                accumulation_steps=1,
                num_workers=0,
                base_channels=4,
                val_views=1,
                device="cpu",
                max_steps=1,
            )
            run_flow_training(settings)
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"], "paused"
            )
            settings.resume = str(run / "training_last.pt")
            settings.max_steps = 0
            run_flow_training(settings)
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"], "complete"
            )
            self.assertEqual(
                torch.load(run / "model_best.pt", weights_only=True)["source"],
                "gaussian",
            )
            uninterrupted = root / "uninterrupted"
            baseline = FlowTrainConfig(
                branch="chi",
                data_dir=str(data),
                cache_dir=str(root / "cache"),
                source_manifest=str(manifest),
                run_dir=str(uninterrupted),
                epochs=2,
                accumulation_steps=1,
                num_workers=0,
                base_channels=4,
                val_views=1,
                device="cpu",
            )
            run_flow_training(baseline)
            for epoch in (1, 2):
                image = (
                    run
                    / "images"
                    / "validation_chi_example00_t050"
                    / f"epoch_{epoch:04d}.png"
                )
                self.assertTrue(image.is_file())
                self.assertEqual(
                    json.loads(image.with_suffix(".json").read_text())["epoch"], epoch
                )
            resumed_state = torch.load(run / "model_best.pt", weights_only=True)[
                "model"
            ]
            baseline_state = torch.load(
                uninterrupted / "model_best.pt", weights_only=True
            )["model"]
            for name in baseline_state:
                torch.testing.assert_close(
                    resumed_state[name], baseline_state[name], rtol=0, atol=0
                )
            background_run = root / "background_run"
            run_flow_training(
                FlowTrainConfig(
                    branch="background",
                    data_dir=str(data),
                    cache_dir=str(root / "cache"),
                    source_manifest=str(manifest),
                    run_dir=str(background_run),
                    epochs=2,
                    accumulation_steps=1,
                    num_workers=0,
                    base_channels=4,
                    val_views=1,
                    device="cpu",
                )
            )
            self.assertEqual(
                torch.load(background_run / "model_best.pt", weights_only=True)[
                    "source"
                ],
                "measured_total_field",
            )
            for epoch in (1, 2):
                self.assertTrue(
                    (
                        background_run
                        / "images"
                        / "validation_background_example00_total_field"
                        / f"epoch_{epoch:04d}.png"
                    ).is_file()
                )
            combined = load_flow_admm(
                run / "model_best.pt",
                background_run / "model_best.pt",
                config=FlowADMMConfig(
                    iterations=1, samples_early=1, samples_middle=1, samples_late=1
                ),
            )
            with torch.no_grad():
                chi, background = combined(
                    torch.randn(1, 1, 8, 8, 8) * 0.1,
                    torch.ones(1, 1, 8, 8, 8),
                    torch.zeros(1, 1, 8, 8, 5),
                    seed=3,
                )
            self.assertTrue(bool(torch.isfinite(chi).all()))
            self.assertTrue(bool(torch.isfinite(background).all()))

    def test_held_out_evaluation_rejects_seen_subject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            image = data / "image_0000.pt"
            torch.save(torch.zeros(1, 8, 8, 8), image)
            torch.save(torch.ones(1, 8, 8, 8), data / "mask_0000.pt")
            subject_manifest = root / "subjects.json"
            subject_manifest.write_text(json.dumps({image.name: "qsm:0000"}))
            split = {"train": [{"subject": "qsm:0000"}], "validation": []}
            prior_dir = root / "prior"
            prior_dir.mkdir()
            (prior_dir / "split.json").write_text(json.dumps(split))
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "split.json").write_text(json.dumps(split))
            ensemble = root / "ensemble.json"
            ensemble.write_text(
                json.dumps(
                    {
                        "format": FLOW_ENSEMBLE_FORMAT,
                        "data_term": FLOW_DATA_TERM,
                        "solver_type": FLOW_SOLVER,
                        "chi_checkpoint": str(prior_dir / "chi.pt"),
                        "background_checkpoint": str(prior_dir / "background.pt"),
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate_held_out(
                    ensemble,
                    baseline_dir / "model_best.pth",
                    data_dir=data,
                    source_manifest=subject_manifest,
                    output=root / "report.json",
                    device="cpu",
                )

    def test_validation_scores_coupled_solver(self):
        solver = FlowADMMNet(
            ConstantVelocity(0),
            ConstantVelocity(0),
            FlowADMMConfig(
                iterations=1, samples_early=1, samples_middle=1, samples_late=1
            ),
        )
        batch = {
            "chi": torch.randn(1, 1, 8, 8, 8) * 0.01,
            "mask": torch.ones(1, 1, 8, 8, 8, dtype=torch.bool),
            "dilated": torch.ones(1, 1, 8, 8, 8, dtype=torch.bool),
            "seed": torch.tensor([7]),
            "subject": ["qsm:1"],
        }
        scores = evaluate_solver(
            solver, [batch], FieldSynthesizer((8, 8, 8), torch.device("cpu")), seed=2
        )
        self.assertEqual(
            set(scores),
            {
                "chi_nrmse",
                "background_nrmse",
                "data_rms",
                "weighted_data_rms",
                "harmonic_rms",
            },
        )
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in scores.values())
        )


if __name__ == "__main__":
    unittest.main()

import copy
import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from whnet.adaptive_wh_tv_validation import (
    ExternalValidationCase,
    physical_external_cases,
)
from whnet.data import (
    TRAINING_CONDITIONS,
    FieldSynthesizer,
    TrainingVolumes,
    _sample_local_perturbations,
    sample_target_chi_rms,
    separable_dilation,
)
from whnet.training import (
    BalancedSourceSampler,
    EMA,
    RunLogger,
    TrainConfig,
    _configure_comet_api_key,
    deep_supervised_loss,
    evaluate_whnet_external_cases,
    learning_rate_multiplier,
    load_trained_model,
    main as training_main,
    make_subject_split,
    run_training,
    supervision_indices,
)
from whnet.physics import continuous_dipole_kernel
from whnet.model import EfficientADMMUnrolledNet, ProximalNetwork


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def inputs(self, shape=(12, 12, 12)):
        phase = torch.randn(1, 1, *shape) * 0.1
        mask = torch.zeros_like(phase)
        mask[:, :, 2:-2, 2:-2, 2:-2] = 1
        dipole = torch.from_numpy(continuous_dipole_kernel(shape)).float()[None, None]
        return phase * mask, mask, dipole

    def model(self, checkpoint=True):
        return EfficientADMMUnrolledNet(
            ProximalNetwork(features=[2, 4], normalized_conditioning=True),
            ProximalNetwork(features=[2, 4], normalized_conditioning=True),
            num_iters=3,
            checkpoint_steps=checkpoint,
        )

    def test_checkpoint_gradients_and_both_branches(self):
        a = self.model()
        a.refresh_spectral_norm()
        b = copy.deepcopy(a)
        b.checkpoint_steps = False
        inputs = self.inputs()
        buffers = {k: v.clone() for k, v in a.named_buffers()}
        gradients = []
        for model in (a, b):
            chi, phi = model(*inputs)
            (chi.square().mean() + phi.square().mean()).backward()
            for branch in (model.chinet, model.phinet):
                self.assertTrue(
                    any(
                        p.grad is not None and bool(p.grad.abs().sum() > 0)
                        for p in branch.parameters()
                    )
                )
            gradients.append(
                torch.cat(
                    [p.grad.flatten() for p in model.parameters() if p.grad is not None]
                )
            )
        torch.testing.assert_close(*gradients, rtol=2e-5, atol=1e-7)
        for key, value in a.named_buffers():
            torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
        self.assertTrue(bool(gradients[0].isfinite().all()))

    def test_projected_physics_matches_full_fft_reference(self):
        model = self.model(checkpoint=False).eval()
        phase, mask, full_d = self.inputs()
        shape = phase.shape[-3:]
        tensors = [torch.randn_like(phase) * 0.1 for _ in range(5)]
        chi, phi_h, ey, eu, ev = tensors
        count = mask.sum(dim=(-3, -2, -1))
        update = torch.full_like(count, 0.01)
        d = full_d[..., : shape[-1] // 2 + 1]

        def fft(x):
            return torch.fft.fftn(x, dim=(-3, -2, -1))

        def inv(x):
            return torch.fft.ifftn(x, dim=(-3, -2, -1)).real

        with torch.no_grad():
            actual = model.efficient_step(
                *tensors,
                phase,
                mask,
                mask,
                d,
                d.square() + 1 + model.eps,
                count,
                torch.tensor(2.0),
                update,
            )
            total = inv(full_d * fft(chi)) + phi_h
            y = (mask * phase + total + ey) / (mask + 1)
            u = model.chinet(chi + eu, 2, update)
            v = model.phinet(phi_h + ev, 2, model._rms(total - phase, mask, count))
            updated_chi = (
                inv(
                    (full_d * fft(y - ey - phi_h) + fft(u - eu))
                    / (full_d.square() + 1 + model.eps)
                )
                * mask
            )
            field = inv(full_d * fft(updated_chi))
            updated_phi = (y - ey - field + v - ev) / 2
            expected = (
                updated_chi,
                updated_phi,
                ey + field + updated_phi - y,
                eu + updated_chi - u,
                ev + updated_phi - v,
            )
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)

    def test_zero_input_has_finite_gradients(self):
        model = self.model()
        phase, mask, d = self.inputs()
        outputs = model(phase * 0, mask, d)
        sum(x.square().mean() for x in outputs).backward()
        self.assertTrue(
            all(
                bool(p.grad.isfinite().all())
                for p in model.parameters()
                if p.grad is not None
            )
        )

    def test_global_phase_scaling_is_equivariant_near_old_floor(self):
        model = self.model(checkpoint=False).eval()
        model.input_scale_floor = 1e-6
        phase, mask, dipole = self.inputs()
        phase = phase * 0.001
        with torch.no_grad():
            base = model(phase, mask, dipole)
            for factor in (0.2125, 0.264, 0.5, 2.0):
                scaled = model(phase * factor, mask, dipole)
                for reference, actual in zip(base, scaled):
                    torch.testing.assert_close(
                        actual, reference * factor, rtol=2e-3, atol=1e-6
                    )

    def test_group_split_keeps_all_augmentations_together(self):
        records = [
            {"subject": f"{cohort}:{subject}", "image": f"{cohort}_{subject}_{view}"}
            for cohort in ("a", "b")
            for subject in range(10)
            for view in range(3)
        ]
        train, val = make_subject_split(records, 0.2, 0)
        self.assertEqual(len(val), 12)
        self.assertFalse({r["subject"] for r in train} & {r["subject"] for r in val})
        self.assertEqual((train, val), make_subject_split(records, 0.2, 0))

    def test_masked_loss_is_padding_invariant_and_supervises_phi(self):
        phase, mask, _ = self.inputs()
        target = phase
        pred = (target + 0.02 * mask).requires_grad_()
        phi_pred = (target + 0.01 * mask).requires_grad_()
        cfg = TrainConfig(lam_chi_high=0)
        loss, parts = deep_supervised_loss(
            [(pred, phi_pred)], target, target, mask, 0, cfg
        )
        padded, _ = deep_supervised_loss(
            [(F.pad(pred, (2,) * 6), F.pad(phi_pred, (2,) * 6))],
            F.pad(target, (2,) * 6),
            F.pad(target, (2,) * 6),
            F.pad(mask, (2,) * 6),
            0,
            cfg,
        )
        torch.testing.assert_close(loss, padded)
        self.assertGreater(float(parts["loss_phi"]), 0)
        loss.backward()
        self.assertGreater(float(phi_pred.grad.abs().sum()), 0)
        perfect, _ = deep_supervised_loss(
            [(target, target)], target, target, mask, 0, cfg
        )
        self.assertEqual(float(perfect), 0)

    def test_high_susceptibility_l1_uses_per_sample_quantile_and_strict_threshold(self):
        increasing = torch.arange(27, dtype=torch.float32).reshape(1, 1, 3, 3, 3)
        target = torch.cat(
            (increasing, -2 * increasing, torch.full_like(increasing, 2)), dim=0
        )
        prediction = (target + 1).detach().requires_grad_()
        selected = torch.zeros_like(target)
        selected[0, 0].flatten()[21:] = 1
        selected[1, 0].flatten()[21:] = 1
        config = types.SimpleNamespace(
            lam_chi=0.0,
            lam_chi_high=1.0,
            lam_phi=0.0,
            lam_grad=0.0,
            final_weight=0.7,
        )
        loss, parts = deep_supervised_loss(
            [(prediction, torch.zeros_like(target))],
            target,
            torch.zeros_like(target),
            torch.ones_like(target),
            0,
            config,
        )
        expected = selected.sum() / target.numel()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(parts["loss_chi_high"], expected)
        loss.backward()
        torch.testing.assert_close(prediction.grad, selected / target.numel())
        with self.assertRaisesRegex(ValueError, "loss weights"):
            TrainConfig(lam_chi_high=-1).validate()

    def test_normalized_high_chi_loss_is_scale_invariant_per_sample(self):
        target = torch.arange(1, 65, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
        target = torch.cat((target, target * 0.1))
        prediction = target * 1.2
        cfg = TrainConfig(lam_chi=1, lam_chi_high=3, lam_phi=0.25, lam_grad=0)
        mask = torch.ones_like(target)
        zeros = torch.zeros_like(target)
        first, _ = deep_supervised_loss(
            [(prediction, zeros)], target, zeros, mask, 0, cfg
        )
        scaled = target * torch.tensor([0.3, 4.0]).reshape(2, 1, 1, 1, 1)
        second, _ = deep_supervised_loss(
            [(scaled * 1.2, zeros)], scaled, zeros, mask, 0, cfg
        )
        torch.testing.assert_close(first, second, rtol=2e-6, atol=1e-6)

    def test_balanced_sampler_counts_and_repeated_views(self):
        records = [{"subject": f"qsm:{index}"} for index in range(4)] + [
            {"subject": f"synthetic:{index}"} for index in range(8)
        ]
        sampler = BalancedSourceSampler(records, 0.25, seed=3)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        self.assertEqual(len(first), 12)
        self.assertEqual(sum(index % 12 >= 4 for index in first), 3)
        self.assertEqual(len(set(first)), 12)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        seen_synthetic = set()
        for epoch in range(3):
            sampler.set_epoch(epoch)
            seen_synthetic.update(index % 12 for index in sampler if index % 12 >= 4)
        self.assertEqual(seen_synthetic, set(range(4, 12)))

    def test_legacy_resume_keeps_old_loss_unless_high_chi_weight_is_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "training_last.pt"
            old_config = asdict(TrainConfig())
            del old_config["lam_chi_high"]
            del old_config["normalize_high_chi_loss"]
            del old_config["input_scale_floor"]
            del old_config["local_perturbations"]
            del old_config["chi_scale_mode"]
            del old_config["background_mode"]
            del old_config["synthetic_fraction"]
            del old_config["external_validation"]
            del old_config["external_validation_every"]
            del old_config["cosmos_simulation"]
            del old_config["sim2_factor"]
            del old_config["sim2_pad"]
            torch.save({"config": old_config}, checkpoint)
            cases = (
                ([], 0.0, False, False),
                (["--lam-chi-high", "1.25"], 1.25, False, False),
                (["--local-perturbations"], 0.0, True, False),
                (["--external-validation"], 0.0, False, True),
            )
            for (
                extra_args,
                expected_loss,
                expected_perturbations,
                expected_external,
            ) in cases:
                with (
                    patch.object(
                        sys,
                        "argv",
                        ["whnet-train", "--resume", str(checkpoint), *extra_args],
                    ),
                    patch("whnet.training.run_training") as run,
                ):
                    training_main()
                    config = run.call_args.args[0]
                    self.assertEqual(config.lam_chi_high, expected_loss)
                    self.assertEqual(config.local_perturbations, expected_perturbations)
                    self.assertEqual(config.external_validation, expected_external)
                    self.assertFalse(config.normalize_high_chi_loss)
                    self.assertEqual(config.input_scale_floor, 1e-3)
                    self.assertEqual(config.chi_scale_mode, "legacy")
                    self.assertEqual(config.background_mode, "legacy")
                    self.assertEqual(config.synthetic_fraction, -1)

    def test_whnet_external_evaluation_uses_sim2_weights_and_cosmos_background(self):
        shape = (1, 1, 4, 4, 4)
        mask = torch.ones(shape)
        sim2_weight = torch.full(shape, 0.25)

        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, phase, mask, dipole, weight, *, max_iters):
                self.calls.append(
                    (phase.clone(), weight.clone(), max_iters, self.training)
                )
                return phase, phase

        model = RecordingModel().train()
        cases = (
            ExternalValidationCase(
                name="cosmos",
                phase=torch.ones(shape),
                mask=mask,
                dipole=torch.ones(shape),
                chi=torch.ones(shape),
                background=torch.ones(shape),
                factor=0.2125,
            ),
            ExternalValidationCase(
                name="sim2",
                phase=torch.full(shape, 2.0),
                mask=mask,
                dipole=torch.ones(shape),
                chi=torch.ones(shape),
                background=None,
                factor=0.264,
                weight=sim2_weight,
            ),
        )
        physical_cases = physical_external_cases(cases)
        self.assertTrue(all(case.factor == 1.0 for case in physical_cases))
        torch.testing.assert_close(
            physical_cases[0].background, torch.full(shape, 0.2125)
        )
        result = evaluate_whnet_external_cases(
            model, physical_cases, torch.device("cpu"), 55
        )
        self.assertEqual(
            result.metrics,
            {
                "external_cosmos_chi_nrmse": 0.0,
                "external_cosmos_background_nrmse": 0.0,
                "external_sim2_chi_nrmse": 1.0,
            },
        )
        torch.testing.assert_close(model.calls[0][0], torch.full(shape, 0.2125))
        torch.testing.assert_close(model.calls[1][0], torch.full(shape, 0.528))
        torch.testing.assert_close(model.calls[0][1], mask)
        torch.testing.assert_close(model.calls[1][1], sim2_weight)
        self.assertTrue(
            all(iters == 55 and not mode for _, _, iters, mode in model.calls)
        )
        self.assertTrue(model.training)
        self.assertEqual(
            set(result.figures),
            {
                "cosmos_chi_comparison",
                "cosmos_background_comparison",
                "sim2_chi_comparison",
            },
        )
        for name, figure in result.figures.items():
            self.assertEqual(len(figure.axes), 9)
            self.assertTrue(all(len(axis.images) == 1 for axis in figure.axes))
            if "chi" in name:
                self.assertTrue(
                    all(
                        axis.images[0].get_clim() == (-0.1, 0.1) for axis in figure.axes
                    )
                )
            else:
                self.assertAlmostEqual(
                    figure.axes[0].images[0].get_clim()[1], 0.2125, places=6
                )
            self.assertIn("Ground truth", figure.axes[0].texts[0].get_text())
            self.assertIn("Difference", figure.axes[6].texts[0].get_text())
        sim2_chi = result.figures["sim2_chi_comparison"]
        self.assertAlmostEqual(
            float(sim2_chi.axes[0].images[0].get_array().mean()), 0.264, places=6
        )
        self.assertAlmostEqual(
            float(sim2_chi.axes[3].images[0].get_array().mean()), 0.528, places=6
        )
        self.assertAlmostEqual(
            float(sim2_chi.axes[6].images[0].get_array().mean()), 0.264, places=6
        )
        import matplotlib.pyplot as plt

        for figure in result.figures.values():
            plt.close(figure)

    def test_dilation_and_deterministic_synthesis(self):
        phase, mask, _ = self.inputs()
        expected = F.max_pool3d(mask, 7, 1, 3).bool()
        torch.testing.assert_close(separable_dilation(mask[0]), expected[0])
        synth = FieldSynthesizer(mask.shape[-3:], "cpu")
        batch = {
            "chi": phase.repeat(3, 1, 1, 1, 1),
            "mask": mask.bool().repeat(3, 1, 1, 1, 1),
            "dilated": expected.repeat(3, 1, 1, 1, 1),
            "seed": torch.tensor([4, 8, 11]),
        }
        together = synth(batch)
        for index in range(3):
            alone = synth(
                {key: value[index : index + 1] for key, value in batch.items()}
            )
            for slot in (0, 1, 2, 4, 5):
                torch.testing.assert_close(
                    together[slot][index : index + 1], alone[slot], rtol=2e-5, atol=1e-6
                )

    def test_ratio_synthesis_hits_requested_field_range(self):
        phase, mask, _ = self.inputs(shape=(24, 24, 24))
        mask.zero_()
        mask[:, :, 6:-6, 6:-6, 6:-6] = 1
        phase = phase * mask
        synth = FieldSynthesizer(mask.shape[-3:], "cpu")
        batch = {
            "chi": phase,
            "mask": mask.bool(),
            "dilated": F.max_pool3d(mask, 7, 1, 3).bool(),
            "seed": torch.tensor([4]),
        }
        *_, diagnostics = synth(batch, background_mode="ratio", return_diagnostics=True)
        ratio = float(diagnostics["local_to_background"][0])
        self.assertGreaterEqual(ratio, 0.03 - 1e-5)
        self.assertLessEqual(ratio, 3.0 + 1e-5)

    def test_sparse_local_perturbation_rate_and_masked_neighbors(self):
        mask = torch.zeros(7, 7, 7, dtype=torch.bool)
        mask[1:6, 1:6, 1:6] = True
        affected = paired = 0
        for seed in range(2000):
            generator = torch.Generator().manual_seed(seed + 2000003)
            perturbations = _sample_local_perturbations(mask, generator)
            if not perturbations:
                continue
            affected += 1
            positive = [value for _, value in perturbations if value > 0]
            negative = [value for _, value in perturbations if value < 0]
            self.assertIn(len(positive), (1, 2, 3))
            self.assertTrue(all(10 <= value <= 100 for value in positive))
            self.assertTrue(all(bool(mask[position]) for position, _ in perturbations))
            if negative:
                paired += 1
                self.assertEqual(len(positive), len(negative))
                for index in range(0, len(perturbations), 2):
                    position, value = perturbations[index]
                    neighbor, opposite = perturbations[index + 1]
                    self.assertEqual(opposite, -value)
                    self.assertEqual(
                        sum(abs(a - b) for a, b in zip(position, neighbor)), 1
                    )
        self.assertLess(abs(affected / 2000 - 0.1), 0.03)
        self.assertLess(abs(paired / affected - 0.5), 0.1)

    def test_sparse_local_perturbations_change_phase_only_before_background(self):
        phase, mask, _ = self.inputs()
        synthesizer = FieldSynthesizer(mask.shape[-3:], "cpu")
        batch = {
            "chi": phase,
            "mask": mask.bool(),
            "dilated": F.max_pool3d(mask, 7, 1, 3).bool(),
            "seed": torch.tensor([4]),
        }
        clean = synthesizer(batch)
        with patch.dict(
            TRAINING_CONDITIONS,
            {
                "local_perturbation_probability": 1.0,
                "local_perturbation_count_min": 1,
                "local_perturbation_count_max": 1,
                "local_perturbation_amplitude_min": 10.0,
                "local_perturbation_amplitude_max": 10.0,
                "local_perturbation_neighbor_probability": 1.0,
            },
        ):
            perturbations = _sample_local_perturbations(
                batch["mask"][0, 0], torch.Generator().manual_seed(4 + 2000003)
            )
            augmented = synthesizer(batch, perturb_local_field=True)
            torch.testing.assert_close(synthesizer(batch)[0], clean[0])
        for slot in (1, 2, 3, 4, 5):
            torch.testing.assert_close(augmented[slot], clean[slot])
        delta = augmented[0] - clean[0]
        self.assertEqual(len(perturbations), 2)
        positions = [position for position, _ in perturbations]
        self.assertEqual(sum(abs(a - b) for a, b in zip(*positions)), 1)
        local = (
            torch.fft.irfftn(
                synthesizer.dipole * torch.fft.rfftn(phase, dim=(-3, -2, -1)),
                s=mask.shape[-3:],
                dim=(-3, -2, -1),
            )
            * mask
        )
        expected = torch.zeros_like(delta)
        for position, amplitude in perturbations:
            expected[(0, 0, *position)] = local[(0, 0, *position)] * (amplitude - 1)
        torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(int((delta * (1 - mask)).abs().count_nonzero()), 0)

    def test_loss_weights_and_lr_floor(self):
        self.assertEqual(supervision_indices(1, 4), [0])
        self.assertEqual(len(supervision_indices(55, 4)), 4)
        self.assertEqual(supervision_indices(55, 4)[-1], 54)
        rates = [learning_rate_multiplier(step, 10, 100, 0.05) for step in range(100)]
        self.assertGreaterEqual(min(rates), 0.05)
        self.assertAlmostEqual(rates[-1], 0.05)
        phase, mask, _ = self.inputs()
        cfg = TrainConfig(lam_grad=0, lam_phi=0.25)
        final_only, _ = deep_supervised_loss(
            [(phase + mask * 0.1, phase)], phase, phase, mask, 0, cfg
        )
        weighted, _ = deep_supervised_loss(
            [(phase, phase)] * 3 + [(phase + mask * 0.1, phase)],
            phase,
            phase,
            mask,
            0,
            cfg,
        )
        torch.testing.assert_close(weighted, final_only * 0.7)

    def test_cache_and_epoch_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase, mask, _ = self.inputs()
            torch.save(phase[0], root / "image.pt")
            torch.save(mask[0].long(), root / "mask.pt")
            records = [
                {
                    "image": str(root / "image.pt"),
                    "mask": str(root / "mask.pt"),
                    "subject": "a:0",
                }
            ]
            dataset = TrainingVolumes(records, root / "cache")
            a = dataset[0]
            b = dataset[0]
            torch.testing.assert_close(a["chi"], b["chi"])
            dataset.set_epoch(1)
            self.assertNotEqual(a["seed"], dataset[0]["seed"])
            packed = torch.load(next((root / "cache").glob("*.pt")), weights_only=True)
            self.assertEqual(packed["mask"].dtype, torch.bool)
            self.assertEqual(packed["dilated"].dtype, torch.bool)

    def test_rms_mixture_samples_physical_range_with_encoded_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chi = torch.full((1, 12, 12, 12), 0.04)
            mask = torch.ones_like(chi, dtype=torch.bool)
            torch.save(chi, root / "image.pt")
            torch.save(mask, root / "mask.pt")
            records = [
                {
                    "image": str(root / "image.pt"),
                    "mask": str(root / "mask.pt"),
                    "subject": "qsm:0",
                }
            ]
            dataset = TrainingVolumes(
                records, root / "cache", chi_scale_mode="rms_mixture"
            )
            samples = [dataset[index] for index in range(10)]
            levels = [float(sample["chi"].square().mean().sqrt()) for sample in samples]
            self.assertTrue(all(0.018 <= value <= 0.08 for value in levels))
            self.assertGreater(len(set(round(value, 6) for value in levels)), 1)

    def test_rms_mixture_is_centered_near_external_references(self):
        generator = torch.Generator().manual_seed(0)
        values = torch.tensor([sample_target_chi_rms(generator) for _ in range(5000)])
        self.assertTrue(0.025 < float(values.median()) < 0.03)
        self.assertTrue(0.04 < float((values > 0.04).float().mean()) < 0.06)
        self.assertTrue(0.029 < float(values.square().mean().sqrt()) < 0.033)

    def test_warm_start_cli_uses_conservative_lr_unless_explicit(self):
        for options, expected in (([], 1e-5), (["--peak-lr", "2e-5"], 2e-5)):
            with (
                patch.object(
                    sys,
                    "argv",
                    ["whnet-train", "--init-ckpt", "initial.pth", *options],
                ),
                patch("whnet.training.run_training") as run,
            ):
                training_main()
                self.assertEqual(run.call_args.args[0].peak_lr, expected)

    def test_ema_state_restoration(self):
        model = self.model()
        ema = EMA(model, 0.9)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(0.001)
        ema.update(model)
        live = {k: v.clone() for k, v in model.state_dict().items()}
        with ema.average_parameters(model):
            model.refresh_spectral_norm()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, live[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(original[k], live[k]) for k in original))

    def test_comet_logger_loads_key_and_resumes_same_experiment(self):
        calls = []

        class FakeExperiment:
            def __init__(self, key):
                self.key = key

            def get_key(self):
                return self.key

            def log_parameters(self, parameters, prefix=None):
                calls.append(("parameters", parameters, prefix))

            def log_metrics(self, metrics, **kwargs):
                calls.append(("metrics", metrics, kwargs))

            def log_figure(self, **kwargs):
                calls.append(("figure", kwargs))

            def end(self):
                calls.append(("end", self.key))

        def start(**kwargs):
            calls.append(("start", kwargs))
            return FakeExperiment(kwargs["experiment_key"] or "test-experiment-key")

        fake_comet = types.SimpleNamespace(
            start=start,
            ExperimentConfig=lambda **kwargs: kwargs,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text("COMET_API_KEY=from-project-env\n")
            directory = root / "run"
            directory.mkdir()
            with (
                patch("whnet.training.PROJECT_ROOT", root),
                patch.dict(os.environ, {"COMET_API_KEY": ""}),
                patch.dict(sys.modules, {"comet_ml": fake_comet}),
            ):
                logger = RunLogger(directory, TrainConfig(comet=True))
                self.assertEqual(os.environ["COMET_API_KEY"], "from-project-env")
                logger.log(
                    {
                        "event": "training_started",
                        "step": 0,
                        "train_volumes": 8,
                        "initialization": "fresh",
                    }
                )
                logger.log(
                    {"event": "train", "step": 3, "epoch": 1, "group": 2, "loss": 0.5}
                )
                logger.log(
                    {
                        "event": "validation",
                        "step": 3,
                        "epoch": 1,
                        "val_chi_k2": 0.4,
                        "external_cosmos_chi_nrmse": 0.3,
                    }
                )
                import matplotlib.pyplot as plt

                figure, axis = plt.subplots()
                axis.plot([0, 1], [0, 1])
                logger.log_figure(
                    figure,
                    name="external_cosmos_chi_comparison_epoch_0001",
                    step=3,
                    epoch=1,
                )
                self.assertTrue(
                    (
                        directory
                        / "images/external_cosmos_chi_comparison_epoch_0001.png"
                    ).is_file()
                )
                logger.log({"event": "paused", "step": 3})
                logger.close()
                self.assertEqual(
                    json.loads((directory / "comet.json").read_text()),
                    {"experiment_key": "test-experiment-key"},
                )
                logger = RunLogger(
                    directory, TrainConfig(comet=True, resume="training_last.pt")
                )
                logger.log(
                    {"event": "training_started", "step": 3, "initialization": "resume"}
                )
                logger.close()

        starts = [call[1] for call in calls if call[0] == "start"]
        self.assertEqual(starts[0]["mode"], "create")
        self.assertEqual(starts[1]["mode"], "get")
        self.assertEqual(starts[1]["experiment_key"], "test-experiment-key")
        self.assertFalse(starts[0]["experiment_config"]["auto_output_logging"])
        metrics = [call for call in calls if call[0] == "metrics"]
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0][1], {"loss": 0.5})
        self.assertEqual(metrics[0][2], {"prefix": "train", "step": 3, "epoch": 1})
        self.assertEqual(
            metrics[1][1],
            {"val_chi_k2": 0.4, "external_cosmos_chi_nrmse": 0.3},
        )
        self.assertEqual(metrics[1][2]["prefix"], "validation")
        figure_logs = [call[1] for call in calls if call[0] == "figure"]
        self.assertEqual(len(figure_logs), 1)
        self.assertEqual(
            figure_logs[0]["figure_name"],
            "external_cosmos_chi_comparison_epoch_0001",
        )
        self.assertEqual(figure_logs[0]["step"], 3)
        self.assertEqual(figure_logs[0]["metadata"], {"epoch": 1})
        self.assertEqual(figure_logs[0]["format"], "png")
        self.assertTrue(figure_logs[0]["overwrite"])
        self.assertEqual(len([call for call in calls if call[0] == "parameters"]), 2)
        self.assertEqual(len([call for call in calls if call[0] == "end"]), 2)

    def test_comet_key_environment_precedence_and_missing_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text("COMET_API_KEY=from-project-env\n")
            with patch("whnet.training.PROJECT_ROOT", root):
                with patch.dict(os.environ, {"COMET_API_KEY": "exported-key"}):
                    _configure_comet_api_key()
                    self.assertEqual(os.environ["COMET_API_KEY"], "exported-key")
                (root / ".env").unlink()
                with patch.dict(os.environ, {"COMET_API_KEY": ""}):
                    with self.assertRaisesRegex(RuntimeError, "COMET_API_KEY"):
                        _configure_comet_api_key()

    def test_end_to_end_resume_matches_uninterrupted_training(self):
        from dataclasses import replace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            identities = {}
            for index in range(8):
                chi = torch.randn(1, 16, 16, 16) * 0.05
                mask = torch.zeros_like(chi, dtype=torch.bool)
                mask[:, 3:-3, 3:-3, 3:-3] = True
                torch.save(chi * mask, data / f"image_{index:04d}.pt")
                torch.save(mask, data / f"mask_{index:04d}.pt")
                identities[f"image_{index:04d}.pt"] = f"cohort:{index // 2}"
            manifest = root / "subjects.json"
            manifest.write_text(json.dumps(identities))
            cfg = TrainConfig(
                data_dir=str(data),
                source_manifest=str(manifest),
                cache_dir=str(root / "cache"),
                checkpoint_dir=str(root / "resumed"),
                epochs=1,
                batch_size=1,
                accumulation_steps=2,
                num_workers=0,
                device="cpu",
                val_frac=0.25,
                val_views=1,
                iter_start=2,
                iter_max=2,
                final_iters=2,
                final_epochs=1,
                warmup_epochs=0,
                wh_warmup_epochs=0,
                max_steps=1,
            )
            phase, mask, dipole = self.inputs(shape=(16, 16, 16))
            cases = (
                ExternalValidationCase(
                    "cosmos", phase, mask, dipole, phase, phase, 0.2125
                ),
                ExternalValidationCase(
                    "sim2",
                    phase,
                    mask,
                    dipole,
                    phase,
                    None,
                    0.264,
                    weight=mask * 0.5,
                ),
            )
            with patch(
                "whnet.training.load_external_cases", return_value=cases
            ) as load:
                run_training(cfg)
                status = json.loads((root / "resumed/status.json").read_text())
                self.assertEqual(status["status"], "paused")
                checkpoint = root / "resumed/training_last.pt"
                run_training(replace(cfg, resume=str(checkpoint), max_steps=0))
                self.assertEqual(
                    json.loads((root / "resumed/status.json").read_text())["status"],
                    "complete",
                )
                run_training(
                    replace(cfg, checkpoint_dir=str(root / "control"), max_steps=0)
                )
                self.assertEqual(load.call_count, 2)
            validation = [
                json.loads(row)
                for row in (root / "resumed/metrics.jsonl").read_text().splitlines()
                if '"event": "validation"' in row
            ]
            self.assertEqual(len(validation), 1)
            for name in (
                "external_cosmos_chi_nrmse",
                "external_cosmos_background_nrmse",
                "external_sim2_chi_nrmse",
            ):
                self.assertIn(name, validation[0])
            self.assertEqual(
                len(list((root / "resumed/images").glob("external_*_epoch_0001.png"))),
                3,
            )
            resumed = torch.load(root / "resumed/model_last.pth", weights_only=True)
            control = torch.load(root / "control/model_last.pth", weights_only=True)
            for key in resumed:
                torch.testing.assert_close(
                    resumed[key], control[key], rtol=2e-5, atol=1e-7
                )
            model = load_trained_model(root / "resumed/model_best.pth")
            self.assertFalse(model.training)
            self.assertEqual(model.num_iters, 2)


if __name__ == "__main__":
    unittest.main()

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from whnet.data import FieldSynthesizer, TrainingVolumes, separable_dilation
from whnet.training import (
    EMA,
    TrainConfig,
    deep_supervised_loss,
    learning_rate_multiplier,
    load_trained_model,
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
        cfg = TrainConfig()
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

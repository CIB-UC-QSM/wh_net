import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.generate_geometric_data import generate_dataset
from whnet.data import training_records
from whnet.synthetic import PATTERN_TYPES, SHAPE_TYPES, generate_volume
from whnet.training import TrainConfig, build_loaders, make_subject_split


class SyntheticDataTests(unittest.TestCase):
    def test_seeded_volumes_have_overlap_and_compatible_values(self):
        first, first_mask, first_info = generate_volume(23, size=64)
        again, again_mask, again_info = generate_volume(23, size=64)
        np.testing.assert_array_equal(first, again)
        np.testing.assert_array_equal(first_mask, again_mask)
        self.assertEqual(first_info, again_info)
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(first_mask.dtype, np.bool_)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.all(first[~first_mask] == 0))
        self.assertGreater(first_info["overlap_voxels"], 0)
        self.assertTrue(0.10 <= first_info["occupancy"] <= 0.20)
        self.assertTrue(4 <= first_info["shape_count"] <= 12)
        self.assertTrue(set(first_info["shape_types"]).issubset(SHAPE_TYPES))
        self.assertTrue(set(first_info["pattern_types"]).issubset(PATTERN_TYPES))

    def test_writes_pairs_manifest_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = torch.zeros((1, 160, 160, 160), dtype=torch.float32)
            original_mask = torch.ones_like(original, dtype=torch.bool)
            torch.save(original, root / "image_0000.pt")
            torch.save(original_mask, root / "mask_0000.pt")
            path = generate_dataset(root, count=2, seed=7)
            mapping = json.loads(path.read_text())
            self.assertEqual(mapping["image_0000.pt"], "qsm:0000")
            self.assertEqual(mapping["image_0651.pt"], "synthetic:0651")
            self.assertEqual(mapping["image_0652.pt"], "synthetic:0652")
            for index in (651, 652):
                chi = torch.load(root / f"image_{index:04d}.pt", weights_only=True)
                mask = torch.load(root / f"mask_{index:04d}.pt", weights_only=True)
                self.assertEqual(chi.shape, (1, 160, 160, 160))
                self.assertEqual(mask.shape, chi.shape)
                self.assertEqual(chi.dtype, torch.float32)
                self.assertEqual(mask.dtype, torch.bool)
                self.assertTrue(bool((chi[~mask] == 0).all()))
            self.assertEqual(len(training_records(root, path)), 3)
            self.assertEqual(training_records(root), training_records(root, path))
            self.assertTrue(
                torch.equal(
                    torch.load(root / "image_0000.pt", weights_only=True), original
                )
            )
            with self.assertRaises(FileExistsError):
                generate_dataset(root, count=2, seed=7)
            self.assertEqual(generate_dataset(root, count=2, seed=7, resume=True), path)
            with self.assertRaises(ValueError):
                generate_dataset(root, count=2, seed=8, resume=True)

    def test_synthetic_cohort_is_training_only(self):
        records = [
            {"subject": f"qsm:{index:04d}", "image": f"real_{index}"}
            for index in range(10)
        ] + [
            {"subject": f"synthetic:{index:04d}", "image": f"synthetic_{index}"}
            for index in range(12)
        ]
        train, validation = make_subject_split(records, 0.2, 0)
        self.assertEqual(len(validation), 2)
        self.assertEqual(
            sum(row["subject"].startswith("synthetic:") for row in train), 12
        )
        self.assertEqual((train, validation), make_subject_split(records, 0.2, 0))

    def test_training_loader_includes_synthetic_even_when_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            mapping = {}
            for index in (*range(4), *range(651, 655)):
                chi = torch.zeros((1, 8, 8, 8))
                mask = torch.ones_like(chi, dtype=torch.bool)
                torch.save(chi, data / f"image_{index:04d}.pt")
                torch.save(mask, data / f"mask_{index:04d}.pt")
                mapping[f"image_{index:04d}.pt"] = (
                    f"synthetic:{index:04d}" if index >= 651 else f"qsm:{index:04d}"
                )
            (data / "subjects.json").write_text(json.dumps(mapping))
            cfg = TrainConfig(
                data_dir=str(data),
                cache_dir=str(root / "cache"),
                device="cpu",
                num_workers=0,
                max_train_volumes=3,
                val_frac=0.25,
            )
            run = root / "run"
            run.mkdir()
            train_loader, validation_loader, shape = build_loaders(cfg, run)
            subjects = [record["subject"] for record in train_loader.dataset.records]
            self.assertEqual(shape, (8, 8, 8))
            self.assertEqual(len(subjects), 3)
            self.assertEqual(sum(name.startswith("synthetic:") for name in subjects), 2)
            self.assertEqual(len(validation_loader.dataset.records), 1)
            self.assertFalse(
                validation_loader.dataset.records[0]["subject"].startswith("synthetic:")
            )


if __name__ == "__main__":
    unittest.main()

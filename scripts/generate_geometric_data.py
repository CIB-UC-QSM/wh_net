"""Add seeded geometric mask/chi pairs to the joint training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from tqdm import tqdm

from whnet.paths import TRAINING_DATA_DIR
from whnet.synthetic import generate_volume


def _write_json(path: Path, value: dict) -> None:
    with NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def _save_tensor(path: Path, value: torch.Tensor) -> None:
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _subject_manifest(output_dir: Path, generated: set[int]) -> dict[str, str]:
    subjects = {}
    for image in sorted(output_dir.glob("image_*.pt")):
        index = int(image.stem.removeprefix("image_"))
        mask = output_dir / f"mask_{index:04d}.pt"
        if not mask.is_file():
            raise ValueError(f"Missing mask for {image}")
        if index in generated:
            subject = f"synthetic:{index:04d}"
        elif 0 <= index < 315:
            subject = f"qsm:{index % 105:04d}"
        elif 315 <= index < 651:
            subject = f"separate:{(index - 315) % 56:04d}"
        else:
            raise ValueError(f"Unknown source identity for {image}")
        subjects[image.name] = subject
    if not generated.issubset({int(name[6:-3]) for name in subjects}):
        raise ValueError("Some generated pairs are missing")
    return subjects


def generate_dataset(
    output_dir: Path,
    *,
    count: int = 651,
    start_index: int = 651,
    seed: int = 0,
    resume: bool = False,
) -> Path:
    """Write compatible pairs and a complete subject manifest without overwrites."""
    if count < 1 or start_index < 651 or seed < 0:
        raise ValueError("Require positive count, start_index >= 651, and seed >= 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "count": count,
        "start_index": start_index,
        "seed": seed,
        "size": 160,
    }
    config_path = output_dir / "synthetic_generation.json"
    generated = set(range(start_index, start_index + count))
    if resume:
        if (
            not config_path.is_file()
            or json.loads(config_path.read_text()) != configuration
        ):
            raise ValueError(
                "Cannot resume with different or missing generation settings"
            )
    else:
        if config_path.exists() or any(
            (output_dir / f"{prefix}_{index:04d}.pt").exists()
            for index in generated
            for prefix in ("image", "mask")
        ):
            raise FileExistsError("Synthetic generation already exists; use --resume")
        _write_json(config_path, configuration)

    for index in tqdm(sorted(generated), desc="Generating geometric QSM volumes"):
        image_path = output_dir / f"image_{index:04d}.pt"
        mask_path = output_dir / f"mask_{index:04d}.pt"
        if image_path.exists() and mask_path.exists():
            if resume:
                continue
            raise FileExistsError(image_path)
        if image_path.exists() or mask_path.exists():
            raise ValueError(f"Incomplete pair at index {index}")
        chi, mask, _ = generate_volume(seed + index - start_index)
        _save_tensor(mask_path, torch.from_numpy(mask).unsqueeze(0))
        _save_tensor(image_path, torch.from_numpy(chi).unsqueeze(0))

    manifest = _subject_manifest(output_dir, generated)
    manifest_path = output_dir / "subjects.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=TRAINING_DATA_DIR)
    parser.add_argument("--count", type=int, default=651)
    parser.add_argument("--start-index", type=int, default=651)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = generate_dataset(
        args.output_dir,
        count=args.count,
        start_index=args.start_index,
        seed=args.seed,
        resume=args.resume,
    )
    print(f"Wrote subject manifest: {manifest}")


if __name__ == "__main__":
    main()

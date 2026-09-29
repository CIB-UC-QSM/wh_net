"""Generate augmented 160³ training tensors from source QSM volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, zoom
from tqdm import tqdm

from whnet.paths import SOURCE_DATA_DIR, TRAINING_DATA_DIR


def process_3d_volume(
    image: np.ndarray, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    mask = image != 0
    if generator.random() > 0.5:
        mask = binary_erosion(binary_dilation(mask))

    coordinates = np.argwhere(mask)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    slices = tuple(slice(start, stop) for start, stop in zip(lower, upper))
    cropped_image = image[slices]
    cropped_mask = mask[slices]

    target_size = generator.integers(80, 140, 3)
    image_scale = tuple(
        size / current for size, current in zip(target_size, cropped_image.shape)
    )
    mask_scale = tuple(
        size / current for size, current in zip(target_size, cropped_mask.shape)
    )
    resized_image = zoom(cropped_image, image_scale, order=3)
    resized_mask = zoom(cropped_mask, mask_scale, order=0)

    remaining = 160 - target_size
    divisors = generator.integers(2, 5, 3)
    padding = tuple(
        (int(size // divisor), int(size - size // divisor))
        for size, divisor in zip(remaining, divisors)
    )
    return (
        np.pad(resized_image, padding, mode="constant"),
        np.pad(resized_mask, padding, mode="constant"),
    )


def augment_qsm_histogram(
    volume: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
    contrast = generator.uniform(0.9, 1.1)
    bias = generator.uniform(-0.02, 0.02)
    gamma = generator.uniform(0.95, 1.05)
    shifted = volume * contrast + bias
    return np.sign(shifted) * np.abs(shifted) ** gamma


def save_tensors(
    image: np.ndarray, mask: np.ndarray, image_path: Path, mask_path: Path
) -> None:
    torch.save(torch.from_numpy(image).float().unsqueeze(0), image_path)
    torch.save(torch.from_numpy(mask).bool().unsqueeze(0), mask_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DATA_DIR / "separate")
    parser.add_argument("--output-dir", type=Path, default=TRAINING_DATA_DIR)
    parser.add_argument("--subjects", type=int, default=56)
    parser.add_argument("--copies", type=int, default=6)
    parser.add_argument("--start-index", type=int, default=315)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(args.seed)
    total = args.subjects * args.copies

    for index in tqdm(range(total), desc="Generating training volumes"):
        source = args.source_dir / f"{index % args.subjects}.nii.gz"
        image = nib.load(source).get_fdata()
        image, mask = process_3d_volume(image, generator)
        if index > 200:
            image = augment_qsm_histogram(image, generator)
        destination = args.start_index + index
        save_tensors(
            image,
            mask,
            args.output_dir / f"image_{destination:04d}.pt",
            args.output_dir / f"mask_{destination:04d}.pt",
        )


if __name__ == "__main__":
    main()

"""Evaluate optimized WH-TV parameters and save COSMOS reconstructions/errors."""

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch import Tensor

from whnet.wh_tv import wh_tv

PARAMETER_NAMES = ("alpha1", "beta", "mu1", "mu2", "muh")
VOLUME_NAMES = (
    "chi_reference",
    "chi_reconstruction",
    "chi_error",
    "chi_absolute_error",
    "background_reference",
    "background_reconstruction",
    "background_error",
    "background_absolute_error",
    "mask",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed optimize_wh_tv.py run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--initial-solution",
        type=Path,
        help="Optional tensor dictionary containing chi and background",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Destination (default: RUN_DIR/evaluation)"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    return args


def masked_metrics(
    prediction: Tensor, reference: Tensor, mask: Tensor
) -> dict[str, float]:
    error = (prediction - reference)[mask.bool()]
    target = reference[mask.bool()]
    return {
        "nrmse": float(
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(target).clamp_min(1e-12)
        ),
        "rmse": float(error.square().mean().sqrt()),
        "mae": float(error.abs().mean()),
        "max_absolute_error": float(error.abs().max()),
    }


def crop(volume: Tensor, pad: int) -> np.ndarray:
    array = volume.detach().float().cpu().squeeze().numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected one 3D volume, got {array.shape}")
    if pad:
        if min(array.shape) <= 2 * pad:
            raise ValueError("Configured padding is too large for the saved simulation")
        array = array[pad:-pad, pad:-pad, pad:-pad]
    return np.asarray(array, dtype=np.float32)


def save_nifti(path: Path, volume: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)


def orthogonal_figure(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    error: np.ndarray,
    destination: Path,
    title: str,
) -> None:
    center = tuple(size // 2 for size in reference.shape)
    slices = (
        lambda x: x[center[0], :, :],
        lambda x: x[:, center[1], :],
        lambda x: x[:, :, center[2]],
    )
    values = np.concatenate((reference.ravel(), reconstruction.ravel()))
    finite = values[np.isfinite(values)]
    image_limit = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
    image_limit = max(image_limit, np.finfo(np.float32).eps)
    error_limit = float(np.percentile(np.abs(error[np.isfinite(error)]), 99.5))
    error_limit = max(error_limit, np.finfo(np.float32).eps)
    rows = (
        (reference, "Reference", "gray", -image_limit, image_limit),
        (reconstruction, "Reconstruction", "gray", -image_limit, image_limit),
        (
            error,
            "Error (reconstruction - reference)",
            "coolwarm",
            -error_limit,
            error_limit,
        ),
    )
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    figure.suptitle(title)
    for row, (volume, label, cmap, low, high) in enumerate(rows):
        for column, select in enumerate(slices):
            axes[row, column].imshow(
                np.rot90(select(volume)), cmap=cmap, vmin=low, vmax=high
            )
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].set_title(label)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu explicitly")

    config_path = args.run_dir / "config.json"
    best_path = args.run_dir / "best.json"
    simulation_path = args.run_dir / "simulation.pt"
    for required in (config_path, best_path, simulation_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    config = json.loads(config_path.read_text())
    best = json.loads(best_path.read_text())
    parameters = {name: float(best[name]) for name in PARAMETER_NAMES}
    if parameters["beta"] < float(config.get("min_beta", 1e6)):
        raise ValueError("Saved best beta violates the optimization minimum")
    iterations = int(config["iterations"])
    # if iterations != 2000:
        # raise ValueError(f"Expected a 2,000-iteration optimization, found {iterations}")

    data = torch.load(simulation_path, map_location=device, weights_only=True)
    required_data = {"phase", "mask", "dipole", "chi", "background"}
    if missing := required_data - data.keys():
        raise KeyError(f"simulation.pt is missing {sorted(missing)}")
    initial = None
    if args.initial_solution:
        saved = torch.load(
            args.initial_solution, map_location=device, weights_only=True
        )
        initial = (saved["chi"], saved["background"])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = wh_tv(
        data["phase"],
        data["dipole"],
        data["mask"],
        initial,
        weight=data["mask"],
        max_iterations=iterations,
        tol_update=None,
        **parameters,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    mask = data["mask"].bool()
    chi_error = (result.chi - data["chi"]) * mask
    background_error = (result.background - data["background"]) * mask
    factor = float(config["factor"])
    physical = {
        "chi_reference": data["chi"] * factor,
        "chi_reconstruction": result.chi * factor,
        "chi_error": chi_error * factor,
        "chi_absolute_error": chi_error.abs() * factor,
        "background_reference": data["background"] * factor,
        "background_reconstruction": result.background * factor,
        "background_error": background_error * factor,
        "background_absolute_error": background_error.abs() * factor,
        "mask": data["mask"],
    }
    pad = int(config.get("pad", 0))
    arrays = {name: crop(value, pad) for name, value in physical.items()}
    output = args.output_dir or args.run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=False)
    for name in VOLUME_NAMES:
        save_nifti(output / f"{name}.nii.gz", arrays[name])
    orthogonal_figure(
        arrays["chi_reference"],
        arrays["chi_reconstruction"],
        arrays["chi_error"],
        output / "chi_comparison.png",
        "WH-TV susceptibility (ppm)",
    )
    orthogonal_figure(
        arrays["background_reference"],
        arrays["background_reconstruction"],
        arrays["background_error"],
        output / "background_comparison.png",
        "WH-TV harmonic background field (ppm)",
    )

    metrics = {
        "source_run": str(args.run_dir.resolve()),
        "parameters": parameters,
        "iterations": result.iterations,
        "update_percent": result.update_percent,
        "seconds": elapsed,
        "factor": factor,
        "snr": config["snr"],
        "bg": config["bg"],
        "chi": masked_metrics(result.chi * factor, data["chi"] * factor, mask),
        "background": masked_metrics(
            result.background * factor, data["background"] * factor, mask
        ),
        "error_definition": "reconstruction - reference",
        "volume_units": "ppm except mask",
        "cropped_padding_voxels_per_side": pad,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()

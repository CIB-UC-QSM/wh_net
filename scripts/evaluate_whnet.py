"""Fast WHNet evaluation on the SNR=100, BG=10 COSMOS simulation."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch import Tensor

from whnet.model import ADMMUnrolledNet, ProximalNetwork
from whnet.paths import CHECKPOINTS_DIR, RUNS_DIR

DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "checkpoints_scratch7" / "model_best.pth"
REQUIRED_SIMULATION_KEYS = {"phase", "mask", "dipole", "chi", "background"}


def find_simulation() -> Path:
    """Return the newest completed COSMOS SNR=100/BG=10 simulation."""
    candidates = []
    for path in RUNS_DIR.glob("wh_tv_cosmos_*/simulation.pt"):
        config_path = path.parent / "config.json"
        status_path = path.parent / "status.json"
        if not config_path.is_file() or not status_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        status = json.loads(status_path.read_text())
        if (
            config.get("snr") == 100
            and config.get("bg") == 10
            and status.get("status") == "complete"
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No completed COSMOS SNR=100/BG=10 simulation found; pass --simulation"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=55)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile both proximal U-Nets (compilation is excluded from timings)",
    )
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New output directory under artifacts/runs by default",
    )
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error(
            "iterations, repeats, and threads must be positive; warmup may be zero"
        )
    if args.compile and args.warmup < 1:
        parser.error("compiled inference requires at least one untimed warm-up")
    return args


def masked_nrmse(prediction: Tensor, reference: Tensor, mask: Tensor) -> float:
    selected = mask.bool()
    return float(
        torch.linalg.vector_norm((prediction - reference)[selected])
        / torch.linalg.vector_norm(reference[selected]).clamp_min(1e-12)
    )


def crop(volume: Tensor, pad: int) -> np.ndarray:
    array = volume.detach().float().cpu().squeeze().numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected a single 3D volume, got {array.shape}")
    if pad:
        if min(array.shape) <= 2 * pad:
            raise ValueError("Configured padding is too large for the simulation")
        array = array[pad:-pad, pad:-pad, pad:-pad]
    return np.asarray(array, dtype=np.float32)


def save_comparison(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    error: np.ndarray,
    destination: Path,
    title: str,
) -> None:
    center = tuple(size // 2 for size in reference.shape)
    selectors = (
        lambda x: x[center[0]],
        lambda x: x[:, center[1]],
        lambda x: x[:, :, center[2]],
    )
    image_limit = max(
        float(
            np.percentile(
                np.abs(np.concatenate((reference.ravel(), reconstruction.ravel()))),
                99.5,
            )
        ),
        np.finfo(np.float32).eps,
    )
    error_limit = max(
        float(np.percentile(np.abs(error), 99.5)), np.finfo(np.float32).eps
    )
    rows = (
        (reference, "Reference", "gray", image_limit),
        (reconstruction, "WHNet", "gray", image_limit),
        (error, "Error (WHNet - reference)", "coolwarm", error_limit),
    )
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    figure.suptitle(title)
    for row, (volume, label, cmap, limit) in enumerate(rows):
        for column, select in enumerate(selectors):
            axes[row, column].imshow(
                np.rot90(select(volume)), cmap=cmap, vmin=-limit, vmax=limit
            )
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].set_title(label)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu explicitly")
    if args.bf16 and device.type != "cuda":
        raise ValueError(
            "BF16 acceleration is supported by this script only on CUDA; use --no-bf16"
        )
    simulation_path = args.simulation or find_simulation()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    data = torch.load(simulation_path, map_location="cpu", weights_only=True)
    if missing := REQUIRED_SIMULATION_KEYS - data.keys():
        raise KeyError(f"Simulation is missing {sorted(missing)}")
    config_path = simulation_path.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if config and (config.get("snr") != 100 or config.get("bg") != 10):
        raise ValueError(
            "The selected simulation is not configured for SNR=100 and BG=10"
        )

    model = ADMMUnrolledNet(
        ProximalNetwork(), ProximalNetwork(), num_iters=args.iterations
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last_3d)
    if args.compile:
        model.chinet.compile()
        model.phinet.compile()

    phase = data["phase"].to(device, non_blocking=True)
    mask = data["mask"].to(device, non_blocking=True)
    dipole = data["dipole"].to(device, non_blocking=True)
    weight = mask
    if args.channels_last:
        phase = phase.contiguous(memory_format=torch.channels_last_3d)
        mask = mask.contiguous(memory_format=torch.channels_last_3d)
        weight = weight.contiguous(memory_format=torch.channels_last_3d)

    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.bf16
        else nullcontext
    )
    compile_seconds = None
    with torch.inference_mode():
        for index in range(args.warmup):
            synchronize(device)
            started = time.perf_counter()
            with autocast():
                reconstruction = model(phase, mask, dipole, weight)
            synchronize(device)
            if index == 0 and args.compile:
                compile_seconds = time.perf_counter() - started
        durations = []
        for _ in range(args.repeats):
            synchronize(device)
            started = time.perf_counter()
            with autocast():
                reconstruction = model(phase, mask, dipole, weight)
            synchronize(device)
            durations.append(time.perf_counter() - started)
    chi, background = reconstruction
    if not bool(torch.isfinite(chi).all() and torch.isfinite(background).all()):
        raise FloatingPointError("WHNet returned a nonfinite reconstruction")

    factor = float(config.get("factor", 0.2125))
    pad = int(config.get("pad", 0))
    mask_bool = mask.bool()
    chi_nrmse = masked_nrmse(chi, data["chi"].to(device), mask_bool)
    background_nrmse = masked_nrmse(
        background, data["background"].to(device), mask_bool
    )
    values = {
        "chi_reference": data["chi"] * factor,
        "chi_reconstruction": chi.cpu() * factor,
        "chi_error": (chi.cpu() - data["chi"]) * data["mask"] * factor,
        "background_reference": data["background"] * factor,
        "background_reconstruction": background.cpu() * factor,
        "background_error": (background.cpu() - data["background"])
        * data["mask"]
        * factor,
        "mask": data["mask"],
    }
    arrays = {name: crop(value, pad) for name, value in values.items()}
    output = args.output_dir or RUNS_DIR / datetime.now(timezone.utc).strftime(
        "whnet_cosmos_%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, array in arrays.items():
        nib.save(nib.Nifti1Image(array, np.eye(4)), output / f"{name}.nii.gz")
    save_comparison(
        arrays["chi_reference"],
        arrays["chi_reconstruction"],
        arrays["chi_error"],
        output / "chi_comparison.png",
        f"WHNet susceptibility, NRMSE={chi_nrmse:.4f}",
    )
    save_comparison(
        arrays["background_reference"],
        arrays["background_reconstruction"],
        arrays["background_error"],
        output / "background_comparison.png",
        f"WHNet harmonic background, NRMSE={background_nrmse:.4f}",
    )
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "simulation": str(simulation_path.resolve()),
        "snr": config.get("snr", 100),
        "bg": config.get("bg", 10),
        "iterations": args.iterations,
        "chi_nrmse": chi_nrmse,
        "background_nrmse": background_nrmse,
        "timing_seconds": {
            "samples": durations,
            "median": statistics.median(durations),
            "minimum": min(durations),
            "mean": statistics.fmean(durations),
            "compile_first_warmup": compile_seconds,
            "excludes_loading_compilation_and_file_output": True,
        },
        "optimization": {
            "inference_mode": True,
            "bf16": args.bf16,
            "channels_last_3d": args.channels_last,
            "compiled_proximals": args.compile,
        },
        "output_units": "ppm except mask",
        "error_definition": "reconstruction - reference",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()

"""Seeded geometric susceptibility volumes for QSM training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

SHAPE_TYPES = ("ellipsoid", "cylinder", "cuboid", "cone")
PATTERN_TYPES = ("constant", "gradient", "texture")
VOLUME_FACTORS = {
    "ellipsoid": 4 * np.pi / 3,
    "cylinder": 2 * np.pi,
    "cuboid": 8.0,
    "cone": 2 * np.pi / 3,
}


@dataclass(frozen=True)
class Shape:
    kind: str
    pattern: str
    center: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray
    susceptibility: float
    gradient_axis: int
    gradient_strength: float


def _draw_shapes(
    size: int, count: int, target: float, generator: np.random.Generator
) -> list[Shape]:
    shapes = []
    signs = generator.choice((-1, 1), size=count)
    signs[:2] = (-1, 1)
    generator.shuffle(signs)
    for index in range(count):
        kind = str(generator.choice(SHAPE_TYPES))
        pattern = str(generator.choice(PATTERN_TYPES))
        anisotropy = generator.uniform(0.72, 1.28, size=3)
        anisotropy /= np.cbrt(np.prod(anisotropy))
        radius = np.cbrt(size**3 * target * 1.28 / (count * VOLUME_FACTORS[kind]))
        radii = (radius * anisotropy).astype(np.float32)
        if index == 1:
            center = shapes[0].center + generator.normal(0, radius * 0.35, size=3)
        elif index > 1 and generator.random() < 0.45:
            previous = shapes[int(generator.integers(index))]
            center = previous.center + generator.normal(0, radius * 0.9, size=3)
        else:
            center = generator.uniform(0.27 * size, 0.73 * size, size=3)
        center = np.clip(center, 0.12 * size, 0.88 * size).astype(np.float32)
        shapes.append(
            Shape(
                kind=kind,
                pattern=pattern,
                center=center,
                radii=radii,
                rotation=Rotation.random(random_state=generator)
                .as_matrix()
                .astype(np.float32),
                susceptibility=float(
                    signs[index]
                    * np.exp(generator.uniform(np.log(0.005), np.log(0.08)))
                ),
                gradient_axis=int(generator.integers(3)),
                gradient_strength=float(generator.uniform(0.3, 0.8)),
            )
        )
    return shapes


def _shape_region(
    size: int, shape: Shape, scale: float
) -> tuple[tuple[slice, slice, slice], np.ndarray, tuple[np.ndarray, ...]]:
    radii = shape.radii * scale
    extent = np.abs(shape.rotation) @ radii
    lower = np.maximum(0, np.floor(shape.center - extent - 1).astype(int))
    upper = np.minimum(size, np.ceil(shape.center + extent + 2).astype(int))
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    offsets = np.ogrid[
        tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    ]
    delta = [
        offset.astype(np.float32) - shape.center[axis]
        for axis, offset in enumerate(offsets)
    ]
    local = tuple(
        sum(shape.rotation[axis, component] * delta[axis] for axis in range(3))
        / radii[component]
        for component in range(3)
    )
    x, y, z = local
    if shape.kind == "ellipsoid":
        inside = x * x + y * y + z * z <= 1
    elif shape.kind == "cylinder":
        inside = (x * x + y * y <= 1) & (np.abs(z) <= 1)
    elif shape.kind == "cuboid":
        inside = (np.abs(x) <= 1) & (np.abs(y) <= 1) & (np.abs(z) <= 1)
    else:
        radial_limit = (1 - z) / 2
        inside = (z >= -1) & (z <= 1) & (x * x + y * y <= radial_limit**2)
    return slices, inside, local


def _render(
    size: int,
    shapes: list[Shape],
    scale: float,
    generator: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    mask = np.zeros((size, size, size), dtype=bool)
    chi = np.zeros(mask.shape, dtype=np.float32) if generator is not None else None
    overlap = np.zeros(mask.shape, dtype=bool)
    for shape in shapes:
        region, inside, local = _shape_region(size, shape, scale)
        overlap[region] |= mask[region] & inside
        mask[region] |= inside
        if chi is None:
            continue
        if shape.pattern == "constant":
            modulation = 1.0
        elif shape.pattern == "gradient":
            modulation = np.clip(
                1 + shape.gradient_strength * local[shape.gradient_axis], 0.2, 1.8
            )
        else:
            noise = generator.standard_normal(inside.shape, dtype=np.float32)
            sigma = tuple(max(2.0, dimension / 12) for dimension in inside.shape)
            noise = gaussian_filter(noise, sigma=sigma)
            selected = noise[inside]
            noise = (noise - selected.mean()) / max(float(selected.std()), 1e-6)
            modulation = np.clip(1 + 0.35 * noise, 0.2, 1.8)
        contribution = chi[region]
        contribution[inside] += shape.susceptibility * (
            np.broadcast_to(modulation, inside.shape)[inside]
            if isinstance(modulation, np.ndarray)
            else modulation
        )
    return chi, mask, int(overlap.sum())


def generate_volume(
    seed: int, size: int = 160
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return a float32 chi, boolean union mask, and generation summary."""
    if size < 32:
        raise ValueError("Volume size must be at least 32")
    generator = np.random.default_rng(seed)
    count = int(generator.integers(4, 13))
    target = float(generator.uniform(0.11, 0.19))
    shapes = _draw_shapes(size, count, target, generator)
    scale = 1.0
    for _ in range(12):
        _, mask, overlap = _render(size, shapes, scale)
        occupancy = float(mask.mean())
        if 0.10 <= occupancy <= 0.20 and overlap > 0:
            break
        scale *= np.clip((target / max(occupancy, 1e-6)) ** (1 / 3), 0.75, 1.35)
    else:
        raise RuntimeError(f"Could not meet occupancy and overlap for seed {seed}")
    chi, mask, overlap = _render(size, shapes, scale, generator)
    assert chi is not None
    return (
        chi,
        mask,
        {
            "seed": seed,
            "shape_count": count,
            "shape_types": [shape.kind for shape in shapes],
            "pattern_types": [shape.pattern for shape in shapes],
            "overlap_voxels": overlap,
            "occupancy": float(mask.mean()),
        },
    )

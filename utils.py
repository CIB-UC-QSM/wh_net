import numpy as np
from numpy.fft import ifftshift
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Any, Literal


def continuous_dipole_kernel(
    shape: tuple[int, int, int],
    voxel_size: tuple[int, int, int] = (1, 1, 1),
    b0_dir: tuple[int, int, int] = (0, 0, 1),
) -> npt.NDArray[np.float64]:
    rx = np.arange(-np.floor(shape[0] / 2), np.ceil(shape[0] / 2))
    ry = np.arange(-np.floor(shape[1] / 2), np.ceil(shape[1] / 2))
    rz = np.arange(-np.floor(shape[2] / 2), np.ceil(shape[2] / 2))

    kx, ky, kz = np.meshgrid(rx, ry, rz, indexing="ij")
    kx /= np.max(np.abs(kx)) * voxel_size[0]
    ky /= np.max(np.abs(ky)) * voxel_size[1]
    kz /= np.max(np.abs(kz)) * voxel_size[2]

    k2 = kx**2 + ky**2 + kz**2
    kernel = ifftshift(
        1 / 3.0
        - ((kx * b0_dir[0] + ky * b0_dir[1] + kz * b0_dir[2]) ** 2)
        / (k2 + np.finfo(np.float64).eps)
    )
    kernel[0, 0, 0] = 0
    return kernel


def rotation_by_permutation(
    im: npt.NDArray[np.floating[Any]], angle: Literal[0, 90, -90, 180]
) -> npt.NDArray[np.floating[Any]]:
    if angle == 90:
        im = im.T
    elif angle == -90:
        im = im.T
        im = im[::-1, :]
        return im
    elif angle == 180:
        im = im[:, :-1]
    return im


def imshow_3d(
    image: npt.NDArray[Any],
    title: str | None = None,
    cmap: str = "gray",
    rango: tuple[float, float] | None = None,
    angles: tuple[int, int, int] | None = None,
) -> None:
    if rango is None:
        rango = (image.min(), image.max())
    if angles is None:
        angles = (0, 0, 0)

    D, H, W = image.shape

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(
        rotation_by_permutation(image[D // 2, :, :], angles[0]),
        cmap=cmap,
        aspect="equal",
        vmin=rango[0],
        vmax=rango[1],
    )
    ax1.axis("off")
    ax2.imshow(
        rotation_by_permutation(image[:, H // 2, :], angles[1]),
        cmap=cmap,
        aspect="equal",
        vmin=rango[0],
        vmax=rango[1],
    )
    ax2.axis("off")
    ax3.imshow(
        rotation_by_permutation(image[:, :, W // 2], angles[2]),
        cmap=cmap,
        aspect="equal",
        vmin=rango[0],
        vmax=rango[1],
    )
    ax3.axis("off")

    if title:
        fig.suptitle(title, fontsize=32)

    plt.tight_layout()
    plt.show()


def rmse(
    pred: npt.NDArray[np.floating[Any]],
    gt: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    rmse_val = np.linalg.norm(pred[mask] - gt[mask]) / np.linalg.norm(gt[mask])
    return 100 * rmse_val


def pad_to_sqr_shape(
    volume: npt.NDArray[np.floating[Any]], zero_pad: int = 0
) -> npt.NDArray[np.floating[Any]]:
    x, y, z = volume.shape
    max_size = np.max([x, y, z]) + zero_pad
    new_volume = np.zeros(3 * (max_size,))
    new_volume[:x, :y, :z] = volume
    return new_volume


def forward_simulation(
    chi_map: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.bool_],
    kernel: npt.NDArray[np.floating[Any]] | None = None,
    magn: npt.NDArray[np.floating[Any]] | None = None,
    snr: int = 100,
    voxel_size: tuple[float, float, float] = (1, 1, 1),
    pad_to_sqr: bool = False,
    zero_pad: int = 0,
) -> npt.NDArray[np.floating[Any]]:
    if magn is None:
        magn = np.ones_like(mask)

    original_size = chi_map.shape
    if pad_to_sqr:
        chi_map = pad_to_sqr_shape(chi_map, zero_pad)
        mask = pad_to_sqr_shape(mask, zero_pad)
        magn = pad_to_sqr_shape(magn, zero_pad)

    if kernel is None:
        kernel = continuous_dipole_kernel(chi_map.shape, voxel_size)

    phase = np.real(np.fft.ifftn(kernel * np.fft.fftn(chi_map * mask)))
    scale = np.pi*0.99 / np.max(np.abs(phase))
    signal = magn * np.exp(1j * phase * scale)

    signal = signal + (
        (1.0 / snr)
        * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    )

    phase = np.angle(signal) * mask / scale
    phase = phase[: original_size[0], : original_size[1], : original_size[2]]
    return phase
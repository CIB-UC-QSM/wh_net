"""GPU-capable linear weak-harmonic weighted TV reconstruction.

Numerical formulation: FANSI WH_wTV.m, Carlos Milovic and Patrich Fuchs,
revision 4371ba3ecf3831af46dc7934cbaf7d32f7a7b16f. See docs/wh-tv.md
for attribution, parameter conventions and differences from the MATLAB driver.
"""

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from torch import Tensor

SPATIAL = (-3, -2, -1)


@dataclass
class WHTVResult:
    chi: Tensor
    background: Tensor
    iterations: int
    update_percent: float


def _gradient(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(x - torch.roll(x, -1, axis) for axis in SPATIAL)


def _divergence(components: tuple[Tensor, ...]) -> Tensor:
    return sum(g - torch.roll(g, 1, axis) for g, axis in zip(components, SPATIAL))


def _positive_laplacian(x: Tensor) -> Tensor:
    return sum(2 * x - torch.roll(x, -1, d) - torch.roll(x, 1, d) for d in SPATIAL)


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@torch.no_grad()
def wh_tv(
    phase: Tensor,
    dipole: Tensor,
    mask: Tensor | None = None,
    initial_solution: tuple[Tensor, Tensor] | None = None,
    *,
    alpha1: float = 1e-3,
    beta: float = 150.0,
    mu1: float | None = None,
    mu2: float = 1.0,
    muh: float = 5.0,
    weight: Tensor | None = None,
    regweight: Tensor | None = None,
    max_iterations: int = 2000,
    tol_update: float | None = None,
    check_every: int = 50,
    callback: Callable[[int, Tensor, Tensor, float], None] | None = None,
) -> WHTVResult:
    """Run the FANSI linear WH-wTV updates on ``phase.device``.

    Volumes are [N, 1, D, H, W]. ``dipole`` is a real, even Fourier kernel
    with DC at the origin, either full-sized or rFFT-sized; broadcasting is
    allowed. Spatial operators use periodic, unit-voxel finite differences.
    ``weight`` is squared internally, as in MATLAB. ``regweight`` may be a
    scalar-volume weight or [N, 1, D, H, W, 3] directional weights.

    ``initial_solution=(chi, background)`` uses the same padded grid and units
    as phase. Both are copied, split variables are initialized consistently,
    and duals start at zero. This is a warm start, not an exact ADMM resume.
    Without an initial solution, use MATLAB's isPrecond=false initialization;
    the separate MSMV/extrapolation preconditioner is not invoked.

    ``tol_update`` is a percentage, checked per sample after every chi update.
    None disables early stopping. As in MATLAB, the final iteration updates
    chi only; returned background is from the preceding split-variable update.
    The reference omits mu2 in the chi numerator; this is preserved here.
    Use mu2=1 for the conventional ADMM coefficient and parameter searches.
    FP16/BF16 inputs are promoted to FP32, FP64 is retained for verification.
    Outputs remain on the input device. This inference solver is not autograd.
    """
    mu1 = 100 * alpha1 if mu1 is None else mu1
    for name, value in dict(
        alpha1=alpha1, beta=beta, mu1=mu1, mu2=mu2, muh=muh
    ).items():
        _positive(name, value)
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not isinstance(check_every, int) or check_every < 1:
        raise ValueError("check_every must be a positive integer")
    if tol_update is not None and (not math.isfinite(tol_update) or tol_update < 0):
        raise ValueError("tol_update must be finite and nonnegative, or None")
    if phase.ndim != 5 or phase.shape[1] != 1 or any(n < 2 for n in phase.shape[-3:]):
        raise ValueError(
            "phase must have shape [N, 1, D, H, W] with spatial sizes >= 2"
        )
    if phase.is_complex():
        raise ValueError("phase must be real")
    dtype = torch.float64 if phase.dtype == torch.float64 else torch.float32
    phase = phase.to(dtype=dtype)
    if not bool(torch.isfinite(phase).all()):
        raise ValueError("phase must be finite")

    def volume(
        value: Tensor, name: str, shape: tuple[int, ...] = phase.shape
    ) -> Tensor:
        value = torch.as_tensor(value, device=phase.device)
        if value.is_complex():
            raise ValueError(f"{name} must be real")
        try:
            value = torch.broadcast_to(value.to(dtype=dtype), shape)
        except RuntimeError as exc:
            raise ValueError(f"{name} cannot broadcast to {shape}") from exc
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        return value

    weights = torch.ones_like(phase) if weight is None else volume(weight, "weight")
    if bool((weights < 0).any()):
        raise ValueError("weight must be nonnegative")
    weights = weights.square()
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("squared weights overflow")
    mask = (weights > 0).to(dtype) if mask is None else volume(mask, "mask")
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool(
        mask.flatten(1).any(1).all()
    ):
        raise ValueError("mask must be binary and nonempty for every sample")
    if regweight is None:
        regularization = (1.0, 1.0, 1.0)
    elif regweight.ndim == 6:
        rw = volume(regweight, "regweight", (*phase.shape, 3))
        regularization = rw.unbind(-1)
    else:
        rw = volume(regweight, "regweight")
        regularization = (rw, rw, rw)
    if any(bool((torch.as_tensor(w) < 0).any()) for w in regularization):
        raise ValueError("regweight must be nonnegative")
    shape = phase.shape[-3:]
    half_shape = (*phase.shape[:-1], shape[-1] // 2 + 1)
    dipole = torch.as_tensor(dipole, device=phase.device)
    if dipole.ndim and dipole.shape[-1] == shape[-1]:
        dipole = volume(dipole, "dipole")[..., : half_shape[-1]]
    else:
        dipole = volume(dipole, "dipole", half_shape)
    lap = _laplacian_spectrum(tuple(shape), phase.device, dtype)

    def fft(x: Tensor) -> Tensor:
        return torch.fft.rfftn(x, dim=SPATIAL)

    def inv(x: Tensor) -> Tensor:
        return torch.fft.irfftn(x, s=shape, dim=SPATIAL)

    # MATLAB's unqualified eps is double precision, including with single arrays.
    epsilon = torch.finfo(torch.float64).eps
    chi_denominator = epsilon + mu2 * dipole.square() + mu1 * lap
    background_denominator = epsilon + mu2 + muh * lap.square()
    fidelity_denominator = weights + mu2
    weighted_phase = weights * phase / fidelity_denominator
    zeros = torch.zeros_like(phase)
    if initial_solution is None:
        chi, background, z2, zh = (zeros.clone() for _ in range(4))
        z_gradient = tuple(zeros.clone() for _ in SPATIAL)
    else:
        if len(initial_solution) != 2:
            raise ValueError("initial_solution must contain (chi, background)")
        chi = volume(initial_solution[0], "initial chi").clone() * mask
        background = volume(initial_solution[1], "initial background").clone()
        z_gradient = _gradient(chi)
        z2 = inv(dipole * fft(chi)) + background
        zh = _positive_laplacian(background)
    s_gradient = tuple(zeros.clone() for _ in SPATIAL)
    s2, sh = zeros.clone(), zeros.clone()
    update = math.inf

    with torch.autocast(device_type=phase.device.type, enabled=False):
        for iteration in range(1, max_iterations + 1):
            previous = chi
            divergence = _divergence(
                tuple(z - s for z, s in zip(z_gradient, s_gradient))
            )
            # Preserve the reference numerator (no mu2 multiplier on this term).
            gradient_rhs = fft(divergence)
            # A periodic derivative has zero DC, even after float32 roundoff.
            gradient_rhs[..., 0, 0, 0] = 0
            rhs = mu1 * gradient_rhs + dipole * fft(z2 - s2 - background)
            chi = inv(rhs / chi_denominator) * mask
            check = iteration % check_every == 0 or iteration == max_iterations
            stop = False
            if check or tol_update is not None:
                delta = torch.linalg.vector_norm((chi - previous).flatten(1), dim=1)
                norm = torch.linalg.vector_norm(chi.flatten(1), dim=1)
                ratios = 100 * delta / norm.clamp_min(torch.finfo(dtype).tiny)
                update = float(ratios.max())
                if not math.isfinite(update) or not bool(
                    torch.isfinite(background).all()
                ):
                    raise FloatingPointError(f"Nonfinite WH-TV iterate at {iteration}")
                stop = tol_update is not None and bool((ratios <= tol_update).all())
            if callback is not None and (check or stop):
                callback(iteration, chi, background, update)
            if stop or iteration == max_iterations:
                break
            gradients = _gradient(chi)
            shifted = tuple(g + s for g, s in zip(gradients, s_gradient))
            z_gradient = tuple(
                v.sign() * (v.abs() - w * alpha1 / mu1).clamp_min(0)
                for v, w in zip(shifted, regularization)
            )
            s_gradient = tuple(v - z for v, z in zip(shifted, z_gradient))
            field = inv(dipole * fft(chi))
            z2 = weighted_phase + mu2 * (field + s2 + background) / fidelity_denominator
            background = inv(
                (muh * lap * fft(zh - sh) + mu2 * fft(z2 - s2 - field))
                / background_denominator
            )
            lap_background = _positive_laplacian(background)
            zh = muh * (lap_background + sh) / (muh + beta * mask)
            s2 = s2 + field - z2 + background
            sh = sh + lap_background - zh
    return WHTVResult(chi, background, iteration, update)


def _laplacian_spectrum(
    shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Positive periodic finite-difference Laplacian on an rFFT grid."""
    half_shape = (shape[0], shape[1], shape[2] // 2 + 1)
    lap = torch.zeros(half_shape, device=device, dtype=dtype)
    for axis, size in enumerate(shape):
        count = size // 2 + 1 if axis == 2 else size
        frequency = torch.arange(count, device=device, dtype=dtype)
        dimensions = [1, 1, 1]
        dimensions[axis] = count
        lap += (4 * torch.sin(torch.pi * frequency / size).square()).reshape(dimensions)
    return lap


def _batch_parameter(value: Tensor | float, name: str, phase: Tensor) -> Tensor:
    """Convert a positive scalar or per-sample vector to [N,1,1,1,1]."""
    result = torch.as_tensor(value, device=phase.device, dtype=phase.dtype)
    if result.ndim == 0:
        result = result.expand(phase.shape[0])
    if result.ndim == 1 and result.shape[0] == phase.shape[0]:
        result = result.reshape(-1, 1, 1, 1, 1)
    try:
        result = torch.broadcast_to(result, (phase.shape[0], 1, 1, 1, 1))
    except RuntimeError as exc:
        raise ValueError(f"{name} must be scalar or have one value per sample") from exc
    detached = result.detach()
    if not bool(torch.isfinite(detached).all()) or bool((detached <= 0).any()):
        raise ValueError(f"{name} must be finite and positive")
    return result


def differentiable_wh_tv(
    phase: Tensor,
    dipole: Tensor,
    mask: Tensor,
    *,
    fidelity_precision: Tensor,
    alpha1: Tensor | float,
    beta: Tensor | float,
    mu1: Tensor | float,
    mu2: Tensor | float,
    muh: Tensor | float,
    max_iterations: int = 2000,
    gradient_tail: int = 25,
    checkpoint_block: int = 5,
) -> WHTVResult:
    """Run WH-TV while differentiating through only its final iterations.

    fidelity_precision is equivalent to weight squared in wh_tv. The prefix is
    evaluated without an autograd graph and the complete ADMM state is detached
    before the final gradient_tail updates. The reconstruction is therefore the
    exact full-horizon result with a deliberately truncated last-K gradient.
    """
    from torch.utils.checkpoint import checkpoint

    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not isinstance(gradient_tail, int) or not 0 <= gradient_tail <= max_iterations:
        raise ValueError("gradient_tail must be between zero and max_iterations")
    if not isinstance(checkpoint_block, int) or checkpoint_block < 1:
        raise ValueError("checkpoint_block must be a positive integer")
    if phase.ndim != 5 or phase.shape[1] != 1 or min(phase.shape[-3:]) < 2:
        raise ValueError("phase must have shape [N, 1, D, H, W]")
    if phase.is_complex():
        raise ValueError("phase must be real")

    dtype = torch.float64 if phase.dtype == torch.float64 else torch.float32
    phase = phase.to(dtype=dtype)
    mask = torch.as_tensor(mask, device=phase.device, dtype=dtype)
    precision = torch.as_tensor(fidelity_precision, device=phase.device, dtype=dtype)
    try:
        mask = torch.broadcast_to(mask, phase.shape)
        precision = torch.broadcast_to(precision, phase.shape)
    except RuntimeError as exc:
        raise ValueError("mask and fidelity_precision must broadcast to phase") from exc
    if not bool(torch.isfinite(phase.detach()).all()):
        raise ValueError("phase must be finite")
    if not bool(((mask.detach() == 0) | (mask.detach() == 1)).all()) or not bool(
        mask.detach().flatten(1).any(1).all()
    ):
        raise ValueError("mask must be binary and nonempty for every sample")
    if not bool(torch.isfinite(precision.detach()).all()) or bool(
        (precision.detach() < 0).any()
    ):
        raise ValueError("fidelity_precision must be finite and nonnegative")

    parameters = tuple(
        _batch_parameter(value, name, phase)
        for name, value in (
            ("alpha1", alpha1),
            ("beta", beta),
            ("mu1", mu1),
            ("mu2", mu2),
            ("muh", muh),
        )
    )
    shape = tuple(phase.shape[-3:])
    half_shape = (*phase.shape[:-1], shape[-1] // 2 + 1)
    dipole = torch.as_tensor(dipole, device=phase.device, dtype=dtype)
    if dipole.ndim and dipole.shape[-1] == shape[-1]:
        try:
            dipole = torch.broadcast_to(dipole, phase.shape)[..., : half_shape[-1]]
        except RuntimeError as exc:
            raise ValueError("dipole cannot broadcast to phase") from exc
    else:
        try:
            dipole = torch.broadcast_to(dipole, half_shape)
        except RuntimeError as exc:
            raise ValueError("dipole cannot broadcast to the phase rFFT grid") from exc
    if not bool(torch.isfinite(dipole.detach()).all()):
        raise ValueError("dipole must be finite")
    laplacian = _laplacian_spectrum(shape, phase.device, dtype)
    epsilon = torch.finfo(torch.float64).eps

    def fft(value: Tensor) -> Tensor:
        return torch.fft.rfftn(value, dim=SPATIAL)

    def inverse(value: Tensor) -> Tensor:
        return torch.fft.irfftn(value, s=shape, dim=SPATIAL)

    zeros = torch.zeros_like(phase)
    # chi, background, z-gradient (3), z2, zh, s-gradient (3), s2, sh
    state = tuple(zeros.clone() for _ in range(12))

    def run_block(
        *arguments: Tensor,
        block_start: int,
        block_count: int,
    ) -> tuple[Tensor, ...]:
        current = arguments[:12]
        p_alpha1, p_beta, p_mu1, p_mu2, p_muh, p_precision = arguments[12:]
        chi_denominator = epsilon + p_mu2 * dipole.square() + p_mu1 * laplacian
        background_denominator = epsilon + p_mu2 + p_muh * laplacian.square()
        fidelity_denominator = p_precision + p_mu2
        weighted_phase = p_precision * phase / fidelity_denominator
        for offset in range(block_count):
            (
                chi,
                background,
                zg0,
                zg1,
                zg2,
                z2,
                zh,
                sg0,
                sg1,
                sg2,
                s2,
                sh,
            ) = current
            z_gradient = (zg0, zg1, zg2)
            s_gradient = (sg0, sg1, sg2)
            divergence = _divergence(
                tuple(z - s for z, s in zip(z_gradient, s_gradient))
            )
            gradient_rhs = fft(divergence).clone()
            gradient_rhs[..., 0, 0, 0] = 0
            rhs = p_mu1 * gradient_rhs + dipole * fft(z2 - s2 - background)
            chi = inverse(rhs / chi_denominator) * mask
            iteration = block_start + offset
            if iteration == max_iterations - 1:
                current = (
                    chi,
                    background,
                    *z_gradient,
                    z2,
                    zh,
                    *s_gradient,
                    s2,
                    sh,
                )
                continue
            gradients = _gradient(chi)
            shifted = tuple(g + s for g, s in zip(gradients, s_gradient))
            threshold = p_alpha1 / p_mu1
            z_gradient = tuple(
                value.sign() * (value.abs() - threshold).clamp_min(0)
                for value in shifted
            )
            s_gradient = tuple(
                value - split for value, split in zip(shifted, z_gradient)
            )
            field = inverse(dipole * fft(chi))
            z2 = (
                weighted_phase
                + p_mu2 * (field + s2 + background) / fidelity_denominator
            )
            background = inverse(
                (p_muh * laplacian * fft(zh - sh) + p_mu2 * fft(z2 - s2 - field))
                / background_denominator
            )
            lap_background = _positive_laplacian(background)
            zh = p_muh * (lap_background + sh) / (p_muh + p_beta * mask)
            s2 = s2 + field - z2 + background
            sh = sh + lap_background - zh
            current = (
                chi,
                background,
                *z_gradient,
                z2,
                zh,
                *s_gradient,
                s2,
                sh,
            )
        return current

    tail = gradient_tail if torch.is_grad_enabled() else 0
    prefix = max_iterations - tail
    detached_arguments = tuple(value.detach() for value in parameters) + (
        precision.detach(),
    )
    with torch.autocast(device_type=phase.device.type, enabled=False):
        with torch.no_grad():
            position = 0
            while position < prefix:
                count = min(checkpoint_block, prefix - position)
                state = run_block(
                    *state,
                    *detached_arguments,
                    block_start=position,
                    block_count=count,
                )
                position += count
        state = tuple(value.detach() for value in state)
        position = prefix
        live_arguments = parameters + (precision,)
        while position < max_iterations:
            count = min(checkpoint_block, max_iterations - position)

            def block(*values: Tensor, start: int = position, size: int = count):
                return run_block(*values, block_start=start, block_count=size)

            state = checkpoint(
                block,
                *state,
                *live_arguments,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            position += count

    chi, background = state[:2]
    return WHTVResult(chi, background, max_iterations, math.nan)

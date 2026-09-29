"""Two-prior FlowADMM reconstruction for masked quantitative susceptibility mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
import json
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SPATIAL = (-3, -2, -1)
FLOW_ENSEMBLE_FORMAT = "whnet-flow-ensemble-v4"
FLOW_DATA_TERM = "weighted_residual_l2"
FLOW_SOLVER = "whnet_closed_form"


def phase_scale(phase: Tensor, mask: Tensor) -> Tensor:
    """The same per-volume scale is used for prior training and reconstruction."""
    count = mask.sum(dim=SPATIAL, keepdim=True).clamp_min(1)
    return (
        ((phase.float().square() * mask).sum(dim=SPATIAL, keepdim=True) / count)
        .sqrt()
        .clamp_min(1e-3)
    )


def _time_batch(t: Tensor | float, x: Tensor) -> Tensor:
    result = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    if result.ndim == 0:
        result = result.expand(x.shape[0])
    if result.shape != (x.shape[0],):
        raise ValueError("time must be scalar or one value per batch item")
    return result


class _FlowBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.film = nn.Linear(time_width, out_channels * 2)

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        x = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.film(time).chunk(2, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        return F.silu(self.norm2(self.conv2(x)) * (1 + scale) + shift)


class FlowVelocity3D(nn.Module):
    """Mask conditioned velocity network for one scalar volume prior."""

    def __init__(
        self, base_channels: int = 8, *, condition_on_field: bool = False
    ) -> None:
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError(
                "base_channels must be a multiple of four and at least four"
            )
        width = base_channels
        self.condition_on_field = condition_on_field
        self.time_mlp = nn.Sequential(
            nn.Linear(32, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.enc1 = _FlowBlock(3 if condition_on_field else 2, width, 128)
        self.enc2 = _FlowBlock(width, 2 * width, 128)
        self.enc3 = _FlowBlock(2 * width, 4 * width, 128)
        self.mid = _FlowBlock(4 * width, 8 * width, 128)
        self.dec3 = _FlowBlock(12 * width, 4 * width, 128)
        self.dec2 = _FlowBlock(6 * width, 2 * width, 128)
        self.dec1 = _FlowBlock(3 * width, width, 128)
        self.out = nn.Conv3d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor | float,
        mask: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        if x_t.ndim != 5 or x_t.shape[1] != 1 or x_t.shape != mask.shape:
            raise ValueError("x_t and mask must have matching [N, 1, D, H, W] shapes")
        if min(x_t.shape[-3:]) < 8:
            raise ValueError("flow network requires at least eight voxels per axis")
        if self.condition_on_field != (condition is not None):
            raise ValueError(
                "field condition is required exactly for the background flow"
            )
        if condition is not None and condition.shape != x_t.shape:
            raise ValueError("field condition must match x_t")
        times = _time_batch(t, x_t)
        frequencies = 2 ** torch.arange(16, device=x_t.device, dtype=x_t.dtype)
        angles = 2 * math.pi * times[:, None] * frequencies[None]
        embedding = self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=1))
        channels = (
            (x_t * mask, mask)
            if condition is None
            else (x_t * mask, condition * mask, mask)
        )
        a = self.enc1(torch.cat(channels, dim=1), embedding)
        b = self.enc2(F.avg_pool3d(a, 2), embedding)
        c = self.enc3(F.avg_pool3d(b, 2), embedding)
        h = self.mid(F.avg_pool3d(c, 2), embedding)
        h = self.dec3(
            torch.cat(
                (
                    F.interpolate(
                        h, size=c.shape[-3:], mode="trilinear", align_corners=False
                    ),
                    c,
                ),
                dim=1,
            ),
            embedding,
        )
        h = self.dec2(
            torch.cat(
                (
                    F.interpolate(
                        h, size=b.shape[-3:], mode="trilinear", align_corners=False
                    ),
                    b,
                ),
                dim=1,
            ),
            embedding,
        )
        h = self.dec1(
            torch.cat(
                (
                    F.interpolate(
                        h, size=a.shape[-3:], mode="trilinear", align_corners=False
                    ),
                    a,
                ),
                dim=1,
            ),
            embedding,
        )
        return self.out(h) * mask


def flow_denoise(
    velocity: nn.Module,
    x_t: Tensor,
    t: Tensor | float,
    mask: Tensor,
    condition: Tensor | None = None,
) -> Tensor:
    times = _time_batch(t, x_t)[:, None, None, None, None]
    predicted = (
        velocity(x_t, t, mask)
        if condition is None
        else velocity(x_t, t, mask, condition=condition)
    )
    return (x_t + (1 - times) * predicted) * mask


def mean_flow_prior(
    velocity: nn.Module,
    x: Tensor,
    mask: Tensor,
    t: float,
    samples: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    if not 0 <= t <= 1 or samples < 1:
        raise ValueError("time must be in [0, 1] and samples must be positive")
    total = torch.zeros_like(x, dtype=torch.float32)
    for _ in range(samples):
        noise = (
            torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            * mask
        )
        x_t = (t * x + (1 - t) * noise) * mask
        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if x.device.type == "cuda"
            else nullcontext()
        ):
            denoised = flow_denoise(velocity, x_t, t, mask)
        total += denoised.float()
    return total.div_(samples) * mask


def conditional_field_prior(
    velocity: nn.Module,
    proposal: Tensor,
    measured_field: Tensor,
    mask: Tensor,
    t: float,
) -> Tensor:
    """Deterministic measured-total-field-to-background flow update."""
    if not 0 <= t <= 1 or proposal.shape != measured_field.shape:
        raise ValueError("invalid time or measured field shape")
    mixed = (t * proposal + (1 - t) * measured_field) * mask
    with (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if proposal.device.type == "cuda"
        else nullcontext()
    ):
        result = flow_denoise(velocity, mixed, t, mask, condition=measured_field)
    return result.float() * mask


def _laplacian(x: Tensor) -> Tensor:
    with torch.autocast(x.device.type, enabled=False):
        kernel = x.new_zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        kernel[0, 0, 1, 1, 1] = -6
        for index in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
            kernel[(0, 0, *index)] = 1
        return F.conv3d(x.float(), kernel, padding=1)


def interior_mask(mask: Tensor) -> Tensor:
    with torch.autocast(mask.device.type, enabled=False):
        stencil = torch.ones((1, 1, 3, 3, 3), device=mask.device, dtype=torch.float32)
        return (F.conv3d(mask.float(), stencil, padding=1) == 27).to(mask.dtype)


def whnet_closed_form_physics(
    split_field: Tensor,
    background: Tensor,
    chi_prior: Tensor,
    background_prior: Tensor,
    dual_field: Tensor,
    dual_chi: Tensor,
    dual_background: Tensor,
    mask: Tensor,
    dipole: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """WH-Net's FFT chi update, support projection, and pointwise b update.

    ``dipole`` is a real rFFT kernel. The Fourier solve is exact before
    projecting chi onto the mask; that projection is the same approximation
    used by EfficientADMMUnrolledNet. Recompute the field from projected chi.
    Background is retained on the full padded grid during the outer loop.
    """
    with torch.autocast(split_field.device.type, enabled=False):
        shape = split_field.shape[-3:]
        rhs = dipole * torch.fft.rfftn(
            (split_field - dual_field - background).float(), dim=SPATIAL
        ) + torch.fft.rfftn((chi_prior - dual_chi).float(), dim=SPATIAL)
        chi = torch.fft.irfftn(rhs / (dipole.square() + 1), s=shape, dim=SPATIAL) * mask
        field = torch.fft.irfftn(
            dipole * torch.fft.rfftn(chi, dim=SPATIAL), s=shape, dim=SPATIAL
        )
        next_background = 0.5 * (
            split_field.float()
            - dual_field
            - field
            + background_prior
            - dual_background
        )
    return chi, next_background, field


@dataclass(frozen=True)
class FlowADMMConfig:
    iterations: int = 30
    tau: float = 1.0
    t_min: float = 0.3
    t_max: float = 0.95
    gamma: float = 1.0
    samples_early: int = 1
    samples_middle: int = 2
    samples_late: int = 5
    first_transition: float = 0.5
    second_transition: float = 0.8

    def validate(self) -> None:
        if self.iterations < 1 or not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("invalid ADMM iteration count or data penalty")
        if (
            not 0 < self.t_min <= self.t_max < 1
            or not math.isfinite(self.gamma)
            or self.gamma <= 0
        ):
            raise ValueError("invalid time schedule")
        if not 0 < self.first_transition < self.second_transition < 1:
            raise ValueError("invalid sample transitions")
        if (
            min(
                self.samples_early,
                self.samples_middle,
                self.samples_late,
            )
            < 1
        ):
            raise ValueError("invalid sample counts")

    def schedule(self, index: int, total: int | None = None) -> tuple[float, int]:
        count = self.iterations if total is None else total
        if not 0 <= index < count:
            raise ValueError("iteration outside schedule")
        time = self.t_min + ((index + 1) / count) ** self.gamma * (
            self.t_max - self.t_min
        )
        fraction = index / count
        samples = (
            self.samples_early
            if fraction < self.first_transition
            else self.samples_middle
            if fraction < self.second_transition
            else self.samples_late
        )
        return time, samples


class FlowADMMNet(nn.Module):
    """Frozen priors coupled by 0.5 * ||W * (F^H D F chi + b - y)||^2.

    ``weights`` multiplies the residual; its square appears in WH-Net's
    measured-field split. Harmonic residuals are diagnostics only.
    All three ADMM split constraints have penalty rho = 1 / tau.
    """

    def __init__(
        self,
        chi_prior: nn.Module,
        background_prior: nn.Module,
        config: FlowADMMConfig | None = None,
    ) -> None:
        super().__init__()
        self.chi_prior = chi_prior.eval().requires_grad_(False)
        self.background_prior = background_prior.eval().requires_grad_(False)
        self.config = config or FlowADMMConfig()
        self.config.validate()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(
        self,
        phase: Tensor,
        mask: Tensor,
        dipole: Tensor,
        weights: Tensor | None = None,
        *,
        max_iters: int | None = None,
        seed: int | None = None,
        return_diagnostics: bool = False,
    ):
        if phase.ndim != 5 or phase.shape[1] != 1 or phase.shape != mask.shape:
            raise ValueError("phase and mask must match [N, 1, D, H, W]")
        if max_iters is not None and max_iters < 1:
            raise ValueError("max_iters must be positive")
        phase = phase.float()
        if not bool(torch.isfinite(phase).all() and torch.isfinite(mask).all()):
            raise ValueError("phase and mask must be finite")
        mask = (mask > 0.5).float()
        if bool((mask.flatten(1).sum(1) == 0).any()):
            raise ValueError("each mask must contain at least one voxel")
        weights = mask if weights is None else weights.float() * mask
        if (
            weights.shape != phase.shape
            or bool((weights < 0).any())
            or not bool(torch.isfinite(weights).all())
        ):
            raise ValueError("weights must match phase and be finite nonnegative")
        # The measured-field split minimizes 0.5 ||W(s-y)||^2 + rho/2 ||s-q||^2.
        weights_squared = weights.square()
        if not bool(torch.isfinite(weights_squared).all()):
            raise ValueError("squared fidelity weights overflow")
        scale = phase_scale(phase, mask)
        measurement = phase / scale
        shape = phase.shape[-3:]
        half_width = shape[-1] // 2 + 1
        if dipole.shape[-3:] == shape:
            dipole = dipole[..., :half_width]
        if dipole.shape[-3:] != (*shape[:-1], half_width):
            raise ValueError("dipole grid does not match phase")
        if dipole.is_complex() or not bool(torch.isfinite(dipole).all()):
            raise ValueError("dipole must be a finite real Fourier kernel")
        dipole = dipole.to(device=phase.device, dtype=torch.float32)
        interior = interior_mask(mask) if return_diagnostics else None

        chi, background, dual_field, dual_chi, dual_background = (
            torch.zeros_like(phase) for _ in range(5)
        )
        field = torch.zeros_like(phase)
        weighted_measurement = self.config.tau * weights_squared * measurement
        fidelity_denominator = 1 + self.config.tau * weights_squared
        total = self.config.iterations if max_iters is None else max_iters
        generator = None
        if seed is not None:
            generator = torch.Generator(device=phase.device).manual_seed(seed)
        diagnostics = []
        for index in range(total):
            previous_chi, previous_background = chi, background
            split_field = (
                weighted_measurement + field + background + dual_field
            ) / fidelity_denominator
            time, samples = self.config.schedule(index, total)
            chi_prior = mean_flow_prior(
                self.chi_prior,
                (chi + dual_chi) * mask,
                mask,
                time,
                samples,
                generator,
            )
            background_prior = conditional_field_prior(
                self.background_prior,
                (background + dual_background) * mask,
                measurement,
                mask,
                time,
            )
            chi, background, field = whnet_closed_form_physics(
                split_field,
                background,
                chi_prior,
                background_prior,
                dual_field,
                dual_chi,
                dual_background,
                mask,
                dipole,
            )
            field_primal = field + background - split_field
            chi_primal = chi - chi_prior
            background_primal = background - background_prior
            dual_field = dual_field + field_primal
            dual_chi = dual_chi + chi_primal
            dual_background = dual_background + background_primal
            if return_diagnostics:
                data_residual = field + background - measurement
                count = mask.sum().clamp_min(1)

                def rms(value: Tensor) -> float:
                    return float((value.square() * mask).sum().div(count).sqrt())

                diagnostics.append(
                    {
                        "iteration": index + 1,
                        "time": time,
                        "chi_samples": samples,
                        "background_evaluations": 1,
                        "data_split_rms": rms(field_primal),
                        "chi_split_rms": rms(chi_primal),
                        "background_split_rms": rms(background_primal),
                        "chi_update_rms": rms(chi - previous_chi),
                        "background_update_rms": rms(background - previous_background),
                        "primal_rms": float(
                            (
                                (
                                    field_primal.square()
                                    + chi_primal.square()
                                    + background_primal.square()
                                )
                                * mask
                            )
                            .sum()
                            .div(count)
                            .sqrt()
                        ),
                        "data_rms": float(
                            (data_residual.square() * mask).sum().div(count).sqrt()
                        ),
                        "weighted_data_rms": float(
                            (weights * data_residual).square().sum().div(count).sqrt()
                        ),
                        "harmonic_rms": float(
                            (_laplacian(background * mask).square() * interior)
                            .sum()
                            .div(interior.sum().clamp_min(1))
                            .sqrt()
                        ),
                    }
                )
        output = chi * scale * mask, background * scale * mask
        if not bool(
            torch.isfinite(output[0]).all() and torch.isfinite(output[1]).all()
        ):
            raise FloatingPointError("nonfinite FlowADMM reconstruction")
        return (*output, diagnostics) if return_diagnostics else output


def load_flow_admm(
    chi_checkpoint: str | Path,
    background_checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    config: FlowADMMConfig | None = None,
) -> FlowADMMNet:
    """Load independently trained priors and construct their coupled ensemble."""
    priors = []
    for expected_branch, path in (
        ("chi", chi_checkpoint),
        ("background", background_checkpoint),
    ):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            state.get("branch") != expected_branch
            or state.get("format") != "whnet-flow-v2"
            or state.get("normalization") != "masked_phase_rms_clamp_1e-3"
            or state.get("source")
            != ("gaussian" if expected_branch == "chi" else "measured_total_field")
        ):
            raise ValueError(f"{path} is not a {expected_branch} flow prior")
        prior = FlowVelocity3D(
            base_channels=state["base_channels"],
            condition_on_field=expected_branch == "background",
        )
        prior.load_state_dict(state["model"], strict=True)
        priors.append(prior)
    return FlowADMMNet(*priors, config=config).to(device).eval()


def read_flow_manifest(path: str | Path) -> dict:
    """Read an ensemble selected for the corrected residual-weighted data term."""
    metadata = json.loads(Path(path).read_text())
    if metadata.get("format") == "whnet-flow-ensemble-v2":
        raise ValueError(
            "FlowADMM ensemble v2 used an incorrect data term; rerun "
            "whnet-flow-tune with the existing v2 prior checkpoints"
        )
    if metadata.get("format") == "whnet-flow-ensemble-v3":
        raise ValueError(
            "FlowADMM ensemble v3 used the CG solver; rerun whnet-flow-tune "
            "with the existing v2 prior checkpoints for the WH-Net closed-form solver"
        )
    if (
        metadata.get("format") != FLOW_ENSEMBLE_FORMAT
        or metadata.get("data_term") != FLOW_DATA_TERM
        or metadata.get("solver_type") != FLOW_SOLVER
    ):
        raise ValueError("invalid FlowADMM ensemble manifest")
    return metadata


def load_flow_run(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> FlowADMMNet:
    """Restore priors and solver settings selected for the corrected data term."""
    metadata = read_flow_manifest(path)
    config = FlowADMMConfig(**metadata["solver"])
    return load_flow_admm(
        metadata["chi_checkpoint"],
        metadata["background_checkpoint"],
        device=device,
        config=config,
    )

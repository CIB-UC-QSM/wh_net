"""Adaptive WH-TV model with spatial variance and scalar parameter prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import DoubleConv, UpsampleConv, _apply_spectral_norm
from .wh_tv import differentiable_wh_tv


_PARAMETER_NAMES = ("alpha1", "beta", "mu1", "mu2", "muh")


@dataclass
class WHTVParameters:
    """Positive per-sample WH-TV parameters, each shaped [N,1,1,1,1]."""

    alpha1: Tensor
    beta: Tensor
    mu1: Tensor
    mu2: Tensor
    muh: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name) for name in _PARAMETER_NAMES}


@dataclass
class AdaptiveWHTVOutput:
    chi: Tensor
    background: Tensor
    variance: Tensor
    precision: Tensor
    parameters: WHTVParameters


class WHTVParameterNet(nn.Module):
    """3D U-Net that predicts relative noise variance and WH-TV scalars."""

    LOG10_BOUNDS = (
        (-7.0, 1.0),  # alpha1
        (6.0, 10.0),  # beta
        (0.0, 5.0),  # mu1 / alpha1
        (-3.0, 3.0),  # mu2
        (-1.0, 5.0),  # muh
    )

    def __init__(
        self,
        features: tuple[int, ...] = (16, 32, 64),
        *,
        spectral_norm: bool = True,
        max_abs_log_variance: float = 2.0,
    ) -> None:
        super().__init__()
        if not features or min(features) < 1:
            raise ValueError("features must contain positive channel counts")
        if max_abs_log_variance <= 0:
            raise ValueError("max_abs_log_variance must be positive")
        self.features = tuple(features)
        self.max_abs_log_variance = float(max_abs_log_variance)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2)

        in_channels = 2
        for channels in self.features:
            self.downs.append(DoubleConv(in_channels, channels))
            in_channels = channels
        bottleneck_channels = self.features[-1] * 2
        self.bottleneck = DoubleConv(self.features[-1], bottleneck_channels)
        for channels in reversed(self.features):
            self.ups.append(UpsampleConv(channels * 2, channels))
            self.ups.append(DoubleConv(channels * 2, channels))
        self.variance_head = nn.Conv3d(self.features[0], 1, kernel_size=1)
        self.parameter_head = nn.Sequential(
            nn.Linear(bottleneck_channels + 2, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )
        self.register_buffer(
            "log10_bounds",
            torch.tensor(self.LOG10_BOUNDS, dtype=torch.float32),
            persistent=True,
        )
        nn.init.zeros_(self.variance_head.weight)
        nn.init.zeros_(self.variance_head.bias)
        nn.init.zeros_(self.parameter_head[-1].weight)
        nn.init.zeros_(self.parameter_head[-1].bias)
        if spectral_norm:
            _apply_spectral_norm(self.downs)
            _apply_spectral_norm(self.bottleneck)
            _apply_spectral_norm(self.ups)

    @staticmethod
    def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
        dimensions = tuple(range(2, value.ndim))
        count = mask.sum(dim=dimensions, keepdim=True).clamp_min(1.0)
        return (value * mask).sum(dim=dimensions, keepdim=True) / count

    def forward(
        self, phase: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, WHTVParameters]:
        if (
            phase.ndim != 5
            or phase.shape[1] != 1
            or phase.shape != mask.shape
            or min(phase.shape[-3:]) < 2 ** len(self.features)
        ):
            raise ValueError(
                "phase and mask must be matching [N,1,D,H,W] tensors large "
                "enough for every pooling level"
            )
        phase = phase.float()
        mask = mask.float()
        count = mask.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0)
        scale = (
            ((phase.square() * mask).sum(dim=(-3, -2, -1), keepdim=True) / count)
            .sqrt()
            .clamp_min(1e-6)
        )
        out = torch.cat((phase / scale, mask), dim=1)
        skips = []
        for down in self.downs:
            out = down(out)
            skips.append(out)
            out = self.pool(out)
        bottleneck = self.bottleneck(out)

        pooled_mask = F.interpolate(mask, size=bottleneck.shape[-3:], mode="nearest")
        pooled_count = pooled_mask.sum(dim=(-3, -2, -1)).clamp_min(1.0)
        pooled = (bottleneck * pooled_mask).sum(dim=(-3, -2, -1)) / pooled_count
        mask_fraction = mask.mean(dim=(-3, -2, -1))
        metadata = torch.cat((scale.flatten(1).log10(), mask_fraction), dim=1)
        # These five outputs span up to seventeen orders of magnitude. Keep
        # the small MLP and all logarithmic transforms in FP32 even when the
        # convolutional U-Net is evaluated under BF16 autocast.
        with torch.autocast(device_type=phase.device.type, enabled=False):
            logits = self.parameter_head(torch.cat((pooled, metadata), dim=1).float())
            bounds = self.log10_bounds.to(logits)
            log_values = bounds[:, 0] + logits.sigmoid() * (bounds[:, 1] - bounds[:, 0])
            alpha1 = 10.0 ** log_values[:, 0]
            beta = 10.0 ** log_values[:, 1]
            mu1 = alpha1 * (10.0 ** log_values[:, 2])
            mu2 = 10.0 ** log_values[:, 3]
            muh = 10.0 ** log_values[:, 4]
        parameters = WHTVParameters(
            *(value.reshape(-1, 1, 1, 1, 1) for value in (alpha1, beta, mu1, mu2, muh))
        )

        out = bottleneck
        for index in range(0, len(self.ups), 2):
            out = self.ups[index](out)
            skip = skips[-1 - index // 2]
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(
                    out, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            out = self.ups[index + 1](torch.cat((skip, out), dim=1))
        # Variance is consumed as a data-fidelity coefficient by a FP32
        # iterative solver, so avoid quantizing this head to BF16 as well.
        with torch.autocast(device_type=phase.device.type, enabled=False):
            log_variance = self.max_abs_log_variance * torch.tanh(
                self.variance_head(out.float())
            )
            log_variance = log_variance - self._masked_mean(log_variance, mask)
            maximum = (
                log_variance.abs().amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
            )
            log_variance = log_variance * (self.max_abs_log_variance / maximum).clamp(
                max=1.0
            )
            log_variance = log_variance * mask
            variance = log_variance.exp() * mask
            precision = (-log_variance).exp() * mask
        return variance, precision, parameters


class AdaptiveWHTV(nn.Module):
    """Predict WH-TV settings and reconstruct chi/background with 2,000 updates."""

    def __init__(
        self,
        predictor: WHTVParameterNet | None = None,
        *,
        solver_iterations: int = 2000,
        gradient_tail: int = 25,
        checkpoint_block: int = 5,
    ) -> None:
        super().__init__()
        if solver_iterations < 1:
            raise ValueError("solver_iterations must be positive")
        if not 0 <= gradient_tail <= solver_iterations:
            raise ValueError("gradient_tail must be within the solver horizon")
        self.predictor = predictor or WHTVParameterNet()
        self.solver_iterations = int(solver_iterations)
        self.gradient_tail = int(gradient_tail)
        self.checkpoint_block = int(checkpoint_block)
        self.register_buffer(
            "_adaptive_wh_tv_version", torch.tensor(1, dtype=torch.int64)
        )

    def forward(
        self,
        phase: Tensor,
        mask: Tensor,
        dipole: Tensor,
        *,
        solver_iterations: int | None = None,
        gradient_tail: int | None = None,
    ) -> AdaptiveWHTVOutput:
        variance, precision, parameters = self.predictor(phase, mask)
        result = differentiable_wh_tv(
            phase,
            dipole,
            mask,
            fidelity_precision=precision,
            max_iterations=(
                self.solver_iterations
                if solver_iterations is None
                else solver_iterations
            ),
            gradient_tail=(
                self.gradient_tail if gradient_tail is None else gradient_tail
            ),
            checkpoint_block=self.checkpoint_block,
            **parameters.as_dict(),
        )
        return AdaptiveWHTVOutput(
            result.chi,
            result.background,
            variance,
            precision,
            parameters,
        )

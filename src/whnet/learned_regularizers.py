"""Fixed-weight proximal regularizers extracted from a trained WHNet checkpoint."""

import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import ADMMUnrolledNet, ProximalNetwork


class FixedResidualProximal(nn.Module):
    """A trained WHNet U-Net residual with a fixed gain and no MLP predictor."""

    def __init__(self, source: ProximalNetwork, weight: float) -> None:
        super().__init__()
        self.downs = source.downs
        self.ups = source.ups
        self.pool = source.pool
        self.bottleneck = source.bottleneck
        self.final_conv = source.final_conv
        self.register_buffer("weight", torch.tensor(0.0, dtype=torch.float32))
        self.set_weight(weight)

    def set_weight(self, weight: float) -> None:
        """Set the nonnegative residual gain without rebuilding the module."""
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("regularizer weight must be finite and nonnegative")
        self.weight.fill_(weight)

    def forward(
        self, x: Tensor, _iteration: Tensor | float = 0, _update: Tensor | float = 0
    ) -> Tensor:
        out = x
        skips = []
        for down in self.downs:
            out = down(out)
            skips.append(out)
            out = self.pool(out)
        out = self.bottleneck(out)
        for index in range(0, len(self.ups), 2):
            out = self.ups[index](out)
            skip = skips[-1 - index // 2]
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(
                    out, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            out = self.ups[index + 1](torch.cat((skip, out), dim=1))
        return x + self.weight.to(dtype=x.dtype) * self.final_conv(out)


def extract_fixed_regularizer_model(
    checkpoint: str | Path,
    chi_weight: float,
    background_weight: float,
    *,
    device: str | torch.device = "cpu",
    num_iters: int = 55,
    features: list[int] | None = None,
    spectral_norm: bool = True,
) -> ADMMUnrolledNet:
    """Load WHNet and discard both dynamic MLPs in favor of fixed gains."""
    if num_iters < 1:
        raise ValueError("num_iters must be positive")
    target = torch.device(device)
    model = ADMMUnrolledNet(
        ProximalNetwork(features=features, spectral_norm=spectral_norm),
        ProximalNetwork(features=features, spectral_norm=spectral_norm),
        num_iters=num_iters,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.chinet = FixedResidualProximal(model.chinet, chi_weight)
    model.phinet = FixedResidualProximal(model.phinet, background_weight)
    return model.eval().requires_grad_(False).to(target)


def set_regularizer_weights(
    model: ADMMUnrolledNet, chi_weight: float, background_weight: float
) -> None:
    """Update both extracted fixed gains in place."""
    if not isinstance(model.chinet, FixedResidualProximal) or not isinstance(
        model.phinet, FixedResidualProximal
    ):
        raise TypeError("model does not contain extracted fixed regularizers")
    model.chinet.set_weight(chi_weight)
    model.phinet.set_weight(background_weight)

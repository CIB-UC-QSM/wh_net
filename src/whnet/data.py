from pathlib import Path
from tempfile import NamedTemporaryFile
import json

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from .physics import continuous_dipole_kernel, imshow_3d
from .paths import LEGACY_TRAINING_DATA_DIR
import torch.nn.functional as F
import os

SPATIAL = (-3, -2, -1)  # FFT solo sobre dimensiones espaciales
TRAINING_CONDITIONS = {
    "snr_min": 40.0,
    "snr_max": 100.0,
    "snr_definition": "masked RMS(clean local field) / masked RMS(noise)",
    "exterior_source_amplitude_min": 0.01,
    "exterior_source_amplitude_max": 20.0,
    "reliability_min": 0.1,
    "background_model": "dipole field from sources outside the 7-voxel dilated mask",
    "local_perturbation_probability": 0.1,
    "local_perturbation_count_min": 1,
    "local_perturbation_count_max": 3,
    "local_perturbation_amplitude_min": 10.0,
    "local_perturbation_amplitude_max": 100.0,
    "local_perturbation_neighbor_probability": 0.5,
}
NEIGHBOR_OFFSETS = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


class QSMDataset(Dataset):
    def __init__(self, data_dir=LEGACY_TRAINING_DATA_DIR):
        self.chi_files = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if "image" in f]
        )
        self.msk_files = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if "mask" in f]
        )
        assert len(self.chi_files) == len(self.msk_files), (
            f"Numero de volumenes 'image' ({len(self.chi_files)}) != 'mask' ({len(self.msk_files)})"
        )

        # IMPORTANTE: la convencion de continuous_dipole_kernel debe coincidir con
        # torch.fft (DC en la esquina, indice [0,0,0]). Si el kernel viene CENTRADO,
        # usar:  D = torch.fft.ifftshift(D, dim=(-3,-2,-1))
        # El bloque __main__ incluye un test de harmonicidad para verificarlo.
        self.D = (
            torch.from_numpy(continuous_dipole_kernel((160, 160, 160)))
            .unsqueeze(0)
            .float()
        )

    def __len__(self):
        return len(self.chi_files)

    def __getitem__(self, idx):
        chi = torch.load(self.chi_files[idx]).float() * torch.empty(1).uniform_(0.5, 3)
        msk = torch.load(self.msk_files[idx]).float()
        perm = torch.randperm(3)
        dims = (0, int(perm[0]) + 1, int(perm[1]) + 1, int(perm[2]) + 1)
        chi = chi.permute(dims).contiguous()
        msk = msk.permute(dims).contiguous()

        flip_dims = [d for d in (1, 2, 3) if torch.rand(1).item() < 0.5]
        if flip_dims:
            chi = torch.flip(chi, dims=flip_dims).contiguous()
            msk = torch.flip(msk, dims=flip_dims).contiguous()

        rval = torch.rand(1).item()
        if rval > 0.0:
            W = msk.clone()
        elif rval > 0.25:
            W = chi.clone()
            W = W - W.min()
            W = (W / W.max()) * msk
        else:
            W = chi.clone() * -1
            W = W - W.min()
            W = (W / W.max()) * msk

        chi_k = torch.fft.fftn(chi * msk, dim=SPATIAL)
        local = torch.real(torch.fft.ifftn(self.D * chi_k, dim=SPATIAL)) * msk

        msk_dil = F.max_pool3d(
            msk.unsqueeze(0), kernel_size=7, stride=1, padding=3
        ).squeeze(0)
        scale = torch.empty(1).uniform_(0.01, 20).item()
        ext_k = torch.fft.fftn((1.0 - msk_dil) * scale, dim=SPATIAL)
        phi = torch.real(torch.fft.ifftn(self.D * ext_k, dim=SPATIAL)) * msk

        inside = msk > 0
        if inside.sum() > 1:
            field_std = torch.max(local[inside].abs())
        else:
            field_std = torch.tensor(0.0)
        snr = torch.empty(1).uniform_(40, 100).item()
        noise_std = field_std / snr
        phase = msk * (local + torch.randn_like(local) * noise_std + phi)

        return phase, msk, W, self.D, chi, phi


if __name__ == "__main__":
    dataset = QSMDataset()

    # --- Auto-diagnostico de convencion del kernel / harmonicidad del fondo ---
    # Si el fondo phi es realmente armonico, |lap(phi)|/|phi| en el interior ~ 0.
    # Un valor grande (> ~0.3) sugiere que el kernel necesita ifftshift.
    # phase, msk, W, D, chi, phi = dataset[0]
    # lap_k = torch.zeros(1, 1, 3, 3, 3)
    # lap_k[0, 0, 1, 1, 1] = -6
    # for d in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
    #     lap_k[0, 0, d[0], d[1], d[2]] = 1
    # interior = (F.max_pool3d((1 - msk).unsqueeze(0), 9, 1, 4).squeeze(0) < 0.5)
    # lap = F.conv3d(phi.unsqueeze(0), lap_k, padding=1).squeeze(0)
    # if interior.sum() > 0:
    #     rel = (lap[interior].abs().mean() / (phi[interior].abs().mean() + 1e-9)).item()
    #     print(f"[diagnostico] |lap(phi)|/|phi| interior = {rel:.3e}  "
    #           f"({'OK, armonico' if rel < 0.3 else 'ALTO -> revisar convencion del kernel (ifftshift?)'})")
    # print(f"[escala] std(phase_total)={torch.std(phase[msk>0]).item():.4f} | "
    #       f"std(phi_fondo)={torch.std(phi[msk>0]).item():.4f}  (verificar que el fondo no domine excesivamente)")

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, pin_memory=torch.cuda.is_available()
    )
    for phase, mask, W, D, chi_gt, phi_gt in dataloader:
        imshow_3d(
            phase.numpy()[0, 0], "phase", rango=(-0.1, 0.1), angles=(-90, -90, 90)
        )
        imshow_3d(mask.numpy()[0, 0], "mask", rango=(0, 1), angles=(-90, -90, 90))
        imshow_3d(W.numpy()[0, 0], "W", rango=(0, 1), angles=(-90, -90, 90))
        imshow_3d(
            chi_gt.numpy()[0, 0], "chi_gt", rango=(-0.1, 0.1), angles=(-90, -90, 90)
        )
        imshow_3d(
            phi_gt.numpy()[0, 0], "phi_gt", rango=(-0.1, 0.1), angles=(-90, -90, 90)
        )
        break
# %%


# The legacy QSMDataset remains available for old notebooks and diagnostics.
# Joint training loads compact source volumes and synthesizes fields on-device.


def separable_dilation(mask):
    """Exactly the same binary dilation as max_pool3d(kernel_size=7)."""
    x = mask.float().unsqueeze(0)
    for kernel, padding in [
        ((7, 1, 1), (3, 0, 0)),
        ((1, 7, 1), (0, 3, 0)),
        ((1, 1, 7), (0, 0, 3)),
    ]:
        x = F.max_pool3d(x, kernel, stride=1, padding=padding)
    return x.squeeze(0).bool()


def training_records(data_dir, source_manifest=None):
    """Return explicit source identities; never silently split augmented files."""
    root = Path(data_dir)
    manifest_path = Path(source_manifest) if source_manifest else root / "subjects.json"
    identities = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    )
    if source_manifest and identities is None:
        raise FileNotFoundError(manifest_path)
    records = []
    for image in sorted(root.glob("image_*.pt")):
        index = int(image.stem.split("_")[-1])
        mask = root / f"mask_{index:04d}.pt"
        if not mask.is_file():
            raise FileNotFoundError(mask)
        if identities is not None:
            subject = identities[image.name]
        elif root.name in {"train_data2", "joint"} and 0 <= index < 651:
            # 0:315 mapping confirmed by the user; 315:651 from gen_data.py.
            subject = (
                f"qsm:{index % 105:04d}"
                if index < 315
                else f"separate:{(index - 315) % 56:04d}"
            )
        else:
            raise ValueError(
                "This dataset needs a JSON filename-to-subject source manifest"
            )
        records.append({"image": str(image), "mask": str(mask), "subject": subject})
    if not records:
        raise ValueError(f"No training volumes in {root}")
    return records


def _sample_local_perturbations(
    mask: torch.Tensor, generator: torch.Generator
) -> list[tuple[tuple[int, int, int], float]]:
    """Draw sparse local-field impulses using a per-sample CPU random stream."""
    conditions = TRAINING_CONDITIONS
    if (
        torch.rand((), generator=generator).item()
        >= conditions["local_perturbation_probability"]
    ):
        return []
    paired = (
        torch.rand((), generator=generator).item()
        < conditions["local_perturbation_neighbor_probability"]
    )
    mask = mask.bool()
    candidates = mask
    if paired:
        candidates = torch.zeros_like(mask)
        for axis in range(3):
            before = [slice(None)] * 3
            after = [slice(None)] * 3
            before[axis] = slice(None, -1)
            after[axis] = slice(1, None)
            before, after = tuple(before), tuple(after)
            adjacent = mask[before] & mask[after]
            candidates[before] |= adjacent
            candidates[after] |= adjacent
    positions = candidates.nonzero(as_tuple=False)
    if positions.numel() == 0:
        raise ValueError("Local-field perturbation needs a masked voxel pair")
    count = int(
        torch.randint(
            conditions["local_perturbation_count_min"],
            conditions["local_perturbation_count_max"] + 1,
            (),
            generator=generator,
        )
    )
    perturbations = []
    for _ in range(count):
        position = tuple(
            positions[
                int(torch.randint(len(positions), (), generator=generator))
            ].tolist()
        )
        amplitude = (
            conditions["local_perturbation_amplitude_min"]
            + (
                conditions["local_perturbation_amplitude_max"]
                - conditions["local_perturbation_amplitude_min"]
            )
            * torch.rand((), generator=generator).item()
        )
        perturbations.append((position, amplitude))
        if paired:
            neighbors = []
            for offset in NEIGHBOR_OFFSETS:
                neighbor = tuple(position[axis] + offset[axis] for axis in range(3))
                if all(
                    0 <= neighbor[axis] < mask.shape[axis] for axis in range(3)
                ) and bool(mask[neighbor]):
                    neighbors.append(neighbor)
            neighbor = neighbors[
                int(torch.randint(len(neighbors), (), generator=generator))
            ]
            perturbations.append((neighbor, -amplitude))
    return perturbations


def sample_target_chi_rms(generator: torch.Generator) -> float:
    """Center physical-unit RMS near both references, with occasional strong cases."""
    tail = bool(torch.rand((), generator=generator) < 0.05)
    low, high = (0.04, 0.08) if tail else (0.018, 0.04)
    fraction = float(torch.rand((), generator=generator))
    return low * (high / low) ** fraction


class TrainingVolumes(Dataset):
    """Cache FP32 chi and boolean masks without modifying original data files."""

    def __init__(
        self,
        records,
        cache_dir,
        *,
        validation=False,
        views=1,
        seed=0,
        chi_scale_mode="legacy",
    ):
        self.records = records
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.validation, self.views, self.seed = validation, views, seed
        if chi_scale_mode not in ("legacy", "rms_mixture"):
            raise ValueError(f"Unknown chi scale mode: {chi_scale_mode}")
        self.chi_scale_mode = chi_scale_mode
        import multiprocessing

        self.epoch = multiprocessing.Value("q", 0)

    def __len__(self):
        return len(self.records) * self.views

    def set_epoch(self, epoch):
        self.epoch.value = epoch

    def _load(self, record):
        import hashlib

        paths = [Path(record[key]) for key in ("image", "mask")]
        signature = repr(
            [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        )
        key = hashlib.sha256(signature.encode()).hexdigest()[:20]
        path = self.cache_dir / f"{paths[0].stem}_{key}.pt"
        if path.is_file():
            return torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        chi = torch.load(paths[0], weights_only=True, map_location="cpu").float()
        mask = torch.load(paths[1], weights_only=True, map_location="cpu").bool()
        if chi.shape != mask.shape or chi.ndim != 4 or chi.shape[0] != 1:
            raise ValueError(f"Invalid image/mask shapes for {paths[0]}")
        if not bool(mask.any()) or not bool(torch.isfinite(chi).all()):
            raise ValueError(f"Empty mask or nonfinite source: {paths[0]}")
        result = {"chi": chi, "mask": mask, "dilated": separable_dilation(mask)}
        with NamedTemporaryFile(
            dir=self.cache_dir, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(result, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return result

    def __getitem__(self, index):
        # Training samplers may encode a repeated view above len(records).
        record = self.records[
            index // self.views if self.validation else index % len(self.records)
        ]
        epoch_stride = len(self) if self.chi_scale_mode == "legacy" else len(self) ** 2
        sample_seed = (
            self.seed
            + index
            + (0 if self.validation else self.epoch.value * epoch_stride)
        )
        generator = torch.Generator().manual_seed(sample_seed)
        loaded = self._load(record)
        shape = loaded["chi"].shape[-3:]
        # Axis swaps are valid here only for cubic, isotropic training volumes.
        permutation = (
            torch.randperm(3, generator=generator).tolist()
            if len(set(shape)) == 1
            else [0, 1, 2]
        )
        dims = (0, *(axis + 1 for axis in permutation))
        flips = tuple(
            axis for axis in (1, 2, 3) if torch.rand((), generator=generator) < 0.5
        )
        result = {}
        for key, volume in loaded.items():
            volume = volume.permute(dims)
            if flips:
                volume = volume.flip(flips)
            result[key] = volume.contiguous()
        if self.chi_scale_mode == "legacy":
            scale = float(torch.empty(()).uniform_(0.5, 3, generator=generator))
        else:
            chi_rms = (
                (result["chi"].square() * result["mask"]).sum() / result["mask"].sum()
            ).sqrt()
            scale = sample_target_chi_rms(generator) / max(float(chi_rms), 1e-8)
        result["chi"] = result["chi"] * scale
        result.update(seed=sample_seed, subject=record["subject"])
        return result


class FieldSynthesizer(torch.nn.Module):
    """Real-FFT dipole simulation, diverse exterior sources and RMS-defined SNR."""

    def __init__(self, shape, device):
        super().__init__()
        kernel = continuous_dipole_kernel(tuple(shape))
        self.register_buffer(
            "dipole",
            torch.from_numpy(kernel[..., : shape[-1] // 2 + 1].copy())
            .float()[None, None]
            .to(device),
        )
        self.shape = tuple(shape)

    @torch.no_grad()
    def forward(
        self,
        batch,
        *,
        perturb_local_field: bool = False,
        background_mode: str = "legacy",
        return_diagnostics: bool = False,
    ):
        if background_mode not in ("legacy", "ratio"):
            raise ValueError(f"Unknown background mode: {background_mode}")
        device = self.dipole.device
        chi = batch["chi"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()
        dilated = batch["dilated"].to(device, non_blocking=True)
        chi = chi * mask
        source_fields, weights, noises, snrs, ratios = [], [], [], [], []
        for index, seed in enumerate(batch["seed"].tolist()):
            # Independent per-sample streams make validation invariant to batch size.
            generator = torch.Generator(device=device).manual_seed(seed + 1000003)

            def rand(*shape):
                return torch.rand(shape, device=device, generator=generator)

            smooth = F.interpolate(
                rand(1, 1, 5, 5, 5),
                size=self.shape,
                mode="trilinear",
                align_corners=False,
            )[0]
            selector = rand()
            uniform = rand() < 0.5
            exterior = torch.where(uniform, torch.ones_like(smooth), 2.0 * smooth - 1.0)
            amplitude = (
                TRAINING_CONDITIONS["exterior_source_amplitude_min"]
                + (
                    TRAINING_CONDITIONS["exterior_source_amplitude_max"]
                    - TRAINING_CONDITIONS["exterior_source_amplitude_min"]
                )
                * rand()
            )
            if background_mode == "ratio":
                ratios.append(0.03 * (3.0 / 0.03) ** float(rand()))
            source_fields.append(exterior * (~dilated[index]) * amplitude)
            signal = chi[index]
            low = signal.masked_fill(mask[index] == 0, torch.inf).amin()
            high = signal.masked_fill(mask[index] == 0, -torch.inf).amax()
            contrast = ((signal - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
            reliability = torch.where(
                selector < 0.5,
                torch.ones_like(smooth),
                torch.where(selector < 0.75, smooth, contrast),
            )
            reliability_floor = TRAINING_CONDITIONS["reliability_min"]
            reliability = (
                reliability_floor + (1.0 - reliability_floor) * reliability
            ) * mask[index]
            weights.append(reliability)
            noises.append(
                torch.randn(signal.shape, device=device, generator=generator)
                / reliability.clamp_min(reliability_floor).sqrt()
            )
            snrs.append(
                TRAINING_CONDITIONS["snr_min"]
                + (TRAINING_CONDITIONS["snr_max"] - TRAINING_CONDITIONS["snr_min"])
                * rand()
            )

        def inverse(x):
            return torch.fft.irfftn(x, s=self.shape, dim=SPATIAL)

        local = inverse(self.dipole * torch.fft.rfftn(chi, dim=SPATIAL)) * mask
        background = (
            inverse(
                self.dipole * torch.fft.rfftn(torch.stack(source_fields), dim=SPATIAL)
            )
            * mask
        )
        count = mask.sum(dim=SPATIAL, keepdim=True).clamp_min(1)

        def rms(x):
            return (
                ((x.square() * mask).sum(dim=SPATIAL, keepdim=True) / count)
                .sqrt()
                .clamp_min(1e-8)
            )

        if background_mode == "ratio":
            target = torch.tensor(ratios, device=device).reshape(-1, 1, 1, 1, 1)
            background = background * rms(local) / (rms(background) * target)

        noise = torch.stack(noises) * mask
        noise = (
            noise
            * rms(local)
            / (rms(noise) * torch.stack(snrs).reshape(-1, 1, 1, 1, 1))
        )
        if perturb_local_field:
            for index, seed in enumerate(batch["seed"].tolist()):
                generator = torch.Generator().manual_seed(seed + 2000003)
                for position, amplitude in _sample_local_perturbations(
                    batch["mask"][index, 0], generator
                ):
                    local[(index, 0, *position)] *= amplitude
        phase = (local + background + noise) * mask
        result = (phase, mask, torch.stack(weights), self.dipole, chi, background)
        if return_diagnostics:
            diagnostics = {
                "chi_rms": rms(chi).flatten(),
                "local_rms": rms(local).flatten(),
                "background_rms": rms(background).flatten(),
                "phase_rms": rms(phase).flatten(),
                "local_to_background": (rms(local) / rms(background)).flatten(),
            }
            return (*result, diagnostics)
        return result

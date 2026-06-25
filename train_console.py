#%%
from comet_ml import Experiment
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from wh_net import ProximalNetwork, ADMMUnrolledNet
from dataset import QSMDataset

# ----------------------------- Configuracion -----------------------------
# --- Entrenamiento DESDE CERO (sin warm-start) ---
INIT_CKPT       = 'checkpoints_scratch4/model_best.pth'               # None = desde cero. La normalizacion espectral
                                        # (parametrizations.weight.original + buffers _u/_v)
                                        # y el MLP de rho cambian el state_dict: los
                                        # checkpoints previos NO cargan con strict=True.
CKPT_DIR        = "checkpoints_scratch4" # carpeta para esta corrida

EPOCHS          = 300
BATCH_SIZE      = 5
PEAK_LR         = 1e-4      # subido: 0.5e-4 era demasiado bajo para mover el warm-start
MIN_LR_FACTOR   = 0.1       # LR final = PEAK_LR * MIN_LR_FACTOR (coseno)
WARMUP_EPOCHS   = 2         # warmup corto: evita que el primer paso desestabilice el warm-start
WH_WARMUP_EPOCHS = 5        # mas largo: el modelo aun no aprendio el termino WH
LAM_WH          = 1000.0      # BAJADO: para ENFATIZAR chi, lam_chi debe dominar a lam_wh

# --- Iteraciones del ADMM desenrollado (NIVEL 1: robustez a la profundidad) ---
NUM_ITERS_MAX   = 20        # objetivo de produccion (referencia)
ITER_START      = 10        # warm-start de un modelo ya entrenado a 30 -> se mantiene 30
ITER_MIN        = 10        # cota INFERIOR del muestreo aleatorio por batch
ITER_SAMPLE_MAX = 20        # cota SUPERIOR del muestreo: un poco MAS ALLA del objetivo (30)
ITER_RAMP_EPOCHS = 10       # epocas para subir la cota superior de ITER_START a ITER_SAMPLE_MAX
DEEP_SUP_K      = 10         # nº de iterados intermedios supervisados (supervision profunda)

# --- Inferencia / produccion: iteracion adaptativa con criterio de parada ---
EVAL_TOL        = 1e-3      # parar cuando ||chi_k - chi_{k-1}|| / ||chi_k|| < tol
EVAL_MAX_ITERS  = 25        # tope de seguridad (puede exceder NUM_ITERS_MAX)

GRAD_CLIP       = 1.0       # el clip ACOTA el paso de los batches dificiles (en vez de descartarlos)
EMA_DECAY       = 0.9     # promedio movil exponencial de peso
VAL_FRAC        = 0.1       # fraccion para validacion (split por volumen)
SEED            = 0
# --------------------------------------------------------------------------

torch.backends.cudnn.benchmark = True


_LAP_KERNEL = torch.tensor(
    [[[
        [[1/13, 3/26, 1/13], [3/26, 3/13, 3/26], [1/13, 3/26, 1/13]],
        [[3/26, 3/13, 3/26], [3/13, -44/13, 3/13], [3/26, 3/13, 3/26]],
        [[1/13, 3/26, 1/13], [3/26, 3/13, 3/26], [1/13, 3/26, 1/13]],
    ]]]
)


def calculate_nrmse(pred, target):
    return torch.norm(pred.flatten() - target.flatten()) / torch.norm(target.flatten())


def spatial_gradient_3d(x):
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return dx, dy, dz


def gradient_loss(pred, target, mask):
    px, py, pz = spatial_gradient_3d(pred)
    tx, ty, tz = spatial_gradient_3d(target)
    mask_x = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
    mask_y = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
    mask_z = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]
    return (F.l1_loss(px * mask_x, tx * mask_x)
            + F.l1_loss(py * mask_y, ty * mask_y)
            + F.l1_loss(pz * mask_z, tz * mask_z))


def weak_harmonic_loss(phi, mask):
    kernel = _LAP_KERNEL.to(device=phi.device, dtype=phi.dtype)
    laplacian_phi = F.conv3d(phi, kernel, padding=1)
    return F.smooth_l1_loss(mask * laplacian_phi, torch.zeros_like(laplacian_phi), beta=0.1)


def hybrid_qsm_loss(chi_pred, chi_gt, phi_pred, phi_gt, mask,
                    lam_chi=1500.0, lam_phi=1.0, lam_grad=1.0, lam_wh=10.0):
    loss_chi = F.l1_loss(chi_pred * mask, chi_gt * mask)
    loss_phi = F.l1_loss(phi_pred * mask, phi_gt * mask)
    loss_grad = gradient_loss(chi_pred, chi_gt, mask)
    loss_wh = weak_harmonic_loss(phi_pred, mask)
    total = lam_chi * loss_chi + lam_phi * loss_phi + lam_grad * loss_grad + lam_wh * loss_wh
    return total, {"loss_chi": loss_chi.item(), "loss_phi": loss_phi.item(),
                   "loss_grad": loss_grad.item(), "loss_wh": loss_wh.item()}


def deep_supervised_loss(iterates, chi_gt, phi_gt, mask, lam_wh):
    """
    Supervision profunda: penaliza un subconjunto de iterados intermedios, no solo
    el ultimo, para que CUALQUIER profundidad produzca una salida razonable. Se eligen
    DEEP_SUP_K iterados equiespaciados (incluyendo el ultimo) con pesos crecientes
    (los tardios pesan mas) normalizados a suma 1, para no alterar la escala de la
    perdida ni el ajuste del LR.
    """
    K = len(iterates)
    if K <= DEEP_SUP_K:
        idxs = list(range(K))
    else:
        idxs = sorted(set(torch.linspace(0, K - 1, DEEP_SUP_K).round().int().tolist()))
    w = [i + 1 for i in range(len(idxs))]          # pesos crecientes por posicion
    s = float(sum(w))
    total, parts_last = 0.0, None
    for wi, idx in zip(w, idxs):
        chi_k, phi_k = iterates[idx]
        loss_k, parts = hybrid_qsm_loss(chi_k, chi_gt, phi_k, phi_gt, mask, lam_wh=lam_wh)
        total = total + (wi / s) * loss_k
        parts_last = parts                          # parts del iterado mas profundo
    return total, parts_last


class EMA:
    """Promedio movil exponencial de los PARAMETROS. Los buffers (p. ej. los
    vectores _u/_v de la iteracion de potencia de la normalizacion espectral) NO se
    promedian: se copian los ultimos. Promediar esos vectores los saca de norma
    unitaria y descalibra la estimacion de sigma -> al validar con los pesos EMA el
    peso normalizado quedaria mal escalado. InstanceNorm3d aqui no lleva running
    stats (track_running_stats=False), asi que no hay buffers de norm que promediar."""
    def __init__(self, model, decay):
        self.decay = decay
        self.param_names = {n for n, _ in model.named_parameters()}
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.param_names and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)   # buffers / no-flotantes: copia directa


def log_ortho_slices(experiment, vol, name, epoch, rango=(-0.1, 0.1)):
    d, h, w = vol.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, sl in zip(axes, [vol[d // 2], vol[:, h // 2], vol[:, :, w // 2]]):
        ax.imshow(sl, cmap='gray', vmin=rango[0], vmax=rango[1]); ax.axis('off')
    plt.tight_layout()
    experiment.log_figure(figure_name=f"{name}_epoch_{epoch}", figure=fig)
    plt.close(fig)


def iter_upper_for_epoch(epoch):
    """Cota superior del muestreo: rampa lineal de ITER_START a ITER_SAMPLE_MAX."""
    frac = min(1.0, epoch / max(1, ITER_RAMP_EPOCHS))
    return int(round(ITER_START + (ITER_SAMPLE_MAX - ITER_START) * frac))


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluacion como en produccion: iteracion adaptativa hasta tolerancia EVAL_TOL."""
    model.eval()
    nch, nph, n = 0.0, 0.0, 0
    for phase, mask, W, D, chi_gt, phi_gt in loader:
        phase, mask, W, D = phase.to(device), mask.to(device), W.to(device), D.to(device)
        chi_gt, phi_gt = chi_gt.to(device), phi_gt.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            chi_pred, phi_pred = model(phase, mask, D, W,
                                       tol=EVAL_TOL, max_iters=EVAL_MAX_ITERS)
        nch += calculate_nrmse(chi_pred * mask, chi_gt * mask).item()
        nph += calculate_nrmse(phi_pred * mask, phi_gt * mask).item()
        n += 1
    model.train()
    return nch / max(1, n), nph / max(1, n)


if __name__ == "__main__":
    experiment = Experiment(project_name="wh-net-qsm")
    experiment.log_parameters({
        "mode": "from_scratch", "init_ckpt": str(INIT_CKPT), "ckpt_dir": CKPT_DIR,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "peak_lr": PEAK_LR,
        "warmup_epochs": WARMUP_EPOCHS, "wh_warmup_epochs": WH_WARMUP_EPOCHS,
        "lam_wh": LAM_WH, "grad_clip": GRAD_CLIP, "ema_decay": EMA_DECAY,
        "num_iters_max": NUM_ITERS_MAX, "iter_start": ITER_START, "iter_min": ITER_MIN,
        "iter_sample_max": ITER_SAMPLE_MAX, "iter_ramp_epochs": ITER_RAMP_EPOCHS,
        "deep_sup_k": DEEP_SUP_K, "eval_tol": EVAL_TOL, "eval_max_iters": EVAL_MAX_ITERS,
    })

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net_chi = ProximalNetwork().to(device)
    net_phi = ProximalNetwork().to(device)
    model = ADMMUnrolledNet(net_chi, net_phi, num_iters=ITER_START).to(device)

    if INIT_CKPT is not None:
        assert os.path.isfile(INIT_CKPT), f"No se encontro el checkpoint inicial: {INIT_CKPT}"
        model.load_state_dict(torch.load(INIT_CKPT, map_location=device), strict=True)
        print(f"Pesos iniciales cargados desde {INIT_CKPT}")
    else:
        print("Entrenamiento desde cero (sin warm-start)")

    optimizer = optim.Adam(model.parameters(), lr=PEAK_LR)

    full = QSMDataset()
    n_val = max(1, int(len(full) * VAL_FRAC))
    n_train = len(full) - n_val
    train_set, val_set = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(SEED))
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=5, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=5, pin_memory=pin)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return MIN_LR_FACTOR + (1 - MIN_LR_FACTOR) * 0.5 * (1 + math.cos(math.pi * progress / 2))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(model, EMA_DECAY)

    global_step = 0
    best_val = float("inf")
    model.train()

    for epoch in range(EPOCHS):
        wh_factor = min(1.0, epoch / max(1, WH_WARMUP_EPOCHS))
        k_hi = iter_upper_for_epoch(epoch)   # cota superior del muestreo en esta epoca
        skipped = 0                          # batches realmente descartados (loss NaN/Inf)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (iters<={k_hi})")
        for phase, mask, W, D, chi_gt, phi_gt in pbar:
            model.num_iters = int(torch.randint(ITER_MIN, k_hi + 1, (1,)).item())

            phase, mask, W, D = phase.to(device), mask.to(device), W.to(device), D.to(device)
            chi_gt, phi_gt = chi_gt.to(device), phi_gt.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.bfloat16):
                iterates = model(phase, mask, D, W, return_iterates=True)
                loss, parts = deep_supervised_loss(iterates, chi_gt, phi_gt, mask,
                                                   lam_wh=LAM_WH * wh_factor)
                chi_pred, phi_pred = iterates[-1]   # salida final (para metricas)


            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            for p in model.parameters():        # sanear gradientes raros -> no descartar el batch
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()
            scheduler.step()
            ema.update(model)

            with torch.no_grad():
                nrmse_chi = calculate_nrmse(chi_pred * mask, chi_gt * mask)
                nrmse_phi = calculate_nrmse(phi_pred * mask, phi_gt * mask)

            experiment.log_metric("loss", loss.item(), step=global_step)
            experiment.log_metric("nrmse_chi", nrmse_chi.item(), step=global_step)
            experiment.log_metric("nrmse_phi", nrmse_phi.item(), step=global_step)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=global_step)
            experiment.log_metric("grad_norm", grad_norm.item(), step=global_step)
            experiment.log_metric("num_iters", model.num_iters, step=global_step)

            for k, v in parts.items():
                experiment.log_metric(k, v, step=global_step)

            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             nrmse_chi=f"{nrmse_chi.item():.3f}",
                             nrmse_phi=f"{nrmse_phi.item():.3f}",
                             it=model.num_iters,
                             lr=f"{scheduler.get_last_lr()[0]:.1e}")
            global_step += 1

        experiment.log_metric("skipped_frac", skipped / max(1, steps_per_epoch), step=global_step)

        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.shadow, strict=True)
        val_chi, val_phi = evaluate(model, val_loader, device)
        model.load_state_dict(backup, strict=True)

        experiment.log_metric("val_nrmse_chi", val_chi, step=global_step)
        experiment.log_metric("val_nrmse_phi", val_phi, step=global_step)
        experiment.log_metric("wh_factor", wh_factor, step=global_step)

        score = val_chi + val_phi
        if score < best_val:
            best_val = score
            torch.save(ema.shadow, os.path.join(CKPT_DIR, "model_best.pth"))
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "model_last.pth"))
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"model_epoch_{epoch+1}.pth"))

        if (epoch + 1) % 5 == 0:
            model.load_state_dict(ema.shadow, strict=True)
            model.eval()
            with torch.no_grad():
                chi_pred, phi_pred = model(phase, mask, D, W,
                                           tol=EVAL_TOL, max_iters=EVAL_MAX_ITERS)
            for vol, name in [(phase, 'phase_in'), (chi_pred, 'chi_pred'),
                              (phi_pred, 'phih_pred'), (chi_gt, 'chi_gt'), (phi_gt, 'phih_gt')]:
                log_ortho_slices(experiment, (vol*mask).detach().cpu().numpy()[0, 0], name, epoch + 1)
            model.load_state_dict(backup, strict=True)
            model.train()

    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "model_best.pth")), strict=True)
    val_chi, val_phi = evaluate(model, val_loader, device)
    print(f"[BEST EMA] Val NRMSE Chi: {val_chi:.4f} | Val NRMSE Phi: {val_phi:.4f}")
    experiment.end()

# %%

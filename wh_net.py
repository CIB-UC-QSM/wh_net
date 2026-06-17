import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# En tensores 5D (B, C, X, Y, Z) la FFT debe aplicarse SOLO a las dimensiones
# espaciales. Aplicarla sobre todo el tensor (default de torch.fft.fftn)
# transforma tambien el batch y mezcla las muestras: bug silencioso.
SPATIAL_DIMS = (-3, -2, -1)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv3d(in_channels, out_channels),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU(),
            DepthwiseSeparableConv3d(out_channels, out_channels),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DepthwiseSeparableConv3d(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='trilinear', align_corners=False)
        return self.conv(x)


class ProximalNetwork(nn.Module):
    """
    Operador proximal aprendido. U-Net residual 3D condicionada en el peso ADMM
    'rho' (le indica a la red la "intensidad" del paso, equivalente al nivel de
    ruido en un denoiser). La salida esta inicializada a cero (final_conv + alpha),
    de modo que la red empieza siendo la identidad: inicializacion estable para
    optimizacion desenrollada.
    """
    def __init__(self, in_channels=1, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [16, 32, 64]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # alpha=1 (NO 0): con final_conv inicializado en cero la red ya empieza como
        # la identidad (x + 1*0 = x), pero el gradiente SI fluye a las convoluciones.
        # Si se inicializa alpha=0, alpha*final_conv=0 bloquea el gradiente de TODO
        # el prox y la red nunca aprende.
        self.alpha = nn.Parameter(torch.ones(1))

        in_c = in_channels + 1  # +1 por el canal de condicionamiento en rho
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(UpsampleConv(feature * 2, feature))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, rho):
        b, _, d, h, w = x.shape
        rho_map = rho.reshape(1, 1, 1, 1, 1).expand(b, 1, d, h, w).to(x.dtype)
        out = torch.cat([x, rho_map], dim=1)

        skip_connections = []
        for down in self.downs:
            out = down(out)
            skip_connections.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            out = self.ups[i](out)
            skip = skip_connections[i // 2]
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:],
                                    mode='trilinear', align_corners=False)
            out = self.ups[i + 1](torch.cat((skip, out), dim=1))

        residual = self.final_conv(out)
        return x + self.alpha * residual


class ADMMUnrolledNet(nn.Module):
    """
    ADMM desenrollado para remocion de campo de fondo (weak-harmonic) y estimacion
    de susceptibilidad de forma conjunta. Resuelve el MISMO problema que WH_wTV.m,
    pero reemplazando los regularizadores analiticos por operadores proximales
    aprendidos:

        min_{chi, phi_h}  1/2 || sqrt(W) (D*chi + phi_h - phi) ||^2
                          + R_chi(chi)       (en WH_wTV: alpha * TV(chi))
                          + R_phi(phi_h)     (en WH_wTV: beta/2 ||M . lap(phi_h)||^2)

      D     : kernel del dipolo en espacio-k (real, par)
      chi   : mapa de susceptibilidad
      phi_h : campo armonico residual (fondo)
      W     : peso de fidelidad espacialmente variable (~ magnitud^2)

    Splitting (ADMM dual escalado), igual estructura que WH_wTV:
      y : consistencia del campo total   (D*chi + phi_h = y)  -> fidelidad de datos
      u : consistencia de chi            (chi   = u)          -> prox aprendido (sustituye TV)
      v : consistencia de phi_h          (phi_h = v)          -> prox aprendido (sustituye weak-harmonic)

    Los sub-problemas en chi y phi_h son cuadraticos y se resuelven en forma
    cerrada en espacio-k, igual que en WH_wTV.
    """
    def __init__(self, chinet, phinet, num_iters=5, eps=1e-6, mask_chi=False):
        super().__init__()
        self.chinet = chinet
        self.phinet = phinet
        self.num_iters = num_iters
        self.eps = eps
        # Si True, fuerza chi=0 fuera de la mascara en cada iteracion, igual que
        # WH_wTV (x = mask.*real(...)). La susceptibilidad solo esta definida en tejido.
        self.mask_chi = mask_chi
        # Pesos ADMM (mu2, mu_chi, mu_phi). softplus garantiza positividad estricta.
        self.rho_y = nn.Parameter(torch.tensor(1.0))
        self.rho_u = nn.Parameter(torch.tensor(1.0))
        self.rho_v = nn.Parameter(torch.tensor(1.0))

    def step_fn(self, chi, phi_h, y, u, v, eta_y, eta_u, eta_v, phi, mask, W, D):
        rho_y = F.softplus(self.rho_y)
        rho_u = F.softplus(self.rho_u)
        rho_v = F.softplus(self.rho_v)

        # Campo directo actual: D*chi + phi_h
        chi_fft = torch.fft.fftn(chi, dim=SPATIAL_DIMS)
        forward_physics = torch.fft.ifftn(D * chi_fft, dim=SPATIAL_DIMS).real
        total_field = forward_physics + phi_h

        # --- y: proximal de fidelidad de datos ponderada (forma cerrada) ---
        #   min_y 1/2||sqrt(W)(y-phi)||^2 + rho_y/2 ||total_field + eta_y - y||^2
        y = (W * phi + rho_y * (total_field + eta_y)) / (W + rho_y + self.eps)

        # --- pasos proximales aprendidos (reemplazan regularizadores analiticos) ---
        u = self.chinet(chi + eta_u, rho_u).float()    # prox de chi  (antes: TV)
        v = self.phinet(phi_h + eta_v, rho_v).float()  # prox de phi_h (antes: weak-harmonic)

        # --- chi: solucion cuadratica en espacio-k ---
        #   (rho_y D^H D + rho_u I) chi = rho_y D^H (y - eta_y - phi_h) + rho_u (u - eta_u)
        rhs_chi = (rho_y * torch.conj(D) * torch.fft.fftn(y - eta_y - phi_h, dim=SPATIAL_DIMS)
                   + rho_u * torch.fft.fftn(u - eta_u, dim=SPATIAL_DIMS))
        denom_chi = rho_y * torch.abs(D) ** 2 + rho_u + self.eps
        chi_fft = rhs_chi / denom_chi
        chi = torch.fft.ifftn(chi_fft, dim=SPATIAL_DIMS).real
        if self.mask_chi:
            chi = chi * mask
            chi_fft = torch.fft.fftn(chi, dim=SPATIAL_DIMS)

        # --- phi_h: solucion cuadratica en espacio-k ---
        #   (rho_y I + rho_v I) phi_h = rho_y (y - eta_y - D*chi) + rho_v (v - eta_v)
        forward_physics = torch.fft.ifftn(D * chi_fft, dim=SPATIAL_DIMS).real
        rhs_phi = (rho_y * torch.fft.fftn(y - eta_y - forward_physics, dim=SPATIAL_DIMS)
                   + rho_v * torch.fft.fftn(v - eta_v, dim=SPATIAL_DIMS))
        denom_phi = rho_y + rho_v + self.eps
        phi_h = torch.fft.ifftn(rhs_phi / denom_phi, dim=SPATIAL_DIMS).real

        # --- actualizacion de variables duales ---
        total_field = forward_physics + phi_h
        eta_y = eta_y + total_field - y
        eta_u = eta_u + chi - u
        eta_v = eta_v + phi_h - v

        return chi, phi_h, y, u, v, eta_y, eta_u, eta_v

    def forward(self, phi, mask, D, W=None, return_iterates=False, tol=None, max_iters=None):
        """
        return_iterates : si True devuelve la lista [(chi_k, phi_h_k), ...] de todos
                          los iterados (para supervision profunda en entrenamiento).
        tol             : si se da (inferencia), itera hasta que la actualizacion
                          relativa de chi sea < tol -> numero de iteraciones adaptativo.
        max_iters       : tope de iteraciones; por defecto self.num_iters.
        """
        phi = phi.float()
        mask = mask.float()
        # W (peso de fidelidad ~ magnitud^2). Si no se entrega, se usa la mascara,
        # equivalente a fidelidad uniforme dentro del ROI.
        W = mask if W is None else W.float()

        rho_y = F.softplus(self.rho_y)

        chi = torch.zeros_like(phi)
        phi_h = torch.zeros_like(phi)
        y = (W * phi) / (W + rho_y + self.eps)
        u = torch.zeros_like(phi)
        v = torch.zeros_like(phi)
        eta_y = torch.zeros_like(phi)
        eta_u = torch.zeros_like(phi)
        eta_v = torch.zeros_like(phi)

        n_iters = self.num_iters if max_iters is None else max_iters
        # checkpoint solo cuando se entrena (con grad); en inferencia no aporta y
        # ademas complica el criterio de parada.
        use_ckpt = torch.is_grad_enabled()
        iterates = []

        for k in range(n_iters):
            chi_prev = chi
            if use_ckpt:
                chi, phi_h, y, u, v, eta_y, eta_u, eta_v = checkpoint(
                    self.step_fn, chi, phi_h, y, u, v, eta_y, eta_u, eta_v,
                    phi, mask, W, D, use_reentrant=False
                )
            else:
                chi, phi_h, y, u, v, eta_y, eta_u, eta_v = self.step_fn(
                    chi, phi_h, y, u, v, eta_y, eta_u, eta_v, phi, mask, W, D
                )

            if return_iterates:
                iterates.append((chi, phi_h))

            # criterio de parada por actualizacion relativa (inferencia / produccion)
            if tol is not None and k > 0:
                rel = torch.norm(chi - chi_prev) / (torch.norm(chi) + self.eps)
                if rel < tol:
                    break

        if return_iterates:
            return iterates
        return chi, phi_h
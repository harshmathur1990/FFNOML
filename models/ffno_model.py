import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# Helpers
# ============================================================
def _to_spacing_value(spacing, x):
    if isinstance(spacing, (float, int)):
        return float(spacing)
    if torch.is_tensor(spacing):
        if spacing.numel() != 1:
            raise ValueError("spacing must be scalar.")
        return float(spacing.detach().item())
    raise TypeError(f"Unsupported spacing type: {type(spacing)}")


def _gn(channels, max_groups=8):
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


def trapezoid_weights(z_scale):
    """The trapezoid quadrature weights |dz| of each column, [B, 1, D, H, W] from z_scale [B, 1, D, H, W], levels in any
    order."""
    if z_scale.shape[2] == 1:
        return torch.ones_like(z_scale)
    step = 0.5 * (z_scale[:, :, 1:] - z_scale[:, :, :-1]).abs()
    return F.pad(step, (0, 0, 0, 0, 0, 1)) + F.pad(step, (0, 0, 0, 0, 1, 0))


def height_fourier_weights(z, k):
    """Weights q_jm with sum_j x_j q_jm = the integral of x(z) exp(-i k_m z) dz over the column, exactly for the
    piecewise-linear x through the levels (Filon quadrature): accurate at any k on any spacing, where the trapezoid sum
    fails once a wave has few levels. z [..., D] (levels in any order), k [M]; returns (real, imag), [..., D, M]. At k = 0
    they are the trapezoid weights."""
    za, zb = z[..., :-1, None], z[..., 1:, None]                  # consecutive levels, [..., D-1, 1]
    low = torch.minimum(za, zb)
    h = (zb - za).abs()
    theta = h * k                                                   # [..., D-1, M]
    small = theta.abs() < 1e-2
    t = torch.where(small, torch.ones_like(theta), theta)
    c, s = torch.cos(t), torch.sin(t)
    # R = int_0^1 u exp(-i theta u) du, L = int_0^1 (1 - u) exp(-i theta u) du
    r_re = torch.where(small, 0.5 - theta ** 2 / 8, (c + t * s - 1) / t ** 2)
    r_im = torch.where(small, -theta / 3 + theta ** 3 / 30, (t * c - s) / t ** 2)
    i_re, i_im = s / t, (c - 1) / t                                 # i (exp(-i theta) - 1) / theta
    l_re = torch.where(small, 0.5 - theta ** 2 / 24, i_re - r_re)
    l_im = torch.where(small, -theta / 6 + theta ** 3 / 120, i_im - r_im)
    phase = low * k
    e_re, e_im = h * torch.cos(phase), -h * torch.sin(phase)        # h exp(-i k z_low)
    lo_re, lo_im = e_re * l_re - e_im * l_im, e_re * l_im + e_im * l_re      # the lower level's share
    hi_re, hi_im = e_re * r_re - e_im * r_im, e_re * r_im + e_im * r_re      # the upper level's
    first_low = za <= zb
    first_re, first_im = torch.where(first_low, lo_re, hi_re), torch.where(first_low, lo_im, hi_im)
    second_re, second_im = torch.where(first_low, hi_re, lo_re), torch.where(first_low, hi_im, lo_im)
    pad_end, pad_start = (0, 0, 0, 1), (0, 0, 1, 0)
    return (F.pad(first_re, pad_end) + F.pad(second_re, pad_start),
            F.pad(first_im, pad_end) + F.pad(second_im, pad_start))


def depth_weights(z_scale):
    """Each level's share of its column's height, [B, 1, D, H, W] from z_scale [B, D, H, W] or [B, 1, D, H, W]: the
    trapezoid |dz| over the column's mean, so that a level counts by the height it stands for, not by how many levels are
    near it. 1 everywhere on a uniform grid."""
    if z_scale.ndim == 4:
        z_scale = z_scale.unsqueeze(1)
    w = trapezoid_weights(z_scale)
    return w / w.mean(dim=2, keepdim=True).clamp_min(1e-12)


class DepthWeightedGroupNorm(nn.GroupNorm):
    """GroupNorm whose statistics weight each level by depth_weights, so that they are means over height and not over
    levels; without weights, nn.GroupNorm. The same parameters as nn.GroupNorm."""

    def forward(self, x, w=None):
        if w is None:
            return super().forward(x)
        B, C = x.shape[:2]
        g = x.reshape(B, self.num_groups, C // self.num_groups, *x.shape[2:])
        wg = w.unsqueeze(1)                                         # [B, 1, 1, D, H, W]
        dims = tuple(range(2, g.dim()))
        total = (wg * torch.ones_like(g)).sum(dim=dims, keepdim=True)
        mean = (wg * g).sum(dim=dims, keepdim=True) / total
        variance = (wg * (g - mean) ** 2).sum(dim=dims, keepdim=True) / total
        y = ((g - mean) * torch.rsqrt(variance + self.eps)).reshape_as(x)
        if self.affine:
            shape = (1, C) + (1,) * (x.dim() - 2)
            y = y * self.weight.view(shape) + self.bias.view(shape)
        return y


def _wgn(channels, max_groups=8):
    g = _gn(channels, max_groups).num_groups
    return DepthWeightedGroupNorm(g, channels)


def _resolve_layer_dropout(rate, layers, layer_idx):
    if not layers:
        return 0.0
    if layer_idx in set(layers):
        return float(rate)
    return 0.0


# MULTI3D's constants (multi3d/src/mconst.f90), cgs
_M3D_EE, _M3D_HH, _M3D_EM, _M3D_BK, _M3D_PI = 1.602189e-12, 6.626176e-27, 9.109534e-28, 1.380662e-16, 3.14159265359
_M3D_LN_CCON = math.log(0.5 * (_M3D_HH / math.sqrt(2.0 * _M3D_PI * _M3D_EM) / math.sqrt(_M3D_BK)) ** 3)


def multi3d_lte_tables(atom, levels):
    """Per level of a MULTI3D atom {"g", "ev", "ion"}: ln(g_i / g_1), ev_i [eV], the Debye-lowering multiple ndxi(ion_i)
    and ion_i - ion_1, as in calc_lte_populations (multi3d/src/phys_mod.f90)."""
    g, ev, ion = [float(v) for v in atom["g"]], [float(v) for v in atom["ev"]], [int(v) for v in atom["ion"]]
    if not len(g) == len(ev) == len(ion) == levels:
        raise ValueError(f"multi3d_lte atom data has {len(g)}, {len(ev)}, {len(ion)} entries for {levels} levels")
    ndxi = [0] * 21                     # Fortran ndxi(1:20), index = ionisation stage
    for i in range(2, 21):
        ndxi[i] = ndxi[i - 1] + (i - 1)
    for i in range(1, 21):              # as MULTI3D's loop, which zeroes ndxi(ion_1) on the way: later stages keep theirs
        ndxi[i] -= ndxi[ion[0]]
    return ([math.log(gi / g[0]) for gi in g], ev, [float(ndxi[i]) for i in ion],
            [float(max(i - ion[0], 0)) for i in ion])


def multi3d_lte_log_weights(log10_t, log10_ne, log_g, ev, ndxi, dion):
    """ln of MULTI3D's unnormalised LTE level weights (calc_lte_populations), [B, levels, ...] in float64, from log10 T [K]
    and log10 ne [m^-3] of shape [B, ...] and per-level tables (multi3d_lte_tables, concatenated over species). Each
    species' softmax of them is MULTI3D's nstar / totn."""
    t = torch.pow(10.0, log10_t.double())[:, None]
    ne = torch.pow(10.0, log10_ne.double() - 6.0)[:, None]          # cm^-3
    shape = (1, -1) + (1,) * (t.dim() - 2)
    table = lambda v: torch.tensor(v, dtype=torch.float64, device=t.device).view(shape)
    conl = _M3D_LN_CCON + torch.log(ne) - 1.5 * torch.log(t)
    dxi = 4.98e-4 * 5040.0 / t * torch.sqrt(ne * _M3D_BK * t)          # eV
    return table(log_g) - (_M3D_EE / _M3D_BK) / t * (table(ev) - dxi * table(ndxi)) - table(dion) * conl


# ============================================================
# Pointwise MLP (small corrective branch)
# ============================================================
class PointwiseMLP(nn.Module):
    def __init__(self, width, expansion=3, dropout=0.0):
        super().__init__()
        hidden = width * expansion

        self.net = nn.Sequential(
            nn.Conv3d(width, hidden, 1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(hidden, width, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Vertical branch: the height operator
# ============================================================
class HeightSpectralConv(nn.Module):
    """
    The vertical branch of every block: a spectral convolution along height, the counterpart of SpectralConv2dFull in x
    and y, on each column's own heights, so that it is the same operator on any vertical grid and for any number of
    levels.

    Wavenumbers k_m = 2 pi m / reference_length, m = 0 .. M - 1, up to the shortest wavelength min_wavelength (M =
    reference_length / min_wavelength + 1); nothing above it. A column is transformed on its heights and the mixed modes
    are evaluated back at the same heights,
        X_m = int x(z) exp(-i k_m z) dz,    y(z_j) = (Y_0 + 2 Re sum_{m>=1} Y_m exp(i k_m z_j)) / reference_length,
    with Y_m = W(k_m) X_m. As in x and y, W(k) is a learned complex channel matrix scaled by a small MLP of the physical
    wavenumber, plus (rank > 0) a low-rank matrix U(k) V(k) from a second MLP: a smooth function of k, not a table.
    X_m is the exact transform of the piecewise-linear column through the levels (height_fourier_weights), so the
    operator does not depend on how many levels sample a wave; the grid only sets how well the column is represented.
    The column's height (z_scale) also enters pointwise. The periodic length L is
    reference_length (z_scale units, Mm), or twice the tallest column when that is longer, so that a column's top and
    bottom are never neighbours in the periodic domain (box). A longer L only samples k more densely over the same band
    0 .. 2 pi / min_wavelength, where W(k) is the trained function: between trained wavenumbers, never beyond them.
    The frequency MLPs see the wavenumber on the fixed physical scale feature_wavelength, t = q / (1 + q) with
    q = k feature_wavelength / 2 pi, so W(k) at a given wavenumber is the same function whatever min_wavelength is.
    Each column keeps only the wavenumbers its own grid resolves, from its median level spacing dz (the spacing where
    the lines form; only the coronal top is coarser): W(k) is tapered from k = 2 pi / (3 dz) to zero at the Nyquist
    wavenumber pi / dz, so modes no column sample can carry are not learned from interpolation.

    Input: x [B, C, D, H, W], z_scale [B, D, H, W] or [B, 1, D, H, W]. Output: [B, C, D, H, W]
    """

    def __init__(self, channels, reference_length, min_wavelength, feature_wavelength, hidden=128, rank=0):
        super().__init__()
        if reference_length <= 0 or min_wavelength <= 0 or feature_wavelength <= 0:
            raise ValueError("the height operator needs z_reference_length, z_min_wavelength and z_feature_wavelength > 0")
        self.reference_length = float(reference_length)
        self.min_wavelength = float(min_wavelength)
        self.feature_wavelength = float(feature_wavelength)
        self.modes = int(math.floor(self.reference_length / self.min_wavelength)) + 1
        self.rank = int(rank)
        scale = 1.0 / math.sqrt(channels)
        self.weight_real = nn.Parameter(scale * torch.randn(channels, channels))
        self.weight_imag = nn.Parameter(scale * torch.randn(channels, channels))
        self.input_gate = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.z_proj = nn.Conv3d(1, channels, 1, bias=False)
        self.freq_mlp = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, 2))
        if self.rank:
            self.freq_lowrank = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
            self.freq_u = nn.Linear(hidden, channels * self.rank)
            self.freq_v = nn.Linear(hidden, self.rank * channels)
            with torch.no_grad():
                self.freq_u.weight.zero_()
                self.freq_u.bias.zero_()
        self.post = nn.Sequential(
            nn.Conv3d(channels, 2 * channels, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(2 * channels, channels, 1, bias=False),
        )
        k = 2 * math.pi * torch.arange(self.modes, dtype=torch.float32) / self.reference_length
        self.register_buffer("k", k, persistent=False)
        self.register_buffer("mode_factor", torch.cat([torch.ones(1), 2 * torch.ones(self.modes - 1)]),
                             persistent=False)

    def k_features(self, k=None):
        """The frequency MLPs' input [M, 2] at wavenumbers k (default: those of reference_length): t and t^2, t a fixed
        function of the physical wavenumber."""
        k = self.k if k is None else k
        q = k * self.feature_wavelength / (2 * math.pi)
        t = q / (1 + q)
        return torch.stack([t, t * t], dim=-1)

    def cutoff(self, columns, k=None):
        """The per-column taper [..., M], for columns [..., D] of heights: 1 up to 2 pi / (3 dz),
        a half cosine down to 0 at pi / dz, dz the column's median level spacing."""
        k = self.k if k is None else k
        dz = (columns[..., 1:] - columns[..., :-1]).abs().median(dim=-1).values[..., None]
        k_nyquist = math.pi / dz
        u = ((k - 2 * k_nyquist / 3) / (k_nyquist / 3)).clamp(0, 1)
        return 0.5 * (1 + torch.cos(math.pi * u))

    def box(self, z_scale):
        """The periodic length L for these columns, its wavenumbers k [M] and the Fourier series' mode factors [M]:
        reference_length and the module's own, unless twice the tallest column is longer; then L is that, with
        floor(L / min_wavelength) + 1 wavenumbers 2 pi m / L, the same band sampled more densely."""
        span = float((z_scale.amax(dim=-3) - z_scale.amin(dim=-3)).max())
        if 2 * span <= self.reference_length:
            return self.reference_length, self.k, self.mode_factor
        length = 2 * span
        modes = int(math.floor(length / self.min_wavelength)) + 1
        k = 2 * math.pi * torch.arange(modes, dtype=self.k.dtype, device=self.k.device) / length
        factor = torch.cat([k.new_ones(1), 2 * k.new_ones(modes - 1)])
        return length, k, factor

    def forward(self, x, z_scale):
        if z_scale.ndim == 4:
            z_scale = z_scale.unsqueeze(1)
        length, k, mode_factor = self.box(z_scale)
        modes = k.numel()
        x = x * self.input_gate(x) + self.z_proj(z_scale)
        z = z_scale - z_scale.amin(dim=2, keepdim=True)              # a shift of z changes nothing but rounding
        shared = bool((z == z[..., :1, :1]).all())                   # one grid for every column (simulations, mesh)
        shared = shared and bool((z == z[:1]).all())                 # ... and of every sample in the batch
        if shared:
            column = z[0, 0, :, 0, 0]
            q_re, q_im = height_fourier_weights(column, k)           # [D, M]
            phase = column[:, None] * k
            x_re = torch.einsum("b c d h w, d m -> b c m h w", x, q_re)
            x_im = torch.einsum("b c d h w, d m -> b c m h w", x, q_im)
            taper = self.cutoff(column, k)[:, None, None]                                   # [M, 1, 1]
        else:
            columns = z[:, 0].permute(0, 2, 3, 1)                    # [B, H, W, D]
            q_re, q_im = height_fourier_weights(columns, k)          # [B, H, W, D, M]
            phase = z[:, 0, ..., None] * k                           # [B, D, H, W, M]
            x_re = torch.einsum("b c d h w, b h w d m -> b c m h w", x, q_re)
            x_im = torch.einsum("b c d h w, b h w d m -> b c m h w", x, q_im)
            taper = self.cutoff(columns, k).permute(0, 3, 1, 2)[:, None]                    # [B, 1, M, H, W]
        cos, sin = torch.cos(phase), torch.sin(phase)
        features = self.k_features(k)
        gate = self.freq_mlp(features)                               # [M, 2]
        gr, gi = gate[:, 0, None, None], gate[:, 1, None, None]      # [M, 1, 1]
        a_re = torch.einsum("b i m h w, i o -> b o m h w", x_re, self.weight_real) - \
            torch.einsum("b i m h w, i o -> b o m h w", x_im, self.weight_imag)
        a_im = torch.einsum("b i m h w, i o -> b o m h w", x_im, self.weight_real) + \
            torch.einsum("b i m h w, i o -> b o m h w", x_re, self.weight_imag)
        y_re, y_im = a_re * gr - a_im * gi, a_re * gi + a_im * gr
        if self.rank:
            h = self.freq_lowrank(features)
            u = self.freq_u(h).reshape(modes, -1, self.rank)         # [M, C, r]
            v = self.freq_v(h).reshape(modes, self.rank, -1)         # [M, r, C]
            low_re = torch.einsum("b i m h w, m r i -> b r m h w", x_re, v)
            low_im = torch.einsum("b i m h w, m r i -> b r m h w", x_im, v)
            y_re = y_re + torch.einsum("b r m h w, m o r -> b o m h w", low_re, u)
            y_im = y_im + torch.einsum("b r m h w, m o r -> b o m h w", low_im, u)
        f = mode_factor[:, None, None] * taper
        y_re, y_im = y_re * f, y_im * f
        if shared:
            y = torch.einsum("b o m h w, d m -> b o d h w", y_re, cos) - \
                torch.einsum("b o m h w, d m -> b o d h w", y_im, sin)
        else:
            y = torch.einsum("b o m h w, b d h w m -> b o d h w", y_re, cos) - \
                torch.einsum("b o m h w, b d h w m -> b o d h w", y_im, sin)
        return self.post(y / length)


# ============================================================
# Stronger Spectral Branch
# ============================================================
class SpectralConv2dFull(nn.Module):
    """
    Full-spectrum metric-aware spectral convolution over (H, W),
    applied independently at each depth plane.

    With rank > 0, each frequency also gets its own low-rank channel mixing U(k) V(k), U: out x rank and
    V: rank x in, from a second small MLP of k; U starts at zero, so the model starts as the one without it.

    Input/Output: [B, C, D, H, W]
    """

    def __init__(self, in_channels, out_channels, hidden=128, post_mix_expansion=2, rank=0):
        super().__init__()
        self.rank = int(rank)

        scale = 1.0 / math.sqrt(in_channels)

        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, dtype=torch.float32)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, dtype=torch.float32)
        )

        self.input_gate = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.freq_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

        if self.rank:
            self.freq_lowrank = nn.Sequential(
                nn.Linear(4, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.freq_u = nn.Linear(hidden, out_channels * self.rank)
            self.freq_v = nn.Linear(hidden, self.rank * in_channels)
            with torch.no_grad():
                self.freq_u.weight.zero_()
                self.freq_u.bias.zero_()

        post_hidden = out_channels * post_mix_expansion
        
        self.post = nn.Sequential(
            nn.Conv3d(out_channels, post_hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(post_hidden, out_channels, 1, bias=False),
        )

    @staticmethod
    def k_features(ky, kx):
        """The frequency MLPs' input [H, Wf, 4] from the wavenumber grids ky, kx [H, Wf] (in 1/m)."""
        freq_multiplier = 1e5

        ky = ky * freq_multiplier

        kx = kx * freq_multiplier

        return torch.stack(
            [
                kx,
                ky,
                torch.sqrt(kx ** 2 + ky ** 2 + 1e-12),
                torch.sign(ky),
            ],
            dim=-1,
        )

    def mix_spectrum(self, x_ft, k_feat):
        """The channel mixing of the spectrum x_ft [B, Cin, D, H, Wf] at frequencies k_feat [H, Wf, 4] (shared with
        the distributed inference path)."""
        gate = self.freq_mlp(k_feat)  # [H, Wf, 2]

        wr = self.weight_real[:, :, None, None]  # [Cin, Cout, 1, 1]
        wi = self.weight_imag[:, :, None, None]

        gr = gate[..., 0][None, None]  # [1, 1, H, Wf]
        gi = gate[..., 1][None, None]

        w_real = wr * gr - wi * gi
        w_imag = wr * gi + wi * gr

        xr = x_ft.real
        xi = x_ft.imag

        out_r = (
            torch.einsum("b i d y x, i o y x -> b o d y x", xr, w_real)
            - torch.einsum("b i d y x, i o y x -> b o d y x", xi, w_imag)
        )

        out_i = (
            torch.einsum("b i d y x, i o y x -> b o d y x", xr, w_imag)
            + torch.einsum("b i d y x, i o y x -> b o d y x", xi, w_real)
        )

        out = torch.complex(out_r, out_i)
        if self.rank:
            h = self.freq_lowrank(k_feat)
            ny, nx = k_feat.shape[:2]
            u = self.freq_u(h).reshape(ny, nx, -1, self.rank).to(x_ft.dtype)   # [H, Wf, Cout, r]
            v = self.freq_v(h).reshape(ny, nx, self.rank, -1).to(x_ft.dtype)   # [H, Wf, r, Cin]
            low = torch.einsum("b i d y x, y x r i -> b r d y x", x_ft, v)
            out = out + torch.einsum("b r d y x, y x o r -> b o d y x", low, u)
        return out

    def forward(self, x, dx, dy):
        B, C, D, H, W = x.shape

        dx = _to_spacing_value(dx, x)
        dy = _to_spacing_value(dy, x)

        # [B, C, D, H, Wf]

        g = self.input_gate(x)
        x = x * g
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        ky = torch.fft.fftfreq(H, d=dy, device=x.device)
        kx = torch.fft.rfftfreq(W, d=dx, device=x.device)
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")

        out = self.mix_spectrum(x_ft, self.k_features(ky, kx))  # k features [H, Wf, 4]
        y = torch.fft.irfft2(out, s=(H, W), dim=(-2, -1))
        y = self.post(y)
        return y


# ============================================================
# Balanced Strong Block
# ============================================================
class FFNOBlock3dBalanced(nn.Module):
    """
    One block: a horizontal spectral branch (SpectralConv2dFull) and a vertical one (HeightSpectralConv) on the same
    input, each normalised, gated pointwise and fused by pointwise channel mixing, then two small pointwise corrective
    paths (pw, mlp) that start at weight 0.5. Every GroupNorm weights levels by their height (DepthWeightedGroupNorm):
    nothing counts grid points in any direction.
    """

    def __init__(
        self,
        width=96,
        dropout=0.0,
        spec_dropout=0.0,
        vertical_dropout=0.0,
        hidden_spectral=128,
        hidden_vertical=128,
        freq_mix_rank=8,
        z_reference_length=40.0,
        z_min_wavelength=0.04,
        z_feature_wavelength=0.3,
    ):
        super().__init__()

        self.spec = SpectralConv2dFull(
            width,
            width,
            hidden=hidden_spectral,
            post_mix_expansion=2,
            rank=freq_mix_rank,
        )
        self.vertical = HeightSpectralConv(width, z_reference_length, z_min_wavelength, z_feature_wavelength,
                                           hidden=hidden_vertical, rank=freq_mix_rank)

        self.pw = nn.Sequential(
            nn.Conv3d(width, width * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(width * 2, width, 1, bias=False),
        )

        self.mlp = PointwiseMLP(width, expansion=2, dropout=dropout)

        self.norm_spec = _wgn(width)
        self.norm_vert = _wgn(width)
        self.branch_gate = nn.Sequential(
            nn.Conv3d(3 * width, width, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(width, 2 * width, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * width, 2 * width, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(2 * width, width, 1, bias=False),
        )
        self.norm_fuse = _wgn(width)
        self.norm_pw = _wgn(width)
        self.norm_mlp = _wgn(width)

        self.res_fused = nn.Parameter(torch.tensor([1.0]))
        self.res_pw = nn.Parameter(torch.tensor([0.5]))
        self.res_mlp = nn.Parameter(torch.tensor([0.5]))

        self.act = nn.GELU()
        self.spec_drop = nn.Dropout(spec_dropout) if spec_dropout > 0 else nn.Identity()
        self.vert_drop = (
            nn.Dropout(vertical_dropout) if vertical_dropout > 0 else nn.Identity()
        )

    def _fuse_branches(self, x, spec, vert, w=None):
        gates = self.branch_gate(torch.cat([x, spec, vert], dim=1))
        spec_gate, vert_gate = gates.chunk(2, dim=1)

        spec_mix = spec * spec_gate
        vert_mix = vert * vert_gate

        fused = self.fuse(torch.cat([spec_mix, vert_mix], dim=1))
        fused = fused + 0.5 * (spec_mix + vert_mix)
        fused = self.act(self.norm_fuse(fused, w))
        return fused, spec_mix, vert_mix, spec_gate, vert_gate

    def forward(self, x, z_scale, dx, dy, collect_stats=False, branch_mask=None):
        w = depth_weights(z_scale)
        if not collect_stats and branch_mask is None:
            residual = x

            spec = self.spec_drop(self.act(self.norm_spec(self.spec(x, dx, dy), w)))
            vert = self.vert_drop(self.act(self.norm_vert(self.vertical(x, z_scale), w)))
            fused, _, _, _, _ = self._fuse_branches(x, spec, vert, w)

            x1 = residual + self.res_fused * fused

            pw = self.norm_pw(self.pw(x1), w)
            x2 = x1 + self.res_pw * pw

            mlp = self.norm_mlp(self.mlp(x2), w)
            out = x2 + self.res_mlp * mlp

            return out

        residual = x
        stats = {} if collect_stats else None
        branch_mask = branch_mask or {}

        spec = self.spec_drop(self.act(self.norm_spec(self.spec(x, dx, dy), w)))
        vert = self.vert_drop(self.act(self.norm_vert(self.vertical(x, z_scale), w)))

        spec_mask = float(branch_mask.get("spec", 1.0))
        vert_mask = float(branch_mask.get("vertical", 1.0))
        pw_mask = float(branch_mask.get("pw", 1.0))
        mlp_mask = float(branch_mask.get("mlp", 1.0))

        spec_eff = spec_mask * spec
        vert_eff = vert_mask * vert
        fused, spec_mix, vert_mix, spec_gate, vert_gate = self._fuse_branches(
            x, spec_eff, vert_eff, w
        )

        x1 = residual + self.res_fused * fused

        pw = self.norm_pw(self.pw(x1), w)
        x2 = x1 + (self.res_pw * pw_mask) * pw

        mlp = self.norm_mlp(self.mlp(x2), w)
        out = x2 + (self.res_mlp * mlp_mask) * mlp

        if collect_stats:
            with torch.no_grad():
                stats["spec_norm"] = spec.abs().mean().item()
                stats["vertical_norm"] = vert.abs().mean().item()
                stats["fuse_norm"] = fused.abs().mean().item()
                stats["pw_norm"] = pw.abs().mean().item()
                stats["mlp_norm"] = mlp.abs().mean().item()
                stats["spec_weight"] = float(spec_mask)
                stats["vertical_weight"] = float(vert_mask)
                stats["spec_weight_effective"] = float(spec_mask)
                stats["vertical_weight_effective"] = float(vert_mask)
                stats["res_fused"] = float(self.res_fused.item())
                stats["res_pw"] = float(self.res_pw.item())
                stats["res_mlp"] = float(self.res_mlp.item())
                stats["spec_gate_mean"] = spec_gate.mean().item()
                stats["vertical_gate_mean"] = vert_gate.mean().item()
                stats["spec_contrib"] = (self.res_fused * spec_mix).abs().mean().item()
                stats["vertical_contrib"] = (self.res_fused * vert_mix).abs().mean().item()
                stats["fuse_contrib"] = (self.res_fused * fused).abs().mean().item()
                stats["pw_contrib"] = ((self.res_pw * pw_mask) * pw).abs().mean().item()
                stats["mlp_contrib"] = ((self.res_mlp * mlp_mask) * mlp).abs().mean().item()
                stats["fused_update"] = (x1 - residual).abs().mean().item()
                stats["pw_update"] = (x2 - x1).abs().mean().item()
                stats["mlp_update"] = (out - x2).abs().mean().item()

            return out, stats

        return out


# ============================================================
# Full Model
# ============================================================
class FFNO3D(nn.Module):
    """
    The FFNO: a pointwise lift, n_layers blocks (FFNOBlock3dBalanced: spectral in x and y, the height operator along
    z, nothing counting grid points), and the level-fraction head. Per species the head predicts one departure logit
    delta_i per level and one total correction u:
        log10 n_i = log10(rho / particle mass) + s + log10 softmax(ln f*_i + delta_i),   s = total_bound * tanh(u / total_bound)
    f*_i(T, ne) is MULTI3D's LTE level fraction (multi3d_lte_log_weights, from the input T and ne and the species'
    "multi3d_lte" atom data), so delta_i is ln b_i, the departure coefficient, up to a constant common to the species'
    levels, and delta = 0 gives MULTI3D's LTE populations. The softmax is each level's share of the species total, and s
    the log10 ratio of that total to rho / mass. Any set of positive levels whose total lies within total_bound dex of
    rho / mass is reproduced exactly; the bound only keeps populations finite in float32. The output is normalised
    log10 populations. The defaults are the final model's (zaug); species comes from config.ATOM_CONFIG.
    """

    def __init__(
        self,
        in_channels=6,
        out_channels=6,
        width=96,
        n_layers=4,
        dropout=0.0,
        spec_dropout=0.0,
        vertical_dropout=0.0,
        spec_dropout_layers=None,
        vertical_dropout_layers=None,
        checkpoint_blocks=True,
        hidden_spectral=128,
        hidden_vertical=128,
        freq_mix_rank=8,
        z_reference_length=40.0,
        z_min_wavelength=0.04,
        z_feature_wavelength=0.3,
        species=None,
        rho_channel=5,
        total_bound=6.0,
        temperature_channel=0,
        ne_channel=4,
    ):
        super().__init__()

        if not species:
            raise ValueError("FFNO3D needs species (config.ATOM_CONFIG: levels, log10_particle_mass, multi3d_lte)")
        self.species_levels = [int(s["levels"]) for s in species]
        if sum(self.species_levels) != out_channels:
            raise ValueError(f"species levels {self.species_levels} do not add up to out_channels={out_channels}")
        head_channels = out_channels + len(self.species_levels)
        self.total_bound = float(total_bound)
        self.rho_channel = int(rho_channel)
        self.register_buffer("log10_particle_mass",
                             torch.tensor([float(s["log10_particle_mass"]) for s in species], dtype=torch.float32))
        # normalisation of the inputs (for rho, T, ne) and of the output; set from the dataset with set_normalization
        # and saved with the checkpoint
        self.register_buffer("mean_X", torch.zeros(in_channels))
        self.register_buffer("std_X", torch.ones(in_channels))
        self.register_buffer("mean_Y", torch.zeros(out_channels))
        self.register_buffer("std_Y", torch.ones(out_channels))
        missing = [i for i, s in enumerate(species) if "multi3d_lte" not in s]
        if missing:
            raise ValueError(f"the level-fraction head needs each species' multi3d_lte atom data (g, ev, ion); "
                             f"missing for {missing}")
        self.temperature_channel, self.ne_channel = int(temperature_channel), int(ne_channel)
        # plain constants, not buffers, so that casting the model never rounds them
        tables = [multi3d_lte_tables(s["multi3d_lte"], s["levels"]) for s in species]
        self.lte_tables = tuple([v for t in tables for v in t[k]] for k in range(4))

        self.checkpoint_blocks = checkpoint_blocks
        # DistributedFSDPBackend wraps each block with PyTorch's composable
        # checkpoint wrapper before FSDP is applied. In that configuration the
        # block call must not be checkpointed a second time around the FSDP
        # module, while lift/head checkpointing remains enabled.
        self.external_block_checkpointing = False

        self.lift = nn.Sequential(
            nn.Conv3d(in_channels, width, 1, bias=False),
            _wgn(width),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            FFNOBlock3dBalanced(
                width=width,
                dropout=dropout,
                spec_dropout=_resolve_layer_dropout(
                    spec_dropout, spec_dropout_layers, layer_idx
                ),
                vertical_dropout=_resolve_layer_dropout(
                    vertical_dropout, vertical_dropout_layers, layer_idx
                ),
                hidden_spectral=hidden_spectral,
                hidden_vertical=hidden_vertical,
                freq_mix_rank=freq_mix_rank,
                z_reference_length=z_reference_length,
                z_min_wavelength=z_min_wavelength,
                z_feature_wavelength=z_feature_wavelength,
            )
            for layer_idx in range(n_layers)
        ])

        self.proj1 = nn.Conv3d(width, width * 2, 1, bias=False)
        self.proj2 = nn.Conv3d(width * 2, head_channels, 1)
        # the total corrections start at s = 0, the total rho / mass
        with torch.no_grad():
            c = 0
            for levels in self.species_levels:
                self.proj2.weight[c + levels].zero_()
                self.proj2.bias[c + levels].zero_()
                c += levels + 1

        self.act = nn.GELU()

    def set_normalization(self, mean_X, std_X, mean_Y, std_Y):
        """Copy the dataset's normalisation into the head."""
        for name, value in (("mean_X", mean_X), ("std_X", std_X), ("mean_Y", mean_Y), ("std_Y", std_Y)):
            buffer = getattr(self, name)
            buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype).broadcast_to(buffer.shape))

    def head_output(self, head, x_in):
        """Normalised log10 populations from the head's channels and the normalised inputs x_in [B, Cin, D, H, W]."""
        rho = self.rho_channel
        log10_rho = x_in[:, rho] * self.std_X[rho] + self.mean_X[rho]
        t, e = self.temperature_channel, self.ne_channel
        lte = multi3d_lte_log_weights(x_in[:, t].double() * self.std_X[t].double() + self.mean_X[t].double(),
                                      x_in[:, e].double() * self.std_X[e].double() + self.mean_X[e].double(),
                                      *self.lte_tables)
        out = []
        c = k = 0                               # head channel and output channel of the species' first level
        for i, levels in enumerate(self.species_levels):
            # in float64: the LTE weights span tens of e-folds
            logits = head[:, c:c + levels].double() + lte[:, k:k + levels]
            log10_fraction = (torch.log_softmax(logits, dim=1) / math.log(10.0)).to(head.dtype)
            s = self.total_bound * torch.tanh(head[:, c + levels] / self.total_bound)
            out.append((log10_rho - self.log10_particle_mass[i] + s)[:, None] + log10_fraction)
            c += levels + 1
            k += levels
        log10_n = torch.cat(out, dim=1)
        return (log10_n - self.mean_Y[None, :, None, None, None]) / self.std_Y[None, :, None, None, None]

    def _run_block(
        self, blk, x, z_scale, dx, dy, collect_stats=False, branch_mask=None
    ):
        # Inversion VJPs run the model in eval mode (dropout disabled) but still
        # require activation checkpointing at production cube sizes.  Gradient
        # mode, rather than train/eval mode, is the correct switch: no-grad
        # prediction avoids recomputation, while autograd VJPs retain only the
        # block inputs and recompute each FSDP-wrapped block during backward.
        if (
            self.checkpoint_blocks
            and not self.external_block_checkpointing
            and torch.is_grad_enabled()
            and not collect_stats
        ):
            return checkpoint(
                lambda t, z: blk(
                    t, z, dx, dy, collect_stats=False, branch_mask=None
                ),
                x,
                z_scale,
                use_reentrant=False,
            )
        return blk(
            x,
            z_scale,
            dx,
            dy,
            collect_stats=collect_stats,
            branch_mask=branch_mask,
        )

    def _lift(self, x, z_scale):
        return self.lift[2](self.lift[1](self.lift[0](x), depth_weights(z_scale)))

    def _run_lift(self, x, z_scale, collect_stats=False):
        if self.checkpoint_blocks and torch.is_grad_enabled() and not collect_stats:
            return checkpoint(self._lift, x, z_scale, use_reentrant=False)
        return self._lift(x, z_scale)

    def _run_head(self, x, collect_stats=False):
        if self.checkpoint_blocks and torch.is_grad_enabled() and not collect_stats:
            return checkpoint(
                lambda t: self.proj2(self.act(self.proj1(t))),
                x,
                use_reentrant=False,
            )
        x = self.act(self.proj1(x))
        return self.proj2(x)

    def forward(self, x, z_scale, dx, dy, collect_stats=False, branch_mask=None):
        x_in = x
        x = self._run_lift(x, z_scale, collect_stats=collect_stats)
        all_stats = [] if collect_stats else None

        for i, blk in enumerate(self.blocks):
            if collect_stats:
                x, s = self._run_block(
                    blk,
                    x,
                    z_scale,
                    dx,
                    dy,
                    collect_stats=True,
                    branch_mask=branch_mask,
                )
                s["layer"] = i
                all_stats.append(s)
            else:
                x = self._run_block(
                    blk,
                    x,
                    z_scale,
                    dx,
                    dy,
                    collect_stats=False,
                    branch_mask=branch_mask,
                )

        x = self.head_output(self._run_head(x, collect_stats=collect_stats), x_in)
        if collect_stats:
            return x, all_stats
        return x

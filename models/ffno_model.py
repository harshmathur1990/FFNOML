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
# Stronger Vertical Branch
# ============================================================
class DepthwiseSeparable1D(nn.Module):
    def __init__(self, ch, hidden_mult=2, k=5, dilation=1, dropout=0.0):
        super().__init__()

        pad = dilation * (k // 2)

        self.dw = nn.Conv1d(
            ch,
            ch,
            kernel_size=k,
            padding=pad,
            dilation=dilation,
            groups=ch,
            bias=False,
        )

        hidden = ch * hidden_mult
        self.pw = nn.Sequential(
            nn.Conv1d(ch, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, ch, 1, bias=False),
        )

        self.gn = _gn(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        y = self.gn(y)
        y = self.act(y)
        y = self.drop(y)
        return y


class LiteVerticalResidualBlock(nn.Module):
    def __init__(self, ch, k=5, dilation=1, dropout=0.0):
        super().__init__()
        self.b1 = DepthwiseSeparable1D(
            ch, hidden_mult=2, k=k, dilation=dilation, dropout=dropout
        )
        self.b2 = DepthwiseSeparable1D(
            ch, hidden_mult=2, k=k, dilation=1, dropout=dropout
        )
        self.out_gn = _gn(ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.b1(x)
        y = self.b2(y)
        return self.act(self.out_gn(x + y))


class LiteMultiScaleVertical(nn.Module):
    def __init__(self, ch, kernels=(3, 7, 15), dropout=0.0):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    ch,
                    ch,
                    kernel_size=k,
                    padding=k // 2,
                    groups=ch,
                    bias=False,
                ),
                nn.Conv1d(ch, ch, 1, bias=False),
            )
            for k in kernels
        ])

        self.mix = nn.Sequential(
            nn.Conv1d(ch, 2 * ch, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(2 * ch, ch, 1, bias=False),
        )

        self.gn = _gn(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = 0.0
        for b in self.branches:
            y = y + b(x)
        y = self.mix(y)
        y = self.gn(y)
        y = self.act(y)
        y = self.drop(y)
        return y


class SpectralConvZ(nn.Module):
    """
    Spectral convolution along depth, per column: the column is zero-padded to twice its length, so that its top and
    bottom are a column's length apart in the periodic domain instead of neighbours, and its lowest `modes` Fourier
    modes are mixed across channels with one complex weight matrix per mode. The weights start at zero, so a model with this branch starts as the model without it.

    Input/Output: [N, C, D]
    """

    def __init__(self, channels, modes):
        super().__init__()
        self.modes = int(modes)
        self.weight_real = nn.Parameter(torch.zeros(channels, channels, self.modes))
        self.weight_imag = nn.Parameter(torch.zeros(channels, channels, self.modes))

    def forward(self, x):
        depth = x.shape[-1]
        x_ft = torch.fft.rfft(x, n=2 * depth, dim=-1)
        m = min(self.modes, x_ft.shape[-1])
        weight = torch.complex(self.weight_real[..., :m], self.weight_imag[..., :m])
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., :m] = torch.einsum("n i k, i o k -> n o k", x_ft[..., :m], weight)
        return torch.fft.irfft(out_ft, n=2 * depth, dim=-1)[..., :depth]


class SpectralConvHeight(nn.Module):
    """
    Spectral convolution along height on each column's own heights: a convolution in z, the same on any vertical grid
    and for any number of levels (SpectralConvZ works on the level index instead). Mode m is the fixed wavenumber
    k_m = 2 pi m / reference_length, and its complex channel mixing W_m is the same in every column. A column is
    transformed by trapezoid quadrature on its heights and its mixed modes are evaluated back at the same heights:
        X_m = sum_j x(z_j) exp(-i k_m z_j) w_j,    y(z_j) = (W_0 X_0 + 2 Re sum_{m>=1} W_m X_m exp(i k_m z_j)) / reference_length
    The reference length (in the units of z_scale, Mm) must be at least twice the tallest column, so that a column's
    top and bottom are never neighbours in the periodic domain (checked by BalancedVerticalPhysicsStack), and the
    shortest wavelength, reference_length / (modes - 1), should be resolved by the coarsest grid spacing. The weights
    start at zero, so a model with this branch starts as the model without it.

    Input: x [N, C, D], z [N, 1, D]. Output: [N, C, D]
    """

    def __init__(self, channels, modes, reference_length):
        super().__init__()
        self.modes = int(modes)
        self.reference_length = float(reference_length)
        self.weight_real = nn.Parameter(torch.zeros(channels, channels, self.modes))
        self.weight_imag = nn.Parameter(torch.zeros(channels, channels, self.modes))
        k = 2 * math.pi * torch.arange(self.modes, dtype=torch.float32) / self.reference_length
        self.register_buffer("k", k, persistent=False)
        self.register_buffer("mode_factor", torch.cat([torch.ones(1), 2 * torch.ones(self.modes - 1)]),
                             persistent=False)

    def forward(self, x, z):
        z = z[:, 0]
        z = z - z.min(dim=-1, keepdim=True).values                  # a shift of z changes nothing but rounding
        w = torch.zeros_like(z)                                      # trapezoid weights |dz|, levels in any order
        if z.shape[-1] > 1:
            step = 0.5 * (z[:, 1:] - z[:, :-1]).abs()
            w = w + F.pad(step, (0, 1)) + F.pad(step, (1, 0))
        phase = z.unsqueeze(-1) * self.k                             # [N, D, M]
        cos, sin = torch.cos(phase), torch.sin(phase)
        xw = x * w.unsqueeze(1)
        x_re = torch.einsum("n i d, n d m -> n i m", xw, cos)
        x_im = -torch.einsum("n i d, n d m -> n i m", xw, sin)
        y_re = torch.einsum("n i m, i o m -> n o m", x_re, self.weight_real) - \
            torch.einsum("n i m, i o m -> n o m", x_im, self.weight_imag)
        y_im = torch.einsum("n i m, i o m -> n o m", x_im, self.weight_real) + \
            torch.einsum("n i m, i o m -> n o m", x_re, self.weight_imag)
        y_re, y_im = y_re * self.mode_factor, y_im * self.mode_factor
        return (torch.einsum("n o m, n d m -> n o d", y_re, cos) -
                torch.einsum("n o m, n d m -> n o d", y_im, sin)) / self.reference_length


class HeightSpectralConv(nn.Module):
    """
    The vertical branch of the height operator (FFNO3D vertical_operator=True): a spectral convolution along height, the
    counterpart of SpectralConv2dFull in x and y, on each column's own heights, so that it is the same operator on any
    vertical grid and for any number of levels.

    Wavenumbers k_m = 2 pi m / reference_length, m = 0 .. M - 1, up to the shortest wavelength min_wavelength (M =
    reference_length / min_wavelength + 1); nothing above it. A column is transformed on its heights and the mixed modes
    are evaluated back at the same heights,
        X_m = int x(z) exp(-i k_m z) dz,    y(z_j) = (Y_0 + 2 Re sum_{m>=1} Y_m exp(i k_m z_j)) / reference_length,
    with Y_m = W(k_m) X_m. As in x and y, W(k) is a learned complex channel matrix scaled by a small MLP of the physical
    wavenumber, plus (rank > 0) a low-rank matrix U(k) V(k) from a second MLP: a smooth function of k, not a table.
    X_m is the exact transform of the piecewise-linear column through the levels (height_fourier_weights), so the
    operator does not depend on how many levels sample a wave; the grid only sets how well the column is represented.
    The column's height (z_scale) enters pointwise, as the old branch's z features did. reference_length (z_scale units,
    Mm) must be at least twice the tallest column, so that a column's top and bottom are never neighbours in the
    periodic domain.

    Input: x [B, C, D, H, W], z_scale [B, D, H, W] or [B, 1, D, H, W]. Output: [B, C, D, H, W]
    """

    def __init__(self, channels, reference_length, min_wavelength, hidden=128, rank=0):
        super().__init__()
        if reference_length <= 0 or min_wavelength <= 0:
            raise ValueError("the height operator needs z_reference_length > 0 and z_min_wavelength > 0")
        self.reference_length = float(reference_length)
        self.min_wavelength = float(min_wavelength)
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

    def k_features(self):
        """The frequency MLPs' input [M, 2]: k over the largest wavenumber, and its square."""
        k = self.k * self.min_wavelength / (2 * math.pi)
        return torch.stack([k, k * k], dim=-1)

    def check_span(self, z_scale):
        span = float((z_scale.amax(dim=-3) - z_scale.amin(dim=-3)).max())
        if 2 * span > self.reference_length:
            raise ValueError(f"a column spans {span:.3f} in z_scale units, more than half of z_reference_length "
                             f"{self.reference_length}")

    def forward(self, x, z_scale):
        if z_scale.ndim == 4:
            z_scale = z_scale.unsqueeze(1)
        self.check_span(z_scale)
        x = x * self.input_gate(x) + self.z_proj(z_scale)
        z = z_scale - z_scale.amin(dim=2, keepdim=True)              # a shift of z changes nothing but rounding
        shared = bool((z == z[..., :1, :1]).all())                   # one grid for every column (simulations, mesh)
        if shared:
            column = z[0, 0, :, 0, 0]
            q_re, q_im = height_fourier_weights(column, self.k)      # [D, M]
            phase = column[:, None] * self.k
            x_re = torch.einsum("b c d h w, d m -> b c m h w", x, q_re)
            x_im = torch.einsum("b c d h w, d m -> b c m h w", x, q_im)
        else:
            columns = z[:, 0].permute(0, 2, 3, 1)                    # [B, H, W, D]
            q_re, q_im = height_fourier_weights(columns, self.k)     # [B, H, W, D, M]
            phase = z[:, 0, ..., None] * self.k                      # [B, D, H, W, M]
            x_re = torch.einsum("b c d h w, b h w d m -> b c m h w", x, q_re)
            x_im = torch.einsum("b c d h w, b h w d m -> b c m h w", x, q_im)
        cos, sin = torch.cos(phase), torch.sin(phase)
        features = self.k_features()
        gate = self.freq_mlp(features)                               # [M, 2]
        gr, gi = gate[:, 0, None, None], gate[:, 1, None, None]      # [M, 1, 1]
        a_re = torch.einsum("b i m h w, i o -> b o m h w", x_re, self.weight_real) - \
            torch.einsum("b i m h w, i o -> b o m h w", x_im, self.weight_imag)
        a_im = torch.einsum("b i m h w, i o -> b o m h w", x_im, self.weight_real) + \
            torch.einsum("b i m h w, i o -> b o m h w", x_re, self.weight_imag)
        y_re, y_im = a_re * gr - a_im * gi, a_re * gi + a_im * gr
        if self.rank:
            h = self.freq_lowrank(features)
            u = self.freq_u(h).reshape(self.modes, -1, self.rank)    # [M, C, r]
            v = self.freq_v(h).reshape(self.modes, self.rank, -1)    # [M, r, C]
            low_re = torch.einsum("b i m h w, m r i -> b r m h w", x_re, v)
            low_im = torch.einsum("b i m h w, m r i -> b r m h w", x_im, v)
            y_re = y_re + torch.einsum("b r m h w, m o r -> b o m h w", low_re, u)
            y_im = y_im + torch.einsum("b r m h w, m o r -> b o m h w", low_im, u)
        f = self.mode_factor[:, None, None]
        y_re, y_im = y_re * f, y_im * f
        if shared:
            y = torch.einsum("b o m h w, d m -> b o d h w", y_re, cos) - \
                torch.einsum("b o m h w, d m -> b o d h w", y_im, sin)
        else:
            y = torch.einsum("b o m h w, b d h w m -> b o d h w", y_re, cos) - \
                torch.einsum("b o m h w, b d h w m -> b o d h w", y_im, sin)
        return self.post(y / self.reference_length)


class BalancedVerticalPhysicsStack(nn.Module):
    """
    Stronger vertical branch:
      - multiscale depth processing
      - residual dilated depth blocks
      - channel mixing
      - learned depth gate
      - optional (z_modes > 0) spectral convolution along depth, gated, added to the residual blocks' output: along
        height with z_reference_length > 0 (SpectralConvHeight), along the level index with 0 (SpectralConvZ)
    """

    def __init__(self, channels, hidden=128, dropout=0.0, chunk=4, z_modes=0, z_reference_length=0.0):
        super().__init__()
        self.chunk = chunk
        self.z_reference_length = float(z_reference_length)

        self.z_proj = nn.Sequential(
            nn.Conv1d(4, hidden, 1, bias=False),
            nn.GELU(),
        )

        self.in_proj = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            _gn(hidden),
            nn.GELU(),
        )

        self.net = nn.Sequential(
            LiteMultiScaleVertical(hidden, kernels=(3, 7, 15), dropout=dropout),
            LiteVerticalResidualBlock(hidden, k=5, dilation=1, dropout=dropout),
            LiteVerticalResidualBlock(hidden, k=5, dilation=2, dropout=dropout),
            LiteVerticalResidualBlock(hidden, k=5, dilation=4, dropout=dropout),
            LiteVerticalResidualBlock(hidden, k=7, dilation=1, dropout=dropout),
        )

        self.depth_gate = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.Sigmoid(),
        )

        if z_modes:
            self.z_spec = (SpectralConvHeight(hidden, z_modes, self.z_reference_length) if self.z_reference_length > 0
                           else SpectralConvZ(hidden, z_modes))
            self.z_gate = nn.Sequential(
                nn.Conv1d(hidden, hidden, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.z_spec = None

        self.out_proj = nn.Conv1d(hidden, channels, 1, bias=False)
        self.out_gn = _gn(channels)

    def _z_features(self, z_scale):
        if z_scale.shape[-1] == 1:
            dz = torch.ones_like(z_scale)
        else:
            dz = torch.empty_like(z_scale)
            dz[:, :, 1:-1] = 0.5 * (z_scale[:, :, 2:] - z_scale[:, :, :-2])
            dz[:, :, 0] = z_scale[:, :, 1] - z_scale[:, :, 0]
            dz[:, :, -1] = z_scale[:, :, -1] - z_scale[:, :, -2]

        z_min = z_scale.min(dim=-1, keepdim=True).values
        z_span = (z_scale.max(dim=-1, keepdim=True).values - z_min).clamp_min(1e-6)
        z_norm = (z_scale - z_min) / z_span
        dz_norm = dz.abs() / dz.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)

        # Keep absolute z_scale as the first channel; derived channels only add geometry context.
        return torch.cat(
            [
                z_scale,
                z_norm,
                dz_norm,
                z_span.expand_as(z_scale),
            ],
            dim=1,
        )

    def check_height_span(self, z_scale):
        """SpectralConvHeight needs every column to span at most half the z reference length."""
        span = float((z_scale.amax(dim=2) - z_scale.amin(dim=2)).max())
        if 2 * span > self.z_reference_length:
            raise ValueError(f"a column spans {span:.3f} in z_scale units, more than half of z_reference_length "
                             f"{self.z_reference_length}")

    def columns(self, xi, zi):
        """The branch on columns xi [N, C, D] with depths zi [N, 1, D], before out_gn (shared with the distributed
        and batched inference paths)."""
        h = self.in_proj(xi) + self.z_proj(self._z_features(zi))
        y = self.net(h)
        if self.z_spec is not None:
            y = y + self.z_gate(h) * (self.z_spec(h, zi) if self.z_reference_length > 0 else self.z_spec(h))
        y = y * self.depth_gate(y)
        return self.out_proj(y)

    def forward(self, x, z_scale):
        B, C, D, H, W = x.shape
        if z_scale.ndim == 4:
            z_scale = z_scale.unsqueeze(1)
        if z_scale.shape != (B, 1, D, H, W):
            raise ValueError(
                f"z_scale must be [B, D, H, W] or [B, 1, D, H, W], got {tuple(z_scale.shape)}"
            )
        if self.z_spec is not None and self.z_reference_length > 0:
            self.check_height_span(z_scale)
        x = x.permute(0, 3, 4, 1, 2).reshape(B, H * W, C, D)
        z_scale = z_scale.permute(0, 3, 4, 1, 2).reshape(B, H * W, 1, D)

        chunks = []
        for i in range(0, H * W, self.chunk):
            j = min(i + self.chunk, H * W)
            xi = x[:, i:j].reshape(-1, C, D)
            zi = z_scale[:, i:j].reshape(-1, 1, D)
            yi = self.columns(xi, zi).reshape(B, j - i, C, D)
            chunks.append(yi)

        y = torch.cat(chunks, dim=1)
        y = y.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2).contiguous()
        return self.out_gn(y)


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
    Spectral and vertical branches are both strengthened and fused by:
      - same input to both branches
      - normalization on both outputs
      - adaptive branch gating before learned fusion
      - learned channel-mixing fusion after concatenation
      - small corrective pointwise/mlp path (open_side_paths: both start at 0.5 instead of 0.15, and the MLP output
        is not squashed by tanh)
    vertical_operator: the vertical branch is HeightSpectralConv, the fusion has no 3x3x3 convolution and every
    GroupNorm weights levels by their height (DepthWeightedGroupNorm): nothing counts grid points in any direction.
    """

    def __init__(
        self,
        width=128,
        dropout=0.05,
        spec_dropout=0.0,
        vertical_dropout=0.0,
        hidden_spectral=128,
        hidden_vertical=128,
        z_spectral_modes=0,
        z_reference_length=0.0,
        freq_mix_rank=0,
        open_side_paths=False,
        vertical_operator=False,
        z_min_wavelength=0.0,
    ):
        super().__init__()
        self.open_side_paths = bool(open_side_paths)
        self.vertical_operator = bool(vertical_operator)
        norm = _wgn if self.vertical_operator else _gn

        self.spec = SpectralConv2dFull(
            width,
            width,
            hidden=hidden_spectral,
            post_mix_expansion=2,
            rank=freq_mix_rank,
        )

        if self.vertical_operator:
            if z_spectral_modes:
                raise ValueError("vertical_operator replaces the vertical stack: z_spectral_modes does not apply")
            self.vertical = HeightSpectralConv(width, z_reference_length, z_min_wavelength, hidden=hidden_vertical,
                                               rank=freq_mix_rank)
        else:
            self.vertical = BalancedVerticalPhysicsStack(
                width,
                hidden=hidden_vertical,
                dropout=dropout,
                chunk=4,
                z_modes=z_spectral_modes,
                z_reference_length=z_reference_length,
            )

        self.pw = nn.Sequential(
            nn.Conv3d(width, width * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(width * 2, width, 1, bias=False),
        )

        self.mlp = PointwiseMLP(width, expansion=2, dropout=dropout)

        self.norm_spec = norm(width)
        self.norm_vert = norm(width)
        self.branch_gate = nn.Sequential(
            nn.Conv3d(3 * width, width, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(width, 2 * width, 1, bias=True),
            nn.Sigmoid(),
        )
        local = [] if self.vertical_operator else [
            nn.Conv3d(
                2 * width,
                2 * width,
                kernel_size=3,
                padding=1,
                groups=2 * width,
                bias=False,
            ),
            nn.GELU(),
        ]
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * width, 2 * width, 1, bias=False),
            nn.GELU(),
            *local,
            nn.Conv3d(2 * width, width, 1, bias=False),
        )
        self.norm_fuse = norm(width)
        self.norm_pw = norm(width)
        self.norm_mlp = norm(width)

        side = 0.5 if self.open_side_paths else 0.15
        self.res_fused = nn.Parameter(torch.tensor([1.0]))
        self.res_pw = nn.Parameter(torch.tensor([side]))
        self.res_mlp = nn.Parameter(torch.tensor([side]))

        self.act = nn.GELU()
        self.spec_drop = nn.Dropout(spec_dropout) if spec_dropout > 0 else nn.Identity()
        self.vert_drop = (
            nn.Dropout(vertical_dropout) if vertical_dropout > 0 else nn.Identity()
        )

    def mlp_activation(self, y):
        """The MLP path's output nonlinearity (shared with the inference paths): tanh, or none with open_side_paths."""
        return y if self.open_side_paths else torch.tanh(y)

    @staticmethod
    def _norm(module, x, w):
        """A GroupNorm of the block, weighted by depth_weights w with vertical_operator (w is None otherwise)."""
        return module(x) if w is None else module(x, w)

    def _fuse_branches(self, x, spec, vert, w=None):
        gates = self.branch_gate(torch.cat([x, spec, vert], dim=1))
        spec_gate, vert_gate = gates.chunk(2, dim=1)

        spec_mix = spec * spec_gate
        vert_mix = vert * vert_gate

        fused = self.fuse(torch.cat([spec_mix, vert_mix], dim=1))
        fused = fused + 0.5 * (spec_mix + vert_mix)
        fused = self.act(self._norm(self.norm_fuse, fused, w))
        return fused, spec_mix, vert_mix, spec_gate, vert_gate

    def forward(self, x, z_scale, dx, dy, collect_stats=False, branch_mask=None):
        w = depth_weights(z_scale) if self.vertical_operator else None
        if not collect_stats and branch_mask is None:
            residual = x

            spec = self.spec_drop(self.act(self._norm(self.norm_spec, self.spec(x, dx, dy), w)))
            vert = self.vert_drop(self.act(self._norm(self.norm_vert, self.vertical(x, z_scale), w)))
            fused, _, _, _, _ = self._fuse_branches(x, spec, vert, w)

            x1 = residual + self.res_fused * fused

            pw = self._norm(self.norm_pw, self.pw(x1), w)
            x2 = x1 + self.res_pw * pw

            mlp = self.mlp_activation(self._norm(self.norm_mlp, self.mlp(x2), w))
            out = x2 + self.res_mlp * mlp

            return out

        residual = x
        stats = {} if collect_stats else None
        branch_mask = branch_mask or {}

        spec = self.spec_drop(self.act(self._norm(self.norm_spec, self.spec(x, dx, dy), w)))
        vert = self.vert_drop(self.act(self._norm(self.norm_vert, self.vertical(x, z_scale), w)))

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

        pw = self._norm(self.norm_pw, self.pw(x1), w)
        x2 = x1 + (self.res_pw * pw_mask) * pw

        mlp = self.mlp_activation(self._norm(self.norm_mlp, self.mlp(x2), w))
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
    def __init__(
        self,
        in_channels=6,
        out_channels=6,
        width=128,
        n_layers=4,
        dropout=0.05,
        spec_dropout=0.0,
        vertical_dropout=0.0,
        spec_dropout_layers=None,
        vertical_dropout_layers=None,
        checkpoint_blocks=True,
        hidden_spectral=128,
        hidden_vertical=128,
        z_spectral_modes=0,
        z_reference_length=0.0,
        freq_mix_rank=0,
        open_side_paths=False,
        vertical_operator=False,
        z_min_wavelength=0.0,
        output_mode="log_population",
        species=None,
        rho_channel=5,
        total_bound=6.0,
        temperature_channel=0,
        ne_channel=4,
    ):
        super().__init__()

        # output_mode "log_population": the head predicts normalised log10 populations directly.
        # "level_fraction": per species, the head predicts one departure logit delta_i per level and one total
        # correction u:
        #     log10 n_i = log10(rho / particle mass) + s + log10 softmax(ln f*_i + delta_i),   s = total_bound * tanh(u / total_bound)
        # f*_i(T, ne) is MULTI3D's LTE level fraction (multi3d_lte_log_weights, from the input T and ne and the species'
        # "multi3d_lte" atom data), so delta_i is ln b_i, the departure coefficient, up to a constant common to the
        # species' levels, and delta = 0 gives MULTI3D's LTE populations. The softmax is each level's share of the
        # species total, and s the log10 ratio of that total to rho / mass. Any set of positive levels whose total lies
        # within total_bound dex of rho / mass is reproduced exactly; the bound only keeps populations finite in
        # float32. The output stays normalised log10 populations, so the loss and the inference paths are unchanged.
        # Capacity switches, all off by default so that existing checkpoints load and give the same outputs:
        # z_spectral_modes (spectral convolution along depth in each vertical branch: along height, independent of the
        # vertical grid, with z_reference_length > 0 in z_scale units (Mm), at least twice the tallest column; along the
        # level index with 0, as in the first capacity screens), freq_mix_rank (per-frequency low-rank channel mixing in
        # each horizontal spectral branch), open_side_paths (see FFNOBlock3dBalanced).
        # vertical_operator: a neural operator in all three directions, nothing counting grid points: the vertical
        # branch is HeightSpectralConv (spectral in height on each column's own heights, wavenumbers 2 pi m /
        # z_reference_length down to wavelength z_min_wavelength, z_scale units), with freq_mix_rank as its low-rank
        # mixing too; the fusion has no 3x3x3 convolution; every GroupNorm, the lift's included, weights levels by height.
        if output_mode not in ("log_population", "level_fraction"):
            raise ValueError(f"unknown output_mode {output_mode!r}")
        self.output_mode = output_mode
        head_channels = out_channels
        if output_mode == "level_fraction":
            if not species:
                raise ValueError("output_mode='level_fraction' needs species")
            self.species_levels = [int(s["levels"]) for s in species]
            if sum(self.species_levels) != out_channels:
                raise ValueError(f"species levels {self.species_levels} do not add up to out_channels={out_channels}")
            head_channels = out_channels + len(self.species_levels)
            self.total_bound = float(total_bound)
            self.rho_channel = int(rho_channel)
            self.register_buffer("log10_particle_mass",
                                 torch.tensor([float(s["log10_particle_mass"]) for s in species], dtype=torch.float32))
            # normalisation of the inputs (for rho) and of the output; set from the dataset with set_normalization
            # and saved with the checkpoint
            self.register_buffer("mean_X", torch.zeros(in_channels))
            self.register_buffer("std_X", torch.ones(in_channels))
            self.register_buffer("mean_Y", torch.zeros(out_channels))
            self.register_buffer("std_Y", torch.ones(out_channels))
            missing = [i for i, s in enumerate(species) if "multi3d_lte" not in s]
            if missing:
                raise ValueError(f"output_mode='level_fraction' needs each species' multi3d_lte atom data (g, ev, ion); "
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

        self.vertical_operator = bool(vertical_operator)
        self.lift = nn.Sequential(
            nn.Conv3d(in_channels, width, 1, bias=False),
            _wgn(width) if self.vertical_operator else _gn(width),
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
                z_spectral_modes=z_spectral_modes,
                z_reference_length=z_reference_length,
                freq_mix_rank=freq_mix_rank,
                vertical_operator=vertical_operator,
                z_min_wavelength=z_min_wavelength,
                open_side_paths=open_side_paths,
            )
            for layer_idx in range(n_layers)
        ])

        self.proj1 = nn.Conv3d(width, width * 2, 1, bias=False)
        self.proj2 = nn.Conv3d(width * 2, head_channels, 1)
        if output_mode == "level_fraction":
            # the total corrections start at s = 0, the total rho / mass
            with torch.no_grad():
                c = 0
                for levels in self.species_levels:
                    self.proj2.weight[c + levels].zero_()
                    self.proj2.bias[c + levels].zero_()
                    c += levels + 1

        self.act = nn.GELU()

    def set_normalization(self, mean_X, std_X, mean_Y, std_Y):
        """Copy the dataset's normalisation into the level-fraction head (a no-op for log_population)."""
        if self.output_mode != "level_fraction":
            return
        for name, value in (("mean_X", mean_X), ("std_X", std_X), ("mean_Y", mean_Y), ("std_Y", std_Y)):
            buffer = getattr(self, name)
            buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype).broadcast_to(buffer.shape))

    def head_output(self, head, x_in):
        """Normalised log10 populations from the head's channels and the normalised inputs x_in [B, Cin, D, H, W]."""
        if self.output_mode == "log_population":
            return head
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
        if not self.vertical_operator:
            return self.lift(x)
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

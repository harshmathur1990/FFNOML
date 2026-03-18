import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# --------------------------------------------
# Helpers
# --------------------------------------------
def _to_spacing_value(spacing, x):
    """
    Convert spacing to a scalar float/tensor usable by torch.fft.fftfreq.

    Accepts:
      - python float/int
      - 0-d torch tensor
      - 1-element torch tensor
    """
    if isinstance(spacing, (float, int)):
        return float(spacing)

    if torch.is_tensor(spacing):
        if spacing.numel() != 1:
            raise ValueError(
                "For this spectral layer, spacing must currently be scalar "
                "(python float/int or tensor with one element)."
            )
        return float(spacing.detach().item())

    raise TypeError(f"Unsupported spacing type: {type(spacing)}")


# --------------------------------------------
# Pointwise MLP
# --------------------------------------------
class PointwiseMLP(nn.Module):
    """Channel MLP using 1x1x1 convs."""

    def __init__(self, width, expansion=2, dropout=0.0):
        super().__init__()
        hidden = width * expansion
        self.fc1 = nn.Conv3d(width, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, width, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# --------------------------------------------
# Factorized Spectral Convs
# --------------------------------------------
class SpectralConv2dFull(nn.Module):
    """
    Full-spectrum metric-aware spectral convolution (NO truncation).

    Differences vs original:
      - uses ALL Fourier modes
      - no zeroing of high frequencies
      - weights are global (Cin x Cout)
      - frequency-dependent gating handles spectral variation
    """

    def __init__(self, in_channels, out_channels, hidden=32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        scale = 1.0 / (in_channels * out_channels)

        # global complex channel mixing
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, dtype=torch.float32)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, dtype=torch.float32)
        )

        # (kx, ky) -> complex gate
        # remove useless sign(kx)
        self.freq_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, dx, dy):
        B, Cin, D, H, W = x.shape

        dx = _to_spacing_value(dx, x)
        dy = _to_spacing_value(dy, x)

        orig_dtype = x.dtype

        with torch.amp.autocast("cuda", enabled=False):
            x32 = x.float()

            x_ft = torch.fft.rfft2(x32, dim=(-2, -1))
            x_ft_real = x_ft.real
            x_ft_imag = x_ft.imag

            # full frequencies (NO truncation)
            ky = torch.fft.fftfreq(H, d=dy, device=x.device)
            kx = torch.fft.rfftfreq(W, d=dx, device=x.device)

            ky = 2 * math.pi * ky
            kx = 2 * math.pi * kx

            k_scale = 1e5   # tune this
            kx = kx * k_scale
            ky = ky * k_scale

            ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")

            k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + 1e-12)

            k_feat = torch.stack([
                kx_grid,
                ky_grid,
                k_mag,
                torch.sign(ky_grid)
            ], dim=-1)

            gate = self.freq_mlp(k_feat)
            gate_real = gate[..., 0]
            gate_imag = gate[..., 1]

            # global complex weights
            wr = self.weight_real[:, :, None, None]
            wi = self.weight_imag[:, :, None, None]

            gr = gate_real[None, None, :, :]
            gi = gate_imag[None, None, :, :]

            w_real = wr * gr - wi * gi
            w_imag = wr * gi + wi * gr

            # complex multiplication
            a = x_ft_real
            b = x_ft_imag
            c = w_real
            d = w_imag

            out_ft_real = (
                torch.einsum("b i d y x, i o y x -> b o d y x", a, c)
                - torch.einsum("b i d y x, i o y x -> b o d y x", b, d)
            )
            out_ft_imag = (
                torch.einsum("b i d y x, i o y x -> b o d y x", a, d)
                + torch.einsum("b i d y x, i o y x -> b o d y x", b, c)
            )

            out_ft = torch.complex(out_ft_real, out_ft_imag)

            y = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))

        return y.to(orig_dtype)


# --------------------------------------------
# Depth mixer
# --------------------------------------------
class DepthMixer1d(nn.Module):
    """
    Local 1D convolution along depth (z) for vertical coupling.
    Operates on each (y,x) column independently.

    Input/Output: [B, C, D, H, W]
    """

    def __init__(self, channels, k=7, dropout=0.0):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(k, 1, 1),
            padding=(p, 0, 0),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        return x


# --------------------------------------------
# FFNO block
# --------------------------------------------
class FFNOBlock3d(nn.Module):
    """
    Factorized block:
      - global spectral mixing in (H,W) per depth plane
      - local vertical mixing in z via conv
      - residual + pointwise MLP
    """

    def __init__(
        self,
        width,
        z_kernel=7,
        dropout=0.0,
        mlp_expansion=2,
    ):
        super().__init__()

        self.spec_hw = SpectralConv2dFull(
            width,
            width
        )
        self.pw_hw = nn.Conv3d(width, width, kernel_size=1)

        self.z_mix = DepthMixer1d(width, k=z_kernel, dropout=dropout)
        self.mlp = PointwiseMLP(
            width,
            expansion=mlp_expansion,
            dropout=dropout,
        )

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.alpha_spec = nn.Parameter(torch.tensor(1.0))
        self.alpha_pw   = nn.Parameter(torch.tensor(1.0))
        self.alpha_z    = nn.Parameter(torch.tensor(1.0))
        self.alpha_mlp  = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, dx, dy):
        # HW spectral + pointwise, residual
        # y = self.spec_hw(x, dx=dx, dy=dy) + self.pw_hw(x)
        y = self.alpha_spec * self.spec_hw(x, dx=dx, dy=dy) + \
            self.alpha_pw   * self.pw_hw(x)
        y = self.act(y)
        y = self.drop(y)
        x = x + y

        # vertical local mixing, residual
        z = self.alpha_z * self.z_mix(x)
        x = x + z

        m = self.alpha_mlp * self.mlp(x)
        x = x + m

        return x


# --------------------------------------------
# Full FFNO model
# --------------------------------------------
class FFNO3D(nn.Module):
    """
    RT-friendly Factorized FNO:
      - full volume in/out
      - spectral in (x,y), vertical conv in z

    Input:  [B, Cin, D, H, W]
    Output: [B, Cout, D, H, W]

    Forward requires dx and dy so the spectral operator is metric-aware.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        width=64,
        n_layers=6,
        z_kernel=9,
        dropout=0.1,
        mlp_expansion=2,
        padding=0,
    ):
        super().__init__()

        self.padding = padding

        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                FFNOBlock3d(
                    width=width,
                    z_kernel=z_kernel,
                    dropout=dropout,
                    mlp_expansion=mlp_expansion,
                )
                for _ in range(n_layers)
            ]
        )

        self.proj1 = nn.Conv3d(width, width * 2, kernel_size=1)
        self.proj2 = nn.Conv3d(width * 2, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x, dx, dy):
        # x: [B, Cin, D, H, W]

        if self.padding > 0:
            p = self.padding
            x = F.pad(x, (p, p, p, p, p, p), mode="replicate")

        x = self.lift(x)

        for blk in self.blocks:
            x = checkpoint(lambda t: blk(t, dx=dx, dy=dy), x, use_reentrant=False)

        x = self.act(self.proj1(x))
        y = self.proj2(x)

        if self.padding > 0:
            p = self.padding
            y = y[:, :, p:-p, p:-p, p:-p]

        return y

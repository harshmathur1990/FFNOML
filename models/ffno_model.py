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


# ============================================================
# Pointwise MLP (WEAK)
# ============================================================
class PointwiseMLP(nn.Module):
    def __init__(self, width, expansion=2, dropout=0.0):
        super().__init__()
        hidden = width * expansion

        self.fc1 = nn.Conv3d(width, hidden, 1)
        self.fc2 = nn.Conv3d(hidden, width, 1)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ============================================================
# LIGHTWEIGHT VERTICAL STACK (~60k params)
# ============================================================
class DepthwiseSeparable1D(nn.Module):
    def __init__(self, ch, k=5, dilation=1, dropout=0.0):
        super().__init__()

        pad = dilation * (k // 2)

        self.dw = nn.Conv1d(
            ch, ch,
            kernel_size=k,
            padding=pad,
            dilation=dilation,
            groups=ch,
            bias=False,
        )

        self.pw = nn.Conv1d(ch, ch, 1, bias=False)

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

        self.b1 = DepthwiseSeparable1D(ch, k, dilation, dropout)
        self.b2 = DepthwiseSeparable1D(ch, k, 1, dropout)

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
            nn.Conv1d(
                ch, ch,
                kernel_size=k,
                padding=k // 2,
                groups=ch,
                bias=False
            )
            for k in kernels
        ])

        self.mix = nn.Conv1d(ch, ch, 1, bias=False)

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


class BalancedVerticalPhysicsStack(nn.Module):
    """
    ~60k parameter vertical physics operator
    """
    def __init__(self, channels, hidden=80, dropout=0.0):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            _gn(hidden),
            nn.GELU(),
        )

        self.net = nn.Sequential(
            LiteMultiScaleVertical(hidden, dropout=dropout),
            LiteVerticalResidualBlock(hidden, dilation=1, dropout=dropout),
            LiteVerticalResidualBlock(hidden, dilation=2, dropout=dropout),
            LiteVerticalResidualBlock(hidden, dilation=4, dropout=dropout),
        )

        self.out_proj = nn.Conv1d(hidden, channels, 1, bias=False)
        self.out_gn = _gn(channels)

    def forward(self, x):
        B, C, D, H, W = x.shape

        x = x.permute(0, 3, 4, 1, 2).reshape(B, H * W, C, D)

        chunk = 128
        outs = []

        for i in range(0, H * W, chunk):
            xi = x[:, i:i+chunk].reshape(-1, C, D)

            yi = self.in_proj(xi)
            yi = self.net(yi)
            yi = self.out_proj(yi)

            yi = yi.reshape(B, -1, C, D)
            outs.append(yi)

        y = torch.cat(outs, dim=1)
        y = y.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2)

        return self.out_gn(y)


# ============================================================
# SPECTRAL (DOMINANT)
# ============================================================
class SpectralConv2dFull(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=96):
        super().__init__()

        scale = 1.0 / math.sqrt(in_channels)

        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels))

        self.freq_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, dx, dy):
        B, C, D, H, W = x.shape

        dx = _to_spacing_value(dx, x)
        dy = _to_spacing_value(dy, x)

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        ky = torch.fft.fftfreq(H, d=dy, device=x.device)
        kx = torch.fft.rfftfreq(W, d=dx, device=x.device)

        ky, kx = torch.meshgrid(ky, kx, indexing="ij")

        k_feat = torch.stack([
            kx, ky,
            torch.sqrt(kx**2 + ky**2 + 1e-12),
            torch.sign(ky),
        ], dim=-1)

        gate = self.freq_mlp(k_feat)

        wr = self.weight_real[:, :, None, None]
        wi = self.weight_imag[:, :, None, None]

        gr = gate[..., 0][None, None]
        gi = gate[..., 1][None, None]

        w_real = wr * gr - wi * gi
        w_imag = wr * gi + wi * gr

        xr, xi = x_ft.real, x_ft.imag

        out_r = torch.einsum("b i d y x, i o y x -> b o d y x", xr, w_real) - \
                torch.einsum("b i d y x, i o y x -> b o d y x", xi, w_imag)

        out_i = torch.einsum("b i d y x, i o y x -> b o d y x", xr, w_imag) + \
                torch.einsum("b i d y x, i o y x -> b o d y x", xi, w_real)

        out = torch.complex(out_r, out_i)

        y = torch.fft.irfft2(out, s=(H, W), dim=(-2, -1))

        return y


# ============================================================
# BALANCED BLOCK
# ============================================================
class FFNOBlock3dBalanced(nn.Module):
    def __init__(self, width=128, dropout=0.05):
        super().__init__()

        self.spec = SpectralConv2dFull(width, width)
        self.vertical = BalancedVerticalPhysicsStack(width, dropout=dropout)

        self.pw = nn.Sequential(
            nn.Conv3d(width, width, 1, bias=False),
            nn.Tanh(),   # bounded correction
        )

        self.mlp = PointwiseMLP(width, expansion=2, dropout=dropout)

        self.norm1 = _gn(width)
        self.norm2 = _gn(width)
        self.norm3 = _gn(width)
        self.norm4 = _gn(width)

        self.res_h = nn.Parameter(torch.tensor([1.0]))
        self.res_v = nn.Parameter(torch.tensor([1.0]))
        self.res_m = nn.Parameter(torch.tensor([0.2]))

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, dx, dy):
        # -------- horizontal --------
        spec = self.norm1(self.spec(x, dx, dy))
        pw   = self.norm2(self.pw(x))

        y = spec + 0.3 * pw
        y = self.act(y)
        y = self.drop(y)

        x = x + self.res_h * y

        # -------- vertical --------
        vert = self.norm3(self.vertical(x))
        z = self.act(vert)
        z = self.drop(z)

        x = x + self.res_v * z

        # -------- correction --------
        m = 0.2 * torch.tanh(self.norm4(self.mlp(x)))
        x = x + self.res_m * m

        return x


# ============================================================
# FULL MODEL
# ============================================================
class FFNO3D(nn.Module):
    def __init__(
        self,
        in_channels=6,
        out_channels=6,
        width=128,
        n_layers=4,
        dropout=0.05,
        checkpoint_blocks=True,
    ):
        super().__init__()

        self.checkpoint_blocks = checkpoint_blocks

        self.lift = nn.Conv3d(in_channels, width, 1)

        self.blocks = nn.ModuleList([
            FFNOBlock3dBalanced(width=width, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.proj1 = nn.Conv3d(width, width * 2, 1)
        self.proj2 = nn.Conv3d(width * 2, out_channels, 1)

        self.act = nn.GELU()

    def _run_block(self, blk, x, dx, dy):
        if self.checkpoint_blocks and self.training:
            return checkpoint(
                lambda t: blk(t, dx, dy),
                x,
                use_reentrant=False
            )
        return blk(x, dx, dy)

    def forward(self, x, dx, dy):
        x = self.lift(x)

        for blk in self.blocks:
            x = self._run_block(blk, x, dx, dy)

        x = self.act(self.proj1(x))
        x = self.proj2(x)

        return x

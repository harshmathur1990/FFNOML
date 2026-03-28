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


class BalancedVerticalPhysicsStack(nn.Module):
    """
    Stronger vertical branch:
      - multiscale depth processing
      - residual dilated depth blocks
      - channel mixing
      - learned depth gate
    """

    def __init__(self, channels, hidden=128, dropout=0.0, chunk=16):
        super().__init__()
        self.chunk = chunk

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

        self.out_proj = nn.Conv1d(hidden, channels, 1, bias=False)
        self.out_gn = _gn(channels)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape

        # [B, H*W, C, D]
        x = x.permute(0, 3, 4, 1, 2).reshape(B, H * W, C, D)

        outs = []
        for i in range(0, H * W, self.chunk):
            xi = x[:, i:i + self.chunk].reshape(-1, C, D)  # [B*chunk, C, D]

            yi = self.in_proj(xi)
            yi = self.net(yi)
            yi = yi * self.depth_gate(yi)
            yi = self.out_proj(yi)

            yi = yi.reshape(B, -1, C, D)
            outs.append(yi)

        y = torch.cat(outs, dim=1)  # [B, H*W, C, D]
        y = y.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2)  # [B, C, D, H, W]

        return self.out_gn(y)


# ============================================================
# Stronger Spectral Branch
# ============================================================
class SpectralConv2dFull(nn.Module):
    """
    Full-spectrum metric-aware spectral convolution over (H, W),
    applied independently at each depth plane.

    Input/Output: [B, C, D, H, W]
    """

    def __init__(self, in_channels, out_channels, hidden=128, post_mix_expansion=2):
        super().__init__()

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

        post_hidden = out_channels * post_mix_expansion
        
        self.post = nn.Sequential(
            nn.Conv3d(out_channels, post_hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(post_hidden, out_channels, 1, bias=False),
        )

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

        k_feat = torch.stack(
            [
                kx,
                ky,
                torch.sqrt(kx ** 2 + ky ** 2 + 1e-12),
                torch.sign(ky),
            ],
            dim=-1,
        )  # [H, Wf, 4]

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
        y = torch.fft.irfft2(out, s=(H, W), dim=(-2, -1))
        y = self.post(y)
        return y


# ============================================================
# Balanced Strong Block
# ============================================================
class FFNOBlock3dBalanced(nn.Module):
    """
    Spectral and vertical branches are both strengthened and kept balanced by:
      - same input to both branches
      - normalization on both outputs
      - learned softmax branch weights
      - small corrective pointwise/mlp path
    """

    def __init__(self, width=128, dropout=0.05):
        super().__init__()

        self.spec = SpectralConv2dFull(
            width,
            width,
            hidden=128,
            post_mix_expansion=2,
        )

        self.vertical = BalancedVerticalPhysicsStack(
            width,
            hidden=48,
            dropout=dropout,
            chunk=16,
        )

        self.pw = nn.Sequential(
            nn.Conv3d(width, width * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(width * 2, width, 1, bias=False),
        )

        self.mlp = PointwiseMLP(width, expansion=2, dropout=dropout)

        self.norm_spec = _gn(width)
        self.norm_vert = _gn(width)
        self.norm_pw = _gn(width)
        self.norm_mlp = _gn(width)

        # Softmax keeps both on comparable footing
        self.branch_logits = nn.Parameter(torch.tensor([0.0, 0.0]))  # [spec, vert]

        self.res_fused = nn.Parameter(torch.tensor([1.0]))
        self.res_pw = nn.Parameter(torch.tensor([0.15]))
        self.res_mlp = nn.Parameter(torch.tensor([0.15]))

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, dx, dy):
        # both branches see same input
        spec = self.drop(self.act(self.norm_spec(self.spec(x, dx, dy))))
        vert = self.drop(self.act(self.norm_vert(self.vertical(x))))

        w = torch.softmax(self.branch_logits, dim=0)
        fused = w[0] * spec + w[1] * vert
        x = x + self.res_fused * fused

        # small corrective paths only
        pw = self.norm_pw(self.pw(x))
        x = x + self.res_pw * pw

        mlp = torch.tanh(self.norm_mlp(self.mlp(x)))
        x = x + self.res_mlp * mlp

        return x


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
        checkpoint_blocks=True,
    ):
        super().__init__()

        self.checkpoint_blocks = checkpoint_blocks

        self.lift = nn.Sequential(
            nn.Conv3d(in_channels, width, 1, bias=False),
            _gn(width),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            FFNOBlock3dBalanced(width=width, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.proj1 = nn.Conv3d(width, width * 2, 1, bias=False)
        self.proj2 = nn.Conv3d(width * 2, out_channels, 1)

        self.act = nn.GELU()

    def _run_block(self, blk, x, dx, dy):
        if self.checkpoint_blocks and self.training:
            return checkpoint(
                lambda t: blk(t, dx, dy),
                x,
                use_reentrant=False,
            )
        return blk(x, dx, dy)

    def forward(self, x, dx, dy):
        x = self.lift(x)

        for blk in self.blocks:
            x = self._run_block(blk, x, dx, dy)

        x = self.act(self.proj1(x))
        x = self.proj2(x)
        return x

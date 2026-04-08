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


def _resolve_layer_dropout(rate, layers, layer_idx):
    if not layers:
        return 0.0
    if layer_idx in set(layers):
        return float(rate)
    return 0.0


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

    def __init__(self, channels, hidden=128, dropout=0.0, chunk=4):
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
        B, C, D, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(B, H * W, C, D)

        chunks = []
        for i in range(0, H * W, self.chunk):
            j = min(i + self.chunk, H * W)
            xi = x[:, i:j].reshape(-1, C, D)

            yi = self.in_proj(xi)
            yi = self.net(yi)
            yi = yi * self.depth_gate(yi)
            yi = self.out_proj(yi)

            yi = yi.reshape(B, j - i, C, D)
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
    Spectral and vertical branches are both strengthened and fused by:
      - same input to both branches
      - normalization on both outputs
      - learned channel-mixing fusion after concatenation
      - small corrective pointwise/mlp path
    """

    def __init__(
        self,
        width=128,
        dropout=0.05,
        spec_dropout=0.0,
        vertical_dropout=0.0,
    ):
        super().__init__()

        self.spec = SpectralConv2dFull(
            width,
            width,
            hidden=128,
            post_mix_expansion=2,
        )

        self.vertical = BalancedVerticalPhysicsStack(
            width,
            hidden=32,
            dropout=dropout,
            chunk=4,
        )

        self.pw = nn.Sequential(
            nn.Conv3d(width, width * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(width * 2, width, 1, bias=False),
        )

        self.mlp = PointwiseMLP(width, expansion=2, dropout=dropout)

        self.norm_spec = _gn(width)
        self.norm_vert = _gn(width)
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * width, 2 * width, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(2 * width, width, 1, bias=False),
        )
        self.norm_fuse = _gn(width)
        self.norm_pw = _gn(width)
        self.norm_mlp = _gn(width)

        self.res_fused = nn.Parameter(torch.tensor([1.0]))
        self.res_pw = nn.Parameter(torch.tensor([0.15]))
        self.res_mlp = nn.Parameter(torch.tensor([0.15]))

        self.act = nn.GELU()
        self.spec_drop = nn.Dropout(spec_dropout) if spec_dropout > 0 else nn.Identity()
        self.vert_drop = (
            nn.Dropout(vertical_dropout) if vertical_dropout > 0 else nn.Identity()
        )

    def forward(self, x, dx, dy, collect_stats=False, branch_mask=None):
        if not collect_stats and branch_mask is None:
            residual = x

            spec = self.spec_drop(self.act(self.norm_spec(self.spec(x, dx, dy))))
            vert = self.vert_drop(self.act(self.norm_vert(self.vertical(x))))
            fused = self.act(self.norm_fuse(self.fuse(torch.cat([spec, vert], dim=1))))

            x1 = residual + self.res_fused * fused

            pw = self.norm_pw(self.pw(x1))
            x2 = x1 + self.res_pw * pw

            mlp = torch.tanh(self.norm_mlp(self.mlp(x2)))
            out = x2 + self.res_mlp * mlp

            return out

        residual = x
        stats = {} if collect_stats else None
        branch_mask = branch_mask or {}

        spec = self.spec_drop(self.act(self.norm_spec(self.spec(x, dx, dy))))
        vert = self.vert_drop(self.act(self.norm_vert(self.vertical(x))))

        spec_mask = float(branch_mask.get("spec", 1.0))
        vert_mask = float(branch_mask.get("vertical", 1.0))
        pw_mask = float(branch_mask.get("pw", 1.0))
        mlp_mask = float(branch_mask.get("mlp", 1.0))

        spec_eff = spec_mask * spec
        vert_eff = vert_mask * vert
        fused = self.act(
            self.norm_fuse(self.fuse(torch.cat([spec_eff, vert_eff], dim=1)))
        )

        x1 = residual + self.res_fused * fused

        pw = self.norm_pw(self.pw(x1))
        x2 = x1 + (self.res_pw * pw_mask) * pw

        mlp = torch.tanh(self.norm_mlp(self.mlp(x2)))
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
                stats["spec_contrib"] = (self.res_fused * spec_eff).abs().mean().item()
                stats["vertical_contrib"] = (self.res_fused * vert_eff).abs().mean().item()
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
    ):
        super().__init__()

        self.checkpoint_blocks = checkpoint_blocks

        self.lift = nn.Sequential(
            nn.Conv3d(in_channels, width, 1, bias=False),
            _gn(width),
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
            )
            for layer_idx in range(n_layers)
        ])

        self.proj1 = nn.Conv3d(width, width * 2, 1, bias=False)
        self.proj2 = nn.Conv3d(width * 2, out_channels, 1)

        self.act = nn.GELU()

    def _run_block(self, blk, x, dx, dy, collect_stats=False, branch_mask=None):
        if self.checkpoint_blocks and self.training and not collect_stats:
            return checkpoint(
                lambda t: blk(t, dx, dy, collect_stats=False, branch_mask=None),
                x,
                use_reentrant=False,
            )
        return blk(x, dx, dy, collect_stats=collect_stats, branch_mask=branch_mask)

    def forward(self, x, dx, dy, collect_stats=False, branch_mask=None):
        x = self.lift(x)
        all_stats = [] if collect_stats else None

        for i, blk in enumerate(self.blocks):
            if collect_stats:
                x, s = self._run_block(
                    blk,
                    x,
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
                    dx,
                    dy,
                    collect_stats=False,
                    branch_mask=branch_mask,
                )

        x = self.act(self.proj1(x))
        x = self.proj2(x)
        if collect_stats:
            return x, all_stats
        return x

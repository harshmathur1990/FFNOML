import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# Helpers
# ============================================================
def _to_spacing_value(spacing, x):
    """
    Convert spacing to scalar float usable by torch.fft.fftfreq.
    Accepts python float/int or 0-d / 1-element tensor.
    """
    if isinstance(spacing, (float, int)):
        return float(spacing)

    if torch.is_tensor(spacing):
        if spacing.numel() != 1:
            raise ValueError(
                "spacing must be scalar (float/int or tensor with one element)."
            )
        return float(spacing.detach().item())

    raise TypeError(f"Unsupported spacing type: {type(spacing)}")


def _gn(channels, max_groups=8):
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return nn.InstanceNorm3d(channels, affine=True)


# def _gn(channels, max_groups=8):
    # return nn.Identity()


# ============================================================
# Basic blocks
# ============================================================
class PointwiseMLP(nn.Module):
    """
    Channel MLP via 1x1x1 convs.
    Input/Output: [B, C, D, H, W]
    """
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


class Residual1DBlock(nn.Module):
    """
    Residual block along depth only.
    Input/Output: [Ncol, C, D]
    """
    def __init__(self, ch, k=5):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, padding=p, bias=False)
        self.gn1 = _gn(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, padding=p, bias=False)
        self.gn2 = _gn(ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act(x + y)


class MultiScaleVertical(nn.Module):
    """
    Parallel depth-only kernels for different vertical coupling scales.
    Input/Output: [Ncol, C, D]
    """
    def __init__(self, ch):
        super().__init__()
        self.k3 = nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.k5 = nn.Conv1d(ch, ch, kernel_size=5, padding=2, bias=False)
        self.k9 = nn.Conv1d(ch, ch, kernel_size=9, padding=4, bias=False)
        self.gn = _gn(ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.k3(x)
        y += self.k5(x)
        y += self.k9(x)
        return self.act(self.gn(y))


class VerticalPhysicsStack(nn.Module):
    """
    Strong vertical mixer operating independently on each (y, x) column.

    Input/Output: [B, C, D, H, W]

    Strategy:
      1. project channels
      2. reshape each (y,x) column into a batch item
      3. run deep multi-scale residual 1D depth network
      4. reshape back
      5. project back to original width
    """
    def __init__(self, channels, hidden=256, dropout=0.0):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            _gn(hidden),
            nn.GELU(),
        )

        self.vertical_net = nn.Sequential(
            MultiScaleVertical(hidden),
            Residual1DBlock(hidden, k=5),
            Residual1DBlock(hidden, k=5),
            MultiScaleVertical(hidden),
            Residual1DBlock(hidden, k=5),
        )

        self.out_proj = nn.Conv1d(hidden, channels, kernel_size=1)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape

        # -> [B, H, W, C, D]
        x = x.permute(0, 3, 4, 1, 2)   # NO contiguous

        # flatten spatial only (safe, view-like)
        x = x.reshape(B, H * W, C, D)  # [B, HW, C, D]

        chunk = 128

        for i in range(0, H * W, chunk):
            # take chunk across ALL batch
            xi = x[:, i:i+chunk]              # [B, chunk, C, D]

            # merge batch + chunk
            xi = xi.reshape(-1, C, D)         # [B*chunk, C, D]

            yi = self.in_proj(xi)
            yi = self.vertical_net(yi)
            yi = self.out_proj(yi)

            # restore shape
            yi = yi.reshape(B, -1, C, D)      # [B, chunk, C, D]

            # write back
            x[:, i:i+chunk] = yi

        # restore spatial
        x = x.reshape(B, H, W, C, D)

        # back to original layout
        x = x.permute(0, 3, 4, 1, 2)   # NO contiguous

        return x


# ============================================================
# Spectral conv in horizontal plane
# ============================================================
class SpectralConv2dFull(nn.Module):
    """
    Full-spectrum metric-aware spectral convolution over (H, W),
    applied independently at each depth plane.

    Input/Output: [B, C, D, H, W]
    """
    def __init__(self, in_channels, out_channels, hidden=32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        scale = 1.0 / math.sqrt(in_channels)

        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, dtype=torch.float32)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, dtype=torch.float32)
        )

        self.freq_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, dx, dy):
        # x: [B, Cin, D, H, W]
        B, Cin, D, H, W = x.shape

        dx = _to_spacing_value(dx, x)
        dy = _to_spacing_value(dy, x)

        orig_dtype = x.dtype

        with torch.amp.autocast("cuda", enabled=False):
            x32 = x.float()

            # FFT only on horizontal plane
            x_ft = torch.fft.rfft2(x32, dim=(-2, -1))
            xr = x_ft.real
            xi = x_ft.imag

            ky = torch.fft.fftfreq(H, d=dy, device=x.device)
            kx = torch.fft.rfftfreq(W, d=dx, device=x.device)

            ky = 2.0 * math.pi * ky
            kx = 2.0 * math.pi * kx

            # rescale for stable MLP inputs
            k_scale = 1e5
            ky = ky * k_scale
            kx = kx * k_scale

            ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
            k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + 1e-12)

            k_feat = torch.stack(
                [
                    kx_grid,
                    ky_grid,
                    k_mag,
                    torch.sign(ky_grid),
                ],
                dim=-1,
            )  # [H, Wf, 4]

            gate = self.freq_mlp(k_feat)  # [H, Wf, 2]
            gate_real = gate[..., 0]
            gate_imag = gate[..., 1]

            wr = self.weight_real[:, :, None, None]   # [Cin, Cout, 1, 1]
            wi = self.weight_imag[:, :, None, None]

            gr = gate_real[None, None, :, :]          # [1, 1, H, Wf]
            gi = gate_imag[None, None, :, :]

            w_real = wr * gr - wi * gi
            w_imag = wr * gi + wi * gr

            out_ft_real = (
                torch.einsum("b i d y x, i o y x -> b o d y x", xr, w_real)
                - torch.einsum("b i d y x, i o y x -> b o d y x", xi, w_imag)
            )
            out_ft_imag = (
                torch.einsum("b i d y x, i o y x -> b o d y x", xr, w_imag)
                + torch.einsum("b i d y x, i o y x -> b o d y x", xi, w_real)
            )

            out_ft = torch.complex(out_ft_real, out_ft_imag)
            y = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))

        return y.to(orig_dtype)


# ============================================================
# FFNO block with strong vertical physics
# ============================================================
class FFNOBlock3d(nn.Module):
    """
    Hybrid FFNO block:
      - global spectral mixing in (H, W)
      - pointwise local channel mixing
      - strong vertical physics stack along D
      - pointwise MLP refinement

    Input/Output: [B, C, D, H, W]
    """
    def __init__(
        self,
        width,
        dropout=0.0,
        mlp_expansion=2,
        vertical_hidden=256,
        use_gating=True,
    ):
        super().__init__()

        self.spec_hw = SpectralConv2dFull(width, width)
        self.pw_hw = nn.Conv3d(width, width, kernel_size=1)

        self.vertical = VerticalPhysicsStack(
            channels=width,
            hidden=vertical_hidden,
            dropout=dropout,
        )

        self.mlp = PointwiseMLP(
            width,
            expansion=mlp_expansion,
            dropout=dropout,
        )

        self.norm1 = _gn(width)
        self.norm2 = _gn(width)
        self.norm3 = _gn(width)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.use_gating = use_gating

        if use_gating:
            self.alpha_spec = nn.Parameter(torch.ones(1) * 3)
            self.alpha_pw   = nn.Parameter(torch.ones(1))
            self.alpha_vert = nn.Parameter(torch.ones(1))
            self.alpha_mlp  = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("alpha_spec", torch.ones(1) * 3)
            self.register_buffer("alpha_pw",   torch.ones(1))
            self.register_buffer("alpha_vert", torch.ones(1))
            self.register_buffer("alpha_mlp",  torch.ones(1))

    def forward(self, x, dx, dy, collect_stats=False):
        stats = {} if collect_stats else None

        # -------------------------------
        # Horizontal operator
        # -------------------------------
        spec = self.spec_hw(x, dx=dx, dy=dy)
        pw   = self.pw_hw(x)

        y = self.alpha_spec * spec + self.alpha_pw * pw

        if collect_stats:
            with torch.no_grad():
                stats["spec_norm"] = spec.abs().mean().item()
                stats["pw_norm"]   = pw.abs().mean().item()
                stats["alpha_spec"] = float(self.alpha_spec.item())
                stats["alpha_pw"]   = float(self.alpha_pw.item())

        y = self.norm1(y)
        y = self.act(y)
        y = self.drop(y)

        x_before = x
        x = x + y

        if collect_stats:
            with torch.no_grad():
                stats["horizontal_update"] = (x - x_before).abs().mean().item()

        # -------------------------------
        # Vertical physics
        # -------------------------------
        vert = self.vertical(x)
        z = self.alpha_vert * vert

        if collect_stats:
            with torch.no_grad():
                stats["vertical_norm"] = vert.abs().mean().item()
                stats["alpha_vert"]    = float(self.alpha_vert.item())

        z = self.norm2(z)
        z = self.act(z)
        z = self.drop(z)

        x_before = x
        x = x + z

        if collect_stats:
            with torch.no_grad():
                stats["vertical_update"] = (x - x_before).abs().mean().item()

        # -------------------------------
        # MLP refinement
        # -------------------------------
        mlp_out = self.mlp(x)
        m = self.alpha_mlp * mlp_out

        if collect_stats:
            with torch.no_grad():
                stats["mlp_norm"] = mlp_out.abs().mean().item()
                stats["alpha_mlp"] = float(self.alpha_mlp.item())

        m = self.norm3(m)
        m = self.act(m)
        m = self.drop(m)

        x_before = x
        x = x + m

        if collect_stats:
            with torch.no_grad():
                stats["mlp_update"] = (x - x_before).abs().mean().item()

        if collect_stats:
            return x, stats
        else:
            return x


# ============================================================
# Full model
# ============================================================
class FFNO3D(nn.Module):
    """
    Updated hybrid FFNO for RT-like 3D volumes.

    Input:  [B, Cin, D, H, W]
    Output: [B, Cout, D, H, W]

    Horizontal nonlocal coupling:
        spectral operator over (H, W)

    Vertical physics:
        deep multi-scale residual column network over D
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        width=64,
        n_layers=6,
        dropout=0.1,
        mlp_expansion=2,
        vertical_hidden=256,
        padding=0,
        checkpoint_blocks=True,
        use_gating=True,
    ):
        super().__init__()

        self.padding = padding
        self.checkpoint_blocks = checkpoint_blocks

        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                FFNOBlock3d(
                    width=width,
                    dropout=dropout,
                    mlp_expansion=mlp_expansion,
                    vertical_hidden=vertical_hidden,
                    use_gating=use_gating,
                )
                for _ in range(n_layers)
            ]
        )

        self.proj1 = nn.Conv3d(width, width * 2, kernel_size=1)
        self.proj2 = nn.Conv3d(width * 2, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def _run_block(self, blk, x, dx, dy, collect_stats=False):
        if self.checkpoint_blocks and self.training and not collect_stats:
            return checkpoint(
                lambda t: blk(t, dx=dx, dy=dy, collect_stats=False),
                x,
                use_reentrant=False,
            )
        else:
            return blk(x, dx=dx, dy=dy, collect_stats=collect_stats)

    def forward(self, x, dx, dy, collect_stats=False):
        all_stats = [] if collect_stats else None

        if self.padding > 0:
            p = self.padding
            x = F.pad(x, (p, p, p, p, p, p), mode="replicate")

        x = self.lift(x)

        for i, blk in enumerate(self.blocks):
            if collect_stats:
                x, s = self._run_block(blk, x, dx, dy, collect_stats=True)
                s["layer"] = i
                all_stats.append(s)
            else:
                x = self._run_block(blk, x, dx, dy, collect_stats=False)

        x = self.act(self.proj1(x))
        y = self.proj2(x)

        if self.padding > 0:
            p = self.padding
            y = y[:, :, p:-p, p:-p, p:-p]

        if collect_stats:
            return y, all_stats
        else:
            return y

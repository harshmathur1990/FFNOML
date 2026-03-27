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


def compute_k_cutoff(dx, dy=None, k_scale=1e5, radial=True):
    """
    Compute training-band cutoff in the SAME scaled k-space
    used by your layer.

    Parameters
    ----------
    dx : float
        Training grid spacing in x.
    dy : float or None
        Training grid spacing in y. If None, dy=dx.
    k_scale : float
        The same scaling factor used in the model.
    radial : bool
        If True, return radial cutoff based on k_mag.
        If False, return per-axis cutoffs.

    Returns
    -------
    radial=True:
        k_cutoff : float

    radial=False:
        kx_cutoff, ky_cutoff : float, float
    """
    if dy is None:
        dy = dx

    kx_cutoff = math.pi / dx * k_scale
    ky_cutoff = math.pi / dy * k_scale

    if radial:
        return math.sqrt(kx_cutoff**2 + ky_cutoff**2)
    else:
        return kx_cutoff, ky_cutoff


def _gn(channels, max_groups=8):
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


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

    SAFE VERSION:
      - no in-place writes
      - autograd friendly
      - chunked processing preserved
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

        # [B, H, W, C, D]
        x = x.permute(0, 3, 4, 1, 2)

        # [B, HW, C, D]
        x = x.reshape(B, H * W, C, D)

        chunk = 128
        outputs = []

        for i in range(0, H * W, chunk):
            xi = x[:, i:i+chunk]              # [B, chunk, C, D]
            xi = xi.reshape(-1, C, D)         # [B*chunk, C, D]

            yi = self.in_proj(xi)
            yi = self.vertical_net(yi)
            yi = self.out_proj(yi)

            yi = yi.reshape(B, -1, C, D)      # [B, chunk, C, D]
            outputs.append(yi)

        # concatenate instead of in-place write
        x = torch.cat(outputs, dim=1)

        # [B, H, W, C, D]
        x = x.reshape(B, H, W, C, D)

        # back to [B, C, D, H, W]
        x = x.permute(0, 3, 4, 1, 2)

        return x

# ============================================================
# Spectral conv in horizontal plane
# ============================================================
# class SpectralConv2dFull(nn.Module):
#     """
#     Full-spectrum metric-aware spectral convolution over (H, W),
#     applied independently at each depth plane.

#     Input/Output: [B, C, D, H, W]
#     """
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         hidden=32,
#         kx_cutoff=None,
#         ky_cutoff=None
#     ):
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         scale = 1.0 / math.sqrt(in_channels)

#         self.weight_real = nn.Parameter(
#             scale * torch.randn(in_channels, out_channels, dtype=torch.float32)
#         )
#         self.weight_imag = nn.Parameter(
#             scale * torch.randn(in_channels, out_channels, dtype=torch.float32)
#         )

#         self.freq_mlp = nn.Sequential(
#             nn.Linear(4, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, 2),
#         )

#         self.kx_cutoff = kx_cutoff
#         self.ky_cutoff = ky_cutoff


#     def forward(self, x, dx, dy):
#         # x: [B, Cin, D, H, W]
#         B, Cin, D, H, W = x.shape

#         dx = _to_spacing_value(dx, x)
#         dy = _to_spacing_value(dy, x)

#         orig_dtype = x.dtype

#         with torch.amp.autocast("cuda", enabled=False):
#             x32 = x.float()

#             # FFT only on horizontal plane
#             x_ft = torch.fft.rfft2(x32, dim=(-2, -1))
#             xr = x_ft.real
#             xi = x_ft.imag

#             ky = torch.fft.fftfreq(H, d=dy, device=x.device)
#             kx = torch.fft.rfftfreq(W, d=dx, device=x.device)

#             ky = 2.0 * math.pi * ky
#             kx = 2.0 * math.pi * kx

#             # rescale for stable MLP inputs
#             k_scale = 1e5
#             ky = ky * k_scale
#             kx = kx * k_scale

#             ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
#             k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + 1e-12)

#             k_feat = torch.stack(
#                 [
#                     kx_grid,
#                     ky_grid,
#                     k_mag,
#                     torch.sign(ky_grid),
#                 ],
#                 dim=-1,
#             )  # [H, Wf, 4]

#             gate = self.freq_mlp(k_feat)  # [H, Wf, 2]
#             gate_real = gate[..., 0]
#             gate_imag = gate[..., 1]

#             wr = self.weight_real[:, :, None, None]   # [Cin, Cout, 1, 1]
#             wi = self.weight_imag[:, :, None, None]

#             gr = gate_real[None, None, :, :]          # [1, 1, H, Wf]
#             gi = gate_imag[None, None, :, :]

#             w_real = wr * gr - wi * gi
#             w_imag = wr * gi + wi * gr

#             out_ft_real = (
#                 torch.einsum("b i d y x, i o y x -> b o d y x", xr, w_real)
#                 - torch.einsum("b i d y x, i o y x -> b o d y x", xi, w_imag)
#             )
#             out_ft_imag = (
#                 torch.einsum("b i d y x, i o y x -> b o d y x", xr, w_imag)
#                 + torch.einsum("b i d y x, i o y x -> b o d y x", xi, w_real)
#             )

#             if self.kx_cutoff is not None:
#                 mask = (kx_grid.abs() <= self.kx_cutoff) & (ky_grid.abs() <= self.ky_cutoff)[None, None, None, :, :]
#                 out_ft_real = out_ft_real * mask
#                 out_ft_imag = out_ft_imag * mask

#             out_ft = torch.complex(out_ft_real, out_ft_imag)
#             y = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))

#         return y.to(orig_dtype)


class SpectralConv2dFull(nn.Module):
    """
    Fully input-conditioned spectral operator.

    ✔ uses full input (no mean over D/H/W)
    ✔ resolution invariant
    ✔ nonlinear
    ✔ per-(k,d) independent → stable & efficient
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden=64,
        rank=16,
        kx_cutoff=None,
        ky_cutoff=None,
        k_scale=1e5,
        use_bias=True,
        apply_mask=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.kx_cutoff = kx_cutoff
        self.ky_cutoff = ky_cutoff
        self.k_scale = k_scale
        self.apply_mask = apply_mask

        scale = 1.0 / math.sqrt(in_channels)

        # low-rank basis
        self.basis_real = nn.Parameter(
            scale * torch.randn(rank, in_channels, out_channels)
        )
        self.basis_imag = nn.Parameter(
            scale * torch.randn(rank, in_channels, out_channels)
        )

        # frequency embedding
        self.freq_net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # FULL input embedding (per k, per depth)
        self.input_net = nn.Sequential(
            nn.Linear(2 * in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # joint → coefficients
        self.joint_net = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * rank),
        )

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    # ------------------------------------------------------------
    def _build_k_grid(self, H, W, dx, dy, device):
        ky = torch.fft.fftfreq(H, d=dy, device=device)
        kx = torch.fft.rfftfreq(W, d=dx, device=device)

        ky = 2.0 * math.pi * ky
        kx = 2.0 * math.pi * kx

        ky_scaled = ky * self.k_scale
        kx_scaled = kx * self.k_scale

        ky_grid, kx_grid = torch.meshgrid(
            ky_scaled, kx_scaled, indexing="ij"
        )

        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + 1e-12)

        k_feat = torch.stack(
            [kx_grid, ky_grid, k_mag, torch.sign(ky_grid)], dim=-1
        )

        return k_feat, kx_grid, ky_grid

    def _build_mask(self, kx_grid, ky_grid):
        if self.kx_cutoff is None and self.ky_cutoff is None:
            return None

        return (
            (kx_grid.abs() <= self.kx_cutoff) &
            (ky_grid.abs() <= self.ky_cutoff)
        )

    # ------------------------------------------------------------
    def forward(self, x, dx, dy):
        B, Cin, D, H, W = x.shape
        device = x.device

        dx = float(dx) if not torch.is_tensor(dx) else float(dx.item())
        dy = float(dy) if not torch.is_tensor(dy) else float(dy.item())

        # --------------------------------------------------------
        # FFT
        # --------------------------------------------------------
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        xr, xi = x_ft.real, x_ft.imag  # [B, Cin, D, H, Wf]

        # --------------------------------------------------------
        # FULL input embedding (NO reduction)
        # --------------------------------------------------------
        z = torch.cat([xr, xi], dim=1)              # [B, 2Cin, D, H, Wf]
        z = z.permute(0, 2, 3, 4, 1)                # [B, D, H, Wf, 2Cin]

        input_emb = self.input_net(z)               # [B, D, H, Wf, hidden]

        # --------------------------------------------------------
        # Frequency embedding
        # --------------------------------------------------------
        k_feat, kx_grid, ky_grid = self._build_k_grid(
            H, W, dx, dy, device
        )

        freq_emb = self.freq_net(k_feat)            # [H, Wf, hidden]

        mask = self._build_mask(kx_grid, ky_grid)

        # --------------------------------------------------------
        # Combine (broadcast over B and D)
        # --------------------------------------------------------
        joint = freq_emb[None, None, :, :, :] + input_emb
        # [B, D, H, Wf, hidden]

        coeffs = self.joint_net(joint)
        coeffs = coeffs.view(B, D, H, x_ft.shape[-1], self.rank, 2)

        coef_real = coeffs[..., 0]
        coef_imag = coeffs[..., 1]

        # --------------------------------------------------------
        # Mask
        # --------------------------------------------------------
        if self.apply_mask and (mask is not None):
            mask_f = mask[None, None, :, :, None].to(coef_real.dtype)
            coef_real *= mask_f
            coef_imag *= mask_f

        # --------------------------------------------------------
        # Low-rank weights
        # --------------------------------------------------------
        br = self.basis_real[None, None, ...]  # [1,1,R,Cin,Cout]
        bi = self.basis_imag[None, None, ...]

        w_real = (
            torch.einsum("bdyxr,brio->bdioyx", coef_real, br)
            - torch.einsum("bdyxr,brio->bdioyx", coef_imag, bi)
        )
        w_imag = (
            torch.einsum("bdyxr,brio->bdioyx", coef_real, bi)
            + torch.einsum("bdyxr,brio->bdioyx", coef_imag, br)
        )

        # --------------------------------------------------------
        # Apply operator
        # --------------------------------------------------------
        out_ft_real = (
            torch.einsum("bidyx,bdioyx->bodyx", xr, w_real)
            - torch.einsum("bidyx,bdioyx->bodyx", xi, w_imag)
        )
        out_ft_imag = (
            torch.einsum("bidyx,bdioyx->bodyx", xr, w_imag)
            + torch.einsum("bidyx,bdioyx->bodyx", xi, w_real)
        )

        out_ft = torch.complex(out_ft_real, out_ft_imag)

        # --------------------------------------------------------
        # Inverse FFT
        # --------------------------------------------------------
        y = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))

        if self.bias is not None:
            y = y + self.bias[None, :, None, None, None]

        return y


# ============================================================
# FFNO block with strong vertical physics
# ============================================================
class FFNOBlock3d(nn.Module):
    """
    STABLE hybrid FFNO block:
      - bounded gates
      - normalized branches
      - residual scaling
      - no runaway amplification
    """

    def __init__(
        self,
        width,
        dropout=0.0,
        mlp_expansion=2,
        vertical_hidden=256,
        use_gating=True,
        spectral_hidden=64,
        spectral_rank=16,
        kx_cutoff=None,
        ky_cutoff=None,
        k_scale=1e5,
        spectral_use_bias=True,
        spectral_apply_mask=True
    ):
        super().__init__()

        # -----------------------------------
        # Operators
        # -----------------------------------
        self.spec_hw = SpectralConv2dFull(
            in_channels=width,
            out_channels=width,
            hidden=spectral_hidden,
            rank=spectral_rank,
            kx_cutoff=kx_cutoff,
            ky_cutoff=ky_cutoff,
            k_scale=k_scale,
            use_bias=spectral_use_bias,
            apply_mask=spectral_apply_mask,
        )

        self.pw_hw   = nn.Conv3d(width, width, kernel_size=1)

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

        # -----------------------------------
        # Branch normalization (CRITICAL)
        # -----------------------------------
        self.norm_spec = _gn(width)
        self.norm_pw   = _gn(width)
        self.norm_vert = _gn(width)
        self.norm_mlp  = _gn(width)

        # -----------------------------------
        # Residual post-norm
        # -----------------------------------
        self.post_norm1 = _gn(width)
        self.post_norm2 = _gn(width)
        self.post_norm3 = _gn(width)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # -----------------------------------
        # Bounded gates (no explosion)
        # -----------------------------------
        self.use_gating = use_gating

        if use_gating:
            self.logit_spec = nn.Parameter(torch.ones(1) * 0.0)
            self.logit_pw   = nn.Parameter(torch.ones(1) * 0.0)
            self.logit_vert = nn.Parameter(torch.ones(1) * -0.5)
            self.logit_mlp  = nn.Parameter(torch.ones(1) * -0.5)
        else:
            self.register_buffer("logit_spec", torch.ones(1) * 0.0)
            self.register_buffer("logit_pw",   torch.ones(1) * 0.0)
            self.register_buffer("logit_vert", torch.ones(1) * 0.0)
            self.register_buffer("logit_mlp",  torch.ones(1) * 0.0)

        # -----------------------------------
        # Residual scaling (VERY IMPORTANT)
        # -----------------------------------
        self.res_h = nn.Parameter(torch.ones(1) * 0.5)
        self.res_v = nn.Parameter(torch.ones(1) * 0.5)
        self.res_m = nn.Parameter(torch.ones(1) * 0.5)

    def _get_alphas(self):
        # bounded in (0, 2)
        alpha_spec = 2.0 * torch.sigmoid(self.logit_spec)
        alpha_pw   = 2.0 * torch.sigmoid(self.logit_pw)
        alpha_vert = 2.0 * torch.sigmoid(self.logit_vert)
        alpha_mlp  = 2.0 * torch.sigmoid(self.logit_mlp)

        return alpha_spec, alpha_pw, alpha_vert, alpha_mlp

    def forward(self, x, dx, dy, collect_stats=False):
        stats = {} if collect_stats else None

        alpha_spec, alpha_pw, alpha_vert, alpha_mlp = self._get_alphas()

        # ============================================================
        # 1. Horizontal mixing
        # ============================================================
        spec = self.norm_spec(self.spec_hw(x, dx=dx, dy=dy))
        pw   = self.norm_pw(self.pw_hw(x))

        y = alpha_spec * spec + alpha_pw * pw
        y = self.act(y)
        y = self.drop(y)

        x_before = x
        x = x + self.res_h * y

        if collect_stats:
            with torch.no_grad():
                stats["spec_norm"] = spec.abs().mean().item()
                stats["pw_norm"]   = pw.abs().mean().item()
                stats["alpha_spec"] = float(alpha_spec.item())
                stats["alpha_pw"]   = float(alpha_pw.item())
                stats["horizontal_update"] = (x - x_before).abs().mean().item()

        x = self.post_norm1(x)

        # ============================================================
        # 2. Vertical physics
        # ============================================================
        vert = self.norm_vert(self.vertical(x))
        z = alpha_vert * vert

        z = self.act(z)
        z = self.drop(z)

        x_before = x
        x = x + self.res_v * z

        if collect_stats:
            with torch.no_grad():
                stats["vertical_norm"] = vert.abs().mean().item()
                stats["alpha_vert"]    = float(alpha_vert.item())
                stats["vertical_update"] = (x - x_before).abs().mean().item()

        x = self.post_norm2(x)

        # ============================================================
        # 3. MLP refinement
        # ============================================================
        mlp_out = self.norm_mlp(self.mlp(x))
        m = alpha_mlp * mlp_out

        m = self.act(m)
        m = self.drop(m)

        x_before = x
        x = x + self.res_m * m

        if collect_stats:
            with torch.no_grad():
                stats["mlp_norm"] = mlp_out.abs().mean().item()
                stats["alpha_mlp"] = float(alpha_mlp.item())
                stats["mlp_update"] = (x - x_before).abs().mean().item()

        x = self.post_norm3(x)

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
        spectral_hidden=64,
        spectral_rank=16,
        dx_cutoff=None,
        dy_cutoff=None,
        k_scale=1e5,
        spectral_use_bias=True,
        spectral_apply_mask=True
    ):
        super().__init__()

        self.padding = padding
        self.checkpoint_blocks = checkpoint_blocks

        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)

        kx_cutoff, ky_cutoff = compute_k_cutoff(dx=dx_cutoff, dy=dy_cutoff, radial=False)

        self.kx_cutoff = kx_cutoff

        self.ky_cutoff = ky_cutoff

        self.blocks = nn.ModuleList(
            [
                FFNOBlock3d(
                    width,
                    dropout=dropout,
                    mlp_expansion=mlp_expansion,
                    vertical_hidden=vertical_hidden,
                    use_gating=use_gating,
                    spectral_hidden=spectral_hidden,
                    spectral_rank=spectral_rank,
                    kx_cutoff=self.kx_cutoff,
                    ky_cutoff=self.ky_cutoff,
                    k_scale=k_scale,
                    spectral_use_bias=spectral_use_bias,
                    spectral_apply_mask=spectral_apply_mask
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

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
# 1D Z Neural Operator Branch
# ============================================================
def _prepare_z_scale(z_scale, x):
    B, _, D, H, W = x.shape
    if z_scale.ndim == 4:
        z_scale = z_scale.unsqueeze(1)
    if z_scale.shape != (B, 1, D, H, W):
        raise ValueError(
            f"z_scale must be [B, D, H, W] or [B, 1, D, H, W], got {tuple(z_scale.shape)}"
        )
    return z_scale


class CoordMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LatentZKernelOperator1d(nn.Module):
    """
    Coordinate-aware 1D neural operator over nonuniform z samples.
    Applied independently to each (x, y) column.
    """

    def __init__(
        self,
        channels,
        n_latents=16,
        kernel_hidden=64,
        point_hidden=128,
        dropout=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.n_latents = n_latents

        self.norm_in = nn.LayerNorm(channels)
        self.norm_latent = nn.LayerNorm(channels)

        self.latent_coords = nn.Parameter(torch.linspace(-1.0, 1.0, n_latents).view(1, n_latents, 1))

        self.input_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.coord_embed = nn.Sequential(
            nn.Conv1d(3, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
        )

        self.latent_init = CoordMLP(1, kernel_hidden, channels, dropout=dropout)

        self.q_point = nn.Linear(channels, channels)
        self.k_point = nn.Linear(channels, channels)
        self.v_point = nn.Linear(channels, channels)
        self.q_latent = nn.Linear(channels, channels)
        self.k_latent = nn.Linear(channels, channels)
        self.v_latent = nn.Linear(channels, channels)

        self.enc_kernel = CoordMLP(5 + 2 * channels, kernel_hidden, channels, dropout=dropout)
        self.lat_kernel = CoordMLP(5 + 2 * channels, kernel_hidden, channels, dropout=dropout)
        self.dec_kernel = CoordMLP(5 + 2 * channels, kernel_hidden, channels, dropout=dropout)

        self.pointwise = nn.Sequential(
            nn.Linear(channels, point_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(point_hidden, channels),
        )

    def _quadrature_weights(self, z):
        if z.shape[-1] == 1:
            return torch.ones_like(z)

        dz = torch.empty_like(z)
        dz[:, 1:-1] = 0.5 * (z[:, 2:] - z[:, :-2])
        dz[:, 0] = z[:, 1] - z[:, 0]
        dz[:, -1] = z[:, -1] - z[:, -2]
        return dz.abs().clamp_min(1e-6)

    def _coord_features(self, z):
        span = (z.max(dim=-1, keepdim=True).values - z.min(dim=-1, keepdim=True).values).clamp_min(1e-6)
        weights = self._quadrature_weights(z)
        local_dz = weights / weights.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        coord = torch.stack(
            [
                z,
                local_dz,
                span.expand_as(z),
            ],
            dim=1,
        )
        return coord, z.unsqueeze(-1), weights

    @staticmethod
    def _pair_features(a_coords, b_coords, a_feat, b_feat, eps=1e-6):
        rel = a_coords.unsqueeze(2) - b_coords.unsqueeze(1)
        rel_norm = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + eps)

        a_c = a_coords.unsqueeze(2).expand(-1, -1, b_coords.shape[1], -1)
        b_c = b_coords.unsqueeze(1).expand(-1, a_coords.shape[1], -1, -1)
        a_f = a_feat.unsqueeze(2).expand(-1, -1, b_feat.shape[1], -1)
        b_f = b_feat.unsqueeze(1).expand(-1, a_feat.shape[1], -1, -1)
        return torch.cat([a_c, b_c, rel, rel_norm, a_f, b_f], dim=-1)

    def forward(self, x, z):
        x = x * self.input_gate(x)
        coord_embed, point_coords, weights = self._coord_features(z)
        x = x + self.coord_embed(coord_embed)

        u = x.transpose(1, 2)
        u0 = u
        u = self.norm_in(u)

        B, D, C = u.shape
        latent_coords = self.latent_coords.to(device=u.device, dtype=u.dtype).expand(B, -1, -1)
        h = self.latent_init(latent_coords)

        p_k = self.k_point(u)
        p_v = self.v_point(u)
        z_q = self.q_latent(h)

        enc_feat = self._pair_features(latent_coords, point_coords, z_q, p_k)
        enc_score = self.enc_kernel(enc_feat)
        enc_alpha = torch.softmax(enc_score.mean(dim=-1), dim=-2).unsqueeze(-1)
        src = (p_v * weights.unsqueeze(-1)).unsqueeze(1)
        h = h + (enc_alpha * src).sum(dim=2)

        h_in = h
        h = self.norm_latent(h)
        z_q2 = self.q_latent(h)
        z_k2 = self.k_latent(h)
        z_v2 = self.v_latent(h)
        lat_feat = self._pair_features(latent_coords, latent_coords, z_q2, z_k2)
        lat_score = self.lat_kernel(lat_feat)
        lat_alpha = torch.softmax(lat_score.mean(dim=-1), dim=-1).unsqueeze(-1)
        h = h_in + (lat_alpha * z_v2.unsqueeze(1)).sum(dim=2)

        p_q = self.q_point(u)
        z_k3 = self.k_latent(h)
        z_v3 = self.v_latent(h)
        dec_feat = self._pair_features(point_coords, latent_coords, p_q, z_k3)
        dec_score = self.dec_kernel(dec_feat)
        dec_alpha = torch.softmax(dec_score.mean(dim=-1), dim=-1).unsqueeze(-1)
        upd = (dec_alpha * z_v3.unsqueeze(1)).sum(dim=2)

        out = u0 + self.pointwise(upd)
        return out.transpose(1, 2)


class ZNeuralOperator1d(nn.Module):
    def __init__(self, channels, hidden=128, dropout=0.0, chunk=4):
        super().__init__()
        self.chunk = chunk

        self.in_proj = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            _gn(hidden),
            nn.GELU(),
        )

        self.operator = LatentZKernelOperator1d(
            hidden,
            n_latents=16,
            kernel_hidden=max(64, hidden),
            point_hidden=2 * hidden,
            dropout=dropout,
        )

        self.local_mix = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 1, bias=False),
        )

        self.depth_gate = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.Sigmoid(),
        )

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_proj = nn.Conv1d(hidden, channels, 1, bias=False)
        self.out_gn = _gn(channels)

    def forward(self, x, z_scale):
        B, C, D, H, W = x.shape
        z_scale = _prepare_z_scale(z_scale, x)

        x_cols = x.permute(0, 3, 4, 1, 2).reshape(B, H * W, C, D)
        z_cols = z_scale.permute(0, 3, 4, 1, 2).reshape(B, H * W, D)

        chunks = []
        for i in range(0, H * W, self.chunk):
            j = min(i + self.chunk, H * W)
            xi = x_cols[:, i:j].reshape(-1, C, D)
            zi = z_cols[:, i:j].reshape(-1, D)

            yi = self.in_proj(xi)
            yi = self.operator(yi, zi) + self.local_mix(yi)
            yi = yi * self.depth_gate(yi)
            yi = self.drop(yi)
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
class FFNOBlock3dZ1D(nn.Module):
    """
    Spectral and vertical branches are both strengthened and fused by:
      - same input to both branches
      - normalization on both outputs
      - adaptive branch gating before learned fusion
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

        self.vertical = ZNeuralOperator1d(
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
        self.branch_gate = nn.Sequential(
            nn.Conv3d(3 * width, width, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(width, 2 * width, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * width, 2 * width, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(
                2 * width,
                2 * width,
                kernel_size=3,
                padding=1,
                groups=2 * width,
                bias=False,
            ),
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

    def _fuse_branches(self, x, spec, vert):
        gates = self.branch_gate(torch.cat([x, spec, vert], dim=1))
        spec_gate, vert_gate = gates.chunk(2, dim=1)

        spec_mix = spec * spec_gate
        vert_mix = vert * vert_gate

        fused = self.fuse(torch.cat([spec_mix, vert_mix], dim=1))
        fused = fused + 0.5 * (spec_mix + vert_mix)
        fused = self.act(self.norm_fuse(fused))
        return fused, spec_mix, vert_mix, spec_gate, vert_gate

    def forward(self, x, z_scale, dx, dy, collect_stats=False, branch_mask=None):
        if not collect_stats and branch_mask is None:
            residual = x

            spec = self.spec_drop(self.act(self.norm_spec(self.spec(x, dx, dy))))
            vert = self.vert_drop(self.act(self.norm_vert(self.vertical(x, z_scale))))
            fused, _, _, _, _ = self._fuse_branches(x, spec, vert)

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
        vert = self.vert_drop(self.act(self.norm_vert(self.vertical(x, z_scale))))

        spec_mask = float(branch_mask.get("spec", 1.0))
        vert_mask = float(branch_mask.get("vertical", 1.0))
        pw_mask = float(branch_mask.get("pw", 1.0))
        mlp_mask = float(branch_mask.get("mlp", 1.0))

        spec_eff = spec_mask * spec
        vert_eff = vert_mask * vert
        fused, spec_mix, vert_mix, spec_gate, vert_gate = self._fuse_branches(
            x, spec_eff, vert_eff
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
class FFNO3DZ1D(nn.Module):
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
            FFNOBlock3dZ1D(
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

    def _run_block(
        self, blk, x, z_scale, dx, dy, collect_stats=False, branch_mask=None
    ):
        if self.checkpoint_blocks and self.training and not collect_stats:
            return checkpoint(
                lambda t: blk(
                    t, z_scale, dx, dy, collect_stats=False, branch_mask=None
                ),
                x,
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

    def _run_lift(self, x, collect_stats=False):
        if self.checkpoint_blocks and self.training and not collect_stats:
            return checkpoint(self.lift, x, use_reentrant=False)
        return self.lift(x)

    def _run_head(self, x, collect_stats=False):
        if self.checkpoint_blocks and self.training and not collect_stats:
            return checkpoint(
                lambda t: self.proj2(self.act(self.proj1(t))),
                x,
                use_reentrant=False,
            )
        x = self.act(self.proj1(x))
        return self.proj2(x)

    def forward(self, x, z_scale, dx, dy, collect_stats=False, branch_mask=None):
        x = self._run_lift(x, collect_stats=collect_stats)
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

        x = self._run_head(x, collect_stats=collect_stats)
        if collect_stats:
            return x, all_stats
        return x

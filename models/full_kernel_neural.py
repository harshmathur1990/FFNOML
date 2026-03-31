import math
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# helpers
# ============================================================

def _to_batch_scalar(
    x: Union[float, int, torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(float(x), device=device, dtype=dtype)
    else:
        x = x.to(device=device, dtype=dtype)

    if x.ndim == 0:
        x = x.expand(batch_size)
    elif x.ndim == 1 and x.shape[0] == batch_size:
        pass
    else:
        raise ValueError(f"{name} must be scalar or [B], got {tuple(x.shape)}")
    return x


def _to_cmass_grid(
    cmass_grid: torch.Tensor,
    batch_size: int,
    depth: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(cmass_grid):
        raise TypeError("cmass_grid must be a torch.Tensor")

    cmass_grid = cmass_grid.to(device=device, dtype=dtype)

    if cmass_grid.ndim == 1:
        if cmass_grid.shape[0] != depth:
            raise ValueError(f"cmass_grid length {cmass_grid.shape[0]} != depth {depth}")
        cmass_grid = cmass_grid.unsqueeze(0).expand(batch_size, depth)
    elif cmass_grid.ndim == 2:
        if cmass_grid.shape != (batch_size, depth):
            raise ValueError(
                f"cmass_grid must be [B, D] = [{batch_size}, {depth}], got {tuple(cmass_grid.shape)}"
            )
    else:
        raise ValueError("cmass_grid must have shape [D] or [B, D]")

    return cmass_grid


def build_coords_and_weights(
    x: torch.Tensor,                    # [B, Cin, D, Ny, Nx]
    dx: Union[float, int, torch.Tensor],
    dy: Union[float, int, torch.Tensor],
    cmass_grid: torch.Tensor,          # [D] or [B, D]
    eps: float = 1e-12,
):
    """
    Returns:
        coords:  [B, N, 3]  with (x_phys, y_phys, log10(cmass))
        weights: [B, N]     quadrature weights ~ dx * dy * d(log10 cmass)
    """
    B, _, D, Ny, Nx = x.shape
    device = x.device
    dtype = x.dtype

    dx = _to_batch_scalar(dx, B, device, dtype, "dx")
    dy = _to_batch_scalar(dy, B, device, dtype, "dy")
    cmass_grid = _to_cmass_grid(cmass_grid, B, D, device, dtype)

    # use log10(cmass) as the coordinate; this is usually numerically easier
    logm = torch.log10(cmass_grid.clamp_min(eps))  # [B, D]

    x_idx = torch.arange(Nx, device=device, dtype=dtype)
    y_idx = torch.arange(Ny, device=device, dtype=dtype)

    x_phys = dx[:, None] * x_idx[None, :]  # [B, Nx]
    y_phys = dy[:, None] * y_idx[None, :]  # [B, Ny]

    X = x_phys[:, None, None, :].expand(B, D, Ny, Nx)
    Y = y_phys[:, None, :, None].expand(B, D, Ny, Nx)
    M = logm[:, :, None, None].expand(B, D, Ny, Nx)

    coords = torch.stack([X, Y, M], dim=-1).reshape(B, D * Ny * Nx, 3)

    # quadrature on log10(cmass)
    dlogm = torch.empty_like(logm)
    if D == 1:
        dlogm[:] = 1.0
    else:
        dlogm[:, 1:-1] = 0.5 * (logm[:, 2:] - logm[:, :-2])
        dlogm[:, 0] = logm[:, 1] - logm[:, 0]
        dlogm[:, -1] = logm[:, -1] - logm[:, -2]
    dlogm = dlogm.abs().clamp_min(eps)

    weights = (
        dx[:, None, None, None]
        * dy[:, None, None, None]
        * dlogm[:, :, None, None]
    ).expand(B, D, Ny, Nx).reshape(B, D * Ny * Nx)

    return coords, weights


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if activation == "gelu":
            act = nn.GELU
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "relu":
            act = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# latent global kernel operator
# ============================================================

class LatentKernelBlock3D(nn.Module):
    """
    Global neural operator block:
      point cloud / voxel field -> latent tokens -> global latent interaction -> back to points

    Complexity:
      O(B * N * M * C) + O(B * M^2 * C)
    instead of O(B * N^2 * C)

    No xy/z splitting.
    """

    def __init__(
        self,
        channels: int,
        n_latents: int = 128,
        coord_dim: int = 3,
        kernel_hidden: int = 128,
        point_hidden: int = 128,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_softmax: bool = True,
    ):
        super().__init__()

        self.channels = channels
        self.n_latents = n_latents
        self.coord_dim = coord_dim
        self.use_softmax = use_softmax

        self.norm_in = nn.LayerNorm(channels)
        self.norm_latent = nn.LayerNorm(channels)

        # learned latent coordinates in normalized operator space
        # actual physical meaning comes from matching against real coords
        self.latent_coords = nn.Parameter(torch.randn(n_latents, coord_dim) * 0.1)

        # point encoders
        self.q_point = nn.Linear(channels, channels)
        self.k_point = nn.Linear(channels, channels)
        self.v_point = nn.Linear(channels, channels)

        # latent encoders
        self.q_latent = nn.Linear(channels, channels)
        self.k_latent = nn.Linear(channels, channels)
        self.v_latent = nn.Linear(channels, channels)

        # initialize latent states from coords alone
        self.latent_init = MLP(
            in_dim=coord_dim,
            hidden_dim=kernel_hidden,
            out_dim=channels,
            n_layers=2,
            activation=activation,
            dropout=dropout,
        )

        # kernels
        # point -> latent
        self.enc_kernel = MLP(
            in_dim=2 * coord_dim + coord_dim + 1 + 2 * channels,  # p, z, p-z, r, feat_p, feat_z
            hidden_dim=kernel_hidden,
            out_dim=channels,
            n_layers=2,
            activation=activation,
            dropout=dropout,
        )

        # latent -> latent
        self.lat_kernel = MLP(
            in_dim=2 * coord_dim + coord_dim + 1 + 2 * channels,
            hidden_dim=kernel_hidden,
            out_dim=channels,
            n_layers=2,
            activation=activation,
            dropout=dropout,
        )

        # latent -> point
        self.dec_kernel = MLP(
            in_dim=2 * coord_dim + coord_dim + 1 + 2 * channels,
            hidden_dim=kernel_hidden,
            out_dim=channels,
            n_layers=2,
            activation=activation,
            dropout=dropout,
        )

        self.pointwise = nn.Sequential(
            nn.Linear(channels, point_hidden),
            nn.GELU() if activation == "gelu" else (nn.SiLU() if activation == "silu" else nn.ReLU()),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(point_hidden, channels),
        )

    @staticmethod
    def _pair_features(a_coords, b_coords, a_feat, b_feat, eps=1e-6):
        """
        a_coords: [B, A, 3]
        b_coords: [B, Bn, 3]
        a_feat:   [B, A, C]
        b_feat:   [B, Bn, C]

        returns features [B, A, Bn, ...]
        """
        rel = a_coords.unsqueeze(2) - b_coords.unsqueeze(1)          # [B, A, Bn, 3]
        rel_norm = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + eps)

        a_c = a_coords.unsqueeze(2).expand(-1, -1, b_coords.shape[1], -1)
        b_c = b_coords.unsqueeze(1).expand(-1, a_coords.shape[1], -1, -1)
        a_f = a_feat.unsqueeze(2).expand(-1, -1, b_feat.shape[1], -1)
        b_f = b_feat.unsqueeze(1).expand(-1, a_feat.shape[1], -1, -1)

        return torch.cat([a_c, b_c, rel, rel_norm, a_f, b_f], dim=-1)

    def forward(self, u, coords, weights):
        """
        u:      [B, N, C]
        coords: [B, N, 3]
        weights:[B, N]
        """
        B, N, C = u.shape
        assert C == self.channels

        u0 = u
        u = self.norm_in(u)

        # normalize coordinates batchwise for numerical stability
        cmin = coords.min(dim=1, keepdim=True).values
        cmax = coords.max(dim=1, keepdim=True).values
        cspan = (cmax - cmin).clamp_min(1e-6)
        coords_n = 2.0 * (coords - cmin) / cspan - 1.0  # [B, N, 3]

        # learned latent coordinates repeated across batch
        z = self.latent_coords.unsqueeze(0).expand(B, -1, -1)  # [B, M, 3]

        # latent initial state from latent coords
        h = self.latent_init(z)  # [B, M, C]

        # --------------------------------------------------------
        # encode: points -> latents
        # --------------------------------------------------------
        p_q = self.q_point(u)
        p_k = self.k_point(u)
        p_v = self.v_point(u)

        z_q = self.q_latent(h)

        enc_feat = self._pair_features(z, coords_n, z_q, p_k)   # [B, M, N, ...]
        enc_w = self.enc_kernel(enc_feat)                       # [B, M, N, C]

        src = (p_v * weights.unsqueeze(-1)).unsqueeze(1)        # [B, 1, N, C]

        if self.use_softmax:
            # attention-like normalization over source points
            score = enc_w.mean(dim=-1)                          # [B, M, N]
            alpha = torch.softmax(score, dim=-1).unsqueeze(-1)
            h = h + (alpha * src).sum(dim=2)
        else:
            h = h + (enc_w * src).sum(dim=2)

        # --------------------------------------------------------
        # global latent interaction: latents -> latents
        # --------------------------------------------------------
        h_in = h
        h = self.norm_latent(h)

        z_q2 = self.q_latent(h)
        z_k2 = self.k_latent(h)
        z_v2 = self.v_latent(h)

        lat_feat = self._pair_features(z, z, z_q2, z_k2)       # [B, M, M, ...]
        lat_w = self.lat_kernel(lat_feat)                       # [B, M, M, C]

        if self.use_softmax:
            score = lat_w.mean(dim=-1)                          # [B, M, M]
            alpha = torch.softmax(score, dim=-1).unsqueeze(-1)
            h = h_in + (alpha * z_v2.unsqueeze(1)).sum(dim=2)
        else:
            h = h_in + (lat_w * z_v2.unsqueeze(1)).sum(dim=2)

        # --------------------------------------------------------
        # decode: latents -> points
        # --------------------------------------------------------
        p_q2 = self.q_point(u)
        z_k3 = self.k_latent(h)
        z_v3 = self.v_latent(h)

        dec_feat = self._pair_features(coords_n, z, p_q2, z_k3)  # [B, N, M, ...]
        dec_w = self.dec_kernel(dec_feat)                         # [B, N, M, C]

        if self.use_softmax:
            score = dec_w.mean(dim=-1)                            # [B, N, M]
            beta = torch.softmax(score, dim=-1).unsqueeze(-1)
            upd = (beta * z_v3.unsqueeze(1)).sum(dim=2)           # [B, N, C]
        else:
            upd = (dec_w * z_v3.unsqueeze(1)).sum(dim=2)

        out = u0 + self.pointwise(upd)
        return out


# ============================================================
# full model
# ============================================================

class GlobalNeuralOperator3D(nn.Module):
    """
    Practical global operator:
      (B, Cin, D, Ny, Nx) -> (B, Cout, D, Ny, Nx)

    Properties:
    - global coupling over whole cube
    - no xy/z split
    - no dependence of parameters on Ny or Nx
    - dx, dy passed in forward
    - cmass_grid passed in forward
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        n_latents: int = 128,
        lifting_hidden: int = 128,
        projection_hidden: int = 128,
        kernel_hidden: int = 128,
        point_hidden: int = 128,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_softmax: bool = True,
    ):
        super().__init__()

        if activation == "gelu":
            act = nn.GELU
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "relu":
            act = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.lift = nn.Sequential(
            nn.Linear(in_channels, lifting_hidden),
            act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(lifting_hidden, hidden_channels),
        )

        self.blocks = nn.ModuleList([
            LatentKernelBlock3D(
                channels=hidden_channels,
                n_latents=n_latents,
                kernel_hidden=kernel_hidden,
                point_hidden=point_hidden,
                activation=activation,
                dropout=dropout,
                use_softmax=use_softmax,
            )
            for _ in range(n_layers)
        ])

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, projection_hidden),
            act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(projection_hidden, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,                   # [B, Cin, D, Ny, Nx]
        dx: Union[float, int, torch.Tensor],
        dy: Union[float, int, torch.Tensor],
        cmass_grid: torch.Tensor,          # [D] or [B, D]
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"x must be [B, Cin, D, Ny, Nx], got {tuple(x.shape)}")

        B, Cin, D, Ny, Nx = x.shape

        coords, weights = build_coords_and_weights(x, dx=dx, dy=dy, cmass_grid=cmass_grid)

        # [B, Cin, D, Ny, Nx] -> [B, N, Cin]
        u = x.permute(0, 2, 3, 4, 1).reshape(B, D * Ny * Nx, Cin)

        u = self.lift(u)
        for block in self.blocks:
            u = block(u, coords, weights)
        u = self.proj(u)

        # [B, N, Cout] -> [B, Cout, D, Ny, Nx]
        y = u.reshape(B, D, Ny, Nx, -1).permute(0, 4, 1, 2, 3).contiguous()
        return y

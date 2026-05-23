import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.ffno_model import SpectralConv2dFull as SpectralConv2dFull3D
from models.ffno_model import BalancedVerticalPhysicsStack
from models.ffno_z1d_model import SpectralConv2dFull as SpectralConv2dFullZ1D
from models.ffno_z1d_model import ZNeuralOperator1d


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def partition_range(n, rank=None, world_size=None):
    rank = get_rank() if rank is None else rank
    world_size = get_world_size() if world_size is None else world_size
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    stop = start + base + (1 if rank < rem else 0)
    return start, stop


def partition_sizes(n, world_size=None):
    world_size = get_world_size() if world_size is None else world_size
    return [partition_range(n, r, world_size)[1] - partition_range(n, r, world_size)[0] for r in range(world_size)]


def _to_spacing_value(spacing):
    if isinstance(spacing, (float, int)):
        return float(spacing)
    if torch.is_tensor(spacing):
        if spacing.numel() != 1:
            raise ValueError("spacing must be scalar.")
        return float(spacing.detach().item())
    raise TypeError(f"Unsupported spacing type: {type(spacing)}")


def _all_to_all_tensor_list(input_list, output_shapes, concat_dim=None):
    if not is_dist():
        return input_list[0] if concat_dim is not None else [input_list[0]]

    device = input_list[0].device
    dtype = input_list[0].dtype
    ndim = input_list[0].ndim
    world_size = get_world_size()

    max_shape = [0] * ndim
    for shape in [tuple(t.shape) for t in input_list] + [tuple(s) for s in output_shapes]:
        for i, value in enumerate(shape):
            max_shape[i] = max(max_shape[i], int(value))

    max_shape_t = torch.tensor(max_shape, device=device, dtype=torch.int64)
    dist.all_reduce(max_shape_t, op=dist.ReduceOp.MAX)
    max_shape = [int(v) for v in max_shape_t.cpu().tolist()]

    padded = []
    for tensor in input_list:
        out = torch.zeros(max_shape, device=device, dtype=dtype)
        slices = tuple(slice(0, s) for s in tensor.shape)
        out[slices] = tensor
        padded.append(out)

    send = torch.stack(padded, dim=0).contiguous()
    del padded

    if send.is_complex():
        send_real = torch.view_as_real(send)
        recv_real = torch.empty(
            (world_size, *max_shape, 2),
            device=device,
            dtype=send_real.dtype,
        )
        dist.all_to_all_single(recv_real, send_real)
        del send_real
        recv = torch.view_as_complex(recv_real.contiguous())
        del recv_real
    else:
        recv = torch.empty((world_size, *max_shape), device=device, dtype=dtype)
        dist.all_to_all_single(recv, send)
    del send

    output_list = []
    for src, shape in enumerate(output_shapes):
        slices = (src, *[slice(0, int(s)) for s in shape])
        output_list.append(recv[slices])
    if concat_dim is not None:
        out = torch.cat(output_list, dim=concat_dim)
        del recv, output_list
        return out
    del recv
    return output_list


class DistributedGroupNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.num_groups = source.num_groups
        self.num_channels = source.num_channels
        self.eps = source.eps
        self.affine = source.affine
        if source.affine:
            self.weight = source.weight
            self.bias = source.bias
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if not is_dist():
            return F.group_norm(
                x,
                self.num_groups,
                self.weight,
                self.bias,
                self.eps,
            )

        b, c = x.shape[:2]
        if c != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {c}")

        xg = x.reshape(b, self.num_groups, c // self.num_groups, *x.shape[2:])
        reduce_dims = tuple(range(2, xg.ndim))
        local_sum = xg.sum(dim=reduce_dims, keepdim=True)
        local_sq = (xg * xg).sum(dim=reduce_dims, keepdim=True)
        local_count = torch.tensor(
            xg[0, 0].numel(),
            device=x.device,
            dtype=x.dtype,
        )

        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        mean = local_sum / local_count
        var = local_sq / local_count - mean * mean
        y = (xg - mean) * torch.rsqrt(var.clamp_min(0.0) + self.eps)
        y = y.reshape_as(x)

        if self.affine:
            y = y * self.weight.view(1, -1, *([1] * (x.ndim - 2)))
            y = y + self.bias.view(1, -1, *([1] * (x.ndim - 2)))
        return y


class DistributedHDepthwiseConv3d(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.conv = source
        self.progress_label = None
        kd, kh, kw = self.conv.kernel_size
        pd, ph, pw = self.conv.padding
        if kh != 3 or ph != 1:
            raise ValueError("DistributedHDepthwiseConv3d currently expects kernel_size/padding 3/1 over H")
        self.local_padding = (pd, 0, pw)

    def _progress(self, message):
        if get_rank() == 0 and self.progress_label is not None:
            print(f"[{self.progress_label}] {message}", flush=True)

    def forward(self, x):
        if not is_dist():
            return self.conv(x)

        rank = get_rank()
        world_size = get_world_size()
        left_rank = rank - 1
        right_rank = rank + 1

        left = torch.zeros_like(x[:, :, :, :1, :])
        right = torch.zeros_like(x[:, :, :, :1, :])

        ops = []
        if left_rank >= 0:
            recv_left = torch.empty_like(left)
            ops.append(dist.P2POp(dist.irecv, recv_left, left_rank))
            ops.append(dist.P2POp(dist.isend, x[:, :, :, :1, :].contiguous(), left_rank))
        else:
            recv_left = left

        if right_rank < world_size:
            recv_right = torch.empty_like(right)
            ops.append(dist.P2POp(dist.irecv, recv_right, right_rank))
            ops.append(dist.P2POp(dist.isend, x[:, :, :, -1:, :].contiguous(), right_rank))
        else:
            recv_right = right

        self._progress("halo exchange")
        for req in dist.batch_isend_irecv(ops):
            req.wait()

        self._progress("local depthwise conv")
        x_halo = torch.cat([recv_left, x, recv_right], dim=3)
        y = F.conv3d(
            x_halo,
            self.conv.weight,
            self.conv.bias,
            self.conv.stride,
            self.local_padding,
            self.conv.dilation,
            self.conv.groups,
        )
        self._progress("done")
        return y


class ProgressVerticalWrapper(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.progress_label = None

    def _progress(self, message):
        if get_rank() == 0 and self.progress_label is not None:
            print(f"[{self.progress_label}] {message}", flush=True)

    def forward(self, x, z_scale):
        self._progress("start")
        if isinstance(self.source, BalancedVerticalPhysicsStack):
            y = self._forward_balanced_vertical(x, z_scale)
        elif isinstance(self.source, ZNeuralOperator1d):
            y = self._forward_z1d_vertical(x, z_scale)
        else:
            y = self.source(x, z_scale)
        self._progress("done")
        return y

    def _iter_columns(self, total):
        iterator = range(0, total, self.source.chunk)
        if get_rank() != 0:
            return iterator

        return tqdm(
            iterator,
            total=(total + self.source.chunk - 1) // self.source.chunk,
            desc=self.progress_label or "vertical",
            leave=False,
        )

    def _forward_balanced_vertical(self, x, z_scale):
        B, C, D, H, W = x.shape
        if z_scale.ndim == 4:
            z_scale = z_scale.unsqueeze(1)
        if z_scale.shape != (B, 1, D, H, W):
            raise ValueError(
                f"z_scale must be [B, D, H, W] or [B, 1, D, H, W], got {tuple(z_scale.shape)}"
            )

        x_cols = x.permute(0, 3, 4, 1, 2).reshape(B, H * W, C, D)
        z_cols = z_scale.permute(0, 3, 4, 1, 2).reshape(B, H * W, 1, D)

        chunks = []
        for i in self._iter_columns(H * W):
            j = min(i + self.source.chunk, H * W)
            xi = x_cols[:, i:j].reshape(-1, C, D)
            zi = z_cols[:, i:j].reshape(-1, 1, D)
            z_feat = self.source._z_features(zi)

            yi = self.source.in_proj(xi) + self.source.z_proj(z_feat)
            yi = self.source.net(yi)
            yi = yi * self.source.depth_gate(yi)
            yi = self.source.out_proj(yi)

            yi = yi.reshape(B, j - i, C, D)
            chunks.append(yi)

        y = torch.cat(chunks, dim=1)
        y = y.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2).contiguous()
        return self.source.out_gn(y)

    def _forward_z1d_vertical(self, x, z_scale):
        B, C, D, H, W = x.shape
        if z_scale.ndim == 4:
            z_scale = z_scale.unsqueeze(1)
        if z_scale.shape != (B, 1, D, H, W):
            raise ValueError(
                f"z_scale must be [B, D, H, W] or [B, 1, D, H, W], got {tuple(z_scale.shape)}"
            )

        x_cols = x.permute(0, 3, 4, 1, 2).reshape(B, H * W, C, D)
        z_cols = z_scale.permute(0, 3, 4, 1, 2).reshape(B, H * W, D)

        chunks = []
        for i in self._iter_columns(H * W):
            j = min(i + self.source.chunk, H * W)
            xi = x_cols[:, i:j].reshape(-1, C, D)
            zi = z_cols[:, i:j].reshape(-1, D)

            yi = self.source.in_proj(xi)
            yi = self.source.operator(yi, zi) + self.source.local_mix(yi)
            yi = yi * self.source.depth_gate(yi)
            yi = self.source.drop(yi)
            yi = self.source.out_proj(yi)

            yi = yi.reshape(B, j - i, C, D)
            chunks.append(yi)

        y = torch.cat(chunks, dim=1)
        y = y.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2).contiguous()
        return self.source.out_gn(y)


class DistributedSpectralConv2dFull(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.progress_label = None

    def _progress(self, message):
        if get_rank() == 0 and self.progress_label is not None:
            print(f"[{self.progress_label}] {message}", flush=True)

    def forward(self, x, dx, dy):
        if not is_dist():
            return self.source(x, dx, dy)

        b, c, d, h_local, w = x.shape
        rank = get_rank()
        world_size = get_world_size()
        h_sizes = [torch.tensor(h_local, device=x.device, dtype=torch.int64)]
        gathered_h = [torch.empty_like(h_sizes[0]) for _ in range(world_size)]
        dist.all_gather(gathered_h, h_sizes[0])
        h_chunks = [int(t.item()) for t in gathered_h]
        h_global = sum(h_chunks)

        dx = _to_spacing_value(dx)
        dy = _to_spacing_value(dy)

        self._progress("input gate")
        g = self.source.input_gate(x)
        x = x * g

        self._progress("rfft over W")
        x_w = torch.fft.rfft(x, dim=-1)
        wf = x_w.shape[-1]
        wf_chunks = partition_sizes(wf, world_size)
        wf_start, wf_stop = partition_range(wf, rank, world_size)
        wf_local = wf_stop - wf_start

        self._progress("all-to-all H -> frequency slabs")
        to_wf_owner = [chunk.contiguous() for chunk in torch.split(x_w, wf_chunks, dim=-1)]
        from_h_owner_shapes = [
            (b, c, d, h_chunks[src], wf_local)
            for src in range(world_size)
        ]
        x_w = _all_to_all_tensor_list(
            to_wf_owner,
            from_h_owner_shapes,
            concat_dim=3,
        )
        del to_wf_owner

        self._progress("fft over H")
        x_ft = torch.fft.fft(x_w, dim=-2)
        del x_w

        ky = torch.fft.fftfreq(h_global, d=dy, device=x.device)
        kx = torch.fft.rfftfreq(w, d=dx, device=x.device)[wf_start:wf_stop]
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")

        k_feat = torch.stack(
            [
                kx,
                ky,
                torch.sqrt(kx ** 2 + ky ** 2 + 1e-12),
                torch.sign(ky),
            ],
            dim=-1,
        )
        gate = self.source.freq_mlp(k_feat)

        wr = self.source.weight_real[:, :, None, None]
        wi = self.source.weight_imag[:, :, None, None]
        gr = gate[..., 0][None, None]
        gi = gate[..., 1][None, None]

        self._progress("spectral channel mix")
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
        del xr, xi, x_ft, w_real, w_imag, wr, wi, gr, gi, gate, k_feat, ky, kx

        out = torch.complex(out_r, out_i)
        del out_r, out_i
        self._progress("ifft over H")
        out = torch.fft.ifft(out, dim=-2)

        self._progress("all-to-all frequency -> H slabs")
        out_channels = out.shape[1]
        to_h_owner = [chunk.contiguous() for chunk in torch.split(out, h_chunks, dim=3)]
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from_wf_owner_shapes = [
            (b, out_channels, d, h_local, wf_chunks[src])
            for src in range(world_size)
        ]
        out_w = _all_to_all_tensor_list(
            to_h_owner,
            from_wf_owner_shapes,
            concat_dim=-1,
        )
        del to_h_owner

        self._progress("irfft over W")
        y = torch.fft.irfft(out_w, n=w, dim=-1)
        del out_w
        self._progress("post mix")
        y = self.source.post(y)
        self._progress("done")
        return y


def _replace_modules(parent):
    for name, child in list(parent.named_children()):
        if isinstance(child, (SpectralConv2dFull3D, SpectralConv2dFullZ1D)):
            setattr(parent, name, DistributedSpectralConv2dFull(child))
            continue

        if isinstance(child, (BalancedVerticalPhysicsStack, ZNeuralOperator1d)):
            setattr(parent, name, ProgressVerticalWrapper(child))
            continue

        if isinstance(child, nn.GroupNorm):
            setattr(parent, name, DistributedGroupNorm(child))
            continue

        if (
            isinstance(child, nn.Conv3d)
            and child.groups == child.in_channels == child.out_channels
            and child.kernel_size == (3, 3, 3)
            and child.padding == (1, 1, 1)
        ):
            setattr(parent, name, DistributedHDepthwiseConv3d(child))
            continue

        _replace_modules(child)


def enable_distributed_inference(model):
    _replace_modules(model)
    return model

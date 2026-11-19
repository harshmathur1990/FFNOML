import torch
import torch.autograd.forward_ad as fwAD
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.ffno_model import SpectralConv2dFull as SpectralConv2dFull3D
from models.ffno_model import BalancedVerticalPhysicsStack, FFNOBlock3dBalanced, HeightSpectralConv, depth_weights

try:
    from config import DEBUG_DIST_INFERENCE
except Exception:
    DEBUG_DIST_INFERENCE = False


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def debug_dist_inference_enabled():
    return bool(DEBUG_DIST_INFERENCE)


def debug_progress(progress_label, message):
    if (
        debug_dist_inference_enabled()
        and get_rank() == 0
        and progress_label is not None
    ):
        print(f"[{progress_label}] {message}", flush=True)


# The batched mode of the FSDP route: a step whose working tensors on the rank's slab would not fit in this fraction of
# the device's free memory runs over the slab a few levels (or, the height operator, rows) at a time; one that fits runs
# in one pass. FREE_MEMORY_BYTES, when set, stands in for the device's free memory (tests).
DEVICE_MEMORY_FRACTION = 0.7
FREE_MEMORY_BYTES = None


def free_device_memory(device):
    """Bytes the caching allocator can still hand out on device (free memory plus its own unused cache); None, no
    limit, off the GPU."""
    if FREE_MEMORY_BYTES is not None:
        return FREE_MEMORY_BYTES
    if device.type != "cuda":
        return None
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def depth_chunk(x, copies, collective=False):
    """Levels of x [B, C, D, H, W] per pass of a step that holds `copies` tensors of the size of its input: all D
    when they fit in DEVICE_MEMORY_FRACTION of the free device memory. collective: the smallest over the ranks, for a
    step with a collective in every pass."""
    depth = x.shape[2]
    chunk = depth
    free = free_device_memory(x.device)
    if free is not None:
        per_level = copies * x[:, :, :1].numel() * x.element_size()
        chunk = int(max(1, min(depth, DEVICE_MEMORY_FRACTION * free // per_level)))
    if collective and is_dist():
        smallest = torch.tensor([chunk], device=x.device)
        dist.all_reduce(smallest, op=dist.ReduceOp.MIN)
        chunk = int(smallest.item())
    return chunk


def depth_spans(depth, chunk):
    return [(a, min(a + chunk, depth)) for a in range(0, depth, chunk)]


def block_depth_weights(block, z_scale):
    """depth_weights for the GroupNorms of a vertical_operator block (DepthWeightedGroupNorm), None for any other."""
    return depth_weights(z_scale) if getattr(block, "vertical_operator", False) else None


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


def _dual_parts(tensor):
    """(primal, tangent) of a forward-mode AD dual tensor; the tangent is None for an ordinary tensor."""
    unpacked = fwAD.unpack_dual(tensor)
    return unpacked.primal, unpacked.tangent


def _linear_collective(operation, tensor):
    """Apply a linear communication step to a dual tensor: to its primal and to its tangent separately.

    The collectives and the halo exchange are autograd Functions without forward-mode rules; being linear, the
    tangent goes through the same exchange. Every rank holds a dual during a JVP, so the calls stay matched.
    """
    primal, tangent = _dual_parts(tensor)
    if tangent is None:
        return operation(tensor)
    return fwAD.make_dual(operation(primal), operation(tangent))


def _as_dist_tensor(tensor):
    if tensor.is_complex():
        return torch.view_as_real(tensor)
    return tensor


def _empty_dist_tensor(shape, *, device, dtype):
    tensor = torch.empty(shape, device=device, dtype=dtype)
    return tensor, _as_dist_tensor(tensor)


def _all_to_all_tensor_list(input_list, output_shapes, concat_dim=None):
    if not is_dist():
        return input_list[0] if concat_dim is not None else [input_list[0]]

    world_size = get_world_size()

    parts = [_dual_parts(value) for value in input_list]
    if any(tangent is not None for _, tangent in parts):
        primals = [primal for primal, _ in parts]
        tangents = [torch.zeros_like(primal) if tangent is None else tangent for primal, tangent in parts]
        out_primal = _all_to_all_tensor_list(primals, output_shapes, concat_dim)
        out_tangent = _all_to_all_tensor_list(tangents, output_shapes, concat_dim)
        if concat_dim is not None:
            return fwAD.make_dual(out_primal, out_tangent)
        return [fwAD.make_dual(p, t) for p, t in zip(out_primal, out_tangent)]

    if len(input_list) != world_size or len(output_shapes) != world_size:
        raise ValueError(
            f"Expected {world_size} tensors/shapes, got "
            f"{len(input_list)} tensors and {len(output_shapes)} shapes."
        )

    # The previous hand-written P2P ring was correct in no-grad prediction but
    # severed autograd at every distributed FFT transpose.  The functional
    # collective supplies the reverse all-to-all needed by the FFNO VJP.
    reference = input_list[0]
    is_complex = reference.is_complex()
    input_dist = [
        (_as_dist_tensor(value.contiguous()) if is_complex else value.contiguous())
        for value in input_list
    ]
    output_dist = []
    for shape in output_shapes:
        real_shape = tuple(int(s) for s in shape) + ((2,) if is_complex else ())
        output_dist.append(torch.empty(
            real_shape,
            device=reference.device,
            dtype=reference.real.dtype if is_complex else reference.dtype,
        ))
    received = list(dist_nn.all_to_all(output_dist, input_dist))
    output_list = (
        [torch.view_as_complex(value) for value in received]
        if is_complex
        else received
    )

    if concat_dim is not None:
        out = torch.cat(output_list, dim=concat_dim)
        del output_list
        return out
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

    def forward(self, x, w=None):
        """w: depth_weights [B, 1, D, H, W] of the local slab for a DepthWeightedGroupNorm (vertical_operator), whose
        statistics weight each level by the height it stands for; None for a plain GroupNorm."""
        if not is_dist() and w is None:
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

        group = lambda t: t.reshape(b, self.num_groups, c // self.num_groups, *t.shape[2:])   # noqa: E731
        reduce_dims = tuple(range(2, x.ndim + 1))
        spans = depth_spans(x.shape[2], depth_chunk(x, 3))        # the batched mode: statistics, then values, by levels
        local_sum = local_sq = local_count = 0
        for a, z in spans:
            xg = group(x[:, :, a:z])
            if w is None:
                local_sum = local_sum + xg.sum(dim=reduce_dims, keepdim=True)
                local_sq = local_sq + (xg * xg).sum(dim=reduce_dims, keepdim=True)
                local_count = local_count + xg[0, 0].numel()
            else:
                wg = w[:, :, a:z].unsqueeze(1)                      # [B, 1, 1, levels, H, W]
                local_sum = local_sum + (wg * xg).sum(dim=reduce_dims, keepdim=True)
                local_sq = local_sq + (wg * xg * xg).sum(dim=reduce_dims, keepdim=True)
                local_count = local_count + wg.sum(dim=reduce_dims, keepdim=True) * xg.shape[2]
        if w is None:
            local_count = torch.tensor(local_count, device=x.device, dtype=x.dtype)

        if is_dist():
            reduce_sum = lambda t: dist_nn.all_reduce(t, op=dist.ReduceOp.SUM)
            local_sum = _linear_collective(reduce_sum, local_sum)
            local_sq = _linear_collective(reduce_sum, local_sq)
            local_count = _linear_collective(reduce_sum, local_count)

        mean = local_sum / local_count
        var = local_sq / local_count - mean * mean
        inverse = torch.rsqrt(var.clamp_min(0.0) + self.eps)

        def normalized(part):
            y = ((group(part) - mean) * inverse).reshape_as(part)
            if self.affine:
                y = y * self.weight.view(1, -1, *([1] * (x.ndim - 2)))
                y = y + self.bias.view(1, -1, *([1] * (x.ndim - 2)))
            return y

        if len(spans) == 1:
            return normalized(x)
        return torch.cat([normalized(x[:, :, a:z]) for a, z in spans], dim=2)


class _HaloExchangeH(torch.autograd.Function):
    """Exchange one H-boundary cell and reverse the exchange in the VJP."""

    @staticmethod
    def forward(ctx, x):
        rank = get_rank()
        world_size = get_world_size()
        left_rank = rank - 1
        right_rank = rank + 1
        left = torch.zeros_like(x[:, :, :, :1, :])
        right = torch.zeros_like(x[:, :, :, :1, :])
        ops = []
        if left_rank >= 0:
            left = torch.empty_like(left)
            ops.append(dist.P2POp(dist.irecv, left, left_rank))
            ops.append(dist.P2POp(
                dist.isend, x[:, :, :, :1, :].contiguous(), left_rank
            ))
        if right_rank < world_size:
            right = torch.empty_like(right)
            ops.append(dist.P2POp(dist.irecv, right, right_rank))
            ops.append(dist.P2POp(
                dist.isend, x[:, :, :, -1:, :].contiguous(), right_rank
            ))
        for request in dist.batch_isend_irecv(ops):
            request.wait()
        ctx.rank = rank
        ctx.world_size = world_size
        return torch.cat([left, x, right], dim=3)

    @staticmethod
    def backward(ctx, grad_halo):
        rank = ctx.rank
        world_size = ctx.world_size
        left_rank = rank - 1
        right_rank = rank + 1
        grad = grad_halo[:, :, :, 1:-1, :].contiguous()
        from_left = None
        from_right = None
        ops = []
        if left_rank >= 0:
            from_left = torch.empty_like(grad[:, :, :, :1, :])
            ops.append(dist.P2POp(dist.irecv, from_left, left_rank))
            ops.append(dist.P2POp(
                dist.isend, grad_halo[:, :, :, :1, :].contiguous(), left_rank
            ))
        if right_rank < world_size:
            from_right = torch.empty_like(grad[:, :, :, -1:, :])
            ops.append(dist.P2POp(dist.irecv, from_right, right_rank))
            ops.append(dist.P2POp(
                dist.isend, grad_halo[:, :, :, -1:, :].contiguous(), right_rank
            ))
        for request in dist.batch_isend_irecv(ops):
            request.wait()
        if from_left is not None:
            grad[:, :, :, :1, :] += from_left
        if from_right is not None:
            grad[:, :, :, -1:, :] += from_right
        return grad


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
        debug_progress(self.progress_label, message)

    def forward(self, x):
        if not is_dist():
            return self.conv(x)
        self._progress("halo exchange")
        x_halo = _linear_collective(_HaloExchangeH.apply, x)
        self._progress("local depthwise conv")
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
        debug_progress(self.progress_label, message)

    def forward(self, x, z_scale):
        self._progress("start")
        if isinstance(self.source, BalancedVerticalPhysicsStack):
            y = self._forward_balanced_vertical(x, z_scale)
        else:
            y = self.source(x, z_scale)
        self._progress("done")
        return y

    def _iter_columns(self, total):
        iterator = range(0, total, self.source.chunk)
        if get_rank() != 0 or not debug_dist_inference_enabled():
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

        if self.source.z_spec is not None and self.source.z_reference_length > 0:
            self.source.check_height_span(z_scale)
        x_cols = x.permute(0, 3, 4, 1, 2).reshape(B, H * W, C, D)
        z_cols = z_scale.permute(0, 3, 4, 1, 2).reshape(B, H * W, 1, D)

        chunks = []
        for i in self._iter_columns(H * W):
            j = min(i + self.source.chunk, H * W)
            xi = x_cols[:, i:j].reshape(-1, C, D)
            zi = z_cols[:, i:j].reshape(-1, 1, D)
            yi = self.source.columns(xi, zi).reshape(B, j - i, C, D)
            chunks.append(yi)

        y = torch.cat(chunks, dim=1)
        y = y.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2).contiguous()
        return self.source.out_gn(y)


class ChunkedHeightSpectralConv(nn.Module):
    """HeightSpectralConv on the local H slab, a few rows of columns at a time when its mode spectra ([B, C, M, rows,
    W], several of them at once) would not fit in the free device memory. The operator acts on each column alone, so the
    chunks give the same result. budget (bytes for the spectra) overrides the free memory."""

    def __init__(self, source, budget=None):
        super().__init__()
        self.source = source
        self.budget = budget
        self.progress_label = None

    def forward(self, x, z_scale):
        if z_scale.ndim == 4:
            z_scale = z_scale.unsqueeze(1)
        b, c, _, h, w = x.shape
        modes = self.source.box(z_scale)[1].numel()               # more than source.modes for columns over L / 2
        per_row = 8 * b * c * modes * w * x.element_size()
        budget = self.budget
        if budget is None:
            free = free_device_memory(x.device)
            # the chunks' outputs and their concatenation take two slabs of their own
            budget = float("inf") if free is None else DEVICE_MEMORY_FRACTION * free - 2 * x.numel() * x.element_size()
        rows = max(1, int(min(h, budget // per_row)))
        if rows >= h:
            return self.source(x, z_scale)
        debug_progress(self.progress_label, f"{h} rows in chunks of {rows}")
        return torch.cat([self.source(x[:, :, :, i:i + rows], z_scale[:, :, :, i:i + rows])
                          for i in range(0, h, rows)], dim=3)


class DepthChunkedPointwise(nn.Module):
    """A block's path that acts on each cell alone (pw, mlp), over the local slab a few levels at a time when its working
    tensors, `copies` of its input's size, would not fit in the free device memory."""

    def __init__(self, source, copies=5):
        super().__init__()
        self.source = source
        self.copies = copies

    def forward(self, x):
        spans = depth_spans(x.shape[2], depth_chunk(x, self.copies))
        if len(spans) == 1:
            return self.source(x)
        return torch.cat([self.source(x[:, :, a:z]) for a, z in spans], dim=2)


def chunked_fuse_branches(block, x, spec, vert, chunk_depth=32, normalize=True):
    """
    Inference-only version of FFNOBlock3dBalanced._fuse_branches.

    The standard path materializes full-volume concatenations with 3*width and
    2*width channels. On full 384^3 slabs this exceeds 16 GB T4 memory. Chunking
    over depth keeps peak memory bounded while preserving the 3x3x3 fuse
    depthwise convolution with a one-cell depth halo.
    """
    _, _, depth, _, _ = x.shape
    fused = None                                   # filled in place, so no chunk list and concatenation on top of it

    for d0 in range(0, depth, chunk_depth):
        d1 = min(d0 + chunk_depth, depth)
        halo0 = max(0, d0 - 1)
        halo1 = min(depth, d1 + 1)
        inner0 = d0 - halo0
        inner1 = inner0 + (d1 - d0)

        x_h = x[:, :, halo0:halo1]
        spec_h = spec[:, :, halo0:halo1]
        vert_h = vert[:, :, halo0:halo1]

        gates = block.branch_gate(torch.cat([x_h, spec_h, vert_h], dim=1))
        spec_gate, vert_gate = gates.chunk(2, dim=1)
        spec_mix = spec_h * spec_gate
        vert_mix = vert_h * vert_gate
        del gates, spec_gate, vert_gate, x_h, spec_h, vert_h

        fused_h = block.fuse(torch.cat([spec_mix, vert_mix], dim=1))
        fused_h = fused_h + 0.5 * (spec_mix + vert_mix)
        if fused is None:
            fused = fused_h.new_empty(fused_h.shape[0], fused_h.shape[1], depth, *fused_h.shape[3:])
        fused[:, :, d0:d1] = fused_h[:, :, inner0:inner1]
        del spec_mix, vert_mix, fused_h

        if torch.cuda.is_available() and x.is_cuda:
            torch.cuda.empty_cache()

    if normalize:
        fused = block.act(block.norm_fuse(fused))
    return fused


class DistributedSpectralConv2dFull(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.progress_label = None

    def _progress(self, message):
        debug_progress(self.progress_label, message)

    def forward(self, x, dx, dy):
        if not is_dist():
            return self.source(x, dx, dy)
        # each level is transformed and mixed on its own: the batched mode runs the transposes a few levels at a time
        spans = depth_spans(x.shape[2], depth_chunk(x, 8, collective=True))
        if len(spans) == 1:
            return self._levels(x, dx, dy)
        self._progress(f"{x.shape[2]} levels in {len(spans)} passes")
        return torch.cat([self._levels(x[:, :, a:z], dx, dy) for a, z in spans], dim=2)

    def _levels(self, x, dx, dy):
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

        # Create the cuBLAS handle before the large spectral work buffers fill
        # the device. Otherwise the small freq_mlp call can fail at cublasCreate.
        if torch.cuda.is_available() and x.is_cuda:
            _ = self.source.freq_mlp(torch.zeros(1, 4, device=x.device, dtype=x.dtype))

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
        if torch.cuda.is_available() and x.is_cuda:
            torch.cuda.empty_cache()

        ky = torch.fft.fftfreq(h_global, d=dy, device=x.device)
        kx = torch.fft.rfftfreq(w, d=dx, device=x.device)[wf_start:wf_stop]
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")

        self._progress("spectral channel mix")
        out = self.source.mix_spectrum(x_ft, self.source.k_features(ky, kx))
        del x_ft, ky, kx
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
        if isinstance(child, SpectralConv2dFull3D):
            setattr(parent, name, DistributedSpectralConv2dFull(child))
            continue

        if isinstance(child, BalancedVerticalPhysicsStack):
            setattr(parent, name, ProgressVerticalWrapper(child))
            continue

        if isinstance(child, HeightSpectralConv):
            setattr(parent, name, ChunkedHeightSpectralConv(child))
            continue

        if isinstance(child, FFNOBlock3dBalanced):
            child.pw = DepthChunkedPointwise(child.pw)
            child.mlp = DepthChunkedPointwise(child.mlp)

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
    """The FSDP route's modules, for a model split over the ranks in H slabs. A vertical_operator model's GroupNorms
    take the slab's depth_weights (block_depth_weights). In the batched mode, the spectral branch, the GroupNorms and the
    pointwise paths run over a few levels at a time, and HeightSpectralConv over a few rows, whenever a step would not fit
    in the device's free memory in one pass (depth_chunk)."""
    _replace_modules(model)
    return model

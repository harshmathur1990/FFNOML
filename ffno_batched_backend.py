"""Forward-only FFNO prediction of a whole cube from host RAM, for GPUs too small to hold its activations.

Not tiling: every operation of FFNO3D is either local or factorizes, so it can be batched without changing the operator.

    kernel-1 layers (lift, gates, post, pw, mlp, head)   any batch
    2D FFT and spectral multiply                         depth planes: each plane is already independent; the
                                                         multiply also over k_y rows, each frequency's own mixing
    vertical branch (HeightSpectralConv)                 rows of columns: it acts on each column alone
    GroupNorm over the volume                            streaming, height-weighted sum and sum of squares, applied in
                                                         a second pass

Hidden states stay in host RAM (22 GB each for 504 x 504 x 452, four at a time), and batches go through the GPUs that are
visible. There is no FSDP and no collective, and the checkpoint's own layers do the work. VJPs and JVPs are not available.

"""

import copy
import queue
import threading
import time

import numpy as np
import torch

from distributed_inference import chunked_fuse_branches
from ffno_fsdp_runtime import LOG10_POPULATION_LIMIT, Z_METRES_PER_MODEL_UNIT, VELOCITY_M_S_PER_MODEL_UNIT, linear_populations
from models.ffno_model import _to_spacing_value, depth_weights
from train_utils import get_checkpoint_io_metadata, get_checkpoint_normalization, load_checkpoint


SPECTRAL_WEIGHT_BYTES = 2 ** 30   # the largest per-frequency weight tensor, C x C x rows x Wf float32, made at once


def spectral_by_frequency_rows(spec, x, dx, dy, weight_bytes=SPECTRAL_WEIGHT_BYTES):
    """SpectralConv2dFull.forward with mix_spectrum over blocks of k_y rows: each frequency's channel mixing depends on
    that frequency alone, so this is the same operator, without the C x C x H x Wf weights of the whole plane at once
    (4.7 GB each at 504 x 504 and C = 96, more than a 16 GB GPU holds with their temporaries)."""
    _, channels, _, height, width = x.shape
    dx, dy = _to_spacing_value(dx, x), _to_spacing_value(dy, x)
    x = x * spec.input_gate(x)
    x_ft = torch.fft.rfft2(x, dim=(-2, -1))
    ky, kx = torch.meshgrid(torch.fft.fftfreq(height, d=dy, device=x.device),
                            torch.fft.rfftfreq(width, d=dx, device=x.device), indexing="ij")
    k_feat = spec.k_features(ky, kx)
    rows = max(1, int(weight_bytes // (4 * channels * channels * x_ft.shape[-1])))
    out = torch.empty_like(x_ft)
    for y0 in range(0, height, rows):
        out[..., y0:y0 + rows, :] = spec.mix_spectrum(x_ft[..., y0:y0 + rows, :], k_feat[y0:y0 + rows])
    del x_ft
    return spec.post(torch.fft.irfft2(out, s=(height, width), dim=(-2, -1)))


def _is_out_of_memory(exception):
    return isinstance(exception, torch.cuda.OutOfMemoryError) or "out of memory" in str(exception).lower()


class _GPUPool:
    """One model replica per visible GPU, and threads that take index ranges from a shared queue. A range that runs out
    of GPU memory is split in two and queued again; its results are written only once it has succeeded."""

    def __init__(self, model, devices):
        self.replicas = {device: copy.deepcopy(model).eval().to(device) for device in devices}
        self.splits = 0

    def run(self, total, step, work):
        tasks = queue.Queue()
        for start in range(0, total, step):
            tasks.put((start, min(start + step, total)))
        lock = threading.Lock()
        pending = [tasks.qsize()]
        failures = []

        def worker(device):
            replica = self.replicas[device]
            with torch.no_grad():  # grad mode is per thread
                while True:
                    with lock:
                        if failures or pending[0] == 0:
                            return
                    try:
                        a, b = tasks.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    try:
                        work(replica, device, a, b)
                    except RuntimeError as exception:
                        if not _is_out_of_memory(exception):
                            with lock:
                                failures.append(exception)
                            return
                        del exception
                        torch.cuda.empty_cache()
                        if b - a < 2:
                            with lock:
                                failures.append(RuntimeError(f"{device}: one index ({a}) does not fit in GPU memory"))
                            return
                        middle = (a + b) // 2
                        with lock:
                            pending[0] += 1
                            self.splits += 1
                        tasks.put((a, middle))
                        tasks.put((middle, b))
                        continue
                    with lock:
                        pending[0] -= 1

        threads = [threading.Thread(target=worker, args=(device,)) for device in self.replicas]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            raise failures[0]


class BatchedFFNOBackend:
    """A checkpoint's FFNO3D on the visible GPUs, predicting a whole cube of any size that fits in host RAM."""

    def __init__(self, checkpoint_path, factory, *, devices=None, chunk_depth=16, chunk_rows=16, log=print):
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        normalization = get_checkpoint_normalization(checkpoint)
        metadata = get_checkpoint_io_metadata(checkpoint)
        if normalization is None or metadata is None:
            raise RuntimeError("checkpoint must contain normalization_stats and io_metadata")
        model = factory(Cin=int(metadata["Cin"]), Cout=int(metadata["Cout"]), device="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        self.levels = int(metadata["Cout"])
        self.epoch = checkpoint.get("epoch")
        self.mean_x = torch.as_tensor(normalization["mean_X"], dtype=torch.float32)[:, None, None, None]
        self.std_x = torch.as_tensor(normalization["std_X"], dtype=torch.float32)[:, None, None, None]
        self.mean_y = torch.as_tensor(normalization["mean_Y"], dtype=torch.float32)[:, None, None, None]
        self.std_y = torch.as_tensor(normalization["std_Y"], dtype=torch.float32)[:, None, None, None]
        if devices is None:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not devices:
            raise RuntimeError("no GPU is visible")
        self.devices = list(devices)
        self.chunk_depth = int(chunk_depth)
        self.chunk_rows = int(chunk_rows)
        self.weights = None                    # depth_weights of the cube being predicted
        self.pool = _GPUPool(model, self.devices)
        self.width = model.lift[0].out_channels
        self.n_blocks = len(model.blocks)
        self.log = log

    # ------------------------------------------------ streaming pieces

    def _map_depth(self, out, source, layer, post=None):
        """`out[:, a:b] = layer(replica)(source[:, a:b])` for a layer that acts on each point separately."""
        def work(replica, device, a, b):
            y = layer(replica)(source[:, a:b].to(device)[None])
            if post is not None:
                y = post(y)
            out[:, a:b] = y[0].cpu()
        self.pool.run(source.shape[1], self.chunk_depth, work)
        return out

    def _groupnorm(self, values, norm, post=None):
        """`norm(values)` in place, with `norm`'s statistics over all of `values` (C, D, H, W): as `DistributedGroupNorm`,
        with the sums accumulated in double precision. `post` (the replica, the tensor) follows on the same batches. The
        statistics weight each level by `self.weights` (depth_weights, (D, H, W)), as DepthWeightedGroupNorm does."""
        weights = self.weights
        channels, depth = values.shape[:2]
        groups = norm(next(iter(self.pool.replicas.values()))).num_groups
        per_group = channels // groups
        totals = torch.zeros(groups, dtype=torch.float64)
        squares = torch.zeros(groups, dtype=torch.float64)
        lock = threading.Lock()

        def accumulate(replica, device, a, b):
            v = values[:, a:b].to(device).double().reshape(groups, per_group, b - a, *values.shape[2:])
            wv = v * weights[a:b].to(device).double()
            total = wv.sum(dim=(1, 2, 3, 4)).cpu()
            square = (wv * v).sum(dim=(1, 2, 3, 4)).cpu()
            with lock:
                totals.add_(total)
                squares.add_(square)

        self.pool.run(depth, self.chunk_depth, accumulate)
        count = float(per_group * weights.double().sum())
        mean = totals / count
        variance = (squares / count - mean * mean).clamp_min(0.0)

        def apply(replica, device, a, b):
            module = norm(replica)
            shape = (groups, per_group, b - a, *values.shape[2:])
            view = (groups, 1, 1, 1, 1)
            m = mean.float().view(view).to(device)
            inverse = torch.rsqrt(variance.float() + module.eps).view(view).to(device)
            y = ((values[:, a:b].to(device).reshape(shape) - m) * inverse).reshape(channels, b - a, *values.shape[2:])
            if module.affine:
                y = y * module.weight.view(-1, 1, 1, 1) + module.bias.view(-1, 1, 1, 1)
            if post is not None:
                y = post(replica, y)
            values[:, a:b] = y.cpu()

        self.pool.run(depth, self.chunk_depth, apply)
        return values

    def _new_hidden(self, shape):
        return torch.empty(self.width, *shape, dtype=torch.float32)

    # ------------------------------------------------ one block

    def _block(self, x, z, dx, dy, index):
        shape = x.shape[1:]
        depth, rows, _ = shape
        block = lambda replica: replica.blocks[index]
        act = lambda replica, y: block(replica).act(y)

        def spectral(replica, device, a, b):
            spec[:, a:b] = spectral_by_frequency_rows(block(replica).spec, x[:, a:b].to(device)[None], dx, dy)[0].cpu()

        spec = self._new_hidden(shape)
        self.pool.run(depth, self.chunk_depth, spectral)
        self._groupnorm(spec, lambda replica: block(replica).norm_spec, act)

        def vertical_columns(replica, device, a, b):
            y = block(replica).vertical(x[:, :, a:b].to(device)[None], z[:, a:b].to(device)[None])
            vert[:, :, a:b] = y[0].cpu()

        vert = self._new_hidden(shape)
        self.pool.run(rows, self.chunk_rows, vertical_columns)
        self._groupnorm(vert, lambda replica: block(replica).norm_vert, act)

        def fusion(replica, device, a, b):
            y = chunked_fuse_branches(block(replica), x[:, a:b].to(device)[None], spec[:, a:b].to(device)[None],
                                      vert[:, a:b].to(device)[None], chunk_depth=b - a, normalize=False)
            fused[:, a:b] = y[0].cpu()

        fused = self._new_hidden(shape)
        self.pool.run(depth, self.chunk_depth, fusion)
        del spec, vert
        self._groupnorm(fused, lambda replica: block(replica).norm_fuse, act)
        template = next(iter(self.pool.replicas.values())).blocks[index]
        x.add_(fused, alpha=float(template.res_fused.item()))
        del fused

        pointwise = self._map_depth(self._new_hidden(shape), x, lambda replica: block(replica).pw)
        self._groupnorm(pointwise, lambda replica: block(replica).norm_pw)
        x.add_(pointwise, alpha=float(template.res_pw.item()))
        del pointwise

        mlp = self._map_depth(self._new_hidden(shape), x, lambda replica: block(replica).mlp)
        self._groupnorm(mlp, lambda replica: block(replica).norm_mlp)
        x.add_(mlp, alpha=float(template.res_mlp.item()))
        del mlp
        return x

    # ------------------------------------------------ whole model

    def predict_log10(self, features, z_scale, dx, dy):
        """log10 populations (levels, D, H, W) from features in the training's units: (6, D, H, W) with log10 T, velocities
        in km/s, log10 ne, log10 rho, and z in Mm, (D, H, W)."""
        started = time.time()
        features = torch.as_tensor(np.asarray(features, dtype=np.float32))
        z = torch.as_tensor(np.asarray(z_scale, dtype=np.float32))
        if features.ndim != 4 or features.shape[0] != 6 or features.shape[1:] != z.shape:
            raise ValueError(f"expected features (6,D,H,W) and z (D,H,W), got {tuple(features.shape)}, {tuple(z.shape)}")
        if not (torch.isfinite(features).all() and torch.isfinite(z).all()):
            raise ValueError("FFNO request contains NaN or Inf")
        dx, dy = float(dx), float(dy)
        depth, rows, columns = z.shape
        shape = (depth, rows, columns)
        self.pool.splits = 0
        normalized = (features - self.mean_x) / self.std_x
        self.weights = depth_weights(z[None, None])[0, 0]

        x = self._new_hidden(shape)
        self._map_depth(x, normalized, lambda replica: replica.lift[0])
        inputs = normalized                    # the level-fraction head takes rho, T and ne from them
        self._groupnorm(x, lambda replica: replica.lift[1], lambda replica, y: replica.lift[2](y))
        self.log(f"BATCHED lift done {time.time() - started:.0f} s")
        for index in range(self.n_blocks):
            self._block(x, z, dx, dy, index)
            self.log(f"BATCHED block {index + 1}/{self.n_blocks} done {time.time() - started:.0f} s")

        output = torch.empty(self.levels, *shape, dtype=torch.float32)

        def head(replica, device, a, b):
            y = replica.proj2(replica.act(replica.proj1(x[:, a:b].to(device)[None])))
            y = replica.head_output(y, inputs[:, a:b].to(device)[None])[0]
            output[:, a:b] = (y * self.std_y.to(device) + self.mean_y.to(device)).cpu()

        self.pool.run(depth, self.chunk_depth, head)
        self.log(f"BATCHED prediction {rows}x{columns}x{depth}: {time.time() - started:.0f} s on {len(self.devices)} GPUs, "
                 f"{self.pool.splits} batches split")
        return output

    def predict_si(self, features, z_m, dx, dy):
        """Linear populations [m^-3], (D, H, W, levels), from the service's SI features: (T, vx, vy, vz, log10 ne, log10 rho)
        with T in K, velocities in m/s, and z in m."""
        features = np.array(features, dtype=np.float32)
        if not (features[0] > 0).all():
            raise ValueError("FFNO request temperature must be positive")
        model_units = torch.as_tensor(features)
        model_units[0] = torch.log10(model_units[0])
        model_units[1:4] /= VELOCITY_M_S_PER_MODEL_UNIT
        z = torch.as_tensor(np.asarray(z_m, dtype=np.float32)) / Z_METRES_PER_MODEL_UNIT
        log10_populations = self.predict_log10(model_units, z, dx, dy)
        limited = int((log10_populations.abs() > LOG10_POPULATION_LIMIT).sum())
        if limited:
            self.log(f"BATCHED {limited} predicted populations beyond 10^+-{LOG10_POPULATION_LIMIT:g} limited to it")
        populations = linear_populations(log10_populations)
        if not torch.isfinite(populations).all() or (populations <= 0).any():
            raise RuntimeError("FFNO returned non-positive or non-finite populations")
        return populations.permute(1, 2, 3, 0).numpy()

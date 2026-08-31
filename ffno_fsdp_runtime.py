"""FSDP plus distributed-H-slab FFNO inference and input VJPs.

Each torchrun rank owns one spatial H slab. FSDP shards parameters, while the
distributed inference adapters preserve the global 2D Fourier operator and
provide differentiable collectives. FSDP alone would replicate the dominant
full-volume activations and is therefore not the inversion execution model.
"""

from functools import partial
import hashlib
import importlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from distributed_inference import enable_distributed_inference
from models.ffno_model import FFNOBlock3dBalanced
from train_utils import (
    get_checkpoint_io_metadata,
    get_checkpoint_normalization,
    load_checkpoint,
)


INPUT_CHANNELS = ("temperature", "vx", "vy", "vz", "log10_ne", "log10_rho")


class DistributedFSDPBackend:
    """One persistent FFNO model collectively owned by all torchrun ranks."""

    def __init__(
        self,
        checkpoint_path,
        factory_module,
        factory_name,
        level_names,
        factory_kwargs=None,
        require_multi_gpu=True,
    ):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "DistributedFSDPBackend requires initialized torch.distributed"
            )
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if require_multi_gpu and self.world_size < 2:
            raise RuntimeError("FSDP FFNO inversion requires at least two GPU processes")
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP FFNO inversion requires CUDA")
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.checkpoint_loaded_locally = self.rank == 0
        # Only torchrun rank 0 reads the full checkpoint.  FSDP's
        # sync_module_states broadcast initializes the other ranks while the
        # constructor immediately shards each wrapped unit.  Loading the
        # checkpoint independently on every GPU would defeat the purpose of
        # the multi-GPU production path during startup.
        checkpoint = (
            load_checkpoint(self.checkpoint_path, map_location="cpu")
            if self.rank == 0
            else None
        )
        startup = [
            {
                "normalization": get_checkpoint_normalization(checkpoint),
                "io_metadata": get_checkpoint_io_metadata(checkpoint),
                "checkpoint_hash": hashlib.sha256(
                    Path(self.checkpoint_path).read_bytes()
                ).hexdigest(),
            }
            if self.rank == 0
            else None
        ]
        dist.broadcast_object_list(startup, src=0, device=self.device)
        normalization = startup[0]["normalization"]
        io_metadata = startup[0]["io_metadata"]
        if normalization is None or io_metadata is None:
            raise RuntimeError("checkpoint must contain normalization_stats and io_metadata")
        if int(io_metadata["Cin"]) != len(INPUT_CHANNELS):
            raise RuntimeError(
                f"checkpoint Cin={io_metadata['Cin']} does not match {INPUT_CHANNELS}"
            )
        self.level_names = tuple(str(value) for value in level_names)
        if int(io_metadata["Cout"]) != len(self.level_names):
            raise RuntimeError("checkpoint Cout differs from configured level_names")

        module = importlib.import_module(factory_module)
        factory = getattr(module, factory_name)
        kwargs = dict(factory_kwargs or {})
        raw_model = factory(
            Cin=int(io_metadata["Cin"]),
            Cout=int(io_metadata["Cout"]),
            device="cpu",
            **kwargs,
        )
        if self.rank == 0:
            raw_model.load_state_dict(checkpoint["model_state"], strict=True)
            del checkpoint
        self.full_parameter_count = sum(
            parameter.numel() for parameter in raw_model.parameters()
        )
        raw_model.eval()
        raw_model = enable_distributed_inference(raw_model)

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={FFNOBlock3dBalanced},
        )
        self.model = FSDP(
            raw_model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=True,
        )
        self.model.eval()
        self.local_parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )

        def normalized(name):
            return torch.as_tensor(
                normalization[name], dtype=torch.float32, device=self.device
            )[None, :, None, None, None]

        self.mean_x = normalized("mean_X")
        self.std_x = normalized("std_X")
        self.mean_y = normalized("mean_Y")
        self.std_y = normalized("std_Y")
        if torch.any(self.std_x <= 0) or torch.any(self.std_y <= 0):
            raise RuntimeError(
                "checkpoint normalization standard deviations must be positive"
            )
        self.checkpoint_hash = startup[0]["checkpoint_hash"]
        self.load_count = 1
        self.call_count = 0
        dist.barrier()

    def describe(self):
        return {
            "backend": "fsdp_distributed_h_slab",
            "input_channels": INPUT_CHANNELS,
            "level_names": self.level_names,
            "checkpoint_hash": self.checkpoint_hash,
            "checkpoint_loaded_locally": self.checkpoint_loaded_locally,
            "device": str(self.device),
            "world_size": self.world_size,
            "fsdp_enabled": isinstance(self.model, FSDP),
            "sharding_strategy": str(self.model.sharding_strategy),
            "full_parameter_count": self.full_parameter_count,
            "local_parameter_count": self.local_parameter_count,
            "output_representation": "linear_population_m3",
        }

    def _inputs(self, features, z_scale, *, requires_grad):
        features = np.asarray(features, dtype=np.float32)
        z_scale = np.asarray(z_scale, dtype=np.float32)
        if features.ndim != 4 or features.shape[0] != len(INPUT_CHANNELS):
            raise ValueError(
                "local features must have shape (6,nz,nx_local,ny)"
            )
        if tuple(features.shape[1:]) != tuple(z_scale.shape):
            raise ValueError(
                "local z_scale must have shape (nz,nx_local,ny)"
            )
        if not np.isfinite(features).all() or not np.isfinite(z_scale).all():
            raise ValueError("FSDP FFNO request contains NaN or Inf")
        x = torch.as_tensor(features, device=self.device)[None]
        z = torch.as_tensor(z_scale, device=self.device)[None]
        x.requires_grad_(requires_grad)
        z.requires_grad_(requires_grad)
        return x, z

    def _forward_linear(self, x, z, dx, dy):
        dx_tensor = torch.as_tensor(
            float(dx), dtype=torch.float32, device=self.device
        )
        dy_tensor = torch.as_tensor(
            float(dy), dtype=torch.float32, device=self.device
        )
        pred_normalized = self.model(
            (x - self.mean_x) / self.std_x,
            z,
            dx_tensor,
            dy_tensor,
        )
        pred_log = pred_normalized * self.std_y + self.mean_y
        return torch.pow(10.0, pred_log)

    @staticmethod
    def _canonical_populations(populations):
        return (
            populations[0]
            .permute(1, 2, 3, 0)
            .float()
            .detach()
            .cpu()
            .numpy()
        )

    def predict_local(self, features, z_scale, dx, dy):
        x, z = self._inputs(features, z_scale, requires_grad=False)
        with torch.no_grad():
            populations = self._forward_linear(x, z, dx, dy)
        if not torch.isfinite(populations).all() or torch.any(populations <= 0):
            raise RuntimeError(
                "FSDP FFNO returned non-positive or non-finite populations"
            )
        torch.cuda.synchronize(self.device)
        self.call_count += 1
        return self._canonical_populations(populations)

    def vjp_local(self, features, z_scale, dx, dy, population_cotangent):
        x, z = self._inputs(features, z_scale, requires_grad=True)
        population_cotangent = np.asarray(
            population_cotangent, dtype=np.float32
        )
        expected = tuple(z_scale.shape) + (len(self.level_names),)
        if tuple(population_cotangent.shape) != expected:
            raise ValueError(
                "local population_cotangent has shape "
                f"{population_cotangent.shape}; expected {expected}"
            )
        if not np.isfinite(population_cotangent).all():
            raise ValueError("FSDP FFNO VJP cotangent contains NaN or Inf")
        populations = self._forward_linear(x, z, dx, dy)
        cotangent = torch.as_tensor(
            population_cotangent, device=self.device
        ).permute(3, 0, 1, 2)[None]
        feature_gradient, z_gradient = torch.autograd.grad(
            populations,
            (x, z),
            grad_outputs=cotangent,
            allow_unused=False,
        )
        if (
            not torch.isfinite(feature_gradient).all()
            or not torch.isfinite(z_gradient).all()
        ):
            raise RuntimeError("FSDP FFNO VJP returned NaN or Inf")
        torch.cuda.synchronize(self.device)
        self.call_count += 1
        return {
            "features": feature_gradient[0].float().detach().cpu().numpy(),
            "z_scale": z_gradient[0].float().detach().cpu().numpy(),
        }

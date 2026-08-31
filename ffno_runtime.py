"""Persistent, file-free-per-evaluation FFNO inference runtime for Julia.

The model factory must return an uninitialized ``torch.nn.Module`` compatible
with ``model(X, z_scale, dx, dy)``. The checkpoint and normalization metadata
are loaded once when this object is constructed.
"""

import hashlib
import importlib
from pathlib import Path

import numpy as np


INPUT_CHANNELS = ("temperature", "vx", "vy", "vz", "log10_ne", "log10_rho")


class PersistentFFNOBackend:
    def __init__(
        self,
        checkpoint_path,
        factory_module,
        factory_name,
        level_names,
        device="cuda",
        factory_kwargs=None,
    ):
        import torch

        from train_utils import get_checkpoint_io_metadata, get_checkpoint_normalization, load_checkpoint

        self.torch = torch
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        requested_device = str(device)
        self.device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        checkpoint = load_checkpoint(self.checkpoint_path, map_location=self.device)
        normalization = get_checkpoint_normalization(checkpoint)
        io_metadata = get_checkpoint_io_metadata(checkpoint)
        if normalization is None or io_metadata is None:
            raise RuntimeError("checkpoint must contain normalization_stats and io_metadata")
        if int(io_metadata["Cin"]) != len(INPUT_CHANNELS):
            raise RuntimeError(f"checkpoint Cin={io_metadata['Cin']} does not match {INPUT_CHANNELS}")
        self.level_names = tuple(str(value) for value in level_names)
        if int(io_metadata["Cout"]) != len(self.level_names):
            raise RuntimeError("checkpoint Cout differs from configured level_names")

        module = importlib.import_module(factory_module)
        factory = getattr(module, factory_name)
        kwargs = dict(factory_kwargs or {})
        self.model = factory(
            Cin=int(io_metadata["Cin"]),
            Cout=int(io_metadata["Cout"]),
            device=self.device,
            **kwargs,
        )
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.to(self.device).eval()
        self.mean_x = torch.as_tensor(normalization["mean_X"], dtype=torch.float32, device=self.device)[None, :, None, None, None]
        self.std_x = torch.as_tensor(normalization["std_X"], dtype=torch.float32, device=self.device)[None, :, None, None, None]
        self.mean_y = torch.as_tensor(normalization["mean_Y"], dtype=torch.float32, device=self.device)[None, :, None, None, None]
        self.std_y = torch.as_tensor(normalization["std_Y"], dtype=torch.float32, device=self.device)[None, :, None, None, None]
        if torch.any(self.std_x <= 0) or torch.any(self.std_y <= 0):
            raise RuntimeError("checkpoint normalization standard deviations must be positive")
        self.checkpoint_hash = hashlib.sha256(Path(self.checkpoint_path).read_bytes()).hexdigest()
        self.load_count = 1
        self.call_count = 0

    def describe(self):
        return {
            "input_channels": INPUT_CHANNELS,
            "level_names": self.level_names,
            "checkpoint_hash": self.checkpoint_hash,
            "device": self.device,
            "output_representation": "linear_population_m3",
        }

    def predict(self, features, z_scale, dx, dy):
        torch = self.torch
        features = np.asarray(features, dtype=np.float32)
        z_scale = np.asarray(z_scale, dtype=np.float32)
        if features.ndim != 4 or features.shape[0] != len(INPUT_CHANNELS):
            raise ValueError("features must have shape (6,nz,nx,ny)")
        if tuple(features.shape[1:]) != tuple(z_scale.shape):
            raise ValueError("z_scale must have shape (nz,nx,ny)")
        if not np.isfinite(features).all() or not np.isfinite(z_scale).all():
            raise ValueError("FFNO request contains NaN or Inf")
        x = torch.as_tensor(features,device=self.device)[None]
        z = torch.as_tensor(z_scale,device=self.device)[None]
        dx_tensor = torch.as_tensor(float(dx),dtype=torch.float32,device=self.device)
        dy_tensor = torch.as_tensor(float(dy),dtype=torch.float32,device=self.device)
        with torch.no_grad():
            pred_normalized = self.model((x-self.mean_x)/self.std_x,z,dx_tensor,dy_tensor)
            pred_log = pred_normalized*self.std_y+self.mean_y
            populations = torch.pow(10.0,pred_log)
        if not torch.isfinite(populations).all() or torch.any(populations <= 0):
            raise RuntimeError("FFNO returned non-positive or non-finite populations")
        self.call_count += 1
        # [1,Cout,nz,nx,ny] -> canonical Julia [nz,nx,ny,Cout]
        return populations[0].permute(1,2,3,0).float().cpu().numpy()

    def vjp(self, features, z_scale, dx, dy, population_cotangent):
        """Apply the exact PyTorch VJP of linear populations.

        The public arrays retain the Julia canonical layouts; only the
        persistent GPU backend performs the channel-first/batch transforms.
        No parameter gradients or dense population Jacobian are constructed.
        """
        torch = self.torch
        features = np.asarray(features, dtype=np.float32)
        z_scale = np.asarray(z_scale, dtype=np.float32)
        population_cotangent = np.asarray(population_cotangent, dtype=np.float32)
        if features.ndim != 4 or features.shape[0] != len(INPUT_CHANNELS):
            raise ValueError("features must have shape (6,nz,nx,ny)")
        if tuple(features.shape[1:]) != tuple(z_scale.shape):
            raise ValueError("z_scale must have shape (nz,nx,ny)")
        expected = tuple(z_scale.shape) + (len(self.level_names),)
        if tuple(population_cotangent.shape) != expected:
            raise ValueError(
                f"population_cotangent has shape {population_cotangent.shape}; expected {expected}"
            )
        if not (
            np.isfinite(features).all()
            and np.isfinite(z_scale).all()
            and np.isfinite(population_cotangent).all()
        ):
            raise ValueError("FFNO VJP request contains NaN or Inf")

        x = torch.as_tensor(features, device=self.device)[None].requires_grad_(True)
        z = torch.as_tensor(z_scale, device=self.device)[None].requires_grad_(True)
        dx_tensor = torch.as_tensor(float(dx), dtype=torch.float32, device=self.device)
        dy_tensor = torch.as_tensor(float(dy), dtype=torch.float32, device=self.device)
        pred_normalized = self.model(
            (x - self.mean_x) / self.std_x, z, dx_tensor, dy_tensor
        )
        pred_log = pred_normalized * self.std_y + self.mean_y
        populations = torch.pow(10.0, pred_log)
        cotangent = torch.as_tensor(population_cotangent, device=self.device)
        cotangent = cotangent.permute(3, 0, 1, 2)[None]
        feature_gradient, z_gradient = torch.autograd.grad(
            populations,
            (x, z),
            grad_outputs=cotangent,
            allow_unused=True,
        )
        if feature_gradient is None:
            feature_gradient = torch.zeros_like(x)
        if z_gradient is None:
            z_gradient = torch.zeros_like(z)
        if not torch.isfinite(feature_gradient).all() or not torch.isfinite(z_gradient).all():
            raise RuntimeError("FFNO VJP returned NaN or Inf")
        self.call_count += 1
        return {
            "features": feature_gradient[0].float().detach().cpu().numpy(),
            "z_scale": z_gradient[0].float().detach().cpu().numpy(),
        }


def create_persistent_backend(**kwargs):
    return PersistentFFNOBackend(**kwargs)

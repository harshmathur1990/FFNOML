import h5py
import torch
from torch.utils.data import Dataset

# ============================================================
# ---------------- DATASET / DATALOADER -----------------------
# ============================================================

class H5PatchDataset(Dataset):
    """
    Lazy HDF5 patch dataset.

    Safe with multiple workers: each worker opens its own file handle.
    """

    def __init__(self, h5_path):
        self.h5_path = h5_path

        self._f = None
        self._X = None
        self._Y = None
        self._dx = None
        self._dy = None
        self._scale = None
        self._weights = None
        self._S = None

        with h5py.File(self.h5_path, "r") as f:
            self.N = int(f["inputs"].shape[0])

            # check optional datasets
            self.has_scale = "scale" in f
            self.has_weights = "weights" in f
            self.has_source_targets = "source_targets" in f

    def _ensure_open(self):

        if self._f is None:

            self._f = h5py.File(self.h5_path, "r")

            self._X = self._f["inputs"]
            self._Y = self._f["targets"]

            self._dx = self._f["dx"]
            self._dy = self._f["dy"]

            if self.has_scale:
                self._scale = self._f["scale"]

            if self.has_weights:
                self._weights = self._f["weights"]

            if self.has_source_targets:
                self._S = self._f["source_targets"]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        self._ensure_open()

        x = torch.from_numpy(self._X[idx])   # [Cin,D,P,P]
        y = torch.from_numpy(self._Y[idx])   # [Cout,D,P,P]

        dx = torch.tensor(self._dx[idx], dtype=torch.float32)
        dy = torch.tensor(self._dy[idx], dtype=torch.float32)

        scale = None
        weight = None

        if self.has_scale:
            scale = torch.tensor(self._scale[idx], dtype=torch.int32)

        if self.has_weights:
            weight = torch.tensor(self._weights[idx], dtype=torch.float32)

        source_target = torch.empty(0, dtype=torch.float32)
        if self.has_source_targets:
            source_target = torch.from_numpy(self._S[idx])

        return x, y, dx, dy, scale, weight, source_target


class H5CubeDataset(Dataset):
    """
    Dataset for full-cube training or evaluation.

    Safe with multiple workers: each worker opens its own file handle.
    """

    def __init__(self, h5_path):

        self.h5_path = h5_path

        self._f = None
        self._X = None
        self._Y = None
        self._dx = None
        self._dy = None
        self._weights = None
        self._scale = None
        self._S = None

        with h5py.File(self.h5_path, "r") as f:

            self.N = int(f["inputs"].shape[0])

            self.has_targets = "targets" in f
            self.has_weights = "weights" in f
            self.has_scale = "scale" in f
            self.has_source_targets = "source_targets" in f

    def _ensure_open(self):

        if self._f is None:

            self._f = h5py.File(self.h5_path, "r")

            self._X = self._f["inputs"]

            if self.has_targets:
                self._Y = self._f["targets"]

            self._dx = self._f["dx"]
            self._dy = self._f["dy"]

            if self.has_weights:
                self._weights = self._f["weights"]

            if self.has_scale:
                self._scale = self._f["scale"]

            if self.has_source_targets:
                self._S = self._f["source_targets"]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        self._ensure_open()

        x = torch.from_numpy(self._X[idx])   # [Cin,D,nx,ny]

        y = None
        if self.has_targets:
            y = torch.from_numpy(self._Y[idx])

        dx = torch.tensor(self._dx[idx], dtype=torch.float32)
        dy = torch.tensor(self._dy[idx], dtype=torch.float32)

        scale = None
        weight = None

        if self.has_scale:
            scale = torch.tensor(self._scale[idx], dtype=torch.int32)

        if self.has_weights:
            weight = torch.tensor(self._weights[idx], dtype=torch.float32)

        source_target = torch.empty(0, dtype=torch.float32)
        if self.has_source_targets:
            source_target = torch.from_numpy(self._S[idx])

        return x, y, dx, dy, scale, weight, source_target

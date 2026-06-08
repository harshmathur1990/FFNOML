import bisect

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _group_names_with_inputs(h5_file):
    if "patch_dataset_names" in h5_file:
        return [str(name) for name in h5_file["patch_dataset_names"].asstr()[...]]

    names = []
    for name, obj in h5_file.items():
        if isinstance(obj, h5py.Group) and "inputs" in obj:
            names.append(name)
    return sorted(names)


def _select_indices_by_weight(weights, fraction, rng):
    weights = np.asarray(weights, dtype=np.float32)
    selected_indices = []
    selected_weights = []

    for weight_value in np.unique(weights):
        class_indices = np.flatnonzero(weights == weight_value)
        if class_indices.size == 1:
            chosen = class_indices
        else:
            n_selected = max(1, int(np.ceil(class_indices.size * fraction)))
            chosen = rng.choice(class_indices, size=n_selected, replace=False)
            chosen.sort()

        total_weight = float(weights[class_indices].sum(dtype=np.float64))
        adjusted_weight = total_weight / float(chosen.size)
        selected_indices.append(chosen.astype(np.int64, copy=False))
        selected_weights.append(
            np.full(chosen.size, adjusted_weight, dtype=np.float32)
        )

    if not selected_indices:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    selected_indices = np.concatenate(selected_indices)
    selected_weights = np.concatenate(selected_weights)
    order = np.argsort(selected_indices)
    return selected_indices[order], selected_weights[order]


class H5PatchDataset(Dataset):
    """
    Lazy HDF5 patch dataset.

    Supports:
    - legacy flat layout with root-level datasets
    - grouped layout with per-simulation datasets preserving native depth
    """

    def __init__(self, h5_path, train_select=1.0, train_select_seed=None):
        self.h5_path = h5_path
        self.train_select = float(train_select)
        if not (0.0 < self.train_select <= 1.0):
            raise ValueError("train_select must be in the interval (0, 1]")
        self.train_select_seed = train_select_seed

        self._f = None
        self._group_names = []
        self._group_lengths = []
        self._group_offsets = []
        self._group_selected_indices = {}
        self._group_selected_weights = {}
        self._selected_indices = None
        self._selected_weights = None
        self._using_train_selection = self.train_select < 1.0
        self.dynamic_length = self._using_train_selection
        self.variable_depth = False

        self._X = None
        self._Y = None
        self._Z = None
        self._dx = None
        self._dy = None
        self._scale = None
        self._weights = None
        self._S = None

        with h5py.File(self.h5_path, "r") as f:
            self._group_names = _group_names_with_inputs(f)
            self.grouped_layout = len(self._group_names) > 0
            if self.grouped_layout:
                self.variable_depth = True
                self._group_lengths = []
                running = 0
                for name in self._group_names:
                    length = int(f[name]["inputs"].shape[0])
                    self._group_lengths.append(length)
                    running += length
                    self._group_offsets.append(running)
                self.N = running
                first = f[self._group_names[0]]
                self.has_scale = "scale" in first
                self.has_weights = "weights" in first
                self.has_source_targets = "source_targets" in first
            else:
                self.N = int(f["inputs"].shape[0])
                self.has_scale = "scale" in f
                self.has_weights = "weights" in f
                self.has_source_targets = "source_targets" in f

        if self._using_train_selection:
            self.resample_train_selection(self.train_select_seed)

    def resample_train_selection(self, seed=None):
        if not self._using_train_selection:
            return

        rng = np.random.default_rng(seed)
        with h5py.File(self.h5_path, "r") as f:
            if self.grouped_layout:
                self._group_lengths = []
                self._group_offsets = []
                self._group_selected_indices = {}
                self._group_selected_weights = {}

                running = 0
                for name in self._group_names:
                    group = f[name]
                    length = int(group["inputs"].shape[0])
                    if "weights" in group:
                        weights = np.asarray(
                            group["weights"][...],
                            dtype=np.float32,
                        )
                    else:
                        weights = np.ones(length, dtype=np.float32)

                    selected, selected_weights = _select_indices_by_weight(
                        weights,
                        self.train_select,
                        rng,
                    )
                    self._group_selected_indices[name] = selected
                    self._group_selected_weights[name] = selected_weights
                    length = int(selected.size)

                    self._group_lengths.append(length)
                    running += length
                    self._group_offsets.append(running)

                self.N = running
                return

            original_n = int(f["inputs"].shape[0])
            if "weights" in f:
                weights = np.asarray(f["weights"][...], dtype=np.float32)
            else:
                weights = np.ones(original_n, dtype=np.float32)

            self._selected_indices, self._selected_weights = _select_indices_by_weight(
                weights,
                self.train_select,
                rng,
            )
            self.N = int(self._selected_indices.size)

    def _ensure_open(self):
        if self._f is not None:
            return

        self._f = h5py.File(self.h5_path, "r")

        if self.grouped_layout:
            return

        self._X = self._f["inputs"]
        self._Y = self._f["targets"]
        self._Z = self._f["z_scale"]
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

    def _locate_group(self, idx):
        group_idx = bisect.bisect_right(self._group_offsets, idx)
        start = 0 if group_idx == 0 else self._group_offsets[group_idx - 1]
        local_idx = idx - start
        group_name = self._group_names[group_idx]
        selected_weight = None
        if self._using_train_selection:
            selected_pos = local_idx
            local_idx = int(self._group_selected_indices[group_name][selected_pos])
            selected_weight = float(
                self._group_selected_weights[group_name][selected_pos]
            )
        return group_name, local_idx, selected_weight

    def __getitem__(self, idx):
        self._ensure_open()

        if self.grouped_layout:
            group_name, local_idx, selected_weight = self._locate_group(idx)
            group = self._f[group_name]

            x = torch.from_numpy(group["inputs"][local_idx])
            y = torch.from_numpy(group["targets"][local_idx])
            z = torch.from_numpy(group["z_scale"][local_idx])
            dx = torch.tensor(group["dx"][local_idx], dtype=torch.float32)
            dy = torch.tensor(group["dy"][local_idx], dtype=torch.float32)

            scale = None
            weight = None
            if "scale" in group:
                scale = torch.tensor(group["scale"][local_idx], dtype=torch.int32)
            if selected_weight is not None:
                weight = torch.tensor(selected_weight, dtype=torch.float32)
            elif "weights" in group:
                weight = torch.tensor(group["weights"][local_idx], dtype=torch.float32)

            source_target = torch.empty(0, dtype=torch.float32)
            if "source_targets" in group:
                source_target = torch.from_numpy(group["source_targets"][local_idx])

            return x, y, z, dx, dy, scale, weight, source_target

        selected_weight = None
        if self._using_train_selection:
            selected_pos = idx
            idx = int(self._selected_indices[selected_pos])
            selected_weight = float(self._selected_weights[selected_pos])

        x = torch.from_numpy(self._X[idx])
        y = torch.from_numpy(self._Y[idx])
        z = torch.from_numpy(self._Z[idx])
        dx = torch.tensor(self._dx[idx], dtype=torch.float32)
        dy = torch.tensor(self._dy[idx], dtype=torch.float32)

        scale = None
        weight = None

        if self.has_scale:
            scale = torch.tensor(self._scale[idx], dtype=torch.int32)

        if selected_weight is not None:
            weight = torch.tensor(selected_weight, dtype=torch.float32)
        elif self.has_weights:
            weight = torch.tensor(self._weights[idx], dtype=torch.float32)

        source_target = torch.empty(0, dtype=torch.float32)
        if self.has_source_targets:
            source_target = torch.from_numpy(self._S[idx])

        return x, y, z, dx, dy, scale, weight, source_target


class H5CubeDataset(Dataset):
    """
    Dataset for full-cube training or evaluation.
    """

    def __init__(self, h5_path):
        self.h5_path = h5_path

        self._f = None
        self._X = None
        self._Y = None
        self._Z = None
        self._dx = None
        self._dy = None
        self._weights = None
        self._scale = None
        self._S = None
        self.variable_depth = False

        with h5py.File(self.h5_path, "r") as f:
            self.N = int(f["inputs"].shape[0])
            self.has_targets = "targets" in f
            self.has_weights = "weights" in f
            self.has_scale = "scale" in f
            self.has_source_targets = "source_targets" in f

    def _ensure_open(self):
        if self._f is not None:
            return

        self._f = h5py.File(self.h5_path, "r")
        self._X = self._f["inputs"]

        if self.has_targets:
            self._Y = self._f["targets"]

        self._Z = self._f["z_scale"]
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

        x = torch.from_numpy(self._X[idx])
        y = None
        if self.has_targets:
            y = torch.from_numpy(self._Y[idx])
        z = torch.from_numpy(self._Z[idx])

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

        return x, y, z, dx, dy, scale, weight, source_target

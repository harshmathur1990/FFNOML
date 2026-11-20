"""ZAugmentDataset: the N stored samples unchanged, then N windows and N uneven selections of their levels, the same in
every tensor; new cuts every training epoch, the same on every rank; validation cuts fixed."""
import numpy as np
import torch

from data.dataset import ZAugmentDataset


class _Columns(torch.utils.data.Dataset):
    def __init__(self, depths):
        self.depths = depths

    def __len__(self):
        return len(self.depths)

    def depth(self, i):
        return self.depths[i]

    def __getitem__(self, i):
        d = self.depths[i]
        level = torch.arange(d, dtype=torch.float32)
        x = level.view(1, d, 1, 1).expand(6, d, 3, 3).clone() + 1000 * i
        y = -x[:5]
        z = level.view(d, 1, 1).expand(d, 3, 3).clone() + 1000 * i
        return x, y, z, torch.tensor(1.0), torch.tensor(1.0), None, torch.tensor(2.0), torch.empty(0)


def _kept(sample):
    return sample[0][0, :, 0, 0].numpy().astype(int) % 1000


def test_stored_then_windows_then_uneven_selections():
    base = _Columns([256, 467, 40])
    data = ZAugmentDataset(base, min_depth=16, random=True, seed=3)
    assert len(data) == 9
    for i in range(3):
        assert torch.equal(data[i][0], base[i][0]) and torch.equal(data[i][2], base[i][2])
    for i in range(3, 9):
        x, y, z, _, _, _, w, lte = data[i]
        kept = _kept(data[i])
        source = (i - 3) % 3
        assert 16 <= len(kept) <= base.depths[source] and np.all(np.diff(kept) > 0)
        assert np.all(x[0, :, 0, 0].numpy() // 1000 == source)
        assert x.shape[1] == y.shape[1] == z.shape[0] and np.array_equal(z[:, 0, 0].numpy().astype(int) % 1000, kept)
        assert torch.equal(y, -x[:5]) and float(w) == 2.0 and lte.numel() == 0
        if i < 6:
            assert np.all(np.diff(kept) == 1)


def test_uneven_selections_are_uneven():
    data = ZAugmentDataset(_Columns([256] * 20), min_depth=16, random=True, seed=1)
    spacings = [set(np.diff(_kept(data[i]))) for i in range(40, 60)]
    assert sum(len(s) > 1 for s in spacings) >= 18


def test_new_cuts_each_epoch_same_for_a_seed():
    a = ZAugmentDataset(_Columns([256, 467]), min_depth=16, random=True)
    b = ZAugmentDataset(_Columns([256, 467]), min_depth=16, random=True)
    a.resample_train_selection(7), b.resample_train_selection(7)
    first = [_kept(a[i]) for i in range(2, 6)]
    assert all(np.array_equal(k, _kept(b[i])) for k, i in zip(first, range(2, 6)))   # every rank alike
    a.resample_train_selection(8)
    assert any(not np.array_equal(k, _kept(a[i])) for k, i in zip(first, range(2, 6)))


def test_validation_cuts_fixed():
    data = ZAugmentDataset(_Columns([256, 467]), min_depth=16, random=False, seed=0)
    assert not hasattr(data._inner_resample, "__call__")
    first = [_kept(data[i]) for i in range(2, 6)]
    assert all(np.array_equal(k, _kept(data[i])) for k, i in zip(first, range(2, 6)))


def test_short_column_kept_whole():
    data = ZAugmentDataset(_Columns([10]), min_depth=16, random=True)
    assert data[1][0].shape[1] == 10 and data[2][0].shape[1] == 10

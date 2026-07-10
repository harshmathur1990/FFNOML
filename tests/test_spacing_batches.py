import unittest
from importlib.util import find_spec

HAS_TORCH = find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from data_builder import ZScaleBucketBatchSampler
    from models.ffno_model import _to_spacing_tensor


class FakeGeometryDataset:
    def __init__(self):
        self.buckets = {
            ("z40",): [0, 1, 2, 3],
            ("z80",): [4],
        }
        self.dynamic_length = False

    def shape_bucket_indices(self):
        return self.buckets


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for spacing batch tests")
class SpacingBatchTests(unittest.TestCase):
    def test_scalar_spacing_expands_to_batch(self):
        x = torch.empty(2, 1)

        spacing = _to_spacing_tensor(1.5, x, 2, "dx")

        self.assertTrue(
            torch.equal(spacing, torch.tensor([1.5, 1.5], dtype=x.dtype))
        )

    def test_mixed_spacing_is_preserved_per_sample(self):
        x = torch.empty(2, 1)

        spacing = _to_spacing_tensor(torch.tensor([1.0, 2.0]), x, 2, "dx")

        self.assertTrue(torch.equal(spacing, torch.tensor([1.0, 2.0])))

    def test_z_scale_sampler_keeps_shape_buckets_separate(self):
        sampler = ZScaleBucketBatchSampler(
            FakeGeometryDataset(),
            batch_size=2,
            shuffle=False,
        )

        batches = list(sampler)

        self.assertEqual(batches, [[0, 1], [2, 3], [4]])


if __name__ == "__main__":
    unittest.main()

"""ffno_batched_backend.py must equal FFNO3D.forward: on the CPU, on a small random model, whatever the batches."""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffno"))
from ffno_batched_backend import BatchedFFNOBackend
from models.ffno_model import FFNO3D

CONFIG = dict(in_channels=6, out_channels=6, width=16, n_layers=2, spec_dropout=0.08, vertical_dropout=0.08,
              spec_dropout_layers=[0, 1], vertical_dropout_layers=[0, 1], hidden_spectral=16, hidden_vertical=16)


class BatchedBackendEqualsForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = FFNO3D(**CONFIG).eval()
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "model.pt")
        zeros, ones = np.zeros(6, np.float32), np.ones(6, np.float32)
        torch.save(dict(model_state=self.model.state_dict(), epoch=1, io_metadata=dict(Cin=6, Cout=6),
                        normalization_stats=dict(mean_X=zeros, std_X=ones, mean_Y=zeros, std_Y=ones)), self.path)
        self.x = torch.randn(6, 24, 11, 9)
        self.z = torch.linspace(14.0, -1.0, 24)[:, None, None].repeat(1, 11, 9) + 0.01 * torch.randn(24, 11, 9)

    def tearDown(self):
        self.directory.cleanup()

    def predict(self, **options):
        backend = BatchedFFNOBackend(self.path, lambda Cin, Cout, device: FFNO3D(**CONFIG), devices=["cpu"],
                                     log=lambda message: None, **options)
        return backend.predict_log10(self.x.numpy(), self.z.numpy(), 48e3, 48e3)

    def test_equals_forward_for_any_batches(self):
        with torch.no_grad():
            expected = self.model(self.x[None], self.z[None], 48e3, 48e3)[0]
        for options in (dict(chunk_depth=24, chunk_rows=11, vertical_chunk=99), dict(chunk_depth=1, chunk_rows=1, vertical_chunk=1),
                        dict(chunk_depth=5, chunk_rows=3, vertical_chunk=7)):
            with self.subTest(**options):
                self.assertLess((self.predict(**options) - expected).abs().max().item(), 2e-4)

    def test_slabs_change_only_the_vertical_statistics(self):
        one, four = self.predict(vertical_norm_slabs=1), self.predict(vertical_norm_slabs=4)
        self.assertGreater((one - four).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()

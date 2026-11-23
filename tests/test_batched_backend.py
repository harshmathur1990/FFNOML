"""ffno_batched_backend.py must equal FFNO3D.forward: on the CPU, on a small random model, whatever the batches."""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffno_batched_backend import BatchedFFNOBackend
from config import ATOM_CONFIG
from models.ffno_model import FFNO3D

H = {k: ATOM_CONFIG["H"][k] for k in ("levels", "log10_particle_mass", "multi3d_lte")}
CONFIG = dict(in_channels=6, out_channels=6, width=16, n_layers=2, spec_dropout=0.08, vertical_dropout=0.08,
              spec_dropout_layers=[0, 1], vertical_dropout_layers=[0, 1], hidden_spectral=16, hidden_vertical=16,
              z_min_wavelength=0.3, freq_mix_rank=4, species=[H])
# the head's T, ne and rho on solar scales; the outputs stay normalised to themselves (mean 0, std 1)
MEAN_X = np.array([3.8, 0.0, 0.0, 0.0, 16.0, -7.0], np.float32)
STD_X = np.array([0.4, 5.0, 5.0, 5.0, 2.0, 2.5], np.float32)


class BatchedBackendEqualsForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = FFNO3D(**CONFIG).eval()
        self.model.set_normalization(MEAN_X, STD_X, np.zeros(6), np.ones(6))
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if ".freq_u." in name:
                    p.normal_(std=0.1)                       # the low-rank mixing, zero at initialisation
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
        for options in (dict(chunk_depth=24, chunk_rows=11), dict(chunk_depth=1, chunk_rows=1),
                        dict(chunk_depth=5, chunk_rows=3)):
            with self.subTest(**options):
                self.assertLess((self.predict(**options) - expected).abs().max().item(), 2e-4)

    def test_populations_beyond_float32_are_limited(self):
        messages = []
        backend = BatchedFFNOBackend(self.path, lambda Cin, Cout, device: FFNO3D(**CONFIG), devices=["cpu"], log=messages.append)
        log10 = torch.full((6, 24, 11, 9), 10.0)
        log10[4, 3, 2, 1], log10[5, 3, 2, 1], log10[0, 4, 2, 1] = 44.9, 41.9, -50.0
        backend.predict_log10 = lambda *arguments: log10
        features = self.x.numpy().copy()
        features[0] = 1e4
        populations = backend.predict_si(features, self.z.numpy() * 1e6, 48e3, 48e3)
        self.assertTrue(np.isfinite(populations).all() and (populations > 0).all())
        self.assertAlmostEqual(np.log10(populations[3, 2, 1, 4]), 37.0, places=4)
        self.assertAlmostEqual(np.log10(populations[4, 2, 1, 0]), -37.0, places=4)
        self.assertAlmostEqual(np.log10(populations[0, 0, 0, 0]), 10.0, places=4)
        self.assertTrue(any("3 predicted populations" in message for message in messages))



if __name__ == "__main__":
    unittest.main()

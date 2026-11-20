"""z_spectral_modes with z_reference_length > 0 (SpectralConvHeight): a convolution along height, the same on any vertical
grid and for any number of levels; as a switch of FFNO3D it starts as a no-op, learns, and the batched backend still
equals forward."""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffno_batched_backend import BatchedFFNOBackend  # noqa: E402
from models.ffno_model import FFNO3D, SpectralConvHeight  # noqa: E402
from test_capacity_switches import CONFIG, inputs, output  # noqa: E402

HEIGHT = dict(z_spectral_modes=8, z_reference_length=40.0)


def profile(z):
    """Smooth channels on heights z (Mm) [D] -> [1, 2, D]."""
    return torch.stack([torch.exp(-((z - 4.0) / 2.5) ** 2), torch.tanh(z / 3.0)])[None].double()


def stretched(depth, bottom=-1.0, top=14.0):
    """Fine low down, coarse higher up, top first like the simulation grids."""
    s = torch.linspace(0.0, 1.0, depth, dtype=torch.float64)
    return (bottom + (top - bottom) * (0.4 * s + 0.6 * s ** 2)).flip(0)


class HeightConvolution(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.conv = SpectralConvHeight(2, 12, 40.0).double()
        with torch.no_grad():
            self.conv.weight_real.normal_(std=0.3)
            self.conv.weight_imag.normal_(std=0.3)

    def apply(self, z):
        with torch.no_grad():
            return self.conv(profile(z), z[None, None])[0]

    def test_same_on_any_grid_and_number_of_levels(self):
        reference_z = torch.linspace(14.0, -1.0, 3001, dtype=torch.float64)
        reference = self.apply(reference_z)
        scale = reference.abs().max().item()
        for z in (torch.linspace(14.0, -1.0, 400, dtype=torch.float64), stretched(250), stretched(497)):
            with self.subTest(levels=len(z)):
                got = self.apply(z)
                want = torch.stack([torch.from_numpy(np.interp(z.numpy(), reference_z.flip(0).numpy(),
                                                               c.flip(0).numpy())) for c in reference])
                self.assertLess((got - want).abs().max().item() / scale, 2e-3)

    def test_level_order_and_height_offset_do_not_matter(self):
        z = stretched(300)
        got = self.apply(z)
        self.assertLess((self.apply(z.flip(0)).flip(-1) - got).abs().max().item(), 1e-10)
        with torch.no_grad():
            shifted = self.conv(profile(z), (z + 3.0)[None, None])[0]
        self.assertLess((shifted - got).abs().max().item(), 1e-10)

    def test_span_beyond_half_the_reference_length_is_refused(self):
        model = FFNO3D(**CONFIG, z_spectral_modes=8, z_reference_length=20.0)
        x, z = inputs()                                     # spans 15 Mm
        with self.assertRaisesRegex(ValueError, "more than half"):
            output(model, x, z)


class HeightSwitch(unittest.TestCase):
    def test_starts_as_no_op(self):
        torch.manual_seed(5)
        off = FFNO3D(**CONFIG)
        on = FFNO3D(**CONFIG, **HEIGHT)
        missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing and all(".vertical.z_" in k for k in missing), missing)
        x, z = inputs()
        self.assertLess((output(on, x, z) - output(off, x, z)).abs().max().item(), 1e-5)

    def test_weights_get_gradients(self):
        torch.manual_seed(5)
        model = FFNO3D(**CONFIG, **HEIGHT).train()
        x, z = inputs()
        model(x, z, 48e3, 48e3).square().mean().backward()
        names = [n for n in model.state_dict() if ".z_spec." in n]
        self.assertTrue(names)
        for name, p in model.named_parameters():
            if ".z_spec." in name:
                with self.subTest(parameter=name):
                    self.assertGreater(p.grad.abs().max().item(), 0.0)

    def test_checkpoint_holds_only_the_weights(self):
        names = [n for n in FFNO3D(**CONFIG, **HEIGHT).state_dict() if ".z_spec." in n]
        self.assertTrue(names and all(n.endswith(("weight_real", "weight_imag")) for n in names), names)

    def test_batched_backend_equals_forward(self):
        torch.manual_seed(3)
        model = FFNO3D(**CONFIG, **HEIGHT).eval()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".z_spec." in name:
                    p.normal_(std=0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            zeros, ones = np.zeros(6, np.float32), np.ones(6, np.float32)
            torch.save(dict(model_state=model.state_dict(), epoch=1, io_metadata=dict(Cin=6, Cout=6),
                            normalization_stats=dict(mean_X=zeros, std_X=ones, mean_Y=zeros, std_Y=ones)), path)
            x, z = inputs()
            with torch.no_grad():
                expected = model(x, z, 48e3, 48e3)[0]
            backend = BatchedFFNOBackend(path, lambda Cin, Cout, device: FFNO3D(**CONFIG, **HEIGHT),
                                         devices=["cpu"], log=lambda message: None, chunk_depth=5, chunk_rows=3,
                                         vertical_chunk=7)
            got = backend.predict_log10(x[0].numpy(), z[0].numpy(), 48e3, 48e3)
            self.assertLess((got - expected).abs().max().item(), 2e-4)


if __name__ == "__main__":
    unittest.main()

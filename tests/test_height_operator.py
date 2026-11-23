"""FFNO3D, a neural operator in x, y and z: the vertical branch is HeightSpectralConv, every GroupNorm weights levels by
height and nothing counts grid points, so the same atmosphere sampled on different vertical grids gives the same
populations at the same heights. The batched backend and the FSDP route equal forward. Columns taller than half of
z_reference_length take a longer periodic length. The final model's checkpoints load into config.py's defaults."""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from config import ATOM_CONFIG  # noqa: E402
from distributed_inference import ChunkedHeightSpectralConv, enable_distributed_inference  # noqa: E402
from ffno_batched_backend import BatchedFFNOBackend  # noqa: E402
from models.ffno_model import (FFNO3D, DepthWeightedGroupNorm, HeightSpectralConv, height_fourier_weights,  # noqa: E402
                               trapezoid_weights)

H = {k: ATOM_CONFIG["H"][k] for k in ("levels", "log10_particle_mass", "multi3d_lte")}
CONFIG = dict(in_channels=6, out_channels=6, width=16, n_layers=2, hidden_spectral=16, hidden_vertical=16,
              spec_dropout=0.0, vertical_dropout=0.0, dropout=0.0, species=[H])
OPERATOR = dict(z_reference_length=40.0, z_min_wavelength=0.3, z_feature_wavelength=0.3, freq_mix_rank=4)
# the head's T, ne and rho on solar scales; the outputs stay normalised to themselves (mean 0, std 1)
MEAN_X = np.array([3.8, 0.0, 0.0, 0.0, 16.0, -7.0], np.float32)
STD_X = np.array([0.4, 5.0, 5.0, 5.0, 2.0, 2.5], np.float32)
FINAL = "/mn/stornext/d9/data/harshm/ffnoml_data/models/zaug"


def model_of(**overrides):
    model = FFNO3D(**{**CONFIG, **OPERATOR, **overrides})
    model.set_normalization(MEAN_X, STD_X, np.zeros(6), np.ones(6))
    return model


def uniform(depth):
    return torch.linspace(14.0, -1.0, depth, dtype=torch.float64)


def stretched(depth):
    """Fine low down, coarse higher up, top first like the simulation grids."""
    s = torch.linspace(0.0, 1.0, depth, dtype=torch.float64)
    return (-1.0 + 15.0 * (0.4 * s + 0.6 * s ** 2)).flip(0)


def atmosphere(z, rows=6, columns=5):
    """Six smooth input channels on heights z [D], varying in x and y: [1, 6, D, H, W] and z_scale [1, D, H, W]."""
    y = torch.linspace(0, 2 * np.pi, rows + 1, dtype=torch.float64)[:-1]
    x = torch.linspace(0, 2 * np.pi, columns + 1, dtype=torch.float64)[:-1]
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    channels = [torch.tanh((zz - 2.0) / 1.5) + 0.3 * torch.sin(xx) * torch.cos(yy),
                0.5 * torch.sin(zz / 2.0 + xx), 0.5 * torch.cos(zz / 3.0 + yy),
                torch.exp(-((zz - 5.0) / 3.0) ** 2) * torch.cos(xx - yy),
                -zz / 7.0 + 0.2 * torch.sin(yy), -zz / 5.0 + 0.1 * torch.cos(xx)]
    return torch.stack(channels)[None].float(), zz[None].float()


def at_heights(values, z_from, z_to):
    """values [..., D] on heights z_from, linearly interpolated to z_to."""
    order = torch.argsort(z_from)
    flat = values.reshape(-1, values.shape[-1])[:, order].double().numpy()
    return torch.from_numpy(np.stack([np.interp(z_to.numpy(), z_from[order].numpy(), row) for row in flat])) \
        .reshape(*values.shape[:-1], len(z_to))


def populations(model, z):
    x, zs = atmosphere(z)
    with torch.no_grad():
        y = model.eval()(x, zs, 48e3, 48e3)
    return y.permute(0, 1, 3, 4, 2)          # depth last


def grid_difference(model):
    """The largest difference, over the output's range, between the model on a 3001-level grid and on three others."""
    reference_z = uniform(3001)
    reference = populations(model, reference_z)
    scale = (reference.max() - reference.min()).item()
    worst = 0.0
    for z in (uniform(400), stretched(250), stretched(497)):
        got = populations(model, z).double()
        worst = max(worst, (got - at_heights(reference, reference_z, z)).abs().max().item() / scale)
    return worst


class PhysicalWavenumbers(unittest.TestCase):
    """W(k) a function of the physical wavenumber whatever the band limit (z_feature_wavelength), and in each column
    only the wavenumbers its grid resolves."""

    def test_features_do_not_depend_on_the_band_limit(self):
        coarse = HeightSpectralConv(8, 40.0, 0.3, 0.3, hidden=8)
        fine = HeightSpectralConv(8, 40.0, 0.04, 0.3, hidden=8)
        self.assertTrue(torch.allclose(coarse.k_features(), fine.k_features()[:coarse.modes]))

    def test_taper_follows_the_column_spacing(self):
        conv = HeightSpectralConv(8, 40.0, 0.04, 0.3, hidden=8)
        for dz in (0.012, 0.02, 0.025, 0.164):
            column = torch.arange(0, 14.0, dz)
            taper, wavelength = conv.cutoff(column), 2 * np.pi / conv.k[1:]
            self.assertTrue(torch.all(taper[1:][wavelength >= 3 * dz * 1.001] == 1))
            self.assertTrue(torch.all(taper[1:][wavelength <= 2 * dz] < 1e-6))
            self.assertEqual(float(taper[0]), 1.0)

    def test_shared_grid_equals_per_column_path(self):
        torch.manual_seed(0)
        conv = HeightSpectralConv(8, 40.0, 0.04, 0.3, hidden=8, rank=2).double()
        x = torch.randn(1, 8, 300, 2, 3, dtype=torch.float64)
        z = torch.linspace(6.0, 0.0, 300, dtype=torch.float64)[None, :, None, None].expand(1, 300, 2, 3).clone()
        shared = conv(x, z)
        z[0, 150, 1, 2] += 1e-9                                      # one column apart: the per-column path
        self.assertLess((conv(x, z) - shared).abs().max().item(), 1e-6)



class HeightOperator(unittest.TestCase):
    def test_same_populations_on_any_vertical_grid(self):
        torch.manual_seed(1)
        got = grid_difference(model_of())
        print(f"\ndifference across vertical grids (of the output range): {got:.2e}")
        self.assertLess(got, 5e-3)

    def test_fourier_weights_are_exact_for_piecewise_linear_columns(self):
        z = stretched(30)
        k = torch.tensor([0.0, 0.7, 5.0, 40.0], dtype=torch.float64)
        q_re, q_im = height_fourier_weights(z, k)
        self.assertLess((q_re[:, 0] - trapezoid_weights(z[None, None, :, None, None])[0, 0, :, 0, 0]).abs().max(), 1e-12)
        self.assertLess(q_im[:, 0].abs().max(), 1e-12)
        x = 2.0 - 0.3 * z                                       # linear: the integral is exact at every k
        lo, hi = z.min(), z.max()
        for m, km in enumerate(k[1:].tolist(), start=1):
            f = lambda zz: np.exp(-1j * km * zz) * ((2.0 - 0.3 * zz) / (-1j * km) - (-0.3) / (-1j * km) ** 2)
            exact = f(hi.item()) - f(lo.item())
            got = complex((x * q_re[:, m]).sum().item(), (x * q_im[:, m]).sum().item())
            self.assertLess(abs(got - exact), 1e-9 * max(1.0, abs(exact)), km)
        flipped_re, flipped_im = height_fourier_weights(z.flip(0), k)
        self.assertLess((flipped_re.flip(0) - q_re).abs().max(), 1e-12)
        self.assertLess((flipped_im.flip(0) - q_im).abs().max(), 1e-12)

    def test_weighted_groupnorm(self):
        torch.manual_seed(2)
        norm = DepthWeightedGroupNorm(4, 16)
        plain = torch.nn.GroupNorm(4, 16)
        plain.load_state_dict(norm.state_dict())
        x = torch.randn(2, 16, 7, 3, 4)
        self.assertLess((norm(x, torch.ones(2, 1, 7, 3, 4)) - plain(x)).abs().max().item(), 1e-5)
        self.assertTrue(torch.equal(norm(x), plain(x)))

    def test_tall_columns_take_a_longer_box(self):
        """A column taller than half of reference_length gets a periodic length of twice its height: the same band
        of wavenumbers, sampled more densely."""
        conv = HeightSpectralConv(4, 20.0, 0.5, 0.3)
        x, z = atmosphere(uniform(50))                     # 15 Mm
        length, k, _ = conv.box(z)
        self.assertAlmostEqual(length, 30.0, places=4)
        self.assertEqual(k.numel(), 61)
        self.assertLessEqual(k.max().item(), 2 * np.pi / 0.5 + 1e-5)
        self.assertTrue(torch.isfinite(conv(torch.randn(1, 4, *z.shape[1:]), z)).all())

    def test_longer_box_changes_little(self):
        """The same weights with a longer periodic length (50, not 40 Mm) on columns within 20 Mm: W(k) is sampled
        between the trained wavenumbers, and the output barely moves."""
        torch.manual_seed(6)
        trained = HeightSpectralConv(8, 40.0, 0.3, 0.3, rank=4).eval()
        longer = HeightSpectralConv(8, 50.0, 0.3, 0.3, rank=4).eval()
        with torch.no_grad():
            for name, p in trained.named_parameters():
                if "freq_u" in name:
                    p.normal_(std=0.1)
        longer.load_state_dict(trained.state_dict())
        x, z = atmosphere(stretched(300))
        x = torch.cat([x, x[:, :2]], 1)
        with torch.no_grad():
            a, b = trained(x, z), longer(x, z)
        self.assertLess(((b - a).abs().max() / (a.max() - a.min())).item(), 1e-2)

    def test_parameters_get_gradients(self):
        torch.manual_seed(3)
        model = model_of().train()
        x, z = atmosphere(stretched(40))
        model(x, z, 48e3, 48e3).square().mean().backward()
        for name, p in model.named_parameters():
            if ".vertical." in name and ".freq_v." not in name and ".freq_lowrank." not in name:   # zero: U starts at 0
                with self.subTest(parameter=name):
                    self.assertIsNotNone(p.grad)
                    self.assertGreater(p.grad.abs().max().item(), 0.0)

    def test_no_grid_point_kernels(self):
        model = model_of()
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv3d)):
                with self.subTest(module=name):
                    self.assertTrue(all(k == 1 for k in module.kernel_size), name)

    def test_batched_backend_equals_forward(self):
        torch.manual_seed(4)
        model = model_of().eval()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".freq_u." in name:
                    p.normal_(std=0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            zeros, ones = np.zeros(6, np.float32), np.ones(6, np.float32)
            torch.save(dict(model_state=model.state_dict(), epoch=1, io_metadata=dict(Cin=6, Cout=6),
                            normalization_stats=dict(mean_X=zeros, std_X=ones, mean_Y=zeros, std_Y=ones)), path)
            x, z = atmosphere(stretched(37), rows=11, columns=9)
            with torch.no_grad():
                expected = model(x, z, 48e3, 48e3)[0]
            backend = BatchedFFNOBackend(path, lambda Cin, Cout, device: model_of(), devices=["cpu"],
                                         log=lambda message: None, chunk_depth=5, chunk_rows=3)
            got = backend.predict_log10(x[0].numpy(), z[0].numpy(), 48e3, 48e3)
            self.assertLess((got - expected).abs().max().item(), 2e-4)

    def test_fsdp_route_equals_forward(self):
        """The --fsdppredict loop (FFNONet._distributed_full_forward) on one rank gives forward's output, in one pass and
        in the batched mode, with the free device memory set so small that the GroupNorms, the pointwise paths and the
        fusion run a few levels at a time and HeightSpectralConv one row at a time."""
        import distributed_inference
        from FFNONet import _distributed_full_forward
        torch.manual_seed(5)
        for rank in (4, 0):
            for free in (None, 400_000):
                with self.subTest(rank=rank, free=free):
                    model = model_of(freq_mix_rank=rank).eval()
                    x, z = atmosphere(stretched(37), rows=11, columns=9)
                    distributed_inference.FREE_MEMORY_BYTES = free
                    try:
                        with torch.no_grad():
                            expected = model(x, z, 48e3, 48e3)
                            route = enable_distributed_inference(model)
                            chunked = [m for m in route.modules() if isinstance(m, ChunkedHeightSpectralConv)]
                            self.assertEqual(len(chunked), CONFIG["n_layers"])
                            if free:
                                hidden = x.new_empty(1, CONFIG["width"], 37, 11, 9)
                                self.assertLess(distributed_inference.depth_chunk(hidden, 3), 37)
                            got, _ = _distributed_full_forward(route, x, z, 48e3, 48e3)
                    finally:
                        distributed_inference.FREE_MEMORY_BYTES = None
                    self.assertLess((got - expected).abs().max().item(), 1e-4)


class FinalModel(unittest.TestCase):
    @unittest.skipUnless(os.path.isdir(FINAL), "the final model's checkpoints are not on this machine")
    def test_checkpoints_load_into_the_defaults(self):
        """Both zaug checkpoints load into config.MODEL_CONFIG with every key matching (each with its own species)."""
        from ffno_model_factory import create_ffno3d
        from train_utils import load_checkpoint
        for atom in ("H", "CA"):
            with self.subTest(atom=atom):
                checkpoint = load_checkpoint(os.path.join(FINAL, atom, f"3D_sim_train_{atom}.pt"), map_location="cpu")
                io = checkpoint["io_metadata"]
                model = create_ffno3d(io["Cin"], io["Cout"], atoms=[atom])
                model.load_state_dict(checkpoint["model_state"], strict=True)


if __name__ == "__main__":
    unittest.main()

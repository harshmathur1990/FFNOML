"""The level-fraction head (FFNO3D output_mode='level_fraction'): log10 n_i = log10(rho / mass) + s + log10 softmax(z)_i.
It reproduces any positive levels whose total lies within total_bound dex of rho / mass, including totals far from it;
the total starts at rho / mass; gradients reach rho; the batched backend gives what forward gives; and the default
log_population model is unchanged."""
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffno_batched_backend import BatchedFFNOBackend
from models.ffno_model import FFNO3D

BASE = dict(in_channels=6, width=16, n_layers=2, hidden_spectral=16, hidden_vertical=16)
H = dict(levels=6, log10_particle_mass=math.log10(2.2697e-27))
CA = dict(levels=6, log10_particle_mass=math.log10(1.0375e-21))
MEAN_X = np.array([3.8, 0.0, 0.0, 0.0, 16.0, -7.0], np.float32)
STD_X = np.array([0.4, 5.0, 5.0, 5.0, 2.0, 2.5], np.float32)


def fraction_model(species, seed=3):
    torch.manual_seed(seed)
    out = sum(s["levels"] for s in species)
    model = FFNO3D(out_channels=out, output_mode="level_fraction", species=species, **BASE).eval()
    model.set_normalization(MEAN_X, STD_X, np.linspace(10.0, 15.0, out), np.linspace(1.0, 2.0, out))
    return model


def inputs(shape=(1, 6, 24, 11, 9), seed=5):
    torch.manual_seed(seed)
    x = torch.randn(*shape)
    z = torch.linspace(14.0, -1.0, shape[2])[:, None, None].repeat(1, *shape[3:])[None].repeat(shape[0], 1, 1, 1)
    return x, z


def log10_populations(model, x, z):
    y = model(x, z, 48e3, 48e3)
    return y * model.std_Y[None, :, None, None, None] + model.mean_Y[None, :, None, None, None]


def log10_reference_total(model, x, species_index):
    log10_rho = x[:, 5] * model.std_X[5] + model.mean_X[5]
    return log10_rho - model.log10_particle_mass[species_index]


class LevelFractionHead(unittest.TestCase):
    def test_any_target_within_the_bound_is_reproduced_exactly(self):
        # levels whose totals span the measured MULTI3D range relative to rho / mass: -3.16 dex (qs006003/689) to
        # +5.40 dex (en024031/109), with level ratios over 30 dex
        model = fraction_model([H, CA]).double()
        torch.manual_seed(11)
        x = torch.randn(1, 6, 5, 4, 3, dtype=torch.float64)
        offsets = torch.tensor([-3.16, -0.5, -0.021, 0.0, 1.93, 5.40], dtype=torch.float64)
        ref = torch.stack([log10_reference_total(model, x, i) for i in (0, 1)], 1)  # [1, 2, D, H, W]
        log10_n = torch.empty(1, 12, 5, 4, 3, dtype=torch.float64)
        head = torch.empty(1, 14, 5, 4, 3, dtype=torch.float64)
        for sp in (0, 1):
            ratios = torch.randn(1, 6, 5, 4, 3, dtype=torch.float64) * 6.0          # log10 level ratios
            frac = ratios - torch.logsumexp(ratios * math.log(10.0), 1, keepdim=True) / math.log(10.0)
            s = offsets[torch.randint(0, 6, (1, 5, 4, 3))]
            log10_n[:, 6 * sp:6 * sp + 6] = ref[:, sp][:, None] + s[:, None] + frac
            head[:, 7 * sp:7 * sp + 6] = ratios * math.log(10.0)                    # logits: softmax gives frac
            head[:, 7 * sp + 6] = model.total_bound * torch.atanh(s / model.total_bound)
        y = model.head_output(head, x) * model.std_Y[None, :, None, None, None] + model.mean_Y[None, :, None, None, None]
        self.assertLess((y - log10_n).abs().max().item(), 1e-9)

    def test_total_starts_at_rho_over_mass(self):
        model = fraction_model([H, CA])
        x, z = inputs()
        with torch.no_grad():
            log10_n = log10_populations(model, x, z).double()
        for sp in (0, 1):
            total = torch.logsumexp(log10_n[:, 6 * sp:6 * sp + 6] * math.log(10.0), 1) / math.log(10.0)
            self.assertLess((total - log10_reference_total(model, x, sp).double()).abs().max().item(), 1e-5)
        self.assertEqual(model.proj2.out_channels, 14)

    def test_populations_stay_finite_in_float32(self):
        model = fraction_model([H])
        with torch.no_grad():
            model.proj2.bias[6] = 1e6                                                 # total correction at its bound
        x, z = inputs()
        with torch.no_grad():
            y = log10_populations(model, x, z)
        self.assertTrue(torch.isfinite(torch.pow(10.0, y)).all())
        self.assertLess((y.max() - (log10_reference_total(model, x, 0).max() + 6.0)).item(), 1e-3)

    def test_gradients(self):
        # the head in float64 (the rest of FFNO3D builds float32 frequency features, so it is checked in float32 below)
        head_model = fraction_model([H, CA]).double()
        torch.manual_seed(7)
        head = torch.randn(1, 14, 4, 3, 2, dtype=torch.float64, requires_grad=True)
        x = torch.randn(1, 6, 4, 3, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(head_model.head_output, (head, x), eps=1e-6, atol=1e-6))
        # through the whole freshly built model, the total is rho / mass, so d log10(total) / d rho_normalised is
        # std_X[rho] and no other input moves it
        model = fraction_model([H])
        x, z = inputs((1, 6, 6, 5, 4))
        x = x.requires_grad_(True)
        y = log10_populations(model, x, z).double()
        total = torch.logsumexp(y * math.log(10.0), dim=1) / math.log(10.0)
        (g,) = torch.autograd.grad(total.sum(), x)
        self.assertLess((g[:, 5] - model.std_X[5]).abs().max().item(), 1e-4)
        self.assertLess(g[:, :5].abs().max().item(), 1e-4)

    def test_batched_backend_equals_forward(self):
        model = fraction_model([H])
        with torch.no_grad():
            model.proj2.weight[6].normal_(0.0, 0.3)                                   # a non-zero total correction
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            stats = {k: getattr(model, k).numpy() for k in ("mean_X", "std_X", "mean_Y", "std_Y")}
            torch.save(dict(model_state=model.state_dict(), epoch=1, io_metadata=dict(Cin=6, Cout=6),
                            normalization_stats=stats), path)
            config = dict(out_channels=6, output_mode="level_fraction", species=[H], **BASE)
            backend = BatchedFFNOBackend(path, lambda Cin, Cout, device: FFNO3D(**config), devices=["cpu"],
                                         log=lambda message: None, chunk_depth=5, chunk_rows=3, vertical_chunk=7)
            x, z = inputs()
            features = x[0] * torch.as_tensor(STD_X)[:, None, None, None] + torch.as_tensor(MEAN_X)[:, None, None, None]
            got = backend.predict_log10(features.numpy(), z[0].numpy(), 48e3, 48e3)
        with torch.no_grad():
            expected = log10_populations(model, x, z)[0]
        self.assertLess((got - expected).abs().max().item(), 2e-4)

    def test_log_population_default_is_unchanged(self):
        torch.manual_seed(3)
        model = FFNO3D(out_channels=6, **BASE).eval()
        self.assertEqual(model.output_mode, "log_population")
        self.assertEqual(model.proj2.out_channels, 6)
        self.assertFalse(any(k in model.state_dict() for k in ("mean_X", "std_X", "mean_Y", "std_Y", "log10_particle_mass")))
        model.set_normalization(MEAN_X, STD_X, np.zeros(6), np.ones(6))
        x, z = inputs()
        with torch.no_grad():
            y = model(x, z, 48e3, 48e3)
            head = model._run_head(model.blocks[1](model.blocks[0](model.lift(x), z, 48e3, 48e3), z, 48e3, 48e3))
        self.assertTrue(torch.equal(y, head))

    def test_species_must_match_out_channels(self):
        with self.assertRaises(ValueError):
            FFNO3D(out_channels=12, output_mode="level_fraction", species=[H], **BASE)


if __name__ == "__main__":
    unittest.main()

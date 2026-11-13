"""The level-fraction head (FFNO3D output_mode='level_fraction'): the levels add up to the species total from rho, or
stay below it with a remainder slot; gradients reach rho; the batched backend gives what forward gives; and the default
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
H = dict(levels=6, log10_particle_mass=math.log10(2.3811e-27), remainder=False)
CA = dict(levels=6, log10_particle_mass=math.log10(2.3811e-27) - (6.34 - 12.0), remainder=True)
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


def log10_total(model, x, species_index):
    log10_rho = x[:, 5] * model.std_X[5] + model.mean_X[5]
    return log10_rho - model.log10_particle_mass[species_index]


class LevelFractionHead(unittest.TestCase):
    def test_levels_add_up_to_the_total_from_rho(self):
        model = fraction_model([H])
        x, z = inputs()
        with torch.no_grad():
            log10_n = log10_populations(model, x, z).double()
        ratio = (10.0 ** (log10_n - log10_total(model, x, 0).double()[:, None])).sum(1)
        self.assertLess((ratio - 1.0).abs().max().item(), 1e-5)

    def test_remainder_slot_keeps_levels_below_the_total(self):
        model = fraction_model([H, CA])
        x, z = inputs()
        with torch.no_grad():
            log10_n = log10_populations(model, x, z).double()
        h = (10.0 ** (log10_n[:, :6] - log10_total(model, x, 0).double()[:, None])).sum(1)
        ca = (10.0 ** (log10_n[:, 6:] - log10_total(model, x, 1).double()[:, None])).sum(1)
        self.assertLess((h - 1.0).abs().max().item(), 1e-5)
        self.assertLess(ca.max().item(), 1.0 + 1e-6)
        self.assertLess(ca.min().item(), 1.0 - 1e-3)
        self.assertEqual(model.proj2.out_channels, 13)

    def test_gradient_reaches_rho(self):
        # the head in float64 (the rest of FFNO3D builds float32 frequency features, so it is checked in float32 below)
        head_model = fraction_model([H, CA]).double()
        torch.manual_seed(7)
        head = torch.randn(1, 13, 4, 3, 2, dtype=torch.float64, requires_grad=True)
        x = torch.randn(1, 6, 4, 3, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(head_model.head_output, (head, x), eps=1e-6, atol=1e-6))
        # through the whole model: the levels add up to the total, so d log10(sum of levels) / d rho_normalised is
        # std_X[rho] everywhere, and nothing else moves the total
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

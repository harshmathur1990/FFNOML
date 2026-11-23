"""Departure coefficients in the level-fraction head: the level logits are MULTI3D's LTE weights from the input T and
ne plus the network's delta = ln b. delta = 0 gives MULTI3D's LTE fractions, as an independent transcription of
calc_lte_populations (multi3d/src/phys_mod.f90) computes them; the atom data is required; gradients reach T and ne; the
batched backend gives what forward gives."""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ffno_batched_backend import BatchedFFNOBackend  # noqa: E402
from models.ffno_model import FFNO3D  # noqa: E402

BASE = dict(in_channels=6, width=16, n_layers=2, hidden_spectral=16, hidden_vertical=16)
# config.ATOM_CONFIG's MULTI3D atom data
H_LTE = dict(g=[2.0, 8.0, 18.0, 32.0, 50.0, 1.0], ion=[1, 1, 1, 1, 1, 2],
             ev=[0.0, 10.198673740091325, 12.087318311066094, 12.748311674749516, 13.054262568222963, 13.60794573029334])
CA_LTE = dict(g=[2.0, 4.0, 6.0, 2.0, 4.0, 1.0], ion=[2, 2, 2, 2, 2, 3],
              ev=[0.0, 1.69242903005873, 1.6999489817497997, 3.12337806212316, 3.151018087686871, 11.87181686885173])
H = dict(levels=6, log10_particle_mass=math.log10(2.2697e-27))
CA = dict(levels=6, log10_particle_mass=math.log10(1.0375e-21))
MEAN_X = np.array([3.8, 0.0, 0.0, 0.0, 16.0, -7.0], np.float32)
STD_X = np.array([0.4, 5.0, 5.0, 5.0, 2.0, 2.5], np.float32)


def multi3d_nstar_fractions(t, ne_cm3, atom):
    """calc_lte_populations line by line, in numpy float64: nstar_i / totn."""
    ee, hh, em, bk, pi = 1.602189e-12, 6.626176e-27, 9.109534e-28, 1.380662e-16, 3.14159265359
    e_k = ee / bk
    g, ev, ion = atom["g"], atom["ev"], atom["ion"]
    ndxi = [0] * 21
    for i in range(2, 21):
        ndxi[i] = ndxi[i - 1] + (i - 1)
    for i in range(1, 21):
        ndxi[i] = ndxi[i] - ndxi[ion[0]]
    ccon = 0.5 * (hh / math.sqrt(2.0 * pi * em) / math.sqrt(bk)) ** 3
    conl = np.log(ccon * ne_cm3) - 1.5 * np.log(t)
    dxi = 4.98e-4 * 5040.0 / t * np.sqrt(ne_cm3 * bk * t)
    tns, sumn = [np.ones_like(t)], np.ones_like(t)
    for i in range(1, len(g)):
        evi = ev[i] - dxi * ndxi[ion[i]]
        tnsl = math.log(g[i]) - math.log(g[0]) - e_k / t * evi
        if ion[i] > ion[0]:
            tnsl = tnsl - float(ion[i] - ion[0]) * conl
        tns.append(np.exp(tnsl))
        sumn = sumn + tns[-1]
    return np.stack(tns, 0) / sumn


def model_for(species, seed=3, zero_delta=False):
    torch.manual_seed(seed)
    out = sum(s["levels"] for s in species)
    model = FFNO3D(out_channels=out, species=species, **BASE).eval()
    model.set_normalization(MEAN_X, STD_X, np.linspace(10.0, 15.0, out), np.linspace(1.0, 2.0, out))
    if zero_delta:
        with torch.no_grad():
            c = 0
            for s in species:
                model.proj2.weight[c:c + s["levels"]].zero_()
                model.proj2.bias[c:c + s["levels"]].zero_()
                c += s["levels"] + 1
    return model


def solar_inputs(shape=(1, 6, 24, 11, 9), seed=5):
    """Normalised inputs whose T and ne span 2,000-60,000 K and 10^14-10^23 m^-3, independently."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g)
    log10_t = math.log10(2000.0) + (math.log10(6e4) - math.log10(2000.0)) * torch.rand(shape[0], *shape[2:], generator=g)
    log10_ne = 14.0 + 9.0 * torch.rand(shape[0], *shape[2:], generator=g)
    x[:, 0] = (log10_t - float(MEAN_X[0])) / float(STD_X[0])
    x[:, 4] = (log10_ne - float(MEAN_X[4])) / float(STD_X[4])
    z = torch.linspace(14.0, -1.0, shape[2])[:, None, None].repeat(1, *shape[3:])[None].repeat(shape[0], 1, 1, 1)
    return x, z


def log10_populations(model, x, z):
    y = model(x, z, 48e3, 48e3)
    return y * model.std_Y[None, :, None, None, None] + model.mean_Y[None, :, None, None, None]


def unnormalised(x, channel):
    return (x[:, channel].double() * float(STD_X[channel]) + float(MEAN_X[channel])).numpy()


class ZeroDeltaIsMulti3dLTE(unittest.TestCase):
    def test_fractions_match_calc_lte_populations(self):
        species = [dict(H, multi3d_lte=H_LTE), dict(CA, multi3d_lte=CA_LTE)]
        model = model_for(species, zero_delta=True)
        x, z = solar_inputs()
        with torch.no_grad():
            log10_n = log10_populations(model, x, z).double()
        t, ne_cm3 = 10.0 ** unnormalised(x, 0)[0], 10.0 ** (unnormalised(x, 4)[0] - 6.0)
        for sp, atom in enumerate((H_LTE, CA_LTE)):
            with self.subTest(species=sp):
                n = log10_n[0, 6 * sp:6 * sp + 6]
                total = torch.logsumexp(n * math.log(10.0), 0) / math.log(10.0)
                got = (n - total).numpy()
                expected = np.log10(np.maximum(multi3d_nstar_fractions(t, ne_cm3, atom), 1e-300))
                usable = expected > -37.0              # float32 populations resolve shares above 10^-37 or so
                self.assertGreater(usable.mean(), 0.5)
                self.assertLess(np.abs(got - expected)[usable].max(), 2e-5)

    def test_ca_iii_debye_lowering_is_multi3d_s(self):
        # MULTI3D's ndxi loop zeroes ndxi(ion_1) on the way, so Ca III is lowered by 3 dxi, not 2
        from models.ffno_model import multi3d_lte_tables
        self.assertEqual(multi3d_lte_tables(CA_LTE, 6)[2], [0.0] * 5 + [3.0])
        self.assertEqual(multi3d_lte_tables(H_LTE, 6)[2], [0.0] * 5 + [1.0])


class Configuration(unittest.TestCase):
    def test_needs_the_atom_data(self):
        with self.assertRaises(ValueError):
            FFNO3D(out_channels=6, species=[H], **BASE)

    def test_adds_no_parameters_or_buffers(self):
        model = model_for([dict(H, multi3d_lte=H_LTE)])
        self.assertEqual(sorted(k for k in model.state_dict() if not k.startswith(("lift.", "blocks.", "proj"))),
                         ["log10_particle_mass", "mean_X", "mean_Y", "std_X", "std_Y"])

    def test_config_gives_the_atom_data(self):
        env = dict(os.environ, FFNOML_ACTIVE_ATOMS="H,CA")
        env.pop("FFNOML_MODEL_CONFIG", None)
        out = subprocess.run([sys.executable, "-c", "import json, config; print(json.dumps(config.MODEL_CONFIG['species']))"],
                             cwd=REPO, env=env, check=True, capture_output=True, text=True).stdout
        species = json.loads(out.strip().splitlines()[-1])
        self.assertEqual([s["multi3d_lte"] for s in species], [H_LTE, CA_LTE])


class Gradients(unittest.TestCase):
    def test_head_gradcheck(self):
        model = model_for([dict(H, multi3d_lte=H_LTE), dict(CA, multi3d_lte=CA_LTE)]).double()
        x, _ = solar_inputs((1, 6, 4, 3, 2))
        head = torch.randn(1, 14, 4, 3, 2, dtype=torch.float64, requires_grad=True)
        x = x.double().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(model.head_output, (head, x), eps=1e-6, atol=1e-6))

    def test_level_2_follows_boltzmann_in_temperature(self):
        # at delta = 0, log10(n2/n1) = log10 4 - e_k ev_2 / (T ln 10), so d log10(n2/n1) / d log10 T = e_k ev_2 / T,
        # times std_X[0] for the normalised input; nothing else moves it
        model = model_for([dict(H, multi3d_lte=H_LTE)], zero_delta=True)
        x, z = solar_inputs((1, 6, 6, 5, 4))
        x = x.requires_grad_(True)
        y = log10_populations(model, x, z)
        (g,) = torch.autograd.grad((y[:, 1] - y[:, 0]).sum(), x)
        t = 10.0 ** unnormalised(x.detach(), 0)
        expected = (1.602189e-12 / 1.380662e-16) * 10.198673740091325 / t * float(STD_X[0])
        self.assertLess(g[:, 1:].abs().max().item(), 1e-6 * np.abs(expected).max())
        self.assertLess(np.abs(g[:, 0].double().numpy() - expected).max() / np.abs(expected).max(), 1e-4)


class BatchedBackend(unittest.TestCase):
    def test_equals_forward(self):
        species = [dict(H, multi3d_lte=H_LTE)]
        model = model_for(species)
        with torch.no_grad():
            model.proj2.weight[6].normal_(0.0, 0.3)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            stats = {k: getattr(model, k).numpy() for k in ("mean_X", "std_X", "mean_Y", "std_Y")}
            torch.save(dict(model_state=model.state_dict(), epoch=1, io_metadata=dict(Cin=6, Cout=6),
                            normalization_stats=stats), path)
            config = dict(out_channels=6, species=species, **BASE)
            backend = BatchedFFNOBackend(path, lambda Cin, Cout, device: FFNO3D(**config), devices=["cpu"],
                                         log=lambda message: None, chunk_depth=5, chunk_rows=3)
            x, z = solar_inputs()
            features = x[0] * torch.as_tensor(STD_X)[:, None, None, None] + torch.as_tensor(MEAN_X)[:, None, None, None]
            got = backend.predict_log10(features.numpy(), z[0].numpy(), 48e3, 48e3)
        with torch.no_grad():
            expected = log10_populations(model, x, z)[0]
        self.assertLess((got - expected).abs().max().item(), 2e-4)


if __name__ == "__main__":
    unittest.main()

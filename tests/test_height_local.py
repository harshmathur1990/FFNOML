"""FFNO3D(z_local_derivatives=True): every block adds a local height path, a 1x1 mix of the block input with its slope
and curvature in height (height_derivatives) on each column's own heights. The finite differences are exact for a
quadratic on non-uniform grids and converge on refinement; at zero initialisation the model equals the one without the
path; trained, the same atmosphere on different vertical grids gives the same populations at the same heights; the
batched backend and the FSDP route equal forward."""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
from ffno_batched_backend import BatchedFFNOBackend  # noqa: E402
from models.ffno_model import FFNO3D, height_derivatives  # noqa: E402
from test_height_operator import CONFIG, OPERATOR, at_heights, atmosphere, populations, stretched, uniform  # noqa: E402

LOCAL = dict(OPERATOR, z_local_derivatives=True, z_local_length=0.1)


def columns(z, f):
    """f(z) on heights z [D] as x [1, 2, D, 2, 3] (two channels, the second 2 f) and z_scale [1, D, 2, 3]."""
    x = torch.stack([f(z), 2.0 * f(z)])[None, :, :, None, None].expand(1, 2, len(z), 2, 3)
    return x, z[None, :, None, None].expand(1, len(z), 2, 3)


def randomise_local(model, std=0.05):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "z_local_mix" in name:
                p.normal_(std=std)


class HeightDerivatives(unittest.TestCase):
    def test_exact_for_a_quadratic(self):
        z = stretched(23)
        x, zs = columns(z, lambda z: 0.3 * z ** 2 - 1.2 * z + 0.7)
        d1, d2 = height_derivatives(x, zs)
        inner = slice(1, -1)
        self.assertLess((d1[0, 0, inner] - (0.6 * z[inner, None, None] - 1.2)).abs().max().item(), 1e-9)
        self.assertLess((d2[0, 0] - 0.6).abs().max().item(), 1e-9)
        self.assertLess((d1[0, 1] - 2.0 * d1[0, 0]).abs().max().item(), 1e-9)

    def test_matches_analytic_and_converges(self):
        """sin(z) on stretched grids: the error falls with the spacing (first order or better, non-uniform)."""
        errors = []
        for depth in (41, 81, 161):
            z = stretched(depth)
            x, zs = columns(z, torch.sin)
            d1, d2 = height_derivatives(x, zs)
            inner = slice(1, -1)
            errors.append(((d1[0, 0, inner] - torch.cos(z[inner, None, None])).abs().max().item(),
                           (d2[0, 0, inner] + torch.sin(z[inner, None, None])).abs().max().item()))
        for (e1, e2), (f1, f2) in zip(errors[:-1], errors[1:]):
            self.assertLess(f1, 0.55 * e1)
            self.assertLess(f2, 0.6 * e2)
        self.assertLess(errors[-1][0], 5e-3)
        self.assertLess(errors[-1][1], 2e-2)

    def test_direction_of_the_levels(self):
        """the derivative is with respect to z whether the levels run up or down"""
        z = uniform(17)
        x, zs = columns(z, lambda z: 2.0 * z)
        d1, _ = height_derivatives(x, zs)
        d1_flip, _ = height_derivatives(x.flip(2), zs.flip(1))
        self.assertLess((d1[0, 0] - 2.0).abs().max().item(), 1e-9)
        self.assertLess((d1_flip[0, 0] - 2.0).abs().max().item(), 1e-9)

    def test_short_columns(self):
        for depth in (1, 2):
            z = uniform(depth) if depth > 1 else torch.zeros(1, dtype=torch.float64)
            x, zs = columns(z, lambda z: 3.0 * z + 1.0)
            d1, d2 = height_derivatives(x, zs)
            self.assertTrue(torch.isfinite(d1).all() and torch.isfinite(d2).all())
            self.assertEqual(d2.abs().max().item(), 0.0)


class LocalHeightPath(unittest.TestCase):
    def test_zero_initialisation_equals_the_model_without(self):
        torch.manual_seed(1)
        off = FFNO3D(**CONFIG, **OPERATOR).eval()
        on = FFNO3D(**CONFIG, **LOCAL).eval()
        missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(missing and all("z_local_mix" in k for k in missing), missing)
        x, z = atmosphere(stretched(29))
        with torch.no_grad():
            self.assertEqual((on(x, z, 48e3, 48e3) - off(x, z, 48e3, 48e3)).abs().max().item(), 0.0)

    def test_switch_off_has_no_new_parameters(self):
        names = [k for k in FFNO3D(**CONFIG, **OPERATOR).state_dict() if "z_local" in k]
        self.assertEqual(names, [])

    def test_needs_the_height_operator(self):
        with self.assertRaises(ValueError):
            FFNO3D(**CONFIG, z_local_derivatives=True)

    def test_no_grid_point_kernels(self):
        model = FFNO3D(**CONFIG, **LOCAL)
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv3d)):
                self.assertTrue(all(k == 1 for k in module.kernel_size), name)

    def test_the_path_learns(self):
        torch.manual_seed(2)
        model = FFNO3D(**CONFIG, **LOCAL).train()
        x, z = atmosphere(stretched(25))
        model(x, z, 48e3, 48e3).square().mean().backward()
        grads = [p.grad for n, p in model.named_parameters() if "z_local_mix" in n]
        self.assertEqual(len(grads), CONFIG["n_layers"])
        self.assertTrue(all(g is not None and g.abs().max().item() > 0 for g in grads))

    def test_same_heights_on_different_grids(self):
        """with the path switched on and non-zero, a fine and a coarser stretched grid agree at common heights about
        as well as the model without it does"""
        torch.manual_seed(3)
        on = FFNO3D(**CONFIG, **LOCAL)
        randomise_local(on)
        off = FFNO3D(**CONFIG, **OPERATOR)
        off.load_state_dict(on.state_dict(), strict=False)
        fine, coarse = stretched(241), stretched(121)
        gaps = []
        for model in (off, on):
            a = populations(model, fine)
            b = populations(model, coarse)
            gaps.append((at_heights(a, fine, coarse).float() - b).abs()[..., 2:-2].max().item())
        self.assertLess(gaps[1], max(2.0 * gaps[0], 2e-2), gaps)

    def test_batched_backend_equals_forward(self):
        torch.manual_seed(4)
        model = FFNO3D(**CONFIG, **LOCAL).eval()
        randomise_local(model)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            zeros, ones = np.zeros(6, np.float32), np.ones(6, np.float32)
            torch.save(dict(model_state=model.state_dict(), epoch=1, io_metadata=dict(Cin=6, Cout=6),
                            normalization_stats=dict(mean_X=zeros, std_X=ones, mean_Y=zeros, std_Y=ones)), path)
            x, z = atmosphere(stretched(37), rows=11, columns=9)
            with torch.no_grad():
                expected = model(x, z, 48e3, 48e3)[0]
            backend = BatchedFFNOBackend(path, lambda Cin, Cout, device: FFNO3D(**CONFIG, **LOCAL), devices=["cpu"],
                                         log=lambda message: None, chunk_depth=5, chunk_rows=3, vertical_chunk=7)
            got = backend.predict_log10(x[0].numpy(), z[0].numpy(), 48e3, 48e3)
            self.assertLess((got - expected).abs().max().item(), 2e-4)

    def test_fsdp_route_equals_forward(self):
        import distributed_inference
        from distributed_inference import enable_distributed_inference
        from FFNONet import _distributed_full_forward
        x, z = atmosphere(stretched(37), rows=11, columns=9)
        for free in (None, 400_000):
            with self.subTest(free=free):
                torch.manual_seed(5)
                model = FFNO3D(**CONFIG, **LOCAL).eval()
                randomise_local(model)
                distributed_inference.FREE_MEMORY_BYTES = free
                try:
                    with torch.no_grad():
                        expected = model(x, z, 48e3, 48e3)
                        route = enable_distributed_inference(model)
                        got, _ = _distributed_full_forward(route, x, z, 48e3, 48e3)
                finally:
                    distributed_inference.FREE_MEMORY_BYTES = None
                self.assertLess((got - expected).abs().max().item(), 2e-4)


if __name__ == "__main__":
    unittest.main()

"""The capacity switches of FFNO3D (z_spectral_modes, freq_mix_rank, open_side_paths): off, the model is the one before
them; on, the depth and low-rank branches start as no-ops and learn, and the batched backend still equals forward."""
import importlib.util
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

CONFIG = dict(in_channels=6, out_channels=6, width=16, n_layers=2, spec_dropout=0.08, vertical_dropout=0.08,
              spec_dropout_layers=[0, 1], vertical_dropout_layers=[0, 1], hidden_spectral=16, hidden_vertical=16)
SWITCHES = dict(z_spectral_modes=8, freq_mix_rank=4, open_side_paths=True)


def inputs(seed=3, depth=24, rows=11, columns=9):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 6, depth, rows, columns, generator=g)
    z = torch.linspace(14.0, -1.0, depth)[None, :, None, None].repeat(1, 1, rows, columns)
    return x, z + 0.01 * torch.randn(z.shape, generator=g)


def output(model, x, z):
    with torch.no_grad():
        return model.eval()(x, z, 48e3, 48e3)


def model_before_switches():
    """FFNO3D from models/ffno_model.py at the last commit without the switches, or None outside a git checkout."""
    for revision in ("HEAD", "HEAD~1", "HEAD~2", "HEAD~3"):
        try:
            source = subprocess.run(["git", "-C", REPO, "show", f"{revision}:models/ffno_model.py"], check=True,
                                    capture_output=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        if "z_spectral_modes" not in source:
            break
    else:
        return None
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "ffno_model_before.py")
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("ffno_model_before", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FFNO3D


class SwitchesOff(unittest.TestCase):
    def test_same_parameters_and_outputs_as_before(self):
        before = model_before_switches()
        if before is None:
            self.skipTest("no git history")
        torch.manual_seed(5)
        old = before(**CONFIG)
        new = FFNO3D(**CONFIG, z_spectral_modes=0, freq_mix_rank=0, open_side_paths=False)
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        new.load_state_dict(old.state_dict(), strict=True)
        x, z = inputs()
        self.assertTrue(torch.equal(output(old, x, z), output(new, x, z)))

    def test_defaults_are_off(self):
        torch.manual_seed(5)
        default = FFNO3D(**CONFIG)
        torch.manual_seed(5)
        off = FFNO3D(**CONFIG, z_spectral_modes=0, freq_mix_rank=0, open_side_paths=False)
        x, z = inputs()
        self.assertTrue(torch.equal(output(default, x, z), output(off, x, z)))


class SwitchesOn(unittest.TestCase):
    def test_new_branches_start_as_no_ops(self):
        torch.manual_seed(5)
        off = FFNO3D(**CONFIG)
        x, z = inputs()
        for switch in ("z_spectral_modes", "freq_mix_rank"):
            with self.subTest(switch=switch):
                on = FFNO3D(**CONFIG, **{switch: SWITCHES[switch]})
                missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
                self.assertFalse(unexpected)
                self.assertTrue(missing and all(".vertical.z_" in k or ".spec.freq_" in k for k in missing), missing)
                self.assertLess((output(on, x, z) - output(off, x, z)).abs().max().item(), 1e-5)

    def test_new_weights_get_gradients(self):
        torch.manual_seed(5)
        model = FFNO3D(**CONFIG, **SWITCHES).train()
        x, z = inputs()
        model(x, z, 48e3, 48e3).square().mean().backward()
        for name, p in model.named_parameters():
            if ".z_spec." in name or ".freq_u." in name:
                with self.subTest(parameter=name):
                    self.assertGreater(p.grad.abs().max().item(), 0.0)

    def test_open_side_paths(self):
        model = FFNO3D(**CONFIG, open_side_paths=True)
        for block in model.blocks:
            self.assertEqual(block.res_pw.item(), 0.5)
            self.assertEqual(block.res_mlp.item(), 0.5)
            y = torch.linspace(-5, 5, 11)
            self.assertTrue(torch.equal(block.mlp_activation(y), y))
        self.assertTrue(torch.equal(FFNO3D(**CONFIG).blocks[0].mlp_activation(y), torch.tanh(y)))


class BatchedBackendWithSwitches(unittest.TestCase):
    def test_equals_forward(self):
        torch.manual_seed(3)
        model = FFNO3D(**CONFIG, **SWITCHES).eval()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".z_spec." in name or ".freq_u." in name:
                    p.normal_(std=0.1)                       # not the zero start, so the branches matter
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            zeros, ones = np.zeros(6, np.float32), np.ones(6, np.float32)
            torch.save(dict(model_state=model.state_dict(), epoch=1, io_metadata=dict(Cin=6, Cout=6),
                            normalization_stats=dict(mean_X=zeros, std_X=ones, mean_Y=zeros, std_Y=ones)), path)
            x, z = inputs()
            with torch.no_grad():
                expected = model(x, z, 48e3, 48e3)[0]
            for options in (dict(chunk_depth=24, chunk_rows=11, vertical_chunk=99),
                            dict(chunk_depth=5, chunk_rows=3, vertical_chunk=7)):
                with self.subTest(**options):
                    backend = BatchedFFNOBackend(path, lambda Cin, Cout, device: FFNO3D(**CONFIG, **SWITCHES),
                                                 devices=["cpu"], log=lambda message: None, **options)
                    got = backend.predict_log10(x[0].numpy(), z[0].numpy(), 48e3, 48e3)
                    self.assertLess((got - expected).abs().max().item(), 2e-4)


if __name__ == "__main__":
    unittest.main()

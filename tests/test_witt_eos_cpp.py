import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

HAS_NUMPY = find_spec("numpy") is not None

if HAS_NUMPY:
    import numpy as np


def shared_library_suffix():
    if platform.system() == "Darwin":
        return ".dylib"
    if platform.system() == "Windows":
        return ".dll"
    return ".so"


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for Witt EOS C++ tests")
class WittEOSCppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = os.environ.get("CXX") or shutil.which("c++")
        if not cls.compiler:
            raise unittest.SkipTest("No C++ compiler found")

        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.lib_path = Path(cls.tmpdir.name) / f"libwitt_eos_cpp_test{shared_library_suffix()}"
        source_path = SCRIPTS / "witt_eos_cpp.cpp"
        command = [
            cls.compiler,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            str(source_path),
            "-o",
            str(cls.lib_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise unittest.SkipTest(f"C++ EOS compile failed: {exc.stderr}") from exc

        cls.lib = ctypes.CDLL(str(cls.lib_path))
        cls.lib.witt_ne_from_rho.argtypes = [
            ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        cls.lib.witt_ne_from_rho.restype = ctypes.c_int

        sys.path.insert(0, str(SCRIPTS))
        from witt import witt

        cls.py_eos = witt()
        cls.pf_path = str(SCRIPTS / "pf_Kurucz.input").encode()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmpdir"):
            cls.tmpdir.cleanup()

    def cpp_ne_from_rho(self, temp, rho_cgs, threads=4):
        temp = np.ascontiguousarray(temp, dtype=np.float64)
        rho_si = np.ascontiguousarray(rho_cgs * 1e3, dtype=np.float64)
        ne = np.empty(temp.shape, dtype=np.float32)
        status = self.lib.witt_ne_from_rho(
            self.pf_path,
            temp,
            rho_si,
            ne,
            temp.size,
            threads,
        )
        self.assertEqual(status, 0)
        return ne.astype(np.float64)

    def python_ne_from_rho(self, temp, rho_cgs):
        out = np.empty(temp.shape, dtype=np.float64)
        for i, (t_cell, rho_cell) in enumerate(zip(temp, rho_cgs)):
            pgas = self.py_eos.pg_from_rho(float(t_cell), float(rho_cell))
            pe = self.py_eos.pe_from_pg(float(t_cell), pgas)
            out[i] = pe / (self.py_eos.BK * float(t_cell)) * 1e6
        return out

    def assert_cpp_matches_python(self, temp, rho_cgs):
        cpp = self.cpp_ne_from_rho(temp, rho_cgs)
        py = self.python_ne_from_rho(temp, rho_cgs)
        rel = np.abs(cpp - py) / np.maximum(np.abs(py), 1e-300)

        self.assertTrue(np.all(np.isfinite(cpp)))
        self.assertLess(float(rel.max()), 1e-6)
        self.assertLess(float(np.median(rel)), 1e-7)

    def test_cpp_matches_python_on_representative_grid(self):
        temperatures = np.array(
            [2500.0, 3500.0, 4500.0, 5770.0, 8000.0, 12000.0, 20000.0],
            dtype=np.float64,
        )
        densities = np.array(
            [1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
            dtype=np.float64,
        )
        temp, rho = np.meshgrid(temperatures, densities, indexing="ij")

        self.assert_cpp_matches_python(temp.ravel(), rho.ravel())

    def test_cpp_matches_python_on_random_samples(self):
        rng = np.random.default_rng(123)
        temp = 10 ** rng.uniform(np.log10(2200.0), np.log10(25000.0), size=200)
        rho = 10 ** rng.uniform(-13.0, -3.5, size=200)

        self.assert_cpp_matches_python(temp, rho)


if __name__ == "__main__":
    unittest.main()

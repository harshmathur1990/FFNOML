import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

HAS_NUMPY = find_spec("numpy") is not None
HAS_ASTROPY = find_spec("astropy") is not None
HAS_H5PY = find_spec("h5py") is not None

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
            str(ROOT.parent / "stic" / "src" / "cop.cc"),
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
            ctypes.c_int,
        ]
        cls.lib.witt_ne_from_rho.restype = ctypes.c_int
        cls.lib.witt_ne_from_pgas.argtypes = [
            ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
        ]
        cls.lib.witt_ne_from_pgas.restype = ctypes.c_int
        cls.lib.witt_opacity500_mass_from_pgas.argtypes = [
            ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]
        cls.lib.witt_opacity500_mass_from_pgas.restype = ctypes.c_int

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
            0,
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

    def cpp_ne_from_pgas(self, temp, pgas_cgs, threads=4):
        temp = np.ascontiguousarray(temp, dtype=np.float64)
        pgas_si = np.ascontiguousarray(pgas_cgs / 10.0, dtype=np.float64)
        ne = np.empty(temp.shape, dtype=np.float32)
        status = self.lib.witt_ne_from_pgas(
            self.pf_path,
            temp,
            pgas_si,
            ne,
            temp.size,
            threads,
            0,
        )
        self.assertEqual(status, 0)
        return ne.astype(np.float64)

    def python_ne_from_pgas(self, temp, pgas_cgs):
        out = np.empty(temp.shape, dtype=np.float64)
        for i, (t_cell, pgas_cell) in enumerate(zip(temp, pgas_cgs)):
            pe = self.py_eos.pe_from_pg(float(t_cell), float(pgas_cell))
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

    def test_cpp_pgas_matches_python_pe_from_pg(self):
        temperatures = np.array(
            [2500.0, 3500.0, 4500.0, 5770.0, 8000.0, 12000.0, 20000.0],
            dtype=np.float64,
        )
        pressures = np.logspace(-2, 6, temperatures.size, dtype=np.float64)

        cpp = self.cpp_ne_from_pgas(temperatures, pressures)
        py = self.python_ne_from_pgas(temperatures, pressures)
        rel = np.abs(cpp - py) / np.maximum(np.abs(py), 1e-300)

        self.assertTrue(np.all(np.isfinite(cpp)))
        self.assertLess(float(rel.max()), 1e-6)

    def test_cpp_opacity500_matches_python_continuum(self):
        temperature = np.ascontiguousarray(
            [3500.0, 4500.0, 5770.0, 8000.0, 12000.0], dtype=np.float64
        )
        pgas_pa = np.ascontiguousarray([0.1, 1.0, 10.0, 100.0, 1000.0])
        actual = np.empty(temperature.size, dtype=np.float64)
        status = self.lib.witt_opacity500_mass_from_pgas(
            self.pf_path, temperature, pgas_pa, actual, temperature.size
        )
        self.assertEqual(status, 0)

        expected = np.empty_like(actual)
        for i, (temp, pressure_pa) in enumerate(zip(temperature, pgas_pa)):
            pressure_cgs = pressure_pa * 10.0
            pe = self.py_eos.pe_from_pg(float(temp), float(pressure_cgs))
            rho_si = self.py_eos.rho_from_pg(float(temp), float(pressure_cgs)) * 1e3
            extinction_cm = self.py_eos.contOpacity(
                float(temp), float(pressure_cgs), pe, np.array([5000.0])
            )[0]
            expected[i] = extinction_cm * 100.0 / rho_si

        relative = np.abs(actual - expected) / expected
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertTrue(np.all(actual > 0.0))
        self.assertLess(float(relative.max()), 3e-4)


@unittest.skipUnless(HAS_NUMPY, "NumPy is required to import the FITS converter")
class ElectronDensitySourceSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        from convert_iris_sim_fits_to_ffno_hdf5 import _find_electron_density_source

        cls.find_source = staticmethod(_find_electron_density_source)

    def quantity_path(self, folder, quantity):
        return Path(folder) / f"MURaM_test_{quantity}_1.fits"

    def test_source_priority_is_lgne_then_lgp_then_lgr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            lgr = self.quantity_path(folder, "lgr")
            lgp = self.quantity_path(folder, "lgp")
            lgne = self.quantity_path(folder, "lgne")

            lgr.touch()
            self.assertEqual(self.find_source(folder, "MURaM", "test", "1")[0], "lgr")
            lgp.touch()
            self.assertEqual(self.find_source(folder, "MURaM", "test", "1")[0], "lgp")
            lgne.touch()
            self.assertEqual(self.find_source(folder, "MURaM", "test", "1")[0], "lgne")

    def test_missing_sources_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                self.find_source(Path(tmpdir), "MURaM", "test", "1")


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for slice selection tests")
class SpatialSliceSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        from convert_iris_sim_fits_to_ffno_hdf5 import (
            _height_indices,
            _parse_slice_text,
            _resolve_slice,
        )

        cls.parse_text = staticmethod(_parse_slice_text)
        cls.resolve = staticmethod(_resolve_slice)
        cls.height_indices = staticmethod(_height_indices)

    def test_compact_and_python_slice_syntax_select_the_requested_stride(self):
        for text in ("0:10:2", "slice(0, 10, 2)"):
            selected = np.arange(10)[self.resolve(self.parse_text(text), None, None, 10, "x")]
            np.testing.assert_array_equal(selected, [0, 2, 4, 6, 8])

    def test_open_bounds_allow_every_third_point(self):
        selected = np.arange(10)[
            self.resolve(self.parse_text("::3"), None, None, 10, "z")
        ]
        np.testing.assert_array_equal(selected, [0, 3, 6, 9])

    def test_explicit_slice_cannot_be_mixed_with_legacy_bounds(self):
        with self.assertRaises(ValueError):
            self.resolve(self.parse_text("0:10:2"), 0, None, 10, "x")

    def test_zero_or_negative_steps_are_rejected(self):
        for text in ("0:10:0", "0:10:-1"):
            with self.assertRaises(ValueError):
                self.resolve(self.parse_text(text), None, None, 10, "x")

    def test_z_slice_is_applied_before_height_filter(self):
        heights = np.arange(8, dtype=np.float64) * 100.0
        candidates = np.arange(8)[self.resolve(self.parse_text("1:8:2"), None, None, 8, "z")]
        actual = self.height_indices(heights, 200.0, 600.0, candidates)
        np.testing.assert_array_equal(actual, [3, 5])


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for resampling tests")
class AtmosphereResamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        import convert_iris_sim_fits_to_ffno_hdf5 as converter
        from convert_iris_sim_fits_to_ffno_hdf5 import (
            _resample_atmosphere,
            _validate_atmosphere,
        )

        cls.converter = converter
        cls.resample = staticmethod(_resample_atmosphere)
        cls.validate = staticmethod(_validate_atmosphere)

    def test_linear_and_boost_cubic_preserve_linear_profiles(self):
        profile = np.arange(4, dtype=np.float32).reshape(1, 1, 4)
        expected = np.linspace(0.0, 3.0, 7).reshape(1, 1, 7)
        for method in ("linear", "cubic"):
            args = types.SimpleNamespace(
                upsample_factors={"z": 2},
                upsample_target_shape={},
                upsample_target_spacing_m={},
                interpolation_default={"x": method, "y": method, "z": method},
                interpolation_channels={},
                log_interpolation_channels=set(),
                continuity_max_relative_increase={},
            )
            fields = {
                "temperature": 5000.0 + profile,
                "rho": 1e-4 + profile,
                "vx": profile,
                "vy": profile,
                "vz": profile,
                "ne": 1e16 + profile,
            }
            actual, height, _, _ = self.resample(
                args, fields, np.array([3.0, 2.0, 1.0, 0.0]), 1.0, 1.0
            )
            np.testing.assert_allclose(actual["vx"], expected, atol=1e-6)
            np.testing.assert_allclose(height, np.linspace(3.0, 0.0, 7))

    def test_channel_specific_methods_and_all_axis_factors(self):
        base = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        fields = {
            "temperature": 5000.0 + base,
            "rho": 10.0 ** (-4.0 + base / 100.0),
            "vx": base,
            "vy": base,
            "vz": base,
            "ne": 10.0 ** (16.0 + base / 100.0),
        }
        args = types.SimpleNamespace(
            upsample_factors={"x": 2, "y": 2, "z": 2},
            upsample_target_shape={},
            upsample_target_spacing_m={},
            interpolation_default={"x": "nearest", "y": "linear", "z": "linear"},
            interpolation_channels={"temperature": {"z": "cubic"}},
            log_interpolation_channels={
                "temperature", "rho", "ne", "pgas", "hydrogen_populations"
            },
            continuity_max_relative_increase={},
        )
        result, height, dx, dy = self.resample(
            args, fields, np.array([100.0, 0.0]), 4.0, 6.0
        )
        self.assertEqual(result["temperature"].shape, (3, 3, 3))
        self.assertTrue(np.all(result["rho"] > 0))
        self.assertTrue(np.all(result["ne"] > 0))
        np.testing.assert_allclose(height, [100.0, 50.0, 0.0])
        self.assertEqual(dx, 2.0)
        self.assertEqual(dy, 3.0)

    def test_pgas_is_interpolated_in_log_space(self):
        ones = np.ones((2, 1, 1), dtype=np.float32)
        fields = {
            "temperature": ones * 5000.0,
            "rho": ones * 1e-4,
            "vx": np.zeros_like(ones),
            "vy": np.zeros_like(ones),
            "vz": np.zeros_like(ones),
            "pgas": np.array([1.0, 100.0], dtype=np.float32).reshape(2, 1, 1),
        }
        args = types.SimpleNamespace(
            upsample_factors={"x": 2},
            upsample_target_shape={},
            upsample_target_spacing_m={},
            interpolation_default={"x": "linear", "y": "linear", "z": "linear"},
            interpolation_channels={},
            log_interpolation_channels={"pgas"},
            continuity_max_relative_increase={"pgas": {"x": 0.0}},
        )

        result, _, _, _ = self.resample(
            args, fields, np.array([0.0]), 1.0, 1.0
        )
        np.testing.assert_allclose(result["pgas"][:, 0, 0], [1.0, 10.0, 100.0])

        args.log_interpolation_channels = set()
        with self.assertRaisesRegex(ValueError, "continuity worsened along x"):
            self.resample(args, fields, np.array([0.0]), 1.0, 1.0)

        args.continuity_max_relative_increase = {}
        result, _, _, _ = self.resample(
            args, fields, np.array([0.0]), 1.0, 1.0
        )
        np.testing.assert_allclose(result["pgas"][:, 0, 0], [1.0, 50.5, 100.0])

    def test_target_physical_spacing_is_used_exactly(self):
        base = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        fields = {
            "temperature": 5000.0 + base,
            "rho": np.full_like(base, 1e-4),
            "vx": base,
            "vy": base,
            "vz": base,
            "ne": np.full_like(base, 1e16),
        }
        args = types.SimpleNamespace(
            upsample_factors={},
            upsample_target_shape={},
            upsample_target_spacing_m={"x": 2.0, "y": 3.0, "z": 25.0},
            interpolation_default={"x": "linear", "y": "linear", "z": "linear"},
            interpolation_channels={},
            log_interpolation_channels={
                "temperature", "rho", "ne", "pgas", "hydrogen_populations"
            },
            continuity_max_relative_increase={},
        )
        result, height, dx, dy = self.resample(
            args, fields, np.array([100.0, 0.0]), 4.0, 6.0
        )

        self.assertEqual(result["temperature"].shape, (5, 3, 5))
        self.assertEqual(dx, 2.0)
        self.assertEqual(dy, 3.0)
        np.testing.assert_allclose(np.abs(np.diff(height)), 25.0)

    def test_positivity_and_continuity_are_enforced(self):
        good = np.ones((2, 1, 1), dtype=np.float32)
        fields = {
            "temperature": good * 5000,
            "rho": good * 1e-4,
            "vx": good,
            "vy": good,
            "vz": good,
            "ne": good * 1e16,
            "pgas": good * 1e4,
            "hydrogen_populations": np.ones((2, 1, 1, 6), dtype=np.float32),
        }
        args = types.SimpleNamespace(
            positive_channels=("temperature", "rho", "ne", "pgas"),
        )
        self.validate(args, fields)
        fields["rho"][1] = 0.0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.validate(args, fields)

    def test_temperature_is_floored_before_eos_uses_interpolated_pgas(self):
        args = types.SimpleNamespace(temperature_floor_k=3250.0)
        pgas = np.array([1.0, 2.0], dtype=np.float32)
        fields = {
            "temperature": np.array([3000.0, 4000.0], dtype=np.float32),
            "pgas": pgas,
        }

        def fake_eos(_args, temperature, pressure):
            np.testing.assert_array_equal(temperature, [3250.0, 4000.0])
            self.assertIs(pressure, pgas)
            return np.array([1e15, 2e15], dtype=np.float32)

        with mock.patch.object(
            self.converter, "_electron_density_from_witt_pgas", side_effect=fake_eos
        ) as eos:
            ne = self.converter._apply_temperature_floor_and_calculate_ne(
                args, "lgp", fields
            )

        eos.assert_called_once()
        np.testing.assert_array_equal(fields["temperature"], [3250.0, 4000.0])
        self.assertIs(fields["ne"], ne)


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for converter config tests")
class ConverterConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        from convert_iris_sim_fits_to_ffno_hdf5 import load_config

        cls.load_config = staticmethod(load_config)

    def test_toml_supplies_crop_resampling_channel_methods_and_old_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "converter.toml"
            config_path.write_text(
                """
[input]
folder = "fits"
simulation_name = "test"
snap = 42

[selection.x]
start = 2
stop = 10
[selection.y]
start = 3
stop = 12
step = 2

[eos]
backend = "python"
workers = 2

[output]
hdf5 = "result.hdf5"
overwrite = true

[resampling.target_spacing_m]
x = 2000.0
y = 3000.0
z = 4000.0
[resampling.default]
x = "linear"
y = "linear"
z = "cubic"
[resampling.channels.temperature]
z = "linear"
[resampling.channels.rho]
x = "cubic"
[resampling.log_space]
pgas = false

[validation]
positive_channels = ["temperature", "rho", "ne"]
[validation.continuity_max_relative_increase.rho]
x = 0.1
y = 0.1
z = 0.2
""",
                encoding="utf-8",
            )
            args = self.load_config(config_path)

            self.assertEqual(args.x_slice, slice(2, 10, None))
            self.assertEqual(args.y_slice, slice(3, 12, 2))
            self.assertEqual(
                args.upsample_target_spacing_m,
                {"x": 2000.0, "y": 3000.0, "z": 4000.0},
            )
            self.assertEqual(args.interpolation_channels["rho"]["x"], "cubic")
            self.assertNotIn("pgas", args.log_interpolation_channels)
            self.assertIn("rho", args.log_interpolation_channels)
            self.assertEqual(
                args.continuity_max_relative_increase["rho"]["z"], 0.2
            )
            self.assertEqual(args.eos_backend, "python")
            self.assertEqual(args.eos_workers, 2)
            self.assertTrue(args.overwrite)
            self.assertEqual(Path(args.output), (Path(tmpdir) / "result.hdf5").resolve())


@unittest.skipUnless(HAS_NUMPY, "NumPy is required to import the FITS converter")
class HydrogenPopulationSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        from convert_iris_sim_fits_to_ffno_hdf5 import (
            _find_hydrogen_population_paths,
        )

        cls.find_paths = staticmethod(_find_hydrogen_population_paths)

    def quantity_path(self, folder, level):
        return Path(folder) / f"BIFROST_en024048_hion_lgn{level}_385.fits"

    def test_complete_lgn1_through_lgn6_set_is_detected_in_level_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = tuple(self.quantity_path(tmpdir, level) for level in range(1, 7))
            for path in expected:
                path.touch()

            actual = self.find_paths(
                Path(tmpdir), "BIFROST", "en024048_hion", "385"
            )

            self.assertEqual(actual, expected)

    def test_incomplete_population_set_is_not_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for level in range(1, 6):
                self.quantity_path(tmpdir, level).touch()

            self.assertIsNone(
                self.find_paths(
                    Path(tmpdir), "BIFROST", "en024048_hion", "385"
                )
            )


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for Multi3D writer tests")
class Multi3dHydrogenWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        import convert_iris_sim_fits_to_ffno_hdf5 as converter

        cls.converter = converter

    def test_complete_populations_enable_and_fill_multi3d_nh(self):
        created = []

        class FakeMulti3dAtmos:
            def __init__(self, _path, nx, ny, nz, **kwargs):
                self.read_nh = kwargs["read_nh"]
                shape = (nx, ny, nz)
                self.ne = np.empty(shape, dtype=np.float32)
                self.temp = np.empty(shape, dtype=np.float32)
                self.vx = np.empty(shape, dtype=np.float32)
                self.vy = np.empty(shape, dtype=np.float32)
                self.vz = np.empty(shape, dtype=np.float32)
                self.rho = np.empty(shape, dtype=np.float32)
                self.nh = np.empty((*shape, 6), dtype=np.float32)
                created.append(self)

        multi3d_module = types.ModuleType("helita.sim.multi3d")
        multi3d_module.Multi3dAtmos = FakeMulti3dAtmos
        sim_module = types.ModuleType("helita.sim")
        sim_module.multi3d = multi3d_module
        helita_module = types.ModuleType("helita")
        helita_module.sim = sim_module
        fake_modules = {
            "helita": helita_module,
            "helita.sim": sim_module,
            "helita.sim.multi3d": multi3d_module,
        }

        shape = (2, 3, 2)
        scalar = np.ones(shape, dtype=np.float32)
        nh = np.arange(np.prod((*shape, 6)), dtype=np.float32).reshape(
            *shape, 6
        )
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            sys.modules, fake_modules
        ), mock.patch.object(
            self.converter, "_import_astropy_units", return_value=mock.MagicMock()
        ), mock.patch.object(
            self.converter,
            "_convert_values",
            side_effect=lambda values, _source, _target: values,
        ):
            self.converter._write_multi3d_atmosphere(
                Path(tmpdir) / "atm3d",
                None,
                temp=scalar,
                rho=scalar,
                vx=scalar,
                vy=scalar,
                vz=scalar,
                ne=scalar,
                nh=nh,
                dx_m=1.0,
                dy_m=1.0,
                height_m=np.array([1.0, 0.0], dtype=np.float32),
                overwrite=False,
            )

        self.assertTrue(created[0].read_nh)
        np.testing.assert_array_equal(created[0].nh, nh)


@unittest.skipUnless(
    HAS_NUMPY and HAS_ASTROPY,
    "NumPy and Astropy are required for FITS converter unit tests",
)
class FitsHeaderUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        from astropy import units as u
        from convert_iris_sim_fits_to_ffno_hdf5 import (
            _convert_values,
            _unit_scale_from_header,
        )

        cls.unit_scale = staticmethod(_unit_scale_from_header)
        cls.convert_values = staticmethod(_convert_values)
        cls.u = u

    def test_units_from_supplied_muram_headers_are_converted_to_si(self):
        self.assertEqual(
            self.unit_scale({"BUNIT": "K"}, "BUNIT", "K", "temperature"),
            1.0,
        )
        self.assertEqual(
            self.unit_scale(
                {"BUNIT": "kg m^(-3)"}, "BUNIT", "kg / m3", "mass density"
            ),
            1.0,
        )
        self.assertEqual(
            self.unit_scale(
                {"BUNIT": "m s^(-1)"}, "BUNIT", "m / s", "velocity"
            ),
            1.0,
        )
        self.assertEqual(
            self.unit_scale(
                {"BUNIT": "N m^(-2)"}, "BUNIT", "Pa", "gas pressure"
            ),
            1.0,
        )
        self.assertEqual(
            self.unit_scale({"CUNIT3": "Mm"}, "CUNIT3", "m", "length"),
            1e6,
        )

    def test_common_cgs_header_units_are_converted_to_si(self):
        self.assertAlmostEqual(
            self.unit_scale(
                {"BUNIT": "g cm^(-3)"}, "BUNIT", "kg / m3", "mass density"
            ),
            1e3,
        )
        self.assertAlmostEqual(
            self.unit_scale(
                {"BUNIT": "dyn cm^(-2)"}, "BUNIT", "Pa", "gas pressure"
            ),
            1e-1,
        )
        self.assertAlmostEqual(
            self.unit_scale(
                {"BUNIT": "cm^(-3)"}, "BUNIT", "1 / m3", "electron density"
            ),
            1e6,
        )

    def test_missing_or_unsupported_header_unit_fails(self):
        with self.assertRaises(ValueError):
            self.unit_scale({}, "BUNIT", "m / s", "velocity")
        with self.assertRaises(ValueError):
            self.unit_scale(
                {"BUNIT": "not_a_physical_unit"}, "BUNIT", "m / s", "velocity"
            )

    def test_multi3d_conversions_are_derived_from_astropy_units(self):
        u = self.u
        np.testing.assert_allclose(
            self.convert_values(np.array([1.0]), u.m / u.s, u.km / u.s),
            [1e-3],
        )
        np.testing.assert_allclose(
            self.convert_values(np.array([1.0]), u.m**-3, u.cm**-3),
            [1e-6],
        )
        np.testing.assert_allclose(
            self.convert_values(np.array([1.0]), u.kg / u.m**3, u.g / u.cm**3),
            [1e-3],
        )
        np.testing.assert_allclose(
            self.convert_values(np.array([1.0]), u.m, u.cm),
            [1e2],
        )


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for coordinate tests")
class TargetCoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        from convert_iris_sim_fits_to_ffno_hdf5 import _reverse_to_target_coordinates

        cls.reverse = staticmethod(_reverse_to_target_coordinates)

    def test_spatial_planes_rotate_and_depth_reverses(self):
        base = np.arange(1, 13, dtype=np.float32).reshape(2, 3, 2)
        temp, rho, vx, vy, vz, ne, height = self.reverse(
            base,
            base + 10,
            base + 20,
            base + 30,
            base + 40,
            base + 50,
            np.array([100.0, 200.0]),
        )

        expected = np.stack(
            [base[:, :, depth][::-1, :].T for depth in (1, 0)], axis=-1
        )
        np.testing.assert_array_equal(temp, expected)
        np.testing.assert_array_equal(rho, expected + 10)
        np.testing.assert_array_equal(vx, expected + 20)
        np.testing.assert_array_equal(vy, expected + 30)
        np.testing.assert_array_equal(vz, expected + 40)
        np.testing.assert_array_equal(ne, expected + 50)
        np.testing.assert_array_equal(height, [200.0, 100.0])


@unittest.skipUnless(
    HAS_NUMPY and HAS_ASTROPY and HAS_H5PY,
    "NumPy, Astropy, and h5py are required for FFNO HDF5 tests",
)
class FFNOHDF5WriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        from convert_iris_sim_fits_to_ffno_hdf5 import _write_ffno_hdf5

        cls.write_hdf5 = staticmethod(_write_ffno_hdf5)

    def test_velocity_channels_are_written_in_km_per_second(self):
        import h5py

        shape = (1, 1, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "solving.hdf5"
            self.write_hdf5(
                output,
                temp=np.full(shape, 5000.0, dtype=np.float32),
                rho=np.full(shape, 1e-4, dtype=np.float32),
                vx=np.full(shape, 1000.0, dtype=np.float32),
                vy=np.full(shape, -2000.0, dtype=np.float32),
                vz=np.full(shape, 3000.0, dtype=np.float32),
                ne=np.full(shape, 1e16, dtype=np.float32),
                height_m=np.array([2e6, 1e6], dtype=np.float32),
                dx_m=192000.0,
                dy_m=192000.0,
                source_folder=Path(tmpdir),
                simulation_code="MURaM",
                simulation_name="test",
                snap="1",
                electron_density_source="lgne",
                compression=1,
                overwrite=False,
            )

            with h5py.File(output, "r") as f:
                np.testing.assert_allclose(f["inputs"][0, 1], 1.0)
                np.testing.assert_allclose(f["inputs"][0, 2], -2.0)
                np.testing.assert_allclose(f["inputs"][0, 3], 3.0)
                self.assertEqual(f.attrs["velocity_unit"], "km s^-1")


if __name__ == "__main__":
    unittest.main()

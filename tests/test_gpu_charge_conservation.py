import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DirectGpuChargeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.python_source = (ROOT / "FFNONet.py").read_text()
        cls.julia_source = (ROOT / "Forward.jl").read_text()
        ast.parse(cls.python_source)

    def test_python_uses_direct_saha_equations_without_lookup_tables(self):
        self.assertIn("def direct_saha_charge_electron_density_gpu(", self.python_source)
        self.assertIn("torch.exp(", self.python_source)
        self.assertIn("saha_factor", self.python_source)
        self.assertIn("dtype=torch.float64", self.python_source)
        self.assertNotIn("charge_lookup_table", self.python_source)

    def test_julia_exports_complete_atomic_model(self):
        for dataset in (
            'group["fixed_hydrogen_density"]',
            'group["atom_offsets"]',
            'group["energies_joule"]',
            'group["statistical_weights"]',
            'group["stages"]',
            'group["abundances"]',
            'group["hydrogen_prediction_indices"]',
            'group["hydrogen_stages"]',
        ):
            self.assertIn(dataset, self.julia_source)

    def test_charge_only_reads_gpu_result_and_se_retains_cpu_fallback(self):
        self.assertIn("read_gpu_charge_electron_density(", self.julia_source)
        self.assertIn("if consistency_mode == :charge_only", self.julia_source)
        self.assertIn("charge_conservation_electron_density(", self.julia_source)
        self.assertIn('backend=consistency_mode == :charge_only ? "gpu" : "cpu"', self.julia_source)

    def test_prediction_writes_gpu_electron_density(self):
        self.assertIn('"electron_density",', self.python_source)
        self.assertIn('"direct_saha_boltzmann_gpu"', self.python_source)

    def test_runtime_compares_gpu_results_with_muspel_samples(self):
        self.assertIn("validate_gpu_charge_samples(", self.julia_source)
        self.assertIn("GPU_CHARGE_VALIDATION_SAMPLES = 64", self.julia_source)
        self.assertIn("GPU_CHARGE_VALIDATION_TOLERANCE = 5e-5", self.julia_source)
        self.assertIn('"gpu_charge_validation_complete"', self.julia_source)


if __name__ == "__main__":
    unittest.main()

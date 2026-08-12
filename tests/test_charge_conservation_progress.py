import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ChargeConservationProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "Forward.jl").read_text()

    def test_progress_counts_chunks_instead_of_individual_cells(self):
        self.assertIn("progress_chunk_elements::Int=100", self.source)
        self.assertIn(
            "Threads.atomic_add!(completed_elements, chunk_elements)",
            self.source,
        )
        self.assertIn('"charge_conservation_progress"', self.source)

    def test_progress_reports_percentage_rate_and_eta(self):
        self.assertIn('"  charge conservation: $(percentage)% | "', self.source)
        self.assertIn('"ETA $(compact_duration(progress.eta_s)) | "', self.source)
        self.assertIn("cells_per_second", self.source)


if __name__ == "__main__":
    unittest.main()

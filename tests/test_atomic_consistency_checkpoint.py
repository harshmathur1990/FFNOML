import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def julia_function_source(name):
    source = (ROOT / "Forward.jl").read_text()
    start = source.index(f"function {name}(")
    next_function = source.find("\nfunction ", start + 1)
    return source[start:] if next_function == -1 else source[start:next_function]


class AtomicConsistencyCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "Forward.jl").read_text()

    def test_staged_file_is_synced_and_validated_before_atomic_rename(self):
        function = julia_function_source("atomic_consistency_update!")

        copy = function.index("cp(result_h5, staging_h5; force=false)")
        update = function.index("operation(staging_h5)", copy)
        sync = function.index("fsync_file(staging_h5)", update)
        validate = function.index("validate_consistency_checkpoint(", sync)
        publish = function.index("atomic_replace_file(staging_h5, result_h5)", validate)
        sync_directory = function.index("fsync_directory(", publish)

        self.assertLess(copy, update)
        self.assertLess(update, sync)
        self.assertLess(sync, validate)
        self.assertLess(validate, publish)
        self.assertLess(publish, sync_directory)

    def test_publication_uses_direct_atomic_rename(self):
        function = julia_function_source("atomic_replace_file")

        self.assertIn("ccall(:rename", function)
        self.assertNotIn("mv(", function)

    def test_iteration_writes_only_to_staging_file(self):
        function = julia_function_source("write_consistency_checkpoint!")

        self.assertIn("atomic_consistency_update!", function)
        self.assertIn('h5open(staging_h5, "r+")', function)
        self.assertIn(
            "write_solving_electron_density!(staging_h5, electron_density)",
            function,
        )
        self.assertNotIn('h5open(result_h5, "r+")', function)
        self.assertLess(
            function.index("atomic_consistency_update!"),
            function.index("rm(work_prediction_h5)"),
        )

    def test_timing_update_is_inside_atomic_publication(self):
        function = julia_function_source("write_consistency_checkpoint!")

        self.assertIn("finalize_consistency_checkpoint_timings!(", function)
        self.assertIn("staging_h5,", function)

    def test_publication_event_precedes_next_distributed_launch(self):
        function = julia_function_source("predict_with_charge_conservation")

        initial_publish = function.index("atomic_consistency_update!(")
        initial_barrier = function.index("parallel_barrier(parallel)", initial_publish)
        launcher = function.index("call_fsdppredict_collective!(", initial_barrier)
        self.assertLess(initial_publish, initial_barrier)
        self.assertLess(initial_barrier, launcher)

        checkpoint = function.index("write_consistency_checkpoint!(", launcher)
        iteration_barrier = function.index("parallel_barrier(parallel)", checkpoint)
        self.assertLess(checkpoint, iteration_barrier)


if __name__ == "__main__":
    unittest.main()

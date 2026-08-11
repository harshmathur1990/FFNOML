import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def julia_function_source(name):
    source = (ROOT / "Forward.jl").read_text()
    start = source.index(f"function {name}(")
    next_function = source.find("\nfunction ", start + 1)
    return source[start:] if next_function == -1 else source[start:next_function]


class ForwardLauncherHandshakeTests(unittest.TestCase):
    def test_nested_launcher_does_not_overlap_an_outer_mpi_collective(self):
        function = julia_function_source("call_fsdppredict_collective!")

        self.assertNotIn("parallel_root_call", function)
        prelaunch_barrier = function.index("fsdppredict_prelaunch_barrier_complete")
        launcher = function.index("call_fsdppredict!", prelaunch_barrier)
        status_received = function.index("fsdppredict_status_received", launcher)
        postlaunch_barrier = function.index(
            "fsdppredict_postlaunch_barrier_start",
            status_received,
        )

        self.assertLess(prelaunch_barrier, launcher)
        self.assertLess(launcher, status_received)
        self.assertLess(status_received, postlaunch_barrier)

    def test_launcher_status_is_event_driven_and_has_a_timeout(self):
        function = julia_function_source("call_fsdppredict_collective!")

        self.assertIn('listen(ip"0.0.0.0", 0)', function)
        self.assertIn("serialize(socket, (success=success, message=message))", function)
        self.assertIn("status = deserialize(control_socket)", function)
        self.assertIn("FORWARD_FSDP_STATUS_TIMEOUT", function)
        self.assertIn('get(ENV, "FORWARD_FSDP_STATUS_TIMEOUT", "0")', function)
        self.assertIn("if timeout_seconds > 0", function)
        self.assertNotIn("sleep(", function)
        self.assertNotIn("while !isfile", function)


if __name__ == "__main__":
    unittest.main()

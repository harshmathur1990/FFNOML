"""Test the real case runner with simulated Slurm/timeout outcomes."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "FFNOInversion.jl/scripts/run_olivia_runtime_tests.sbatch"


class OliviaCaseOutcomeTests(unittest.TestCase):
    def run_case(self, expectation, status, marker=True, watchdog=False,
                 cleanup=True, command_diagnostic=False):
        source = BATCH.read_text()
        function = "run_case() {" + source.split("run_case() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            timeout = work / "timeout"
            # Simulate only GNU timeout's watchdog decision; execute the real
            # wrapper and child to verify stdout/stderr separation as well.
            timeout.write_text('''#!/bin/bash
while [[ "$1" == --* ]]; do shift; done
shift
if [[ "$TEST_WATCHDOG" == 1 ]]; then
    echo "timeout: sending signal TERM to command bash" >&2
fi
exec "$@"
''')
            timeout.chmod(0o755)
            command = work / "command"
            command.write_text(f'''#!/bin/bash
{ 'echo expected_event >&2' if marker else ':' }
{ 'echo "timeout: sending signal TERM to command bash" >&2' if command_diagnostic else ':' }
exit {status}
''')
            command.chmod(0o755)
            harness = '''set -eu
snapshot_scheduler() { :; }
sample_case() { :; }
capture_node_health() { :; }
cleanup_numeric_steps() { return "$TEST_CLEANUP_STATUS"; }
wait_for_interconnect_recovery() { return 0; }
diagnostics_root=$TEST_DIRECTORY
summary=$TEST_DIRECTORY/summary.tsv
failures=0
''' + function + '''
run_case probe "$TEST_EXPECTATION" 90 expected_event terminal "$TEST_DIRECTORY/command"
'''
            env = dict(os.environ, PATH=f"{work}:{os.environ['PATH']}",
                       TEST_DIRECTORY=str(work), TEST_EXPECTATION=expectation,
                       TEST_WATCHDOG=str(int(watchdog)),
                       TEST_CLEANUP_STATUS="0" if cleanup else "1")
            result = subprocess.run(["bash", "-c", harness], env=env,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            return (work / "summary.tsv").read_text().split("\t")[2]

    def test_slurm_kill_after_internal_timeout_passes(self):
        self.assertEqual(self.run_case("controlled_failure", 137), "PASS")

    def test_watchdog_kill_cannot_pass_internal_timeout(self):
        for status in (124, 137):
            with self.subTest(status=status):
                self.assertEqual(self.run_case("controlled_failure", status, watchdog=True), "FAIL")

    def test_signal_exit_requires_internal_marker_and_cleanup(self):
        self.assertEqual(self.run_case("controlled_failure", 137, marker=False), "FAIL")
        self.assertEqual(self.run_case("controlled_failure", 137, cleanup=False), "FAIL")

    def test_external_timeout_requires_watchdog_and_stall_marker(self):
        for status in (124, 137):
            with self.subTest(status=status):
                self.assertEqual(self.run_case("timeout", status, watchdog=True), "PASS")
                self.assertEqual(self.run_case("timeout", status), "FAIL")
        self.assertEqual(self.run_case("timeout", 137, watchdog=True, marker=False), "FAIL")

    def test_command_stderr_cannot_impersonate_watchdog(self):
        self.assertEqual(self.run_case("controlled_failure", 137, command_diagnostic=True), "PASS")

    def test_ordinary_outcomes(self):
        self.assertEqual(self.run_case("success", 0), "PASS")
        self.assertEqual(self.run_case("success", 0, marker=False), "FAIL")
        self.assertEqual(self.run_case("success", 0, watchdog=True), "FAIL")
        self.assertEqual(self.run_case("controlled_failure", 1), "PASS")
        self.assertEqual(self.run_case("controlled_failure", 0), "FAIL")


if __name__ == "__main__":
    unittest.main()

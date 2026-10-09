"""Exercise the nested Slurm launcher without requiring Slurm or GPUs."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "FFNOInversion.jl/scripts/ffno_fsdp_service.sh"


class FSDPServiceLauncherTests(unittest.TestCase):
    def test_static_rendezvous_preserves_host_order_with_reversed_slurm_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)

            def executable(name, source):
                path = work / name
                path.write_text("#!/bin/bash\nset -eu\n" + source)
                path.chmod(0o755)
                return path

            executable("scontrol", "printf 'gpu-1-73\\ngpu-1-106\\n'\n")
            executable("hostname", 'printf "%s\\n" "$TEST_HOST"\n')
            executable("srun", 'while [[ "$1" != env ]]; do shift; done\nexec "$@"\n')
            python = executable("capture-python", 'printf "%s\\n" "$@" > "$TEST_ARGS"\n')
            manifest = work / "manifest.toml"
            manifest.write_text("")
            captured = work / "arguments.txt"
            environment = dict(os.environ,
                PATH=f"{work}:{os.environ['PATH']}",
                SLURM_JOB_ID="2290846", SLURM_NNODES="2",
                SLURM_JOB_NODELIST="gpu-1-[73,106]",
                OLIVIA_REPO_DIR=str(ROOT / "FFNOInversion.jl"),
                OLIVIA_PYTHON=str(python), TEST_ARGS=str(captured),
                FFNO_FSDP_GPUS_PER_NODE="4")

            # The second host launches first and receives Slurm task rank 0.
            for host, task_rank, node_rank in [("gpu-1-106", "0", "1"),
                                                ("gpu-1-73", "1", "0")]:
                with self.subTest(host=host):
                    result = subprocess.run(
                        ["bash", str(LAUNCHER), str(manifest)],
                        env=dict(environment, TEST_HOST=host, SLURM_PROCID=task_rank),
                        capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    arguments = captured.read_text().splitlines()
                    self.assertEqual(arguments[:2], ["-m", "torch.distributed.run"])
                    options = dict(arg[2:].split("=", 1) for arg in arguments
                                   if arg.startswith("--"))
                    # Only static rendezvous honors the supplied node rank.
                    self.assertEqual(options["rdzv_backend"], "static")
                    self.assertEqual(options["node_rank"], node_rank)
                    self.assertEqual(options["rdzv_endpoint"], "gpu-1-73:36846")
                    self.assertEqual(options["nnodes"], "2")
                    self.assertEqual(options["nproc_per_node"], "4")
                    self.assertEqual(Path(arguments[-1]), manifest.resolve())


if __name__ == "__main__":
    unittest.main()

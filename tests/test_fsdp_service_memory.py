"""The FSDP service's MEMORY command: every GPU rank's peak allocated, reserved and total memory, gathered on rank 0 and
sent to Julia as one line per rank (host, local rank, bytes). Needs CUDA (one GPU, one rank)."""

import os
from pathlib import Path
import socket
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

CUDA = torch.cuda.is_available()


class _Diagnostics:
    def record(self, *args, **kwargs):
        pass

    def record_memory(self, *args, **kwargs):
        pass

    def record_array(self, *args, **kwargs):
        pass

    def record_libraries(self, *args, **kwargs):
        pass


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@unittest.skipUnless(CUDA, "the FSDP service runs on CUDA devices")
class MemoryCommandTest(unittest.TestCase):
    def test_memory_reports_each_rank_and_resets_the_peak(self):
        import torch.distributed as dist
        import ffno_fsdp_service as service

        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=1)
        try:
            device = torch.device("cuda", 0)
            torch.zeros(1, device=device)                  # initialize the device, as the service's model load does
            torch.cuda.reset_peak_memory_stats(device)
            block = torch.empty(64 * 2**20, dtype=torch.uint8, device=device)        # 64 MiB at the peak
            entry = service._memory_entry(device, 0)
            self.assertEqual(entry[0], socket.gethostname())
            self.assertEqual(entry[1], 0)
            self.assertGreaterEqual(entry[2], 64 * 2**20)
            self.assertGreaterEqual(entry[3], entry[2] - 2**20)
            self.assertGreater(entry[4], entry[3])
            del block

            port = _free_port()
            configuration = {"service": {"port": port, "token": "t0k", "bind_host": "127.0.0.1"}}
            replies = []

            def client():
                for _ in range(200):
                    try:
                        connection = socket.create_connection(("127.0.0.1", port))
                        break
                    except OSError:
                        threading.Event().wait(0.05)
                reader = connection.makefile("rb")
                connection.sendall(b"HELLO t0k\n")
                replies.append(reader.readline().decode())
                for _ in range(2):                      # the second report starts from the reset peak
                    connection.sendall(b"MEMORY\n")
                    header = reader.readline().decode().split()
                    replies.append((header, [reader.readline().decode().split() for _ in range(int(header[2]))]))
                connection.sendall(b"SHUTDOWN\n")
                replies.append(reader.readline().decode())
                connection.close()

            thread = threading.Thread(target=client)
            thread.start()
            service.serve(configuration, {}, None, _Diagnostics())
            thread.join()

            self.assertTrue(replies[0].startswith("READY "))
            (header, rows), (_, second) = replies[1], replies[2]
            self.assertEqual(header, ["OK", "MEMORY", "1"])
            self.assertEqual(len(rows), 1)
            host, local_rank, peak, reserved, total = rows[0]
            self.assertEqual((host, local_rank), (socket.gethostname(), "0"))
            self.assertGreaterEqual(int(peak), 64 * 2**20)
            self.assertGreater(int(total), int(reserved))
            self.assertLess(int(second[0][2]), 64 * 2**20)          # the 64 MiB block is gone and the peak was reset
            self.assertEqual(replies[3], "BYE\n")
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()

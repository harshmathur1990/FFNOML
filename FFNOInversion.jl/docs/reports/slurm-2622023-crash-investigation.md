# Job 2622023 crash investigation

Inspected 25 October 2026: copied stdout/stderr and `diagnostics/slurm-2622023/`, plus current source (`7f1ab1c`). The exact submitted checkout and run-directory factory/config were not independently recovered; source explanations below agree with the logged stack traces. No cluster commands, reruns, or code changes were performed.

## Finding

The job failed during its initial distributed FFNO population prediction. Python GPU-service workers exited with SIGSEGV. Julia then received a truncated service response; cleanup raised EPIPE and obscured the prediction failure in the top-level exception. The logs do not identify the native instruction/library responsible for the segfaults.

HSEMHS installation succeeded: stderr lines 14–17 report HSEMHS, FFNOInversion and its Muspel extension successfully precompiled. Stdout reports `FFNO_OLIVIA_JULIA_ENVIRONMENT_READY`. The model-factory stack reaches population prediction after equilibrium reconstruction; no HSEMHS exception is recorded.

## Evidence and sequence (Europe/Oslo, CET)

1. **15:39:07:** allocation starts on 8 nodes. Runtime setup builds and loads the ARM aarch64 Wittmann library and instantiates/precompiles the Julia environment.
2. The GPU service launches **32 Python GPU ranks**, 4 per node. `ffno-fsdp-service-1261006.log:2765` records `FFNO_FSDP_SERVICE_LISTENING ... world=32 models=CA,H`; line 2766 records the connected Julia client. This excludes a failure to start or connect to the service.
3. Population inference enters the model. Periodic Python stack dumps include `predict_local`, `_forward_linear`, `_forward_balanced_vertical`, convolutions, group normalization, and z-feature construction. These are sampled stacks, not proof of the crashing native operation. The printed `Timeout (0:01:00)!` headers come from the repeating `faulthandler.dump_traceback_later` diagnostic in `ffno_fsdp_service.py`, not necessarily a fatal request timeout.
4. **15:42:36:** torchrun reports SIGSEGV on GPU-service rank **10**, local rank 2, PID 1573876, host **gpu-1-16**; rank 11 on that host also segfaults (service log lines 7783–7817). Another node's first-observed report lists rank **20** on **gpu-1-66** at the same second (lines 8007–8041). The merged log does not establish a unique first rank across hosts. Further SIGSEGV reports include ranks 26/27 on gpu-1-79 and rank 16 on gpu-1-52.
5. **15:42:39:** rank 4 reports NCCL `NET/OFI ... PTLTE_NOT_FOUND` and failure of all-reduce sequence 108 (8 elements), after the first reported segfaults (lines 7820 onward). This is consistent with peer loss during teardown; it does not establish OFI as the initiating cause.
6. **15:42:47:** Julia's prediction control call returns an error: `truncated FSDP service response` (`slurm-2622023.err:102` onward; rank-0 event log also records it).
7. **15:42:50:** cleanup tries to send `SHUTDOWN` to the dead service and raises `IOError: write: broken pipe (EPIPE)`. This is the exception shown first in stderr. `build_production_model` catches the prediction exception and calls `close_distributed_population_model!`; that cleanup exception supersedes its planned `rethrow()`.
8. **15:43:24:** job ends with exit 137 and forced termination of remaining Julia tasks. This final SIGKILL is not evidence that memory exhaustion initiated the failure. No explicit CUDA OOM or Slurm OOM event appears in the inspected logs.

## Confirmed secondary error-handling defect

`FSDPService.jl:close_fsdp_service!` writes to the socket without handling a peer that has already died. Its `finally` closes the socket, but an exception skips subsequent process waiting and log-stream cleanup. The factory's catch block also skips `close_equilibrium!` if population shutdown throws.

Recommended fix: preserve the primary exception, record shutdown failures separately, and perform resource cleanup through nested `try/finally`. Make service closure idempotent and tolerant of already-dead peers, while keeping shutdown symmetric across MPI ranks. Add a dead-service test that checks primary-error preservation, log/process cleanup, and equilibrium closure. This improves diagnosis but does not fix the Python segfault.

## Next diagnostic run

1. Give each torchrun rank separate stdout/stderr and capture PID/host/rank, exact Python/PyTorch/CUDA/cuDNN/NCCL/OFI versions and loaded libraries. Enable core dumps subject to site limits; obtain a native backtrace for the first SIGSEGV. Current Python stacks alone cannot distinguish CUDA, cuDNN, NCCL/OFI, or another native extension.
2. Add per-rank progress records around input scatter, model blocks, prediction completion and output gather. Record input shape, strides, dtype and finite/positive checks. For one diagnostic run use `CUDA_LAUNCH_BLOCKING=1` to improve attribution of asynchronous GPU failures; do not assume it is a performance configuration or a fix.
3. Reproduce the same first prediction with a smaller allocation/control input, then increase ranks. Distinguish one-node from multi-node behavior. A controlled network-backend comparison can follow if native traces or reproducibility implicate OFI; do not label the later OFI error as the root cause solely from this log.
4. Rerun after primary-error preservation is fixed so a dead service cannot hide the triggering exception behind SHUTDOWN EPIPE.

Conclusion: setup and HSEMHS installation completed; the observed initiating failure is native SIGSEGV in the distributed Python inference service. Exact native cause remains unresolved pending rank-separated logs/core backtraces or a controlled reproduction.

## Core-dump addendum (25 October 2026)

Inspected the newly supplied repository-root `core` (approximately 3.5 GiB) with macOS LLDB and a read-only parser of ELF program headers and Linux notes. LLDB successfully loads its aarch64 thread registers and unwinds raw addresses. The matching Linux executable/shared libraries are not present locally, so function names and instruction disassembly could not be recovered.

### Process identity and fault

- ELF64 Linux ARM aarch64 core, Python process **PID 426597**, with **14 recorded threads**.
- This PID matches the service log at lines 7971–7976: **GPU rank 13, local rank 1, host gpu-1-36**, reported SIGSEGV at **15:42:42 CET**. This core is from a confirmed failed worker, not the earliest reported failure: rank 14 on the same node was reported at 15:42:36, alongside failures on other nodes.
- Faulting thread: **TID 427389**. Do not assume it is Python's main thread; its TID differs from the process PID.
- Linux `NT_SIGINFO`: signal **11**, `si_code = 1` (**SEGV_MAPERR**), fault address **0x00000000000f5584**.
- Program counter **0x000040001878eea8**, link register **0x000040001878eea0**, stack pointer **0x0000400343a3e7c0**. Register `x0 = 0x00000000000f557c`; the fault address is `x0 + 8`. This is consistent with an invalid pointer access, but without the instruction and symbols it does not identify the object or its origin.

### Native module attribution

The ELF `NT_FILE` mapping containing the faulting PC is:

```text
0x0000400018670000–0x0000400018ab0000, file offset 0
/cluster/software/NRIS/neoverse_v2/software/Python/3.12.3-GCCcore-13.3.0/lib/libpython3.12.so.1.0
```

LLDB recovered this raw stack for the faulting thread:

| Frame | PC | Mapped module / offset from mapping start |
| --- | --- | --- |
| 0 | `0x40001878eea8` | libpython3.12, `0x11eea8` |
| 1 | `0x40001878f128` | libpython3.12, `0x11f128` |
| 2 | `0x40001890f4f4` | libpython3.12, `0x29f4f4` |
| 3 | `0x4000188cad4c` | libpython3.12, `0x25ad4c` |
| 4 | `0x400018de6df0` | `/lib64/libc.so.6` |

The segment containing the fault PC stores only its first 65,536 bytes out of a mapped 4,456,448 bytes; the fault PC lies beyond those stored bytes. Thus LLDB's “read memory ... failed” for code addresses reflects omitted file-backed pages, not evidence that the instruction address itself was unmapped. ELF file mappings still identify the library. Mapping-relative offsets above should be resolved using the matching ELF load layout and binary, not an arbitrary local Python build.

### Updated interpretation

This core places the final fault in **CPython's shared library on a non-main OS thread**, rather than directly at an identified CUDA/NCCL instruction. It does **not** prove a CPython bug: misuse from native extensions, memory corruption, or a thread/GIL/lifetime error could surface there. It also does not prove the crash represents the initiating failure across all workers. Retain the distinction between module attribution and root cause.

The next most useful artifact is the **exact** `libpython3.12.so.1.0` above (plus debug symbols if available), the venv's resolved Python executable, and matching `/lib64/libc.so.6`. Run a native debugger on Olivia against the core where those paths exist, or copy the exact binaries and configure debugger path mappings locally. Capture `thread apply all bt full`, registers and disassembly around the PC. With Python's matching GDB helpers available, also capture Python stacks (`py-bt` on relevant threads). Preserve the core: another run's dump may overwrite a file named simply `core`.

Read-only inspection commands used locally:

```sh
file core
lldb --batch -c core -o 'process status' -o 'thread backtrace all'
lldb --batch -c core -o 'register read pc lr sp x0 x1 x2 x3 x29' -o 'thread info'
```

Core extraction strengthens the diagnosis of a native Python-service failure, but a symbolized backtrace remains necessary before choosing a corrective change to the inference runtime.

## Addendum: the faulthandler watchdog as the crashing code (Claude, 25 October 2026)

Combining the core with the service log points at the service's own periodic traceback dump rather than CUDA, NCCL or the model:

- **Timing.** NCCL initialises at 15:41:34. `main()` starts `faulthandler.dump_traceback_later(60, repeat=True)` right after `init_process_group`, which synchronizes all ranks. The first SIGSEGVs are at 15:42:36, at the first 60 s dump, on several nodes in the same second.
- **Count.** The log has exactly 32 `Timeout (0:01:00)!` headers, one per rank: every rank was in its first dump. Several dumps stop mid-line (`File Thread 0x"`). The torchrun SIGSEGV reports follow immediately, with no other output in between.
- **Stack shape.** The core's faulting thread is not the main thread. Its stack is four libpython frames entered directly from libc's thread start. That matches faulthandler's C watchdog thread (`faulthandler_thread` → `_Py_DumpTracebackThreads`). A Python `threading` thread would show eval-loop frames; NCCL and CUDA threads would show their own libraries.
- **No "Fatal Python error".** faulthandler's fatal-signal handler was enabled on every rank, yet none printed this message. That is consistent with the fault happening inside faulthandler's own dump.
- **What changed from earlier runs.** Job 2473982 made 960 periodic dumps without crashing. None of its stacks contained `<shim>` frames. In 2622023 the dumped stacks contain `<shim>` / `<interpreter trampoline>` frames from torch's dynamo eval-frame hook (`torch/_dynamo/eval_frame.py`, entered through the activation-checkpoint wrapper). The watchdog walks other threads' frames without the GIL. On Python 3.12.3 that is unsafe while those frames change, and it is especially fragile with shim frames.

**Confirmed (symbolized on Olivia, 25 October 2026)** with `addr2line -i -f -p` against the job's exact `libpython3.12.so.1.0`:

```text
0x11eea8  PyType_HasFeature (object.h:966)
          inlined by dump_frame (traceback.c:1248)
          inlined by dump_traceback (traceback.c:1322)
0x11f128  PyThreadState_Next (pystate.c:1968), inlined by _Py_DumpTracebackThreads (traceback.c:1439)
0x29f4f4  faulthandler_thread (faulthandler.c:582)
0x25ad4c  pythread_wrapper (thread_pthread.h:239)
```

The faulting thread is faulthandler's watchdog. `dump_frame` read the type (offset 8, so the fault address is `x0 + 8`) of an object taken from another thread's live frame and got a garbage pointer. The periodic dump caused the crash; CUDA, NCCL, the checkpoints and the model are not implicated. Earlier wording kept for the record:

This remained a hypothesis until it was symbolized. A symbolized backtrace of `libpython3.12.so.1.0+0x11eea8` would confirm it. On Olivia, `addr2line -f -e <libpython> 0x11eea8`, or `gdb` against the core, should resolve to a frame-dumping function in `Modules/faulthandler.c` or `Python/traceback.c`.

Changes made (`ffno_fsdp_service.py`, `ffno_fsdp_diagnostics.py`, `FSDPService.jl`, `Execution.jl`):

- Periodic dumps are now opt-in: `FFNO_FSDP_TRACEBACK_INTERVAL` defaults to 0. The fatal-signal handler stays on.
- Opt-in per-rank diagnostics (`FFNO_FSDP_DIAGNOSTICS`, `FFNO_FSDP_CORE_DUMPS`) and per-rank torchrun stdout/stderr.
- `FFNO_POPULATION_MODEL_ORDER` sets the order in which the population models are requested.
- `close_fsdp_service!` tolerates a dead service and is idempotent. `close_model_resources!` always closes both the population service and the equilibrium without replacing the error being handled.

Discriminating runs:

1. Rerun unchanged except for the new default (no periodic dump). The hypothesis predicts success.
2. Optionally, rerun with `FFNO_FSDP_TRACEBACK_INTERVAL=60`. The hypothesis predicts the crash again, at the first dump.
3. If run 1 still crashes, use `FFNO_FSDP_DIAGNOSTICS=1 FFNO_FSDP_CORE_DUMPS=1 CUDA_LAUNCH_BLOCKING=1`, and swap the model order.

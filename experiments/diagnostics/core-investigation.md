# Core investigation: job 2622023

The core belongs to Python PID 426597 (gpu-1-36, local GPU rank 1, global rank 13), running ffno_fsdp_service.py with diagnostics/slurm-2622023/ffno-fsdp-service-1261006.toml. It is an AArch64 ELF core, approximately 3.6 GiB apparent size.

## Confirmed crash

GDB reports SIGSEGV in watchdog thread LWP 427389:

```
PyType_HasFeature
dump_frame                         Python/traceback.c:1248
dump_traceback                     Python/traceback.c:1322
_Py_DumpTracebackThreads            Python/traceback.c:1438
faulthandler_thread                Modules/faulthandler.c:579
pythread_wrapper
```

The main thread (LWP 426597) was executing Python attribute access through Py_INCREF, _PyObject_GenericGetAttrWithDict, and _PyEval_EvalFrameDefault. The immediate crashing stack is Python traceback collection, not a CUDA kernel or NCCL operation.

ffno_fsdp_service.py:285–288 enables faulthandler and schedules repeating traceback dumps every 60 seconds by default. The service log starts printing `Timeout (0:01:00)!` at line 6736; these messages are periodic diagnostics, not evidence that an inference deadline expired. Interleaved stacks show workers inside FFNO forward inference. Several workers then exit with SIGSEGV around 15:42:36. NCCL OFI PTLTE_NOT_FOUND errors appear at 15:42:39, after those exits. Julia subsequently reports service connection failures and broken pipes, including failures while cleaning up the service.

## Interpretation and next diagnostic step

The repeating traceback watchdog is the leading immediate crash trigger. This core alone cannot establish whether invalid Python frame state originated in a CPython/PyTorch interaction or earlier memory corruption. Nor does it prove that inference will complete once traceback collection is disabled.

For the next controlled run, export `FFNO_FSDP_TRACEBACK_INTERVAL=0` into the GPU-service worker environment. The existing code supports this without a source change. Confirm that the launcher propagates the variable to torchrun workers. This disables periodic traceback traversal while retaining faulthandler.enable for fatal signals. If the crash persists, obtain a new core and inspect it with the compute-node AArch64 runtime libraries available.

## Debugger limitations and artifacts

Analysis was performed from an x86_64 host with GDB capable of reading this AArch64 core and Python symbols. Some compute-node system/CUDA libraries are missing or differ in build ID; most background native threads therefore cannot be reliably unwound. GDB's terminal `corrupt stack?` message is not by itself proof of stack corruption.

- core-threads.txt: all-thread native backtraces and thread identifiers.
- core-gdb.txt: initial backtrace and mapped ELF section details.
- slurm-2622023/ffno-fsdp-service-1261006.log: worker tracebacks and failure timeline.

No repository source or run configuration was changed, and no new job was submitted.

# The cost of the alternatives to one shared CUDA context

This file records experiment this experiment of the design (`../../docs/DESIGN.md`, not included in this archive) (section 1.1): what a GPU analytics engine pays on this machine when it gives up one shared CUDA context for the known alternatives. It covers five measurements:

1. The latency of one operator call in a separate per-tenant process.
2. The time and memory to give each tenant process its own copy of a shared table.
3. Throughput and latency when several tenants issue small kernels.
4. Fault containment between processes.
5. Two instrumentation baselines: compute-sanitizer memcheck and Guardian-style address fencing.

All runs used GPU 0 (RTX A6000) on 2026-09-26. The method section at the end lists the versions and the measurement rules. The operators are the Sonnet CUDA C++ `gather` (`out[i] = col[sel[i]]`, random `sel`) and `projection` (`out[i] = price[i] * (1 - discount[i])`) kernels from `eval/corpus/synthesized/sonnet/`, loaded from their `kernel.ptx` with the driver API. Row counts are 1e4, 1e6, and 1e8.

In the tables, a number is a median. A range in brackets is the minimum to maximum over trials, unless the table says otherwise. Times are in microseconds (us) or milliseconds (ms).

## Summary

| Alternative | Cost measured on this machine |
| --- | --- |
| Process per tenant, operator call by a pipe | +11 to +13 us per call (+109 us for the 16 ms gather). A 6.4 us call becomes 18 us. |
| Process per tenant, operator call by spinning on shared memory | +0.2 us per call, but one CPU core spins in each process |
| Process per tenant, each new intermediate buffer | 0.1 to 7 ms per buffer (export, import, and release) |
| Process per tenant, shared table by copy | 262 MiB of context and one copy of the table per tenant. For an 8 GiB table and 4 tenants: 33 GiB and 109 ms from device memory, or 1.36 s from host memory. An 8 GiB table for 8 tenants does not fit in 48 GiB. |
| Process per tenant, small kernels from N tenants | 22x (N = 2) to 86x (N = 8) lower throughput for the 1e4-row gather. About 13,000 kernels per second in total, whatever N is. |
| Process per tenant, fault containment | Works: other processes continue with correct results. They stall for about 142 ms while the driver handles the fault. |
| compute-sanitizer memcheck | Kernel time 9x to 39x |
| Guardian-style fencing | Kernel time +0% to +4% on these memory-bound kernels. Needs one power-of-two partition per tenant and cannot share a buffer between partitions. |

## 1. Operator call latency: same context against a per-tenant process

**Setup.** An engine process allocates the operator's three buffers and initializes them. A worker process (the tenant's process) maps the same buffers once at startup. An operator call is measured on the engine's host thread from the request to the completion of the kernel. There are three modes:

- Same context: the engine launches the kernel and synchronizes its stream.
- Pipe hand-off: the engine writes one byte to a pipe and blocks in `read`. The worker, blocked in `read`, launches the kernel, synchronizes, and writes one byte back.
- Shared-memory hand-off: both sides spin on sequence numbers in shared memory.

The buffers are VMM allocations (`cuMemCreate`), exported as POSIX file descriptors. The table under "Imported mapping speed" shows why the experiment does not use `cudaIpcOpenMemHandle` for them. Each size has 10 trials, and each trial has 1000 calls per mode (100 at 1e8) in rotating order. GPU kernel time is from a CUDA graph of 100 launches (10 at 1e8), timed with events.

| Operator | Rows | Kernel (GPU) | Same context | Pipe | Added by pipe | Shared memory | Added by shared memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gather | 1e4 | 1.34 | 6.46 | 17.95 | 11.48 [11.18 to 13.59] | 6.70 | 0.24 [0.19 to 0.39] |
| gather | 1e6 | 53.55 | 59.04 | 72.25 | 13.21 [13.17 to 13.27] | 59.25 | 0.22 [0.12 to 0.26] |
| gather | 1e8 | 15944 | 15950 | 16059 | 108.5 [13.8 to 115.1] | 15951 | 0.22 [-0.31 to 1.89] |
| projection | 1e4 | 1.20 | 6.29 | 17.80 | 11.49 [11.15 to 11.63] | 6.55 | 0.26 [0.23 to 0.32] |
| projection | 1e6 | 37.91 | 43.47 | 54.86 | 11.38 [11.30 to 11.52] | 43.03 | -0.45 [-0.48 to -0.37] |
| projection | 1e8 | 3501.6 | 3507.6 | 3520.6 | 13.40 [12.20 to 15.88] | 3507.0 | -0.50 [-1.46 to 0.38] |

All values are in us. "Added" is the per-trial difference from the same-context median. At 1e4 rows, the p99 is 7.0 us in the same context and 21.3 us through the pipe.

For the 16 ms gather, the pipe adds 109 to 115 us in 9 of 10 trials instead of 13 us. The shared-memory hand-off, where no process sleeps, adds under 2 us at every size, so the extra time comes from waking a process that slept for 16 ms. This experiment did not find out why that wake-up is slower.

**Per-buffer cost.** A per-tenant process must also map every new buffer that the engine gives it, such as an intermediate. Each value is the median of 30 new buffers of the operator's output size, in us:

| Buffer | CUDA IPC export | CUDA IPC import | CUDA IPC release | VMM export | VMM import and map | VMM unmap and release |
| --- | --- | --- | --- | --- | --- | --- |
| 80 KB | 5.0 | 52.2 | 36.6 | 671 | 85.6 | 40.3 |
| 8 MB | 5.1 | 110.1 | 51.2 | 684 | 111.2 | 50.8 |
| 800 MB | 14.6 | 5840 | 1224 | 780 | 308 | 102.5 |

CUDA IPC is `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, and `cudaIpcCloseMemHandle`. VMM is `cuMemExportToShareableHandle`, then `cuMemImportFromShareableHandle` with `cuMemMap`. These values come from the gather runs. The projection runs give values within 27% of them.

**Imported mapping speed.** A random gather over a buffer imported with `cudaIpcOpenMemHandle` can run much slower than the same gather over a local buffer. The table gives the GPU time of the gather (median of 5 launches, us) when its `col` input is one of four buffers:

- The worker's own `cudaMalloc` buffer.
- The engine's `cudaMalloc` buffer of the exact size, imported with CUDA IPC.
- The same, with the `cudaMalloc` size rounded up to a multiple of 2 MiB.
- The engine's VMM buffer, imported by file descriptor.

| Operator, rows (col size) | Own | CUDA IPC, exact size | CUDA IPC, size rounded to 2 MiB | VMM file descriptor |
| --- | --- | --- | --- | --- |
| gather, 1e4 (80 KB) | 5.1 | 5.1 | 5.0 | 5.0 |
| gather, 1e6 (8 MB) | 57.2 | 75.8 (1.32x) | 57.3 | 57.1 |
| gather, 1e7 (80 MB) | 1467 | 1468 | 1468 | 1467 |
| gather, 2e7 (160 MB) | 3073 | 23364 (7.6x) | 3076 | 3073 |
| gather, 3e7 (240 MB) | 4687 | 95657 (20.4x) | 4684 | 4687 |
| gather, 1e8 (800 MB) | 15953 | 1115920 (70x) | 15945 | 15941 |
| projection, 1e6 | 38.9 | 38.9 | 38.9 | 38.9 |
| projection, 1e8 | 3507 | 3507 | 3513 | 3505 |

Torch's CUDA IPC with the default allocator uses the same `cudaIpc*` functions (they are symbols in `libc10_cuda.so`). Its caching allocator requests segments in multiples of 2 MiB. A torch `index_select` gather over a column received through torch IPC ran as fast as over a local copy: 16286 us against 16275 us at 1e8 rows, and within 2% at 1e6, 1e7, and 3e7 rows. This experiment did not find the cause of the slowdown.

**Interpretation.** A per-tenant process adds about 11 to 13 us to every operator call through a pipe, which is almost twice the whole same-context call of a small operator, or 0.2 us through a spinning hand-off that uses one CPU core per process. It also adds 0.1 to 7 ms for every new buffer it must import.

## 2. Sharing a table between tenants

**What CUDA IPC exports.** The engine process uses torch 2.11 with the default caching allocator (`PYTORCH_CUDA_ALLOC_CONF` unset) and runs these steps:

1. It frees a 64 MiB tensor, so that the segment stays in torch's cache.
2. It allocates tenant A's tensor (16 MiB, values 1.0) and tenant B's tensor (16 MiB, values 7.0). `torch.cuda.memory_snapshot()` shows both in one 64 MiB segment.
3. It sends only A's tensor to a tenant process with `torch.multiprocessing`.

The tenant process calls `cuMemGetAddressRange` on the pointer it received. The driver's mapping is 64 MiB: the whole segment, not the 16 MiB tensor. The tenant searched the mapping and found all 4,194,304 of B's values. It overwrote them, and afterwards the engine found 0 of B's 4,194,304 values intact.

With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which is this machine's environment default, the import fails in the tenant process with `RuntimeError: pidfd_getfd: Operation not permitted`. Torch needs `pidfd_getfd` to share expandable segments, and Yama `ptrace_scope` is 1 on this machine. So with that setting, torch cannot share a tensor between processes at all.

**Copying the table into each tenant's process.** An engine process holds a table in device memory (a VMM allocation exported as a file descriptor) and in pinned host memory shared with the tenants. N tenant processes each allocate their own copy with `cudaMalloc`, then fill it at the same time as the others. There are two sources:

- Device to device: from the engine's device copy, through the imported mapping.
- Host to device: from the shared pinned host memory.

Each cell has 5 trials, which alternate between the two sources. Each tenant's copy was checked against the table after the last trial. The wall time runs from the start signal to the end of the last tenant's copy. Memory per tenant process is from `nvidia-smi`.

| Table | Tenants | Wall, device to device (ms) | Wall, host to device (ms) | Extra device memory |
| --- | --- | --- | --- | --- |
| 100 MiB | 2 | 0.73 [0.70 to 0.77] | 8.31 [8.31 to 8.31] | 724 MiB |
| 100 MiB | 4 | 1.54 [1.47 to 1.55] | 16.62 [16.61 to 16.62] | 1.4 GiB |
| 100 MiB | 8 | 3.08 [3.08 to 3.09] | 33.23 [33.23 to 33.23] | 2.8 GiB |
| 1 GiB | 2 | 6.74 [6.71 to 6.80] | 85.0 [85.0 to 85.1] | 2.5 GiB |
| 1 GiB | 4 | 13.54 [13.52 to 13.60] | 170.0 [170.0 to 170.1] | 5.0 GiB |
| 1 GiB | 8 | 27.20 [27.19 to 27.23] | 340.2 [340.2 to 340.2] | 10.0 GiB |
| 8 GiB | 2 | 54.67 [54.60 to 54.75] | 682.2 [682.2 to 682.2] | 16.5 GiB |
| 8 GiB | 4 | 109.36 [109.31 to 109.43] | 1363.3 [1363.3 to 1363.3] | 33.0 GiB |
| 8 GiB | 8 | not run: does not fit | not run | 66.0 GiB, plus the engine's 8.3 GiB, against 48 GiB |

Each tenant process uses the table's size plus 262 MiB of CUDA context: `nvidia-smi` shows 362, 1286, and 8454 MiB per process for the three table sizes. "Extra device memory" is N times (table + 262 MiB).

The aggregate rate stays constant as N grows: 272 to 319 GB/s device to device and 25.2 GB/s host to device. So the copies of different processes share the bandwidth and do not overlap in useful work, and the time grows linearly with N. In one context, two tenants read the same buffer, granted to both, at no copy time and no extra memory.

A table in its own allocation, outside the pool, can be mapped into each tenant process without a copy. The VMM mapping of part 1 does that, at the per-buffer cost that part 1 measured. The pool itself cannot be shared that way, because the mapping gives every tenant the whole pool.

**Interpretation.** Separate processes cannot share a pooled table without giving every tenant the whole pool. The alternative, a copy per tenant, costs one table plus 262 MiB of device memory per tenant, and 0.7 ms to 1.4 s of copy time per sharing event on this GPU.

## 3. Concurrency: N tenants issuing small kernels

**Setup.** Each tenant runs a closed loop: it launches the gather kernel on its own buffers, synchronizes, and records the latency. There are two modes:

- One context: one process with one host thread and one non-blocking stream per tenant.
- Process per tenant: one process, and so one context, per tenant, without MPS.

After a 0.3 s warmup of all tenants at the same time, each tenant runs for up to 1 s or 10,000 calls. Throughput counts the calls that complete in the interval when all tenants are active. The latency percentiles pool all tenants. Each cell has 3 trials, with the order of the two modes alternated. Throughput varied at most 2.3% across trials, and p50 at most 1.6%.

| Rows | N | One context: throughput (calls/s) | p50 (us) | p99 (us) | Process per tenant: throughput (calls/s) | p50 (us) | p99 (us) | Throughput ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e4 | 1 | 155,500 | 6.3 | 6.9 | | | | |
| 1e4 | 2 | 291,400 | 6.7 | 7.7 | 13,000 | 154.6 | 155.0 | 22.5 [22.4 to 22.7] |
| 1e4 | 4 | 580,800 | 6.7 | 8.5 | 12,900 | 310.6 | 311.6 | 45.3 [44.9 to 45.8] |
| 1e4 | 8 | 1,095,200 | 7.0 | 11.8 | 12,800 | 625.0 | 626.3 | 85.6 [85.5 to 85.8] |
| 1e6 | 1 | 16,900 | 59.0 | 59.9 | | | | |
| 1e6 | 2 | 18,300 | 109.1 | 112.8 | 7,500 | 265.3 | 266.3 | 2.4 [2.4 to 2.4] |
| 1e6 | 4 | 18,500 | 216.3 | 221.4 | 7,500 | 530.9 | 532.5 | 2.5 [2.5 to 2.5] |
| 1e6 | 8 | 18,600 | 430.6 | 438.5 | 7,500 | 1062.1 | 1064.3 | 2.5 [2.5 to 2.5] |

**Mechanism.** Without MPS, each process has its own context, and the GPU runs kernels of one context at a time. The GPU switches to another context when the running context has no more work or its time slice ends.

To see this, each tenant ran 100 kernels of one block that spin for 1 ms on the GPU global timer and record their start and end times on that timer. The overlap is the sum of the kernel durations divided by the length of their union: N means that all tenants run at the same time, and 1 means that they never overlap. The gap is the median idle time between the end of one tenant's kernel and the start of another tenant's kernel.

| N | One context: call p50 (us) | One context: overlap | Process per tenant: call p50 (us) | Process per tenant: overlap | Gap between tenants (us) |
| --- | --- | --- | --- | --- | --- |
| 2 | 1006.6 | 2.00 | 2160.6 | 1.00 | 79.9 |
| 4 | 1006.7 | 3.99 | 4306.9 | 1.00 | 76.8 |
| 8 | 1006.6 | 7.95 | 8619.0 | 1.00 | 76.8 |

In the process-per-tenant mode, the kernels never overlap, and each context switch leaves the GPU idle for about 77 to 80 us. In a closed loop of small kernels, a switch comes before every kernel. So the GPU completes about one kernel per switch, about 13,000 per second for the 1e4-row gather (1 / 77 us) whatever N is, and each tenant's latency is N times (kernel time + switch). The 1e6-row gather pays 54 us of kernel and 79 us of switch per call. In one context, kernels from different streams run at the same time on different SMs, so throughput grows with N until the GPU is full. The 1e6-row gather fills the GPU by itself, so its throughput stays at 17,000 to 19,000 calls per second.

The time slice itself shows in part 4: while another process runs a kernel that fills the GPU for 2 s, each of the tenant's 59 us calls takes 2.36 ms.

**Interpretation.** Without MPS, per-tenant processes serialize the GPU at context granularity, and a switch costs about 77 us. For small operators this cuts total throughput by 22x to 86x and raises latency by N times the switch cost.

## 4. Fault containment

**Setup.** Tenant B runs the 1e6-row gather in a closed loop and checks its output. Tenant A runs the same gather with every index set to `INT_MAX`, which reads 16 GiB past its column. The fault is the same as in the Dask worker experiment.

- One context: A and B are two streams, driven by two host threads of one process. A faults 300 ms after B starts.
- Process per tenant: A and B are separate processes, without MPS, and each fault uses a new process A:
  - Phase 1: A faults while B runs its loop.
  - Phase 2: A faults while B has a 500 ms kernel in flight.
  - Phase 3: a non-faulting check. A runs a 2 s kernel that fills every SM (252 blocks of 512 threads) while B runs its loop.
- A last new process checks that the GPU still runs kernels.

**Results.**

| Case | Tenant A | Tenant B |
| --- | --- | --- |
| One context, fault | `CUDA_ERROR_ILLEGAL_ADDRESS` | 5,031 calls succeeded. The call in flight at the fault, launched 39 us after A's kernel, took 148.7 ms and failed with `CUDA_ERROR_ILLEGAL_ADDRESS`, and so did all 54,969 later calls. After that, `cudaMalloc`, a `cudaMemcpy` of B's column, and a new launch all returned error 700. |
| Separate processes, phase 1 | The fault was reported 145 ms after the launch. A's later `cudaMalloc` failed with `cudaErrorIllegalAddress`. | 22,827 calls, 0 errors, 0 wrong output rows before and after. The one call that overlapped A's fault took 142.3 ms (median 59.2 us). |
| Separate processes, phase 2 | The fault was reported 145 ms after the launch. | B's 500 ms kernel returned `cudaSuccess`, and its GPU timestamps show 500.0 ms. |
| Separate processes, phase 3 | A's 2 s kernel returned success. | 17,778 calls, 0 errors. The 885 calls during A's kernel took a median of 2362 us (59 us outside it). |
| One context, phase 3 case | A's 2 s kernel returned success. | The first call after A's launch waited 2.0 s, the whole length of A's kernel. |
| New process at the end | | Runs the gather with correct output. |

**Interpretation.** Without MPS, a fault in one tenant's process does not fail or corrupt another process's kernels, including one in flight. The others stall for about 142 ms while the driver handles the fault. In one context, the same fault ends every tenant's use of CUDA.

A kernel that runs long in one context also delays the other streams until it ends, because streams in one context do not preempt each other, while separate contexts get time slices of about 2.3 ms.

## 5. Instrumentation baselines

### compute-sanitizer memcheck

**Check that it runs.** `/usr/local/cuda-12.9/bin/compute-sanitizer` (version 2025.2.1) runs programs built with nvcc 13.0 on this driver (KMD 610.57.04, CUDA UMD 13.3). For the faulting gather of part 4, it reports `Invalid __global__ read of size 8 bytes ... is 17,146,314,233 bytes after the nearest allocation`, so it instruments CUDA 13 code.

**Setup.** The same binary runs natively and under `compute-sanitizer --tool memcheck`: three native and memcheck pairs, in alternating order. Each run has three trials after a 0.5 s warmup. Kernel time comes from events around k launches issued back to back (k = 1000, 200, and 10 for the three sizes), divided by k. Call time is the wall time of one launch plus a stream synchronize (median of 20). The table uses eager timing for both modes. At 1e4 rows, the native eager kernel time includes launch gaps: it is 2.55 us, against 1.27 us in a graph. The ratio is the median of the three per-pair ratios.

| Operator | Rows | Kernel, native (us) | Kernel, memcheck (us) | Kernel ratio | Call, native (us) | Call, memcheck (us) | Call ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gather | 1e4 | 2.55 | 43.70 | 17.2 [17.0 to 19.2] | 6.32 | 63.44 | 10.1 [10.0 to 10.1] |
| gather | 1e6 | 54.72 | 1416.8 | 25.9 [25.9 to 25.9] | 59.22 | 1463.1 | 24.7 [24.7 to 24.8] |
| gather | 1e8 | 15939 | 142091 | 8.9 [8.9 to 8.9] | 15943 | 142120 | 8.9 [8.9 to 8.9] |
| projection | 1e4 | 2.56 | 43.25 | 16.9 [16.0 to 17.2] | 6.08 | 63.31 | 10.5 [10.4 to 11.0] |
| projection | 1e6 | 37.25 | 1362.1 | 36.6 [36.5 to 36.6] | 41.75 | 1408.8 | 33.7 [32.7 to 33.8] |
| projection | 1e8 | 3504 | 136329 | 38.9 [38.9 to 38.9] | 3509 | 136355 | 38.9 [38.9 to 38.9] |

**Interpretation.** memcheck makes these operators 9x to 39x slower. As section 10.4 of the design found, it checks allocation bounds, so it would not report an access that stays inside one pool allocation.

### Guardian-style address fencing

**Setup.** A small script rewrites the PTX. It adds two `.u64` parameters (partition base and mask), loads them once, and replaces the address of each `ld.global`, `st.global`, `atom.global`, and `red.global` with `base | (addr & mask)`: one `and.b64` and one `or.b64` per access. The script stops with an error on any access without an explicit state space. It fenced 3 accesses in each kernel.

The driver loads the fenced PTX with `cuModuleLoadData`, as it loads the original. The three buffers lie in one VMM allocation whose size is the next power of two that holds them, aligned to that size. Both kernels run on the same buffers, and the output of each is checked before and after timing.

Kernel time comes from a CUDA graph of k launches (k = 1000, 200, and 10), timed with events, with 20 trials that pair the two kernels in rotating order. The ratio is the median of the 20 per-trial ratios. `ptxas` 13.0 turns each fenced access into 4 `LOP3.LUT` instructions (the and and the or on the two 32-bit halves of the address), 12 per kernel.

| Operator | Rows | Partition | Unfenced (us) | Fenced (us) | Ratio |
| --- | --- | --- | --- | --- | --- |
| gather | 1e4 | 2 MiB | 1.272 | 1.319 | 1.036 [0.939 to 1.042] |
| gather | 1e6 | 32 MiB | 53.64 | 53.67 | 1.0005 [0.9976 to 1.0010] |
| gather | 1e8 | 2 GiB | 15937.6 | 15937.8 | 1.0000 [0.9998 to 1.0002] |
| projection | 1e4 | 2 MiB | 1.161 | 1.207 | 1.040 [0.956 to 1.041] |
| projection | 1e6 | 32 MiB | 36.20 | 36.28 | 1.002 [0.997 to 1.004] |
| projection | 1e8 | 4 GiB | 3503.2 | 3503.4 | 0.9999 [0.9994 to 1.0006] |

**Interpretation.** On these memory-bound operators, fencing costs at most 4% (47 ns at 1e4 rows) and nothing that this experiment can measure at 1e6 and 1e8 rows. Its cost for an analytics engine is in what it cannot express:

- Each tenant needs one power-of-two, size-aligned partition. The 1e8-row projection needs 4 GiB for 2.4 GB of buffers.
- A table that two tenants share must be copied into each partition, as in part 2.
- An access outside the partition goes silently to an address inside the tenant's own partition instead of failing.

## Method

**Machine.**

- CPU and memory: 2 x Intel Xeon Gold 6338 (64 cores), 1 TiB of memory, Linux 5.15.0-190-generic.
- GPU: GPU 0 of two NVIDIA RTX A6000 (48 GiB, 84 SMs, VBIOS 94.02.5C.00.02).
- Driver: KMD 610.57.04, CUDA UMD 13.3.
- GPU settings: persistence mode on, compute mode Default, compute time-slice policy Default. No MPS control daemon ran, and no GPU mode was changed.
- Yama `ptrace_scope` is 1.

**Software.**

- nvcc 13.0.88 (`-O2 -arch=sm_86`). The kernels are the corpus PTX files (nvcc 13.0 output), JIT-compiled by the driver at its default optimization level.
- ptxas 13.0 and cuobjdump 12.9 for the SASS inspection.
- compute-sanitizer 2025.2.1 from CUDA 12.9.
- Python 3.12.9, torch 2.11.0+cu130, cuda-python 13.2.0. Numba was not used, because the corpus's CUDA C++ versions of the two operators served for all measurements.

**Clocks.** Clocks were not locked, because that needs root. Before the runs, the GPU sat at an idle P-state (P0, P3, or P5) with SM clocks of 855 to 1935 MHz. After the runs, the SM clock was 1800 to 1935 MHz (maximum 2100 MHz), and the temperature was 31 to 62 C.

**Idle check.**

- Before each of the 119 measurement runs, a script waited until `nvidia-smi` showed GPU 0 at 5% utilization or less with no compute process. The recorded maximum before any run is 0% with 0 processes.
- Each JSON file records the GPU 0 state (utilization, memory, clocks, P-state, temperature, power, process count) before and after its run.
- The state of GPU 1 was not controlled.
- Every run had `CUDA_VISIBLE_DEVICES=0`, and every process was pinned to CPUs 0 to 31 (NUMA node 0, the CPU affinity of GPU 0) with `taskset`.

**Timing rules, following section 9.3 of the design.**

- Kernel time comes from CUDA graphs timed with events (part 1, and the fencing of part 5), or from events around back-to-back launches where a graph was not used (part 5 memcheck, for both modes).
- Call latencies are host wall time (`steady_clock`) from request to completion. The spin kernels of part 3 use the GPU global timer.
- Warmup is by time: 1 s per mode in part 1 and 0.1 s before each block, 0.5 s per variant in part 5, and 0.3 s in part 3.
- Each measured cell is paired with its baseline within a trial, in rotating or alternating order. The report gives the median and the range over trials.
- Every output was checked against a reference before timing, or, for part 2 and part 3, after it.

**Deviations from section 9.3.** The code ran from a scratch directory, not from a frozen worktree, because the repository has no commits yet. The memcheck comparison uses eager launches in both modes, because it compares against the same method under the sanitizer.

**Raw data.** `records/e4_2026-09-26.tar.gz` (`../records/e4_2026-09-26.tar.gz`, not included in this archive) holds every raw sample, as JSON files, one per run, under `e4/results/`. The part 3 files are gzip-compressed because of their size. The archive also holds the programs as they ran (`e4_latency.cu`, `e4_ipcprobe.cu`, `e4_copy.cu`, `e4_conc.cu`, `e4_fault.cu`, `e4_ktime.cu`, `common.h`), the fencing script `fence.py`, the torch checks, the run scripts, and `analyze.py`, which computes every number in this file. The programs are a record of this run; they are not held to the project's lint and type rules, and the paper's runs port them.

## What did not work or stays open

- Torch CUDA IPC with `expandable_segments:True` (this machine's default environment setting) fails with `pidfd_getfd: Operation not permitted`. The export check therefore used the default allocator.
- The 8 GiB table for 8 tenants was not run: 8 copies of 8.3 GiB plus the engine's copy exceed the 48 GiB GPU.
- The slowdown of random gathers over `cudaIpcOpenMemHandle` mappings of `cudaMalloc` buffers whose size is not a multiple of 2 MiB (up to 70x) has no known cause. It is not monotonic in size: 1.32x at 8 MB, none at 80 MB, 7.6x at 160 MB.
- The pipe hand-off after a 16 ms kernel costs about 110 us instead of 13 us. The cause was not found.
- Clocks were not locked.
- MPS was not measured, because starting its control daemon on this shared machine was not allowed. The process-per-tenant numbers are for time-sliced contexts without MPS. With MPS, kernels from different processes can run at the same time, but MPS limits fault containment (DESIGN section 1), and it changes neither the per-call hand-off, the per-buffer import, nor the table copies.

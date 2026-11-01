# Two Dask clients on one cuDF GPU worker

Recorded 2026-09-27. This is the real-system motivation experiment for this experiment.
Two separately named Dask `Client` connections submit work to one
`dask-cuda` worker. Each client persists its own cuDF column. The worker uses
one fixed RMM pool and one CUDA context. Client A receives a future for only
A's column; client B's future is never passed to A's task.

The executable experiment is
[`dask_worker.py`](dask_worker.py). Its machine-readable result is
`records/e1_dask_cudf_2026-09-27.json` (`../records/e1_dask_cudf_2026-09-27.json`, not included in this archive),
and the sanitizer output is
`records/e1_dask_cudf_memcheck_2026-09-27.log` (`../records/e1_dask_cudf_memcheck_2026-09-27.log`, not included in this archive).
The earlier [`cudf_udf.py`](cudf_udf.py) and its records are preserved unchanged
as the complementary `Series.apply` experiment.

## Why this is a real system path

Dask Distributed documents a scheduler that handles concurrent requests from
several clients and a `Client.submit(function, *args)` interface for custom
work. `dask-cuda.LocalCUDACluster` is the RAPIDS worker implementation: it uses
one GPU per process and exposes a per-worker RMM pool. The cuDF UDF guide
documents passing a `Series` directly to a Numba CUDA kernel through the CUDA
array interface. Its example includes an explicit bounds guard and warns that
omitting the guard can produce undefined behavior.

The experiment composes those documented paths without a mock scheduler,
allocator, column, or kernel launcher:

1. `LocalCUDACluster` starts one worker process on GPU 0 with a fixed 256 MiB
   `PoolMemoryResource`.
2. `Client(name="tenant-a")` persists A's 32-element `int64` cuDF column.
3. `Client(name="tenant-b")` persists B's 32-element secret column.
4. A submits a normal `@cuda.jit` kernel over the CuPy view returned by
   `Series.values`.
5. The harness asks the Dask scheduler for the submitted task's dependency
   keys, and records the worker address, PID, RMM resource, and column
   addresses.

The names A and B describe principals at the service layer. Dask itself does
not provide a mutually untrusted tenant boundary. Its security documentation
says that Dask executes arbitrary user code, that deployments must isolate and
govern access, and that Dask does not manage users. This experiment therefore does not claim
a Dask or cuDF vulnerability. It demonstrates the concrete GPU-worker layout
that requires that trust today and the failure OpGate is intended to prevent
when a service wants to consolidate untrusted operators.

Official documentation used to establish the path:

- [Dask Distributed architecture and multiple clients](https://distributed.dask.org/en/stable/)
- [Dask security and user-management scope](https://docs.dask.org/en/stable/faq.html#is-dask-secure)
- [`dask-cuda.LocalCUDACluster` and its RMM pool](https://docs.rapids.ai/api/dask-cuda/legacy/api/)
- [cuDF UDFs and direct Numba kernels on `Series`](https://docs.rapids.ai/api/cudf/latest/user_guide/guide-to-udfs/)

## Setup

- NVIDIA RTX A6000, compute capability 8.6, driver 610.57.04.
- Python 3.12.9, cuDF 26.08.01, RMM 26.08.00, dask-cuda 26.08.00,
  Dask and Distributed 2026.7.1, Numba 0.64.0, and numba-cuda 0.30.4.
- One Dask GPU worker, one thread, device spilling disabled.
- One 256 MiB RMM `PoolMemoryResource`, with equal initial and maximum size.
- Two `Client` connections with distinct scheduler IDs. The driver holds both
  connections so that it can order and measure the experiment; the scheduler
  still records them as different clients.
- Each case runs in a new process. The availability case cannot poison the
  process used by the confidentiality case.

Install the optional RAPIDS dependencies and run both cases:

```console
uv sync --extra cudf
.venv/bin/python eval/motivation/dask_worker.py all \
  --output eval/records/e1_dask_cudf_2026-09-27.json
```

## Confidentiality result

The fixed pool placed B's column exactly 256 bytes after A's 256-byte column.
The raw record contains both addresses. The harness aborts rather than
searching or changing the layout if this condition is false.

A submitted this ordinary kernel operation:

```python
output[0] = source[source.size]
```

There is no pointer conversion, compiler intrinsic, stale pointer, worker-data
inspection, or B future in the task. Dask reported the task's complete
dependency set as `['tenant-a-live-column']`. The task ran in the same worker
and PID as B's column.

B initially filled its column with `0x11223344556677`. A's one-past load
returned `0x11223344556677`. B then changed its live column in place to
`0x77665544332211`; its address did not change. A repeated the same task, with
the same sole dependency on A's column, and received `0x77665544332211`. B's
own reduction confirmed that all 32 elements held the second value.

This differential step rules out a constant, an A-owned value, or a stale
output: A's result followed a change made only through B's client while A's
task graph remained unchanged.

Compute Sanitizer 2025.2.1 followed all Dask subprocesses for the successful
leak case:

```console
/usr/local/cuda-12.9/bin/compute-sanitizer \
  --tool memcheck --target-processes all --report-api-errors no \
  --error-exitcode 99 \
  --log-file eval/records/e1_dask_cudf_memcheck_2026-09-27.log \
  .venv/bin/python eval/motivation/dask_worker.py leak-child
```

It reported `ERROR SUMMARY: 0 errors`. Both logical columns are suballocations
inside one live RMM allocation, so allocation-level memcheck accepts the load.
`--report-api-errors no` suppresses unrelated `cuGetProcAddress_v2`
compatibility reports between CUDA 12.9 Compute Sanitizer and the CUDA 13
user-mode driver.

## Availability result

In a fresh deployment, B's ordinary cuDF `Series.sum()` returned 32. A then
submitted another normal Numba kernel whose only dependency was
`tenant-a-crash-input`; it loaded `source[2**32]` and raised CUDA error 700.
B's next `Series.sum()` failed with `cudaErrorIllegalAddress`, and B's direct
context synchronization failed with error 700. A and B's pre-fault tasks ran
in the same recorded worker and PID.

This is the worker-level blast radius relevant to OpGate: a defect in A's
submitted GPU operator prevents B's valid cuDF work from using the shared
context. Dask correctly reports the task failures; it cannot restore a CUDA
context inside the still-running worker process.

## Paper claim supported by this experiment

this experiment supports a narrow, reproducible claim: when cuDF columns from two client
workloads reside in one dask-cuda worker and RMM pool, an ordinary unchecked
Numba access from one workload can read the neighboring workload's column,
and an illegal access prevents the other workload from continuing in that
worker's CUDA context. Neither Dask nor cuDF assigns memory authority to the
kernel from its task dependencies or argument list.

The result does not establish that Dask promises hostile-client isolation, that
all deployments share one cluster across users, or that an out-of-bounds read
will find a chosen column under every allocator history. The recorded
deployment makes the adjacency explicit and fails fast when it is absent.
The unchecked corpus run supplies the non-adversarial complement: 24 synthesized-operator defects
crossed neighboring pooled buffers silently, and three poisoned the context.

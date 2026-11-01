# Unsafe PTX and handles through the multi-tenant service

Recorded 2026-09-29. This experiment sends the frozen PTX of the paired replay through the implemented
multi-tenant service in `opgate.hosts.service`. It leaves the Dask worker experiment--the paired replay and their
records unchanged. The machine-readable result is
`records/e8_service_2026-09-29.json` (`../records/e8_service_2026-09-29.json`, not included in this archive),
and the harness is [`service_boundary.py`](service_boundary.py).

## Boundary under test

The service listens on a Unix domain socket and accepts strict,
length-prefixed JSON. A connection credential maps to a server-owned
`TenantId`; the client receives a principal name and opaque handles. One owner
thread creates and uses the executor, one 256 MiB pool, the RMM resource, and
the cuDF hook. Socket threads only authenticate, parse, and queue work.

The generated-operator protocol accepts PTX and a closed plan-node kind. It
does not accept a binding, device address, tenant id, Python callable, pickle,
Dask task, launch grid, raw argument list, or generic CUDA launch. For this experiment the
`read_one_i64` schema derives the Numba ABI binding and the one-thread launch
on the server. Unsupported operations and unknown plan kinds are input errors;
there is no alternative execution path.

Each accepted or refused registration records the submitted PTX, the
server-derived binding, the decision and reason, and
module-load and successful-launch counts. Only an administrator principal can
read those records. Buffer and operator handles belong to the authenticated
connection session. Disconnect cleanup is queued after that session's
submitted work and frees its buffers before pool reuse.

## Method

The current harness reads the actual PTX strings stored under
`registrations.e7_audit_rows` in the successful this experiment archive. It submits those
strings unchanged and compares the resulting audit text directly. Missing or
non-ASCII PTX fails before service startup. The original September 29
measurement tied these inputs to the paired replay using its recorded procedure; that
historical record remains unchanged. Current replay checks the archived
service inputs without recompiling them.

Two spawned OS client processes first authenticate as different principals and
simultaneously request the service's typed numeric cuDF node. The owner thread
serializes both requests in one worker. Tenant A then registers the three paired-replay
modules. The safe control runs on A's catalog buffers. Tenant B places a secret
in its own catalog buffer; A attempts to read and free B's opaque handle. B
then runs another cuDF query. B also uploads four catalog-owned columns from a
deterministic TPC-H-distribution data set at scale 0.01. The same columns run
through two distinct paths: the service-authored cuDF Q6 node, and Sonnet's
recorded model-written `q6_sum` PTX registered under the server-derived
`tpch_q6_sum` schema. A registers the same accepted PTX under its own session
and attempts both paths with B's handles, so the generated-kernel attack reaches
buffer authorization instead of failing because A used B's operator handle.
Finally A disconnects, reconnects with the same credential, and tries the old
buffer and operator handles.

Run it with:

```sh
.venv/bin/python eval/rq1_safety/service_boundary.py \
  --archive eval/records/e8_service_2026-09-29.json \
  --output eval/records/e8_service_replay_2026-09-29.json
```

## Result

| submitted paired-replay module | decision | module loads | launches | observed safe value |
| --- | --- | ---: | ---: | ---: |
| one-past read | refused at its out-of-buffer `ld.global.u64` | 0 | 0 | -- |
| wild read | refused at its out-of-buffer `ld.global.u64` | 0 | 0 | -- |
| `source[0]` control | accepted | 1 | 1 | 11 |

The two client processes returned the expected cuDF values. The cuDF UDF
module loaded once and ran three times: once for each concurrent client and
once for B after A's two unsafe refusals. A's read and free of B's handle were
refused, B's secret was unchanged, and B's post-refusal query returned the
expected `[20.0, 23.0]`. Both old handles were refused after A reconnected.

Both Q6 paths processed 60,261 lineitems. The cuDF path returned
`1163423.5682`; the submitted Sonnet kernel returned
`1163423.5681999999`. Both match the NumPy reference within the declared
floating-point tolerance. A's two Q6 requests with B's four handles were
refused at buffer authorization. The cuDF row UDF loaded and launched once.
The actual model-written PTX is retained in the audit. B's registration loaded
and launched it once, while A's byte-identical registration
reused the verified module and had zero launches. All four input handles were
explicitly freed.

Final service telemetry recorded four module loads and six launches in total,
five accepted registrations, zero input or fatal errors, zero live buffers,
zero service buffer handles, and zero service operator handles. The run used
cuDF 26.08.01, RMM 26.08.00, numba-cuda 0.30.4, CUDA 13, driver 610.57.04,
and one RTX A6000.

The record above predates the guard pass in the service (design decision
38). The script now expects the current outcome: the one-past read is still
refused, at the guarded text's checked load; the wild read is accepted with
the pass's checks and loaded, and its launch after B's secret is written is
answered `check_failed` to A while B's secret stays unchanged; the
telemetry records five module loads, seven launches, and one failed check.
`eval/export/rq1.py` validates the archived record and still expects both
unsafe modules refused.

The integration exposed real host-lifecycle and clean-process issues before
this record was collected. cuDF caches each UDF helper's execution context by
the helper instance; that helper owns its input frame, and cuDF also caches
device scalars. A fast two-client disconnect additionally showed that relying
on destruction of a NumPy-to-cuDF input could retain its 24-byte tenant
allocation. The adapter clears the frame-bearing and scalar caches, and the
service now allocates, writes, exposes, and frees that input explicitly through
the catalog. The CUDA array view now supplies the `strides` field cuDF expects.
Finally, the clean-process run found that numba-cuda can leave its dynamically
referenced `datamodel` submodule detached after multiprocessing initialization;
the frontend initializes and checks it at the compilation boundary. Tests and
this experiment require zero surviving query buffers and fail the worker on an unexpected
live-buffer change.

## Claim and remaining work

this experiment connects the Dask worker experiment/the paired replay's exact disclosure and fault bytes to a restricted service
path: both unsafe registrations reach the gate and neither reaches the CUDA
loader, the accepted control reaches the executor, cross-tenant and stale
handles fail, another tenant remains healthy, and both the cuDF plan and the
submitted recorded model-written Q6 operator return the reference result. This
supports the narrow claim that the implemented service boundary prevents these
measured effects while accepted model-written code participates in Q6.

This experiment is one row set for RQ1, not the full evaluation. The cross-buffer write,
free-and-reuse, index-poisoning, graph-replay, cancellation/timeout, and
wrong-size cases still need service-level records. A CUDA activity trace and a
packaged clean-environment replay remain open. RQ3 now has one service-level
Q6 point but still needs fixed plans for the other 21 TPC-H queries at the
declared scale factors. RQ4 still needs the native, process, process-plus-IPC,
MPS, throughput, tail-latency, and memory measurements. The paper must not use
this experiment as evidence for those unfinished results.

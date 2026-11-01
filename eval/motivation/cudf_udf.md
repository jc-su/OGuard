# A cuDF UDF reads another tenant's column

Recorded 2026-09-27. This experiment runs code submitted through cuDF's ordinary
`Series.apply` path in the same process, CUDA context, and RMM pool as another
tenant's live column. The script and both cases are in
[`cudf_udf.py`](cudf_udf.py). The machine-readable result is
`records/e1_cudf_2026-09-27.json` (`../records/e1_cudf_2026-09-27.json`, not included in this archive).

## Setup

- NVIDIA RTX A6000, compute capability 8.6, driver 610.57.04.
- Python 3.12.9, cuDF 26.08.01, RMM 26.08.00, Numba 0.64.0, and
  numba-cuda 0.30.4.
- One 256 MiB `PoolMemoryResource`, with equal initial and maximum size.
- Tenant A and tenant B are two principals represented by separate live cuDF
  columns. cuDF does not attach an owner to either allocation.
- Each case runs in a fresh subprocess, so the availability case cannot poison
  the process running the other case.

Run both cases and keep the raw result:

```console
.venv/bin/python eval/motivation/cudf_udf.py all \
  --output eval/records/e1_cudf_2026-09-27.json
```

## Confidentiality

The submitted UDF contains a Numba lowering intrinsic that converts an integer
to a device pointer and performs one 64-bit load. Tenant A allocates a 256-byte
cuDF column, records its address, and frees it. Tenant B then allocates its
same-sized secret column, and RMM reuses the exact released block. A submits
the stale address as a scalar argument to a UDF over a different live A
column. The UDF dereferences it without knowing B's address or predicting an
adjacent allocation.

This tests memory authority under one explicit address assumption: the tenant
can learn and retain the address of a buffer it owns. It does not claim that
cuDF's scalar UDF API discloses an address by itself. The custom lowering
intrinsic represents a deliberately malicious low-level UDF rather than the
accidental indexing defects measured by the unchecked corpus run.

Tenant A's current input and B's reused block were distinct live 256-byte
buffers in the same RMM pool. B initially held `0x11223344556677`. Every one of A's 32
`Series.apply` outputs returned that value. The harness then changed only B's
live column, in place, to `0x77665544332211`; B's pointer stayed unchanged and
every A output changed to the second value. cuDF returned both calls normally.

The same successful leakage case ran under CUDA 12.9 Compute Sanitizer 2025.2.1
memcheck with `--report-api-errors no`. That option suppresses unrelated
`cuGetProcAddress_v2` compatibility errors between this sanitizer and the CUDA
13 user-mode driver. Memcheck reported `ERROR SUMMARY: 0 errors`; its log is
`records/e1_cudf_memcheck_2026-09-27.log` (`../records/e1_cudf_memcheck_2026-09-27.log`, not included in this archive).
The target address is within the RMM pool's upstream allocation, so an
allocation-bounds checker considers the load valid.

## Availability

The second submitted UDF made the same 64-bit load from aligned, unmapped
address 8. Tenant B's
ordinary `sum()` returned 496 before the launch. Tenant A's UDF then raised
`cudaErrorIllegalAddress`. B's next `sum()` failed with the same CUDA error,
and the following direct context synchronization returned driver error 700.

## Result

this experiment establishes two properties of the evaluated cuDF path:

1. cuDF compiles and launches submitted code with the process's device-memory
   authority; it does not confine loads to the column whose values the UDF was
   asked to process.
2. A memory fault in that submitted code poisons the CUDA context used by the
   other tenant's tested cuDF operation.

The experiment does not establish an address-disclosure channel, whether a
particular service accepts compiler-extension code, remote code submission, or
isolation between separate CUDA contexts. Those are outside this experiment's claim.

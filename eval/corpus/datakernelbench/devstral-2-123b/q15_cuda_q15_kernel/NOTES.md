# q15_cuda_q15_kernel

Kernel `q15_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/full/queries/q15/round_05.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 5`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q15_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.

## Launch

`q15_launcher` (plan lines 62-75) launches `q15_kernel<<<(l_suppkey.numel() + 1023) / 1024, 1024>>>` with 0 bytes of dynamic shared memory; `run_query` calls it at lines 144-153.

## Lengths and ranges

- `l_suppkey` (read, int64, length `total_rows`): `lineitem["l_suppkey"]` (line 131), supplier keys that the plan does not bound; `total_rows = l_suppkey.numel()` (line 74).
- `l_extendedprice` (read, float64, length `total_rows`): `lineitem["l_extendedprice"]` (line 132).
- `l_discount` (read, float64, length `total_rows`): `lineitem["l_discount"]` (line 133).
- `l_shipdate_days` (read, int32, length `total_rows`): `lineitem["l_shipdate"]` in days (lines 134-136).
- `revenue_accum` (readwrite, float64, length `n_sup`): `torch.zeros(n_sup)` with `n_sup = len(supplier)` (lines 140-141); the kernel adds into it with `atomicAdd`.
- `present_accum` (readwrite, int32, length `n_sup`): `torch.zeros(n_sup, dtype=torch.int32)` (line 142); the kernel updates it with `atomicMax`.
- `date_lo` (scalar, int32): from the date parameter `shipdate_lo` (line 128).
- `date_hi` (scalar, int32): from the date parameter `shipdate_hi` (line 129).
- `n_sup` (size, int64): `revenue_accum.numel()` (line 73).
- `total_rows` (size, int64): `l_suppkey.numel()` (line 74).
- size symbol `total_rows`, least value 0.
- size symbol `n_sup`, least value 0.

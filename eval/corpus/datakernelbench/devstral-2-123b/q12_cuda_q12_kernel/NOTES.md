# q12_cuda_q12_kernel

Kernel `q12_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/full/queries/q12/round_03.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 3`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q12_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Left out `#include <cuda_bf16.hpp>` and `#define BLOCK_SIZE 1024`, which the kernel does not use.

## Launch

`q12_launcher` (plan lines 103-116) launches `q12_kernel<<<(n + 1023) / 1024, 1024>>>` with 0 bytes of dynamic shared memory, `n = shipmode_code.numel()`; `run_query` calls it at lines 197-200.

## Lengths and ranges

- `shipmode_code` (index-in into `high_out`, int32, length `n_elements`): `(joined["l_shipmode"] == shipmode2).astype("int32")` (line 176), so every value is 0 or 1, inside `[0, 2)`, the length of `high_out` and of the shared arrays it indexes; `n_elements = shipmode_code.numel()` (line 103).
- `priority_high` (read, int32, length `n_elements`): a 0/1 flag of the same `joined` rows (lines 177-180).
- `commit_days` (read, int32, length `n_elements`): `l_commitdate` in days of the same rows (lines 182-184).
- `receipt_days` (read, int32, length `n_elements`): `l_receiptdate` in days of the same rows (lines 185-187).
- `ship_days` (read, int32, length `n_elements`): `l_shipdate` in days of the same rows (lines 188-190).
- `date_lo` (scalar, int32): from the date parameter `receipt_lo` (line 167).
- `date_hi` (scalar, int32): from the date parameter `receipt_hi` (line 168).
- `high_out` (readwrite, uint64, length `2`): `torch.zeros(2, dtype=torch.int64)` (line 194), passed as `unsigned long long*`; the kernel adds into it with `atomicAdd`.
- `low_out` (readwrite, uint64, length `2`): `torch.zeros(2, dtype=torch.int64)` (line 195), as `high_out`.
- `n_elements` (size, int64): `shipmode_code.numel()` (line 103).
- size symbol `n_elements`, least value 0.

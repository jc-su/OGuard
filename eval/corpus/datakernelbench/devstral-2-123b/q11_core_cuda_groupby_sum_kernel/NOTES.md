# q11_core_cuda_groupby_sum_kernel

Kernel `groupby_sum_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/core/queries/q11/round_08.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 8`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `groupby_sum_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Left out `#include <cuda_runtime.h>`, which only the host code uses.
- Kept `#define BLOCK_SIZE 512`, the size of the kernel's shared arrays, and the device function `atomicAdd_ull`, which the kernel uses.
- The parameters keep the plan's `size_t` (`unique_count`, `max_unique`, `n`); `signature.json` states them as `uint64`.

## Launch

`groupby_sum_launcher` (plan lines 71-98) launches `groupby_sum_kernel<<<(n + 511) / 512, 512>>>` (lines 84-93) with 0 bytes of dynamic shared memory, with `unique_count` a `cudaMalloc`ed `size_t` set to 0 (lines 79-82); `_query_core` calls it at line 137 with `n = ps_partkey.numel()` and `max_unique = 8000000` (line 133), and `run_query` at line 185.

## Lengths and ranges

- `keys` (read, int64, length `n`): `ps_partkey_t` (line 181), from `joined` (line 179).
- `values` (read, float64, length `n`): `ps_supplycost * ps_availqty` (line 129), the same rows.
- `unique_keys` (write, int64, length `max_unique`): `torch.zeros(max_unique)` (line 134).
- `out_values` (write, float64, length `max_unique`): `torch.zeros(max_unique)` (line 135).
- `unique_count` (readwrite, uint64, length `1`): one `size_t` on the device (lines 79-82); the kernel takes append positions from it with `atomicAdd`.
- `max_unique` (size, uint64): 8000000 (line 133), the length of `unique_keys` and `out_values`.
- `n` (size, uint64): `ps_partkey.numel()` (line 137).
- size symbol `n`, least value 0.
- size symbol `max_unique`, least value 0.

# q05_cuda_q5_kernel

Kernel `q5_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/full/queries/q05/round_06.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 6`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q5_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Left out `#define BLOCK_SIZE 1024` and `#define WARP_SIZE 32`, which the kernel does not use.

## Launch

`q5_launcher` (plan lines 58-66) launches `q5_kernel<<<(num_rows + 1023) / 1024, 1024>>>` with 0 bytes of dynamic shared memory; `run_query` calls it at lines 175-182.

## Lengths and ranges

- `l_extendedprice` (read, float64, length `num_rows`): `final_df["l_extendedprice"]` (line 166); `num_rows = len(final_df)` (line 169).
- `l_discount` (read, float64, length `num_rows`): `final_df["l_discount"]` (line 167).
- `nation_id_ptr` (index-in into `out_revenue`, int64, length `num_rows`): `final_df["nation_id"]` (line 168); the plan assigns `nation_id` as `range(len(nations_unique))` (line 157) and joins it on `n_nationkey` (lines 158-161), so every value lies in `[0, num_nations)`.
- `out_revenue` (readwrite, float64, length `num_nations`): `torch.zeros(num_nations)` (line 172), zeroed again by the launcher (line 55); the kernel adds into it with `atomicAdd`.
- `num_nations` (size, int64): `int(len(nations_unique))` (line 163), the length of `out_revenue`.
- `num_rows` (size, int64): `int(len(final_df))` (line 169), the length of the three row buffers.
- size symbol `num_rows`, least value 0.
- size symbol `num_nations`, least value 0.

# q05_core_cuda_q5_kernel

Kernel `q5_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/core/queries/q05/round_10.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 10`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q5_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Left out `#define BLOCK_SIZE 1024`, which the kernel does not use.

## Launch

`q5_launcher` (plan lines 31-62) launches `q5_kernel<<<(n + 1023) / 1024, 1024>>>` (lines 51-60) with 0 bytes of dynamic shared memory, `n = extendedprice.numel()` (line 50), after zeroing `out_revenue` with `cudaMemset` (line 52); `_query_core` calls it at lines 96-102 and `run_query` at lines 184-189.

## Lengths and ranges

- `extendedprice_ptr` (read, float64, length `n_elements`): `final_df["l_extendedprice"]` (line 186).
- `discount_ptr` (read, float64, length `n_elements`): `final_df["l_discount"]` (line 187).
- `nation_id_ptr` (index-in into `revenue_ptr`, int64, length `n_elements`): `final_df["nation_id"]` (line 185); the plan assigns `nation_id` as `range(len(nations_unique))` (line 178) and joins it on `n_nationkey` (line 180), so every value lies in `[0, num_nations)`.
- `revenue_ptr` (readwrite, float64, length `num_nations`): `torch.zeros(num_nations)` (line 95), zeroed again by the launcher (line 52); the kernel adds into it with `atomicAdd`.
- `num_nations` (size, int64): `int(len(nations_unique))` (line 182), the length of `revenue_ptr`.
- `n_elements` (size, int64): `extendedprice.numel()` (line 50), the length of the three row buffers, which come from the same `final_df`.
- size symbol `n_elements`, least value 0.
- size symbol `num_nations`, least value 0.

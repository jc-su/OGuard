# q14_core_cuda_promo_fuse_kernel

Kernel `promo_fuse_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/core/queries/q14/round_01.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 1`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `promo_fuse_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Kept `#define BLOCK_SIZE 1024`, the size of the kernel's shared arrays.

## Launch

`promo_revenue_launcher` (plan lines 54-91) launches `promo_fuse_kernel<<<(n + 1023) / 1024, 1024>>>` (lines 77-87) with 0 bytes of dynamic shared memory, `n = price.numel()` (line 74); `_query_core` calls it at lines 124-127 and `run_query` at lines 163-170.

## Lengths and ranges

- `price_ptr` (read, float64, length `n_elements`): `joined["l_extendedprice"]` (line 159); the four columns come from the same `joined` (line 154).
- `discount_ptr` (read, float64, length `n_elements`): `joined["l_discount"]` (line 160).
- `shipdate_ptr` (read, int32, length `n_elements`): `joined["l_shipdate"]` in days (lines 156-158).
- `promo_ptr` (read, bool, length `n_elements`): `joined["p_type_is_promo"]` as bool (line 161).
- `date_lo`, `date_hi` (scalar, int32): the day counts of the parameters `shipdate_lo` and `shipdate_hi` (lines 145-146).
- `total_ptr` (readwrite, float64, length `1`): `torch::zeros({1})` (line 75); the kernel adds into it with `atomicAdd`.
- `promo_ptr_out` (readwrite, float64, length `1`): `torch::zeros({1})` (line 76), as `total_ptr`.
- `n_elements` (size, int64): `price.numel()` (line 74).
- size symbol `n_elements`, least value 0.

# q14_cuda_promo_revenue_kernel

Kernel `promo_revenue_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/full/queries/q14/round_01.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 1`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `promo_revenue_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Left out `#include <cuda_fp16.h>`, which the kernel does not use.

## Launch

`promo_revenue_launcher` (plan lines 74-87) launches `promo_revenue_kernel<<<(n + 1023) / 1024, 1024>>>` with 0 bytes of dynamic shared memory, `n = price.numel()`; `run_query` calls it at lines 153-156.

## Lengths and ranges

- `price_ptr` (read, float64, length `n_elements`): `joined["l_extendedprice"]` (line 149).
- `discount_ptr` (read, float64, length `n_elements`): `joined["l_discount"]` (line 150).
- `shipdate_ptr` (read, int32, length `n_elements`): `joined["l_shipdate"]` in days (lines 146-148).
- `promo_ptr` (read, bool, length `n_elements`): `joined["p_type_is_promo"]` as bool (line 151).
- `date_lo` (scalar, int32): from the date parameter `shipdate_lo` (line 135).
- `date_hi` (scalar, int32): from the date parameter `shipdate_hi` (line 136).
- `total_rev_out` (readwrite, float64, length `1`): `torch::zeros({1})` (line 75); the kernel adds into it with `atomicAdd`.
- `promo_rev_out` (readwrite, float64, length `1`): `torch::zeros({1})` (line 76), as `total_rev_out`.
- `n_elements` (size, int64): `price.numel()` (line 74); the four columns come from the same `joined`.
- size symbol `n_elements`, least value 0.
